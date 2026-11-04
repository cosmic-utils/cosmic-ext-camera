// SPDX-License-Identifier: GPL-3.0-only

//! Timelapse video encoder
//!
//! Receives camera frames in real-time via a channel, converts each to
//! RGBA using the app's GPU compute pipeline (supporting all formats
//! including packed Bayer), and encodes them into a video at 30 fps.
//!
//! Pipeline: appsrc (RGBA) → videoconvert → encoder → muxer → filesink

use super::encoder_selection::{
    EncoderConfig, VideoEncoderInput, prepare_video_encoder_input, select_encoders,
};
use super::muxer::{create_muxer, link_muxer_to_sink, link_video_to_muxer};
use super::recorder::{apply_video_tags_to_muxer, convert_frame_to_zoomed_rgba};
use super::{RecordingMonitor, RecordingPhase};
use crate::backends::camera::types::CameraFrame;
use crate::backends::camera::zoom::ZoomControl;
use crate::media::encoders::video::EncoderInfo;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use tracing::{error, info, warn};

/// Target framerate for the output timelapse video.
const TIMELAPSE_FPS: u32 = 30;

/// Run the timelapse encoding loop.
///
/// Receives `CameraFrame`s from `frame_rx`, converts each to RGBA via
/// the GPU compute pipeline (handles YUV, Bayer, and all other formats),
/// pushes it into a GStreamer encoding pipeline at [`TIMELAPSE_FPS`], and
/// finalises the file when the channel closes.
///
/// This function is intended to be spawned as an async task.
/// Compatibility entry point for callers without a failure subscription.
pub async fn run_timelapse_encoder(
    frame_rx: tokio::sync::mpsc::UnboundedReceiver<Arc<CameraFrame>>,
    output_path: PathBuf,
    encoder_info: Option<EncoderInfo>,
    bitrate_kbps: Option<u32>,
    live_filter_code: Arc<AtomicU32>,
    live_zoom: Arc<ZoomControl>,
    capture_metadata: crate::pipelines::capture_metadata::CaptureMetadata,
) -> Result<String, String> {
    let monitor = RecordingMonitor::new_waiting(
        encoder_info
            .as_ref()
            .map(|info| info.element_name.clone())
            .unwrap_or_else(|| "automatic".into()),
    );
    run_timelapse_encoder_monitored(
        frame_rx,
        output_path,
        encoder_info,
        bitrate_kbps,
        live_filter_code,
        live_zoom,
        capture_metadata,
        monitor,
    )
    .await
}

/// Run on an owned OS thread: even a kernel-blocked state change or destructor
/// must not block the caller's async executor. The caller subscribes to `monitor`
/// independently (using `RecordingMonitor::new_waiting`); abandoning this future
/// does not cancel or retry the worker.
#[allow(clippy::too_many_arguments)] // Keep the live controls explicit at the worker boundary.
pub async fn run_timelapse_encoder_monitored(
    frame_rx: tokio::sync::mpsc::UnboundedReceiver<Arc<CameraFrame>>,
    output_path: PathBuf,
    encoder_info: Option<EncoderInfo>,
    bitrate_kbps: Option<u32>,
    live_filter_code: Arc<AtomicU32>,
    live_zoom: Arc<ZoomControl>,
    capture_metadata: crate::pipelines::capture_metadata::CaptureMetadata,
    monitor: RecordingMonitor,
) -> Result<String, String> {
    let worker_monitor = monitor.clone();
    run_timelapse_worker(monitor, move || async move {
        let mut guard = TimelapsePipeline::new(worker_monitor.clone());
        let result = run_timelapse_encoder_inner(
            frame_rx,
            output_path,
            encoder_info,
            bitrate_kbps,
            live_filter_code,
            live_zoom,
            capture_metadata,
            &mut guard,
        )
        .await;
        // Publish errors before entering potentially uninterruptible NULL/drop.
        if let Err(detail) = &result {
            worker_monitor.report_error(detail.clone());
        }
        let cleanup = guard.cleanup();
        result.and_then(|path| cleanup.map(|()| path))
    })
    .await
}

async fn run_timelapse_worker<F, Fut>(monitor: RecordingMonitor, work: F) -> Result<String, String>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<String, String>>,
{
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let worker_monitor = monitor.clone();
    let spawn = std::thread::Builder::new()
        .name("timelapse-encoder".into())
        .spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| format!("Timelapse runtime: {e}"))?;
                runtime.block_on(work())
            }))
            .unwrap_or_else(|_| Err("Timelapse encoder worker panicked".into()));
            if let Err(detail) = &result {
                worker_monitor.report_error(detail.clone());
            }
            worker_monitor.finish();
            let _ = sender.send(result);
        });
    if let Err(error) = spawn {
        let detail = format!("Cannot start timelapse encoder worker: {error}");
        monitor.report_error(detail.clone());
        monitor.finish();
        return Err(detail);
    }
    // Deliberately do not join: a stuck driver cannot be forcibly cancelled.
    receiver
        .await
        .unwrap_or_else(|_| Err("Timelapse encoder worker disconnected".into()))
}

/// Own the pipeline through *all* early returns and unwinding. Never finish the
/// monitor here: Gst object destruction after NULL can itself block in a driver.
struct TimelapsePipeline {
    pipeline: Option<gst::Pipeline>,
    monitor: RecordingMonitor,
}

impl TimelapsePipeline {
    fn new(monitor: RecordingMonitor) -> Self {
        Self {
            pipeline: None,
            monitor,
        }
    }

    fn cleanup(&mut self) -> Result<(), String> {
        // Take before entering Gst so Drop cannot retry NULL or reset its clock.
        let Some(pipeline) = self.pipeline.take() else {
            return Ok(());
        };
        self.monitor.set_phase(RecordingPhase::Cleanup);
        let result = pipeline
            .set_state(gst::State::Null)
            .map(|_| ())
            .map_err(|e| format!("set NULL: {e:?}"));
        if let Err(detail) = &result {
            self.monitor.report_error(detail.clone());
        }
        drop(pipeline);
        result
    }
}

impl Drop for TimelapsePipeline {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.monitor
                .report_error("Timelapse encoder worker panicked".into());
        }
        let _ = self.cleanup();
    }
}

/// Observe without removing messages needed by the final EOS/error wait.
fn install_timelapse_observer(pipeline: &gst::Pipeline, monitor: &RecordingMonitor) {
    if let Some(bus) = pipeline.bus() {
        let monitor = monitor.clone();
        bus.set_sync_handler(move |_, message| {
            if let gst::MessageView::Error(error) = message.view() {
                monitor.report_error(format!(
                    "GStreamer error: {} ({})",
                    error.error(),
                    error.debug().unwrap_or_default(),
                ));
            }
            gst::BusSyncReply::Pass
        });
    }
}

async fn receive_timelapse_frame(
    frames: &mut tokio::sync::mpsc::UnboundedReceiver<Arc<CameraFrame>>,
    status: &mut tokio::sync::watch::Receiver<super::RecordingStatus>,
) -> Result<Option<Arc<CameraFrame>>, String> {
    loop {
        if let Some(detail) = status.borrow().error.clone() {
            return Err(detail);
        }
        tokio::select! {
            biased;
            changed = status.changed() => {
                changed.map_err(|_| "Timelapse monitor disconnected".to_string())?;
            }
            frame = frames.recv() => return Ok(frame),
        }
    }
}

#[allow(clippy::too_many_arguments)] // Mirrors the public monitored entry point.
async fn run_timelapse_encoder_inner(
    mut frame_rx: tokio::sync::mpsc::UnboundedReceiver<Arc<CameraFrame>>,
    output_path: PathBuf,
    encoder_info: Option<EncoderInfo>,
    bitrate_kbps: Option<u32>,
    live_filter_code: Arc<AtomicU32>,
    live_zoom: Arc<ZoomControl>,
    capture_metadata: crate::pipelines::capture_metadata::CaptureMetadata,
    guard: &mut TimelapsePipeline,
) -> Result<String, String> {
    let monitor = guard.monitor.clone();
    // The caller supplies new_waiting(): sparse input (even frame one) is idle,
    // not startup. Do not opt into the continuous-recording activity watchdog.
    monitor.set_phase(RecordingPhase::WaitingForFrames);
    // Wait for the first frame so we know the dimensions.
    let first_frame = frame_rx
        .recv()
        .await
        .ok_or_else(|| "Channel closed before first frame".to_string())?;

    monitor.set_phase(RecordingPhase::Starting);
    let width = first_frame.width;
    let height = first_frame.height;

    info!(
        width, height,
        fps = TIMELAPSE_FPS,
        format = ?first_frame.format,
        output = %output_path.display(),
        "Starting timelapse encoder"
    );

    // --- encoder selection ---------------------------------------------------
    let encoder_config = EncoderConfig {
        width,
        height,
        bitrate_override_kbps: bitrate_kbps,
        ..Default::default()
    };

    let encoders = select_encoders(&encoder_config, encoder_info.as_ref(), false)
        .map_err(|e| format!("Encoder selection failed: {e}"))?;

    let video_enc = encoders.video;

    let enc_name = video_enc
        .encoder
        .factory()
        .map(|f| f.name().to_string())
        .unwrap_or_default();
    monitor.set_encoder(enc_name.clone());
    // Retain the resolved encoder in the guard before input negotiation can fail.
    // Otherwise an early `?` could drop the device-backed element before the
    // caller can publish the error and enter monitored cleanup.
    guard.pipeline = Some(gst::Pipeline::new());
    let pipeline = guard.pipeline.as_ref().unwrap();
    install_timelapse_observer(pipeline, &monitor);
    let mut status = monitor.subscribe();
    pipeline.add(&video_enc.encoder).map_err(|e| {
        let detail = format!("pipeline add encoder: {e}");
        monitor.report_error(detail.clone());
        detail
    })?;
    let extension = video_enc.extension;
    let encoder_input =
        prepare_video_encoder_input(&video_enc.encoder, &enc_name, width, height, TIMELAPSE_FPS)
            .map_err(|e| format!("Encoder input preparation failed: {e}"))?;

    let mut final_output = output_path;
    final_output.set_extension(extension);

    info!(
          encoder = %enc_name,
          input_format = ?encoder_input.format,
          encode_width = encoder_input.width,
          encode_height = encoder_input.height,
          output = %final_output.display(),
          "Timelapse encoder selected");

    // --- build GStreamer pipeline (always RGBA input) -------------------------

    let appsrc = gst_app::AppSrc::builder()
        .caps(
            &gst::Caps::builder("video/x-raw")
                .field("format", "RGBA")
                .field("width", width as i32)
                .field("height", height as i32)
                .field("framerate", gst::Fraction::new(TIMELAPSE_FPS as i32, 1))
                .build(),
        )
        .format(gst::Format::Time)
        .is_live(false)
        .build();

    let videoconvert = gst::ElementFactory::make("videoconvert")
        .build()
        .map_err(|e| format!("videoconvert: {e}"))?;

    let encoder_elem = video_enc.encoder;
    let input_chain = create_encoder_input_chain(&encoder_input)?;
    let parser = video_enc.parser;
    let muxer_elem = video_enc.muxer;
    let muxer_cfg = create_muxer(muxer_elem, &final_output)?;
    apply_video_tags_to_muxer(&muxer_cfg.muxer, &capture_metadata);

    // Add elements
    pipeline
        .add_many([
            appsrc.upcast_ref(),
            &videoconvert,
            &input_chain.videoscale,
            &input_chain.capsfilter,
            &muxer_cfg.muxer,
            &muxer_cfg.filesink,
        ])
        .map_err(|e| format!("pipeline add: {e}"))?;

    // Link
    appsrc
        .link(&videoconvert)
        .map_err(|_| "link appsrc→videoconvert")?;
    link_converter_to_encoder(&videoconvert, &input_chain, &encoder_elem)?;

    if let Some(ref p) = parser {
        pipeline
            .add(p)
            .map_err(|e| format!("pipeline add parser: {e}"))?;
        encoder_elem.link(p).map_err(|_| "link encoder→parser")?;
        link_video_to_muxer(p, &muxer_cfg.muxer)?;
    } else {
        link_video_to_muxer(&encoder_elem, &muxer_cfg.muxer)?;
    }
    link_muxer_to_sink(&muxer_cfg.muxer, &muxer_cfg.filesink)?;

    // Start
    pipeline
        .set_state(gst::State::Playing)
        .map_err(|e| format!("set PLAYING: {e:?}"))?;
    monitor.set_phase(RecordingPhase::Recording);

    // --- push frames --------------------------------------------------------
    let frame_duration_ns: u64 = 1_000_000_000 / TIMELAPSE_FPS as u64;
    let frame_duration = gst::ClockTime::from_nseconds(frame_duration_ns);
    let mut frame_index: u64 = 0;

    // Push the first frame we already received
    let rgba = convert_and_filter(&first_frame, &live_filter_code, &live_zoom).await?;
    push_rgba(&appsrc, rgba, frame_index, frame_duration)?;
    frame_index += 1;

    // Receive remaining frames until the channel is closed (timelapse stopped)
    while let Some(frame) = receive_timelapse_frame(&mut frame_rx, &mut status).await? {
        if frame.width != width || frame.height != height {
            warn!(
                expected_w = width,
                expected_h = height,
                actual_w = frame.width,
                actual_h = frame.height,
                "Skipping frame with mismatched dimensions"
            );
            continue;
        }
        match convert_and_filter(&frame, &live_filter_code, &live_zoom).await {
            Ok(rgba) => {
                if let Err(e) = push_rgba(&appsrc, rgba, frame_index, frame_duration) {
                    error!(error = %e, frame = frame_index, "Failed to push frame, stopping");
                    return Err(e);
                }
                frame_index += 1;
            }
            Err(e) => {
                warn!(error = %e, frame = frame_index, "Skipping frame (conversion failed)");
                continue;
            }
        }

        if frame_index.is_multiple_of(30) {
            info!(frames = frame_index, "Timelapse encoding progress");
        }
    }

    info!(
        total_frames = frame_index,
        "Channel closed, finalising timelapse video"
    );

    if frame_index == 0 {
        let _ = std::fs::remove_file(&final_output);
        return Err("No frames were encoded".into());
    }

    finalize_timelapse_pipeline(
        pipeline,
        &appsrc,
        &monitor,
        gst::ClockTime::from_seconds(30),
    )?;

    info!(path = %final_output.display(), frames = frame_index, "Timelapse video saved");
    Ok(final_output.display().to_string())
}

fn finalize_timelapse_pipeline(
    pipeline: &gst::Pipeline,
    appsrc: &gst_app::AppSrc,
    monitor: &RecordingMonitor,
    timeout: gst::ClockTime,
) -> Result<(), String> {
    monitor.set_phase(RecordingPhase::Finalizing);
    // Signal end of stream
    appsrc
        .end_of_stream()
        .map_err(|e| format!("end_of_stream: {e}"))?;

    // Wait for EOS
    let bus = pipeline.bus().ok_or("No pipeline bus")?;
    loop {
        match bus.timed_pop_filtered(timeout, &[gst::MessageType::Eos, gst::MessageType::Error]) {
            Some(msg) => match msg.view() {
                gst::MessageView::Eos(..) => {
                    info!("Timelapse video EOS received");
                    break;
                }
                gst::MessageView::Error(e) => {
                    let err_msg = format!(
                        "GStreamer error: {} ({})",
                        e.error(),
                        e.debug().unwrap_or_default()
                    );
                    return Err(err_msg);
                }
                _ => {}
            },
            None => {
                return Err("Timeout waiting for pipeline EOS".into());
            }
        }
    }

    Ok(())
}

struct EncoderInputChain {
    videoscale: gst::Element,
    capsfilter: gst::Element,
}

fn create_encoder_input_chain(input: &VideoEncoderInput) -> Result<EncoderInputChain, String> {
    let videoscale = gst::ElementFactory::make("videoscale")
        .build()
        .map_err(|e| format!("videoscale: {e}"))?;
    let capsfilter = gst::ElementFactory::make("capsfilter")
        .property("caps", input.caps(TIMELAPSE_FPS))
        .build()
        .map_err(|e| format!("capsfilter: {e}"))?;

    Ok(EncoderInputChain {
        videoscale,
        capsfilter,
    })
}

fn link_converter_to_encoder(
    videoconvert: &gst::Element,
    input_chain: &EncoderInputChain,
    encoder: &gst::Element,
) -> Result<(), String> {
    gst::Element::link_many([
        videoconvert,
        &input_chain.videoscale,
        &input_chain.capsfilter,
        encoder,
    ])
    .map_err(|_| "link videoconvert→videoscale→capsfilter→encoder".to_string())
}

/// Convert a frame to RGBA and apply the current live filter (if any).
async fn convert_and_filter(
    frame: &CameraFrame,
    live_filter_code: &AtomicU32,
    live_zoom: &ZoomControl,
) -> Result<Vec<u8>, String> {
    let mut rgba = convert_frame_to_zoomed_rgba(frame, live_zoom.zoom()).await?;

    let filter_code = live_filter_code.load(Ordering::Relaxed);
    if filter_code != 0 {
        let filter = crate::app::FilterType::from_gpu_filter_code(filter_code);
        rgba = crate::shaders::apply_filter_gpu_rgba(&rgba, frame.width, frame.height, filter)
            .await
            .map_err(|e| format!("Filter failed: {e}"))?;
    }

    Ok(rgba)
}

/// Push tightly-packed RGBA bytes into appsrc with correct PTS.
fn push_rgba(
    appsrc: &gst_app::AppSrc,
    rgba: Vec<u8>,
    index: u64,
    duration: gst::ClockTime,
) -> Result<(), String> {
    let pts = duration * index;
    let mut buffer = gst::Buffer::from_mut_slice(rgba);
    {
        let buf = buffer.get_mut().unwrap();
        buf.set_pts(pts);
        buf.set_duration(duration);
    }

    appsrc
        .push_buffer(buffer)
        .map(|_| ())
        .map_err(|e| format!("push_buffer: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn timelapse_zoom_samples_each_frame_and_resets_without_resizing() {
        let frame = CameraFrame {
            width: 8,
            height: 4,
            stride: 32,
            format: crate::backends::camera::types::PixelFormat::RGBA,
            data: crate::backends::camera::types::FrameData::from_owned_vec(
                (0..128).map(|i| i as u8).collect(),
            ),
            yuv_planes: None,
            captured_at: std::time::Instant::now(),
            sensor_timestamp_ns: None,
            libcamera_metadata: None,
        };
        let filter = AtomicU32::new(0);
        let zoom = ZoomControl::default();
        let first = convert_and_filter(&frame, &filter, &zoom).await.unwrap();
        zoom.set_zoom(2.0);
        let middle = convert_and_filter(&frame, &filter, &zoom).await.unwrap();
        zoom.set_zoom(1.0);
        let reset = convert_and_filter(&frame, &filter, &zoom).await.unwrap();
        assert_eq!(first.len(), 128);
        assert_eq!(middle.len(), 128);
        assert_ne!(middle, first, "timelapse must sample zoom for every frame");
        assert_eq!(&middle[..4], &frame.data.as_ref()[40..44]);
        assert_eq!(reset, first);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn monitored_timelapse_closed_input_reports_failure_and_finishes() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        drop(tx);
        let monitor = super::super::RecordingMonitor::new("chosen".into());
        let status = monitor.subscribe();
        let result = run_timelapse_encoder_monitored(
            rx,
            PathBuf::from("unused-timelapse-output"),
            None,
            None,
            Arc::new(AtomicU32::new(0)),
            Arc::new(ZoomControl::default()),
            Default::default(),
            monitor,
        )
        .await;
        assert_eq!(result.unwrap_err(), "Channel closed before first frame");
        assert_eq!(
            status.borrow().error.as_deref(),
            Some("Channel closed before first frame")
        );
        assert_eq!(
            status.borrow().phase,
            super::super::RecordingPhase::Finished
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn monitored_timelapse_sparse_input_is_idle_not_starting() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let monitor = RecordingMonitor::new_waiting("chosen".into());
        let status = monitor.subscribe();
        let task = tokio::spawn(run_timelapse_encoder_monitored(
            rx,
            PathBuf::from("unused-timelapse-output"),
            None,
            None,
            Arc::new(AtomicU32::new(0)),
            Arc::new(ZoomControl::default()),
            Default::default(),
            monitor,
        ));
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let phase = status.borrow().phase;
        drop(tx);
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(2), task)
                .await
                .unwrap()
                .unwrap()
                .is_err()
        );
        assert_eq!(phase, RecordingPhase::WaitingForFrames);
    }

    #[test]
    fn monitored_timelapse_drop_cleans_pipeline_without_finishing_or_resetting_deadline() {
        gst::init().unwrap();
        let monitor = RecordingMonitor::new("software".into());
        let status = monitor.subscribe();
        let pipeline = gst::Pipeline::new();
        pipeline.set_state(gst::State::Ready).unwrap();
        let mut guard = TimelapsePipeline::new(monitor.clone());
        guard.pipeline = Some(pipeline.clone());
        guard.cleanup().unwrap();
        let cleanup_started = status.borrow().phase_started;
        drop(guard);
        assert_eq!(pipeline.current_state(), gst::State::Null);
        assert_eq!(status.borrow().phase, RecordingPhase::Cleanup);
        assert_eq!(status.borrow().phase_started, cleanup_started);

        pipeline.set_state(gst::State::Ready).unwrap();
        let mut guard = TimelapsePipeline::new(monitor.clone());
        guard.pipeline = Some(pipeline.clone());
        drop(guard);
        assert_eq!(pipeline.current_state(), gst::State::Null);
        assert_eq!(status.borrow().phase, RecordingPhase::Cleanup);
        assert_eq!(status.borrow().phase_started, cleanup_started);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn monitored_timelapse_software_resolves_encoder_and_writes_video() {
        gst::init().unwrap();
        let scratch = std::env::temp_dir();
        let output = scratch.join(format!("timelapse-monitor-{}.mp4", uuid::Uuid::new_v4()));
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let frame = Arc::new(CameraFrame {
            width: 32,
            height: 32,
            data: crate::backends::camera::types::FrameData::from_owned_vec(vec![128; 32 * 32 * 4]),
            format: crate::backends::camera::types::PixelFormat::RGBA,
            stride: 32 * 4,
            yuv_planes: None,
            captured_at: std::time::Instant::now(),
            sensor_timestamp_ns: None,
            libcamera_metadata: None,
        });
        tx.send(frame.clone()).unwrap();
        tx.send(frame).unwrap();
        drop(tx);
        let monitor = RecordingMonitor::new("unresolved".into());
        let status = monitor.subscribe();
        let encoder = EncoderInfo {
            element_name: "x264enc".into(),
            display_name: "software fixture".into(),
            codec: crate::media::encoders::video::VideoCodec::H264,
            is_hardware: false,
            priority: 0,
        };
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            run_timelapse_encoder_monitored(
                rx,
                output.clone(),
                Some(encoder),
                None,
                Arc::new(AtomicU32::new(0)),
                Arc::new(ZoomControl::default()),
                Default::default(),
                monitor,
            ),
        )
        .await;
        let bytes = std::fs::read(&output);
        let _ = std::fs::remove_file(&output);
        assert_eq!(result.unwrap().unwrap(), output.display().to_string());
        assert!(bytes.unwrap().len() > 100);
        assert_eq!(status.borrow().encoder, "x264enc");
        assert_eq!(status.borrow().phase, RecordingPhase::Finished);
        assert!(status.borrow().error.is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn monitored_timelapse_bus_error_is_visible_while_input_remains_open() {
        gst::init().unwrap();
        let output =
            std::env::temp_dir().join(format!("timelapse-error-{}.mp4", uuid::Uuid::new_v4()));
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Arc::new(CameraFrame {
            width: 33,
            height: 33,
            // x264 rejects odd I420 dimensions during asynchronous negotiation.
            data: crate::backends::camera::types::FrameData::from_owned_vec(vec![0; 33 * 33 * 4]),
            format: crate::backends::camera::types::PixelFormat::RGBA,
            stride: 33 * 4,
            yuv_planes: None,
            captured_at: std::time::Instant::now(),
            sensor_timestamp_ns: None,
            libcamera_metadata: None,
        }))
        .unwrap();
        let monitor = RecordingMonitor::new("unresolved".into());
        let mut status = monitor.subscribe();
        let encoder = EncoderInfo {
            element_name: "x264enc".into(),
            display_name: "software fixture".into(),
            codec: crate::media::encoders::video::VideoCodec::H264,
            is_hardware: false,
            priority: 0,
        };
        let task = tokio::spawn(run_timelapse_encoder_monitored(
            rx,
            output.clone(),
            Some(encoder),
            None,
            Arc::new(AtomicU32::new(0)),
            Arc::new(ZoomControl::default()),
            Default::default(),
            monitor,
        ));
        let observed = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                if status.borrow().error.is_some() {
                    break;
                }
                status.changed().await.unwrap();
            }
        })
        .await;
        drop(tx);
        let result = tokio::time::timeout(std::time::Duration::from_secs(3), task).await;
        let _ = std::fs::remove_file(&output);
        assert!(
            observed.is_ok(),
            "Gst Error must be reported before the input closes"
        );
        assert!(result.unwrap().unwrap().is_err());
        assert!(
            status
                .borrow()
                .error
                .as_ref()
                .unwrap()
                .contains("GStreamer error")
        );
        assert_eq!(status.borrow().phase, RecordingPhase::Finished);
    }

    #[test]
    fn monitored_timelapse_marks_finalizing_before_eos_is_observed() {
        gst::init().unwrap();
        let pipeline = gst::Pipeline::new();
        let appsrc = gst_app::AppSrc::builder().format(gst::Format::Time).build();
        let sink = gst::ElementFactory::make("fakesink").build().unwrap();
        pipeline.add_many([appsrc.upcast_ref(), &sink]).unwrap();
        appsrc.link(&sink).unwrap();
        let monitor = RecordingMonitor::new("software".into());
        let status = monitor.subscribe();
        let observed = Arc::new(std::sync::Mutex::new(None));
        let callback_observed = observed.clone();
        let callback_status = status.clone();
        pipeline.bus().unwrap().set_sync_handler(move |_, message| {
            if matches!(message.view(), gst::MessageView::Eos(_)) {
                *callback_observed.lock().unwrap() = Some(callback_status.borrow().phase);
            }
            gst::BusSyncReply::Pass
        });
        pipeline.set_state(gst::State::Playing).unwrap();
        let result = finalize_timelapse_pipeline(
            &pipeline,
            &appsrc,
            &monitor,
            gst::ClockTime::from_seconds(1),
        );
        pipeline.set_state(gst::State::Null).unwrap();
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(*observed.lock().unwrap(), Some(RecordingPhase::Finalizing));
        assert_eq!(status.borrow().phase, RecordingPhase::Finalizing);
    }

    #[test]
    fn monitored_timelapse_eos_bus_noise_does_not_extend_deadline() {
        gst::init().unwrap();
        let pipeline = gst::Pipeline::new();
        pipeline.set_state(gst::State::Ready).unwrap();
        let appsrc = gst_app::AppSrc::builder().build();
        appsrc.set_state(gst::State::Paused).unwrap();
        let bus = pipeline.bus().unwrap();
        for _ in 0..100 {
            bus.post(
                gst::message::Application::builder(gst::Structure::new_empty(
                    "timelapse-test-noise",
                ))
                .build(),
            )
            .unwrap();
        }
        bus.post(gst::message::Eos::builder().build()).unwrap();
        let result = finalize_timelapse_pipeline(
            &pipeline,
            &appsrc,
            &RecordingMonitor::new("software".into()),
            gst::ClockTime::from_nseconds(1),
        );
        appsrc.set_state(gst::State::Null).unwrap();
        pipeline.set_state(gst::State::Null).unwrap();
        assert_eq!(result.unwrap_err(), "Timeout waiting for pipeline EOS");
    }

    #[test]
    fn monitored_timelapse_rejected_eos_is_an_error_not_success() {
        gst::init().unwrap();
        let pipeline = gst::Pipeline::new();
        pipeline.set_state(gst::State::Ready).unwrap();
        let appsrc = gst_app::AppSrc::builder().build();
        appsrc.set_state(gst::State::Paused).unwrap();
        appsrc.set_state(gst::State::Null).unwrap();
        pipeline
            .bus()
            .unwrap()
            .post(gst::message::Eos::builder().build())
            .unwrap();
        let result = finalize_timelapse_pipeline(
            &pipeline,
            &appsrc,
            &RecordingMonitor::new("software".into()),
            gst::ClockTime::from_seconds(1),
        );
        pipeline.set_state(gst::State::Null).unwrap();
        assert!(result.unwrap_err().contains("end_of_stream"));
    }

    // Supplemental finite characterization of the worker boundary and observer.
    #[tokio::test(flavor = "current_thread")]
    async fn monitored_timelapse_worker_keeps_executor_responsive_until_cleanup_returns() {
        let monitor = RecordingMonitor::new("software".into());
        let status = monitor.subscribe();
        let worker_monitor = monitor.clone();
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let caller = std::thread::current().id();
        let task = tokio::spawn(run_timelapse_worker(monitor, move || async move {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            worker_monitor.report_error("synthetic error before cleanup".into());
            worker_monitor.set_phase(RecordingPhase::Cleanup);
            entered_tx.send(std::thread::current().id()).unwrap();
            release_rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .unwrap();
            Err("synthetic error before cleanup".into())
        }));
        let worker = tokio::time::timeout(std::time::Duration::from_secs(1), entered_rx)
            .await
            .unwrap()
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        let phase_while_blocked = status.borrow().phase;
        let error_while_blocked = status.borrow().error.clone();
        release_tx.send(()).unwrap();
        let result = tokio::time::timeout(std::time::Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap();
        assert_ne!(worker, caller);
        assert_eq!(phase_while_blocked, RecordingPhase::Cleanup);
        assert_eq!(
            error_while_blocked.as_deref(),
            Some("synthetic error before cleanup")
        );
        assert!(result.is_err());
        assert_eq!(status.borrow().phase, RecordingPhase::Finished);
    }

    #[test]
    fn monitored_timelapse_observer_preserves_errors_for_finalization() {
        gst::init().unwrap();
        let pipeline = gst::Pipeline::new();
        pipeline.set_state(gst::State::Ready).unwrap();
        let monitor = RecordingMonitor::new("software".into());
        let status = monitor.subscribe();
        install_timelapse_observer(&pipeline, &monitor);
        let bus = pipeline.bus().unwrap();
        bus.post(
            gst::message::Error::builder(gst::CoreError::Failed, "synthetic driver error")
                .src(&pipeline)
                .build(),
        )
        .unwrap();
        assert!(
            status
                .borrow()
                .error
                .as_ref()
                .unwrap()
                .contains("synthetic driver error")
        );
        let appsrc = gst_app::AppSrc::builder().build();
        appsrc.set_state(gst::State::Paused).unwrap();
        let result = finalize_timelapse_pipeline(
            &pipeline,
            &appsrc,
            &monitor,
            gst::ClockTime::from_seconds(1),
        );
        appsrc.set_state(gst::State::Null).unwrap();
        pipeline.set_state(gst::State::Null).unwrap();
        assert!(result.unwrap_err().contains("synthetic driver error"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn monitored_timelapse_panic_cleans_before_finishing() {
        gst::init().unwrap();
        let monitor = RecordingMonitor::new("software".into());
        let status = monitor.subscribe();
        let worker_monitor = monitor.clone();
        let pipeline = gst::Pipeline::new();
        pipeline.set_state(gst::State::Ready).unwrap();
        let worker_pipeline = pipeline.clone();
        let result = run_timelapse_worker(monitor, move || async move {
            let mut guard = TimelapsePipeline::new(worker_monitor);
            guard.pipeline = Some(worker_pipeline);
            panic!("synthetic worker panic");
        })
        .await;
        assert_eq!(result.unwrap_err(), "Timelapse encoder worker panicked");
        assert_eq!(pipeline.current_state(), gst::State::Null);
        assert_eq!(status.borrow().phase, RecordingPhase::Finished);
        assert_eq!(
            status.borrow().error.as_deref(),
            Some("Timelapse encoder worker panicked")
        );
    }

    #[test]
    fn v4l2_timelapse_chain_remains_linked_after_elements_enter_pipeline() {
        gst::init().unwrap();
        let pipeline = gst::Pipeline::new();
        let appsrc = gst::ElementFactory::make("appsrc").build().unwrap();
        let videoconvert = gst::ElementFactory::make("videoconvert").build().unwrap();
        let caps = gst::Caps::builder("video/x-raw")
            .field("format", "NV12")
            .field("width", gst::IntRange::<i32>::new(16, 4096))
            .field("height", gst::IntRange::<i32>::new(16, 4096))
            .build();
        let encoder = gst::ElementFactory::make("capsfilter")
            .property("caps", caps)
            .build()
            .unwrap();
        let input = super::super::encoder_selection::prepare_video_encoder_input(
            &encoder,
            "v4l2h265enc",
            1436,
            1080,
            TIMELAPSE_FPS,
        )
        .unwrap();

        let chain = create_encoder_input_chain(&input).unwrap();
        pipeline
            .add_many([
                &appsrc,
                &videoconvert,
                &chain.videoscale,
                &chain.capsfilter,
                &encoder,
            ])
            .unwrap();
        appsrc.link(&videoconvert).unwrap();
        link_converter_to_encoder(&videoconvert, &chain, &encoder).unwrap();

        assert_eq!((input.width, input.height), (1440, 1088));
        assert_eq!(input.format.as_deref(), Some("NV12"));
        assert!(videoconvert.static_pad("src").unwrap().is_linked());
        assert!(chain.videoscale.static_pad("src").unwrap().is_linked());
        assert!(chain.capsfilter.static_pad("src").unwrap().is_linked());
    }
}
