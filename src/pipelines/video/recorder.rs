// SPDX-License-Identifier: GPL-3.0-only

//! Video recording pipeline with intelligent encoder selection
//!
//! This module implements video recording with:
//! - Automatic hardware encoder detection and selection
//! - Preview continues through independent shared-frame delivery
//! - Audio integration
//! - Quality presets

use super::encoder_selection::{
    EncoderConfig, VideoEncoderInput, prepare_video_encoder_input, select_encoders,
};
use super::muxer::link_audio_to_muxer;
use super::recording_monitor::{FinishRecording, RecordingMonitor, RecordingPhase};
use super::stats::{
    RECORDING_STATS, RecordingDiagnostics, clear_recording_diagnostics,
    publish_recording_diagnostics,
};
use crate::backends::camera::types::{CameraFrame, PixelFormat, RecordingFrame};
use crate::backends::camera::zoom::{ZoomControl, residual_zoom};
use crate::media::encoders::video::SelectedVideoEncoder;
use crate::pipelines::audio_level::install_level_sync_handler as install_shared_level_sync_handler;
use crate::pipelines::audio_level::{PULSESRC_BUFFER_TIME_US, PULSESRC_SLAVE_METHOD};
use chrono::{Datelike, Timelike, Utc};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, error, info, warn};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// How often to emit periodic progress log messages (every Nth frame).
const LOG_EVERY_N_FRAMES: u64 = 60;

/// Maximum number of frames retained by appsrc while the encoder is slower
/// than capture. Live recording must prefer a recent frame over unbounded
/// latency and memory growth.
const APPSRC_MAX_BUFFERS: u64 = 3;

pub use crate::pipelines::audio_level::{AudioLevels, SharedAudioLevels};

/// Common recording configuration.
pub struct RecorderConfig<'a> {
    /// Video width
    pub width: u32,
    /// Video height
    pub height: u32,
    /// Video framerate
    pub framerate: u32,
    /// Output file path
    pub output_path: PathBuf,
    /// Encoder configuration
    pub encoder_config: EncoderConfig,
    /// Whether to record audio
    pub enable_audio: bool,
    /// Optional audio device path
    pub audio_device: Option<&'a str>,
    /// Native sample rate (Hz) of the selected audio source. Used to pin the
    /// audio capsfilter so PulseAudio passes through unchanged when Opus
    /// accepts the rate. A defensive resampler handles negotiation exceptions.
    /// `0` means "unknown" and falls back to 48 kHz.
    pub audio_source_rate_hz: u32,
    /// Specific encoder info (if None, auto-select)
    pub encoder_info: Option<&'a crate::media::encoders::video::EncoderInfo>,
    /// Immutable metadata captured at recording start.
    pub capture_metadata: crate::pipelines::capture_metadata::CaptureMetadata,
    /// Pre-created shared audio levels handle (UI reads this for live meters)
    pub audio_levels: SharedAudioLevels,
}

/// Appsrc-specific recording configuration (libcamera backend).
///
/// Frames are pushed from the application via a `tokio::sync::mpsc` channel
/// instead of using `pipewiresrc`. This avoids camera contention when the
/// native libcamera pipeline already holds the device.
pub struct AppsrcRecorderConfig<'a> {
    /// Common recording settings
    pub base: RecorderConfig<'a>,
    /// Pixel format of incoming frames
    pub pixel_format: crate::backends::camera::types::PixelFormat,
    /// Live filter code (read each frame via AtomicU32, updated by UI thread).
    /// Value is `FilterType::gpu_filter_code()`. 0 = Standard (no filter).
    pub live_filter_code: Arc<std::sync::atomic::AtomicU32>,
    /// Desired zoom sampled for each frame; native-applied zoom is not repeated.
    pub live_zoom: Arc<ZoomControl>,
}

/// Video recorder using the new pipeline architecture
#[derive(Debug)]
pub struct VideoRecorder {
    pipeline: gst::Pipeline,
    file_path: PathBuf,
    /// Lifetime-tied PA source-volume restore. Constructed when a non-default
    /// audio device is configured; dropping (here or when `VideoRecorder`
    /// drops) restores the user's prior PA volume. `None` for default-device
    /// recordings or when the PulseAudio socket is unreachable.
    _pulse_volume_guard: Option<crate::backends::audio::PulseSourceVolumeGuard>,
    /// Cancellation is cooperative: a stuck native call cannot be interrupted.
    /// Dropping this handle never joins the native worker.
    pusher_handle: Option<PusherTask>,
    monitor: RecordingMonitor,
    cleanup_complete: bool,
    _finish: FinishRecording,
}

/// A blocking task's abort handle cannot cancel it once started. Own an explicit
/// cancellation channel instead, including cancellation when the owner is dropped.
#[derive(Debug)]
struct PusherTask {
    worker: tokio::task::JoinHandle<()>,
    cancel: Option<tokio::sync::oneshot::Sender<()>>,
}

impl Drop for PusherTask {
    fn drop(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(());
        }
    }
}

impl std::future::Future for PusherTask {
    type Output = Result<(), tokio::task::JoinError>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        std::pin::Pin::new(&mut self.get_mut().worker).poll(cx)
    }
}

/// Field order guarantees the lease outlives the appsrc's final native unref,
/// including unwinding and cancellation of asynchronous frame processing.
struct PusherResources {
    appsrc: gst_app::AppSrc,
    _lease: super::recording_monitor::RecordingResourceLease,
}

/// Observe errors synchronously before starting any recording-like pipeline.
/// Audio levels still update; errors and EOS remain available on the bus.
/// Callers own phase updates and must publish Cleanup before NULL.
pub fn install_recording_observer(
    pipeline: &gst::Pipeline,
    audio_levels: &SharedAudioLevels,
    monitor: &RecordingMonitor,
) {
    if let Some(pad) = pipeline
        .by_name("recording-encoder")
        .and_then(|encoder| encoder.static_pad("src"))
    {
        let activity = monitor.clone();
        pad.add_probe(gst::PadProbeType::BUFFER, move |_, _| {
            activity.note_output();
            gst::PadProbeReturn::Ok
        });
    }
    let monitor = monitor.clone();
    let weak_pipeline = pipeline.downgrade();
    crate::pipelines::audio_level::install_level_sync_handler_with_observer(
        pipeline,
        audio_levels,
        move |message| {
            if let gst::MessageView::Error(err) = message.view() {
                let source = err
                    .src()
                    .map(|s| s.name().to_string())
                    .unwrap_or_else(|| "pipeline".into());
                monitor.report_error(format!(
                    "{source}: {} ({})",
                    err.error(),
                    err.debug().as_deref().unwrap_or("no debug details")
                ));
            } else if let gst::MessageView::StateChanged(state) = message.view()
                && state.current() == gst::State::Playing
                && weak_pipeline
                    .upgrade()
                    .is_some_and(|pipeline| message.src() == Some(pipeline.upcast_ref()))
            {
                let starting = monitor.subscribe().borrow().phase == RecordingPhase::Starting;
                if starting {
                    monitor.set_phase(RecordingPhase::Recording);
                }
            }
        },
    );
}

fn encoder_input_caps_filter(input: &VideoEncoderInput) -> String {
    let format = input
        .format
        .as_deref()
        .map(|format| format!(",format={format}"))
        .unwrap_or_default();
    format!(
        "! capsfilter caps=video/x-raw,width={},height={}{}",
        input.width, input.height, format
    )
}

fn decoded_video_processing_chain_for_encoder(
    input: &VideoEncoderInput,
    needs_scaling: bool,
) -> String {
    let mut chain = if needs_scaling {
        format!(
            "! videoconvert ! videoscale \
             ! capsfilter caps=video/x-raw,format=I420,width={width},height={height} \
             ! videoconvert",
            width = input.width,
            height = input.height,
        )
    } else {
        "! videoconvert".to_string()
    };

    chain.push(' ');
    chain.push_str(&encoder_input_caps_filter(input));
    chain
}

fn video_tags(metadata: &crate::pipelines::capture_metadata::CaptureMetadata) -> gst::TagList {
    let mut tags = gst::TagList::new();
    let tags = tags.get_mut().expect("new tag list must be writable");
    let replace = gst::TagMergeMode::ReplaceAll;
    let application_name = crate::pipelines::capture_metadata::CaptureMetadata::application_name();

    tags.add::<gst::tags::ApplicationName>(&application_name.as_str(), replace);
    if let Some(version) = metadata.libcamera_version.as_deref() {
        let encoder = format!("libcamera {version}");
        tags.add::<gst::tags::Encoder>(&encoder.as_str(), replace);
    }
    tags.add::<gst::tags::ImageOrientation>(&metadata.orientation.gstreamer_tag(), replace);

    if let Some(make) = metadata.device_make.as_deref() {
        tags.add::<gst::tags::DeviceManufacturer>(&make, replace);
    }
    if let Some(model) = metadata.device_model.as_deref() {
        tags.add::<gst::tags::DeviceModel>(&model, replace);
    }
    if let Some(description) = metadata.description() {
        tags.add::<gst::tags::Description>(&description.as_str(), replace);
    }
    if let Some(captured_at) = metadata.captured_at.as_ref() {
        let captured_at = captured_at.with_timezone(&Utc);
        let seconds =
            captured_at.second() as f64 + f64::from(captured_at.nanosecond()) / 1_000_000_000.0;
        if let Ok(date_time) = gst::DateTime::new(
            0.0f32,
            captured_at.year(),
            captured_at.month() as i32,
            captured_at.day() as i32,
            captured_at.hour() as i32,
            captured_at.minute() as i32,
            seconds,
        ) {
            tags.add::<gst::tags::DateTime>(&date_time, replace);
        }
    }

    tags.to_owned()
}

fn apply_video_tags(
    pipeline: &gst::Pipeline,
    metadata: &crate::pipelines::capture_metadata::CaptureMetadata,
) {
    let Some(muxer) = pipeline.by_name("recording-muxer") else {
        warn!("Recording muxer not found; video metadata was not applied");
        return;
    };
    apply_video_tags_to_muxer(&muxer, metadata);
}

pub(super) fn apply_video_tags_to_muxer(
    muxer: &gst::Element,
    metadata: &crate::pipelines::capture_metadata::CaptureMetadata,
) {
    let Ok(tag_setter) = muxer.clone().dynamic_cast::<gst::TagSetter>() else {
        warn!("Recording muxer does not implement TagSetter; video metadata was not applied");
        return;
    };

    tag_setter.merge_tags(&video_tags(metadata), gst::TagMergeMode::ReplaceAll);
}

/// Restore the selected PA source's prior volume when recording ends.
/// Disabled audio, the default source, and unreachable PA need no guard.
fn build_pulse_volume_guard(
    enable_audio: bool,
    audio_device: Option<&str>,
) -> Option<crate::backends::audio::PulseSourceVolumeGuard> {
    if !enable_audio {
        return None;
    }
    audio_device.and_then(crate::backends::audio::PulseSourceVolumeGuard::normalize_for_recording)
}

/// Prepared encoders, audio branch, timing, and output for appsrc recording.
struct RecorderSetup {
    audio_elements: Option<AudioBranch>,
    encoder_name: String,
    encoder_input: VideoEncoderInput,
    parser_str: String,
    muxer_name: String,
    output_path: PathBuf,
    frame_duration_ns: i64,
}

/// Select encoders, negotiate input, create audio, and resolve the output path.
fn prepare_recorder(
    encoder_info: Option<&crate::media::encoders::video::EncoderInfo>,
    encoder_config: &EncoderConfig,
    enable_audio: bool,
    audio_device: Option<&str>,
    audio_source_rate_hz: u32,
    output_path: PathBuf,
    framerate: u32,
) -> Result<RecorderSetup, String> {
    let encoders = select_encoders(encoder_config, encoder_info, enable_audio)?;

    let audio_elements = if let Some(audio_encoder_config) = encoders.audio {
        Some(VideoRecorder::create_audio_branch(
            audio_device,
            audio_source_rate_hz,
            audio_encoder_config,
        )?)
    } else {
        None
    };

    info!(
        video_codec = ?encoders.video.codec,
        audio = audio_elements.is_some(),
        container = ?encoders.video.container,
        "Selected encoders"
    );

    let frame_duration_ns = 1_000_000_000i64 / framerate as i64;

    let selected_encoder = encoders
        .video
        .encoder
        .factory()
        .map(|f| f.name().to_string())
        .unwrap_or_else(|| "openh264enc".to_string());

    let is_v4l2 = selected_encoder.starts_with("v4l2");
    let selected_input = prepare_video_encoder_input(
        &encoders.video.encoder,
        &selected_encoder,
        encoder_config.width,
        encoder_config.height,
        framerate,
    );

    let (encoder_name, encoder_input, parser_str, muxer_name) = if is_v4l2
        && selected_input.is_err()
    {
        warn!(
            selected = %selected_encoder,
            width = encoder_config.width,
            height = encoder_config.height,
            framerate,
            "V4L2 encoder has no compatible CPU-memory input caps, falling back to openh264enc"
        );
        (
            "openh264enc".to_string(),
            named_encoder_input("openh264enc", encoder_config, framerate)?,
            "! h264parse".to_string(),
            "mp4mux".to_string(),
        )
    } else {
        let selected_input = selected_input?;
        let (parser, muxer) = parser_and_muxer_names(&encoders.video);
        // Probe hardware encoders to catch cases where the element exists in
        // the registry but can't actually encode (e.g. VA-API backed by NVENC
        // in a flatpak sandbox that lacks libnvidia-encode.so).
        let is_software = selected_encoder == "openh264enc"
            || selected_encoder == "x264enc"
            || selected_encoder == "x265enc";
        if !is_software
            && !is_v4l2
            && !crate::media::encoders::detection::probe_single_encoder(&selected_encoder)
        {
            warn!(
                selected = %selected_encoder,
                "Hardware encoder probe failed, falling back to openh264enc"
            );
            (
                "openh264enc".to_string(),
                named_encoder_input("openh264enc", encoder_config, framerate)?,
                "! h264parse".to_string(),
                "mp4mux".to_string(),
            )
        } else {
            if let Some(format) = selected_input.format.as_deref() {
                info!(encoder = %selected_encoder, format, "Using negotiated V4L2 encoder input");
            }
            (selected_encoder, selected_input, parser, muxer)
        }
    };
    let extension = if encoder_name == "openh264enc" {
        "mp4"
    } else {
        encoders.video.extension
    };
    let output_path = output_path.with_extension(extension);

    Ok(RecorderSetup {
        audio_elements,
        encoder_name,
        encoder_input,
        parser_str,
        muxer_name,
        output_path,
        frame_duration_ns,
    })
}

fn named_encoder_input(
    encoder_name: &str,
    encoder_config: &EncoderConfig,
    framerate: u32,
) -> Result<VideoEncoderInput, String> {
    let encoder = gst::ElementFactory::make(encoder_name)
        .build()
        .map_err(|e| format!("Failed to create {encoder_name}: {e}"))?;
    prepare_video_encoder_input(
        &encoder,
        encoder_name,
        encoder_config.width,
        encoder_config.height,
        framerate,
    )
}

/// Extract parser name (with `! ` prefix) and muxer name from a selected video encoder.
fn parser_and_muxer_names(video: &SelectedVideoEncoder) -> (String, String) {
    let parser = video
        .parser
        .as_ref()
        .and_then(|p| p.factory().map(|f| format!("! {}", f.name())))
        .unwrap_or_default();
    let muxer = video
        .muxer
        .factory()
        .map(|f| f.name().to_string())
        .unwrap_or_else(|| "mp4mux".to_string());
    (parser, muxer)
}

/// Read `CLOCK_BOOTTIME` in nanoseconds (same clock domain as libcamera
/// sensor timestamps).
fn read_clock_boottime_ns() -> u64 {
    use std::mem::MaybeUninit;
    unsafe {
        let mut ts = MaybeUninit::<libc::timespec>::uninit();
        if libc::clock_gettime(libc::CLOCK_BOOTTIME, ts.as_mut_ptr()) != 0 {
            return 0;
        }
        let ts = ts.assume_init();
        ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
    }
}

/// Add audio branch elements to the pipeline, link the chain, connect to
/// the muxer, and install the level sync handler.
fn add_audio_branch_to_pipeline(
    pipeline: &gst::Pipeline,
    audio_branch: &AudioBranch,
    audio_levels: &SharedAudioLevels,
) -> Result<(), String> {
    let chain = [
        &audio_branch.source,
        &audio_branch.queue,
        &audio_branch.convert,
        &audio_branch.resample,
        &audio_branch.capsfilter,
        &audio_branch.compressor,
        &audio_branch.makeup_gain,
        &audio_branch.limiter,
        &audio_branch.level,
        &audio_branch.encoder,
    ];
    pipeline
        .add_many(chain)
        .map_err(|e| format!("Failed to add audio elements to pipeline: {}", e))?;

    gst::Element::link_many(chain).map_err(|_| "Failed to link audio chain")?;

    let muxer = pipeline
        .by_name("recording-muxer")
        .ok_or("Failed to find recording-muxer for audio linking")?;
    link_audio_to_muxer(&audio_branch.encoder, &muxer)?;

    install_shared_level_sync_handler(pipeline, audio_levels);

    Ok(())
}

/// Bound the appsrc queue used by live recording.
///
/// `push_buffer` does not automatically stop accepting data after appsrc emits
/// `enough-data`. Without an explicit leaky limit, a slow encoder can therefore
/// retain every full-resolution frame until the process is killed by the OOM
/// killer. Dropping the oldest queued frame keeps latency and memory bounded.
fn configure_live_appsrc(appsrc: &gst_app::AppSrc) {
    appsrc.set_max_buffers(APPSRC_MAX_BUFFERS);
    appsrc.set_max_bytes(0);
    appsrc.set_block(false);
    appsrc.set_leaky_type(gst_app::AppLeakyType::Downstream);
}

/// Return whether appsrc has reached its configured live-frame limit.
///
/// Checking before format conversion avoids spending CPU/GPU time on a frame
/// that cannot enter the encoding pipeline yet. The next captured frame gets
/// another chance as soon as downstream frees a slot.
fn appsrc_queue_is_full(appsrc: &gst_app::AppSrc) -> bool {
    queue_at_capacity(appsrc.current_level_buffers(), appsrc.max_buffers())
}

fn queue_at_capacity(current_buffers: u64, max_buffers: u64) -> bool {
    max_buffers > 0 && current_buffers >= max_buffers
}

fn fallback_pts(capture_index: u64, frame_duration_ns: u64) -> u64 {
    capture_index * frame_duration_ns
}

/// Parse a GStreamer pipeline description and perform common configuration.
///
/// Returns the pipeline and appsrc element after:
/// - Parsing the pipeline description
/// - Extracting the `camera-appsrc` element
/// - Configuring the video encoder (bitrate / quality)
/// - Adding the audio branch (if present)
/// - Installing muxer fixup probes
fn build_recorder_pipeline(
    pipeline_desc: &str,
    encoder_name: &str,
    encoder_config: &EncoderConfig,
    encode_width: u32,
    encode_height: u32,
    audio_elements: Option<&AudioBranch>,
    audio_levels: &SharedAudioLevels,
) -> Result<(gst::Pipeline, gst_app::AppSrc), String> {
    let pipeline = gst::parse::launch(pipeline_desc)
        .map_err(|e| format!("Failed to parse pipeline: {}", e))?
        .dynamic_cast::<gst::Pipeline>()
        .map_err(|_| "Failed to cast to Pipeline")?;

    let appsrc = pipeline
        .by_name("camera-appsrc")
        .ok_or("Failed to find camera-appsrc in pipeline")?
        .dynamic_cast::<gst_app::AppSrc>()
        .map_err(|_| "Failed to cast to AppSrc")?;
    configure_live_appsrc(&appsrc);

    if let Some(enc_element) = pipeline.by_name("recording-encoder") {
        crate::media::encoders::video::configure_video_encoder(
            &enc_element,
            encoder_name,
            encoder_config.video_quality,
            encode_width,
            encode_height,
            encoder_config.bitrate_override_kbps,
        );
    }

    if let Some(audio_branch) = audio_elements {
        add_audio_branch_to_pipeline(&pipeline, audio_branch, audio_levels)?;
        info!("Audio branch added to recording pipeline");
    }

    install_muxer_fixup_probes(&pipeline);

    Ok((pipeline, appsrc))
}

/// Result of PTS computation for a single frame.
enum PtsResult {
    /// Computed PTS in nanoseconds — push this buffer.
    Pts(u64),
    /// Frame should be skipped (pipeline not yet PLAYING).
    Skip,
}

/// Compute PTS for a recording frame using sensor timestamps and pipeline
/// running-time for A/V sync. Falls back to frame-count-based PTS if
/// sensor timestamps are unavailable.
fn compute_pts(
    appsrc: &gst_app::AppSrc,
    sensor_ts: Option<u64>,
    capture_index: u64,
    frame_duration_ns: u64,
    pipeline_playing: &mut bool,
    ts_offset: &mut Option<(u64, u64)>,
) -> PtsResult {
    let Some(ts) = sensor_ts else {
        return PtsResult::Pts(fallback_pts(capture_index, frame_duration_ns));
    };

    // A clock can be assigned before PLAYING, while base_time is still zero.
    // Its running time is then system uptime, not the recording timeline.
    if !*pipeline_playing {
        if appsrc.current_state() != gst::State::Playing || appsrc.current_running_time().is_none()
        {
            RECORDING_STATS
                .pusher_skipped
                .fetch_add(1, Ordering::Relaxed);
            return PtsResult::Skip;
        }
        *pipeline_playing = true;
        info!("Pipeline is PLAYING, starting video capture");
    }
    let rt = match appsrc.current_running_time() {
        Some(t) => t.nseconds(),
        None => {
            RECORDING_STATS
                .pusher_skipped
                .fetch_add(1, Ordering::Relaxed);
            return PtsResult::Skip;
        }
    };

    // Record processing delay for diagnostics
    let now_boot = read_clock_boottime_ns();
    let processing_delay = now_boot.saturating_sub(ts);
    RECORDING_STATS
        .last_processing_delay_us
        .store(processing_delay / 1_000, Ordering::Relaxed);

    // On first frame, establish PTS base accounting for processing
    // delay so video timestamps reflect actual capture time.
    let is_first = ts_offset.is_none();
    let (first_ts, pts_base) = *ts_offset.get_or_insert((ts, rt.saturating_sub(processing_delay)));
    let pts = pts_base + ts.saturating_sub(first_ts);
    if is_first {
        info!(
            running_time_ms = rt / 1_000_000,
            processing_delay_ms = processing_delay / 1_000_000,
            pts_base_ms = pts_base / 1_000_000,
            pts_ms = pts / 1_000_000,
            sensor_ts_ms = ts / 1_000_000,
            "First video frame A/V sync: pts_base = running_time - processing_delay"
        );
    }
    PtsResult::Pts(pts)
}

/// Install muxer sink pad probes that fix PTS=NONE (copies DTS → PTS).
///
/// NVENC encoders (nvh265enc/nvh264enc) add a 3 600 000 s offset to PTS/DTS
/// **and** to the segment event.  The aggregator-based mp4mux in GStreamer 1.28
/// converts PTS to running-time via `PTS − segment.start`, so the offset
/// cancels out.  Stripping the offset from buffers without also adjusting the
/// segment causes the muxer to clip every video buffer as "outside segment",
/// resulting in 0 video samples in the output file.
fn install_muxer_fixup_probes(pipeline: &gst::Pipeline) {
    let Some(muxer) = pipeline.by_name("recording-muxer") else {
        return;
    };
    for pad in muxer.sink_pads() {
        let pad_name = pad.name().to_string();
        let mux_probe_count = std::sync::Arc::new(AtomicU64::new(0));
        let mpc = mux_probe_count.clone();
        pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
            if let Some(buffer) = info.buffer_mut() {
                let buf = buffer.make_mut();
                // Fix PTS=NONE (some encoders set only DTS)
                if buf.pts().is_none()
                    && let Some(dts) = buf.dts()
                {
                    buf.set_pts(dts);
                }
            }
            let n = mpc.fetch_add(1, Ordering::Relaxed);
            if n < 3
                && let Some(buffer) = info.buffer()
            {
                debug!(
                    pad = pad_name.as_str(),
                    frame = n,
                    pts_ms = buffer.pts().map(|p| p.mseconds()),
                    dts_ms = buffer.dts().map(|d| d.mseconds()),
                    "Muxer sink pad buffer"
                );
            }
            gst::PadProbeReturn::Ok
        });
    }
}

/// Wrap tightly-packed RGBA frame storage directly for Standard-filter recording.
///
/// The GStreamer buffer owns a clone of [`FrameData`], so the underlying shared
/// allocation stays alive until downstream releases the buffer. Filtered frames,
/// padded RGBA, and non-RGBA formats return `None` and use the processing path.
fn try_shared_rgba_buffer(
    frame: &CameraFrame,
    filter_type: crate::app::FilterType,
) -> Option<gst::Buffer> {
    let row_bytes = frame.width.checked_mul(4)?;
    let expected_len = row_bytes.checked_mul(frame.height)? as usize;

    (filter_type == crate::app::FilterType::Standard
        && frame.format == PixelFormat::RGBA
        && frame.stride == row_bytes
        && frame.data.len() == expected_len)
        .then(|| gst::Buffer::from_slice(frame.data.clone()))
}

/// Apply residual zoom per buffer, never by mutating a downstream element:
/// queued frames must retain their own geometry and appsrc caps stay fixed.
async fn recording_frame_buffer(
    frame: &CameraFrame,
    filter_type: crate::app::FilterType,
    zoom: f32,
) -> Result<gst::Buffer, String> {
    let zoom = if zoom.is_finite() { zoom.max(1.0) } else { 1.0 };
    if zoom == 1.0
        && let Some(buffer) = try_shared_rgba_buffer(frame, filter_type)
    {
        return Ok(buffer);
    }
    let rgba = apply_rgba_zoom(
        convert_frame_to_rgba(frame).await?,
        frame.width,
        frame.height,
        zoom,
    )?;
    let output = if filter_type == crate::app::FilterType::Standard {
        rgba
    } else {
        match crate::shaders::apply_filter_gpu_rgba(&rgba, frame.width, frame.height, filter_type)
            .await
        {
            Ok(data) => data,
            Err(e) => {
                warn!(error = %e, "Failed to apply filter, using unfiltered RGBA");
                rgba
            }
        }
    };
    Ok(gst::Buffer::from_mut_slice(output))
}

/// Center-crop and upscale one RGBA buffer, retaining its original dimensions.
/// `zoom` is residual software zoom, not the camera's total desired zoom.
fn apply_rgba_zoom(rgba: Vec<u8>, width: u32, height: u32, zoom: f32) -> Result<Vec<u8>, String> {
    if !zoom.is_finite() || zoom <= 1.0 {
        return Ok(rgba);
    }
    if width == 0 || height == 0 {
        return Err("Empty recording frame".into());
    }
    let source =
        image::RgbaImage::from_raw(width, height, rgba).ok_or("Invalid recording RGBA layout")?;
    let crop_width = ((width as f32 / zoom).floor() as u32).clamp(1, width);
    let crop_height = ((height as f32 / zoom).floor() as u32).clamp(1, height);
    let cropped = image::imageops::crop_imm(
        &source,
        (width - crop_width) / 2,
        (height - crop_height) / 2,
        crop_width,
        crop_height,
    );
    Ok(image::imageops::resize(
        &cropped.to_image(),
        width,
        height,
        image::imageops::FilterType::Triangle,
    )
    .into_raw())
}

/// Convert and apply only zoom not already baked into this frame by the ISP.
/// Output stays tightly packed at source dimensions for fixed-caps sinks.
pub(crate) async fn convert_frame_to_zoomed_rgba(
    frame: &CameraFrame,
    desired_zoom: f32,
) -> Result<Vec<u8>, String> {
    apply_rgba_zoom(
        convert_frame_to_rgba(frame).await?,
        frame.width,
        frame.height,
        residual_zoom(desired_zoom, frame),
    )
}

/// Convert a camera frame to tightly-packed RGBA using the GPU compute shader.
///
/// For frames already in RGBA format, strips stride padding.
/// For YUV and other formats, uses the GPU compute pipeline.
pub(crate) async fn convert_frame_to_rgba(frame: &CameraFrame) -> Result<Vec<u8>, String> {
    if frame.format == PixelFormat::RGBA {
        let row_bytes = (frame.width * 4) as usize;
        let stride = frame.stride as usize;
        if stride <= row_bytes {
            return Ok(frame.data.to_vec());
        }
        let mut out = Vec::with_capacity(row_bytes * frame.height as usize);
        for y in 0..frame.height as usize {
            out.extend_from_slice(&frame.data[y * stride..y * stride + row_bytes]);
        }
        return Ok(out);
    }

    let input = crate::shaders::GpuFrameInput::from_camera_frame(frame)?;

    let mut pipeline_guard = crate::shaders::get_gpu_convert_pipeline()
        .await
        .map_err(|e| format!("Failed to get GPU convert pipeline: {}", e))?;

    let pipeline = pipeline_guard
        .as_mut()
        .ok_or("GPU convert pipeline not initialized")?;

    // Photo quality selection must not move realtime Bayer recording onto
    // CPU demosaic. Retain the shared float core's GPU bilinear route.
    let result = if frame.format.is_bayer() {
        pipeline
            .convert_bayer_with_method(&input, crate::shaders::DemosaicMethod::Bilinear)
            .map_err(|e| e.to_string())
    } else {
        pipeline.convert(&input)
    };
    result.map_err(|e| format!("GPU conversion failed: {}", e))?;

    pipeline
        .read_rgba_to_cpu(frame.width, frame.height)
        .await
        .map_err(|e| format!("Failed to read RGBA from GPU: {}", e))
}

impl VideoRecorder {
    /// Create an appsrc-based video recorder for the libcamera backend.
    ///
    /// Frames from the native capture pipeline are received via `frame_rx` and
    /// pushed into a GStreamer encoding pipeline through `appsrc`. The preview
    /// continues uninterrupted because the same frames are displayed in the UI
    /// and forwarded here.
    ///
    /// The returned recorder is already started. When the `frame_rx`
    /// channel closes (sender dropped), the pusher task sends EOS and the
    /// pipeline finalizes gracefully.
    pub fn new_from_appsrc(
        config: AppsrcRecorderConfig<'_>,
        frame_rx: tokio::sync::mpsc::Receiver<RecordingFrame>,
    ) -> Result<Self, String> {
        Self::new_from_appsrc_monitored(config, frame_rx, RecordingMonitor::new("automatic".into()))
    }

    /// Construct and eagerly start, publishing failures before blocking cleanup.
    pub fn new_from_appsrc_monitored(
        config: AppsrcRecorderConfig<'_>,
        frame_rx: tokio::sync::mpsc::Receiver<RecordingFrame>,
        monitor: RecordingMonitor,
    ) -> Result<Self, String> {
        monitor.enable_activity_watchdog();
        monitor.set_phase(RecordingPhase::Starting);
        let result = Self::new_from_appsrc_inner(config, frame_rx, monitor.clone());
        if let Err(detail) = &result {
            monitor.report_error(detail.clone());
            // The inner call has released all constructor-owned resources,
            // even if it failed before constructing the recorder's drop guard.
            // finish() still waits for any independently owned pusher leases.
            monitor.set_phase(RecordingPhase::Cleanup);
            monitor.finish();
        }
        result
    }

    fn new_from_appsrc_inner(
        config: AppsrcRecorderConfig<'_>,
        frame_rx: tokio::sync::mpsc::Receiver<RecordingFrame>,
        monitor: RecordingMonitor,
    ) -> Result<Self, String> {
        let AppsrcRecorderConfig {
            base:
                RecorderConfig {
                    width,
                    height,
                    framerate,
                    output_path,
                    encoder_config,
                    enable_audio,
                    audio_device,
                    audio_source_rate_hz,
                    encoder_info,
                    capture_metadata,
                    audio_levels,
                },
            pixel_format,
            live_filter_code,
            live_zoom,
        } = config;

        // Keep stable RGBA appsrc caps for the whole recording so live filter
        // changes never renegotiate the GStreamer pipeline. Standard-filter
        // frames may still share tightly-packed RGBA storage directly.
        let initial_filter_code = live_filter_code.load(std::sync::atomic::Ordering::Relaxed);

        info!(
            width,
            height,
            framerate,
            format = ?pixel_format,
            initial_filter = initial_filter_code,
            output = %output_path.display(),
            audio = enable_audio,
            audio_device = ?audio_device,
            orientation = capture_metadata.orientation.gstreamer_tag(),
            "Creating appsrc-based video recorder (libcamera backend)"
        );

        // Boost the PA source to 100% before pulsesrc opens — see
        // `PulseSourceVolumeGuard`. Created here so it lives at least as long
        // as the recorder pipeline; dropped automatically when the
        // `VideoRecorder` itself drops (i.e. when recording stops).
        let pulse_volume_guard = build_pulse_volume_guard(enable_audio, audio_device);

        let setup = prepare_recorder(
            encoder_info,
            &encoder_config,
            enable_audio,
            audio_device,
            audio_source_rate_hz,
            output_path,
            framerate,
        )?;

        let (base_width, base_height) = (width, height);
        monitor.set_encoder(setup.encoder_name.clone());
        let (final_width, final_height) = (setup.encoder_input.width, setup.encoder_input.height);

        // Only insert videoscale/capsfilter when downscaling is needed.
        // Capture orientation is stored as container metadata, so recording never
        // rotates or mirrors full video frames in software.
        let needs_scaling = final_width != base_width || final_height != base_height;

        // Always use RGBA input: the filtered pusher converts each frame to RGBA
        // (via GPU compute shader), applies the current filter, and pushes RGBA.
        // This lets the user toggle filters mid-recording.
        let initial_gst_format = "RGBA";

        let processing_chain =
            decoded_video_processing_chain_for_encoder(&setup.encoder_input, needs_scaling);

        let pipeline_desc = format!(
            "appsrc name=camera-appsrc \
               caps=video/x-raw,format={fmt},width={w},height={h},framerate={fps}/1 \
               is-live=true do-timestamp=false format=time \
               min-latency={lat} max-latency={lat} \
             ! queue max-size-buffers=5 max-size-time=1000000000 \
             {processing} \
             ! {encoder} name=recording-encoder \
             {parser} \
             ! {muxer} name=recording-muxer \
             ! filesink location={loc}",
            fmt = initial_gst_format,
            w = width,
            h = height,
            fps = framerate,
            lat = setup.frame_duration_ns,
            processing = processing_chain,
            encoder = setup.encoder_name,
            parser = setup.parser_str,
            muxer = setup.muxer_name,
            loc = setup.output_path.display(),
        );

        info!(desc = %pipeline_desc, "Launching appsrc pipeline");

        let (pipeline, appsrc) = build_recorder_pipeline(
            &pipeline_desc,
            &setup.encoder_name,
            &encoder_config,
            final_width,
            final_height,
            setup.audio_elements.as_ref(),
            &audio_levels,
        )?;
        apply_video_tags(&pipeline, &capture_metadata);
        install_recording_observer(&pipeline, &audio_levels, &monitor);

        info!(
            initial_filter = initial_filter_code,
            "Pusher will apply live GPU filter (RGBA output)"
        );
        let pusher_handle = Self::spawn_filtered_pusher(
            appsrc,
            frame_rx,
            framerate,
            live_filter_code,
            live_zoom,
            monitor.clone(),
        );

        // Publish diagnostics for the insights drawer
        let mode = if needs_scaling {
            "Filtered RGBA (videoconvert + scale)"
        } else {
            "Filtered RGBA (videoconvert)"
        };
        publish_recording_diagnostics(RecordingDiagnostics {
            mode: mode.to_string(),
            pipeline_string: pipeline_desc.clone(),
            encoder: setup.encoder_name.clone(),
            resolution: format!("{}x{}", final_width, final_height),
            framerate,
        });

        let recorder = VideoRecorder {
            pipeline,
            file_path: setup.output_path,
            _pulse_volume_guard: pulse_volume_guard,
            pusher_handle: Some(pusher_handle),
            monitor: monitor.clone(),
            cleanup_complete: false,
            _finish: FinishRecording(monitor.clone()),
        };

        // Eagerly start so synchronous encoder failures reach the caller.
        recorder.start()?;

        Ok(recorder)
    }

    /// Spawn a pusher task that shares Standard-filter RGBA frames directly or
    /// converts and filters frames before pushing them to the same RGBA appsrc.
    ///
    /// Reads the current filter code from `live_filter_code` each frame so
    /// filter changes during recording are reflected in the output file.
    /// When filter code is 0 (Standard), the RGBA data is pushed without
    /// running the filter shader.
    fn spawn_filtered_pusher(
        appsrc: gst_app::AppSrc,
        frame_rx: tokio::sync::mpsc::Receiver<RecordingFrame>,
        framerate: u32,
        live_filter_code: Arc<std::sync::atomic::AtomicU32>,
        live_zoom: Arc<ZoomControl>,
        monitor: RecordingMonitor,
    ) -> PusherTask {
        let resources = PusherResources {
            appsrc,
            _lease: monitor.resource_lease(),
        };
        let runtime = tokio::runtime::Handle::current();
        let (cancel, cancelled) = tokio::sync::oneshot::channel();
        let worker = tokio::task::spawn_blocking(move || {
            // Native push, EOS, queries and final unref stay off libcosmic's
            // sole async worker. GPU conversion still awaits without busy-waiting.
            runtime.block_on(async {
                tokio::select! {
                    biased;
                    _ = cancelled => {}
                    _ = Self::run_filtered_pusher(
                        &resources.appsrc, frame_rx, framerate, live_filter_code, live_zoom, monitor,
                    ) => {}
                }
            });
            drop(resources);
        });
        PusherTask {
            worker,
            cancel: Some(cancel),
        }
    }

    async fn run_filtered_pusher(
        appsrc: &gst_app::AppSrc,
        mut frame_rx: tokio::sync::mpsc::Receiver<RecordingFrame>,
        framerate: u32,
        live_filter_code: Arc<std::sync::atomic::AtomicU32>,
        live_zoom: Arc<ZoomControl>,
        monitor: RecordingMonitor,
    ) {
        let initial = live_filter_code.load(std::sync::atomic::Ordering::Relaxed);
        info!(
            initial_filter_code = initial,
            "Filtered appsrc pusher task started"
        );

        let start_epoch_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        RECORDING_STATS
            .pusher_start_epoch_ns
            .store(start_epoch_ns, Ordering::Relaxed);

        let mut frame_count: u64 = 0;
        let mut capture_frame_count: u64 = 0;
        let start_time = std::time::Instant::now();
        let frame_duration_ns = 1_000_000_000u64 / framerate as u64;
        let mut pipeline_playing = false;
        let mut ts_offset: Option<(u64, u64)> = None;

        while let Some(rec_frame) = frame_rx.recv().await {
            // Offer is activity even if downstream backpressure drops it.
            monitor.note_input();
            let frame = rec_frame;

            let capture_index = capture_frame_count;
            capture_frame_count += 1;
            if appsrc_queue_is_full(appsrc) {
                RECORDING_STATS
                    .pusher_skipped
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }

            let sensor_ts = frame.sensor_timestamp_ns;
            let sequence = frame.libcamera_metadata.as_ref().and_then(|m| m.sequence);

            // Read the live filter for every frame so Standard ↔ filtered
            // switching remains seamless within one fixed-caps pipeline.
            let filter_code = live_filter_code.load(std::sync::atomic::Ordering::Relaxed);
            let filter_type = crate::app::FilterType::from_gpu_filter_code(filter_code);

            let t0 = std::time::Instant::now();
            let zoom = residual_zoom(live_zoom.zoom(), &frame);
            let mut buffer = match recording_frame_buffer(&frame, filter_type, zoom).await {
                Ok(buffer) => buffer,
                Err(e) => {
                    warn!(error = %e, "Failed to process recording frame, skipping");
                    continue;
                }
            };

            RECORDING_STATS
                .last_convert_time_us
                .store(t0.elapsed().as_micros() as u64, Ordering::Relaxed);

            let pts_ns = match compute_pts(
                appsrc,
                sensor_ts,
                capture_index,
                frame_duration_ns,
                &mut pipeline_playing,
                &mut ts_offset,
            ) {
                PtsResult::Pts(pts) => pts,
                PtsResult::Skip => continue,
            };

            {
                let buf_ref = buffer.get_mut().unwrap();
                buf_ref.set_pts(gst::ClockTime::from_nseconds(pts_ns));
                buf_ref.set_duration(gst::ClockTime::from_nseconds(frame_duration_ns));
            }

            RECORDING_STATS.last_pts_ns.store(pts_ns, Ordering::Relaxed);

            if appsrc.push_buffer(buffer).is_err() {
                warn!("Filtered pusher: failed to push buffer, stopping");
                break;
            }

            RECORDING_STATS
                .pusher_pushed
                .fetch_add(1, Ordering::Relaxed);
            frame_count += 1;
            if frame_count.is_multiple_of(LOG_EVERY_N_FRAMES) {
                let elapsed = start_time.elapsed().as_secs_f64();
                debug!(
                    frames = frame_count,
                    seq = ?sequence,
                    sensor_ts_ms = ?sensor_ts.map(|t| t / 1_000_000),
                    pts_ms = pts_ns / 1_000_000,
                    elapsed_secs = format!("{:.1}", elapsed),
                    effective_fps = format!("{:.1}", frame_count as f64 / elapsed),
                    filter_time_us = t0.elapsed().as_micros(),
                    "Filtered pusher progress"
                );
            }
        }

        info!(
            total_frames = frame_count,
            "Frame channel closed, sending EOS to filtered appsrc"
        );
        monitor.set_phase(RecordingPhase::Finalizing);
        let _ = appsrc.end_of_stream();
    }

    /// Create audio branch elements
    ///
    /// Uses `pulsesrc` (PipeWire's PulseAudio compatibility layer) for reliable
    /// audio capture from all device types including pro-audio (multi-channel)
    /// and standard stereo/mono sources.
    ///
    /// All input channels are mixed down to mono via a capsfilter. The same
    /// capsfilter pins the sample rate to `opus_target_rate(audio_source_rate_hz)`
    /// — the source's native rate when Opus can encode it, else 48 kHz.
    /// PulseAudio normally converts internally; a defensive `audioresample`
    /// handles sources that cannot negotiate that rate.
    fn create_audio_branch(
        audio_device: Option<&str>,
        audio_source_rate_hz: u32,
        audio_encoder_config: crate::media::encoders::audio::SelectedAudioEncoder,
    ) -> Result<AudioBranch, String> {
        let mut source_builder = gst::ElementFactory::make("pulsesrc")
            // Resample against the pipeline clock so audio stays synchronized
            // without `skew` advancing the capture pointer and dropping mic
            // samples when video processing temporarily starves the process.
            //
            // The value is sourced from `PULSESRC_SLAVE_METHOD` so the
            // pre-recording probe in `audio_probe.rs` configures pulsesrc
            // identically.
            .property_from_str("slave-method", PULSESRC_SLAVE_METHOD)
            .property("buffer-time", PULSESRC_BUFFER_TIME_US)
            .property("provide-clock", false);

        // pulsesrc `device` property takes the PipeWire/PulseAudio node name
        // (e.g. "alsa_input.usb-Focusrite_Scarlett_4i4_4th_Gen_...-00.pro-input-0")
        if let Some(device) = audio_device {
            if !device.is_empty() {
                info!(device = %device, "Using audio source device");
                source_builder = source_builder.property("device", device);
            }
        } else {
            info!("Using default audio source");
        }

        let source = source_builder
            .build()
            .map_err(|e| format!("Failed to create audio source: {}", e))?;

        let queue = gst::ElementFactory::make("queue")
            .property("max-size-buffers", 200u32)
            .property("max-size-time", 2_000_000_000u64)
            .build()
            .map_err(|e| format!("Failed to create audio queue: {}", e))?;

        let convert = gst::ElementFactory::make("audioconvert")
            .build()
            .map_err(|e| format!("Failed to create audioconvert: {}", e))?;

        // Defensive `audioresample`: in the steady-state case the source
        // already emits at the rate `opus_target_rate` requested (PA negotiates
        // it server-side, see the capsfilter below), and this element acts as a
        // zero-cost pass-through. It exists for the default-source edge case
        // where `audio_source_rate_hz` is 0 (unknown), `opus_target_rate`
        // defaults to 48 kHz, and PA happens to serve a card at a rate it
        // won't or can't resample (some PA configs disable resampling). In
        // that case GStreamer needs an in-pipeline resampler or the audio
        // branch fails to negotiate.
        let resample = gst::ElementFactory::make("audioresample")
            .build()
            .map_err(|e| format!("Failed to create audioresample: {}", e))?;

        // Force mono output + an Opus-compatible sample rate. `opus_target_rate`
        // returns the source's native rate when Opus accepts it (no resampling
        // anywhere); otherwise 48 kHz, in which case either PulseAudio resamples
        // internally or the `audioresample` element above picks up the slack.
        let target_rate =
            crate::media::encoders::audio::opus_target_rate(audio_source_rate_hz) as i32;
        let capsfilter = gst::ElementFactory::make("capsfilter")
            .property(
                "caps",
                gst::Caps::builder("audio/x-raw")
                    .field("channels", 1i32)
                    .field("rate", target_rate)
                    .build(),
            )
            .build()
            .map_err(|e| format!("Failed to create audio capsfilter: {}", e))?;
        info!(
            source_rate_hz = audio_source_rate_hz,
            target_rate_hz = target_rate,
            "Audio capsfilter rate negotiated"
        );

        // Soft-knee downward compressor — squashes loud peaks so the following
        // makeup-gain stage can lift quiet content without clipping. Without
        // this, on-camera mics with low sensitivity (USB webcams) or PA
        // sources that get attenuated by their ALSA UCM profile (e.g. the
        // Pixel 3a's VoiceCall mic clamped at -30 dB) record audio at
        // -40 LUFS or worse.
        //
        // gstreamer1.0-plugins-good `audiodynamic`:
        // - mode=compressor, characteristics=soft-knee
        // - threshold is normalized [0..1] amplitude; 0.2 ≈ -14 dBFS.
        // - ratio is the *output* slope above threshold: 0.3 ≈ 3:1 ratio
        //   (signals 12 dB over threshold land 4 dB over).
        // NB: `audiodynamic.threshold` / `.ratio` are GLib `gfloat` (f32),
        // not `gdouble` — passing f64 triggers a runtime "can't be set from
        // the given type" GObject type-mismatch panic. Parameter values are
        // shared with the settings probe via `pipelines::audio_level::dynamics`.
        use crate::pipelines::audio_level::dynamics;
        let compressor = gst::ElementFactory::make("audiodynamic")
            .property_from_str("mode", "compressor")
            .property_from_str("characteristics", "soft-knee")
            .property("threshold", dynamics::COMPRESSOR_THRESHOLD)
            .property("ratio", dynamics::COMPRESSOR_RATIO)
            .build()
            .map_err(|e| format!("Failed to create audio compressor: {}", e))?;

        // Makeup gain — lifts the compressed signal back to a sane recording
        // level. Linear scale: 2.0 ≈ +6 dB. With the upstream
        // `PulseSourceVolumeGuard` already boosting PA to 100%, +6 dB sits
        // safely below the brick-wall limiter that follows; on platforms
        // without `pactl` (so PA is wherever the user left it) +6 dB still
        // provides a noticeable lift.
        let makeup_gain = gst::ElementFactory::make("volume")
            .property("volume", dynamics::MAKEUP_GAIN)
            .build()
            .map_err(|e| format!("Failed to create makeup-gain element: {}", e))?;

        // Brick-wall limiter after makeup gain. Catches transients that the
        // 3:1 compressor lets slip through, capping output around -0.45 dBFS
        // (threshold 0.95 → 20·log10(0.95)). Ratio 0.05 ≈ 20:1, effectively
        // a limiter; characteristics=hard-knee for the sharpest cutoff.
        // Without this the boosted Pixel 3a recordings clip at +4 dBFS.
        let limiter = gst::ElementFactory::make("audiodynamic")
            .property_from_str("mode", "compressor")
            .property_from_str("characteristics", "hard-knee")
            .property("threshold", dynamics::LIMITER_THRESHOLD)
            .property("ratio", dynamics::LIMITER_RATIO)
            .build()
            .map_err(|e| format!("Failed to create audio limiter: {}", e))?;

        // Single level meter AFTER compressor + makeup gain — the UI then
        // reads the actual recorded signal level, not the raw mic level.
        let level = gst::ElementFactory::make("level")
            .name("audio-level-output")
            .property("post-messages", true)
            .property("interval", 100_000_000u64) // 100ms
            .build()
            .map_err(|e| format!("Failed to create level meter: {}", e))?;

        let encoder = audio_encoder_config.encoder;

        Ok(AudioBranch {
            source,
            queue,
            convert,
            resample,
            capsfilter,
            compressor,
            makeup_gain,
            limiter,
            level,
            encoder,
        })
    }

    /// Start recording (idempotent — no-op if already playing)
    pub fn start(&self) -> Result<(), String> {
        // Starting an already-playing recorder is idempotent.
        if self.pipeline.current_state() == gst::State::Playing {
            info!("Pipeline already playing, skipping start");
            return Ok(());
        }

        info!("Starting video recording pipeline");

        // Log pipeline element names for diagnostics
        let mut element_names = Vec::new();
        for e in self.pipeline.iterate_elements().into_iter().flatten() {
            element_names.push(e.name().to_string());
        }
        info!(elements = ?element_names, "Pipeline elements");

        let result = self.pipeline.set_state(gst::State::Playing).map_err(|e| {
            let detail = format!("Failed to start recording: {e}");
            self.monitor.report_error(detail.clone());
            detail
        })?;
        info!(state_change = ?result, "Pipeline set to Playing");
        if result != gst::StateChangeSuccess::Async {
            self.monitor.set_phase(RecordingPhase::Recording);
        }
        Ok(())
    }

    /// Stop recording and finalize the file
    pub fn stop(self) -> Result<PathBuf, String> {
        self.stop_with_timeout(gst::ClockTime::from_seconds(60))
    }

    fn stop_with_timeout(mut self, eos_wait: gst::ClockTime) -> Result<PathBuf, String> {
        self.monitor.set_phase(RecordingPhase::Finalizing);
        info!("Stopping video recording");
        // Stop accepting frames before any native EOS/finalization call can
        // block. Cancellation is a signal, never a join on the native worker.
        drop(self.pusher_handle.take());
        clear_recording_diagnostics();

        // Send EOS directly to every source element's src pad.
        // pipeline.send_event(EOS) doesn't reliably reach live sources like pulsesrc,
        // so the muxer (aggregator) never sees EOS on the audio pad and hangs.
        // By pushing EOS on each source's src pad, both appsrc and pulsesrc branches
        // propagate EOS through to the muxer, allowing it to finalize.
        info!("Sending EOS to all source elements");
        let iter = self.pipeline.iterate_sources();
        let mut eos_sent = 0u32;
        for src in iter {
            let Ok(src) = src else { continue };
            let name = src.name().to_string();
            // Use element-level send_event (not pad-level) — for source elements
            // this routes downstream events via gst_pad_push_event on the src pad.
            // pad.send_event() would send upstream, which is wrong for EOS.
            debug!(element = %name, "Sending EOS to source element");
            src.send_event(gst::event::Eos::new());
            eos_sent += 1;
        }
        if eos_sent == 0 {
            // Fallback: send EOS to pipeline
            warn!("No source pads found, sending EOS to pipeline");
            if !self.pipeline.send_event(gst::event::Eos::new()) {
                warn!("Failed to send EOS event to pipeline");
            }
        } else {
            info!(eos_sent, "EOS sent to source elements");
        }

        let mut eos_timeout = false;

        // Wait for EOS to propagate through the entire pipeline.
        // The bus posts an EOS message only after ALL sink elements have received
        // EOS, which means the muxer has finalized (written moov atom for MP4,
        // duration for WebM, etc.) and the filesink has flushed.
        if let Some(bus) = self.pipeline.bus() {
            info!("Waiting for pipeline EOS on bus...");
            match bus
                .timed_pop_filtered(eos_wait, &[gst::MessageType::Eos, gst::MessageType::Error])
            {
                Some(msg) => match msg.view() {
                    gst::MessageView::Eos(_) => {
                        info!("Pipeline EOS received — file finalized");
                    }
                    gst::MessageView::Error(err) => {
                        error!(
                            error = %err.error(),
                            debug = ?err.debug(),
                            source = ?err.src().map(|s| s.name()),
                            "GStreamer error while waiting for EOS"
                        );
                        self.monitor
                            .report_error(format!("Error finalizing recording: {}", err.error()));
                        eos_timeout = true;
                    }
                    _ => {}
                },
                None => {
                    warn!("Timeout (60s) waiting for pipeline EOS, forcing shutdown");
                    self.monitor
                        .report_error("Timeout waiting for recording EOS".into());
                    eos_timeout = true;
                }
            }
        } else {
            // Fallback: no bus available, use fixed sleep
            warn!("No pipeline bus available, using fixed sleep fallback");
            self.monitor
                .report_error("Recording pipeline has no bus for EOS finalization".into());
            std::thread::sleep(std::time::Duration::from_millis(1000));
            eos_timeout = true;
        }

        // Set pipeline to NULL state - this will trigger final cleanup
        info!("Setting pipeline to NULL state");
        self.cleanup()?;

        let file_path = std::mem::take(&mut self.file_path);
        if eos_timeout || self.monitor.subscribe().borrow().error.is_some() {
            warn!(path = %file_path.display(), "Recording may be incomplete (EOS timeout)");
            Err(format!(
                "Recording saved but may be incomplete: {}",
                file_path.display()
            ))
        } else {
            info!(path = %file_path.display(), "Recording saved");
            Ok(file_path)
        }
    }
}

impl VideoRecorder {
    fn cleanup(&mut self) -> Result<(), String> {
        if self.cleanup_complete {
            return Ok(());
        }
        self.monitor.set_phase(RecordingPhase::Cleanup);
        let result = self.pipeline.set_state(gst::State::Null).map_err(|e| {
            let detail = format!("Failed to stop pipeline: {e}");
            self.monitor.report_error(detail.clone());
            detail
        });
        // Never finish until NULL returned, nor retry a failed transition in Drop.
        self.cleanup_complete = true;
        if let Some(bus) = self.pipeline.bus() {
            bus.unset_sync_handler();
        }
        result.map(|_| ())
    }
}

impl Drop for VideoRecorder {
    fn drop(&mut self) {
        // Request cancellation; the resource lease keeps Cleanup observable
        // until the independent worker actually releases its appsrc.
        drop(self.pusher_handle.take());

        // Ensure pipeline is properly stopped — this disconnects pulsesrc from
        // PulseAudio and releases all GStreamer resources.
        let _ = self.cleanup();
    }
}

/// Audio branch elements
struct AudioBranch {
    source: gst::Element,
    queue: gst::Element,
    convert: gst::Element,
    /// Zero-cost pass-through when the source already produces the target rate;
    /// only does work when PA can't (or won't) serve the requested rate.
    resample: gst::Element,
    capsfilter: gst::Element,
    /// Soft-knee downward compressor. Squashes loud peaks so the following
    /// makeup-gain stage can lift quiet content without clipping. Same chain
    /// is used by the settings audio probe so the meter reflects what the
    /// recording captures.
    compressor: gst::Element,
    /// Makeup gain applied after the compressor. Linear scale; the user-facing
    /// "audio gain" setting will eventually drive this property.
    makeup_gain: gst::Element,
    /// Brick-wall limiter after makeup gain — caps output around -0.4 dBFS
    /// so a strong source plus +6 dB makeup never clips the encoder.
    limiter: gst::Element,
    /// Level meter after compressor + makeup gain so the UI reads the actual
    /// recorded signal level. The pre-mix per-channel meter was removed to
    /// cut CPU pressure on weak ARM hardware; the defensive resampler remains.
    level: gst::Element,
    encoder: gst::Element,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn recording_zoom_crops_the_center_without_changing_resolution() {
        gst::init().unwrap();
        let mut frame = test_rgba_frame(8, 4, 32);
        let mut pixels = vec![0u8; 8 * 4 * 4];
        for y in 0..4 {
            for x in 0..8 {
                let i = (y * 8 + x) * 4;
                pixels[i..i + 4].copy_from_slice(&[x as u8 * 30, y as u8 * 50, 90, 255]);
            }
        }
        frame.data = crate::backends::camera::types::FrameData::from_owned_vec(pixels);
        let buffer = recording_frame_buffer(&frame, crate::app::FilterType::Standard, 2.0)
            .await
            .unwrap();
        let mapped = buffer.map_readable().unwrap();
        assert_eq!(mapped.len(), 8 * 4 * 4);
        assert_eq!(
            &mapped[..4],
            &[60, 50, 90, 255],
            "2x must start at the centered ROI, not the source edge"
        );
        assert_eq!(&mapped[mapped.len() - 4..], &[150, 100, 90, 255]);
    }

    #[tokio::test]
    async fn recording_zoom_pusher_samples_live_zoom_and_reset_at_fixed_caps() {
        gst::init().unwrap();
        let pipeline = gst::parse::launch(
            "appsrc name=input is-live=true format=time caps=video/x-raw,format=RGBA,width=8,height=4,framerate=30/1 ! appsink name=output sync=false async=false",
        ).unwrap().downcast::<gst::Pipeline>().unwrap();
        let appsrc = pipeline
            .by_name("input")
            .unwrap()
            .downcast::<gst_app::AppSrc>()
            .unwrap();
        let sink = pipeline
            .by_name("output")
            .unwrap()
            .downcast::<gst_app::AppSink>()
            .unwrap();
        let zoom = Arc::new(ZoomControl::default());
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        pipeline.set_state(gst::State::Playing).unwrap();
        let pusher = VideoRecorder::spawn_filtered_pusher(
            appsrc,
            rx,
            30,
            Arc::new(std::sync::atomic::AtomicU32::new(0)),
            zoom.clone(),
            RecordingMonitor::new("identity".into()),
        );
        let frame = test_rgba_frame(8, 4, 32);
        let mut samples = Vec::new();
        for (desired, native_applied) in [(1.0, false), (2.0, false), (1.0, false), (2.0, true)] {
            zoom.set_zoom(desired);
            let mut offered = frame.clone();
            if native_applied {
                use crate::backends::camera::zoom::{ScalerCropBounds, ZoomRect};
                let base = ZoomRect {
                    x: 0,
                    y: 0,
                    width: 4000,
                    height: 3000,
                };
                let minimum = ZoomRect {
                    width: 100,
                    height: 100,
                    ..base
                };
                let bounds = ScalerCropBounds::new(minimum, base, base, base, false).unwrap();
                offered.libcamera_metadata = Some(crate::backends::camera::types::FrameMetadata {
                    scaler_zoom: bounds.applied(Some(bounds.crop(2.0))),
                    ..Default::default()
                });
            }
            tx.send(offered.into()).await.unwrap();
            let sink = sink.clone();
            samples.push(
                tokio::task::spawn_blocking(move || {
                    sink.try_pull_sample(gst::ClockTime::from_seconds(3))
                })
                .await
                .unwrap(),
            );
        }
        drop(tx);
        pusher.await.unwrap();
        pipeline.set_state(gst::State::Null).unwrap();
        let samples: Vec<_> = samples.into_iter().map(Option::unwrap).collect();
        for (index, sample) in samples.iter().enumerate() {
            let caps = sample.caps().unwrap().structure(0).unwrap();
            assert_eq!(caps.get::<i32>("width").unwrap(), 8);
            assert_eq!(caps.get::<i32>("height").unwrap(), 4);
            let buffer = sample.buffer().unwrap();
            assert_eq!(buffer.size(), 8 * 4 * 4);
            assert_eq!(buffer.pts().unwrap().nseconds(), index as u64 * 33_333_333);
            assert_eq!(buffer.duration().unwrap().nseconds(), 33_333_333);
        }
        let first = samples[0].buffer().unwrap().map_readable().unwrap();
        let middle = samples[1].buffer().unwrap().map_readable().unwrap();
        let reset = samples[2].buffer().unwrap().map_readable().unwrap();
        assert_eq!(first.as_slice(), frame.data.as_ref());
        assert_ne!(
            middle.as_slice(),
            first.as_slice(),
            "live zoom must affect recording after it started"
        );
        assert_eq!(&middle[..4], &frame.data.as_ref()[40..44]);
        assert_eq!(reset.as_slice(), first.as_slice());
        let native = samples[3].buffer().unwrap().map_readable().unwrap();
        assert_eq!(
            native.as_slice(),
            first.as_slice(),
            "native 2x must not be cropped a second time by the pusher"
        );
    }

    #[tokio::test]
    async fn recording_zoom_native_applied_crop_is_not_repeated_and_stays_shared() {
        gst::init().unwrap();
        let mut frame = test_rgba_frame(8, 4, 32);
        use crate::backends::camera::zoom::{ScalerCropBounds, ZoomRect};
        let base = ZoomRect {
            x: 0,
            y: 0,
            width: 4000,
            height: 3000,
        };
        let minimum = ZoomRect {
            width: 100,
            height: 100,
            ..base
        };
        let bounds = ScalerCropBounds::new(minimum, base, base, base, false).unwrap();
        frame.libcamera_metadata = Some(crate::backends::camera::types::FrameMetadata {
            scaler_zoom: bounds.applied(Some(bounds.crop(2.0))),
            ..Default::default()
        });
        let output = convert_frame_to_zoomed_rgba(&frame, 2.0).await.unwrap();
        assert_eq!(output, frame.data.as_ref());
        let buffer = recording_frame_buffer(
            &frame,
            crate::app::FilterType::Standard,
            residual_zoom(2.0, &frame),
        )
        .await
        .unwrap();
        assert_eq!(
            buffer.map_readable().unwrap().as_ptr(),
            frame.data.as_ref().as_ptr()
        );
        // Hardware may clamp below the request: software applies only 4 / 2.
        let partial = convert_frame_to_zoomed_rgba(&frame, 4.0).await.unwrap();
        let expected = recording_frame_buffer(&frame, crate::app::FilterType::Standard, 2.0)
            .await
            .unwrap();
        assert_eq!(partial, expected.map_readable().unwrap().as_slice());
        assert_ne!(partial, output);
    }

    #[tokio::test]
    async fn recording_zoom_repacked_rgba_keeps_pixels_and_storage_size() {
        let frame = test_rgba_frame(8, 4, 40);
        let unzoomed = convert_frame_to_rgba(&frame).await.unwrap();
        let output = convert_frame_to_zoomed_rgba(&frame, 2.0).await.unwrap();
        assert_eq!(output.len(), 8 * 4 * 4);
        assert_eq!(&output[..4], &frame.data.as_ref()[48..52]);
        assert_eq!(
            convert_frame_to_zoomed_rgba(&frame, 1.0).await.unwrap(),
            unzoomed
        );
    }

    #[test]
    fn recording_zoom_keeps_live_filter_after_crop_at_fixed_dimensions() {
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu(
                "recording_zoom_keeps_live_filter_after_crop_at_fixed_dimensions",
            );
            return;
        };
        gst::init().unwrap();
        pollster::block_on(async {
            let frame = test_rgba_frame(8, 4, 32);
            let zoomed = convert_frame_to_zoomed_rgba(&frame, 2.0).await.unwrap();
            let expected =
                crate::shaders::apply_filter_gpu_rgba(&zoomed, 8, 4, crate::app::FilterType::Sepia)
                    .await
                    .unwrap();
            let actual = recording_frame_buffer(&frame, crate::app::FilterType::Sepia, 2.0)
                .await
                .unwrap();
            assert_eq!(actual.size(), 8 * 4 * 4);
            assert_eq!(actual.map_readable().unwrap().as_slice(), expected);
        });
    }

    #[test]
    fn sensor_pts_waits_for_playing_even_when_a_clock_is_available() {
        gst::init().unwrap();
        let appsrc = gst_app::AppSrc::builder().is_live(true).build();
        appsrc.set_clock(Some(&gst::SystemClock::obtain())).unwrap();
        appsrc.set_state(gst::State::Ready).unwrap();
        assert_eq!(appsrc.base_time(), Some(gst::ClockTime::ZERO));
        assert!(appsrc.current_running_time().is_some());
        let mut pipeline_playing = false;
        let mut ts_offset = None;
        let result = compute_pts(
            &appsrc,
            Some(read_clock_boottime_ns()),
            0,
            1_000_000_000 / 30,
            &mut pipeline_playing,
            &mut ts_offset,
        );
        appsrc.set_state(gst::State::Null).unwrap();

        assert!(
            matches!(result, PtsResult::Skip),
            "a clock assigned before PLAYING must not seed video PTS with system uptime"
        );
        assert!(!pipeline_playing);
        assert!(ts_offset.is_none());
    }

    #[test]
    fn sensor_pts_uses_recording_running_time_after_playing() {
        gst::init().unwrap();
        let pipeline = gst::parse::launch(
            "appsrc name=input is-live=true format=time ! fakesink sync=false async=false",
        )
        .unwrap()
        .downcast::<gst::Pipeline>()
        .unwrap();
        let appsrc = pipeline
            .by_name("input")
            .unwrap()
            .downcast::<gst_app::AppSrc>()
            .unwrap();
        pipeline.set_state(gst::State::Playing).unwrap();
        assert_eq!(appsrc.current_state(), gst::State::Playing);
        let sensor_ts = read_clock_boottime_ns();
        let mut pipeline_playing = false;
        let mut ts_offset = None;
        let first = compute_pts(
            &appsrc,
            Some(sensor_ts),
            0,
            33_333_333,
            &mut pipeline_playing,
            &mut ts_offset,
        );
        let next = compute_pts(
            &appsrc,
            Some(sensor_ts + 33_333_333),
            1,
            33_333_333,
            &mut pipeline_playing,
            &mut ts_offset,
        );
        let running_time = appsrc.current_running_time().unwrap().nseconds();
        pipeline.set_state(gst::State::Null).unwrap();

        let PtsResult::Pts(first) = first else {
            panic!("a PLAYING source must accept sensor-timestamped frames");
        };
        let PtsResult::Pts(next) = next else {
            panic!("the next sensor frame must remain on the recording timeline");
        };
        assert!(first <= running_time);
        assert_eq!(next - first, 33_333_333);
        assert!(pipeline_playing);
    }

    fn construction_config<'a>(
        encoder: &'a crate::media::encoders::video::EncoderInfo,
        output_path: PathBuf,
    ) -> AppsrcRecorderConfig<'a> {
        AppsrcRecorderConfig {
            base: RecorderConfig {
                width: 16,
                height: 16,
                framerate: 30,
                output_path,
                encoder_config: EncoderConfig {
                    width: 16,
                    height: 16,
                    ..Default::default()
                },
                enable_audio: false,
                audio_device: None,
                audio_source_rate_hz: 0,
                encoder_info: Some(encoder),
                capture_metadata: Default::default(),
                audio_levels: SharedAudioLevels::default(),
            },
            pixel_format: PixelFormat::RGBA,
            live_filter_code: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            live_zoom: Arc::new(ZoomControl::default()),
        }
    }

    #[tokio::test]
    async fn monitored_constructor_encoder_selection_failure_finishes_after_releasing_input() {
        gst::init().unwrap();
        let encoder = crate::media::encoders::video::EncoderInfo {
            element_name: "camera-nonexistent-encoder".into(),
            display_name: "Missing test encoder".into(),
            codec: crate::media::encoders::video::VideoCodec::H264,
            is_hardware: false,
            priority: 0,
        };
        let monitor = RecordingMonitor::new(encoder.element_name.clone());
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        tx.try_send(test_rgba_frame(2, 2, 8).into()).unwrap();
        let error = VideoRecorder::new_from_appsrc_monitored(
            construction_config(&encoder, PathBuf::from("unused.mp4")),
            rx,
            monitor.clone(),
        )
        .unwrap_err();
        assert!(error.contains(&encoder.element_name), "{error}");
        assert!(tx.is_closed(), "constructor retained its input receiver");
        let failure =
            super::super::recording_monitor::wait_for_recording_failure(monitor.subscribe())
                .await
                .unwrap();
        assert_eq!(failure.detail, error);
        assert_eq!(monitor.subscribe().borrow().phase, RecordingPhase::Finished);
    }

    // Post-GREEN characterizations of the other construction/lifetime branches.
    #[tokio::test]
    async fn monitored_constructor_pipeline_parse_failure_finishes() {
        gst::init().unwrap();
        let encoder = construction_software_encoder();
        let monitor = RecordingMonitor::new("requested encoder".into());
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        // An unavailable element deterministically fails real parsing after
        // encoder selection, without opening a file or touching devices.
        let error = VideoRecorder::new_from_appsrc_monitored(
            construction_config(
                &encoder,
                PathBuf::from("unused ! camera-nonexistent-element"),
            ),
            rx,
            monitor.clone(),
        )
        .unwrap_err();
        assert!(error.starts_with("Failed to parse pipeline:"), "{error}");
        assert!(tx.is_closed());
        let failure =
            super::super::recording_monitor::wait_for_recording_failure(monitor.subscribe())
                .await
                .unwrap();
        assert_eq!(failure.detail, error);
        assert_eq!(monitor.subscribe().borrow().encoder, "x264enc");
        assert_eq!(monitor.subscribe().borrow().phase, RecordingPhase::Finished);
    }

    fn construction_software_encoder() -> crate::media::encoders::video::EncoderInfo {
        crate::media::encoders::video::EncoderInfo {
            element_name: "x264enc".into(),
            display_name: "Synthetic software H264".into(),
            codec: crate::media::encoders::video::VideoCodec::H264,
            is_hardware: false,
            priority: 0,
        }
    }

    #[tokio::test]
    async fn monitored_constructor_failure_does_not_bypass_outstanding_resource_lease() {
        gst::init().unwrap();
        let mut encoder = construction_software_encoder();
        encoder.element_name = "camera-nonexistent-encoder".into();
        let monitor = RecordingMonitor::new(encoder.element_name.clone());
        let lease = monitor.resource_lease();
        let (_tx, rx) = tokio::sync::mpsc::channel(1);
        assert!(
            VideoRecorder::new_from_appsrc_monitored(
                construction_config(&encoder, PathBuf::from("unused.mp4")),
                rx,
                monitor.clone(),
            )
            .is_err()
        );
        assert!(monitor.subscribe().borrow().error.is_some());
        assert_eq!(monitor.subscribe().borrow().phase, RecordingPhase::Cleanup);
        drop(lease);
        assert_eq!(monitor.subscribe().borrow().phase, RecordingPhase::Finished);
    }

    #[tokio::test]
    async fn monitored_constructor_success_stays_live_until_recorder_and_pusher_release() {
        gst::init().unwrap();
        let encoder = construction_software_encoder();
        let output = std::env::temp_dir().join(format!(
            "camera-construction-success-{}.mp4",
            uuid::Uuid::new_v4()
        ));
        let monitor = RecordingMonitor::new("requested encoder".into());
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let recorder = VideoRecorder::new_from_appsrc_monitored(
            construction_config(&encoder, output.clone()),
            rx,
            monitor.clone(),
        )
        .unwrap();
        let status = monitor.subscribe().borrow().clone();
        assert!(status.error.is_none());
        assert!(matches!(
            status.phase,
            RecordingPhase::Starting | RecordingPhase::Recording
        ));
        assert!(!tx.is_closed());
        // Successful construction must not even request deferred completion:
        // registering a resource after finish_requested would panic.
        let lease = monitor.resource_lease();
        drop(recorder);
        assert_eq!(monitor.subscribe().borrow().phase, RecordingPhase::Cleanup);
        drop(lease);
        let mut completion = monitor.subscribe();
        tokio::time::timeout(std::time::Duration::from_secs(3), async {
            while completion.borrow_and_update().phase != RecordingPhase::Finished {
                completion.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
        assert!(tx.is_closed());
        std::fs::remove_file(output).unwrap();
    }

    #[test]
    fn native_pusher_stall_and_destruction_do_not_starve_monitor_or_finish_early() {
        use std::sync::mpsc;
        use std::time::Duration;
        gst::init().unwrap();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let pipeline =
            gst::parse::launch("appsrc name=input ! identity ! fakesink sync=false async=false")
                .unwrap()
                .downcast::<gst::Pipeline>()
                .unwrap();
        let appsrc = pipeline
            .by_name("input")
            .unwrap()
            .downcast::<gst_app::AppSrc>()
            .unwrap();
        // A full byte queue emits enough-data synchronously inside push_buffer.
        // Keep the pipeline in NULL: no streaming thread or device is involved.
        appsrc.set_max_buffers(0);
        appsrc.set_max_bytes(1);
        appsrc.set_block(false);
        appsrc
            .push_buffer(gst::Buffer::with_size(16).unwrap())
            .unwrap();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = std::sync::Mutex::new(release_rx);
        appsrc.set_callbacks(
            gst_app::AppSrcCallbacks::builder()
                .enough_data(move |_| {
                    let _ = entered_tx.send(());
                    let _ = release_rx
                        .lock()
                        .unwrap()
                        .recv_timeout(Duration::from_secs(5));
                })
                .build(),
        );
        let monitor = RecordingMonitor::new("identity".into());
        let (destroy_tx, destroy_rx) = mpsc::channel();
        let (destroy_release_tx, destroy_release_rx) = mpsc::channel();
        let destruction_monitor = monitor.clone();
        let _notify = appsrc.add_weak_ref_notify(move || {
            let _ = destroy_tx.send(destruction_monitor.subscribe().borrow().phase);
            let _ = destroy_release_rx.recv_timeout(Duration::from_secs(5));
        });
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let pusher = {
            let _entered = runtime.enter();
            VideoRecorder::spawn_filtered_pusher(
                appsrc,
                rx,
                30,
                Arc::new(std::sync::atomic::AtomicU32::new(0)),
                Arc::new(ZoomControl::default()),
                monitor.clone(),
            )
        };
        tx.try_send(test_rgba_frame(2, 2, 8).into()).unwrap();
        entered_rx.recv_timeout(Duration::from_secs(3)).unwrap();
        let recorder = VideoRecorder {
            pipeline,
            file_path: PathBuf::from("synthetic"),
            _pulse_volume_guard: None,
            pusher_handle: Some(pusher),
            monitor: monitor.clone(),
            cleanup_complete: false,
            _finish: FinishRecording(monitor.clone()),
        };
        // This is an external OS deadline, not a timer on the blocked executor.
        let (observed_tx, observed_rx) = mpsc::channel();
        let status = monitor.subscribe();
        runtime.spawn(async move {
            let failure = super::super::recording_monitor::wait_for_recording_failure(status).await;
            observed_tx.send(failure.unwrap().detail).unwrap();
        });
        let watchdog_monitor = monitor.clone();
        runtime.spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            watchdog_monitor.report_error("native call blocked".into());
        });
        let observed = observed_rx.recv_timeout(Duration::from_millis(500));
        drop(recorder); // Must signal cancellation, not wait for native completion.
        let phase_while_native_blocked = monitor.subscribe().borrow().phase;
        release_tx.send(()).unwrap();
        let destruction_phase = destroy_rx.recv_timeout(Duration::from_secs(3));
        let phase_while_destructor_blocked = monitor.subscribe().borrow().phase;
        // A gated GObject final-unref must also be isolated from the executor.
        let (tick_tx, tick_rx) = mpsc::channel();
        runtime.spawn(async move {
            tick_tx.send(()).unwrap();
        });
        let destructor_tick = tick_rx.recv_timeout(Duration::from_millis(500));
        destroy_release_tx.send(()).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while monitor.subscribe().borrow().phase != RecordingPhase::Finished
            && std::time::Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(1));
        }
        drop(tx);
        runtime.shutdown_timeout(Duration::from_secs(3));
        assert_eq!(
            (
                observed.ok(),
                phase_while_native_blocked,
                destruction_phase.ok(),
                phase_while_destructor_blocked,
                destructor_tick.is_ok(),
            ),
            (
                Some("native call blocked".into()),
                RecordingPhase::Cleanup,
                Some(RecordingPhase::Cleanup),
                RecordingPhase::Cleanup,
                true,
            ),
            "native pusher must preserve timer/subscription liveness and resource lifetime",
        );
        assert_eq!(monitor.subscribe().borrow().phase, RecordingPhase::Finished);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn stop_cancels_pusher_before_waiting_for_pipeline_eos() {
        use std::time::Duration;
        gst::init().unwrap();
        let monitor = RecordingMonitor::new("identity".into());
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let pusher = VideoRecorder::spawn_filtered_pusher(
            gst_app::AppSrc::builder().build(),
            rx,
            30,
            Arc::new(std::sync::atomic::AtomicU32::new(0)),
            Arc::new(ZoomControl::default()),
            monitor.clone(),
        );
        // This empty pipeline never publishes EOS. Pusher cancellation must
        // precede the blocking bus wait, not depend on its timeout returning.
        let recorder = VideoRecorder {
            pipeline: gst::Pipeline::new(),
            file_path: PathBuf::from("synthetic"),
            _pulse_volume_guard: None,
            pusher_handle: Some(pusher),
            monitor: monitor.clone(),
            cleanup_complete: false,
            _finish: FinishRecording(monitor.clone()),
        };
        let stop = tokio::task::spawn_blocking(move || {
            recorder.stop_with_timeout(gst::ClockTime::from_seconds(1))
        });
        let cancelled = tokio::time::timeout(Duration::from_millis(250), tx.closed()).await;
        assert!(stop.await.unwrap().is_err());
        assert!(
            cancelled.is_ok(),
            "pusher cancellation waited for native EOS finalization"
        );
    }

    // Post-GREEN characterization: cancelling an idle blocking worker must not
    // depend on the producer closing its bounded channel or on JoinHandle::abort.
    #[tokio::test(flavor = "current_thread")]
    async fn dropping_pusher_cancels_idle_receiver_and_releases_appsrc() {
        use std::time::Duration;
        gst::init().unwrap();
        let appsrc = gst_app::AppSrc::builder().build();
        let monitor = RecordingMonitor::new("identity".into());
        let (destroyed_tx, destroyed_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let _notify = appsrc.add_weak_ref_notify(move || {
            let _ = destroyed_tx.send(());
            let _ = release_rx.recv();
        });
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let pusher = VideoRecorder::spawn_filtered_pusher(
            appsrc,
            rx,
            30,
            Arc::new(std::sync::atomic::AtomicU32::new(0)),
            Arc::new(ZoomControl::default()),
            monitor.clone(),
        );
        drop(pusher);
        tokio::time::timeout(Duration::from_secs(3), destroyed_rx)
            .await
            .unwrap()
            .unwrap();
        assert!(tx.is_closed());
        assert_eq!(
            monitor.subscribe().borrow().phase,
            RecordingPhase::Starting,
            "pusher completion alone must not complete the recorder owner"
        );
        monitor.finish();
        let mut completion = monitor.subscribe();
        assert_eq!(
            completion.borrow().phase,
            RecordingPhase::Starting,
            "final native unref must finish before the pusher lease is released"
        );
        release_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(3), async {
            while completion.borrow_and_update().phase != RecordingPhase::Finished {
                completion.changed().await.unwrap();
            }
        })
        .await
        .expect("pusher cleanup must publish completion after final native unref");
    }

    #[test]
    fn monitored_encoder_output_probe_records_progress_without_replacing_existing_probes() {
        gst::init().unwrap();
        let pipeline = gst::parse::launch(
            "appsrc name=input is-live=true ! identity name=recording-encoder ! fakesink name=sink sync=false async=false"
        ).unwrap().downcast::<gst::Pipeline>().unwrap();
        let monitor = RecordingMonitor::new("identity".into());
        monitor.enable_activity_watchdog();
        let original_count = Arc::new(AtomicU64::new(0));
        let count = original_count.clone();
        pipeline
            .by_name("recording-encoder")
            .unwrap()
            .static_pad("src")
            .unwrap()
            .add_probe(gst::PadProbeType::BUFFER, move |_, _| {
                count.fetch_add(1, Ordering::Relaxed);
                gst::PadProbeReturn::Ok
            });
        install_recording_observer(&pipeline, &SharedAudioLevels::default(), &monitor);
        let (tx, rx) = std::sync::mpsc::channel();
        let m = monitor.clone();
        pipeline
            .by_name("sink")
            .unwrap()
            .static_pad("sink")
            .unwrap()
            .add_probe(gst::PadProbeType::BUFFER, move |_, _| {
                let _ = tx.send(m.has_pending_input());
                gst::PadProbeReturn::Ok
            });
        pipeline.set_state(gst::State::Playing).unwrap();
        let appsrc = pipeline
            .by_name("input")
            .unwrap()
            .downcast::<gst_app::AppSrc>()
            .unwrap();
        monitor.note_input();
        assert!(monitor.has_pending_input());
        appsrc
            .push_buffer(gst::Buffer::with_size(16).unwrap())
            .unwrap();
        let pending_at_sink = rx.recv_timeout(std::time::Duration::from_secs(3));
        pipeline.set_state(gst::State::Null).unwrap();
        assert!(
            !pending_at_sink.unwrap(),
            "encoder output was not recorded as progress"
        );
        assert_eq!(original_count.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn monitored_pusher_notes_offered_input_even_when_appsrc_queue_is_full() {
        gst::init().unwrap();
        let appsrc = gst_app::AppSrc::builder().build();
        configure_live_appsrc(&appsrc);
        for _ in 0..APPSRC_MAX_BUFFERS {
            appsrc
                .push_buffer(gst::Buffer::with_size(16).unwrap())
                .unwrap();
        }
        assert!(appsrc_queue_is_full(&appsrc));
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let monitor = RecordingMonitor::new("blocked".into());
        monitor.enable_activity_watchdog();
        let handle = VideoRecorder::spawn_filtered_pusher(
            appsrc.clone(),
            rx,
            30,
            Arc::new(std::sync::atomic::AtomicU32::new(0)),
            Arc::new(ZoomControl::default()),
            monitor.clone(),
        );
        tx.send(test_rgba_frame(2, 2, 8).into()).await.unwrap();
        drop(tx);
        handle.await.unwrap();
        assert!(
            monitor.has_pending_input(),
            "queue-full skip hid offered input"
        );
        assert_eq!(appsrc.current_level_buffers(), APPSRC_MAX_BUFFERS);
    }

    // Post-GREEN public-route characterization: explicit software encoder only;
    // never run automatic hardware discovery, audio capture or GPU conversion.
    #[tokio::test]
    async fn monitored_constructor_reports_resolved_software_encoder_on_start_failure() {
        gst::init().unwrap();
        let info = crate::media::encoders::video::EncoderInfo {
            element_name: "x264enc".into(),
            display_name: "Synthetic software H264".into(),
            codec: crate::media::encoders::video::VideoCodec::H264,
            is_hardware: false,
            priority: 0,
        };
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let monitor = RecordingMonitor::new("requested encoder".into());
        let missing = std::env::temp_dir()
            .join(format!("camera-absent-{}", uuid::Uuid::new_v4()))
            .join("video.mp4");
        let result = VideoRecorder::new_from_appsrc_monitored(
            AppsrcRecorderConfig {
                base: RecorderConfig {
                    width: 16,
                    height: 16,
                    framerate: 30,
                    output_path: missing,
                    encoder_config: EncoderConfig {
                        width: 16,
                        height: 16,
                        ..Default::default()
                    },
                    enable_audio: false,
                    audio_device: None,
                    audio_source_rate_hz: 0,
                    encoder_info: Some(&info),
                    capture_metadata: Default::default(),
                    audio_levels: SharedAudioLevels::default(),
                },
                pixel_format: PixelFormat::RGBA,
                live_filter_code: Arc::new(std::sync::atomic::AtomicU32::new(0)),
                live_zoom: Arc::new(ZoomControl::default()),
            },
            rx,
            monitor.clone(),
        );
        assert!(result.is_err());
        let status = monitor.subscribe().borrow().clone();
        assert_eq!(status.encoder, "x264enc");
        monitor.note_input();
        assert!(
            monitor.has_pending_input(),
            "normal constructor did not enable activity monitoring"
        );
        assert!(status.error.is_some());
        // Constructor failure cancels the pusher, but its native unref runs
        // independently. Observe completion rather than assuming immediate drop.
        let mut completion = monitor.subscribe();
        tokio::time::timeout(std::time::Duration::from_secs(3), async {
            while completion.borrow_and_update().phase != RecordingPhase::Finished {
                completion.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
        drop(tx);
    }

    #[test]
    fn monitored_async_start_keeps_startup_deadline_until_playing() {
        gst::init().unwrap();
        let pipeline = gst::parse::launch("appsrc name=input is-live=true ! fakesink sync=false")
            .unwrap()
            .downcast::<gst::Pipeline>()
            .unwrap();
        let monitor = RecordingMonitor::new("synthetic".into());
        install_recording_observer(&pipeline, &SharedAudioLevels::default(), &monitor);
        let recorder = VideoRecorder {
            pipeline: pipeline.clone(),
            file_path: PathBuf::from("synthetic"),
            _pulse_volume_guard: None,
            pusher_handle: None,
            monitor: monitor.clone(),
            cleanup_complete: false,
            _finish: FinishRecording(monitor.clone()),
        };
        recorder.start().unwrap();
        assert_eq!(monitor.subscribe().borrow().phase, RecordingPhase::Starting);
        let appsrc = pipeline
            .by_name("input")
            .unwrap()
            .downcast::<gst_app::AppSrc>()
            .unwrap();
        appsrc
            .push_buffer(gst::Buffer::with_size(16).unwrap())
            .unwrap();
        pipeline.state(gst::ClockTime::from_seconds(3)).0.unwrap();
        assert_eq!(
            monitor.subscribe().borrow().phase,
            RecordingPhase::Recording
        );
        drop(recorder);
        assert_eq!(monitor.subscribe().borrow().phase, RecordingPhase::Finished);
    }

    #[test]
    fn cleanup_remains_monitored_through_pipeline_destruction() {
        gst::init().unwrap();
        let monitor = RecordingMonitor::new("synthetic".into());
        let recorder = VideoRecorder {
            pipeline: gst::Pipeline::new(),
            file_path: PathBuf::from("synthetic"),
            _pulse_volume_guard: None,
            pusher_handle: None,
            monitor: monitor.clone(),
            cleanup_complete: false,
            _finish: FinishRecording(monitor.clone()),
        };
        let observed = Arc::new(std::sync::Mutex::new(None));
        let at_destruction = observed.clone();
        let status = monitor.clone();
        let _notify = recorder.pipeline.add_weak_ref_notify(move || {
            *at_destruction.lock().unwrap() = Some(status.subscribe().borrow().phase);
        });
        drop(recorder);
        assert_eq!(*observed.lock().unwrap(), Some(RecordingPhase::Cleanup));
        assert_eq!(monitor.subscribe().borrow().phase, RecordingPhase::Finished);
    }

    #[tokio::test]
    async fn monitored_pusher_channel_close_marks_finalizing_before_eos() {
        gst::init().unwrap();
        let appsrc = gst_app::AppSrc::builder().build();
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let monitor = RecordingMonitor::new("synthetic".into());
        monitor.set_phase(RecordingPhase::Recording);
        let handle = VideoRecorder::spawn_filtered_pusher(
            appsrc,
            rx,
            30,
            Arc::new(std::sync::atomic::AtomicU32::new(0)),
            Arc::new(ZoomControl::default()),
            monitor.clone(),
        );
        drop(tx);
        handle.await.unwrap();
        assert_eq!(
            monitor.subscribe().borrow().phase,
            RecordingPhase::Finalizing
        );
    }

    #[test]
    fn monitored_eos_timeout_is_visible_before_null_cleanup() {
        gst::init().unwrap();
        // A source that cannot produce EOS on its own, with an injected zero wait.
        let pipeline = gst::parse::launch("appsrc is-live=true ! fakesink sync=false")
            .unwrap()
            .downcast::<gst::Pipeline>()
            .unwrap();
        let monitor = RecordingMonitor::new("synthetic".into());
        let observed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let seen = observed.clone();
        let m = monitor.clone();
        pipeline.bus().unwrap().set_sync_handler(move |_, message| {
            if let gst::MessageView::StateChanged(state) = message.view()
                && state.current() == gst::State::Ready
            {
                let status = m.subscribe().borrow().clone();
                if status.phase == RecordingPhase::Cleanup {
                    seen.store(
                        status.error.as_ref().is_some_and(|e| e.contains("EOS")),
                        Ordering::Relaxed,
                    );
                }
            }
            gst::BusSyncReply::Pass
        });
        let recorder = VideoRecorder {
            pipeline,
            file_path: PathBuf::from("synthetic"),
            _pulse_volume_guard: None,
            pusher_handle: None,
            monitor: monitor.clone(),
            cleanup_complete: false,
            _finish: FinishRecording(monitor.clone()),
        };
        recorder.start().unwrap();
        assert!(recorder.stop_with_timeout(gst::ClockTime::ZERO).is_err());
        assert!(observed.load(Ordering::Relaxed));
        assert!(monitor.subscribe().borrow().error.is_some());
    }

    #[test]
    fn monitored_synthetic_recorder_reports_error_before_cleanup_and_keeps_bus_error() {
        gst::init().unwrap();
        let pipeline = gst::parse::launch(
            "videotestsrc is-live=true ! identity name=recording-encoder ! fakesink sync=false",
        )
        .unwrap()
        .downcast::<gst::Pipeline>()
        .unwrap();
        let monitor = super::super::RecordingMonitor::new("identity".into());
        install_recording_observer(&pipeline, &SharedAudioLevels::default(), &monitor);
        let recorder = VideoRecorder {
            pipeline: pipeline.clone(),
            file_path: PathBuf::from("synthetic"),
            _pulse_volume_guard: None,
            pusher_handle: None,
            monitor: monitor.clone(),
            cleanup_complete: false,
            _finish: FinishRecording(monitor.clone()),
        };
        recorder.start().unwrap();
        pipeline.state(gst::ClockTime::from_seconds(3)).0.unwrap();
        assert_eq!(
            monitor.subscribe().borrow().phase,
            super::super::RecordingPhase::Recording
        );
        let encoder = pipeline.by_name("recording-encoder").unwrap();
        encoder
            .post_message(
                gst::message::Error::builder(gst::CoreError::Failed, "synthetic encode failure")
                    .src(&encoder)
                    .build(),
            )
            .unwrap();
        let error = monitor.subscribe().borrow().error.clone().unwrap();
        assert!(error.contains("recording-encoder"));
        assert!(error.contains("synthetic encode failure"));
        assert!(recorder.stop().is_err());
        assert_eq!(monitor.subscribe().borrow().error.as_ref(), Some(&error));
        assert_eq!(
            monitor.subscribe().borrow().phase,
            super::super::RecordingPhase::Finished
        );
    }

    #[test]
    fn monitored_stop_publishes_finalizing_before_eos_and_finishes_after_null() {
        gst::init().unwrap();
        let pipeline = gst::parse::launch("videotestsrc is-live=true ! fakesink sync=false")
            .unwrap()
            .downcast::<gst::Pipeline>()
            .unwrap();
        let monitor = super::super::RecordingMonitor::new("synthetic".into());
        let observed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let seen = observed.clone();
        let m = monitor.clone();
        pipeline
            .iterate_sources()
            .into_iter()
            .next()
            .unwrap()
            .unwrap()
            .static_pad("src")
            .unwrap()
            .add_probe(gst::PadProbeType::EVENT_DOWNSTREAM, move |_, info| {
                if info
                    .event()
                    .is_some_and(|event| event.type_() == gst::EventType::Eos)
                {
                    seen.store(
                        m.subscribe().borrow().phase == RecordingPhase::Finalizing,
                        Ordering::Relaxed,
                    );
                }
                gst::PadProbeReturn::Ok
            });
        let recorder = VideoRecorder {
            pipeline: pipeline.clone(),
            file_path: PathBuf::from("synthetic"),
            _pulse_volume_guard: None,
            pusher_handle: None,
            monitor: monitor.clone(),
            cleanup_complete: false,
            _finish: FinishRecording(monitor.clone()),
        };
        recorder.start().unwrap();
        assert_eq!(recorder.stop().unwrap(), PathBuf::from("synthetic"));
        assert!(observed.load(Ordering::Relaxed));
        assert_eq!(pipeline.current_state(), gst::State::Null);
        assert_eq!(
            monitor.subscribe().borrow().phase,
            super::super::RecordingPhase::Finished
        );
        assert!(monitor.subscribe().borrow().error.is_none());
    }

    #[test]
    fn recording_audio_branch_adds_and_links_every_stage_in_order() {
        use crate::media::encoders::audio::SelectedAudioEncoder;
        gst::init().unwrap();
        let pipeline = gst::Pipeline::new();
        let muxer = gst::ElementFactory::make("webmmux")
            .name("recording-muxer")
            .build()
            .unwrap();
        pipeline.add(&muxer).unwrap();
        let branch = VideoRecorder::create_audio_branch(
            None,
            48_000,
            SelectedAudioEncoder {
                encoder: gst::ElementFactory::make("opusenc").build().unwrap(),
            },
        )
        .unwrap();
        add_audio_branch_to_pipeline(&pipeline, &branch, &SharedAudioLevels::default()).unwrap();
        assert_eq!(
            branch.source.property::<i64>("buffer-time"),
            PULSESRC_BUFFER_TIME_US
        );
        assert!(!branch.source.property::<bool>("provide-clock"));
        assert!(!branch.source.property::<bool>("do-timestamp"));
        assert_eq!(
            branch
                .source
                .property_value("slave-method")
                .serialize()
                .unwrap()
                .as_str(),
            PULSESRC_SLAVE_METHOD
        );
        let caps = branch.capsfilter.property::<gst::Caps>("caps");
        assert_eq!(
            caps.structure(0).unwrap().get::<i32>("channels").unwrap(),
            1
        );
        assert_eq!(
            caps.structure(0).unwrap().get::<i32>("rate").unwrap(),
            48_000
        );
        use crate::pipelines::audio_level::dynamics;
        assert_eq!(
            branch.compressor.property::<f32>("threshold"),
            dynamics::COMPRESSOR_THRESHOLD
        );
        assert_eq!(
            branch.compressor.property::<f32>("ratio"),
            dynamics::COMPRESSOR_RATIO
        );
        assert_eq!(
            branch.makeup_gain.property::<f64>("volume"),
            dynamics::MAKEUP_GAIN
        );
        assert_eq!(
            branch.limiter.property::<f32>("threshold"),
            dynamics::LIMITER_THRESHOLD
        );
        assert_eq!(
            branch.limiter.property::<f32>("ratio"),
            dynamics::LIMITER_RATIO
        );
        assert!(branch.level.property::<bool>("post-messages"));
        assert_eq!(branch.level.property::<u64>("interval"), 100_000_000);
        // Inspect the real graph independently of its construction list. NULL
        // state is sufficient for linking and does not open a microphone.
        let mut element = branch.source.clone();
        for expected in [
            "pulsesrc",
            "queue",
            "audioconvert",
            "audioresample",
            "capsfilter",
            "audiodynamic",
            "volume",
            "audiodynamic",
            "level",
            "opusenc",
            "webmmux",
        ] {
            assert_eq!(element.factory().unwrap().name(), expected);
            assert_eq!(element.parent(), Some(pipeline.clone().upcast()));
            if expected != "webmmux" {
                element = element
                    .static_pad("src")
                    .unwrap()
                    .peer()
                    .unwrap()
                    .parent_element()
                    .unwrap();
            }
        }
        assert_eq!(element, muxer);
        assert_eq!(pipeline.children().len(), 11);
        assert_eq!(pipeline.current_state(), gst::State::Null);
    }

    #[test]
    fn bayer_recording_retains_gpu_bilinear_reconstruction() {
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu("bayer_recording_retains_gpu_bilinear_reconstruction");
            return;
        };
        pollster::block_on(async {
            use crate::backends::camera::types::{FrameData, FrameMetadata};
            use crate::shaders::{DemosaicMethod, GpuConvertPipeline, GpuFrameInput};
            let frame = CameraFrame {
                width: 16,
                height: 16,
                stride: 16,
                format: PixelFormat::BayerRGGB,
                data: FrameData::from_owned_vec(
                    (0..256)
                        .map(|i| (32 + (i * 37 + i / 16 * 53) % 160) as u8)
                        .collect(),
                ),
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: Some(FrameMetadata {
                    colour_gains: Some([1.0, 1.0]),
                    ..Default::default()
                }),
            };
            let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
            let mut pipeline = GpuConvertPipeline::new().await.unwrap();
            pipeline
                .convert_bayer_with_method(&input, DemosaicMethod::Bilinear)
                .unwrap();
            let expected = pipeline.read_rgba_to_cpu(16, 16).await.unwrap();
            pipeline
                .convert_bayer_with_method(&input, DemosaicMethod::Rcd)
                .unwrap();
            let rcd = pipeline.read_rgba_to_cpu(16, 16).await.unwrap();
            assert_ne!(rcd, expected, "fixture must distinguish RCD from Bilinear");
            let actual = convert_frame_to_rgba(&frame).await.unwrap();
            assert_eq!(
                actual, expected,
                "photo default must not change realtime recording reconstruction"
            );
        });
    }

    #[test]
    fn bayer_recording_preserves_signed_headroom_through_ccm() {
        let Some((_guard, _device, _queue)) = crate::test_gpu::headless_device_with_compute()
        else {
            crate::test_gpu::skip_no_gpu("bayer_recording_preserves_signed_headroom_through_ccm");
            return;
        };
        pollster::block_on(async {
            use crate::backends::camera::types::{FrameData, FrameMetadata, RawLayout};
            for (format, sites) in [
                (PixelFormat::BayerRGGB, [0, 1, 2, 3]),
                (PixelFormat::BayerBGGR, [3, 2, 1, 0]),
                (PixelFormat::BayerGRBG, [1, 0, 3, 2]),
                (PixelFormat::BayerGBRG, [2, 3, 0, 1]),
            ] {
                let mut data = vec![0xa5; 24];
                for y in 0..4 {
                    for x in 0..4 {
                        data[y * 6 + x] = [16, 64, 64, 96][sites[(y % 2) * 2 + x % 2]];
                    }
                }
                let frame = CameraFrame {
                    width: 4,
                    height: 4,
                    stride: 6,
                    format,
                    data: FrameData::from_owned_vec(data),
                    yuv_planes: None,
                    captured_at: std::time::Instant::now(),
                    sensor_timestamp_ns: None,
                    libcamera_metadata: Some(FrameMetadata {
                        raw_layout: Some(RawLayout::U8),
                        colour_gains: Some([1.0, 1.0]),
                        black_level: Some(32.0 / 255.0),
                        colour_correction_matrix: Some([
                            [1.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ]),
                        ..Default::default()
                    }),
                };
                let rgba = convert_frame_to_rgba(&frame).await.unwrap();
                assert_eq!(rgba.len(), 4 * 4 * 4);
                for pixel in rgba.as_chunks::<4>().0 {
                    for (actual, expected) in pixel[..3].iter().zip([76, 106, 146]) {
                        assert!(actual.abs_diff(expected) <= 1, "{format:?}: {pixel:?}");
                    }
                    assert_eq!(pixel[3], 255);
                }
            }
        });
    }

    fn test_rgba_frame(width: u32, height: u32, stride: u32) -> CameraFrame {
        let len = stride as usize * height as usize;
        CameraFrame {
            width,
            height,
            data: crate::backends::camera::types::FrameData::from_owned_vec(
                (0..len).map(|value| value as u8).collect(),
            ),
            format: PixelFormat::RGBA,
            stride,
            yuv_planes: None,
            captured_at: std::time::Instant::now(),
            sensor_timestamp_ns: None,
            libcamera_metadata: None,
        }
    }

    #[test]
    fn decoded_recording_orientation_does_not_transform_pixels() {
        let input = VideoEncoderInput {
            width: 1080,
            height: 1920,
            format: None,
        };
        let processing = decoded_video_processing_chain_for_encoder(&input, true);

        assert!(!processing.contains("videoflip"));
        assert!(processing.contains("width=1080,height=1920"));
    }

    #[test]
    fn software_encoder_rgba_processing_constrains_chroma_to_i420() {
        let input = VideoEncoderInput {
            width: 4000,
            height: 3000,
            format: Some("I420".to_string()),
        };
        let x264 = decoded_video_processing_chain_for_encoder(&input, false);
        let x265 = decoded_video_processing_chain_for_encoder(&input, false);

        assert!(x264.contains("format=I420"));
        assert!(x265.contains("format=I420"));
    }

    #[test]
    fn hardware_encoder_rgba_processing_uses_negotiated_input_format() {
        let input = VideoEncoderInput {
            width: 1440,
            height: 1088,
            format: Some("NV12".to_string()),
        };
        let processing = decoded_video_processing_chain_for_encoder(&input, false);

        assert!(processing.contains("format=NV12"));
    }

    #[test]
    fn decoded_v4l2_processing_applies_negotiated_format_and_aligned_dimensions() {
        let input = VideoEncoderInput {
            width: 1440,
            height: 1088,
            format: Some("NV12".into()),
        };
        let processing = decoded_video_processing_chain_for_encoder(&input, true);
        assert!(processing.contains("videoscale"));
        assert!(processing.contains("width=1440,height=1088"));
        assert!(processing.contains("format=NV12"));
    }

    #[test]
    fn standard_filter_wraps_tightly_packed_rgba_without_copying() {
        gst::init().unwrap();
        let frame = test_rgba_frame(4, 2, 16);
        let source_ptr = frame.data.as_ref().as_ptr();

        let buffer = try_shared_rgba_buffer(&frame, crate::app::FilterType::Standard)
            .expect("tightly packed Standard RGBA should use shared storage");
        let mapped = buffer.map_readable().expect("buffer should be readable");

        assert_eq!(mapped.as_slice(), frame.data.as_ref());
        assert_eq!(mapped.as_slice().as_ptr(), source_ptr);
    }

    #[test]
    fn live_filter_switching_only_uses_shared_storage_for_standard_frames() {
        gst::init().unwrap();
        let frame = test_rgba_frame(4, 2, 16);

        assert!(try_shared_rgba_buffer(&frame, crate::app::FilterType::Standard).is_some());
        assert!(try_shared_rgba_buffer(&frame, crate::app::FilterType::Sepia).is_none());
        assert!(try_shared_rgba_buffer(&frame, crate::app::FilterType::Standard).is_some());
    }

    #[test]
    fn padded_standard_rgba_still_uses_the_repacking_path() {
        gst::init().unwrap();
        let frame = test_rgba_frame(4, 2, 20);

        assert!(try_shared_rgba_buffer(&frame, crate::app::FilterType::Standard).is_none());
    }

    #[test]
    fn video_tags_include_standard_orientation_and_device_fields() {
        gst::init().unwrap();
        let metadata = crate::pipelines::capture_metadata::CaptureMetadata {
            device_make: Some("Example".to_string()),
            device_model: Some("Phone".to_string()),
            captured_at: Some(
                chrono::DateTime::parse_from_rfc3339("2026-09-28T12:34:56.789+02:00").unwrap(),
            ),
            orientation: crate::pipelines::capture_metadata::CaptureOrientation::FlipRotate90,
            ..Default::default()
        };

        let tags = video_tags(&metadata);

        assert_eq!(
            tags.get::<gst::tags::ImageOrientation>().unwrap().get(),
            "flip-rotate-90"
        );
        assert_eq!(
            tags.get::<gst::tags::DeviceManufacturer>().unwrap().get(),
            "Example"
        );
        assert_eq!(tags.get::<gst::tags::DeviceModel>().unwrap().get(), "Phone");
        assert_eq!(
            tags.get::<gst::tags::ApplicationName>().unwrap().get(),
            crate::pipelines::capture_metadata::CaptureMetadata::application_name()
        );
        assert!(tags.get::<gst::tags::DateTime>().is_some());
    }

    #[test]
    fn live_appsrc_queue_stays_bounded_when_downstream_is_slow() {
        gst::init().unwrap();
        let pipeline = gst::parse::launch(
            "appsrc name=test-appsrc is-live=true format=time \
             ! identity sleep-time=250000 \
             ! fakesink sync=false",
        )
        .unwrap()
        .downcast::<gst::Pipeline>()
        .unwrap();
        let appsrc = pipeline
            .by_name("test-appsrc")
            .unwrap()
            .downcast::<gst_app::AppSrc>()
            .unwrap();

        configure_live_appsrc(&appsrc);
        pipeline.set_state(gst::State::Playing).unwrap();

        for _ in 0..32 {
            appsrc
                .push_buffer(gst::Buffer::with_size(1024).unwrap())
                .unwrap();
        }

        assert_eq!(appsrc.max_buffers(), APPSRC_MAX_BUFFERS);
        assert_eq!(appsrc.max_bytes(), 0);
        assert!(!appsrc.is_block());
        assert_eq!(appsrc.leaky_type(), gst_app::AppLeakyType::Downstream);
        assert!(appsrc.current_level_buffers() <= APPSRC_MAX_BUFFERS);
        pipeline.set_state(gst::State::Null).unwrap();
    }

    #[test]
    fn queue_capacity_treats_zero_as_unlimited() {
        assert!(!queue_at_capacity(100, 0));
        assert!(!queue_at_capacity(
            APPSRC_MAX_BUFFERS - 1,
            APPSRC_MAX_BUFFERS
        ));
        assert!(queue_at_capacity(APPSRC_MAX_BUFFERS, APPSRC_MAX_BUFFERS));
    }

    #[test]
    fn fallback_timestamps_preserve_time_across_dropped_frames() {
        let frame_duration_ns = 40_000_000;

        assert_eq!(fallback_pts(0, frame_duration_ns), 0);
        // Capture index 1 represents a frame discarded under backpressure.
        assert_eq!(fallback_pts(2, frame_duration_ns), 80_000_000);
    }
}
