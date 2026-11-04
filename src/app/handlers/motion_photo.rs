// SPDX-License-Identifier: GPL-3.0-only
//! Continuous bounded encoded RAM recording: nominal 1.25s pre + 1.25s post.
//! Processing, encoder startup, snapshot and muxing run off the UI thread.
//! The preview source is used even for raw and HDR stills. Their exact exposure
//! correspondence is unknown, so the Motion Photo presentation timestamp is -1.

use crate::app::recording_error::EncoderOperation;
use crate::app::state::AppModel;
use crate::backends::camera::types::{CameraFrame, SensorRotation};
use crate::media::encoders::video::EncoderInfo;
use crate::pipelines::capture_metadata::CaptureMetadata;
use crate::pipelines::motion_photo_ring::{MemoryEncoder, POST_ROLL_NS};
use crate::pipelines::photo::processing::{PostProcessingConfig, PostProcessor};
use crate::pipelines::video::{RecordingMonitor, RecordingPhase};
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

const PROCESSING_TIMEOUT: Duration = Duration::from_millis(500);

/// Deletes temporary video on success, failure and task cancellation.
pub struct TemporaryClip(pub PathBuf);
impl Drop for TemporaryClip {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Geometry and filter snapshot shared by the clip and delayed HDR still processing.
#[derive(Clone)]
pub struct MotionPhotoSnapshot {
    pub photo_output_format: crate::config::PhotoOutputFormat,
    pub camera_metadata: CaptureMetadata,
    pub filter: crate::app::FilterType,
    pub zoom: f32,
    pub media_rotation: SensorRotation,
    pub mirror: bool,
    geometry_rotation: SensorRotation,
    ratio: crate::app::state::PhotoAspectRatio,
    fit: bool,
    portrait: bool,
    screen_width: f32,
    screen_height: f32,
    insets: crate::app::preview_geometry::BarInsets,
}
impl MotionPhotoSnapshot {
    fn crop(&self, width: u32, height: u32, include_zoom: bool) -> Option<(u32, u32, u32, u32)> {
        let (fw, fh) = if self.geometry_rotation.swaps_dimensions() {
            (height, width)
        } else {
            (width, height)
        };
        let crop = if self.fit {
            self.ratio.crop_rect(fw, fh, self.portrait)
        } else {
            crate::app::preview_geometry::cover_capture_crop(
                fw,
                fh,
                self.screen_width,
                self.screen_height,
                self.insets,
                self.ratio.display_ratio(self.portrait),
            )
        };
        let crop = crate::app::preview_geometry::inverse_oriented_crop(
            crop,
            width,
            height,
            self.geometry_rotation,
        );
        let config = PostProcessingConfig {
            crop_rect: Some(crop),
            zoom_level: if include_zoom { self.zoom } else { 1.0 },
            ..Default::default()
        };
        let region = PostProcessor::final_region(width, height, &config).ok()?;
        (region != (0, 0, width, height)).then_some(region)
    }
}

#[derive(PartialEq)]
struct RecorderRecipe {
    dimensions: (u32, u32),
    crop: Option<(u32, u32, u32, u32)>,
    filter: crate::app::FilterType,
    rotation: SensorRotation,
    mirror: bool,
    encoder: String,
    bitrate: u32,
    fps: u32,
    audio: Option<String>,
    geometry_rotation: SensorRotation,
    ratio: crate::app::PhotoAspectRatio,
    fit: bool,
    portrait: bool,
    screen_size: (u32, u32),
    insets: crate::app::preview_geometry::BarInsets,
}
/// A request holds the producer only through snapshot collection, not muxing
/// or still encoding. Drop also releases failed/rejected/abandoned requests.
struct PostrollLease(Arc<AtomicUsize>);
impl Drop for PostrollLease {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

struct ShutterRequest {
    completion: WorkerCompletion,
    postroll: Option<PostrollLease>,
    shutter: Instant,
    path: PathBuf,
    cancel: Arc<AtomicBool>,
    result: tokio::sync::oneshot::Sender<Result<TemporaryClip, String>>,
}
pub(crate) struct MotionPhotoRecorder {
    monitor: RecordingMonitor,
    #[cfg(test)]
    pre_roll_ready: Arc<AtomicBool>,
    draining: Arc<AtomicBool>,
    postroll_requests: Arc<AtomicUsize>,
    recipe: RecorderRecipe,
    worker: Option<std::thread::JoinHandle<()>>,
    frame_tx: std::sync::mpsc::SyncSender<Arc<CameraFrame>>,
    shutter_tx: std::sync::mpsc::SyncSender<ShutterRequest>,
    cancel: Arc<AtomicBool>,
    source_cancel: Arc<AtomicBool>,
}
impl MotionPhotoRecorder {
    #[cfg(test)]
    pub(crate) fn pre_roll_ready(&self) -> bool {
        self.pre_roll_ready.load(Ordering::Acquire)
    }

    pub(crate) async fn stop_and_wait(mut self) -> Result<(), String> {
        self.cancel.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            tokio::task::spawn_blocking(move || {
                worker
                    .join()
                    .map_err(|_| "Motion Photo recorder panicked during shutdown".to_string())
            })
            .await
            .map_err(|e| e.to_string())??;
        }
        Ok(())
    }
}
impl Drop for MotionPhotoRecorder {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Release);
    }
}

pub(crate) struct PhotoJob {
    pub live: Option<MotionPhotoCapture>,
    pub still_path: Option<String>,
}

pub struct MotionPhotoCapture {
    pub snapshot: MotionPhotoSnapshot,
    pub result: Option<tokio::task::JoinHandle<Result<TemporaryClip, String>>>,
    pub cancel: Arc<AtomicBool>,
    pub temporary_path: PathBuf,
}
impl Drop for MotionPhotoCapture {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Release);
        // TemporaryClip owns all disk cleanup on the encoder/mux/embedding path.
        // Never unlink an accepted clip from the UI while it is being muxed.
    }
}

impl AppModel {
    pub(crate) fn should_show_motion_photo_button(&self) -> bool {
        self.mode == crate::app::CameraMode::Photo && self.motion_photo_format_compatible()
    }

    pub(crate) fn motion_photo_enabled(&self) -> bool {
        self.config.motion_photo
    }

    pub(crate) fn handle_toggle_motion_photo_capture(
        &mut self,
    ) -> cosmic::Task<cosmic::Action<crate::app::Message>> {
        if self.should_show_motion_photo_button() {
            let setting = self.handle_toggle_motion_photo(!self.config.motion_photo);
            return cosmic::Task::batch([
                setting,
                self.show_toggle_feedback(crate::app::toggle_feedback::ToggleKind::MotionPhoto),
            ]);
        }
        cosmic::Task::none()
    }

    pub(crate) fn handle_dismiss_motion_photo_error(
        &mut self,
    ) -> cosmic::Task<cosmic::Action<crate::app::Message>> {
        self.motion_photo_error = None;
        cosmic::Task::none()
    }

    #[cfg(test)]
    pub(crate) fn motion_photo_burst_crop(
        &self,
        width: u32,
        height: u32,
    ) -> Option<(u32, u32, u32, u32)> {
        self.motion_photo_capture
            .as_ref()
            .and_then(|live| live.snapshot.crop(width, height, true))
    }

    /// Crop a burst frame using the shutter recipe and that frame's ISP evidence.
    pub(crate) fn motion_photo_burst_frame_crop(
        &self,
        frame: &CameraFrame,
    ) -> Option<(u32, u32, u32, u32)> {
        self.motion_photo_capture.as_ref().and_then(|live| {
            // Keep the stored shutter zoom total: RAW and processed frames can
            // coexist, and only the latter have a completed crop baked in.
            let mut snapshot = live.snapshot.clone();
            snapshot.zoom = crate::backends::camera::zoom::residual_zoom(snapshot.zoom, frame);
            snapshot.crop(frame.width, frame.height, true)
        })
    }

    /// Snapshot settings once, before any photo path branches.
    pub(crate) fn start_motion_photo_capture(&mut self) -> Result<(), String> {
        if self.encoder_restart_required() {
            return Err(
                "Motion Photo video encoder failed; restart the device before recording again"
                    .into(),
            );
        }
        if self.motion_photo_capture.is_some()
            || self.recording.is_recording()
            || (self.timelapse.is_running() || self.timelapse.is_finalising())
        {
            return Err("Motion Photo cannot overlap another capture or recording".into());
        }
        if self.current_frame_is_file_source {
            return Err("Motion Photo requires a live camera, not a file preview source".into());
        }
        let _frame = self
            .current_frame
            .as_ref()
            .ok_or("Motion Photo requires a preview frame")?;
        let encoder = self
            .available_video_encoders
            .get(self.current_video_encoder_index)
            .ok_or("Motion Photo requires a selected video encoder")?
            .clone();
        crate::pipelines::motion_photo_encoder::validate_encoder(&encoder)?;
        let _audio_device = if self.config.record_audio {
            Some(
                self.available_audio_devices
                    .get(self.current_audio_device_index)
                    .ok_or("Motion Photo audio is enabled but no microphone is selected")?
                    .node_name
                    .clone(),
            )
        } else {
            None
        };
        let snapshot = self.motion_photo_snapshot();
        let started = Instant::now();
        let cancel = Arc::new(AtomicBool::new(false));
        let path = dirs::cache_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("cosmic-camera/motion-photo")
            .join(format!("{}.mp4", uuid::Uuid::new_v4()));
        self.recording_session_counter = self.recording_session_counter.wrapping_add(1);
        let operation = EncoderOperation::MotionPhoto(self.recording_session_counter);
        let monitor = RecordingMonitor::new(encoder.element_name.clone());
        // Post-roll collection does not start a pipeline or finalize anything.
        monitor.set_phase(RecordingPhase::Recording);
        self.register_encoder_operation(operation, monitor.clone());
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let mut request = ShutterRequest {
            completion: WorkerCompletion(monitor.clone()),
            postroll: None,
            shutter: started,
            path: path.clone(),
            cancel: cancel.clone(),
            result: result_tx,
        };
        // A bounded timestamp-only operation: no mutex, pipeline work or await.
        if let Some(recorder) = &self.motion_photo_recorder {
            recorder.postroll_requests.fetch_add(1, Ordering::AcqRel);
            request.postroll = Some(PostrollLease(recorder.postroll_requests.clone()));
            if let Err(e) = recorder.shutter_tx.try_send(request) {
                let request = match e {
                    std::sync::mpsc::TrySendError::Full(p)
                    | std::sync::mpsc::TrySendError::Disconnected(p) => p,
                };
                let _ = request
                    .result
                    .send(Err("Motion Photo recorder is unavailable or busy".into()));
            }
        } else {
            let _ = request.result.send(Err(
                "Motion Photo prebuffer is not ready (warming or unavailable)".into(),
            ));
        }
        let watchdog_cancel = cancel.clone();
        let result = tokio::spawn(await_clip_result(
            result_rx,
            watchdog_cancel,
            Duration::from_secs(10),
            monitor,
        ));
        self.motion_photo_capture = Some(MotionPhotoCapture {
            snapshot,
            result: Some(result),
            cancel,
            temporary_path: path,
        });
        Ok(())
    }

    fn motion_photo_snapshot(&self) -> MotionPhotoSnapshot {
        MotionPhotoSnapshot {
            photo_output_format: self.config.photo_output_format,
            camera_metadata: self.build_camera_metadata(),
            filter: self.selected_filter,
            zoom: self.current_zoom_level(),
            media_rotation: self.capture_media_rotation(),
            mirror: self.should_mirror_captures(),
            geometry_rotation: self.capture_geometry_rotation(),
            ratio: self.photo_aspect_ratio,
            fit: self.preview_fit_to_view,
            portrait: self.screen_is_portrait(),
            screen_width: self.screen_width,
            screen_height: self.screen_height,
            insets: self.settled_bar_insets(),
        }
    }

    pub(crate) fn motion_photo_format_compatible(&self) -> bool {
        matches!(
            self.config.photo_output_format,
            crate::config::PhotoOutputFormat::Jpeg | crate::config::PhotoOutputFormat::Avif
        )
    }

    fn retire_motion_photo_recorder(&mut self) {
        if let Some(mut recorder) = self.motion_photo_recorder.take() {
            recorder.cancel.store(true, Ordering::Release);
            self.motion_photo_retiring = recorder.worker.take();
        }
    }

    pub(crate) fn reconcile_shutdown_preview(&mut self) {
        if !self.pending_close {
            return;
        }
        let needs_postroll = self.motion_photo_recorder.as_ref().is_some_and(|recorder| {
            recorder.draining.store(true, Ordering::Release);
            recorder.postroll_requests.load(Ordering::Acquire) != 0
        });
        if !needs_postroll
            && !self.still_capture.has_request()
            && !self.burst_mode.is_collecting_frames()
        {
            self.camera_cancel_flag.store(true, Ordering::Release);
        }
    }

    pub(crate) fn sync_motion_photo_recorder(&mut self) {
        let dimensions = self.current_frame.as_ref().map(|f| (f.width, f.height));
        self.sync_motion_photo_recorder_dimensions(dimensions);
    }
    fn sync_motion_photo_recorder_dimensions(&mut self, dimensions: Option<(u32, u32)>) {
        // Check sticky worker errors even before the UI consumes its subscription.
        if self.encoder_restart_required()
            || self
                .motion_photo_recorder
                .as_ref()
                .is_some_and(|r| r.monitor.subscribe().borrow().error.is_some())
        {
            self.retire_motion_photo_recorder();
            return;
        }
        if self
            .motion_photo_retiring
            .as_ref()
            .is_some_and(|worker| !worker.is_finished())
        {
            return;
        }
        self.motion_photo_retiring = None;
        // Accepted capture windows survive close and settings changes until
        // their own completion; no new shutter is accepted once pending_close.
        if self.pending_close {
            return;
        }
        let allowed = self.motion_photo_enabled()
            && self.motion_photo_format_compatible()
            && self.mode == crate::app::CameraMode::Photo
            && !self.current_frame_is_file_source
            && !self.virtual_camera.is_file_source()
            && !self.recording.is_recording()
            && !self.timelapse.is_active()
            && !self.pending_close
            && !self.camera_cancel_flag.load(Ordering::Acquire);
        if !allowed {
            self.retire_motion_photo_recorder();
            return;
        }
        let Some((width, height)) = dimensions else {
            self.retire_motion_photo_recorder();
            return;
        };
        let Some(info) = self
            .available_video_encoders
            .get(self.current_video_encoder_index)
            .cloned()
        else {
            self.retire_motion_photo_recorder();
            return;
        };
        let audio = if self.config.record_audio {
            let Some(device) = self
                .available_audio_devices
                .get(self.current_audio_device_index)
            else {
                self.retire_motion_photo_recorder();
                return;
            };
            Some(device.node_name.clone())
        } else {
            None
        };
        let snapshot = self.motion_photo_snapshot();
        let recipe = RecorderRecipe {
            dimensions: (width, height),
            crop: snapshot.crop(width, height, false),
            filter: snapshot.filter,
            rotation: snapshot.media_rotation,
            mirror: snapshot.mirror,
            encoder: info.element_name.clone(),
            bitrate: self.config.bitrate_preset.bitrate_kbps(width, height),
            fps: self
                .active_format
                .as_ref()
                .and_then(|f| f.framerate)
                .map(|f| f.as_int())
                .unwrap_or(30)
                .clamp(1, 24),
            audio,
            geometry_rotation: snapshot.geometry_rotation,
            ratio: snapshot.ratio,
            fit: snapshot.fit,
            portrait: snapshot.portrait,
            screen_size: (
                snapshot.screen_width.to_bits(),
                snapshot.screen_height.to_bits(),
            ),
            insets: snapshot.insets,
        };
        if self.motion_photo_recorder.as_ref().is_some_and(|r| {
            r.recipe == recipe && Arc::ptr_eq(&r.source_cancel, &self.camera_cancel_flag)
        }) {
            return;
        }
        self.retire_motion_photo_recorder();
        if self.motion_photo_retiring.is_some() {
            return;
        }
        let (frame_tx, frames) = std::sync::mpsc::sync_channel(2);
        let (shutter_tx, shutters) = std::sync::mpsc::sync_channel(8);
        let cancel = Arc::new(AtomicBool::new(false));
        let source_cancel = self.camera_cancel_flag.clone();
        let processing = PostProcessingConfig {
            filter_type: recipe.filter,
            crop_rect: recipe.crop,
            ..Default::default()
        };
        let draining = Arc::new(AtomicBool::new(false));
        let worker_draining = draining.clone();
        let postroll_requests = Arc::new(AtomicUsize::new(0));
        let job_cancel = cancel.clone();
        let job_source_cancel = source_cancel.clone();
        let live_zoom = self.zoom_control.clone();
        let bitrate = recipe.bitrate;
        let fps = recipe.fps;
        let audio = recipe.audio.clone();
        let rotation = recipe.rotation;
        let mirror = recipe.mirror;
        #[cfg(test)]
        let pre_roll_ready = Arc::new(AtomicBool::new(false));
        #[cfg(test)]
        let worker_pre_roll_ready = pre_roll_ready.clone();
        self.recording_session_counter = self.recording_session_counter.wrapping_add(1);
        let operation = EncoderOperation::MotionPhoto(self.recording_session_counter);
        // An enabled prebuffer waiting for its first camera frame is not startup.
        let monitor = RecordingMonitor::new_waiting(info.element_name.clone());
        self.register_encoder_operation(operation, monitor.clone());
        let worker_monitor = monitor.clone();
        let job = std::thread::Builder::new()
            .name("motion-photo-memory".into())
            .spawn(move || {
                run_memory_recorder(
                    frames,
                    shutters,
                    processing,
                    live_zoom,
                    rotation,
                    mirror,
                    info,
                    bitrate,
                    fps,
                    audio,
                    job_cancel.clone(),
                    job_source_cancel,
                    worker_draining,
                    worker_monitor,
                    #[cfg(test)]
                    worker_pre_roll_ready,
                );
                job_cancel.store(true, Ordering::Release);
            });
        let worker = match job {
            Ok(worker) => worker,
            Err(e) => {
                let error = format!("Motion Photo worker could not start: {e}");
                monitor.report_error(error.clone());
                monitor.finish();
                self.finish_encoder_operation(operation, Some(error));
                return;
            }
        };
        self.motion_photo_recorder = Some(MotionPhotoRecorder {
            monitor,
            #[cfg(test)]
            pre_roll_ready,
            draining,
            postroll_requests,
            recipe,
            worker: Some(worker),
            frame_tx,
            shutter_tx,
            cancel,
            source_cancel,
        });
    }

    pub(crate) fn forward_motion_photo_frame(&mut self, frame: &Arc<CameraFrame>) {
        if !self.pending_close
            && (!self.motion_photo_enabled()
                || self.current_frame_is_file_source
                || self.mode != crate::CameraMode::Photo
                || self.recording.is_recording()
                || self.timelapse.is_active())
        {
            self.retire_motion_photo_recorder();
            return;
        }
        // Two integers + Arc identity only on the steady-state frame path.
        if self.motion_photo_enabled()
            && self.motion_photo_recorder.as_ref().is_none_or(|r| {
                r.recipe.dimensions != (frame.width, frame.height)
                    || !Arc::ptr_eq(&r.source_cancel, &self.camera_cancel_flag)
            })
        {
            self.sync_motion_photo_recorder_dimensions(Some((frame.width, frame.height)));
        }
        if let Some(recorder) = &self.motion_photo_recorder
            && !recorder.cancel.load(Ordering::Acquire)
            && !recorder.source_cancel.load(Ordering::Acquire)
        {
            let _ = recorder.frame_tx.try_send(frame.clone());
        }
    }
}

/// Wait for the independently fed clip, then atomically enrich the saved still.
/// The temporary clip is lifetime-owned and removed even when embedding fails.
pub(crate) async fn embed_completed_clip(
    path: PathBuf,
    job: tokio::task::JoinHandle<Result<TemporaryClip, String>>,
) -> Result<String, String> {
    let clip = job
        .await
        .map_err(|e| {
            format!(
                "Still saved at {}, Motion Photo task failed: {e}",
                path.display()
            )
        })?
        .map_err(|e| {
            format!(
                "Still saved at {}, Motion Photo failed: {e}",
                path.display()
            )
        })?;
    tokio::task::spawn_blocking(move || {
        crate::pipelines::motion_photo::embed_motion_photo(&path, &clip.0, -1)
            .map(|p| {
                // Keep the original on every encoding/publication failure. Only
                // remove the intermediate once the complete MP file is published.
                if let Err(error) = std::fs::remove_file(&path) {
                    tracing::warn!(%error, path = %path.display(), "Motion Photo saved but duplicate still could not be removed");
                }
                p.display().to_string()
            })
            .map_err(|e| {
                format!(
                    "Still saved at {}, Motion Photo embedding failed: {e}",
                    path.display()
                )
            })
    })
    .await
    .map_err(|e| format!("Motion Photo writer task failed: {e}"))?
}

fn orient_clip_frame(
    image: image::RgbImage,
    rotation: SensorRotation,
    mirror: bool,
) -> image::RgbImage {
    let image = match rotation {
        SensorRotation::None => image,
        SensorRotation::Rotate90 => image::imageops::rotate270(&image),
        SensorRotation::Rotate180 => image::imageops::rotate180(&image),
        SensorRotation::Rotate270 => image::imageops::rotate90(&image),
    };
    if mirror {
        image::imageops::flip_horizontal(&image)
    } else {
        image
    }
}

/// Watchdog for failed/unresponsive recording; the still job remains independent.
async fn await_clip_result(
    receiver: tokio::sync::oneshot::Receiver<Result<TemporaryClip, String>>,
    cancel: Arc<AtomicBool>,
    deadline: Duration,
    monitor: RecordingMonitor,
) -> Result<TemporaryClip, String> {
    match tokio::time::timeout(deadline, receiver).await {
        Ok(result) => {
            result.map_err(|_| "Motion Photo worker stopped before finalization".to_string())?
        }
        Err(_) => {
            cancel.store(true, Ordering::Release);
            let error = "Motion Photo worker exceeded its finalization deadline; still preserved"
                .to_string();
            monitor.report_error(error.clone());
            Err(error)
        }
    }
}

fn continuous_processor(processing: PostProcessingConfig) -> PostProcessor {
    PostProcessor::new(processing)
        .for_continuous_frames()
        .without_gpu_pacing()
}

/// Process in source coordinates, then orient for the encoder.
async fn process_clip_frame(
    processor: &PostProcessor,
    frame: Arc<CameraFrame>,
    desired_zoom: f32,
    dimensions: &mut Option<(u32, u32)>,
    rotation: SensorRotation,
    mirror: bool,
) -> Result<image::RgbImage, String> {
    let target = *dimensions.get_or_insert((frame.width, frame.height));
    let processed = processor.process_with_zoom(frame, desired_zoom).await?;
    // Zoom changes framing, never encoder resolution. Resample the entire
    // residual crop back to the unzoomed aspect region, then orient it.
    let image = if processed.image.dimensions() == target {
        processed.image
    } else {
        image::imageops::resize(
            &processed.image,
            target.0,
            target.1,
            image::imageops::FilterType::Triangle,
        )
    };
    Ok(orient_clip_frame(image, rotation, mirror))
}

struct WorkerCompletion(RecordingMonitor);
impl Drop for WorkerCompletion {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.report_error("Motion Photo worker panicked".into());
        }
        self.0.set_phase(RecordingPhase::Cleanup);
        self.0.finish();
    }
}

#[allow(clippy::too_many_arguments)]
fn run_memory_recorder(
    frames: std::sync::mpsc::Receiver<Arc<CameraFrame>>,
    shutters: std::sync::mpsc::Receiver<ShutterRequest>,
    processing: PostProcessingConfig,
    live_zoom: Arc<crate::backends::camera::zoom::ZoomControl>,
    rotation: SensorRotation,
    mirror: bool,
    info: EncoderInfo,
    bitrate: u32,
    fps: u32,
    audio: Option<String>,
    cancel: Arc<AtomicBool>,
    source_cancel: Arc<AtomicBool>,
    draining: Arc<AtomicBool>,
    monitor: RecordingMonitor,
    #[cfg(test)] pre_roll_ready: Arc<AtomicBool>,
) {
    // Declared before the encoder so Finished is published only after Drop's
    // potentially blocking driver cleanup has actually returned.
    let _completion = WorkerCompletion(monitor.clone());
    let origin = Instant::now();
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(error) => {
            monitor.report_error(format!("Motion Photo runtime could not start: {error}"));
            return;
        }
    };
    let mut clip_dimensions = processing
        .crop_rect
        .map(|(_, _, width, height)| (width, height));
    let processor = continuous_processor(processing);
    let mut encoder: Option<MemoryEncoder> = None;
    let mut pending: Vec<ShutterRequest> = Vec::new();
    let mut failure: Option<String> = None;
    let mut next_pts = Duration::ZERO;
    loop {
        if cancel.load(Ordering::Acquire) || source_cancel.load(Ordering::Acquire) {
            break;
        }
        #[cfg(test)]
        pre_roll_ready.store(
            encoder
                .as_ref()
                .is_some_and(|encoder| encoder.pre_roll_ready(origin.elapsed().as_nanos() as u64)),
            Ordering::Release,
        );
        while pending.len() < 8 {
            match shutters.try_recv() {
                Ok(p) => pending.push(p),
                Err(_) => break,
            }
        }
        let mut i = 0;
        while i < pending.len() {
            if pending[i].cancel.load(Ordering::Acquire) {
                let p = pending.remove(i);
                let _ = p.result.send(Err("Motion Photo capture cancelled".into()));
                continue;
            }
            if pending[i].shutter.elapsed()
                < Duration::from_nanos(POST_ROLL_NS) + Duration::from_millis(200)
            {
                i += 1;
                continue;
            }
            let mut p = pending.remove(i);
            let shutter = p.shutter.saturating_duration_since(origin).as_nanos() as u64;
            let window = if let Some(error) = &failure {
                Err(error.clone())
            } else if let Some(encoder) = &encoder {
                encoder.snapshot_window(shutter)
            } else {
                Err("Motion Photo prebuffer has no encoded frames".into())
            };
            // Snapshot owns its bytes. Producer may now stop even if muxing
            // and the still encoder have not finished yet.
            drop(p.postroll.take());
            let writer_draining = draining.clone();
            let writer_cancel = cancel.clone();
            let writer_source_cancel = source_cancel.clone();
            // The shared encoder keeps running while each bounded immutable
            // snapshot is muxed independently. At most eight accepted jobs.
            let _ = std::thread::Builder::new()
                .name("motion-photo-mux".into())
                .spawn(move || {
                    let result = window.and_then(|window| {
                        let temporary = TemporaryClip(p.path);
                        std::fs::create_dir_all(
                            temporary.0.parent().ok_or("Invalid clip cache path")?,
                        )
                        .map_err(|e| e.to_string())?;
                        let midpoint =
                            window.write_monitored(&temporary.0, p.completion.0.clone())?;
                        if p.cancel.load(Ordering::Acquire)
                            || (!writer_draining.load(Ordering::Acquire)
                                && (writer_cancel.load(Ordering::Acquire)
                                    || writer_source_cancel.load(Ordering::Acquire)))
                        {
                            return Err("Motion Photo source changed during mux".into());
                        }
                        tracing::debug!(
                            shutter_offset_us = midpoint / 1000,
                            "Motion Photo shutter position; exact still exposure unknown"
                        );
                        Ok(temporary)
                    });
                    let _ = p.result.send(result);
                });
        }
        // During shutdown, no prebuffer work after the last accepted window
        // has been copied. Mux writers are independent of this producer.
        if draining.load(Ordering::Acquire) && pending.is_empty() {
            break;
        }
        if let Some(enc) = &encoder
            && let Err(e) = enc.check_error()
        {
            monitor.report_error(e.clone());
            failure = Some(e);
        }
        let frame = match frames.recv_timeout(Duration::from_millis(10)) {
            Ok(f) => f,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        };
        if failure.is_some() || frame.captured_at < origin {
            continue;
        }
        let elapsed = frame.captured_at.duration_since(origin);
        if elapsed < next_pts {
            continue;
        }
        let processed = runtime.block_on(async {
            tokio::time::timeout(
                PROCESSING_TIMEOUT,
                process_clip_frame(
                    &processor,
                    frame,
                    live_zoom.zoom(),
                    &mut clip_dimensions,
                    rotation,
                    mirror,
                ),
            )
            .await
        });
        let image = match processed {
            Ok(Ok(image)) => image,
            Ok(Err(e)) => {
                monitor.report_error(e.clone());
                failure = Some(e);
                continue;
            }
            Err(_) => continue,
        };
        if cancel.load(Ordering::Acquire) || source_cancel.load(Ordering::Acquire) {
            break;
        }
        if encoder.is_none() {
            match MemoryEncoder::new_monitored(
                &info,
                image.width(),
                image.height(),
                fps,
                bitrate,
                audio.as_deref(),
                origin,
                monitor.clone(),
            ) {
                Ok(enc) => encoder = Some(enc),
                Err(e) => {
                    monitor.report_error(e.clone());
                    failure = Some(e);
                    continue;
                }
            }
        }
        if let Err(e) = encoder
            .as_mut()
            .unwrap()
            .push(image, elapsed.as_nanos() as u64)
        {
            monitor.report_error(e.clone());
            failure = Some(e);
        }
        next_pts = elapsed + Duration::from_nanos(1_000_000_000 / fps as u64);
    }
    // No frame is needed to stop: the bounded tick observes cancellation.
    for p in pending {
        let _ = p.result.send(Err(
            "Motion Photo cancelled because source/settings changed".into(),
        ));
    }
    while let Ok(p) = shutters.try_recv() {
        let _ = p.result.send(Err("Motion Photo recorder stopped".into()));
    }
}

#[cfg(test)]
#[path = "motion_photo_zoom_tests.rs"]
mod zoom_tests;

#[cfg(test)]
#[path = "motion_photo_pacing_tests.rs"]
mod pacing_tests;

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn failed_motion_clip_preserves_valid_independent_still_bytes() {
        let path =
            std::env::temp_dir().join(format!("motion-failed-clip-{}.jpg", uuid::Uuid::new_v4()));
        image::RgbImage::from_pixel(16, 16, image::Rgb([70, 80, 90]))
            .save(&path)
            .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let job = tokio::spawn(async { Err("synthetic video failure".into()) });
        let error = embed_completed_clip(path.clone(), job).await.unwrap_err();
        assert!(error.contains("Still saved at"));
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert!(image::open(&path).is_ok());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn failed_encoder_rejects_new_motion_clip_before_creating_work() {
        let mut model = AppModel::default();
        let monitor = RecordingMonitor::new("venus-encoder".into());
        monitor.report_error("synthetic driver failure".into());
        model.register_encoder_operation(EncoderOperation::MotionPhoto(1), monitor);
        let error = model.start_motion_photo_capture().unwrap_err();
        assert!(
            error.contains("restart"),
            "must reject from sticky monitor before validating a new capture: {error}"
        );
        assert!(model.motion_photo_capture.is_none());
        assert_eq!(model.encoder_operations.len(), 1);
    }

    #[tokio::test]
    async fn clip_deadline_reports_to_monitor_without_finishing_a_still_running_mux() {
        let monitor = RecordingMonitor::new("venus-encoder".into());
        monitor.set_phase(RecordingPhase::Finalizing);
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let result = await_clip_result(
            receiver,
            cancel.clone(),
            Duration::from_millis(1),
            monitor.clone(),
        )
        .await;
        assert!(result.is_err());
        assert!(cancel.load(Ordering::Acquire));
        assert!(
            monitor
                .subscribe()
                .borrow()
                .error
                .as_deref()
                .unwrap()
                .contains("deadline")
        );
        assert_eq!(
            monitor.subscribe().borrow().phase,
            RecordingPhase::Finalizing,
            "a watchdog must not pretend an unresponsive mux was cleaned up"
        );
        drop(sender);
    }

    #[tokio::test]
    async fn memory_worker_is_monitored_while_idle_and_retirement_cannot_restart_after_error() {
        let mut model = AppModel::default();
        model.config.record_audio = false;
        model.config.motion_photo = true;
        model.available_video_encoders.push(EncoderInfo {
            element_name: "x264enc".into(),
            display_name: "H264".into(),
            codec: crate::media::encoders::video::VideoCodec::H264,
            is_hardware: false,
            priority: 0,
        });
        model.sync_motion_photo_recorder_dimensions(Some((64, 48)));
        assert_eq!(
            model.encoder_operations.len(),
            1,
            "register before any camera input"
        );
        let (&operation, monitor) = model.encoder_operations.iter().next().unwrap();
        let monitor = monitor.clone();
        assert_eq!(
            monitor.subscribe().borrow().phase,
            crate::pipelines::video::RecordingPhase::WaitingForFrames,
            "no startup deadline while waiting for first preview frame"
        );
        monitor.report_error("synthetic stalled encoder".into());
        model.sync_motion_photo_recorder_dimensions(Some((64, 48)));
        assert!(
            model.motion_photo_recorder.is_none(),
            "failure must retire the worker synchronously"
        );
        assert!(
            model.encoder_operations.contains_key(&operation),
            "retirement must not unregister cleanup"
        );
        let worker = model.motion_photo_retiring.take().unwrap();
        tokio::task::spawn_blocking(move || worker.join().unwrap())
            .await
            .unwrap();
        assert_eq!(
            monitor.subscribe().borrow().phase,
            crate::pipelines::video::RecordingPhase::Finished
        );
        model.finish_encoder_operation(operation, None);
        model.sync_motion_photo_recorder_dimensions(Some((64, 48)));
        assert!(
            model.motion_photo_recorder.is_none(),
            "completed failed worker must not auto-restart"
        );
        assert!(model.recording_failure.is_some());
    }

    #[tokio::test]
    async fn unresponsive_recorder_reports_deadline_without_losing_the_still_job() {
        let (_sender, receiver) = tokio::sync::oneshot::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let result = tokio::time::timeout(
            Duration::from_millis(50),
            await_clip_result(
                receiver,
                cancel.clone(),
                Duration::from_millis(1),
                RecordingMonitor::new("test".into()),
            ),
        )
        .await;
        assert!(
            matches!(result, Ok(Err(ref message)) if message.contains("deadline")),
            "worker watchdog must report an explicit clip failure"
        );
        assert!(cancel.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn recorder_lifecycle_starts_before_shutter_and_stops_without_frames() {
        let mut model = AppModel::default();
        model.config.record_audio = false;
        model.current_frame = Some(Arc::new(CameraFrame {
            width: 64,
            height: 48,
            stride: 256,
            data: crate::backends::camera::types::FrameData::from_copied(
                vec![80; 64 * 48 * 4].into(),
            ),
            format: crate::backends::camera::types::PixelFormat::RGBA,
            yuv_planes: None,
            captured_at: Instant::now(),
            sensor_timestamp_ns: None,
            libcamera_metadata: None,
        }));
        model.available_video_encoders.push(EncoderInfo {
            element_name: "x264enc".into(),
            display_name: "H264".into(),
            codec: crate::media::encoders::video::VideoCodec::H264,
            is_hardware: false,
            priority: 0,
        });
        let _ = model.update(crate::app::Message::ToggleMotionPhoto(true));
        assert!(
            model.motion_photo_recorder.is_some(),
            "record before shutter, not just post-roll"
        );
        assert!(model.motion_photo_capture.is_none());
        let _ = model.update(crate::app::Message::ToggleMotionPhotoCapture);
        assert!(
            model.motion_photo_recorder.is_none(),
            "off must release without needing another camera frame"
        );
        assert!(!model.config.motion_photo);
    }

    #[test]
    fn incompatible_still_formats_hide_header_toggle_without_erasing_preference() {
        use crate::config::PhotoOutputFormat;
        let mut model = AppModel::default();
        model.config.motion_photo = true;
        for format in [PhotoOutputFormat::Png, PhotoOutputFormat::Dng] {
            model.config.photo_output_format = format;
            assert!(!model.should_show_motion_photo_button());
            assert!(!model.motion_photo_format_compatible());
            let _ = model.update(crate::app::Message::ToggleMotionPhotoCapture);
            assert!(
                model.config.motion_photo,
                "queued unsupported top-bar toggle must be ignored"
            );
            let _ = model.update(crate::app::Message::ToggleMotionPhoto(true));
            assert!(model.config.motion_photo);
            assert!(model.motion_photo_recorder.is_none());
            model.config.motion_photo = false;
            let _ = model.update(crate::app::Message::ToggleMotionPhoto(true));
            assert!(
                !model.config.motion_photo,
                "queued unsupported master enable must be ignored"
            );
            model.config.motion_photo = true;
        }
        for format in [PhotoOutputFormat::Jpeg, PhotoOutputFormat::Avif] {
            model.config.photo_output_format = format;
            assert!(model.motion_photo_format_compatible());
            assert!(model.should_show_motion_photo_button());
            assert!(model.motion_photo_enabled());
        }
    }

    #[test]
    fn top_bar_toggle_updates_preference_and_can_be_reenabled() {
        let mut model = AppModel::default();
        assert!(model.should_show_motion_photo_button());
        let _ = model.update(crate::app::Message::ToggleMotionPhoto(true));
        assert!(model.should_show_motion_photo_button());
        assert!(model.motion_photo_enabled());
        let _ = model.update(crate::app::Message::ToggleMotionPhotoCapture);
        assert!(!model.config.motion_photo);
        assert!(model.should_show_motion_photo_button());
        assert!(!model.motion_photo_enabled());
        let _ = model.update(crate::app::Message::ToggleMotionPhotoCapture);
        assert!(model.motion_photo_enabled());
        model.mode = crate::app::CameraMode::Video;
        assert!(!model.should_show_motion_photo_button());
        let _ = model.update(crate::app::Message::ToggleMotionPhotoCapture);
        assert!(model.config.motion_photo);
        model.mode = crate::app::CameraMode::Photo;
        let _ = model.update(crate::app::Message::ToggleMotionPhoto(false));
        assert!(model.should_show_motion_photo_button());
        assert!(!model.motion_photo_enabled());
    }

    #[test]
    fn top_bar_and_settings_toggle_share_persisted_motion_photo_preference() {
        use cosmic::cosmic_config::{Config as Store, CosmicConfigEntry};

        let root = std::env::temp_dir().join(format!(
            "camera-motion-photo-header-{}",
            uuid::Uuid::new_v4()
        ));
        let store =
            Store::with_custom_path("camera-test", crate::config::Config::VERSION, root.clone())
                .unwrap();
        let mut model = AppModel {
            config_handler: Some(store),
            ..Default::default()
        };

        for format in [
            crate::config::PhotoOutputFormat::Jpeg,
            crate::config::PhotoOutputFormat::Avif,
        ] {
            model.config.photo_output_format = format;
            for (message, enabled) in [
                (crate::app::Message::ToggleMotionPhoto(true), true),
                (crate::app::Message::ToggleMotionPhotoCapture, false),
                (crate::app::Message::ToggleMotionPhotoCapture, true),
                (crate::app::Message::ToggleMotionPhoto(false), false),
                (crate::app::Message::ToggleMotionPhotoCapture, true),
                (crate::app::Message::ToggleMotionPhotoCapture, false),
            ] {
                let _ = model.update(message);
                assert_eq!(model.config.motion_photo, enabled);
                assert_eq!(model.motion_photo_enabled(), enabled);
                assert!(model.should_show_motion_photo_button());
                model.config_persistence.flush();
                assert_eq!(
                    crate::config::Config::get_entry(model.config_handler.as_ref().unwrap())
                        .unwrap()
                        .motion_photo,
                    enabled
                );
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn dismiss_motion_photo_error_clears_only_the_warning() {
        let mut model = AppModel {
            motion_photo_error: Some("Synthetic encoder failure".into()),
            last_media_path: Some("preserved.jpg".into()),
            ..Default::default()
        };
        let _ = model.update(crate::app::Message::DismissMotionPhotoError);
        assert!(model.motion_photo_error.is_none());
        assert_eq!(model.last_media_path.as_deref(), Some("preserved.jpg"));
    }
    #[test]
    fn clip_pixels_match_still_inverse_sensor_rotation_and_upright_mirror() {
        let image = image::RgbImage::from_fn(2, 3, |x, y| image::Rgb([x as u8, y as u8, 0]));
        let expected = image::imageops::flip_horizontal(&image::imageops::rotate270(&image));
        assert_eq!(
            orient_clip_frame(image, SensorRotation::Rotate90, true),
            expected
        );
    }
}
