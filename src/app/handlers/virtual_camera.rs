// SPDX-License-Identifier: GPL-3.0-only

//! Virtual camera handlers
//!
//! Handles virtual camera streaming from camera feed or file sources (images/videos),
//! including preview playback, seeking, and play/pause controls.

use crate::app::state::{
    AppModel, FileSource, FilterType, Message, VideoPlaybackCommand, VirtualCameraState,
};
use cosmic::Task;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

/// Parameters for video file streaming to virtual camera
struct VideoStreamParams<'a> {
    zoom_control: Arc<crate::backends::camera::zoom::ZoomControl>,
    initial_filter: FilterType,
    filter_rx: &'a mut tokio::sync::watch::Receiver<FilterType>,
    stop_rx: tokio::sync::oneshot::Receiver<()>,
    preview_tx:
        tokio::sync::mpsc::UnboundedSender<Arc<crate::backends::camera::types::CameraFrame>>,
    progress_tx: tokio::sync::mpsc::UnboundedSender<(f64, f64, f64)>,
    control_rx: tokio::sync::mpsc::UnboundedReceiver<VideoPlaybackCommand>,
    initial_seek_position: f64,
    initial_paused: bool,
}

// Keep output preparation separate from preview ownership: callers retain the source.
async fn virtual_output_frame<'a>(
    frame: &'a crate::backends::camera::types::CameraFrame,
    zoom: &crate::backends::camera::zoom::ZoomControl,
) -> Result<std::borrow::Cow<'a, crate::backends::camera::types::CameraFrame>, String> {
    use crate::backends::camera::types::{CameraFrame, FrameData, PixelFormat};
    let desired = zoom.zoom();
    // Native processed frames may already contain all requested zoom. Keep the
    // original RGBA push fast path, including its stride, in that case.
    if frame.format == PixelFormat::RGBA
        && crate::backends::camera::zoom::residual_zoom(desired, frame) <= 1.0
    {
        return Ok(std::borrow::Cow::Borrowed(frame));
    }
    let rgba =
        crate::pipelines::video::recorder::convert_frame_to_zoomed_rgba(frame, desired).await?;
    Ok(std::borrow::Cow::Owned(CameraFrame {
        data: FrameData::from_owned_vec(rgba),
        format: PixelFormat::RGBA,
        stride: frame.width * 4,
        yuv_planes: None,
        ..frame.clone()
    }))
}

impl AppModel {
    // =========================================================================
    // Virtual Camera Handlers
    // =========================================================================

    pub(crate) fn handle_toggle_virtual_camera(&mut self) -> Task<cosmic::Action<Message>> {
        if self.virtual_camera.is_streaming() {
            self.animate_capture_scale(1.0);
        } else {
            self.animate_capture_scale(0.82);
        }
        if self.virtual_camera.is_streaming() {
            info!("Stopping virtual camera streaming");

            // For video file sources, save the current playback position BEFORE stopping
            // so we can resume from this position later
            if matches!(self.virtual_camera_file_source, Some(FileSource::Video(_)))
                && let Some((current_position, _, _)) = self.video_file_progress
            {
                self.video_preview_seek_position = current_position;
            }

            if let Some(sender) = self.virtual_camera.take_stop_sender() {
                let _ = sender.send(());
            }
            // Set to Idle immediately so UI updates (button state changes)
            // but don't send VirtualCameraStopped - the streaming thread will send it
            // when it actually stops. This avoids duplicate messages.
            self.virtual_camera = VirtualCameraState::Idle;
            self.invalidate_qr_detection();

            // Clear current frame to avoid accessing invalid mapped buffers
            // The frame might contain a GStreamer mapped buffer that becomes invalid
            // when the pipeline stops
            self.current_frame = None;

            return Task::none();
        }

        // Check if we have a file source
        if let Some(file_source) = &self.virtual_camera_file_source {
            return self.start_virtual_camera_from_file(file_source.clone());
        }

        // Start virtual camera from camera
        let Some(format) = &self.active_format else {
            error!("No active format for virtual camera");
            return Task::none();
        };

        let width = format.width;
        let height = format.height;
        let filter_type = self.selected_filter;
        let zoom_control = Arc::clone(&self.zoom_control);

        info!(
            width,
            height,
            ?filter_type,
            "Starting virtual camera streaming from camera"
        );

        let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel();
        let (frame_tx, mut frame_rx) = tokio::sync::mpsc::unbounded_channel();
        let (filter_tx, mut filter_rx) = tokio::sync::watch::channel(filter_type);
        self.virtual_camera = VirtualCameraState::start(stop_tx, frame_tx, filter_tx, false);

        // Start the virtual camera streaming on a DEDICATED THREAD
        // This is critical: CPU filtering is blocking and must NOT run on the async executor
        //
        // Use a oneshot channel to communicate completion back to the async task
        // This avoids blocking the executor with handle.join()
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();

        // Spawn the dedicated thread immediately (fire-and-forget from thread perspective)
        std::thread::spawn(move || {
            use crate::backends::virtual_camera::VirtualCameraManager;

            // Create and start the virtual camera on this dedicated thread
            let mut manager = VirtualCameraManager::new();
            manager.set_filter(filter_type);

            // Build a tokio runtime once. The virtual-camera output pipeline
            // accepts RGBA only, so YUV-format frames (e.g. MJPEG → I420 from
            // libcamera) must be converted before they're pushed; that
            // conversion lives behind an async API. Without it, the appsrc
            // buffer would interpret Y-only data as RGBA and downstream
            // consumer apps would see a stretched / tiled grayscale image
            // (4× horizontal tiles for a width:stride ratio of 1:4).
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    let _ =
                        done_tx.send(Err(format!("Failed to create virtual-cam runtime: {}", e)));
                    return;
                }
            };

            let result = (|| {
                if let Err(e) = manager.start(width, height) {
                    return Err(format!("Failed to start virtual camera: {}", e));
                }

                info!("Virtual camera started on dedicated thread, processing frames");

                // Process frames until channel closes
                let mut frame_count = 0u64;
                let mut dropped_count = 0u64;

                loop {
                    // Check for an explicit stop signal first — the channel-close
                    // fallback only fires when the frame sender is dropped, which
                    // we can't rely on if a future caller forgets to drop it.
                    if let Ok(()) = stop_rx.try_recv() {
                        info!("Received stop signal, stopping virtual camera");
                        break;
                    }

                    // Check for filter updates (non-blocking)
                    if filter_rx.has_changed().unwrap_or(false) {
                        let new_filter = *filter_rx.borrow_and_update();
                        manager.set_filter(new_filter);
                        info!(?new_filter, "Virtual camera filter updated");
                    }

                    // Wait for at least one frame (blocking is OK on dedicated thread)
                    let first_frame = match frame_rx.blocking_recv() {
                        Some(f) => f,
                        None => {
                            info!("Frame channel closed, stopping virtual camera");
                            break;
                        }
                    };

                    // Drain any additional frames, keeping only the latest
                    let mut latest_frame = first_frame;
                    while let Ok(newer_frame) = frame_rx.try_recv() {
                        latest_frame = newer_frame;
                        dropped_count += 1;
                    }

                    frame_count += 1;
                    if frame_count.is_multiple_of(30) {
                        debug!(
                            frame = frame_count,
                            dropped = dropped_count,
                            "Processing virtual camera frame"
                        );
                    }

                    // Convert non-RGBA and apply only source-aware residual zoom.
                    let push_result = match rt
                        .block_on(virtual_output_frame(&latest_frame, &zoom_control))
                    {
                        Ok(output) => manager.push_frame(&output),
                        Err(e) => {
                            warn!(?e, format = ?latest_frame.format, "Virtual output conversion failed; dropping frame");
                            continue;
                        }
                    };
                    if let Err(e) = push_result {
                        warn!(?e, "Failed to push frame to virtual camera");
                    }
                }

                info!("Shutting down virtual camera");

                if let Err(e) = manager.stop() {
                    warn!(?e, "Error stopping virtual camera");
                }

                Ok(())
            })();

            // Signal completion to the async task (ignore if receiver dropped)
            let _ = done_tx.send(result);
        });

        // Create async task that waits for thread completion WITHOUT blocking
        let streaming_task = Task::perform(
            async move {
                // This awaits the oneshot channel - doesn't block the executor!
                match done_rx.await {
                    Ok(result) => result,
                    Err(_) => Err("Virtual camera thread terminated unexpectedly".to_string()),
                }
            },
            |result| cosmic::Action::App(Message::VirtualCameraStopped(result)),
        );

        let start_signal = Task::done(cosmic::Action::App(Message::VirtualCameraStarted));

        Task::batch([start_signal, streaming_task])
    }

    /// Start virtual camera streaming from a file source (image or video)
    pub(crate) fn start_virtual_camera_from_file(
        &mut self,
        file_source: FileSource,
    ) -> Task<cosmic::Action<Message>> {
        // Stop any preview playback before starting streaming
        self.stop_video_preview_playback();

        let filter_type = self.selected_filter;
        let zoom_control = Arc::clone(&self.zoom_control);
        let is_video = matches!(file_source, FileSource::Video(_));

        info!(
            ?file_source,
            ?filter_type,
            "Starting virtual camera from file source"
        );

        // Create channels
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
        let (frame_tx, _frame_rx) = tokio::sync::mpsc::unbounded_channel();
        let (filter_tx, mut filter_rx) = tokio::sync::watch::channel(filter_type);

        // Create preview channel for sending frames back to UI
        let (preview_tx, preview_rx) = tokio::sync::mpsc::unbounded_channel();
        self.file_source_preview_receiver =
            Some(std::sync::Arc::new(tokio::sync::Mutex::new(preview_rx)));

        // Create progress channel for video files
        let (progress_tx, progress_rx) = tokio::sync::mpsc::unbounded_channel::<(f64, f64, f64)>();

        // Create playback control channel for video files
        let (control_tx, control_rx) =
            tokio::sync::mpsc::unbounded_channel::<VideoPlaybackCommand>();

        // Use start_file_source to mark this as file source streaming
        self.virtual_camera = VirtualCameraState::start(stop_tx, frame_tx, filter_tx, true);

        // For video files, keep the current progress (with stored seek position) until
        // the streaming thread sends actual progress updates. This prevents the slider
        // from jumping to 0 briefly before showing the correct position.
        // For non-video files, clear the progress.
        // Note: We preserve video_file_paused state so streaming respects play/pause.
        if !is_video {
            self.video_file_progress = None;
        }

        // Store control channel for video files
        if is_video {
            self.video_playback_control_tx = Some(control_tx);
        } else {
            self.video_playback_control_tx = None;
        }

        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();

        // Get the stored seek position and paused state to apply when streaming starts
        let initial_seek_position = self.video_preview_seek_position;
        let initial_paused = self.video_file_paused;

        // Spawn dedicated thread for file source streaming
        std::thread::spawn(move || {
            let result = match file_source {
                FileSource::Image(path) => Self::stream_image_to_virtual_camera(
                    &path,
                    filter_type,
                    &mut filter_rx,
                    stop_rx,
                    preview_tx,
                    zoom_control,
                ),
                FileSource::Video(path) => Self::stream_video_to_virtual_camera(
                    &path,
                    VideoStreamParams {
                        zoom_control,
                        initial_filter: filter_type,
                        filter_rx: &mut filter_rx,
                        stop_rx,
                        preview_tx,
                        progress_tx,
                        control_rx,
                        initial_seek_position,
                        initial_paused,
                    },
                ),
            };

            let _ = done_tx.send(result);
        });

        let streaming_task = Task::perform(
            async move {
                match done_rx.await {
                    Ok(result) => result,
                    Err(_) => Err("Virtual camera thread terminated unexpectedly".to_string()),
                }
            },
            |result| cosmic::Action::App(Message::VirtualCameraStopped(result)),
        );

        let start_signal = Task::done(cosmic::Action::App(Message::VirtualCameraStarted));

        // For video files, also spawn a task to receive progress updates
        if is_video {
            let progress_task = Task::run(
                futures::stream::unfold(progress_rx, |mut rx| async move {
                    rx.recv().await.map(|(pos, dur, progress)| {
                        (Message::VideoFileProgress(pos, dur, progress), rx)
                    })
                }),
                cosmic::Action::App,
            );
            Task::batch([start_signal, streaming_task, progress_task])
        } else {
            Task::batch([start_signal, streaming_task])
        }
    }

    /// Stream an image file to the virtual camera at ~30fps
    fn stream_image_to_virtual_camera(
        path: &std::path::Path,
        initial_filter: FilterType,
        filter_rx: &mut tokio::sync::watch::Receiver<FilterType>,
        mut stop_rx: tokio::sync::oneshot::Receiver<()>,
        preview_tx: tokio::sync::mpsc::UnboundedSender<
            Arc<crate::backends::camera::types::CameraFrame>,
        >,
        zoom_control: Arc<crate::backends::camera::zoom::ZoomControl>,
    ) -> Result<(), String> {
        use crate::backends::virtual_camera::{VirtualCameraManager, load_image_as_frame};

        // Load the image
        let frame =
            load_image_as_frame(path).map_err(|e| format!("Failed to load image: {}", e))?;

        let width = frame.width;
        let height = frame.height;

        // Create and start virtual camera manager.
        //
        // File sources are streamed exactly as they appear in the file — the
        // app's `mirror_preview` setting is for selfie-cam ergonomics and
        // doesn't apply here. Pin `flip_horizontal` to `false` defensively so
        // a future change to the default can't silently start mirroring
        // streamed images to consumer apps.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("Failed to create virtual-cam runtime: {}", e))?;
        let mut manager = VirtualCameraManager::new();
        manager.set_filter(initial_filter);

        if let Err(e) = manager.start(width, height) {
            return Err(format!("Failed to start virtual camera: {}", e));
        }

        info!(width, height, "Streaming image to virtual camera");

        // Stream at approximately 30fps
        use crate::constants::virtual_camera as vc_timing;
        let frame_duration = vc_timing::IMAGE_STREAM_FRAME_DURATION;
        let mut frame_count = 0u64;

        // Wrap frame in Arc for preview
        let frame_arc = Arc::new(frame);

        loop {
            // Check for stop signal
            match stop_rx.try_recv() {
                Ok(()) | Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    info!("Stop signal received, stopping image stream");
                    break;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {}
            }

            // Check for filter updates
            if filter_rx.has_changed().unwrap_or(false) {
                let new_filter = *filter_rx.borrow_and_update();
                manager.set_filter(new_filter);
                info!(?new_filter, "Virtual camera filter updated");
            }

            // Push the frame to virtual camera (file sources are never mirrored - they
            // display content as-is, unlike camera selfie preview which mirrors)
            if let Err(e) = rt.block_on(async {
                let output = virtual_output_frame(&frame_arc, &zoom_control).await?;
                manager.push_frame(&output).map_err(|e| e.to_string())
            }) {
                warn!(?e, "Failed to push frame to virtual camera");
            }

            // Send frame to preview (ignore errors if UI is not consuming fast enough)
            let _ = preview_tx.send(Arc::clone(&frame_arc));

            frame_count += 1;
            if frame_count.is_multiple_of(30) {
                debug!(frame = frame_count, "Image streaming");
            }

            std::thread::sleep(frame_duration);
        }

        if let Err(e) = manager.stop() {
            warn!(?e, "Error stopping virtual camera");
        }

        Ok(())
    }

    /// Stream a video file to the virtual camera with looping
    fn stream_video_to_virtual_camera(
        path: &std::path::Path,
        params: VideoStreamParams<'_>,
    ) -> Result<(), String> {
        let VideoStreamParams {
            zoom_control,
            initial_filter,
            filter_rx,
            mut stop_rx,
            preview_tx,
            progress_tx,
            control_rx,
            initial_seek_position,
            initial_paused,
        } = params;
        use crate::backends::virtual_camera::{VideoDecoder, VirtualCameraManager};

        // Create video decoder
        let decoder = VideoDecoder::new(path)
            .map_err(|e| format!("Failed to create video decoder: {}", e))?;

        // Apply initial seek position if set (user sought while not streaming)
        if initial_seek_position > 0.0 {
            info!(initial_seek_position, "Applying initial seek position");
            decoder.seek(initial_seek_position);
        }

        let (width, height) = decoder.dimensions();

        // Create and start virtual camera manager.
        //
        // File sources are streamed exactly as they appear in the file — the
        // app's `mirror_preview` setting is for selfie-cam ergonomics and
        // doesn't apply here. Pin `flip_horizontal` to `false` defensively so
        // a future change to the default can't silently start mirroring
        // streamed video files to consumer apps.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("Failed to create virtual-cam runtime: {}", e))?;
        let mut manager = VirtualCameraManager::new();
        manager.set_filter(initial_filter);

        if let Err(e) = manager.start(width, height) {
            return Err(format!("Failed to start virtual camera: {}", e));
        }

        info!(
            width,
            height,
            has_audio = decoder.has_audio(),
            path = %path.display(),
            "Streaming video to virtual camera (looping)"
        );

        // Get preroll frame immediately for instant preview.
        // We keep an Arc handle to the most recent frame so we can re-push it
        // while playback is paused — otherwise consumer apps connecting to
        // the virtual-camera PipeWire node receive a single frame and then
        // nothing, which most apps render as a black window (whereas the
        // image-source path stays "alive" by re-pushing the same image at
        // 30fps).
        let mut last_frame: Option<Arc<crate::backends::camera::types::CameraFrame>> = None;
        if let Some(preroll) = decoder.preroll_frame() {
            let frame_arc = Arc::new(preroll);
            if let Err(e) = rt.block_on(async {
                let output = virtual_output_frame(&frame_arc, &zoom_control).await?;
                manager.push_frame(&output).map_err(|e| e.to_string())
            }) {
                warn!(?e, "Failed to push preroll frame to virtual camera");
            }
            let _ = preview_tx.send(Arc::clone(&frame_arc));
            last_frame = Some(frame_arc);
        }

        let mut frame_count = 0u64;
        let mut last_progress_update = std::time::Instant::now();
        let mut paused = initial_paused;
        let mut control_rx = control_rx;

        // Apply initial paused state
        if initial_paused {
            decoder.set_paused(true);
            info!("Starting video in paused state");
        }

        loop {
            // Check for stop signal
            match stop_rx.try_recv() {
                Ok(()) | Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    info!("Stop signal received, stopping video stream");
                    break;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {}
            }

            // Check for playback control commands. Coalesce queued Seeks so
            // a continuous slider drag only triggers one (expensive)
            // GStreamer seek per loop iteration — the intermediate positions
            // would have produced frames the user never sees anyway.
            let mut needs_frame_update = false;
            let mut latest_seek: Option<f64> = None;
            while let Ok(cmd) = control_rx.try_recv() {
                match cmd {
                    VideoPlaybackCommand::Seek(position) => {
                        latest_seek = Some(position);
                        // When seeking while paused, we need to pull a frame to update display
                        if paused {
                            needs_frame_update = true;
                        }
                    }
                    VideoPlaybackCommand::TogglePause => {
                        paused = !paused;
                        decoder.set_paused(paused);
                        info!(paused, "Video pause toggled");
                    }
                    VideoPlaybackCommand::SetPaused(p) => {
                        paused = p;
                        decoder.set_paused(paused);
                        info!(paused, "Video pause set");
                    }
                }
            }
            if let Some(position) = latest_seek {
                info!(position, "Seeking video (coalesced)");
                decoder.seek(position);
            }

            // Check for filter updates
            if filter_rx.has_changed().unwrap_or(false) {
                let new_filter = *filter_rx.borrow_and_update();
                manager.set_filter(new_filter);
                info!(?new_filter, "Virtual camera filter updated");
            }

            // Send progress updates at regular intervals
            use crate::constants::virtual_camera as vc_timing;
            if last_progress_update.elapsed() >= vc_timing::PROGRESS_UPDATE_INTERVAL {
                if let (Some(pos), Some(dur)) = (decoder.position(), decoder.duration()) {
                    let progress = if dur > 0.0 {
                        (pos / dur).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let _ = progress_tx.send((pos, dur, progress));
                }
                last_progress_update = std::time::Instant::now();
            }

            // If paused and no frame update needed, keep re-pushing the last
            // frame so the consumer app sees a still image instead of going
            // black. The decoder itself stays paused — we just refresh the
            // virtual-camera pipeline's input.
            if paused && !needs_frame_update {
                if let Some(ref frame_arc) = last_frame
                    && let Err(e) = rt.block_on(async {
                        let output = virtual_output_frame(frame_arc, &zoom_control).await?;
                        manager.push_frame(&output).map_err(|e| e.to_string())
                    })
                {
                    warn!(?e, "Failed to re-push paused frame to virtual camera");
                }
                std::thread::sleep(vc_timing::PAUSE_CHECK_INTERVAL);
                continue;
            }

            // If we need a frame update while paused (after seeking), temporarily unpause to get a frame
            if needs_frame_update && paused {
                decoder.set_paused(false);
            }

            // Get next frame from video
            match decoder.next_frame() {
                Some(frame) => {
                    // Wrap frame in Arc
                    let frame_arc = Arc::new(frame);

                    // Push to virtual camera (file sources are never mirrored - they
                    // display content as-is, unlike camera selfie preview which mirrors)
                    if let Err(e) = rt.block_on(async {
                        let output = virtual_output_frame(&frame_arc, &zoom_control).await?;
                        manager.push_frame(&output).map_err(|e| e.to_string())
                    }) {
                        warn!(?e, "Failed to push frame to virtual camera");
                    }

                    // Send frame to preview
                    let _ = preview_tx.send(Arc::clone(&frame_arc));
                    last_frame = Some(Arc::clone(&frame_arc));

                    // If we got a frame update while paused (after seeking), re-pause and send progress
                    if needs_frame_update && paused {
                        decoder.set_paused(true);
                        // Send immediate progress update
                        if let (Some(pos), Some(dur)) = (decoder.position(), decoder.duration()) {
                            let progress = if dur > 0.0 {
                                (pos / dur).clamp(0.0, 1.0)
                            } else {
                                0.0
                            };
                            let _ = progress_tx.send((pos, dur, progress));
                        }
                    }

                    frame_count += 1;
                    if frame_count.is_multiple_of(30) {
                        debug!(frame = frame_count, "Video streaming");
                    }
                }
                None => {
                    // If we temporarily unpaused for a frame update, re-pause even if no frame
                    if needs_frame_update && paused {
                        decoder.set_paused(true);
                    }
                    // Check if we hit end of stream
                    if decoder.is_eos() {
                        info!("Video ended, restarting for loop");
                        if let Err(e) = decoder.restart() {
                            warn!(?e, "Failed to restart video, continuing");
                        }
                    } else {
                        // No frame available, wait a bit
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                }
            }
        }

        decoder.stop();

        if let Err(e) = manager.stop() {
            warn!(?e, "Error stopping virtual camera");
        }

        Ok(())
    }

    pub(crate) fn handle_virtual_camera_started(&mut self) -> Task<cosmic::Action<Message>> {
        info!("Virtual camera streaming started successfully");
        self.update_idle_inhibit();
        Self::delay_task(1000, Message::UpdateVirtualCameraDuration)
    }

    pub(crate) fn handle_virtual_camera_stopped(
        &mut self,
        result: Result<(), String>,
    ) -> Task<cosmic::Action<Message>> {
        self.virtual_camera = VirtualCameraState::Idle;
        self.update_idle_inhibit();
        // Clear the file source preview receiver (only relevant for file source streaming)
        self.file_source_preview_receiver = None;

        // For video file sources, preserve the current position and duration
        // This allows continued seeking while not streaming from where playback stopped
        let mut post_stop_task = None;
        if let Some((current_position, duration, _)) = self.video_file_progress {
            if let Some(FileSource::Video(ref path)) = self.virtual_camera_file_source {
                // Use the current playback position, not the stored seek position
                let position = current_position;
                let progress = if duration > 0.0 {
                    position / duration
                } else {
                    0.0
                };
                self.video_file_progress = Some((position, duration, progress));
                // Update the stored seek position so it's used if streaming restarts
                self.video_preview_seek_position = position;

                // If video was playing during streaming, continue playing in preview
                // Otherwise just load a static frame at the current position
                if !self.video_file_paused {
                    info!("Video was playing during streaming, continuing preview playback");
                    // Use Task::done to trigger playback in next update cycle
                    // This ensures clean state after streaming has fully stopped
                    post_stop_task = Some(Task::done(cosmic::Action::App(
                        Message::StartVideoPreviewPlayback,
                    )));
                } else {
                    // Load preview frame at the current seek position
                    let path = path.clone();
                    post_stop_task = Some(Task::perform(
                        async move {
                            use crate::backends::virtual_camera::load_video_frame_at_position;

                            match load_video_frame_at_position(&path, position) {
                                Ok(frame) => Some(Arc::new(frame)),
                                Err(e) => {
                                    warn!(?e, "Failed to load preview frame after stopping");
                                    None
                                }
                            }
                        },
                        |frame| cosmic::Action::App(Message::VideoSeekPreviewLoaded(frame)),
                    ));
                }
            } else {
                self.video_file_progress = None;
            }
        }

        // Note: We preserve video_file_paused state so it's applied when streaming restarts
        self.video_playback_control_tx = None;

        // Clear any active transition to ensure UI is enabled
        // This is important when stopping file source streaming, as no frames will arrive
        // to naturally clear the transition
        self.transition_state.clear();

        match result {
            Ok(()) => {
                info!("Virtual camera stopped successfully");
            }
            Err(err) => {
                error!(error = %err, "Virtual camera error");
            }
        }

        post_stop_task.unwrap_or_else(Task::none)
    }

    pub(crate) fn handle_update_virtual_camera_duration(
        &mut self,
    ) -> Task<cosmic::Action<Message>> {
        if self.virtual_camera.is_streaming() {
            return Self::delay_task(1000, Message::UpdateVirtualCameraDuration);
        }
        Task::none()
    }

    pub(crate) fn handle_open_virtual_camera_file(&self) -> Task<cosmic::Action<Message>> {
        info!("Opening file picker for virtual camera source");

        Task::perform(
            async {
                use rfd::AsyncFileDialog;

                let file = AsyncFileDialog::new()
                    .add_filter(
                        crate::fl!("virtual-camera-file-filter-name"),
                        &[
                            "png", "jpg", "jpeg", "gif", "bmp", "webp", "mp4", "mkv", "webm",
                            "avi", "mov",
                        ],
                    )
                    .pick_file()
                    .await;

                if let Some(file) = file {
                    let path = file.path().to_path_buf();
                    let extension = path
                        .extension()
                        .and_then(|e| e.to_str())
                        .map(|e| e.to_lowercase())
                        .unwrap_or_default();

                    // Determine if it's an image or video based on extension
                    use crate::constants::file_formats;
                    if file_formats::is_image_extension(&extension) {
                        Some(FileSource::Image(path))
                    } else if file_formats::is_video_extension(&extension) {
                        Some(FileSource::Video(path))
                    } else {
                        None
                    }
                } else {
                    None
                }
            },
            |file_source| cosmic::Action::App(Message::VirtualCameraFileSelected(file_source)),
        )
    }

    pub(crate) fn handle_virtual_camera_file_selected(
        &mut self,
        file_source: Option<FileSource>,
    ) -> Task<cosmic::Action<Message>> {
        if let Some(ref source) = file_source {
            info!(?source, "Virtual camera file source selected");
            self.invalidate_qr_detection();

            // Get the path to load preview from
            let path = match source {
                FileSource::Image(p) | FileSource::Video(p) => p.clone(),
            };

            let is_video = matches!(source, FileSource::Video(_));
            self.virtual_camera_file_source = file_source;

            // Reset seek position when a new file is selected
            // Start in paused state since the video isn't playing yet (just showing preview frame)
            self.video_preview_seek_position = 0.0;
            self.video_file_paused = true;

            // Load preview frame (and duration for videos) asynchronously
            return Task::perform(
                async move {
                    use crate::backends::virtual_camera::{get_video_duration, load_preview_frame};

                    let frame = match load_preview_frame(&path) {
                        Ok(frame) => Some(Arc::new(frame)),
                        Err(e) => {
                            warn!(?e, "Failed to load preview frame");
                            None
                        }
                    };

                    // For videos, also get the duration
                    let duration = if is_video {
                        match get_video_duration(&path) {
                            Ok(dur) => Some(dur),
                            Err(e) => {
                                warn!(?e, "Failed to get video duration");
                                None
                            }
                        }
                    } else {
                        None
                    };

                    (frame, duration)
                },
                |(frame, duration)| {
                    cosmic::Action::App(Message::FileSourcePreviewLoaded(frame, duration))
                },
            );
        } else {
            info!("File picker cancelled");
        }
        self.virtual_camera_file_source = file_source;
        Task::none()
    }

    pub(crate) fn handle_file_source_preview_loaded(
        &mut self,
        frame: Option<Arc<crate::backends::camera::types::CameraFrame>>,
        duration: Option<f64>,
    ) -> Task<cosmic::Action<Message>> {
        let mut qr_task = Task::none();
        if let Some(frame) = frame {
            info!(
                width = frame.width,
                height = frame.height,
                ?duration,
                "File source preview loaded"
            );
            self.invalidate_qr_detection();
            qr_task = self.start_qr_detection(&frame);
            self.current_frame = Some(frame);
            self.current_frame_is_file_source = true;
            // Reset aspect ratio to Native so the file source displays uncropped
            self.photo_aspect_ratio = crate::app::state::PhotoAspectRatio::Native;
            // Readiness signal for the screenshot harness: the first preview
            // frame is now set, so the app has something to draw and is ready to
            // capture. Emitted on a dedicated `preview_ready` target so it is
            // filtered out of normal runs unless the harness opts in via
            // RUST_LOG=...,preview_ready=debug (see preview/capture-previews.sh).
            tracing::debug!(target: "preview_ready", "first preview frame ready");
        } else {
            warn!("Failed to load file source preview");
        }

        // If we have video duration, set up initial progress (position at 0 or stored seek position)
        if let Some(dur) = duration {
            let position = self.video_preview_seek_position;
            let progress = if dur > 0.0 { position / dur } else { 0.0 };
            self.video_file_progress = Some((position, dur, progress));

            // Kick off the persistent preview decoder (in whatever paused
            // state the user is in — defaults to true right after picking).
            // This way subsequent slider scrubs route a Seek command through
            // the existing pipeline instead of spinning a new one per scrub
            // via `load_video_frame_at_position`.
            if matches!(self.virtual_camera_file_source, Some(FileSource::Video(_))) {
                return Task::batch([qr_task, self.start_video_preview_playback()]);
            }
        }

        qr_task
    }

    pub(crate) fn handle_clear_virtual_camera_file(&mut self) -> Task<cosmic::Action<Message>> {
        info!("Clearing virtual camera file source, switching back to camera");
        // Stop any preview playback
        self.stop_video_preview_playback();
        self.virtual_camera_file_source = None;
        self.current_frame_is_file_source = false;
        self.video_file_progress = None;
        self.video_preview_seek_position = 0.0;
        self.video_file_paused = false;
        self.invalidate_qr_detection();
        // Clear current frame so camera subscription can update it with camera feed
        self.current_frame = None;
        Task::none()
    }

    pub(crate) fn handle_video_file_progress(
        &mut self,
        position: f64,
        duration: f64,
        progress: f64,
    ) -> Task<cosmic::Action<Message>> {
        self.video_file_progress = Some((position, duration, progress));
        Task::none()
    }

    pub(crate) fn handle_video_file_seek(
        &mut self,
        position: f64,
    ) -> Task<cosmic::Action<Message>> {
        self.invalidate_qr_detection();
        // Always store the seek position for when streaming starts
        self.video_preview_seek_position = position;

        if self.virtual_camera.is_streaming() {
            // If streaming, send seek command to the streaming decoder
            if let Some(ref tx) = self.video_playback_control_tx
                && tx.send(VideoPlaybackCommand::Seek(position)).is_ok()
            {
                info!(position, "Seeking streaming video to position");
            }
            Task::none()
        } else if let Some(ref tx) = self.video_preview_control_tx {
            // If preview playback is active, send seek command to preview decoder
            if tx.send(VideoPlaybackCommand::Seek(position)).is_ok() {
                info!(position, "Seeking preview video to position");
            }
            Task::none()
        } else {
            // If not streaming and no preview playback, update progress and load a preview frame
            if let Some((_, duration, _)) = self.video_file_progress {
                let progress = if duration > 0.0 {
                    position / duration
                } else {
                    0.0
                };
                self.video_file_progress = Some((position, duration, progress));
            }

            // Load preview frame at the new position
            if let Some(FileSource::Video(ref path)) = self.virtual_camera_file_source {
                let path = path.clone();
                info!(position, "Loading preview frame at seek position");
                return Task::perform(
                    async move {
                        use crate::backends::virtual_camera::load_video_frame_at_position;

                        match load_video_frame_at_position(&path, position) {
                            Ok(frame) => Some(Arc::new(frame)),
                            Err(e) => {
                                warn!(?e, "Failed to load seek preview frame");
                                None
                            }
                        }
                    },
                    |frame| cosmic::Action::App(Message::VideoSeekPreviewLoaded(frame)),
                );
            }
            Task::none()
        }
    }

    pub(crate) fn handle_video_seek_preview_loaded(
        &mut self,
        frame: Option<Arc<crate::backends::camera::types::CameraFrame>>,
    ) -> Task<cosmic::Action<Message>> {
        if let Some(frame) = frame {
            debug!(
                width = frame.width,
                height = frame.height,
                "Seek preview frame loaded"
            );
            let qr_task = self.start_qr_detection(&frame);
            self.current_frame = Some(frame);
            self.current_frame_is_file_source = true;
            return qr_task;
        }
        Task::none()
    }

    pub(crate) fn handle_toggle_video_play_pause(&mut self) -> Task<cosmic::Action<Message>> {
        // Always toggle the paused state
        self.video_file_paused = !self.video_file_paused;
        info!(paused = self.video_file_paused, "Video play/pause toggled");

        if self.virtual_camera.is_streaming() {
            // If streaming, send the command to the streaming decoder
            if let Some(ref tx) = self.video_playback_control_tx {
                let _ = tx.send(VideoPlaybackCommand::TogglePause);
            }
        } else if let Some(ref tx) = self.video_preview_control_tx {
            // Persistent preview decoder is already running — just toggle it.
            let _ = tx.send(VideoPlaybackCommand::SetPaused(self.video_file_paused));
        } else if !self.video_file_paused {
            // Defensive: preview decoder isn't up (decoder failed to start, or
            // the file source was loaded before this code path existed). Start
            // one so play actually plays.
            return self.start_video_preview_playback();
        }
        Task::none()
    }

    pub(crate) fn handle_video_preview_playback_update(
        &mut self,
        frame: Arc<crate::backends::camera::types::CameraFrame>,
        position: f64,
        duration: f64,
        progress: f64,
    ) -> Task<cosmic::Action<Message>> {
        let qr_task = self.start_qr_detection(&frame);
        self.current_frame = Some(frame);
        self.current_frame_is_file_source = true;
        self.video_file_progress = Some((position, duration, progress));
        self.video_preview_seek_position = position;
        qr_task
    }

    pub(crate) fn handle_video_preview_playback_stopped(
        &mut self,
    ) -> Task<cosmic::Action<Message>> {
        info!("Video preview playback stopped");
        self.video_preview_control_tx = None;
        self.video_preview_stop_tx = None;
        Task::none()
    }

    /// Start video preview playback (when not streaming).
    ///
    /// Kept alive for the entire lifetime of the video file source so seek
    /// requests reuse the same `VideoDecoder` instead of spinning a fresh
    /// GStreamer pipeline per scrub (which the old `load_video_frame_at_position`
    /// fallback did, and which made scrubbing the slider visibly laggy).
    pub(crate) fn start_video_preview_playback(&mut self) -> Task<cosmic::Action<Message>> {
        // Only start if we have a video file and are not already playing
        let path = match &self.virtual_camera_file_source {
            Some(FileSource::Video(p)) => p.clone(),
            _ => return Task::none(),
        };

        // Stop any existing preview playback
        self.stop_video_preview_playback();

        info!(
            paused = self.video_file_paused,
            "Starting video preview playback"
        );

        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
        let (control_tx, control_rx) = tokio::sync::mpsc::unbounded_channel();
        let (frame_tx, frame_rx) = tokio::sync::mpsc::unbounded_channel();

        self.video_preview_stop_tx = Some(stop_tx);
        self.video_preview_control_tx = Some(control_tx);

        let initial_position = self.video_preview_seek_position;
        let initial_paused = self.video_file_paused;

        // Spawn preview playback thread
        std::thread::spawn(move || {
            Self::run_video_preview_playback(
                path,
                initial_position,
                initial_paused,
                stop_rx,
                control_rx,
                frame_tx,
            );
        });

        // Return a task that receives frames and sends messages
        use futures::StreamExt;
        Task::run(
            futures::stream::unfold(frame_rx, |mut rx| async move {
                rx.recv().await.map(|(frame, pos, dur, progress)| {
                    (
                        Message::VideoPreviewPlaybackUpdate(frame, pos, dur, progress),
                        rx,
                    )
                })
            })
            .chain(futures::stream::once(async {
                Message::VideoPreviewPlaybackStopped
            })),
            cosmic::Action::App,
        )
    }

    /// Stop video preview playback
    pub(crate) fn stop_video_preview_playback(&mut self) {
        if let Some(stop_tx) = self.video_preview_stop_tx.take() {
            let _ = stop_tx.send(());
        }
        self.video_preview_control_tx = None;
    }

    /// Run video preview playback in a background thread
    fn run_video_preview_playback(
        path: std::path::PathBuf,
        initial_position: f64,
        initial_paused: bool,
        mut stop_rx: tokio::sync::oneshot::Receiver<()>,
        mut control_rx: tokio::sync::mpsc::UnboundedReceiver<VideoPlaybackCommand>,
        frame_tx: tokio::sync::mpsc::UnboundedSender<(
            Arc<crate::backends::camera::types::CameraFrame>,
            f64,
            f64,
            f64,
        )>,
    ) {
        use crate::backends::virtual_camera::VideoDecoder;

        let decoder = match VideoDecoder::new(&path) {
            Ok(d) => d,
            Err(e) => {
                warn!(?e, "Failed to create preview decoder");
                return;
            }
        };

        // Seek to initial position
        if initial_position > 0.0 {
            decoder.seek(initial_position);
        }

        use crate::constants::virtual_camera as vc_timing;

        let mut paused = initial_paused;
        if paused {
            decoder.set_paused(true);
        }
        let mut last_frame_time = std::time::Instant::now();
        let frame_duration = vc_timing::IMAGE_STREAM_FRAME_DURATION; // ~30fps

        loop {
            // Check for stop signal
            match stop_rx.try_recv() {
                Ok(()) | Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    break;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {}
            }

            // Check for control commands. When we Seek while paused we need
            // to temporarily unpause the decoder to pull one frame at the new
            // position — otherwise the slider preview would stay stuck on the
            // pre-seek frame. Coalesce queued Seeks (e.g. from a slider drag)
            // so we only execute the latest one — intermediate positions
            // would have produced frames the user never sees.
            let mut needs_frame_update = false;
            let mut latest_seek: Option<f64> = None;
            while let Ok(cmd) = control_rx.try_recv() {
                match cmd {
                    VideoPlaybackCommand::Seek(position) => {
                        latest_seek = Some(position);
                        if paused {
                            needs_frame_update = true;
                        }
                    }
                    VideoPlaybackCommand::TogglePause => {
                        paused = !paused;
                        decoder.set_paused(paused);
                    }
                    VideoPlaybackCommand::SetPaused(p) => {
                        paused = p;
                        decoder.set_paused(paused);
                    }
                }
            }
            if let Some(position) = latest_seek {
                decoder.seek(position);
            }

            if paused && !needs_frame_update {
                std::thread::sleep(vc_timing::PAUSE_CHECK_INTERVAL);
                continue;
            }

            // Briefly unpause to pull a single frame at the new seek position.
            if needs_frame_update && paused {
                decoder.set_paused(false);
            }

            // Get next frame
            if let Some(frame) = decoder.next_frame() {
                let frame_arc = Arc::new(frame);

                // Get progress info
                let position = decoder.position().unwrap_or(0.0);
                let duration = decoder.duration().unwrap_or(1.0);
                let progress = if duration > 0.0 {
                    position / duration
                } else {
                    0.0
                };

                // Send frame to UI
                if frame_tx
                    .send((frame_arc, position, duration, progress))
                    .is_err()
                {
                    break; // Receiver dropped
                }

                // Re-pause after a seek-while-paused frame pull.
                if needs_frame_update && paused {
                    decoder.set_paused(true);
                }

                // Rate limiting
                let elapsed = last_frame_time.elapsed();
                if elapsed < frame_duration {
                    std::thread::sleep(frame_duration - elapsed);
                }
                last_frame_time = std::time::Instant::now();
            } else {
                // If we temporarily unpaused to satisfy a seek, re-pause even if
                // no frame came back so we don't accidentally play.
                if needs_frame_update && paused {
                    decoder.set_paused(true);
                }
                if decoder.is_eos() {
                    // Video ended, loop
                    if decoder.restart().is_err() {
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod zoom_tests {
    use super::virtual_output_frame;
    use crate::backends::camera::types::{CameraFrame, FrameData, PixelFormat};
    use crate::backends::camera::zoom::ZoomControl;

    fn source_frame() -> CameraFrame {
        let data: Vec<u8> = (0..8)
            .flat_map(|_| (0..8).flat_map(|x| [x * 30, 0, 0, 255]))
            .collect();
        CameraFrame {
            width: 8,
            height: 8,
            stride: 32,
            format: PixelFormat::RGBA,
            data: FrameData::from_owned_vec(data),
            yuv_planes: None,
            captured_at: std::time::Instant::now(),
            sensor_timestamp_ns: Some(42),
            libcamera_metadata: None,
        }
    }

    #[tokio::test]
    async fn paused_file_source_reads_live_zoom_without_accumulating_or_changing_preview() {
        let source = source_frame();
        let original = source.data.to_vec();
        let zoom = ZoomControl::default();
        let first = virtual_output_frame(&source, &zoom).await.unwrap();
        assert!(std::ptr::eq(first.as_ref(), &source));
        zoom.set_zoom(2.0);
        let second = virtual_output_frame(&source, &zoom).await.unwrap();
        assert_ne!(
            &*second.data,
            original.as_slice(),
            "virtual output must follow zoom changes while reusing a paused source"
        );
        assert_eq!((second.width, second.height, second.stride), (8, 8, 32));
        assert_eq!(second.sensor_timestamp_ns, source.sensor_timestamp_ns);
        assert_eq!(second.captured_at, source.captured_at);
        let repeated = virtual_output_frame(&source, &zoom).await.unwrap();
        assert_eq!(&*repeated.data, &*second.data);
        assert_eq!(
            &*source.data,
            original.as_slice(),
            "preview source must remain unzoomed"
        );
        zoom.set_zoom(1.0);
        let restored = virtual_output_frame(&source, &zoom).await.unwrap();
        assert!(std::ptr::eq(restored.as_ref(), &source));
        assert_eq!(&*restored.data, original.as_slice());
    }

    fn native_frame() -> CameraFrame {
        use crate::backends::camera::types::FrameMetadata;
        use crate::backends::camera::zoom::{ScalerCropBounds, ZoomRect};
        let base = ZoomRect {
            x: 0,
            y: 0,
            width: 800,
            height: 800,
        };
        let crop = ZoomRect {
            x: 200,
            y: 200,
            width: 400,
            height: 400,
        };
        let min = ZoomRect {
            x: 0,
            y: 0,
            width: 100,
            height: 100,
        };
        let applied = ScalerCropBounds::new(min, base, base, base, false)
            .unwrap()
            .applied(Some(crop));
        let mut frame = source_frame();
        frame.libcamera_metadata = Some(FrameMetadata {
            scaler_zoom: applied,
            ..Default::default()
        });
        frame
    }

    #[tokio::test]
    async fn native_applied_zoom_preserves_original_rgba_fast_path() {
        let frame = native_frame();
        let zoom = ZoomControl::default();
        zoom.set_zoom(2.0);
        let output = virtual_output_frame(&frame, &zoom).await.unwrap();
        assert!(
            std::ptr::eq(output.as_ref(), &frame),
            "already baked native zoom must not crop twice"
        );
    }

    #[tokio::test]
    async fn partial_native_zoom_matches_software_residual_at_fixed_dimensions() {
        let native = native_frame();
        let file = source_frame();
        let native_zoom = ZoomControl::default();
        native_zoom.set_zoom(4.0);
        let file_zoom = ZoomControl::default();
        file_zoom.set_zoom(2.0);
        let native_output = virtual_output_frame(&native, &native_zoom).await.unwrap();
        let file_output = virtual_output_frame(&file, &file_zoom).await.unwrap();
        assert_ne!(&*native_output.data, &*native.data);
        assert_eq!(&*native_output.data, &*file_output.data);
        assert_eq!(
            (native_output.width, native_output.height),
            (native.width, native.height)
        );
    }
}
