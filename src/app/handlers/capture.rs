// SPDX-License-Identifier: GPL-3.0-only

//! Capture operations handlers
//!
//! Handles photo capture, video recording, flash, zoom, and timer functionality.

use crate::app::state::{AppModel, CameraMode, Message, RecordingState, TimelapseState};
use crate::backends::camera::types::RecordingFrame;
use crate::backends::camera::v4l2_controls::read_exposure_metadata;
use crate::pipelines::photo::burst_mode::BurstModeConfig;
use crate::pipelines::photo::burst_mode::burst::{
    calculate_adaptive_params, estimate_scene_brightness,
};
use cosmic::Task;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

/// Configuration for appsrc recording pipeline (libcamera backend)
struct AppsrcRecordingConfig {
    width: u32,
    height: u32,
    framerate: u32,
    format: crate::backends::camera::types::CameraFormat,
    output_path: PathBuf,
    sensor_rotation: crate::backends::camera::types::SensorRotation,
    audio_device: Option<String>,
    /// Native sample rate of the selected audio source; 0 if unknown.
    audio_source_rate_hz: u32,
    selected_encoder: Option<crate::media::encoders::video::EncoderInfo>,
    bitrate_kbps: u32,
}

/// Delay in ms before resetting burst mode state after successful capture
const BURST_MODE_SUCCESS_DISPLAY_MS: u64 = 2000;
/// Delay in ms before resetting burst mode state after an error
const BURST_MODE_ERROR_DISPLAY_MS: u64 = 3000;

impl AppModel {
    // =========================================================================
    // Flash Hardware Helpers
    // =========================================================================

    /// Check if the current camera is a back-facing camera (based on libcamera location property)
    pub(crate) fn is_back_camera(&self) -> bool {
        self.available_cameras
            .get(self.current_camera_index)
            .and_then(|c| c.camera_location.as_deref())
            == Some("back")
    }

    /// Check if the current camera supports multistream (simultaneous preview + raw capture)
    pub(crate) fn is_current_camera_multistream(&self) -> bool {
        self.available_cameras
            .get(self.current_camera_index)
            .map(|cam| cam.supports_multistream)
            .unwrap_or(false)
    }

    /// Check if hardware flash should be used (back camera with writable flash LEDs)
    pub(crate) fn use_hardware_flash(&self) -> bool {
        self.is_back_camera() && self.flash.hardware.has_devices()
    }

    /// Turn on hardware flash LEDs (only for back cameras with writable hardware)
    pub(crate) fn turn_on_flash_hardware(&self) {
        if self.use_hardware_flash() {
            info!("Turning on hardware flash LEDs");
            crate::flash::all_on(&self.flash.hardware.devices);
        }
    }

    /// Turn off hardware flash LEDs (safe to call even if no hardware)
    pub(crate) fn turn_off_flash_hardware(&self) {
        if !self.flash.hardware.devices.is_empty() {
            crate::flash::all_off(&self.flash.hardware.devices);
        }
    }

    /// Handle dismissing the flash permission error popup
    pub(crate) fn handle_dismiss_flash_error(&mut self) -> Task<cosmic::Action<Message>> {
        self.flash.error_popup = None;
        Task::none()
    }

    // =========================================================================
    // Capture Operations Handlers
    // =========================================================================

    /// Create a delayed task that sends a message after the specified milliseconds
    pub(crate) fn delay_task(millis: u64, message: Message) -> Task<cosmic::Action<Message>> {
        Task::perform(
            async move {
                tokio::time::sleep(tokio::time::Duration::from_millis(millis)).await;
                message
            },
            cosmic::Action::App,
        )
    }

    /// Generic 16 ms-tick driver shared by all animation tracks. Given the
    /// animation's elapsed time (`None` when not animating), the duration,
    /// the tick message to reschedule, and a callback that clears the
    /// animation when it ends, return either the next tick or `Task::none`.
    /// Keeps the per-track tick arms in `update.rs` to one expression each.
    pub(crate) fn tick_animation_until(
        &mut self,
        duration: std::time::Duration,
        tick: Message,
        elapsed: Option<std::time::Duration>,
        on_done: impl FnOnce(&mut Self),
    ) -> Task<cosmic::Action<Message>> {
        match elapsed {
            Some(e) if e >= duration => {
                on_done(self);
                Task::none()
            }
            Some(_) => Self::delay_task(16, tick),
            None => Task::none(),
        }
    }

    /// Persist the current `Config` to disk on a blocking thread so the UI thread
    /// doesn't stall on filesystem I/O.  Used by settings toggles (fit/fill,
    /// aspect ratio, etc.) where the visible state has already been updated
    /// in-memory and the on-disk write just needs to follow eventually.
    pub(crate) fn persist_config_async(&self) {
        use cosmic::cosmic_config::CosmicConfigEntry;
        let Some(handler) = self.config_handler.clone() else {
            return;
        };
        let config = self.config.clone();
        tokio::task::spawn_blocking(move || {
            if let Err(err) = config.write_entry(&handler) {
                tracing::error!(?err, "Failed to persist config");
            }
        });
    }

    /// Check if burst mode would be triggered based on current scene brightness
    ///
    /// Returns true if Auto mode would use more than 1 frame (actual burst capture)
    /// or if a fixed frame count > 1 is set, AND the user hasn't overridden it.
    pub fn would_use_burst_mode(&self) -> bool {
        use crate::config::BurstModeSetting;

        // User override takes precedence
        if self.hdr_override_disabled {
            return false;
        }

        match self.config.burst_mode_setting {
            BurstModeSetting::Off => false,
            BurstModeSetting::Frames4
            | BurstModeSetting::Frames6
            | BurstModeSetting::Frames8
            | BurstModeSetting::Frames50 => true, // Fixed frame counts always use burst
            BurstModeSetting::Auto => {
                // Use the cached auto-detected frame count (updated every 1 second)
                self.auto_detected_frame_count > 1
            }
        }
    }

    /// Build camera metadata (name, driver, exposure info) for photo encoding.
    fn build_camera_metadata(&self) -> crate::pipelines::photo::CameraMetadata {
        self.available_cameras
            .get(self.current_camera_index)
            .map(|cam| {
                let mut metadata = crate::pipelines::photo::CameraMetadata {
                    camera_name: Some(cam.name.clone()),
                    camera_driver: cam.device_info.as_ref().map(|info| info.driver.clone()),
                    ..Default::default()
                };
                if let Some(device_info) = &cam.device_info {
                    let exposure = read_exposure_metadata(&device_info.path);
                    metadata.exposure_time = exposure.exposure_time;
                    metadata.iso = exposure.iso;
                    metadata.gain = exposure.gain;
                }
                metadata
            })
            .unwrap_or_default()
    }

    /// Capture the current frame as a photo with the selected filter and zoom
    pub(crate) fn capture_photo(&mut self) -> Task<cosmic::Action<Message>> {
        self.capture_photo_with_frame(None)
    }

    /// Capture a photo, optionally using a pre-captured frame (zero-shutter-lag).
    /// Falls back to `self.current_frame` if `zsl_frame` is `None`.
    fn capture_photo_with_frame(
        &mut self,
        zsl_frame: Option<Arc<crate::backends::camera::types::CameraFrame>>,
    ) -> Task<cosmic::Action<Message>> {
        // Use HDR+ burst mode only if it would actually be used (frame_count > 1)
        // This respects auto-detected brightness and user override.
        // Skip when file source is active — burst needs multiple live frames.
        if self.would_use_burst_mode() && !self.current_frame_is_file_source {
            return self.capture_burst_mode_photo();
        }

        // In multistream mode, capture from the raw stream (full sensor resolution)
        // instead of the preview stream (1080p).
        // Skip when file source is active — there is no capture thread to provide raw frames.
        if self.is_current_camera_multistream() && !self.current_frame_is_file_source {
            return self.capture_photo_from_raw_stream();
        }

        let frame_arc = if let Some(frame) = zsl_frame {
            frame
        } else {
            let Some(frame) = &self.current_frame else {
                info!("No frame available to capture");
                return Task::none();
            };
            Arc::clone(frame)
        };

        info!("Capturing photo...");
        self.is_capturing = true;

        let save_dir = crate::app::get_photo_directory(&self.config.save_folder_name);
        let filter_type = self.selected_filter;
        let zoom_level = self.zoom_level;
        let mirror_horizontal = self.should_mirror_captures();

        let geometry_rotation = self.capture_geometry_rotation();
        let media_rotation = self.capture_media_rotation();

        // Calculate crop rectangle:
        // - Cover mode: crop to screen-visible area, then apply aspect ratio
        // - Fit mode: apply aspect ratio directly on the full frame
        let (fw, fh) = if geometry_rotation.swaps_dimensions() {
            (frame_arc.height, frame_arc.width)
        } else {
            (frame_arc.width, frame_arc.height)
        };
        let portrait = self.screen_is_portrait();
        let crop_rect = {
            let (x, y, w, h) = if self.preview_fit_to_view {
                self.photo_aspect_ratio.crop_rect(fw, fh, portrait)
            } else {
                // Map the on-screen frame rect (the same one the canvas
                // crop overlay highlights) back to sensor coords via the
                // preview's Cover scaling. This stays aligned with what
                // the user sees inside the crop bars even when the UI
                // bars are asymmetric (top 47 vs bottom ~174) — a
                // sensor-centered crop would drift off-axis here.
                crate::app::preview_geometry::cover_capture_crop(
                    fw,
                    fh,
                    self.screen_width,
                    self.screen_height,
                    self.settled_bar_insets(),
                    self.photo_aspect_ratio.display_ratio(portrait),
                )
            };
            if x == 0 && y == 0 && w == fw && h == fh {
                None
            } else {
                Some(crate::app::preview_geometry::inverse_oriented_crop(
                    (x, y, w, h),
                    frame_arc.width,
                    frame_arc.height,
                    geometry_rotation,
                ))
            }
        };

        // Get the encoding format from config
        let encoding_format: crate::pipelines::photo::EncodingFormat =
            self.config.photo_output_format.into();

        let camera_metadata = self.build_camera_metadata();

        let save_task = Task::perform(
            async move {
                use crate::pipelines::photo::{
                    EncodingQuality, PhotoPipeline, PostProcessingConfig,
                };
                let config = PostProcessingConfig {
                    filter_type,
                    crop_rect,
                    zoom_level,
                    rotation: media_rotation,
                    mirror_horizontal,
                    ..Default::default()
                };
                let mut pipeline =
                    PhotoPipeline::with_config(config, encoding_format, EncodingQuality::High);
                pipeline.set_camera_metadata(camera_metadata);
                pipeline
                    .capture_and_save(frame_arc, save_dir)
                    .await
                    .map(|p| p.display().to_string())
            },
            |result| cosmic::Action::App(Message::PhotoSaved(result)),
        );

        let animation_task = Self::delay_task(150, Message::ClearCaptureAnimation);
        Task::batch([save_task, animation_task])
    }

    /// Capture a photo from the raw stream (multistream mode)
    ///
    /// Requests a full-resolution raw frame from the dedicated raw stream,
    /// which bypasses the ISP and captures at the sensor's native resolution.
    fn capture_photo_from_raw_stream(&mut self) -> Task<cosmic::Action<Message>> {
        info!("Capturing photo from raw stream (multistream mode)...");
        self.is_capturing = true;

        // Request a still capture from the raw stream
        self.still_capture_requested
            .store(true, std::sync::atomic::Ordering::Release);

        let still_frame = Arc::clone(&self.latest_still_frame);
        let still_frame_notify = Arc::clone(&self.still_frame_notify);
        let save_dir = crate::app::get_photo_directory(&self.config.save_folder_name);
        let filter_type = self.selected_filter;
        let zoom_level = self.zoom_level;
        let mirror_horizontal = self.should_mirror_captures();

        let geometry_rotation = self.capture_geometry_rotation();
        let media_rotation = self.capture_media_rotation();

        // For raw stream capture, use native aspect ratio (no crop from preview)
        // The raw frame may have a different aspect ratio than the preview.
        // Snapshot every value the cover-crop math needs so the spawned task
        // doesn't have to borrow `self`. The cover map (screen dims + UI bar
        // heights + display ratio) lets the saved photo match the on-screen
        // framed rect — see `cover_capture_crop`.
        let photo_aspect_ratio = self.photo_aspect_ratio;
        let preview_fit_to_view = self.preview_fit_to_view;
        let portrait = self.screen_is_portrait();
        let cover_screen_w = self.screen_width;
        let cover_screen_h = self.screen_height;
        let cover_insets = self.settled_bar_insets();
        let cover_target_ratio = self.photo_aspect_ratio.display_ratio(portrait);

        let encoding_format: crate::pipelines::photo::EncodingFormat =
            self.config.photo_output_format.into();

        let camera_metadata = self.build_camera_metadata();

        let save_task = Task::perform(
            async move {
                use crate::pipelines::photo::{
                    EncodingQuality, PhotoPipeline, PostProcessingConfig,
                };

                // Wait for the raw frame with a timeout, blocking on the
                // capture-thread notifier instead of polling.
                let timeout = std::time::Duration::from_secs(2);
                let frame = wait_for_still_frame(&still_frame, &still_frame_notify, timeout)
                    .await
                    .ok_or_else(|| "Timeout waiting for raw frame from still stream".to_string())?;

                info!(
                    width = frame.width,
                    height = frame.height,
                    format = ?frame.format,
                    "Raw frame captured from still stream"
                );

                let (rw, rh) = if geometry_rotation.swaps_dimensions() {
                    (frame.height, frame.width)
                } else {
                    (frame.width, frame.height)
                };
                let crop_rect = {
                    let (x, y, w, h) = if preview_fit_to_view {
                        photo_aspect_ratio.crop_rect(rw, rh, portrait)
                    } else {
                        crate::app::preview_geometry::cover_capture_crop(
                            rw,
                            rh,
                            cover_screen_w,
                            cover_screen_h,
                            cover_insets,
                            cover_target_ratio,
                        )
                    };
                    if x == 0 && y == 0 && w == rw && h == rh {
                        None
                    } else {
                        Some(crate::app::preview_geometry::inverse_oriented_crop(
                            (x, y, w, h),
                            frame.width,
                            frame.height,
                            geometry_rotation,
                        ))
                    }
                };

                let config = PostProcessingConfig {
                    filter_type,
                    crop_rect,
                    zoom_level,
                    rotation: media_rotation,
                    mirror_horizontal,
                    ..Default::default()
                };
                let mut pipeline =
                    PhotoPipeline::with_config(config, encoding_format, EncodingQuality::High);
                pipeline.set_camera_metadata(camera_metadata);
                pipeline
                    .capture_and_save(Arc::new(frame), save_dir)
                    .await
                    .map(|p| p.display().to_string())
            },
            |result| cosmic::Action::App(Message::PhotoSaved(result)),
        );

        let animation_task = Self::delay_task(150, Message::ClearCaptureAnimation);
        Task::batch([save_task, animation_task])
    }

    /// Capture a burst mode photo using multi-frame burst capture
    fn capture_burst_mode_photo(&mut self) -> Task<cosmic::Action<Message>> {
        // Validate state - prevent starting if already active
        if self.burst_mode.is_active() {
            warn!(
                stage = ?self.burst_mode.stage,
                "Cannot start burst mode capture: already active"
            );
            return Task::none();
        }

        // Determine frame count: use config if set, otherwise use cached auto-detected value
        let frame_count = match self.config.burst_mode_setting.frame_count() {
            Some(count) => {
                info!(frame_count = count, "Using configured frame count");
                count
            }
            None => {
                // Use the cached auto-detected frame count (updated every 1 second)
                let auto_count = self.auto_detected_frame_count;
                info!(
                    auto_frame_count = auto_count,
                    "Using cached auto-detected frame count"
                );
                auto_count
            }
        };

        self.is_capturing = true;
        self.burst_mode.start_capture(frame_count);

        // If flash is enabled, turn it on for the entire burst capture duration
        if self.flash.enabled {
            if self.use_hardware_flash() {
                info!("Flash enabled - turning on hardware flash during burst capture");
                self.turn_on_flash_hardware();
            } else {
                info!("Flash enabled - keeping screen flash on during burst capture");
                self.flash.active = true;
            }
        }

        // In multistream mode, capture raw Bayer frames from the raw stream
        // This gives full-resolution raw sensor data (e.g. 3280x2464 SRGGB10_CSI2P)
        // instead of low-res ISP-processed viewfinder frames (e.g. 1436x1080 ABGR8888)
        // HDR+ paper Section 2: "we begin from Bayer raw frames"
        if self.is_current_camera_multistream() {
            // Use the shared still capture Arcs (same mechanism as single photo capture)
            // These are connected to the subscription's NativeLibcameraPipeline
            let still_requested = Arc::clone(&self.still_capture_requested);
            let still_frame = Arc::clone(&self.latest_still_frame);
            let still_frame_notify = Arc::clone(&self.still_frame_notify);
            info!(
                frame_count,
                "Starting burst mode capture - raw frames from raw stream (multistream)"
            );

            return Task::perform(
                async move {
                    let mut frames: Vec<Arc<crate::backends::camera::types::CameraFrame>> =
                        Vec::with_capacity(frame_count);

                    for i in 0..frame_count {
                        // Request a still capture from the raw stream
                        still_requested.store(true, std::sync::atomic::Ordering::Release);

                        // Wait for raw frame with timeout
                        let timeout = std::time::Duration::from_secs(2);
                        let frame = wait_for_still_frame(
                            &still_frame,
                            &still_frame_notify,
                            timeout,
                        )
                        .await
                        .ok_or_else(|| {
                            format!(
                                "Failed to capture raw frame {}/{}: timeout waiting for raw stream",
                                i + 1,
                                frame_count
                            )
                        })?;

                        info!(
                            frame = i + 1,
                            total = frame_count,
                            width = frame.width,
                            height = frame.height,
                            format = ?frame.format,
                            "Raw burst frame captured"
                        );

                        frames.push(Arc::new(frame));
                    }

                    Ok(frames)
                },
                |result| cosmic::Action::App(Message::BurstModeRawFramesCaptured(result)),
            );
        }

        // Fallback: collect viewfinder frames from the stream
        info!(
            frame_count,
            "Starting burst mode capture - collecting frames from stream..."
        );
        // Frames will be collected in handle_camera_frame
        // When enough frames are collected, BurstModeFramesCollected message is sent
        Task::none()
    }

    /// Handle raw burst frames captured via capture_photo() (multistream mode)
    pub(crate) fn handle_burst_mode_raw_frames_captured(
        &mut self,
        result: Result<Vec<Arc<crate::backends::camera::types::CameraFrame>>, String>,
    ) -> Task<cosmic::Action<Message>> {
        match result {
            Ok(frames) => {
                info!(
                    frames = frames.len(),
                    "Raw burst frames captured successfully"
                );
                // Store the raw frames in burst mode state
                self.burst_mode.set_frames(frames);
                // Delegate to the existing processing flow
                self.handle_burst_mode_frames_collected()
            }
            Err(e) => {
                error!("Failed to capture raw burst frames: {}", e);
                self.burst_mode.error();
                self.is_capturing = false;
                // Turn off flash
                self.turn_off_flash_hardware();
                if self.flash.active {
                    self.flash.active = false;
                }
                Task::none()
            }
        }
    }

    /// Handle when all burst mode frames have been collected
    pub(crate) fn handle_burst_mode_frames_collected(&mut self) -> Task<cosmic::Action<Message>> {
        info!(
            frames = self.burst_mode.frames_captured(),
            "Burst mode frames collected, starting processing"
        );

        // Capture blur state up front so the HDR+ processing blur shows the
        // burst still frame with the rotation/mirror of the camera that
        // produced it. Without this, `blur_frame_rotation` keeps whatever
        // value was set the last time a transition fired — or `None` if the
        // user never switched cameras this session — and the burst frame
        // renders rotated wrong on 90°/270° sensors. Mirrors the existing
        // capture in `handle_burst_mode_complete`.
        self.blur_frame_rotation = self.current_frame_rotation;
        self.blur_frame_mirror = self.should_mirror_preview();
        self.blur_frame_zoom = self.current_zoom_level();

        // Turn off flash now that capture is complete (before processing)
        self.turn_off_flash_hardware();
        if self.flash.active {
            info!("Turning off screen flash - burst capture complete");
            self.flash.active = false;
        }

        // Stop the camera stream during HDR+ processing
        // This frees GPU/CPU resources for burst processing
        // The stream will be restarted in handle_burst_mode_complete
        info!("Stopping camera stream for HDR+ processing");
        self.camera_cancel_flag
            .store(true, std::sync::atomic::Ordering::Release);

        // Take the frames from the buffer
        let frames: Vec<Arc<crate::backends::camera::types::CameraFrame>> =
            self.burst_mode.take_frames();

        if frames.len() < 2 {
            error!("Not enough frames collected for burst mode");
            self.burst_mode.error();
            self.is_capturing = false;
            return Task::none();
        }

        let save_dir = crate::app::get_photo_directory(&self.config.save_folder_name);

        // Get encoding format and camera metadata (including exposure info)
        let encoding_format: crate::pipelines::photo::EncodingFormat =
            self.config.photo_output_format.into();

        let camera_metadata = self.build_camera_metadata();

        let geometry_rotation = self.capture_geometry_rotation();
        let media_rotation = self.capture_media_rotation();

        // Calculate crop rectangle based on preview mode and aspect ratio.
        // Cover-mode crop is computed by inverse-mapping the on-screen
        // framed rect back to sensor coords (`cover_capture_crop`) so the
        // saved photo matches what the user sees inside the crop bars.
        let portrait = self.screen_is_portrait();
        let crop_rect = if let Some(frame) = frames.first() {
            let (rw, rh) = if geometry_rotation.swaps_dimensions() {
                (frame.height, frame.width)
            } else {
                (frame.width, frame.height)
            };
            let (x, y, w, h) = if self.preview_fit_to_view {
                self.photo_aspect_ratio.crop_rect(rw, rh, portrait)
            } else {
                crate::app::preview_geometry::cover_capture_crop(
                    rw,
                    rh,
                    self.screen_width,
                    self.screen_height,
                    self.settled_bar_insets(),
                    self.photo_aspect_ratio.display_ratio(portrait),
                )
            };
            if x == 0 && y == 0 && w == rw && h == rh {
                None
            } else {
                Some(crate::app::preview_geometry::inverse_oriented_crop(
                    (x, y, w, h),
                    frame.width,
                    frame.height,
                    geometry_rotation,
                ))
            }
        } else {
            None
        };

        // Create burst mode config with user's settings
        let mut config = BurstModeConfig::default();
        config.crop_rect = crop_rect;
        config.encoding_format = encoding_format;
        config.camera_metadata = camera_metadata;
        config.save_burst_raw_dng = self.config.save_burst_raw;
        config.rotation = media_rotation;
        config.mirror_horizontal = self.should_mirror_captures();

        // Calculate adaptive processing parameters based on scene brightness
        // estimate_scene_brightness assumes RGBA data, so skip for raw Bayer frames
        if let Some(first_frame) = frames.first()
            && !first_frame.format.is_bayer()
        {
            let (_luminance, brightness) = estimate_scene_brightness(first_frame);
            let adaptive_params = calculate_adaptive_params(brightness);
            config.shadow_boost = adaptive_params.shadow_boost;
            config.local_contrast = adaptive_params.local_contrast;
            config.robustness = adaptive_params.robustness;
            debug!(
                ?brightness,
                shadow_boost = adaptive_params.shadow_boost,
                local_contrast = adaptive_params.local_contrast,
                robustness = adaptive_params.robustness,
                "Adaptive burst mode parameters applied"
            );
        }

        // Get selected filter to apply after processing
        let selected_filter = self.selected_filter;

        // Start processing task - BurstModeState handles the communication channels
        let (progress_atomic, result_tx) = self.burst_mode.start_processing_task();

        // Spawn processing on a dedicated OS thread - completely separate from UI/tokio
        // This ensures the event loop stays responsive even during blocking GPU operations
        std::thread::spawn(move || {
            // Create a new tokio runtime for this thread
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create tokio runtime for burst mode processing");

            let result = rt.block_on(async move {
                process_burst_mode_frames_with_atomic(
                    frames,
                    save_dir,
                    config,
                    progress_atomic,
                    selected_filter,
                )
                .await
            });
            let _ = result_tx.send(result);
        });

        // Start a timer to periodically poll progress and check for completion (every 100ms)
        Self::delay_task(100, Message::PollBurstModeProgress)
    }

    /// Poll burst mode progress and check for completion
    pub(crate) fn handle_poll_burst_mode_progress(&mut self) -> Task<cosmic::Action<Message>> {
        // Only poll if we're in processing stage
        if self.burst_mode.stage != crate::app::state::BurstModeStage::Processing {
            self.burst_mode.clear_processing_state();
            return Task::none();
        }

        // Update progress from atomic
        self.burst_mode.poll_progress();

        // Check if result is ready (non-blocking)
        if let Some(result) = self.burst_mode.try_get_result() {
            return self.handle_burst_mode_complete(result);
        }

        // Schedule next poll
        Self::delay_task(100, Message::PollBurstModeProgress)
    }

    /// Handle periodic brightness evaluation tick
    ///
    /// Evaluates scene brightness every 1 second and updates auto_detected_frame_count.
    /// Only runs when in Auto mode and not overridden.
    pub(crate) fn handle_brightness_evaluation_tick(&mut self) -> Task<cosmic::Action<Message>> {
        use crate::config::BurstModeSetting;
        use crate::pipelines::photo::burst_mode::burst::{
            calculate_adaptive_params, estimate_scene_brightness,
        };

        // Only evaluate in Auto mode when not overridden
        if !matches!(self.config.burst_mode_setting, BurstModeSetting::Auto) {
            return Task::none();
        }

        if self.hdr_override_disabled {
            return Task::none();
        }

        // Evaluate brightness from current frame
        if let Some(frame) = &self.current_frame {
            let (_luminance, brightness) = estimate_scene_brightness(frame);
            let params = calculate_adaptive_params(brightness);

            // Only update and log if frame count changed
            if params.frame_count != self.auto_detected_frame_count {
                debug!(
                    old_count = self.auto_detected_frame_count,
                    new_count = params.frame_count,
                    brightness = ?brightness,
                    "Auto frame count updated based on scene brightness"
                );
                self.auto_detected_frame_count = params.frame_count;
            }
        }

        Task::none()
    }

    pub(crate) fn handle_capture(&mut self) -> Task<cosmic::Action<Message>> {
        // Animate the correct button depending on recording state
        if self.recording.is_recording() {
            self.haptic_tap();
            self.animate_photo_btn_scale(0.82);
        } else {
            self.animate_capture_scale(0.82);
        }

        // If quick-record is active, ignore direct Capture messages
        // (the quick-record state machine handles capture)
        if self.quick_record.is_pressed() || self.quick_record.is_recording() {
            return Task::none();
        }

        // If timer countdown is active, abort it
        if self.photo_timer_countdown.is_some() {
            return self.handle_abort_photo_timer();
        }

        // In Photo mode with timer set, start countdown
        if self.mode == CameraMode::Photo
            && self.photo_timer_setting != crate::app::state::PhotoTimerSetting::Off
        {
            let seconds = self.photo_timer_setting.seconds();
            info!(seconds, "Starting photo timer countdown");
            self.photo_timer_countdown = Some(seconds);
            self.photo_timer_tick_start = Some(std::time::Instant::now());
            return Self::delay_task(1000, Message::PhotoTimerTick);
        }

        // Normal capture flow (with flash check)
        if self.mode == CameraMode::Photo && self.flash.enabled && !self.flash.active {
            if self.use_hardware_flash() {
                info!("Flash enabled - turning on hardware flash before capture");
                self.turn_on_flash_hardware();
            } else {
                info!("Flash enabled - showing screen flash before capture");
                self.flash.active = true;
            }
            return Self::delay_task(1000, Message::FlashComplete);
        }
        self.capture_photo()
    }

    pub(crate) fn handle_toggle_flash(&mut self) -> Task<cosmic::Action<Message>> {
        // If trying to enable flash on a back camera with permission errors, show popup
        if !self.flash.enabled && self.is_back_camera() && self.flash.hardware.has_error() {
            warn!("Flash hardware detected but not writable — showing permission error");
            self.flash.error_popup = self.flash.hardware.permission_error.clone();
            return Task::none();
        }

        self.flash.enabled = !self.flash.enabled;
        info!(
            flash_enabled = self.flash.enabled,
            is_back = self.is_back_camera(),
            has_hardware = self.flash.hardware.has_devices(),
            "Flash toggled"
        );
        Task::none()
    }

    pub(crate) fn handle_toggle_burst_mode(&mut self) -> Task<cosmic::Action<Message>> {
        use crate::config::BurstModeSetting;
        use cosmic::cosmic_config::CosmicConfigEntry;

        // Track if config changed (need to save)
        let mut config_changed = false;

        match self.config.burst_mode_setting {
            BurstModeSetting::Off => {
                // Turning ON: go to Auto mode, clear any override
                self.config.burst_mode_setting = BurstModeSetting::Auto;
                self.hdr_override_disabled = false;
                config_changed = true;
            }
            BurstModeSetting::Auto => {
                if self.hdr_override_disabled {
                    // Already overridden - toggle back to enabled
                    self.hdr_override_disabled = false;
                    info!("HDR+ override cleared - auto mode re-enabled");
                } else if self.auto_detected_frame_count > 1 {
                    // HDR+ would be active - set override to disable it
                    self.hdr_override_disabled = true;
                    info!("HDR+ override enabled - auto mode disabled until next toggle");
                } else {
                    // HDR+ already not active (bright scene, 1 frame) - switch to Off
                    self.config.burst_mode_setting = BurstModeSetting::Off;
                    config_changed = true;
                }
            }
            _ => {
                // Fixed frame count modes - toggle override
                if self.hdr_override_disabled {
                    self.hdr_override_disabled = false;
                    info!("HDR+ override cleared for fixed frame count mode");
                } else {
                    self.hdr_override_disabled = true;
                    info!("HDR+ override enabled for fixed frame count mode");
                }
            }
        }

        info!(
            setting = ?self.config.burst_mode_setting,
            override_disabled = self.hdr_override_disabled,
            auto_frame_count = self.auto_detected_frame_count,
            "HDR+ toggled"
        );

        // Save config only if setting changed
        if config_changed
            && let Some(handler) = self.config_handler.as_ref()
            && let Err(err) = self.config.write_entry(handler)
        {
            error!(?err, "Failed to save HDR+ setting");
        }
        Task::none()
    }

    pub(crate) fn handle_set_burst_mode_frame_count(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
        use cosmic::cosmic_config::CosmicConfigEntry;

        // Don't allow changing frame count during active capture
        if self.burst_mode.is_active() {
            warn!("Cannot change frame count during active capture");
            return Task::none();
        }

        use crate::config::BurstModeSetting;
        // Index 0 = Off, 1 = Auto, 2 = 4 frames, 3 = 6 frames, 4 = 8 frames, 5 = 50 frames
        self.config.burst_mode_setting = match index {
            0 => BurstModeSetting::Off,
            1 => BurstModeSetting::Auto,
            2 => BurstModeSetting::Frames4,
            3 => BurstModeSetting::Frames6,
            4 => BurstModeSetting::Frames8,
            5 => BurstModeSetting::Frames50,
            _ => BurstModeSetting::Auto,
        };

        // Reset override when manually changing setting
        self.hdr_override_disabled = false;

        info!(
            setting = ?self.config.burst_mode_setting,
            "HDR+ setting changed (override cleared)"
        );

        if let Some(handler) = self.config_handler.as_ref()
            && let Err(err) = self.config.write_entry(handler)
        {
            error!(?err, "Failed to save burst mode frame count setting");
        }
        Task::none()
    }

    pub(crate) fn handle_cycle_photo_aspect_ratio(&mut self) -> Task<cosmic::Action<Message>> {
        // Get frame dimensions to determine if native matches a defined ratio
        let (width, height) = self
            .current_frame
            .as_ref()
            .map(|f| (f.width, f.height))
            .unwrap_or((0, 0));

        self.photo_aspect_ratio = self.photo_aspect_ratio.next_for_frame(width, height);
        info!(aspect_ratio = ?self.photo_aspect_ratio, "Photo aspect ratio changed");

        self.config.photo_aspect_ratio = self.photo_aspect_ratio;
        self.persist_config_async();

        Task::none()
    }

    pub(crate) fn handle_flash_complete(&mut self) -> Task<cosmic::Action<Message>> {
        info!("Flash complete - capturing photo");
        self.turn_off_flash_hardware();
        self.flash.active = false;
        self.capture_photo()
    }

    pub(crate) fn handle_cycle_photo_timer(&mut self) -> Task<cosmic::Action<Message>> {
        self.photo_timer_setting = self.photo_timer_setting.next();
        info!(
            timer = ?self.photo_timer_setting,
            "Photo timer setting changed"
        );
        Task::none()
    }

    pub(crate) fn handle_photo_timer_tick(&mut self) -> Task<cosmic::Action<Message>> {
        if let Some(remaining) = self.photo_timer_countdown {
            if remaining <= 1 {
                // Countdown complete - capture the photo
                info!("Photo timer countdown complete - capturing");
                self.photo_timer_countdown = None;
                self.photo_timer_tick_start = None;
                self.animate_capture_scale(1.0);
                // Check if flash is enabled
                if self.flash.enabled && !self.flash.active {
                    if self.use_hardware_flash() {
                        info!("Flash enabled - turning on hardware flash before capture");
                        self.turn_on_flash_hardware();
                    } else {
                        info!("Flash enabled - showing screen flash before capture");
                        self.flash.active = true;
                    }
                    return Self::delay_task(1000, Message::FlashComplete);
                }
                return self.capture_photo();
            } else {
                // Continue countdown
                self.photo_timer_countdown = Some(remaining - 1);
                self.photo_timer_tick_start = Some(std::time::Instant::now());
                info!(remaining = remaining - 1, "Photo timer tick");
                return Self::delay_task(1000, Message::PhotoTimerTick);
            }
        }
        Task::none()
    }

    pub(crate) fn handle_abort_photo_timer(&mut self) -> Task<cosmic::Action<Message>> {
        if self.photo_timer_countdown.is_some() {
            info!("Photo timer countdown aborted");
            self.photo_timer_countdown = None;
            self.photo_timer_tick_start = None;
            self.animate_capture_scale(1.0);
        }
        Task::none()
    }

    pub(crate) fn handle_zoom_in(&mut self) -> Task<cosmic::Action<Message>> {
        // Multiplicative zoom: each step is ~2% magnification change,
        // so it feels consistent at any zoom level.
        let new_zoom = (self.zoom_level * 1.02).min(10.0);
        if (new_zoom - self.zoom_level).abs() > 0.001 {
            self.zoom_level = new_zoom;
            self.zoom_animation = None; // step zoom is real-time
            debug!(zoom = self.zoom_level, "Zoom in");
        }
        Task::none()
    }

    pub(crate) fn handle_zoom_out(&mut self) -> Task<cosmic::Action<Message>> {
        // Multiplicative zoom: each step is ~2% magnification change.
        let new_zoom = (self.zoom_level / 1.02).max(1.0);
        if (new_zoom - self.zoom_level).abs() > 0.001 {
            self.zoom_level = new_zoom;
            self.zoom_animation = None; // step zoom is real-time
            debug!(zoom = self.zoom_level, "Zoom out");
        }
        Task::none()
    }

    /// Reset the zoom level to 1× with an ease-out animation. If a tick chain
    /// is already in flight (re-press during the animation), no new chain is
    /// spawned — the existing one picks up the replaced animation, same
    /// pattern as `start_fit_animation`.
    pub(crate) fn handle_reset_zoom(&mut self) -> Task<cosmic::Action<Message>> {
        let from = self.current_zoom_level();
        if (from - 1.0).abs() <= 0.001 {
            return Task::none();
        }
        self.zoom_level = 1.0;
        let was_idle = self.zoom_animation.is_none();
        self.zoom_animation = Some(crate::app::state::ZoomAnimation {
            start: std::time::Instant::now(),
            from,
        });
        debug!(from, "Zoom reset to 1.0 (animated)");
        if was_idle {
            Self::delay_task(16, Message::ZoomAnimationTick)
        } else {
            Task::none()
        }
    }

    pub(crate) fn handle_pinch_zoom(&mut self, level: f32) -> Task<cosmic::Action<Message>> {
        let new_zoom = level.clamp(1.0, 10.0);
        if (new_zoom - self.zoom_level).abs() > 0.001 {
            self.zoom_level = new_zoom;
            self.zoom_animation = None; // pinch is real-time
        }
        Task::none()
    }

    pub(crate) fn handle_photo_saved(
        &mut self,
        result: Result<String, String>,
    ) -> Task<cosmic::Action<Message>> {
        // Always release the capture lock here; the separate
        // `ClearCaptureAnimation` task may be dropped on backpressure or panic,
        // which would otherwise leave the capture button permanently disabled.
        self.is_capturing = false;
        match result {
            Ok(path) => {
                info!(path = %path, "Photo saved successfully");
                self.last_media_path = Some(path.clone());

                return Task::done(cosmic::Action::App(Message::RefreshGalleryThumbnail));
            }
            Err(err) => {
                let expected_dir = crate::app::get_photo_directory(&self.config.save_folder_name);
                error!(
                    error = %err,
                    expected_directory = %expected_dir.display(),
                    "Failed to save photo. This may be caused by: \
                     1) Insufficient disk space, \
                     2) Missing write permissions to the Pictures directory, \
                     3) Flatpak sandbox restrictions (ensure xdg-pictures access is granted), \
                     4) XDG_PICTURES_DIR pointing to an inaccessible location"
                );
            }
        }
        Task::none()
    }

    pub(crate) fn handle_clear_capture_animation(&mut self) -> Task<cosmic::Action<Message>> {
        self.is_capturing = false;
        if self.recording.is_recording() {
            self.animate_photo_btn_scale(1.0);
        } else {
            self.animate_capture_scale(1.0);
        }
        Task::none()
    }

    /// Terminate via the kernel's `_exit` syscall instead of libc `exit`.
    /// `exit` runs atexit handlers — including Mesa's, which destroys the
    /// EGL/GL context — while a libcamera `DebayerEGL` worker is mid-call
    /// against that context, causing a SIGSEGV during teardown. The
    /// application's libcamera pipeline lives in a long-running iced
    /// subscription that we cannot shut down synchronously from here, so
    /// we skip userspace cleanup entirely and let the kernel reap the
    /// process; all worker threads die atomically with no in-flight calls.
    fn shutdown_and_exit(&self, status: i32) -> ! {
        // SAFETY: `_exit` makes no assumptions about program state; it
        // unconditionally terminates the process via the syscall.
        unsafe { libc::_exit(status) }
    }

    /// Handle a window-close request. If a recording or timelapse is in
    /// progress, signal it to stop and defer the process exit until the
    /// completion handler (`handle_recording_stopped` /
    /// `handle_timelapse_assembly_complete`) sees `pending_close == true`
    /// and exits. Otherwise exit immediately.
    pub(crate) fn handle_window_close(&mut self) -> Task<cosmic::Action<Message>> {
        // Already closing — second click is a no-op.
        if self.pending_close {
            return Task::none();
        }

        // Turn off hardware flash LEDs regardless of the exit path so we
        // never leave them on after the app window goes away.
        if self.flash.enabled {
            self.flash.enabled = false;
        }
        self.turn_off_flash_hardware();

        if self.recording.is_recording() {
            info!("Window close requested during recording — signalling EOS first");
            self.pending_close = true;
            if let Some(sender) = self.recording.take_stop_sender() {
                let _ = sender.send(());
            }
            // Leave RecordingState::Recording in place so the UI still
            // shows the timer until the recorder task emits
            // RecordingStopped.
            return Task::none();
        }

        if self.timelapse.is_running() {
            info!("Window close requested during timelapse — finalising first");
            self.pending_close = true;
            // Transition to Finalising; the frame sender is dropped,
            // closing the channel so the encoder task drains and emits
            // TimelapseAssemblyComplete.
            return self.stop_timelapse();
        }

        if self.timelapse.is_finalising() {
            // Already finalising — just defer the exit.
            info!("Window close requested while timelapse is finalising — waiting");
            self.pending_close = true;
            return Task::none();
        }

        info!("Window close — exiting");
        self.shutdown_and_exit(0);
    }

    pub(crate) fn handle_toggle_recording(&mut self) -> Task<cosmic::Action<Message>> {
        self.haptic_tap();
        if self.recording.is_recording() {
            // Stopping: animate release (scale back up)
            self.animate_capture_scale(1.0);
            // Turn off torch when stopping recording
            if self.flash.enabled {
                self.turn_off_flash_hardware();
            }
            if let Some(sender) = self.recording.take_stop_sender() {
                info!("Sending stop signal to recorder");
                let _ = sender.send(());
            }
            self.recording = RecordingState::Idle;
            self.update_idle_inhibit();
        } else {
            // Starting: validate before animating
            if self
                .available_cameras
                .get(self.current_camera_index)
                .is_none()
            {
                error!("No camera available for recording");
                return Task::none();
            }
            if self.active_format.is_none() {
                error!("No active format for recording");
                return Task::none();
            }
            // Animate to recording size (after guards pass)
            self.animate_capture_scale(0.82);
            return Task::done(cosmic::Action::App(Message::StartRecordingAfterDelay));
        }
        Task::none()
    }

    pub(crate) fn handle_recording_started(
        &mut self,
        path: String,
    ) -> Task<cosmic::Action<Message>> {
        info!(path = %path, "Recording started successfully");
        self.update_idle_inhibit();
        self.sync_audio_probe();
        Self::delay_task(1000, Message::UpdateRecordingDuration)
    }

    pub(crate) fn handle_recording_stopped(
        &mut self,
        session: u64,
        result: Result<String, String>,
    ) -> Task<cosmic::Action<Message>> {
        // If a new recording has already taken over `self.recording`, this
        // stop event is from a previous session and must NOT touch the
        // current-session state. Resetting `self.recording` here would
        // drop the new session's `stop_sender`, causing its async task to
        // wake from `stop_rx.await` with an error and self-terminate via
        // `recorder.stop()`. We still log the result and update
        // `last_media_path` / the gallery for the just-finalized file.
        let is_current = self.recording.session() == Some(session);

        if is_current {
            self.recording = RecordingState::Idle;
            self.sync_audio_probe();
            self.quick_record = crate::app::state::QuickRecordState::Idle;
            self.update_idle_inhibit();
            // Turn off torch when recording ends
            self.turn_off_flash_hardware();
        }

        if is_current && self.pending_close {
            match &result {
                Ok(path) => info!(path = %path, "Recording finalized — exiting"),
                Err(err) => error!(error = %err, "Recording finalize failed — exiting anyway"),
            }
            self.shutdown_and_exit(0);
        }

        match result {
            Ok(path) => {
                info!(session, path = %path, "Recording saved successfully");
                self.last_media_path = Some(path.clone());
                Task::done(cosmic::Action::App(Message::RefreshGalleryThumbnail))
            }
            Err(err) => {
                let expected_dir = crate::app::get_photo_directory(&self.config.save_folder_name);
                error!(
                    session,
                    error = %err,
                    expected_directory = %expected_dir.display(),
                    "Failed to save recording. This may be caused by: \
                     1) Insufficient disk space, \
                     2) Missing write permissions to the Pictures directory, \
                     3) Flatpak sandbox restrictions (ensure xdg-pictures access is granted), \
                     4) XDG_PICTURES_DIR pointing to an inaccessible location"
                );
                Task::none()
            }
        }
    }

    pub(crate) fn handle_update_recording_duration(&mut self) -> Task<cosmic::Action<Message>> {
        if self.recording.is_recording() {
            return Self::delay_task(1000, Message::UpdateRecordingDuration);
        }
        Task::none()
    }

    /// Photo mode: capture frame on press, start 300ms timer for quick-record.
    pub(crate) fn handle_capture_button_pressed(&mut self) -> Task<cosmic::Action<Message>> {
        self.haptic_tap();
        use crate::app::state::QuickRecordState;

        // Only handle in Photo mode when idle
        if self.mode != CameraMode::Photo
            || self.recording.is_recording()
            || self.burst_mode.is_active()
            || self.quick_record.is_recording()
        {
            return Task::none();
        }

        // If photo timer is counting down, abort it
        if self.photo_timer_countdown.is_some() {
            return self.handle_abort_photo_timer();
        }

        // Capture current frame for zero-shutter-lag photo
        let captured_frame = self.current_frame.clone();

        self.quick_record = QuickRecordState::Pressed {
            press_time: std::time::Instant::now(),
            captured_frame,
        };
        self.animate_capture_scale(0.82);

        // Schedule 300ms threshold check
        Self::delay_task(300, Message::QuickRecordThreshold)
    }

    /// Photo mode: finger lifted — either process photo or stop recording.
    pub(crate) fn handle_capture_button_released(&mut self) -> Task<cosmic::Action<Message>> {
        use crate::app::state::QuickRecordState;

        match std::mem::take(&mut self.quick_record) {
            QuickRecordState::Pressed { captured_frame, .. } => {
                // Short tap: route through timer/flash logic before capturing
                self.quick_record = QuickRecordState::Idle;

                // If timer countdown is active, abort it
                if self.photo_timer_countdown.is_some() {
                    self.animate_capture_scale(1.0);
                    return self.handle_abort_photo_timer();
                }

                // In Photo mode with timer set, start countdown
                // Keep button in pressed state until capture completes
                if self.mode == CameraMode::Photo
                    && self.photo_timer_setting != crate::app::state::PhotoTimerSetting::Off
                {
                    let seconds = self.photo_timer_setting.seconds();
                    info!(seconds, "Starting photo timer countdown");
                    self.photo_timer_countdown = Some(seconds);
                    self.photo_timer_tick_start = Some(std::time::Instant::now());
                    return Self::delay_task(1000, Message::PhotoTimerTick);
                }

                // Screen flash (front camera) or hardware flash (back camera)
                self.animate_capture_scale(1.0);
                if self.flash.enabled && !self.flash.active {
                    if self.use_hardware_flash() {
                        info!("Flash enabled - turning on hardware flash before capture");
                        self.turn_on_flash_hardware();
                    } else {
                        info!("Flash enabled - showing screen flash before capture");
                        self.flash.active = true;
                    }
                    return Self::delay_task(1000, Message::FlashComplete);
                }

                // No timer or flash — use the zero-shutter-lag frame
                self.capture_photo_with_frame(captured_frame)
            }
            QuickRecordState::Recording => {
                self.animate_capture_scale(1.0);
                // Long press ended: stop recording
                self.quick_record = QuickRecordState::Idle;

                // Stop recording (same as toggle off)
                if let Some(sender) = self.recording.take_stop_sender() {
                    let _ = sender.send(());
                }
                self.recording = RecordingState::Idle;
                self.update_idle_inhibit();
                Task::none()
            }
            QuickRecordState::Idle => {
                self.animate_capture_scale(1.0);
                Task::none()
            }
        }
    }

    /// 300ms elapsed — if still pressed, start recording.
    pub(crate) fn handle_quick_record_threshold(&mut self) -> Task<cosmic::Action<Message>> {
        use crate::app::state::QuickRecordState;
        // Only act if still in Pressed state (finger hasn't lifted)
        if !self.quick_record.is_pressed() {
            return Task::none();
        }
        self.haptic_tap();
        // Animate to recording scale
        self.animate_capture_scale(0.82);

        // Discard the captured photo frame, start recording
        self.quick_record = QuickRecordState::Recording;
        self.start_quick_recording()
    }

    /// Start quick-recording using the existing appsrc infrastructure.
    /// Same as normal video recording but initiated from Photo mode.
    fn start_quick_recording(&mut self) -> Task<cosmic::Action<Message>> {
        if self
            .available_cameras
            .get(self.current_camera_index)
            .is_none()
        {
            self.quick_record = crate::app::state::QuickRecordState::Idle;
            self.animate_capture_scale(1.0);
            return Task::none();
        }
        if self.active_format.is_none() {
            self.quick_record = crate::app::state::QuickRecordState::Idle;
            self.animate_capture_scale(1.0);
            return Task::none();
        }

        let format = self.active_format.as_ref().unwrap();

        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S_%3f");
        let filename = format!("VID_{}.mp4", timestamp);
        let save_dir = crate::app::get_video_directory(&self.config.save_folder_name);
        let output_path = save_dir.join(&filename);

        info!(
            output = %output_path.display(),
            "Starting quick-record (long-press in Photo mode)"
        );

        // Snapshot the physical device orientation at recording start so the
        // resulting video is upright and remains stable for the whole session.
        let sensor_rotation = self.capture_media_rotation();
        let framerate = format.framerate.map(|f| f.as_int()).unwrap_or(30);

        // Use viewfinder frame dimensions (not raw format dimensions)
        let (appsrc_width, appsrc_height) = self
            .current_frame
            .as_ref()
            .map(|f| (f.width, f.height))
            .unwrap_or((format.width, format.height));

        let selected_audio_device = self
            .available_audio_devices
            .get(self.current_audio_device_index);
        let audio_device = if self.config.record_audio {
            selected_audio_device.map(|dev| dev.node_name.clone())
        } else {
            None
        };
        // Gate symmetrically with `audio_device`: when the user disabled audio,
        // both `audio_device` and `audio_source_rate_hz` are zero/None so the
        // downstream capsfilter can't accidentally pin a rate that doesn't
        // match whatever PA would otherwise produce.
        let audio_source_rate_hz = if self.config.record_audio {
            selected_audio_device.map(|d| d.sample_rate).unwrap_or(0)
        } else {
            0
        };

        let selected_encoder = self
            .available_video_encoders
            .get(self.current_video_encoder_index)
            .cloned();

        let appsrc_bitrate = self
            .config
            .bitrate_preset
            .bitrate_kbps(appsrc_width, appsrc_height);

        self.start_appsrc_recording(AppsrcRecordingConfig {
            width: appsrc_width,
            height: appsrc_height,
            framerate,
            format: format.clone(),
            output_path,
            sensor_rotation,
            audio_device,
            audio_source_rate_hz,
            selected_encoder,
            bitrate_kbps: appsrc_bitrate,
        })
    }

    pub(crate) fn handle_start_recording_after_delay(&mut self) -> Task<cosmic::Action<Message>> {
        let Some(camera) = self.available_cameras.get(self.current_camera_index) else {
            error!("Camera disappeared");
            self.recording = RecordingState::Idle;
            return Task::none();
        };

        let Some(format) = &self.active_format else {
            error!("Format disappeared");
            self.recording = RecordingState::Idle;
            return Task::none();
        };

        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S_%3f");
        let filename = format!("VID_{}.mp4", timestamp);
        let save_dir = crate::app::get_video_directory(&self.config.save_folder_name);
        let output_path = save_dir.join(&filename);

        info!(
            device = %camera.path,
            width = format.width,
            height = format.height,
            fps = ?format.framerate,
            output = %output_path.display(),
            "Starting video recording"
        );

        // Turn on hardware flash as torch during video recording
        if self.flash.enabled && self.use_hardware_flash() {
            info!("Flash enabled - turning on hardware flash torch for video recording");
            self.turn_on_flash_hardware();
        }

        // Snapshot the physical device orientation at recording start so the
        // resulting video is upright and remains stable for the whole session.
        let sensor_rotation = self.capture_media_rotation();
        let width = format.width;
        let height = format.height;
        let framerate = format.framerate.map(|f| f.as_int()).unwrap_or(30);

        // Only get audio device if audio recording is enabled in settings
        let selected_audio_device = self
            .available_audio_devices
            .get(self.current_audio_device_index);
        let audio_device = if self.config.record_audio {
            selected_audio_device.map(|dev| dev.node_name.clone())
        } else {
            None
        };
        // Gate symmetrically with `audio_device`: when the user disabled audio,
        // both `audio_device` and `audio_source_rate_hz` are zero/None so the
        // downstream capsfilter can't accidentally pin a rate that doesn't
        // match whatever PA would otherwise produce.
        let audio_source_rate_hz = if self.config.record_audio {
            selected_audio_device.map(|d| d.sample_rate).unwrap_or(0)
        } else {
            0
        };

        let selected_encoder = self
            .available_video_encoders
            .get(self.current_video_encoder_index)
            .cloned();

        // For the appsrc pipeline, use the actual viewfinder frame dimensions
        // (not active_format, which may be the raw/Bayer stream resolution).
        let (appsrc_width, appsrc_height) = self
            .current_frame
            .as_ref()
            .map(|f| (f.width, f.height))
            .unwrap_or((width, height));
        let appsrc_bitrate = self
            .config
            .bitrate_preset
            .bitrate_kbps(appsrc_width, appsrc_height);
        self.start_appsrc_recording(AppsrcRecordingConfig {
            width: appsrc_width,
            height: appsrc_height,
            framerate,
            format: format.clone(),
            output_path,
            sensor_rotation,
            audio_device,
            audio_source_rate_hz,
            selected_encoder,
            bitrate_kbps: appsrc_bitrate,
        })
    }

    /// Start recording using the appsrc pipeline (libcamera backend).
    ///
    /// Frames from the native capture thread are forwarded via an mpsc channel
    /// into a GStreamer appsrc encoding pipeline. The preview continues
    /// uninterrupted since we reuse the same frames.
    fn start_appsrc_recording(
        &mut self,
        config: AppsrcRecordingConfig,
    ) -> Task<cosmic::Action<Message>> {
        use crate::backends::camera::types::PixelFormat;

        let AppsrcRecordingConfig {
            width,
            height,
            framerate,
            format,
            output_path,
            sensor_rotation,
            audio_device,
            audio_source_rate_hz,
            selected_encoder,
            bitrate_kbps,
        } = config;
        let mirror_horizontal = self.should_mirror_captures();

        // Determine pixel format for the appsrc pipeline
        let pixel_format = self
            .current_frame
            .as_ref()
            .map(|f| f.format)
            .unwrap_or_else(|| {
                // Fallback: parse from format string
                PixelFormat::from_gst_format(&format.pixel_format).unwrap_or(PixelFormat::I420)
            });

        // Check if we can use the VA-API JPEG zero-copy path:
        // - Camera outputs MJPEG
        // - A VA-API JPEG decoder is available that handles this camera's
        //   chroma subsampling (e.g. 4:2:0 → I420, 4:2:2 → Y42B)
        // - No sensor rotation needed (GPU JPEG decode → encoder is direct)
        let is_mjpeg = format.pixel_format == "MJPEG" || format.pixel_format.contains("MJPG");
        let decoded_yuv_format = self
            .current_frame
            .as_ref()
            .map(|f| f.gst_format_string())
            .unwrap_or("I420");
        let va_jpeg_dec = if is_mjpeg {
            crate::media::encoders::detection::detect_va_jpeg_decoder(decoded_yuv_format)
        } else {
            None
        };
        let use_jpeg_pipeline = is_mjpeg
            && va_jpeg_dec.is_some()
            && sensor_rotation == crate::backends::camera::types::SensorRotation::None;

        if use_jpeg_pipeline {
            info!(
                va_jpeg_dec = ?va_jpeg_dec,
                "Using VA-API JPEG zero-copy recording pipeline"
            );
        } else {
            info!(
                pixel_format = ?pixel_format,
                is_mjpeg,
                va_jpeg_dec = ?va_jpeg_dec,
                rotation = %sensor_rotation,
                "Using appsrc recording pipeline (libcamera backend)"
            );
        }

        let channel_size = if use_jpeg_pipeline { 30 } else { 15 };
        let (frame_tx, frame_rx) = tokio::sync::mpsc::channel(channel_size);
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();

        let path_for_message = output_path.display().to_string();

        // When a file source is active (--preview-source), there is no capture
        // thread to forward frames. Spawn a task that pushes the static frame
        // into the recording channel at the configured framerate.
        if self.current_frame_is_file_source {
            if let Some(ref frame) = self.current_frame {
                let frame = Arc::new(frame.as_ref().clone());
                let tx = frame_tx;
                let interval = std::time::Duration::from_millis(1000 / framerate.max(1) as u64);
                tokio::spawn(async move {
                    let mut ticker = tokio::time::interval(interval);
                    loop {
                        ticker.tick().await;
                        if tx
                            .send(RecordingFrame::Decoded(Arc::clone(&frame)))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                });
                // frame_tx moved into the task; create a dummy sender for the
                // backend manager so the rest of the code doesn't panic.
                let (dummy_tx, _dummy_rx) = tokio::sync::mpsc::channel(1);
                if let Some(ref manager) = self.backend_manager {
                    manager.set_recording_sender(Some(dummy_tx));
                    manager.set_jpeg_recording_mode(false);
                }
            }
        } else {
            // Direct path: capture thread → appsrc (bypasses UI thread entirely)
            if let Some(ref manager) = self.backend_manager {
                manager.set_recording_sender(Some(frame_tx));
                manager.set_jpeg_recording_mode(use_jpeg_pipeline);
            }
        }

        // Create audio levels handle upfront so the UI can read it immediately.
        // The recorder is created in the async task to avoid blocking the UI thread.
        let audio_levels: crate::pipelines::video::SharedAudioLevels = Default::default();
        self.recording_session_counter += 1;
        let session = self.recording_session_counter;
        self.recording = RecordingState::start(
            session,
            path_for_message.clone(),
            stop_tx,
            Some(audio_levels.clone()),
        );

        let backend_manager = self.backend_manager.clone();
        let va_jpeg_dec_name = va_jpeg_dec.map(|s| s.to_string());
        let live_filter = self.recording_filter_code.clone();
        let record_audio = self.config.record_audio;

        let recording_task = Task::perform(
            async move {
                // Capture runtime handle — spawn_blocking doesn't propagate the
                // tokio context, but the appsrc pusher needs it for tokio::spawn.
                let rt_handle = tokio::runtime::Handle::current();

                // Clone for use inside spawn_blocking's fallback path
                let backend_manager_inner = backend_manager.clone();

                let recorder = tokio::task::spawn_blocking(move || {
                    // Enter the runtime context so tokio::spawn works inside
                    // new_from_appsrc (used by the appsrc pusher task).
                    let _guard = rt_handle.enter();

                    use crate::pipelines::video::{
                        AppsrcRecorderConfig, AudioChannels, AudioQuality, EncoderConfig,
                        RecorderConfig, VideoQuality, VideoRecorder,
                    };

                    // Use Low quality for appsrc path (x264 veryfast preset)
                    // to stay real-time on ARM devices.
                    let config = EncoderConfig {
                        video_quality: VideoQuality::Low,
                        audio_quality: AudioQuality::High,
                        audio_channels: AudioChannels::Mono,
                        width,
                        height,
                        bitrate_override_kbps: Some(bitrate_kbps),
                    };

                    let make_appsrc_config =
                        |audio_levels: crate::pipelines::video::SharedAudioLevels| {
                            AppsrcRecorderConfig {
                                base: RecorderConfig {
                                    width,
                                    height,
                                    framerate,
                                    output_path: output_path.clone(),
                                    encoder_config: config.clone(),
                                    // Honor the user's `record_audio` preference directly. With
                                    // an empty device list (e.g. PipeWire not enumerating, fallback
                                    // also empty), `audio_device` is None — let `pulsesrc` open the
                                    // PA default source rather than dropping the audio branch.
                                    enable_audio: record_audio,
                                    audio_device: audio_device.as_deref(),
                                    audio_source_rate_hz,
                                    encoder_info: selected_encoder.as_ref(),
                                    rotation: sensor_rotation,
                                    mirror_horizontal,
                                    audio_levels,
                                },
                                pixel_format,
                                live_filter_code: live_filter.clone(),
                            }
                        };

                    // Try VA-API JPEG pipeline first, fall back to legacy
                    let recorder = if use_jpeg_pipeline {
                        let dec = va_jpeg_dec_name.as_deref().unwrap();
                        match VideoRecorder::new_from_appsrc_jpeg(
                            make_appsrc_config(audio_levels.clone()),
                            dec,
                            frame_rx,
                        ) {
                            Ok(r) => r,
                            Err(e) => {
                                warn!(
                                    error = %e,
                                    "VA-API JPEG pipeline failed, falling back to legacy"
                                );
                                // Clear jpeg mode so capture thread sends decoded frames
                                if let Some(ref manager) = backend_manager_inner {
                                    manager.set_jpeg_recording_mode(false);
                                }
                                // Need a new channel since the old rx was consumed
                                // This is a rare error path — the recording will start
                                // without any buffered frames, which is acceptable.
                                let (fallback_tx, fallback_rx) = tokio::sync::mpsc::channel(15);
                                if let Some(ref manager) = backend_manager_inner {
                                    manager.set_recording_sender(Some(fallback_tx));
                                }
                                VideoRecorder::new_from_appsrc(
                                    make_appsrc_config(audio_levels),
                                    fallback_rx,
                                )?
                            }
                        }
                    } else {
                        VideoRecorder::new_from_appsrc(make_appsrc_config(audio_levels), frame_rx)?
                    };

                    recorder.start()?;
                    let path = output_path.display().to_string();
                    Ok::<_, String>((recorder, path))
                })
                .await
                .unwrap_or_else(|e| Err(format!("Task join error: {}", e)))?;

                let (recorder, path) = recorder;

                // Wait for stop signal
                let _ = stop_rx.await;

                // Clear the direct recording sender — drops the Sender, closing
                // the channel so the appsrc pusher sees None and sends EOS.
                if let Some(ref manager) = backend_manager {
                    manager.set_recording_sender(None);
                    manager.set_jpeg_recording_mode(false);
                }

                // Give a brief moment for EOS to propagate before stopping the pipeline.
                tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

                tokio::task::spawn_blocking(move || {
                    recorder.stop().map(|_| path).map_err(|e| e.to_string())
                })
                .await
                .unwrap_or_else(|e| Err(format!("Task join error: {}", e)))
            },
            move |result| cosmic::Action::App(Message::RecordingStopped { session, result }),
        );

        let start_signal = Task::done(cosmic::Action::App(Message::RecordingStarted(
            path_for_message,
        )));

        Task::batch([start_signal, recording_task])
    }

    /// Handle burst mode progress update
    pub(crate) fn handle_burst_mode_progress(
        &mut self,
        progress: f32,
    ) -> Task<cosmic::Action<Message>> {
        self.burst_mode.processing_progress = progress;

        debug!(
            progress,
            stage = ?self.burst_mode.stage,
            "Burst mode progress"
        );
        Task::none()
    }

    /// Handle burst mode capture complete
    pub(crate) fn handle_burst_mode_complete(
        &mut self,
        result: Result<String, String>,
    ) -> Task<cosmic::Action<Message>> {
        self.is_capturing = false;

        // Start a short blur transition (200ms) after stream restarts
        // This keeps the last frame blurred until new frames arrive, then fades out smoothly
        // Don't disable UI since capture is complete
        self.blur_frame_rotation = self.current_frame_rotation;
        self.blur_frame_mirror = self.should_mirror_preview();
        self.blur_frame_zoom = self.current_zoom_level();
        let _ = self.transition_state.start_with_duration(200, false);

        // Restart the camera stream after HDR+ processing
        // The stream was stopped when processing began to free GPU resources.
        // Increment the restart counter to change the subscription ID and trigger restart.
        // Create a new cancel flag so the new subscription isn't immediately cancelled.
        info!("Restarting camera stream after HDR+ processing");
        self.camera_stream_restart_counter = self.camera_stream_restart_counter.wrapping_add(1);
        self.camera_cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

        match result {
            Ok(path) => {
                info!(path, "Burst mode capture complete");
                self.burst_mode.complete();

                // Reset burst mode state after a short delay
                let reset_task =
                    Self::delay_task(BURST_MODE_SUCCESS_DISPLAY_MS, Message::ResetBurstModeState);

                // Trigger the same photo saved flow
                let saved_task = Task::done(cosmic::Action::App(Message::PhotoSaved(Ok(path))));

                Task::batch([saved_task, reset_task])
            }
            Err(e) => {
                error!(error = %e, "Burst mode capture failed");
                self.burst_mode.error();

                // Reset after showing error
                Self::delay_task(BURST_MODE_ERROR_DISPLAY_MS, Message::ResetBurstModeState)
            }
        }
    }

    // =========================================================================
    // Idle Inhibit
    // =========================================================================

    /// Update the system idle/suspend inhibit based on current activity.
    ///
    /// Inhibit is active whenever the app is in a usable state — cameras have
    /// finished initializing, or any active session (recording / virtual
    /// camera streaming / timelapse) is in progress. The session triggers
    /// also act as fallbacks if the camera list becomes empty mid-operation
    /// (e.g. hotplug-removed during recording).
    ///
    /// Call this whenever recording / streaming / timelapse state changes,
    /// or when the camera list transitions to/from empty.
    ///
    /// Uses the XDG Inhibit portal (org.freedesktop.portal.Inhibit) to block
    /// idle lock + suspend while the camera is active. Issue #365.
    ///
    /// The portal is used because it needs no finish-args: portals are always
    /// reachable in the Flatpak sandbox, unlike direct D-Bus calls to logind
    /// (system bus) or the screensaver. Desktops that ship no Inhibit backend
    /// (COSMIC: pop-os/xdg-desktop-portal-cosmic#342) return an error here and
    /// simply won't inhibit; we keep no direct-D-Bus fallback on purpose.
    pub(crate) fn update_idle_inhibit(&mut self) {
        let should_inhibit = !self.available_cameras.is_empty()
            || self.recording.is_recording()
            || self.virtual_camera.is_streaming()
            || self.timelapse.is_active();

        let is_inhibited = self.idle_inhibit.is_some();

        if should_inhibit && !is_inhibited {
            // XDG Inhibit portal (prevents idle lock + suspend, no finish-args).
            // Connection must stay alive: the inhibition ends when it drops.
            match portal_inhibit() {
                Ok(guard) => {
                    info!(
                        handle = guard.handle.as_str(),
                        "idle+suspend inhibit active"
                    );
                    self.idle_inhibit = Some(guard);
                }
                Err(e) => warn!(error = %e, "idle+suspend inhibit failed"),
            }
        } else if !should_inhibit && is_inhibited {
            // Release the inhibit (Drop calls Request.Close on the handle)
            if self.idle_inhibit.take().is_some() {
                info!("idle+suspend inhibit released");
            }
        }
    }

    // =========================================================================
    // Timelapse Handlers
    // =========================================================================

    pub(crate) fn handle_toggle_timelapse(&mut self) -> Task<cosmic::Action<Message>> {
        self.haptic_tap();
        if self.timelapse.is_running() {
            self.animate_capture_scale(1.0);
        } else {
            self.animate_capture_scale(0.82);
        }
        if self.timelapse.is_running() {
            info!("Stopping timelapse");
            return self.stop_timelapse();
        }

        if self.timelapse.is_finalising() {
            return Task::none();
        }

        let interval_ms = self.config.timelapse_interval.millis();
        info!(interval_ms, "Starting timelapse");

        // Create channel for sending frames to the encoder
        let (frame_tx, frame_rx) = tokio::sync::mpsc::unbounded_channel();

        self.timelapse = TimelapseState::Running {
            start_time: std::time::Instant::now(),
            shots_taken: 0,
            interval_ms,
            frame_sender: frame_tx,
        };

        // Build output path
        let folder_name = self.config.save_folder_name.clone();
        let encoder_info = self
            .available_video_encoders
            .get(self.current_video_encoder_index)
            .cloned();
        let (w, h) = self
            .active_format
            .as_ref()
            .map(|f| (f.width, f.height))
            .unwrap_or((1920, 1080));
        let bitrate_kbps = Some(self.config.bitrate_preset.bitrate_kbps(w, h));
        let live_filter_code = Arc::clone(&self.recording_filter_code);
        let rotation = self.capture_media_rotation();
        let mirror_horizontal = self.should_mirror_captures();

        // Spawn the encoder task — it runs until the channel is closed
        let encoder_task = Task::perform(
            async move {
                let video_dir = crate::app::get_video_directory(&folder_name);
                if let Err(e) = std::fs::create_dir_all(&video_dir) {
                    return Err(format!("Failed to create video directory: {e}"));
                }

                let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S_%3f");
                let output_path = video_dir.join(format!("timelapse_{timestamp}.mp4"));

                crate::pipelines::video::timelapse::run_timelapse_encoder(
                    frame_rx,
                    output_path,
                    encoder_info,
                    bitrate_kbps,
                    live_filter_code,
                    rotation,
                    mirror_horizontal,
                )
                .await
            },
            |result| cosmic::Action::App(Message::TimelapseAssemblyComplete(result)),
        );

        self.update_idle_inhibit();

        // Send first frame immediately, then schedule next tick
        self.timelapse_send_current_frame();

        let tick_task = Self::delay_task(interval_ms, Message::TimelapseTick);

        Task::batch([encoder_task, tick_task])
    }

    /// Send the current preview frame to the timelapse encoder.
    fn timelapse_send_current_frame(&mut self) {
        if let Some(frame) = &self.current_frame
            && self.timelapse.send_frame(Arc::clone(frame))
        {
            self.timelapse.increment_shots();
        }
    }

    /// Stop timelapse capture. Dropping the sender closes the channel,
    /// which causes the encoder task to finalise the video and send
    /// `TimelapseAssemblyComplete`.
    fn stop_timelapse(&mut self) -> Task<cosmic::Action<Message>> {
        let shots = self.timelapse.shots_taken();
        let start_time = match &self.timelapse {
            TimelapseState::Running { start_time, .. } => *start_time,
            _ => std::time::Instant::now(),
        };

        // Transition to Finalising — the sender is dropped, closing the channel
        self.timelapse = TimelapseState::Finalising {
            start_time,
            shots_taken: shots,
        };

        Task::none()
    }

    pub(crate) fn handle_timelapse_assembly_complete(
        &mut self,
        result: Result<String, String>,
    ) -> Task<cosmic::Action<Message>> {
        self.timelapse = TimelapseState::Idle;
        self.update_idle_inhibit();

        if self.pending_close {
            match &result {
                Ok(path) => info!(path = %path, "Timelapse finalized — exiting"),
                Err(err) => error!(error = %err, "Timelapse finalize failed — exiting anyway"),
            }
            self.shutdown_and_exit(0);
        }

        match result {
            Ok(path) => {
                info!(path = %path, "Timelapse video saved");
                self.last_media_path = Some(path);
                Task::done(cosmic::Action::App(Message::RefreshGalleryThumbnail))
            }
            Err(e) => {
                error!(error = %e, "Timelapse video encoding failed");
                Task::none()
            }
        }
    }

    pub(crate) fn handle_timelapse_tick(&mut self) -> Task<cosmic::Action<Message>> {
        if !self.timelapse.is_running() {
            return Task::none();
        }

        let interval_ms = match &self.timelapse {
            TimelapseState::Running { interval_ms, .. } => *interval_ms,
            _ => return Task::none(),
        };

        // Send the current frame to the encoder
        self.timelapse_send_current_frame();

        // Schedule next tick
        Self::delay_task(interval_ms, Message::TimelapseTick)
    }

    pub(crate) fn handle_set_timelapse_interval(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
        use cosmic::cosmic_config::CosmicConfigEntry;
        if let Some(&interval) = crate::config::TimelapseInterval::ALL.get(index) {
            self.config.timelapse_interval = interval;
            if let Some(handler) = self.config_handler.as_ref()
                && let Err(err) = self.config.write_entry(handler)
            {
                error!(?err, "Failed to save timelapse interval");
            }
        }
        Task::none()
    }
}

/// Call the XDG Inhibit portal (org.freedesktop.portal.Inhibit) to block idle
/// lock + suspend. Returns a guard whose Drop closes the request handle.
///
/// The portal is always reachable inside the Flatpak sandbox, so it needs no
/// finish-args, unlike direct D-Bus calls to logind (system bus) or the
/// screensaver. Desktops that ship no Inhibit backend (e.g. COSMIC, tracked at
/// pop-os/xdg-desktop-portal-cosmic#342) return an error here and the caller
/// leaves the session un-inhibited; there is no direct-D-Bus fallback.
fn portal_inhibit() -> Result<crate::app::state::InhibitGuard, String> {
    use std::collections::HashMap;
    use zbus::zvariant::Value;

    // Inhibit flags bitmask: 4 = Suspend, 8 = Idle. Together these match the
    // previous logind "idle:sleep" request.
    const INHIBIT_SUSPEND: u32 = 4;
    const INHIBIT_IDLE: u32 = 8;

    let conn = zbus::blocking::Connection::session()
        .map_err(|e| format!("D-Bus session connection: {e}"))?;
    let options: HashMap<&str, Value> = HashMap::from([("reason", Value::from("Camera active"))]);
    let handle: zbus::zvariant::OwnedObjectPath = conn
        .call_method(
            Some("org.freedesktop.portal.Desktop"),
            "/org/freedesktop/portal/desktop",
            Some("org.freedesktop.portal.Inhibit"),
            "Inhibit",
            // (window, flags, options): empty window means no parent surface.
            &("", INHIBIT_SUSPEND | INHIBIT_IDLE, options),
        )
        .map_err(|e| format!("portal Inhibit: {e}"))?
        .body()
        .deserialize()
        .map_err(|e| format!("portal Inhibit response: {e}"))?;
    Ok(crate::app::state::InhibitGuard {
        connection: conn,
        handle,
    })
}

/// Async function to process collected burst mode frames (GPU-only)
///
/// Uses the unified GPU pipeline for all processing:
/// 1. Initialize GPU pipeline
/// 2. Select reference frame (GPU sharpness)
/// 3. Align frames (GPU)
/// 4. Merge frames (GPU spatial or FFT)
/// 5. Apply tone mapping (GPU)
/// 6. Apply selected filter (GPU)
/// 7. Apply aspect ratio crop (if configured)
/// 8. Save output
///
/// Progress updates are sent via the provided atomic counter (progress * 1000).
async fn process_burst_mode_frames_with_atomic(
    frames: Vec<Arc<crate::backends::camera::types::CameraFrame>>,
    save_dir: PathBuf,
    config: BurstModeConfig,
    progress_atomic: Arc<std::sync::atomic::AtomicU32>,
    filter: crate::app::FilterType,
) -> Result<String, String> {
    use crate::pipelines::photo::burst_mode::{
        ProgressCallback, SaveOutputParams, export_burst_frames_dng, process_burst_mode,
        save_output,
    };

    info!(
        frame_count = frames.len(),
        crop_rect = ?config.crop_rect,
        encoding_format = ?config.encoding_format,
        save_burst_raw_dng = config.save_burst_raw_dng,
        filter = ?filter,
        "Processing burst mode frames (GPU-only FFT pipeline)"
    );

    // Store fields before moving config
    let crop_rect = config.crop_rect;
    let encoding_format = config.encoding_format;
    let camera_metadata = config.camera_metadata.clone();
    let save_burst_raw_dng = config.save_burst_raw_dng;
    let rotation = config.rotation;
    let mirror_horizontal = config.mirror_horizontal;

    // Export raw burst frames as DNG if enabled (before processing)
    if save_burst_raw_dng {
        match export_burst_frames_dng(&frames, save_dir.clone(), &camera_metadata).await {
            Ok(burst_dir) => {
                info!(burst_dir = %burst_dir.display(), "Raw burst frames saved as DNG");
            }
            Err(e) => {
                error!(error = %e, "Failed to export raw burst frames as DNG");
                // Continue with processing even if export fails
            }
        }
    }

    // Save first frame as reference (before HDR+ processing)
    if let Some(first_frame) = frames.first()
        && let Err(e) = save_first_burst_frame(
            first_frame,
            &save_dir,
            crop_rect,
            encoding_format,
            &camera_metadata,
            filter,
            rotation,
            mirror_horizontal,
        )
        .await
    {
        warn!(error = %e, "Failed to save first burst frame");
        // Continue with HDR+ processing even if first frame save fails
    }

    // Create progress callback that updates the atomic counter
    let progress_callback: ProgressCallback = Arc::new(move |progress: f32| {
        let progress_int = (progress * 1000.0) as u32;
        progress_atomic.store(progress_int, std::sync::atomic::Ordering::Relaxed);
    });

    // Process using the unified GPU pipeline with progress reporting
    let merged = process_burst_mode(frames, config, Some(progress_callback)).await?;

    // Save output with optional crop, filter, rotation, and selected encoding format
    let output_path = save_output(
        &merged,
        SaveOutputParams {
            output_dir: save_dir,
            crop_rect,
            encoding_format,
            camera_metadata,
            filter: Some(filter),
            rotation,
            filename_suffix: Some("_HDR+"),
            mirror_horizontal,
        },
    )
    .await?;

    info!(path = %output_path.display(), "Burst mode photo saved");
    Ok(output_path.display().to_string())
}

/// Save the first frame of a burst as a separate file for comparison
#[allow(clippy::too_many_arguments)]
async fn save_first_burst_frame(
    frame: &crate::backends::camera::types::CameraFrame,
    save_dir: &std::path::Path,
    crop_rect: Option<(u32, u32, u32, u32)>,
    encoding_format: crate::pipelines::photo::EncodingFormat,
    camera_metadata: &crate::pipelines::photo::CameraMetadata,
    filter: crate::app::FilterType,
    rotation: crate::backends::camera::types::SensorRotation,
    mirror_horizontal: bool,
) -> Result<PathBuf, String> {
    use crate::pipelines::photo::burst_mode::{MergedFrame, SaveOutputParams, save_output};

    // Skip first-frame comparison save for raw Bayer frames — the HDR+ output is the
    // important artifact. Converting a single raw Bayer frame to RGBA just for comparison
    // adds complexity and latency (GPU debayer required).
    if frame.format.is_bayer() {
        return Err("Skipping first-frame comparison for raw Bayer input".to_string());
    }

    // Convert CameraFrame to MergedFrame for save_output
    // Strip stride padding if present (viewfinder RGBA may have padded rows)
    let data = {
        let row_bytes = (frame.width * 4) as usize;
        let stride = frame.stride as usize;
        if stride > row_bytes && stride > 0 {
            let mut out = Vec::with_capacity(row_bytes * frame.height as usize);
            for y in 0..frame.height as usize {
                out.extend_from_slice(&frame.data[y * stride..y * stride + row_bytes]);
            }
            out
        } else {
            frame.data.to_vec()
        }
    };
    let merged = MergedFrame {
        data,
        width: frame.width,
        height: frame.height,
    };

    // Reuse save_output with no filename suffix (plain IMG_{timestamp})
    let path = save_output(
        &merged,
        SaveOutputParams {
            output_dir: save_dir.to_path_buf(),
            crop_rect,
            encoding_format,
            camera_metadata: camera_metadata.clone(),
            filter: Some(filter),
            rotation,
            filename_suffix: None, // No suffix for first frame
            mirror_horizontal,
        },
    )
    .await?;

    info!(path = %path.display(), "First burst frame saved for comparison");
    Ok(path)
}

/// Wait for a still frame to appear in the shared mutex.
///
/// Awaits a notification from the capture thread (no polling), then takes the
/// frame out of the shared slot. Returns `None` on timeout.
///
/// Subscribing to `Notify` *before* checking the slot is important: the
/// capture thread may have already stored the frame between the caller's
/// `still_requested.store(true, …)` and this await, in which case we'd miss
/// the notification. The pre-await check below catches that case.
pub(super) async fn wait_for_still_frame(
    still_frame: &std::sync::Mutex<Option<crate::backends::camera::types::CameraFrame>>,
    still_frame_notify: &tokio::sync::Notify,
    timeout: std::time::Duration,
) -> Option<crate::backends::camera::types::CameraFrame> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        // Register interest BEFORE checking the slot, otherwise we'd race
        // with a frame stored between the take attempt and the await.
        let notified = still_frame_notify.notified();
        if let Ok(mut guard) = still_frame.lock()
            && let Some(frame) = guard.take()
        {
            return Some(frame);
        }
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return None;
        }
        if tokio::time::timeout(remaining, notified).await.is_err() {
            return None;
        }
    }
}
