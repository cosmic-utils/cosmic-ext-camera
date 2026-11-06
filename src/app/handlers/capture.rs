// SPDX-License-Identifier: GPL-3.0-only

//! Capture operations handlers
//!
//! Handles photo capture, video recording, flash, zoom, and timer functionality.

use crate::app::state::{AppModel, CameraMode, Message, RecordingState, TimelapseState};

use crate::pipelines::capture_metadata::CaptureMetadata;
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

/// Keep synchronous GPU waits and photo-sized CPU work off libcosmic's single
/// async worker so preview subscriptions and animation tasks can keep running.
/// Once started, the blocking job survives cancellation of this outer future
/// and may finish writing the photo; its workspaces retire when the job ends.
async fn run_photo_save(
    gate: Arc<tokio::sync::Semaphore>,
    job: impl std::future::Future<Output = Result<String, String>> + Send + 'static,
) -> Result<String, String> {
    let permit = gate
        .acquire_owned()
        .await
        .map_err(|error| format!("Photo queue closed: {error}"))?;
    let runtime = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        // The blocking worker, not its cancellable async caller, owns the gate.
        let _permit = permit;
        runtime.block_on(job)
    })
    .await
    .map_err(|error| format!("Photo save task failed: {error}"))?
}

impl AppModel {
    /// Acquisition and bounded processing admission shared by every photo shutter.
    /// Processing jobs alone do not lock acquisition while a slot remains.
    pub(crate) fn can_accept_photo(&self) -> bool {
        !self.pending_close
            && !self.transition_state.ui_disabled
            && !self.is_capturing
            && self.hardware_photo_id.is_none()
            && self.motion_photo_capture.is_none()
            && !self.burst_mode.is_active()
            && self.photo_jobs.len() < 8
            && self.current_frame.as_deref().is_none_or(|frame| {
                crate::backends::camera::zoom::frame_can_represent_zoom(
                    self.current_zoom_level(),
                    frame,
                )
            })
    }

    /// Snapshot photo-only processing preferences before dispatching a save job.
    fn photo_processing_config(&self) -> crate::pipelines::photo::PostProcessingConfig {
        crate::pipelines::photo::PostProcessingConfig {
            demosaic_method: self.config.demosaic_method,
            ..Default::default()
        }
    }

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
    pub(crate) fn persist_config_async(&mut self) {
        let Some(handler) = self.config_handler.as_ref() else {
            return;
        };
        self.config_persistence.save(handler, &self.config);
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
    pub(super) fn build_camera_metadata(&self) -> CaptureMetadata {
        let host = crate::pipelines::capture_metadata::cached_host_device_metadata();
        let mut metadata = self
            .available_cameras
            .get(self.current_camera_index)
            .map(|cam| CaptureMetadata {
                camera_name: Some(cam.name.clone()),
                camera_driver: cam.device_info.as_ref().map(|info| info.driver.clone()),
                sensor_model: cam.sensor_model.clone(),
                camera_location: cam.camera_location.clone(),
                pipeline_handler: cam.pipeline_handler.clone(),
                libcamera_version: cam.libcamera_version.clone(),
                device_make: host.and_then(|h| h.make.clone()),
                device_model: host.and_then(|h| h.model.clone()),
                captured_at: Some(chrono::Local::now().fixed_offset()),
                ..Default::default()
            })
            .unwrap_or_else(|| CaptureMetadata {
                device_make: host.and_then(|h| h.make.clone()),
                device_model: host.and_then(|h| h.model.clone()),
                captured_at: Some(chrono::Local::now().fixed_offset()),
                ..Default::default()
            });
        // Last observed controls, not a claim of exact RAW/HDR/ZSL exposure.
        // Never open/ioctl a device or initialize host discovery at shutter.
        if let Some(settings) = &self.exposure_settings {
            metadata.exposure_time = settings.exposure_time.map(|v| f64::from(v) * 0.0001);
            metadata.iso = settings.iso.and_then(|v| u32::try_from(v).ok());
            metadata.gain = settings.gain;
        }
        metadata
    }

    /// Capture the current frame as a photo with the selected filter and zoom
    pub(crate) fn capture_photo(&mut self) -> Task<cosmic::Action<Message>> {
        self.capture_photo_with_frame(None)
    }

    /// Capture a photo, optionally using a pre-captured frame (zero-shutter-lag).
    /// Falls back to `self.current_frame` if ZSL cannot represent the shutter zoom.
    fn capture_photo_with_frame(
        &mut self,
        zsl_frame: Option<Arc<crate::backends::camera::types::CameraFrame>>,
    ) -> Task<cosmic::Action<Message>> {
        if !self.can_accept_photo() {
            return Task::none();
        }
        let zoom_level = self.current_zoom_level();
        let suitable = |frame: &Arc<crate::backends::camera::types::CameraFrame>| {
            crate::backends::camera::zoom::frame_can_represent_zoom(zoom_level, frame)
        };
        let frame_arc = zsl_frame.filter(suitable).or_else(|| {
            self.current_frame
                .as_ref()
                .filter(|frame| suitable(frame))
                .cloned()
        });
        // Ordinary stills need usable pixels before accepting a Motion Photo
        // or allocating a photo ID. RAW/HDR acquire their own frames.
        if frame_arc.is_none()
            && (self.current_frame_is_file_source
                || (!self.would_use_burst_mode() && !self.is_current_camera_multistream()))
        {
            info!("No frame available for the requested capture zoom");
            return Task::none();
        }
        if self.motion_photo_enabled()
            && matches!(
                self.config.photo_output_format,
                crate::config::PhotoOutputFormat::Jpeg | crate::config::PhotoOutputFormat::Avif
            )
        {
            if let Err(error) = self.start_motion_photo_capture() {
                error!(%error, "Motion Photo unavailable; still shutter continues");
                self.motion_photo_error = Some(error);
            } else {
                self.motion_photo_error = None;
            }
        }
        let capture_id = self.next_photo_id.max(1);
        self.next_photo_id = capture_id + 1;
        // Use HDR+ burst mode only if it would actually be used (frame_count > 1)
        // This respects auto-detected brightness and user override.
        // Skip when file source is active — burst needs multiple live frames.
        if self.would_use_burst_mode() && !self.current_frame_is_file_source {
            self.hardware_photo_id = Some(capture_id);
            let task = self.capture_burst_mode_photo();
            return Task::batch([
                task,
                Self::delay_task(60_000, Message::PhotoEncodingWatchdog(capture_id)),
            ]);
        }

        // In multistream mode, capture from the raw stream (full sensor resolution)
        // instead of the preview stream (1080p).
        // Skip when file source is active — there is no capture thread to provide raw frames.
        if self.is_current_camera_multistream() && !self.current_frame_is_file_source {
            self.hardware_photo_id = Some(capture_id);
            let task = self.capture_photo_from_raw_stream(capture_id);
            return Task::batch([
                task,
                Self::delay_task(60_000, Message::PhotoEncodingWatchdog(capture_id)),
            ]);
        }

        let Some(frame_arc) = frame_arc else {
            return Task::none();
        };

        info!("Capturing photo...");
        self.photo_jobs.insert(
            capture_id,
            crate::app::handlers::motion_photo::PhotoJob {
                live: self.motion_photo_capture.take(),
                still_path: None,
            },
        );
        // Ordinary acquisition is an Arc snapshot, so encoding must not lock
        // the shutter. Hardware/HDR still retains its explicit acquisition lock.

        let save_dir = crate::app::get_photo_directory(&self.config.save_folder_name);
        let filter_type = self.selected_filter;
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
                // Fit preview applies the canonical crop in sensor space.
                // Only the preview rotation, not window shape, swaps its axes.
                self.photo_aspect_ratio
                    .crop_rect(fw, fh, geometry_rotation.swaps_dimensions())
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

        let photo_config = self.photo_processing_config();
        let photo_session = crate::shaders::current_photo_session();
        let mut camera_metadata = self.build_camera_metadata();
        camera_metadata.orientation =
            crate::pipelines::capture_metadata::CaptureOrientation::from_capture(
                media_rotation,
                mirror_horizontal,
            );

        let processing_gate = Arc::clone(&self.photo_processing_gate.0);
        let save_task = Task::perform(
            run_photo_save(processing_gate, async move {
                use crate::pipelines::photo::{
                    EncodingQuality, PhotoPipeline, PostProcessingConfig,
                };
                let config = PostProcessingConfig {
                    filter_type,
                    crop_rect,
                    zoom_level,
                    ..photo_config
                };
                let mut pipeline =
                    PhotoPipeline::with_config(config, encoding_format, EncodingQuality::High);
                pipeline.set_photo_session(photo_session);
                pipeline.set_camera_metadata(camera_metadata);
                pipeline
                    .capture_and_save(frame_arc, save_dir)
                    .await
                    .map(|p| p.display().to_string())
            }),
            move |result| cosmic::Action::App(Message::PhotoCaptureSaved(capture_id, result)),
        );

        let animation_task = Self::delay_task(150, Message::ClearCaptureAnimation);
        Task::batch([
            save_task,
            animation_task,
            Self::delay_task(60_000, Message::PhotoEncodingWatchdog(capture_id)),
        ])
    }

    /// Capture a photo from the raw stream (multistream mode)
    ///
    /// Requests a full-resolution raw frame from the dedicated raw stream,
    /// which bypasses the ISP and captures at the sensor's native resolution.
    fn capture_photo_from_raw_stream(&mut self, capture_id: u64) -> Task<cosmic::Action<Message>> {
        info!("Capturing photo from raw stream (multistream mode)...");
        self.is_capturing = true;

        // Request a still capture from the raw stream
        let still_request = self.still_capture.request();
        let preview_frame = self.current_frame.as_ref().map(Arc::clone);
        let save_dir = crate::app::get_photo_directory(&self.config.save_folder_name);
        let filter_type = self.selected_filter;
        let zoom_level = self.current_zoom_level();
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
        let cover_screen_w = self.screen_width;
        let cover_screen_h = self.screen_height;
        let cover_insets = self.settled_bar_insets();
        let cover_target_ratio = self
            .photo_aspect_ratio
            .display_ratio(self.screen_is_portrait());

        let encoding_format: crate::pipelines::photo::EncodingFormat =
            self.config.photo_output_format.into();

        let photo_config = self.photo_processing_config();
        let mut camera_metadata = self.build_camera_metadata();
        camera_metadata.orientation =
            crate::pipelines::capture_metadata::CaptureOrientation::from_capture(
                media_rotation,
                mirror_horizontal,
            );

        let raw_photo_config = photo_config.clone();
        let raw_photo_session = crate::shaders::current_photo_session();
        let raw_save_dir = save_dir.clone();
        let raw_camera_metadata = camera_metadata.clone();
        let processing_gate = Arc::clone(&self.photo_processing_gate.0);
        let save_task = Task::run(
            async_stream::stream! {
                // Acquisition must not wait for processing or hold the UI shutter
                // lock through GPU work and file encoding (including fallback).
                let frame = wait_for_still_frame(
                    still_request, std::time::Duration::from_secs(2),
                ).await.ok_or_else(|| "Timeout waiting for raw frame from still stream".to_string());
                yield Message::PhotoCaptureAcquired(capture_id);
                let result = run_photo_save(processing_gate, capture_with_preview_fallback(
                    move || async move {
                        use crate::pipelines::photo::{
                            EncodingQuality, PhotoPipeline, PostProcessingConfig,
                        };

                        let frame = frame?;

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
                                photo_aspect_ratio.crop_rect(
                                    rw,
                                    rh,
                                    geometry_rotation.swaps_dimensions(),
                                )
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
                            ..raw_photo_config
                        };
                        let mut pipeline =
                            PhotoPipeline::with_config(config, encoding_format, EncodingQuality::High);
                        pipeline.set_photo_session(raw_photo_session);
                        pipeline.set_camera_metadata(raw_camera_metadata);
                        pipeline
                            .capture_and_save(Arc::new(frame), raw_save_dir)
                            .await
                            .map(|p| p.display().to_string())
                    },
                    move || async move {
                        use crate::pipelines::photo::{
                            EncodingQuality, PhotoPipeline, PostProcessingConfig,
                        };

                        let frame = preview_frame.ok_or_else(|| {
                            "Raw capture failed and no preview frame is available".to_string()
                        })?;
                        info!(
                            width = frame.width,
                            height = frame.height,
                            format = ?frame.format,
                            "Saving ISP-processed preview frame as CPU fallback"
                        );

                        let (fw, fh) = if geometry_rotation.swaps_dimensions() {
                            (frame.height, frame.width)
                        } else {
                            (frame.width, frame.height)
                        };
                        let crop_rect = {
                            let (x, y, w, h) = if preview_fit_to_view {
                                photo_aspect_ratio.crop_rect(
                                    fw,
                                    fh,
                                    geometry_rotation.swaps_dimensions(),
                                )
                            } else {
                                crate::app::preview_geometry::cover_capture_crop(
                                    fw,
                                    fh,
                                    cover_screen_w,
                                    cover_screen_h,
                                    cover_insets,
                                    cover_target_ratio,
                                )
                            };
                            if x == 0 && y == 0 && w == fw && h == fh {
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
                            ..photo_config
                        };
                        let mut pipeline =
                            PhotoPipeline::with_config(config, encoding_format, EncodingQuality::High);
                        pipeline.set_camera_metadata(camera_metadata);
                        pipeline
                            .capture_and_save(frame, save_dir)
                            .await
                            .map(|p| p.display().to_string())
                    },
                )).await;
                yield Message::PhotoCaptureSaved(capture_id, result);
            },
            cosmic::Action::App,
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
        self.burst_mode.capture_zoom = self.current_zoom_level();
        // Stop an in-flight reset animation at the shutter's framing. A
        // burst must not change field of view between alignment inputs.
        self.zoom_level = self.burst_mode.capture_zoom;
        self.zoom_animation = None;
        self.sync_zoom_control();
        self.burst_mode.demosaic_method = self.config.demosaic_method;

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
            let still_capture = Arc::clone(&self.still_capture);
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
                        let still_request = still_capture.request();

                        // Wait for raw frame with timeout
                        let timeout = std::time::Duration::from_secs(2);
                        let frame = wait_for_still_frame(
                            still_request,
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

    /// A lost viewfinder cannot finish HDR collection. Encoding/processing
    /// already dispatched owns its bytes and must continue independently.
    pub(crate) fn fail_unfinished_viewfinder_hdr(&mut self, error: &str) {
        if !self.burst_mode.is_collecting_frames()
            || self.is_current_camera_multistream()
            // All frames may already be owned while FramesCollected is queued.
            || self.burst_mode.frames_captured() >= self.burst_mode.target_frame_count
        {
            return;
        }
        self.burst_mode.take_frames();
        self.burst_mode.error();
        self.is_capturing = false;
        self.hardware_photo_id = None;
        // Drop cancels only this acquisition's motion request, not photo_jobs.
        self.motion_photo_capture = None;
        self.motion_photo_error = Some(error.to_owned());
        self.turn_off_flash_hardware();
        self.flash.active = false;
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
                if self.motion_photo_capture.take().is_some() {
                    self.motion_photo_error = Some(e.clone());
                }
                self.burst_mode.error();
                self.is_capturing = false;
                self.hardware_photo_id = None;
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
        if self.motion_photo_capture.is_none() {
            info!("Stopping camera stream for HDR+ processing");
            self.camera_cancel_flag
                .store(true, std::sync::atomic::Ordering::Release);
        }

        // Take the frames from the buffer
        let frames: Vec<Arc<crate::backends::camera::types::CameraFrame>> =
            self.burst_mode.take_frames();

        if frames.len() < 2 {
            if self.motion_photo_capture.take().is_some() {
                self.motion_photo_error = Some("Not enough frames collected for burst mode".into());
            }
            error!("Not enough frames collected for burst mode");
            self.burst_mode.error();
            self.is_capturing = false;
            self.hardware_photo_id = None;
            return Task::none();
        }

        let save_dir = crate::app::get_photo_directory(&self.config.save_folder_name);

        // Motion Photo stills and clips share the shutter snapshot even if the
        // user changes controls while a raw/HDR burst is being collected.
        let live_snapshot = self
            .motion_photo_capture
            .as_ref()
            .map(|live| &live.snapshot);
        let encoding_format: crate::pipelines::photo::EncodingFormat = live_snapshot
            .map(|s| s.photo_output_format)
            .unwrap_or(self.config.photo_output_format)
            .into();
        let mut camera_metadata = live_snapshot
            .map(|s| s.camera_metadata.clone())
            .unwrap_or_else(|| self.build_camera_metadata());
        let geometry_rotation = self.capture_geometry_rotation();
        let media_rotation = live_snapshot
            .map(|s| s.media_rotation)
            .unwrap_or_else(|| self.capture_media_rotation());
        let mirror = live_snapshot
            .map(|s| s.mirror)
            .unwrap_or_else(|| self.should_mirror_captures());
        camera_metadata.orientation =
            crate::pipelines::capture_metadata::CaptureOrientation::from_capture(
                media_rotation,
                mirror,
            );

        // Calculate crop rectangle based on preview mode and aspect ratio.
        // Cover-mode crop is computed by inverse-mapping the on-screen
        // framed rect back to sensor coords (`cover_capture_crop`) so the
        // saved photo matches what the user sees inside the crop bars.
        let portrait = self.screen_is_portrait();
        let crop_rect = if self.motion_photo_capture.is_some()
            && let Some(frame) = frames.first()
        {
            self.motion_photo_burst_frame_crop(frame)
        } else if let Some(frame) = frames.first() {
            let (rw, rh) = if geometry_rotation.swaps_dimensions() {
                (frame.height, frame.width)
            } else {
                (frame.width, frame.height)
            };
            let (x, y, w, h) = if self.preview_fit_to_view {
                self.photo_aspect_ratio
                    .crop_rect(rw, rh, geometry_rotation.swaps_dimensions())
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

        let crop_rect = if self.motion_photo_capture.is_none() {
            match frames
                .first()
                .map(|frame| {
                    crate::app::zoom::burst_zoom_crop(
                        frame,
                        crop_rect,
                        self.burst_mode.capture_zoom,
                    )
                })
                .transpose()
            {
                Ok(crop) => crop.flatten(),
                Err(error) => return self.handle_burst_mode_complete(Err(error)),
            }
        } else {
            crop_rect
        };

        // Create burst mode config with user's settings
        let mut config = BurstModeConfig::default();
        config.crop_rect = crop_rect;
        config.encoding_format = encoding_format;
        config.camera_metadata = camera_metadata;
        config.save_burst_raw_dng = self.config.save_burst_raw;
        config.demosaic_method = self.burst_mode.demosaic_method;

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

        // Preserve the shutter filter while a Motion Photo is in flight.
        let selected_filter = live_snapshot
            .map(|s| s.filter)
            .unwrap_or(self.selected_filter);

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

    fn start_photo_timer(&mut self) -> Task<cosmic::Action<Message>> {
        let seconds = self.photo_timer_setting.seconds();
        info!(seconds, "Starting photo timer countdown");
        self.photo_timer_countdown = Some(seconds);
        self.photo_timer_tick_start = Some(std::time::Instant::now());
        Self::delay_task(1000, Message::PhotoTimerTick)
    }

    fn start_preflash(&mut self) -> Task<cosmic::Action<Message>> {
        if self.use_hardware_flash() {
            info!("Flash enabled - turning on hardware flash before capture");
            self.turn_on_flash_hardware();
        } else {
            info!("Flash enabled - showing screen flash before capture");
            self.flash.active = true;
        }
        Self::delay_task(1000, Message::FlashComplete)
    }

    pub(crate) fn handle_capture(&mut self) -> Task<cosmic::Action<Message>> {
        // If quick-record is active, ignore direct Capture messages
        // (the quick-record state machine handles capture)
        if self.quick_record.is_pressed() || self.quick_record.is_recording() {
            return Task::none();
        }

        // If timer countdown is active, abort it
        if self.photo_timer_countdown.is_some() {
            return self.handle_abort_photo_timer();
        }

        if !self.can_accept_photo() {
            return Task::none();
        }

        // Animate the correct button depending on recording state
        if self.recording.is_recording() {
            self.haptic_tap();
            self.animate_photo_btn_scale(0.82);
        } else {
            self.animate_capture_scale(0.82);
        }

        // In Photo mode with timer set, start countdown
        if self.mode == CameraMode::Photo
            && self.photo_timer_setting != crate::app::state::PhotoTimerSetting::Off
        {
            return self.start_photo_timer();
        }

        // Normal capture flow (with flash check)
        if self.mode == CameraMode::Photo && self.flash.enabled && !self.flash.active {
            return self.start_preflash();
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
        self.show_toggle_feedback(crate::app::toggle_feedback::ToggleKind::Flash)
    }

    pub(crate) fn handle_toggle_burst_mode(&mut self) -> Task<cosmic::Action<Message>> {
        use crate::config::BurstModeSetting;

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
        if config_changed {
            self.persist_config_async();
        }
        self.show_toggle_feedback(crate::app::toggle_feedback::ToggleKind::Hdr)
    }

    pub(crate) fn handle_set_burst_mode_frame_count(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
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

        self.persist_config_async();
        Task::none()
    }

    pub(crate) fn handle_cycle_photo_aspect_ratio(&mut self) -> Task<cosmic::Action<Message>> {
        self.photo_aspect_ratio = self.photo_aspect_ratio.next();
        info!(aspect_ratio = ?self.photo_aspect_ratio, "Photo aspect ratio changed");

        self.config.photo_aspect_ratio = self.photo_aspect_ratio;
        self.persist_config_async();

        Task::none()
    }

    pub(crate) fn handle_flash_complete(&mut self) -> Task<cosmic::Action<Message>> {
        if self.pending_close {
            return Task::none();
        }
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
        if self.pending_close {
            return Task::none();
        }
        if let Some(remaining) = self.photo_timer_countdown {
            if remaining <= 1 {
                // Countdown complete - capture the photo
                info!("Photo timer countdown complete - capturing");
                self.photo_timer_countdown = None;
                self.photo_timer_tick_start = None;
                self.animate_capture_scale(1.0);
                // Check if flash is enabled
                if self.flash.enabled && !self.flash.active {
                    return self.start_preflash();
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

    pub(crate) fn sync_zoom_control(&self) {
        self.zoom_control.set_zoom(self.current_zoom_level());
    }

    pub(crate) fn handle_zoom_in(&mut self) -> Task<cosmic::Action<Message>> {
        if self.burst_mode.is_collecting_frames() {
            return Task::none();
        }
        // Multiplicative zoom: each step is ~2% magnification change,
        // so it feels consistent at any zoom level.
        let new_zoom = (self.zoom_level * 1.02).min(10.0);
        if (new_zoom - self.zoom_level).abs() > 0.001 {
            self.zoom_level = new_zoom;
            self.zoom_animation = None; // step zoom is real-time
            debug!(zoom = self.zoom_level, "Zoom in");
        }
        self.sync_zoom_control();
        Task::none()
    }

    pub(crate) fn handle_zoom_out(&mut self) -> Task<cosmic::Action<Message>> {
        if self.burst_mode.is_collecting_frames() {
            return Task::none();
        }
        // Multiplicative zoom: each step is ~2% magnification change.
        let new_zoom = (self.zoom_level / 1.02).max(1.0);
        if (new_zoom - self.zoom_level).abs() > 0.001 {
            self.zoom_level = new_zoom;
            self.zoom_animation = None; // step zoom is real-time
            debug!(zoom = self.zoom_level, "Zoom out");
        }
        self.sync_zoom_control();
        Task::none()
    }

    /// Reset the zoom level to 1× with an ease-out animation. If a tick chain
    /// is already in flight (re-press during the animation), no new chain is
    /// spawned — the existing one picks up the replaced animation, same
    /// pattern as `start_fit_animation`.
    pub(crate) fn handle_reset_zoom(&mut self) -> Task<cosmic::Action<Message>> {
        if self.burst_mode.is_collecting_frames() {
            return Task::none();
        }
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
        self.sync_zoom_control();
        if was_idle {
            Self::delay_task(16, Message::ZoomAnimationTick)
        } else {
            Task::none()
        }
    }

    pub(crate) fn handle_pinch_zoom(&mut self, level: f32) -> Task<cosmic::Action<Message>> {
        if self.burst_mode.is_collecting_frames() {
            return Task::none();
        }
        let new_zoom = level.clamp(1.0, 10.0);
        if (new_zoom - self.zoom_level).abs() > 0.001 {
            self.zoom_level = new_zoom;
            self.zoom_animation = None; // pinch is real-time
        }
        self.sync_zoom_control();
        Task::none()
    }

    pub(crate) fn handle_photo_capture_acquired(
        &mut self,
        id: u64,
    ) -> Task<cosmic::Action<Message>> {
        // Stale/duplicate acquisition cannot release a newer RAW/HDR shutter
        // or steal its Motion Photo clip.
        if self.hardware_photo_id != Some(id) || self.photo_jobs.contains_key(&id) {
            return Task::none();
        }
        self.photo_jobs.insert(
            id,
            crate::app::handlers::motion_photo::PhotoJob {
                live: self.motion_photo_capture.take(),
                still_path: None,
            },
        );
        self.hardware_photo_id = None;
        self.is_capturing = false;
        Task::none()
    }

    pub(crate) fn handle_photo_capture_saved(
        &mut self,
        id: u64,
        result: Result<String, String>,
    ) -> Task<cosmic::Action<Message>> {
        let Some(job) = self.photo_jobs.get_mut(&id) else {
            return Task::none();
        };
        if let Ok(path) = &result {
            job.still_path = Some(path.clone());
            if let Some(live) = &mut job.live
                && let Some(clip) = live.result.take()
            {
                return Task::perform(
                    crate::app::handlers::motion_photo::embed_completed_clip(
                        PathBuf::from(path),
                        clip,
                    ),
                    move |result| cosmic::Action::App(Message::PhotoCaptureFinished(id, result)),
                );
            }
        }
        self.handle_photo_capture_finished(id, result)
    }

    pub(crate) fn handle_photo_capture_finished(
        &mut self,
        id: u64,
        result: Result<String, String>,
    ) -> Task<cosmic::Action<Message>> {
        let Some(job) = self.photo_jobs.remove(&id) else {
            return Task::none();
        };
        let path = match result {
            Ok(path) => Some(path),
            Err(error) => {
                error!(capture_id = id, %error, "Photo/Motion Photo completion failed");
                self.motion_photo_error = Some(error);
                job.still_path
            }
        };
        // A late completion cannot clear another job or replace a newer image.
        if let Some(path) = path
            && id > self.last_published_photo_id
        {
            self.last_published_photo_id = id;
            self.last_media_path = Some(path);
            return Task::done(cosmic::Action::App(Message::RefreshGalleryThumbnail));
        }
        Task::none()
    }

    pub(crate) fn handle_photo_saved(
        &mut self,
        result: Result<String, String>,
    ) -> Task<cosmic::Action<Message>> {
        if let Some(capture_id) = self.hardware_photo_id.take() {
            self.is_capturing = false;
            self.hardware_photo_id = None;
            self.photo_jobs.insert(
                capture_id,
                crate::app::handlers::motion_photo::PhotoJob {
                    live: self.motion_photo_capture.take(),
                    still_path: None,
                },
            );
            return self.handle_photo_capture_saved(capture_id, result);
        }
        if let Some(live) = &mut self.motion_photo_capture {
            if let Ok(path) = &result
                && let Some(job) = live.result.take()
            {
                self.is_capturing = true;
                self.last_media_path = Some(path.clone());
                let path = PathBuf::from(path);
                return Task::perform(
                    crate::app::handlers::motion_photo::embed_completed_clip(path, job),
                    |result| cosmic::Action::App(Message::PhotoSaved(result)),
                );
            }
            if let Err(error) = &result {
                self.motion_photo_error = Some(error.clone());
            }
            self.motion_photo_capture = None;
        }
        // Always release the capture lock here; the separate
        // `ClearCaptureAnimation` task may be dropped on backpressure or panic,
        // which would otherwise leave the capture button permanently disabled.
        self.is_capturing = false;
        self.hardware_photo_id = None;
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
        if self.pending_close {
            return Task::none();
        }
        self.pending_close = true;
        self.current_frame = None;
        self.core.window.show_context = false;
        self.tools_menu_visible = false;
        self.format_picker_visible = false;
        self.exposure_picker_visible = false;
        self.color_picker_visible = false;
        self.motor_picker_visible = false;
        self.recording_keybind = None;
        self.fit_animation = None;
        self.zoom_animation = None;
        crate::backends::volume_keys::set_focused(false);
        self.sync_audio_probe();
        if let Some(stop) = self.video_preview_stop_tx.take() {
            let _ = stop.send(());
        }
        if let Some(stop) = self.virtual_camera.take_stop_sender() {
            let _ = stop.send(());
        }
        self.quick_record = crate::app::state::QuickRecordState::Idle;
        self.photo_timer_countdown = None;
        self.photo_timer_tick_start = None;
        self.flash.enabled = false;
        self.flash.active = false;
        self.turn_off_flash_hardware();
        // Keep camera/encoded memory worker alive until all accepted post-roll
        // windows and still encoders finish. Closing never cancels accepted media.
        self.request_recording_stop();
        let timelapse_stop = if self.timelapse.is_running() {
            self.stop_timelapse()
        } else {
            Task::none()
        };
        self.reconcile_shutdown_preview();
        Task::batch([
            timelapse_stop,
            Self::delay_task(30_000, Message::ShutdownDrainWatchdog),
        ])
    }

    pub(crate) fn handle_shutdown_drain_watchdog(&mut self) -> Task<cosmic::Action<Message>> {
        if self.pending_close
            && (self.shutdown_has_pending_media() || self.shutdown_waiting_for_recorder)
        {
            let error = "Shutdown drain exceeded its watchdog deadline; waiting for accepted media, not abandoning it".to_string();
            error!(%error);
            self.motion_photo_error = Some(error);
        }
        Task::none()
    }

    pub(crate) fn handle_photo_encoding_watchdog(
        &mut self,
        id: u64,
    ) -> Task<cosmic::Action<Message>> {
        if self.photo_jobs.contains_key(&id) || self.hardware_photo_id == Some(id) {
            let error = format!(
                "Photo {id} encoder exceeded its watchdog deadline; still waiting to preserve accepted media"
            );
            error!(%error);
            self.motion_photo_error = Some(error);
        }
        Task::none()
    }

    /// Count accepted output files, not acquisition/encoding phases. A Live
    /// Photo's still and embedded clip are one save, as is an HDR frame burst.
    pub(crate) fn shutdown_pending_saves(&self) -> usize {
        let acquisition = self.hardware_photo_id.map_or_else(
            || usize::from(self.is_capturing || self.motion_photo_capture.is_some()),
            |id| usize::from(!self.photo_jobs.contains_key(&id)),
        );
        let active_video = self
            .recording
            .session()
            .map_or(0, |id| usize::from(!self.recording_jobs.contains(&id)));
        self.photo_jobs.len()
            + acquisition
            + self.recording_jobs.len()
            + active_video
            + usize::from(self.timelapse.is_active())
    }

    pub(crate) fn shutdown_has_pending_media(&self) -> bool {
        self.shutdown_pending_saves() != 0
    }

    pub(crate) fn maybe_finish_shutdown(&mut self) -> Task<cosmic::Action<Message>> {
        if !self.pending_close
            || self.shutdown_waiting_for_recorder
            || self.shutdown_has_pending_media()
        {
            return Task::none();
        }
        self.shutdown_waiting_for_recorder = true;
        let recorder = self.motion_photo_recorder.take();
        let retired = self.motion_photo_retiring.take();
        let preferences = std::mem::take(&mut self.config_persistence);
        Task::perform(
            async move {
                preferences.finish().await?;
                if let Some(recorder) = recorder {
                    recorder.stop_and_wait().await?;
                }
                if let Some(worker) = retired {
                    tokio::task::spawn_blocking(move || {
                        worker
                            .join()
                            .map_err(|_| "Retired Motion Photo worker panicked".to_string())
                    })
                    .await
                    .map_err(|e| e.to_string())??;
                }
                Ok(())
            },
            |result| cosmic::Action::App(Message::ShutdownRecorderStopped(result)),
        )
    }

    pub(crate) fn handle_shutdown_recorder_stopped(
        &mut self,
        result: Result<(), String>,
    ) -> Task<cosmic::Action<Message>> {
        self.shutdown_waiting_for_recorder = false;
        if let Err(error) = result {
            self.motion_photo_error = Some(error);
        }
        if self.pending_close && !self.shutdown_has_pending_media() {
            self.shutdown_and_exit(0);
        }
        Task::none()
    }

    /// Arm the deadline before handing Stop to a worker that may already be stuck.
    /// Retain job ownership until native cleanup really completes.
    pub(crate) fn request_recording_stop(&mut self) {
        if let Some(session) = self.recording.session()
            && let Some(monitor) =
                self.encoder_operations
                    .get(&crate::app::recording_error::EncoderOperation::Video(
                        session,
                    ))
        {
            monitor.set_phase(crate::pipelines::video::RecordingPhase::Finalizing);
        }
        if let Some(sender) = self.recording.take_stop_sender() {
            let _ = sender.send(());
        }
    }

    pub(crate) fn handle_toggle_recording(&mut self) -> Task<cosmic::Action<Message>> {
        if self.encoder_restart_required() {
            self.show_encoder_restart_warning();
            return Task::none();
        }
        if self.pending_close || (!self.recording.is_recording() && !self.recording_jobs.is_empty())
        {
            self.motion_photo_error =
                Some("Video shutter waits for prior video finalization or shutdown".into());
            return Task::none();
        }
        self.haptic_tap();
        if self.recording.is_recording() {
            // Stopping: animate release (scale back up)
            self.animate_capture_scale(1.0);
            // Turn off torch when stopping recording
            if self.flash.enabled {
                self.turn_off_flash_hardware();
            }
            self.request_recording_stop();
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
        if self.encoder_restart_required() {
            return Task::none();
        }
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
        let known_session = self.recording_jobs.contains(&session) || is_current;
        let operation = crate::app::recording_error::EncoderOperation::Video(session);
        if known_session && result.is_err() && !self.encoder_operations.contains_key(&operation) {
            let encoder = self
                .available_video_encoders
                .get(self.current_video_encoder_index)
                .map(|encoder| encoder.element_name.clone())
                .unwrap_or_else(|| crate::fl!("recording-error-automatic"));
            // Legacy/synthetic completions have no native owner to publish Finished.
            let monitor = crate::pipelines::video::RecordingMonitor::new(encoder);
            monitor.finish();
            self.register_encoder_operation(operation, monitor);
        }
        if self.encoder_operations.contains_key(&operation) {
            self.finish_encoder_operation(operation, result.as_ref().err().cloned());
        } else {
            self.recording_jobs.remove(&session);
        }

        if is_current {
            self.recording = RecordingState::Idle;
            self.sync_audio_probe();
            self.quick_record = crate::app::state::QuickRecordState::Idle;
            self.update_idle_inhibit();
            // Turn off torch when recording ends
            self.turn_off_flash_hardware();
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
        use crate::app::state::QuickRecordState;

        // Only handle in Photo mode when idle
        if self.pending_close
            || self.mode != CameraMode::Photo
            || self.recording.is_recording()
            || self.quick_record.is_recording()
        {
            return Task::none();
        }

        // If photo timer is counting down, abort it
        if self.photo_timer_countdown.is_some() {
            return self.handle_abort_photo_timer();
        }

        if !self.can_accept_photo() {
            return Task::none();
        }

        self.haptic_tap();

        // Compatible Motion Photos accept the shutter on press, so both the
        // still snapshot and the centered motion window use the press instant.
        // Timer/flash and ordinary long-press quick-record keep their delays.
        if self.motion_photo_enabled()
            && matches!(
                self.config.photo_output_format,
                crate::config::PhotoOutputFormat::Jpeg | crate::config::PhotoOutputFormat::Avif
            )
            && self.photo_timer_setting == crate::app::state::PhotoTimerSetting::Off
            && !self.flash.enabled
        {
            self.quick_record = QuickRecordState::Idle;
            self.animate_capture_scale(0.82);
            return self.capture_photo_with_frame(self.current_frame.clone());
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
        if self.pending_close {
            return Task::none();
        }
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

                if !self.can_accept_photo() {
                    self.animate_capture_scale(1.0);
                    return Task::none();
                }

                // In Photo mode with timer set, start countdown
                // Keep button in pressed state until capture completes
                if self.mode == CameraMode::Photo
                    && self.photo_timer_setting != crate::app::state::PhotoTimerSetting::Off
                {
                    return self.start_photo_timer();
                }

                // Screen flash (front camera) or hardware flash (back camera)
                self.animate_capture_scale(1.0);
                if self.flash.enabled && !self.flash.active {
                    return self.start_preflash();
                }

                // No timer or flash — use the zero-shutter-lag frame
                self.capture_photo_with_frame(captured_frame)
            }
            QuickRecordState::Recording => {
                self.animate_capture_scale(1.0);
                // Long press ended: stop recording
                self.quick_record = QuickRecordState::Idle;

                // Stop recording (same as toggle off).
                self.request_recording_stop();
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
        if self.encoder_restart_required() {
            self.quick_record = crate::app::state::QuickRecordState::Idle;
            self.show_encoder_restart_warning();
            return Task::none();
        }
        use crate::app::state::QuickRecordState;
        // Only act if still in Pressed state (finger hasn't lifted)
        if self.pending_close
            || !matches!(&self.quick_record,
            QuickRecordState::Pressed { press_time, .. } if press_time.elapsed() >= std::time::Duration::from_millis(300))
        {
            return Task::none();
        }
        if !self.photo_jobs.is_empty() || !self.recording_jobs.is_empty() {
            debug!("Quick recording waits for pending photo/video encoding");
            self.quick_record = QuickRecordState::Idle;
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

        let config = self.appsrc_recording_config(format, output_path);
        self.start_appsrc_recording(config)
    }

    pub(crate) fn handle_start_recording_after_delay(&mut self) -> Task<cosmic::Action<Message>> {
        if self.encoder_restart_required() {
            self.show_encoder_restart_warning();
            return Task::none();
        }
        if self.pending_close || !self.recording_jobs.is_empty() {
            self.motion_photo_error =
                Some("Video capture cannot start during finalization or shutdown".into());
            return Task::none();
        }
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

        let config = self.appsrc_recording_config(format, output_path);
        self.start_appsrc_recording(config)
    }

    /// Snapshot the shared normal/quick-record settings before opening a pipeline.
    fn appsrc_recording_config(
        &self,
        format: &crate::backends::camera::types::CameraFormat,
        output_path: PathBuf,
    ) -> AppsrcRecordingConfig {
        // The active format may describe a full-resolution Bayer stream rather
        // than the viewfinder frames actually supplied to the encoder.
        let (width, height) = self
            .current_frame
            .as_ref()
            .map(|frame| (frame.width, frame.height))
            .unwrap_or((format.width, format.height));
        // Gate both the device and its native rate together: disabled audio
        // must never leave a rate that can accidentally constrain pulsesrc.
        let audio = self
            .available_audio_devices
            .get(self.current_audio_device_index)
            .filter(|_| self.config.record_audio);
        AppsrcRecordingConfig {
            width,
            height,
            framerate: format.framerate.map(|fps| fps.as_int()).unwrap_or(30),
            format: format.clone(),
            output_path,
            // Keep the orientation stable for the complete recording session.
            sensor_rotation: self.capture_media_rotation(),
            audio_device: audio.map(|device| device.node_name.clone()),
            audio_source_rate_hz: audio.map(|device| device.sample_rate).unwrap_or(0),
            selected_encoder: self
                .available_video_encoders
                .get(self.current_video_encoder_index)
                .cloned(),
            bitrate_kbps: self.config.bitrate_preset.bitrate_kbps(width, height),
        }
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
        if self.encoder_restart_required() {
            self.quick_record = crate::app::state::QuickRecordState::Idle;
            self.show_encoder_restart_warning();
            return Task::none();
        }
        use crate::backends::camera::types::PixelFormat;

        if self.motion_photo_capture.is_some() {
            self.quick_record = crate::app::state::QuickRecordState::Idle;
            self.motion_photo_error =
                Some("Wait for Motion Photo capture to complete before recording".into());
            return Task::none();
        }
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
        let mut capture_metadata = self.build_camera_metadata();
        capture_metadata.orientation =
            crate::pipelines::capture_metadata::CaptureOrientation::from_capture(
                sensor_rotation,
                mirror_horizontal,
            );

        // Determine pixel format for the appsrc pipeline
        let pixel_format = self
            .current_frame
            .as_ref()
            .map(|f| f.format)
            .unwrap_or_else(|| {
                // Fallback: parse from format string
                PixelFormat::from_gst_format(&format.pixel_format).unwrap_or(PixelFormat::I420)
            });

        // Recording always consumes decoded frames so filters can change live.
        info!(?pixel_format, rotation = %sensor_rotation, "Using appsrc recording pipeline");
        let (frame_tx, frame_rx) = tokio::sync::mpsc::channel(15);
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();

        let path_for_message = output_path.display().to_string();

        // When a file source is active (--preview-source), there is no capture
        // thread to forward frames. Spawn a task that pushes the static frame
        // into the recording channel at the configured framerate.
        if self.current_frame_is_file_source {
            if let Some(ref frame) = self.current_frame {
                let frame = Arc::clone(frame);
                let tx = frame_tx;
                let interval = std::time::Duration::from_millis(1000 / framerate.max(1) as u64);
                tokio::spawn(async move {
                    let mut ticker = tokio::time::interval(interval);
                    loop {
                        ticker.tick().await;
                        if tx.send(Arc::clone(&frame)).await.is_err() {
                            break;
                        }
                    }
                });
                // frame_tx moved into the task; create a dummy sender for the
                // recording bridge so the rest of the code doesn't panic.
                let (dummy_tx, _dummy_rx) = tokio::sync::mpsc::channel(1);
                if let Some(ref manager) = self.recording_bridge {
                    manager.set_recording_sender(Some(dummy_tx));
                }
            }
        } else {
            // Direct path: capture thread → appsrc (bypasses UI thread entirely)
            if let Some(ref manager) = self.recording_bridge {
                manager.set_recording_sender(Some(frame_tx));
            }
        }

        // Create audio levels handle upfront so the UI can read it immediately.
        // The recorder is created in the async task to avoid blocking the UI thread.
        let audio_levels: crate::pipelines::video::SharedAudioLevels = Default::default();
        self.recording_session_counter += 1;
        let session = self.recording_session_counter;
        let monitor = crate::pipelines::video::RecordingMonitor::new(
            selected_encoder
                .as_ref()
                .map(|encoder| encoder.element_name.clone())
                .unwrap_or_else(|| crate::fl!("recording-error-automatic")),
        );
        self.register_encoder_operation(
            crate::app::recording_error::EncoderOperation::Video(session),
            monitor.clone(),
        );
        self.recording_jobs.insert(session);
        self.recording = RecordingState::start(
            session,
            path_for_message.clone(),
            stop_tx,
            Some(audio_levels.clone()),
        );

        let recording_bridge = self.recording_bridge.clone();

        let live_filter = self.recording_filter_code.clone();
        let live_zoom = Arc::clone(&self.zoom_control);
        let record_audio = self.config.record_audio;

        let recording_task = Task::perform(
            async move {
                // Capture runtime handle — spawn_blocking doesn't propagate the
                // tokio context, but the appsrc pusher needs it for tokio::spawn.
                let rt_handle = tokio::runtime::Handle::current();

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
                                    capture_metadata: capture_metadata.clone(),
                                    audio_levels,
                                },
                                pixel_format,
                                live_filter_code: live_filter.clone(),
                                live_zoom: Arc::clone(&live_zoom),
                            }
                        };

                    let recorder = VideoRecorder::new_from_appsrc_monitored(
                        make_appsrc_config(audio_levels),
                        frame_rx,
                        monitor,
                    )?;
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
                if let Some(ref manager) = recording_bridge {
                    manager.set_recording_sender(None);
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

    /// Handle burst mode capture complete
    pub(crate) fn handle_burst_mode_complete(
        &mut self,
        result: Result<String, String>,
    ) -> Task<cosmic::Action<Message>> {
        // Successful HDR still publication is delivered by the queued
        // PhotoSaved message below. Keep its logical save alive until then.
        self.is_capturing = result.is_ok();
        if result.is_err() {
            self.hardware_photo_id = None;
        }

        // Start a short blur transition (200ms) after stream restarts
        // This keeps the last frame blurred until new frames arrive, then fades out smoothly
        // Don't disable UI since capture is complete
        self.blur_frame_rotation = self.current_frame_rotation;
        self.blur_frame_mirror = self.should_mirror_preview();
        self.blur_frame_zoom = self.current_zoom_level();
        self.transition_state.start_with_duration(200, false);

        // Restart the camera stream after HDR+ processing
        // The stream was stopped when processing began to free GPU resources.
        // Increment the restart counter to change the subscription ID and trigger restart.
        // Create a new cancel flag so the new subscription isn't immediately cancelled.
        if !self.pending_close && self.motion_photo_capture.is_none() {
            info!("Restarting camera stream after HDR+ processing");
            self.camera_stream_restart_counter = self.camera_stream_restart_counter.wrapping_add(1);
            self.camera_cancel_flag =
                std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        }
        if let Err(error) = &result {
            self.motion_photo_capture = None;
            self.motion_photo_error = Some(error.clone());
        }

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
        if self.encoder_restart_required() {
            self.show_encoder_restart_warning();
            return Task::none();
        }
        if self.pending_close {
            return Task::none();
        }
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

        if self.timelapse.is_finalising()
            || self.motion_photo_capture.is_some()
            || self.is_capturing
        {
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
        let live_zoom = Arc::clone(&self.zoom_control);
        self.recording_session_counter += 1;
        let operation = crate::app::recording_error::EncoderOperation::Timelapse(
            self.recording_session_counter,
        );
        let monitor = crate::pipelines::video::RecordingMonitor::new_waiting(
            encoder_info
                .as_ref()
                .map(|encoder| encoder.element_name.clone())
                .unwrap_or_else(|| crate::fl!("recording-error-automatic")),
        );
        self.register_encoder_operation(operation, monitor.clone());
        let mut capture_metadata = self.build_camera_metadata();
        capture_metadata.orientation =
            crate::pipelines::capture_metadata::CaptureOrientation::from_capture(
                self.capture_media_rotation(),
                self.should_mirror_captures(),
            );

        // Spawn the encoder task — it runs until the channel is closed
        let encoder_task = Task::perform(
            async move {
                let video_dir = crate::app::get_video_directory(&folder_name);
                if let Err(e) = std::fs::create_dir_all(&video_dir) {
                    let error = format!("Failed to create video directory: {e}");
                    monitor.report_error(error.clone());
                    // No encoder worker was created, so no native resources remain.
                    monitor.finish();
                    return Err(error);
                }

                let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S_%3f");
                let output_path = video_dir.join(format!("timelapse_{timestamp}.mp4"));

                crate::pipelines::video::timelapse::run_timelapse_encoder_monitored(
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
        for (operation, monitor) in &self.encoder_operations {
            if matches!(
                operation,
                crate::app::recording_error::EncoderOperation::Timelapse(_)
            ) {
                monitor.set_phase(crate::pipelines::video::RecordingPhase::Finalizing);
            }
        }
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
        if let Some(operation) = self.encoder_operations.keys().copied().find(|operation| {
            matches!(
                operation,
                crate::app::recording_error::EncoderOperation::Timelapse(_)
            )
        }) {
            self.finish_encoder_operation(operation, result.as_ref().err().cloned());
        }
        self.timelapse = TimelapseState::Idle;
        self.update_idle_inhibit();

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
        if let Some(&interval) = crate::config::TimelapseInterval::ALL.get(index) {
            self.config.timelapse_interval = interval;
            self.persist_config_async();
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
    // libcosmic enables zbus's Tokio backend. Its block_on must run outside
    // the app's executor, including when a synchronous update runs in a task.
    std::thread::spawn(|| zbus::block_on(portal_inhibit_async()))
        .join()
        .map_err(|_| "portal Inhibit worker panicked".to_string())?
}

async fn portal_inhibit_async() -> Result<crate::app::state::InhibitGuard, String> {
    use std::collections::HashMap;
    use zbus::Variant;

    // Inhibit flags bitmask: 4 = Suspend, 8 = Idle. Together these match the
    // previous logind "idle:sleep" request.
    const INHIBIT_SUSPEND: u32 = 4;
    const INHIBIT_IDLE: u32 = 8;

    let conn = zbus::Connection::session()
        .await
        .map_err(|e| format!("D-Bus session connection: {e}"))?;
    let options: HashMap<&str, Variant> =
        HashMap::from([("reason", Variant::from("Camera active"))]);
    let handle: zbus::OwnedObjectPath = conn
        .call_method(
            Some("org.freedesktop.portal.Desktop"),
            "/org/freedesktop/portal/desktop",
            Some("org.freedesktop.portal.Inhibit"),
            "Inhibit",
            // (window, flags, options): empty window means no parent surface.
            &("", INHIBIT_SUSPEND | INHIBIT_IDLE, options),
        )
        .await
        .map_err(|e| format!("portal Inhibit: {e}"))?
        .body()
        .deserialize()
        .map_err(|e| format!("portal Inhibit response: {e}"))?;
    Ok(crate::app::state::InhibitGuard {
        connection: conn,
        handle,
    })
}

#[cfg(test)]
mod portal_inhibit_tests {
    use std::{collections::HashMap, sync::mpsc, time::Duration};

    const REQUEST_PATH: &str = "/org/freedesktop/portal/desktop/request/camera/test";

    struct Inhibit;

    #[zbus::interface(name = "org.freedesktop.portal.Inhibit")]
    impl Inhibit {
        fn inhibit(
            &self,
            window: &str,
            flags: u32,
            options: HashMap<String, zbus::OwnedVariant>,
        ) -> zbus::OwnedObjectPath {
            assert_eq!(window, "");
            assert_eq!(flags, 12);
            assert_eq!(
                <&str>::try_from(&options["reason"]).unwrap(),
                "Camera active"
            );
            zbus::OwnedObjectPath::try_from(REQUEST_PATH).unwrap()
        }
    }

    struct Request(mpsc::Sender<()>);

    #[zbus::interface(name = "org.freedesktop.portal.Request")]
    impl Request {
        fn close(&self) {
            self.0.send(()).unwrap();
        }
    }

    #[test]
    fn portal_inhibit_keeps_connection_alive_and_closes_request_on_drop() {
        // Never claim the real desktop portal's name or inhibit the user's session.
        const CHILD: &str = "CAMERA_TEST_PRIVATE_PORTAL_BUS";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new("timeout")
                .args(["30s", "dbus-run-session", "--"])
                .arg(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "app::handlers::capture::portal_inhibit_tests::portal_inhibit_keeps_connection_alive_and_closes_request_on_drop",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .status()
                .expect("portal test requires timeout and dbus-run-session");
            assert!(status.success(), "isolated portal test failed: {status}");
            return;
        }

        let (closed, receive_close) = mpsc::channel();
        let service = zbus::block_on(async {
            zbus::connection::Builder::session()
                .name("org.freedesktop.portal.Desktop")
                .serve_at("/org/freedesktop/portal/desktop", Inhibit)
                .serve_at(REQUEST_PATH, Request(closed))
                .build()
                .await
                .unwrap()
        });

        let guard = super::portal_inhibit().unwrap();
        assert_eq!(guard.handle.as_str(), REQUEST_PATH);
        assert!(receive_close.try_recv().is_err());
        // This separate call proves connection work continues between block_on calls.
        zbus::block_on(guard.connection.peer_creds()).unwrap();
        drop(guard);
        receive_close.recv_timeout(Duration::from_secs(5)).unwrap();

        // App updates and guard cleanup can also run inside an existing Tokio
        // executor. Exercise both runtime flavors without touching the desktop.
        for runtime in [
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap(),
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap(),
        ] {
            let guard = super::portal_inhibit().unwrap();
            runtime.block_on(async {
                // The retained connection must keep processing traffic even
                // when no thread is inside zbus::block_on.
                tokio::time::timeout(
                    Duration::from_secs(5),
                    guard.connection.call_method(
                        Some("org.freedesktop.DBus"),
                        "/org/freedesktop/DBus",
                        Some("org.freedesktop.DBus"),
                        "ListNames",
                        &(),
                    ),
                )
                .await
                .expect("portal connection must progress between synchronous calls")
                .unwrap();
                drop(guard);
            });
            receive_close.recv_timeout(Duration::from_secs(5)).unwrap();

            runtime.block_on(async {
                let guard = super::portal_inhibit().unwrap();
                assert_eq!(guard.handle.as_str(), REQUEST_PATH);
                assert!(receive_close.try_recv().is_err());
                drop(guard);
            });
            receive_close.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        drop(service);
    }
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

    // Save output with optional crop, filter, and selected encoding format.
    let output_path = save_output(
        &merged,
        SaveOutputParams {
            output_dir: save_dir,
            crop_rect,
            encoding_format,
            camera_metadata,
            filter: Some(filter),
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
    camera_metadata: &CaptureMetadata,
    filter: crate::app::FilterType,
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

    // Reuse save_output with standard timestamp naming and collision-safe creation
    let path = save_output(
        &merged,
        SaveOutputParams {
            output_dir: save_dir.to_path_buf(),
            crop_rect,
            encoding_format,
            camera_metadata: camera_metadata.clone(),
            filter: Some(filter),
        },
    )
    .await?;

    info!(path = %path.display(), "First burst frame saved for comparison");
    Ok(path)
}

/// Run the full-resolution raw save first, then reuse the existing
/// ISP-processed preview save path when raw processing is unavailable.
async fn capture_with_preview_fallback<T, Raw, RawFuture, Preview, PreviewFuture>(
    raw_capture: Raw,
    preview_capture: Preview,
) -> Result<T, String>
where
    Raw: FnOnce() -> RawFuture,
    RawFuture: std::future::Future<Output = Result<T, String>>,
    Preview: FnOnce() -> PreviewFuture,
    PreviewFuture: std::future::Future<Output = Result<T, String>>,
{
    match raw_capture().await {
        Ok(value) => Ok(value),
        Err(raw_error) => {
            warn!(
                error = %raw_error,
                "Raw photo processing unavailable; falling back to ISP-processed preview frame"
            );
            preview_capture().await.map_err(|preview_error| {
                format!(
                    "Raw photo processing failed: {raw_error}; preview fallback failed: {preview_error}"
                )
            })
        }
    }
}

/// Await and retire this request before releasing the application's hardware lock.
pub(super) async fn wait_for_still_frame(
    request: crate::backends::camera::still_capture::StillRequest,
    timeout: std::time::Duration,
) -> Option<crate::backends::camera::types::CameraFrame> {
    request.wait(timeout).await
}

#[cfg(test)]
mod tests {
    use super::capture_with_preview_fallback;
    use crate::backends::camera::types::SensorRotation;
    use std::cell::Cell;
    use std::sync::Arc;

    fn live_model() -> crate::app::state::AppModel {
        let mut model = crate::app::state::AppModel::default();
        model.config.motion_photo = true;
        model.config.record_audio = false;
        model.config.burst_mode_setting = crate::config::BurstModeSetting::Off;
        model.current_frame = Some(Arc::new(crate::backends::camera::types::CameraFrame {
            width: 64,
            height: 48,
            data: crate::backends::camera::types::FrameData::from_copied(
                vec![80; 64 * 48 * 4].into(),
            ),
            format: crate::backends::camera::types::PixelFormat::RGBA,
            stride: 256,
            yuv_planes: None,
            captured_at: std::time::Instant::now(),
            sensor_timestamp_ns: None,
            libcamera_metadata: None,
        }));
        model
            .available_video_encoders
            .push(crate::media::encoders::video::EncoderInfo {
                element_name: "x264enc".into(),
                display_name: "H.264".into(),
                codec: crate::media::encoders::video::VideoCodec::H264,
                is_hardware: false,
                priority: 0,
            });
        model
    }

    fn raw_queue_model() -> crate::app::state::AppModel {
        let mut model = live_model();
        model.config.motion_photo = false;
        model.config.photo_output_format = crate::config::PhotoOutputFormat::Png;
        model.preview_fit_to_view = true;
        model.photo_aspect_ratio = crate::app::state::PhotoAspectRatio::Native;
        model
            .available_cameras
            .push(crate::backends::camera::types::CameraDevice {
                supports_multistream: true,
                ..Default::default()
            });
        model.config.save_folder_name = std::env::temp_dir()
            .join(format!("camera-raw-queue-{}", uuid::Uuid::new_v4()))
            .to_string_lossy()
            .into_owned();
        model
    }

    fn native_zoom_frame(zoom: f32) -> Arc<crate::backends::camera::types::CameraFrame> {
        use crate::backends::camera::types::FrameMetadata;
        use crate::backends::camera::zoom::{ScalerCropBounds, ZoomRect};

        let base = ZoomRect {
            x: 0,
            y: 0,
            width: 4000,
            height: 3000,
        };
        let min = ZoomRect {
            width: 100,
            height: 100,
            ..base
        };
        let bounds = ScalerCropBounds::new(min, base, base, base, false).unwrap();
        let mut frame = live_model().current_frame.unwrap().as_ref().clone();
        frame.libcamera_metadata = Some(FrameMetadata {
            scaler_zoom: bounds.applied(Some(bounds.crop(zoom))),
            ..Default::default()
        });
        Arc::new(frame)
    }

    #[tokio::test]
    async fn zsl_zoom_out_prefers_suitable_current_frame() {
        let mut model = live_model();
        model.config.motion_photo = false;
        model.zoom_level = 1.0;
        let current = native_zoom_frame(1.0);
        let zsl = native_zoom_frame(4.0);
        model.current_frame = Some(Arc::clone(&current));
        assert!(model.can_accept_photo());

        let _task = model.capture_photo_with_frame(Some(Arc::clone(&zsl)));

        assert_eq!(
            Arc::strong_count(&current),
            3,
            "save must own the suitable current frame"
        );
        assert_eq!(
            Arc::strong_count(&zsl),
            1,
            "old native 4x must not be exported at 1x"
        );
        assert_eq!(model.photo_jobs.len(), 1);
    }

    #[tokio::test]
    async fn zsl_zoom_out_without_suitable_frame_rejects_before_motion_or_photo_jobs() {
        let mut model = live_model();
        model.zoom_level = 1.0;
        model.current_frame = None;
        let next_id = model.next_photo_id;
        assert!(model.can_accept_photo());

        let task = model.capture_photo_with_frame(Some(native_zoom_frame(4.0)));

        assert!(cosmic::iced::runtime::task::into_stream(task).is_none());
        assert!(model.photo_jobs.is_empty());
        assert!(model.motion_photo_capture.is_none());
        assert!(model.motion_photo_error.is_none());
        assert!(model.hardware_photo_id.is_none());
        assert!(!model.still_capture.has_request());
        assert!(!model.is_capturing);
        assert_eq!(model.next_photo_id, next_id);
        assert_eq!(model.shutdown_pending_saves(), 0);
    }

    #[tokio::test]
    async fn zsl_compatible_frame_preserves_arc_identity_including_raw() {
        for raw in [false, true] {
            let mut model = live_model();
            model.config.motion_photo = false;
            model.zoom_level = if raw { 1.0 } else { 4.0 };
            let current = native_zoom_frame(1.0);
            let mut zsl = native_zoom_frame(if raw { 4.0 } else { 2.0 });
            if raw {
                Arc::get_mut(&mut zsl).unwrap().format =
                    crate::backends::camera::types::PixelFormat::BayerRGGB;
            }
            model.current_frame = Some(Arc::clone(&current));

            let _task = model.capture_photo_with_frame(Some(Arc::clone(&zsl)));

            assert_eq!(
                Arc::strong_count(&zsl),
                2,
                "save must retain the original ZSL Arc (raw={raw})"
            );
            assert_eq!(Arc::strong_count(&current), 2);
            assert_eq!(model.photo_jobs.len(), 1);
        }
    }

    async fn next_capture_message<S>(stream: &mut S) -> super::Message
    where
        S: futures::Stream<Item = cosmic::iced::runtime::Action<cosmic::Action<super::Message>>>
            + Unpin,
    {
        use futures::StreamExt;
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            while let Some(action) = stream.next().await {
                if let cosmic::iced::runtime::Action::Output(cosmic::Action::App(message)) = action
                {
                    if matches!(
                        message,
                        super::Message::ClearCaptureAnimation
                            | super::Message::PhotoEncodingWatchdog(_)
                    ) {
                        continue;
                    }
                    return message;
                }
            }
            panic!("capture task ended without a message");
        })
        .await
        .expect("capture task must make progress")
    }

    #[tokio::test]
    async fn timelapse_directory_failure_retires_encoder_operation() {
        use crate::app::recording_error::EncoderOperation;
        use crate::pipelines::video::RecordingPhase;
        use futures::StreamExt;

        let blocked_dir =
            std::env::temp_dir().join(format!("camera-timelapse-blocked-{}", uuid::Uuid::new_v4()));
        std::fs::write(&blocked_dir, b"not a directory").unwrap();
        let mut model = live_model();
        model.config.motion_photo = false;
        model.config.save_folder_name = blocked_dir.to_string_lossy().into_owned();
        let task = model.handle_toggle_timelapse();
        let operation = EncoderOperation::Timelapse(model.recording_session_counter);
        let monitor = model.encoder_operations[&operation].clone();
        assert_eq!(
            monitor.subscribe().borrow().phase,
            RecordingPhase::WaitingForFrames
        );
        let mut recipes =
            iced_futures::subscription::into_recipes(model.encoder_failure_subscriptions());
        let mut observer = recipes
            .pop()
            .unwrap()
            .stream(Box::pin(futures::stream::pending()));
        let mut stream = cosmic::iced::runtime::task::into_stream(task).unwrap();
        let message = next_capture_message(&mut stream).await;
        std::fs::remove_file(&blocked_dir).unwrap();
        let super::Message::TimelapseAssemblyComplete(Err(ref detail)) = message else {
            panic!("expected directory failure, got {message:?}");
        };
        assert!(detail.starts_with("Failed to create video directory: "));
        let detail = detail.clone();
        drop(model.update(message));

        assert!(
            model.encoder_operations.is_empty(),
            "no encoder worker owns this operation"
        );
        assert!(!model.timelapse.is_active());
        assert!(!model.shutdown_has_pending_media());
        assert_eq!(model.recording_failure.as_ref().unwrap().detail, detail);
        let status = monitor.subscribe().borrow().clone();
        assert_eq!(status.phase, RecordingPhase::Finished);
        assert_eq!(status.error.as_deref(), Some(detail.as_str()));
        let failure = tokio::time::timeout(std::time::Duration::from_secs(1), observer.next())
            .await
            .unwrap()
            .unwrap();
        assert!(
            matches!(failure, super::Message::EncoderFailed { operation: actual, .. } if actual == operation)
        );
        let finished = tokio::time::timeout(std::time::Duration::from_secs(1), observer.next())
            .await
            .unwrap()
            .unwrap();
        assert!(
            matches!(finished, super::Message::EncoderOperationFinished(actual) if actual == operation)
        );
    }

    #[tokio::test]
    async fn direct_and_release_timers_keep_countdown_and_tick_task() {
        use crate::app::state::PhotoTimerSetting::{Sec3, Sec5, Sec10};
        for setting in [Sec3, Sec5, Sec10] {
            for release in [false, true] {
                let mut model = raw_queue_model();
                model.photo_timer_setting = setting;
                let task = if release {
                    drop(model.handle_capture_button_pressed());
                    model.handle_capture_button_released()
                } else {
                    model.handle_capture()
                };
                assert_eq!(model.photo_timer_countdown, Some(setting.seconds()));
                assert!(model.photo_timer_tick_start.is_some());
                assert!(!model.flash.active && !model.is_capturing);
                let mut stream = cosmic::iced::runtime::task::into_stream(task).unwrap();
                assert!(matches!(
                    next_capture_message(&mut stream).await,
                    super::Message::PhotoTimerTick
                ));
                drop(model.handle_abort_photo_timer());
                assert!(
                    model.photo_timer_countdown.is_none() && model.photo_timer_tick_start.is_none()
                );
            }
        }
    }

    #[tokio::test]
    async fn all_preflash_callers_keep_screen_flash_delay_and_close_guard() {
        for location in [None, Some("front"), Some("back")] {
            for route in 0..3 {
                let mut model = raw_queue_model();
                model.available_cameras[0].camera_location = location.map(str::to_string);
                // No flash devices exist in this test: back cameras fall back to screen too.
                assert!(model.flash.hardware.devices.is_empty());
                model.flash.enabled = true;
                let task = match route {
                    0 => model.handle_capture(),
                    1 => {
                        drop(model.handle_capture_button_pressed());
                        model.handle_capture_button_released()
                    }
                    _ => {
                        model.photo_timer_countdown = Some(1);
                        model.photo_timer_tick_start = Some(std::time::Instant::now());
                        model.handle_photo_timer_tick()
                    }
                };
                assert!(model.flash.active && !model.is_capturing);
                assert!(model.photo_timer_countdown.is_none());
                let mut stream = cosmic::iced::runtime::task::into_stream(task).unwrap();
                assert!(matches!(
                    next_capture_message(&mut stream).await,
                    super::Message::FlashComplete
                ));
                model.pending_close = true;
                assert!(
                    cosmic::iced::runtime::task::into_stream(model.handle_flash_complete())
                        .is_none()
                );
                assert!(!model.is_capturing);
            }
        }
    }

    #[test]
    fn rejected_direct_photo_does_not_animate_a_disabled_shutter() {
        let mut model = raw_queue_model();
        model.capture_scale_to = 1.0;
        model.photo_btn_scale_to = 1.0;
        model.is_capturing = true;
        let _ = model.handle_capture();
        assert_eq!(model.capture_scale_to, 1.0);
        assert!(model.capture_anim_start.is_none());
        assert!(model.photo_btn_anim_start.is_none());
    }

    #[test]
    fn backpressured_quick_record_threshold_is_silent() {
        let mut model = raw_queue_model();
        model.photo_jobs.insert(
            1,
            crate::app::handlers::motion_photo::PhotoJob {
                live: None,
                still_path: None,
            },
        );
        model.quick_record = crate::app::state::QuickRecordState::Pressed {
            press_time: std::time::Instant::now() - std::time::Duration::from_secs(1),
            captured_frame: None,
        };
        model.motion_photo_error = Some("existing diagnostic".into());
        let _ = model.handle_quick_record_threshold();
        assert_eq!(
            model.motion_photo_error.as_deref(),
            Some("existing diagnostic")
        );
        assert!(!model.quick_record.is_pressed());
        assert!(!model.recording.is_recording());
    }

    #[test]
    fn release_rechecks_photo_admission_before_starting_timer() {
        let mut model = raw_queue_model();
        model.photo_timer_setting = crate::app::state::PhotoTimerSetting::Sec3;
        let _press = model.handle_capture_button_pressed();
        assert!(model.quick_record.is_pressed());
        model.is_capturing = true;
        assert!(
            cosmic::iced::runtime::task::into_stream(model.handle_capture_button_released())
                .is_none()
        );
        assert!(model.photo_timer_countdown.is_none());
        assert!(!model.quick_record.is_pressed());
    }

    #[test]
    fn full_queue_rejects_press_before_quick_record_or_timer_side_effects() {
        for motion in [false, true] {
            let mut model = raw_queue_model();
            model.config.motion_photo = motion;
            for id in 1..=8 {
                model.photo_jobs.insert(
                    id,
                    crate::app::handlers::motion_photo::PhotoJob {
                        live: None,
                        still_path: None,
                    },
                );
            }
            let next_id = model.next_photo_id;
            let task = model.handle_capture_button_pressed();
            assert!(cosmic::iced::runtime::task::into_stream(task).is_none());
            assert!(!model.quick_record.is_pressed());
            model.photo_timer_setting = crate::app::state::PhotoTimerSetting::Sec3;
            let task = model.handle_capture();
            assert!(cosmic::iced::runtime::task::into_stream(task).is_none());
            assert!(model.photo_timer_countdown.is_none());
            assert_eq!(model.next_photo_id, next_id);
            assert!(!model.still_capture.has_request());
            assert!(model.motion_photo_error.is_none());
        }
    }

    #[test]
    fn busy_photo_rejection_is_silent_and_does_not_allocate() {
        for motion in [false, true] {
            let mut model = raw_queue_model();
            model.config.motion_photo = motion;
            model.is_capturing = true;
            model.motion_photo_error = Some("existing diagnostic".into());
            let next_id = model.next_photo_id;
            assert!(cosmic::iced::runtime::task::into_stream(model.capture_photo()).is_none());
            assert_eq!(
                model.motion_photo_error.as_deref(),
                Some("existing diagnostic")
            );
            assert_eq!(model.next_photo_id, next_id);
            assert!(!model.still_capture.has_request());
        }
    }

    #[tokio::test]
    async fn hdr_raw_burst_uses_fresh_tickets_without_changing_exclusive_acquisition() {
        use futures::StreamExt;
        use std::task::Poll;
        let mut model = raw_queue_model();
        model.auto_detected_frame_count = 3;
        let mut stream =
            cosmic::iced::runtime::task::into_stream(model.capture_burst_mode_photo()).unwrap();
        let mut previous = None;
        for value in [40, 80, 120] {
            let ticket = tokio::time::timeout(
                std::time::Duration::from_secs(1),
                futures::future::poll_fn(|cx| {
                    assert!(
                        stream.poll_next_unpin(cx).is_pending(),
                        "burst must wait for the next RAW frame"
                    );
                    match model.still_capture.pending() {
                        Some(ticket) => Poll::Ready(ticket),
                        None => Poll::Pending,
                    }
                }),
            )
            .await
            .expect("burst must issue its next request");
            assert_ne!(Some(ticket), previous);
            let mut frame = model.current_frame.as_ref().unwrap().as_ref().clone();
            if let Some(old) = previous {
                assert!(!model.still_capture.publish(old, frame.clone()));
                assert_eq!(model.still_capture.pending(), Some(ticket));
            }
            frame.data =
                crate::backends::camera::types::FrameData::from_owned_vec(vec![value; 64 * 48 * 4]);
            assert!(model.still_capture.publish(ticket, frame));
            previous = Some(ticket);
            assert!(model.is_capturing);
        }
        let super::Message::BurstModeRawFramesCaptured(Ok(frames)) =
            next_capture_message(&mut stream).await
        else {
            panic!("HDR must acquire its entire ordered RAW burst");
        };
        assert_eq!(
            frames
                .iter()
                .map(|frame| frame.data.as_ref()[0])
                .collect::<Vec<_>>(),
            [40, 80, 120]
        );
        assert!(!model.still_capture.has_request());
        assert!(
            model.is_capturing,
            "HDR remains exclusive through processing"
        );
    }

    #[tokio::test]
    async fn expired_raw_producer_cannot_publish_into_next_acquisition() {
        use futures::FutureExt;

        // Cover both the stale-slot window before the UI receives Acquired and
        // the producer finishing after the UI has already admitted B.
        for complete_before_ui in [true, false] {
            let mut model = raw_queue_model();
            let save_dir = model.config.save_folder_name.clone();
            let mut a = cosmic::iced::runtime::task::into_stream(model.capture_photo()).unwrap();
            let capture = model.still_capture.clone();
            let old_frame = model.current_frame.as_ref().unwrap().as_ref().clone();
            let (observed_tx, observed_rx) = tokio::sync::oneshot::channel();
            let (complete_tx, complete_rx) = tokio::sync::oneshot::channel();
            let producer = tokio::spawn(async move {
                // Same snapshot/publication API used around native process_buffer.
                let ticket = capture.pending().unwrap();
                observed_tx.send(()).unwrap();
                complete_rx.await.unwrap();
                assert!(!capture.publish(ticket, old_frame));
            });
            observed_rx.await.unwrap();
            let acquired = next_capture_message(&mut a).await;
            assert!(matches!(acquired, super::Message::PhotoCaptureAcquired(1)));
            assert!(
                !model.still_capture.has_request(),
                "timeout retires A before UI unlock"
            );
            let mut completion = Some((complete_tx, producer));
            if complete_before_ui {
                let (tx, producer) = completion.take().unwrap();
                tx.send(()).unwrap();
                producer.await.unwrap();
            }
            let _ = model.update(acquired);
            let mut b = cosmic::iced::runtime::task::into_stream(model.capture_photo()).unwrap();
            assert_eq!(model.hardware_photo_id, Some(2));
            let ticket_b = model.still_capture.pending().unwrap();
            if let Some((tx, producer)) = completion {
                tx.send(()).unwrap();
                producer.await.unwrap();
            }
            assert_eq!(
                model.still_capture.pending(),
                Some(ticket_b),
                "A must not erase B's request"
            );
            assert!(
                next_capture_message(&mut b).now_or_never().is_none(),
                "B must not acquire the retired producer A's frame"
            );
            let mut frame_b = model.current_frame.as_ref().unwrap().as_ref().clone();
            frame_b.data =
                crate::backends::camera::types::FrameData::from_owned_vec(vec![160; 64 * 48 * 4]);
            assert!(model.still_capture.publish(ticket_b, frame_b));
            let acquired = next_capture_message(&mut b).await;
            assert!(matches!(acquired, super::Message::PhotoCaptureAcquired(2)));
            let _ = model.update(acquired);
            let saved = next_capture_message(&mut b).await;
            let super::Message::PhotoCaptureSaved(2, Ok(path)) = saved else {
                panic!("B must save its own frame");
            };
            assert_eq!(
                image::open(path).unwrap().to_rgb8().get_pixel(0, 0).0,
                [160; 3]
            );
            std::fs::remove_dir_all(save_dir).unwrap();
        }
    }

    #[tokio::test]
    async fn raw_acquisition_unlocks_shutter_before_save_completion() {
        let mut model = raw_queue_model();
        let save_dir = model.config.save_folder_name.clone();
        let mut stream = cosmic::iced::runtime::task::into_stream(model.capture_photo()).unwrap();
        assert!(model.still_capture.publish(
            model.still_capture.pending().unwrap(),
            model.current_frame.as_ref().unwrap().as_ref().clone()
        ));
        let first = next_capture_message(&mut stream).await;
        assert!(
            !matches!(
                first,
                super::Message::PhotoSaved(_) | super::Message::PhotoCaptureSaved(..)
            ),
            "RAW acquisition must produce its own event before saving finishes"
        );
        let _ = model.update(first);
        assert!(!model.is_capturing);
        assert!(model.hardware_photo_id.is_none());
        assert_eq!(model.photo_jobs.len(), 1);
        assert!(model.photo_jobs[&1].still_path.is_none());
        let _second = model.capture_photo();
        assert_eq!(model.hardware_photo_id, Some(2));
        assert_eq!(model.next_photo_id, 3);
        let saved = next_capture_message(&mut stream).await;
        let path = match saved {
            super::Message::PhotoCaptureSaved(1, ref result) => result.as_ref().unwrap().clone(),
            _ => panic!("RAW save must be identified"),
        };
        let _ = model.update(saved);
        assert_eq!(model.hardware_photo_id, Some(2));
        assert!(model.is_capturing);
        assert!(image::open(path).is_ok());
        std::fs::remove_dir_all(save_dir).unwrap();
    }

    #[tokio::test]
    async fn photo_and_recording_snapshot_keep_the_same_default_source_geometry() {
        use crate::app::state::{CameraMode, RecordingState};
        use crate::backends::camera::types::{CameraFormat, FrameData, Framerate};

        let mut model = raw_queue_model();
        model.available_cameras[0].supports_multistream = false;
        model.available_formats = [(160, 120), (120, 160)]
            .into_iter()
            .map(|(width, height)| CameraFormat {
                width,
                height,
                framerate: Some(Framerate::from_int(30)),
                pixel_format: "NV12".into(),
                hardware_accelerated: false,
            })
            .collect();
        model.screen_width = 600.0;
        model.screen_height = 800.0;
        let save_dir = model.config.save_folder_name.clone();
        let (stop_sender, mut stop_receiver) = tokio::sync::oneshot::channel();
        let mut stop_sender = Some(stop_sender);

        for mode in [CameraMode::Photo, CameraMode::Video] {
            model.mode = mode;
            model.select_format_from_cache(mode);
            if mode == CameraMode::Video {
                model.recording = RecordingState::start(
                    1,
                    "synthetic.mp4".into(),
                    stop_sender.take().unwrap(),
                    None,
                );
            }
            let format = model.active_format.as_ref().unwrap();
            let mut frame = model.current_frame.as_ref().unwrap().as_ref().clone();
            frame.width = format.width;
            frame.height = format.height;
            frame.stride = format.width * 4;
            frame.data =
                FrameData::from_copied(vec![80; (frame.stride * frame.height) as usize].into());
            model.current_frame = Some(Arc::new(frame));

            let mut stream =
                cosmic::iced::runtime::task::into_stream(model.handle_capture()).unwrap();
            let saved = next_capture_message(&mut stream).await;
            let super::Message::PhotoCaptureSaved(_, Ok(path)) = &saved else {
                panic!("photo must save in {mode:?}");
            };
            assert_eq!(
                image::open(path).unwrap().to_rgb8().dimensions(),
                (160, 120)
            );
            let _ = model.update(saved);
            assert_eq!(model.recording.is_recording(), mode == CameraMode::Video);
            assert!(model.photo_jobs.is_empty());
        }
        assert_eq!(
            stop_receiver.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        );
        std::fs::remove_dir_all(save_dir).unwrap();
    }

    #[tokio::test]
    async fn ordinary_photo_processing_waits_for_the_app_queue() {
        for raw in [false, true] {
            let mut model = raw_queue_model();
            model.available_cameras[0].supports_multistream = raw;
            let save_dir = model.config.save_folder_name.clone();
            let gate = model.photo_processing_gate.0.clone();
            let held = gate.acquire_owned().await.unwrap();
            let mut stream =
                cosmic::iced::runtime::task::into_stream(model.capture_photo()).unwrap();
            if raw {
                assert!(model.still_capture.publish(
                    model.still_capture.pending().unwrap(),
                    model.current_frame.as_ref().unwrap().as_ref().clone()
                ));
                let acquired = next_capture_message(&mut stream).await;
                assert!(matches!(acquired, super::Message::PhotoCaptureAcquired(1)));
                let _ = model.update(acquired);
            }
            assert!(
                tokio::time::timeout(
                    std::time::Duration::from_millis(100),
                    next_capture_message(&mut stream)
                )
                .await
                .is_err(),
                "ordinary processing must wait while the app worker is occupied (raw={raw})"
            );
            drop(held);
            let saved = next_capture_message(&mut stream).await;
            assert!(matches!(saved, super::Message::PhotoCaptureSaved(1, Ok(_))));
            let _ = model.update(saved);
            assert!(model.photo_jobs.is_empty());
            std::fs::remove_dir_all(save_dir).unwrap();
        }
    }

    #[tokio::test]
    async fn raw_timeout_finishes_acquisition_before_fallback_or_error() {
        for fallback in [false, true] {
            let mut model = raw_queue_model();
            let save_dir = model.config.save_folder_name.clone();
            if !fallback {
                model.current_frame = None;
            }
            let mut stream =
                cosmic::iced::runtime::task::into_stream(model.capture_photo()).unwrap();
            let acquired = next_capture_message(&mut stream).await;
            assert!(matches!(acquired, super::Message::PhotoCaptureAcquired(1)));
            let _ = model.update(acquired);
            assert!(!model.is_capturing);
            assert!(
                !model.still_capture.has_request(),
                "timed-out acquisition must not leave a request keeping the source alive"
            );
            assert_eq!(model.shutdown_pending_saves(), 1);
            let _second = model.capture_photo();
            let saved = next_capture_message(&mut stream).await;
            match &saved {
                super::Message::PhotoCaptureSaved(1, result) => {
                    assert_eq!(result.is_ok(), fallback)
                }
                _ => panic!("timeout completion must retain its ID"),
            }
            let _ = model.update(saved);
            assert_eq!(model.hardware_photo_id, Some(2));
            assert!(model.is_capturing);
            assert!(model.photo_jobs.is_empty());
            if fallback {
                std::fs::remove_dir_all(save_dir).unwrap();
            }
        }
    }

    #[tokio::test]
    async fn raw_queue_accepts_eight_owned_frames_and_rejects_ninth_without_request() {
        use futures::FutureExt;
        let mut model = raw_queue_model();
        let save_dir = model.config.save_folder_name.clone();
        let held = model
            .photo_processing_gate
            .0
            .clone()
            .acquire_owned()
            .await
            .unwrap();
        let mut streams = Vec::new();
        for id in 1..=8 {
            let mut stream =
                cosmic::iced::runtime::task::into_stream(model.capture_photo()).unwrap();
            assert_eq!(model.hardware_photo_id, Some(id));
            assert!(!model.can_accept_photo());
            assert!(model.still_capture.has_request());
            assert_eq!(model.shutdown_pending_saves(), id as usize);
            let mut frame = model.current_frame.as_ref().unwrap().as_ref().clone();
            frame.data = crate::backends::camera::types::FrameData::from_owned_vec(vec![
                id as u8
                    * 20;
                64 * 48 * 4
            ]);
            assert!(
                model
                    .still_capture
                    .publish(model.still_capture.pending().unwrap(), frame)
            );
            let acquired = next_capture_message(&mut stream).await;
            assert!(matches!(acquired, super::Message::PhotoCaptureAcquired(got) if got == id));
            let _ = model.update(acquired);
            assert_eq!(model.photo_jobs.len(), id as usize);
            assert_eq!(model.can_accept_photo(), id < 8);
            assert_eq!(model.shutdown_pending_saves(), id as usize);
            assert!(!model.is_capturing);
            assert!(next_capture_message(&mut stream).now_or_never().is_none());
            streams.push(stream);
        }
        assert!(cosmic::iced::runtime::task::into_stream(model.capture_photo()).is_none());
        assert_eq!(model.next_photo_id, 9);
        assert!(model.motion_photo_error.is_none());
        assert!(!model.still_capture.has_request());
        assert!(model.hardware_photo_id.is_none());
        // Shutter-time frame/encoding/crop/directory snapshots must survive
        // queued work and changes made while the worker is occupied.
        model.zoom_level = 3.0;
        model.selected_filter = crate::app::state::FilterType::Negative;
        model.photo_aspect_ratio = crate::app::state::PhotoAspectRatio::Ratio2x1;
        model.config.photo_output_format = crate::config::PhotoOutputFormat::Jpeg;
        model.config.save_folder_name = format!("{save_dir}/changed");
        let _close = model.update(super::Message::WindowClose);
        assert_eq!(model.shutdown_pending_saves(), 8);
        drop(held);
        let mut paths = std::collections::BTreeSet::new();
        for (index, stream) in streams.iter_mut().enumerate() {
            let saved = next_capture_message(stream).await;
            let path = match &saved {
                super::Message::PhotoCaptureSaved(id, Ok(path)) => {
                    assert_eq!(*id as usize, index + 1);
                    path.clone()
                }
                _ => panic!("queued RAW photo must save with its ID"),
            };
            let image = image::open(&path).unwrap().to_rgb8();
            assert_eq!(image.dimensions(), (64, 48));
            assert_eq!(image.get_pixel(0, 0).0, [(index as u8 + 1) * 20; 3]);
            assert!(path.ends_with(".png"));
            assert_eq!(
                std::path::Path::new(&path).parent().unwrap(),
                std::path::Path::new(&save_dir)
            );
            assert!(paths.insert(path));
            let _ = model.update(saved);
            assert_eq!(model.shutdown_pending_saves(), 7 - index);
        }
        assert_eq!(paths.len(), 8);
        std::fs::remove_dir_all(save_dir).unwrap();
    }

    #[tokio::test]
    async fn raw_acquisition_event_remains_accepted_during_shutdown() {
        let mut model = raw_queue_model();
        let save_dir = model.config.save_folder_name.clone();
        let held = model
            .photo_processing_gate
            .0
            .clone()
            .acquire_owned()
            .await
            .unwrap();
        let mut stream = cosmic::iced::runtime::task::into_stream(model.capture_photo()).unwrap();
        assert!(model.still_capture.publish(
            model.still_capture.pending().unwrap(),
            model.current_frame.as_ref().unwrap().as_ref().clone()
        ));
        let acquired = next_capture_message(&mut stream).await;
        let _close = model.update(super::Message::WindowClose);
        assert!(model.pending_close);
        assert_eq!(model.shutdown_pending_saves(), 1);
        let _ = model.update(acquired);
        assert_eq!(model.photo_jobs.len(), 1);
        assert!(!model.is_capturing);
        assert_eq!(model.shutdown_pending_saves(), 1);
        assert!(
            model
                .camera_cancel_flag
                .load(std::sync::atomic::Ordering::Acquire)
        );
        drop(held);
        let saved = next_capture_message(&mut stream).await;
        assert!(matches!(saved, super::Message::PhotoCaptureSaved(1, Ok(_))));
        let _finish = model.update(saved);
        assert_eq!(model.shutdown_pending_saves(), 0);
        std::fs::remove_dir_all(save_dir).unwrap();
    }

    #[tokio::test]
    async fn stale_raw_acquisition_and_completion_preserve_newer_clip_and_hardware() {
        use std::sync::atomic::Ordering;
        let mut model = raw_queue_model();
        let held = model
            .photo_processing_gate
            .0
            .clone()
            .acquire_owned()
            .await
            .unwrap();
        let mut stream = cosmic::iced::runtime::task::into_stream(model.capture_photo()).unwrap();
        // Start real clip state without requiring a hardware encoder or frames.
        model.start_motion_photo_capture().unwrap();
        let first_clip = model.motion_photo_capture.as_ref().unwrap().cancel.clone();
        assert!(model.still_capture.publish(
            model.still_capture.pending().unwrap(),
            model.current_frame.as_ref().unwrap().as_ref().clone()
        ));
        let acquired = next_capture_message(&mut stream).await;
        let _ = model.update(acquired);
        assert!(Arc::ptr_eq(
            &model.photo_jobs[&1].live.as_ref().unwrap().cancel,
            &first_clip
        ));
        assert!(model.motion_photo_capture.is_none());
        let _second = model.capture_photo();
        model.start_motion_photo_capture().unwrap();
        let second_clip = model.motion_photo_capture.as_ref().unwrap().cancel.clone();
        for stale in [1, 0, 999, 1] {
            let _ = model.update(super::Message::PhotoCaptureAcquired(stale));
            assert_eq!(model.hardware_photo_id, Some(2));
            assert!(model.is_capturing);
            assert!(model.still_capture.has_request());
            assert_eq!(model.photo_jobs.len(), 1);
            assert!(Arc::ptr_eq(
                &model.motion_photo_capture.as_ref().unwrap().cancel,
                &second_clip
            ));
        }
        let _ = model.update(super::Message::PhotoCaptureSaved(
            1,
            Err("first worker failed".into()),
        ));
        assert!(first_clip.load(Ordering::Acquire));
        assert!(!second_clip.load(Ordering::Acquire));
        assert_eq!(model.hardware_photo_id, Some(2));
        assert!(model.is_capturing);
        // A queued RAW completion cannot unlock an HDR acquisition either.
        model.burst_mode.stage = crate::app::state::BurstModeStage::Processing;
        let _ = model.update(super::Message::PhotoCaptureSaved(1, Ok("late.png".into())));
        assert_eq!(model.hardware_photo_id, Some(2));
        assert!(model.is_capturing);
        assert_eq!(
            model.burst_mode.stage,
            crate::app::state::BurstModeStage::Processing
        );
        drop(stream);
        drop(held);
    }

    #[tokio::test]
    async fn photo_queue_runs_fifo_one_worker_and_recovers_from_errors() {
        use futures::FutureExt;
        let model = raw_queue_model();
        let gate = model.photo_processing_gate.0.clone();
        let held = gate.clone().acquire_owned().await.unwrap();
        let (started, mut starts) = tokio::sync::mpsc::unbounded_channel();
        let mut jobs = Vec::new();
        let mut releases = Vec::new();
        for id in 1..=3 {
            let (release, wait) = tokio::sync::oneshot::channel();
            releases.push(release);
            let started = started.clone();
            let mut job = Box::pin(super::run_photo_save(gate.clone(), async move {
                started.send(id).unwrap();
                wait.await.unwrap();
                if id == 1 {
                    Err("controlled worker error".into())
                } else {
                    Ok(format!("{id}.png"))
                }
            }));
            assert!(job.as_mut().now_or_never().is_none());
            jobs.push(tokio::spawn(job));
        }
        assert!(starts.try_recv().is_err());
        drop(held);
        for (index, release) in releases.into_iter().enumerate() {
            assert_eq!(
                tokio::time::timeout(std::time::Duration::from_secs(5), starts.recv())
                    .await
                    .unwrap(),
                Some(index + 1)
            );
            assert!(
                starts.try_recv().is_err(),
                "only one processing worker may enter"
            );
            release.send(()).unwrap();
            let result = (&mut jobs[index]).await.unwrap();
            assert_eq!(result.is_err(), index == 0);
        }
        assert_eq!(gate.available_permits(), 1);
    }

    #[tokio::test]
    async fn photo_queue_cancellation_keeps_started_worker_exclusive() {
        use futures::FutureExt;
        let model = raw_queue_model();
        let gate = model.photo_processing_gate.0.clone();
        let (entered, ready) = tokio::sync::oneshot::channel();
        let (release, wait) = tokio::sync::oneshot::channel();
        let first = tokio::spawn(super::run_photo_save(gate.clone(), async move {
            entered.send(()).unwrap();
            wait.await.unwrap();
            Ok("first.png".into())
        }));
        tokio::time::timeout(std::time::Duration::from_secs(5), ready)
            .await
            .unwrap()
            .unwrap();
        first.abort();
        assert!(first.await.unwrap_err().is_cancelled());
        assert_eq!(
            gate.available_permits(),
            0,
            "blocking worker must still own the gate"
        );
        // Cancelling a waiter must remove it, without consuming a later permit.
        let mut cancelled = Box::pin(super::run_photo_save(gate.clone(), async {
            panic!("cancelled waiter started")
        }));
        assert!(cancelled.as_mut().now_or_never().is_none());
        drop(cancelled);
        let (entered, mut ready) = tokio::sync::oneshot::channel();
        let mut second = Box::pin(super::run_photo_save(gate.clone(), async move {
            entered.send(()).unwrap();
            Ok("second.png".into())
        }));
        assert!(second.as_mut().now_or_never().is_none());
        assert!(ready.try_recv().is_err());
        release.send(()).unwrap();
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_secs(5), second)
                .await
                .unwrap(),
            Ok("second.png".into())
        );
        ready.await.unwrap();
        assert_eq!(gate.available_permits(), 1);
        let panic =
            super::run_photo_save(gate.clone(), async { panic!("controlled worker panic") }).await;
        assert!(panic.unwrap_err().contains("Photo save task failed"));
        assert_eq!(gate.available_permits(), 1);
        assert_eq!(
            super::run_photo_save(gate, async { Ok("survived.png".into()) }).await,
            Ok("survived.png".into())
        );
    }

    #[tokio::test]
    async fn raw_processing_error_uses_shutter_preview_without_unlocking_next_acquisition() {
        let mut model = raw_queue_model();
        let save_dir = model.config.save_folder_name.clone();
        let held = model
            .photo_processing_gate
            .0
            .clone()
            .acquire_owned()
            .await
            .unwrap();
        let mut stream = cosmic::iced::runtime::task::into_stream(model.capture_photo()).unwrap();
        let mut bad_raw = model.current_frame.as_ref().unwrap().as_ref().clone();
        bad_raw.stride = 1; // Invalid layout fails the real post-processor.
        assert!(
            model
                .still_capture
                .publish(model.still_capture.pending().unwrap(), bad_raw)
        );
        let acquired = next_capture_message(&mut stream).await;
        let _ = model.update(acquired);
        let _second = model.capture_photo();
        model.current_frame = None;
        model.zoom_level = 4.0;
        model.config.save_folder_name = format!("{save_dir}/changed");
        drop(held);
        let saved = next_capture_message(&mut stream).await;
        let path = match &saved {
            super::Message::PhotoCaptureSaved(1, Ok(path)) => path.clone(),
            _ => panic!("RAW processing error must fall back to the shutter-time preview"),
        };
        let image = image::open(path).unwrap().to_rgb8();
        assert_eq!(image.dimensions(), (64, 48));
        assert_eq!(image.get_pixel(0, 0).0, [80; 3]);
        let _ = model.update(saved);
        assert_eq!(model.hardware_photo_id, Some(2));
        assert!(model.is_capturing);
        assert!(model.photo_jobs.is_empty());
        std::fs::remove_dir_all(save_dir).unwrap();
    }

    #[tokio::test]
    async fn raw_save_completion_does_not_release_a_later_hdr_shutter() {
        let mut model = raw_queue_model();
        let save_dir = model.config.save_folder_name.clone();
        let held = model
            .photo_processing_gate
            .0
            .clone()
            .acquire_owned()
            .await
            .unwrap();
        let mut stream = cosmic::iced::runtime::task::into_stream(model.capture_photo()).unwrap();
        assert!(model.still_capture.publish(
            model.still_capture.pending().unwrap(),
            model.current_frame.as_ref().unwrap().as_ref().clone()
        ));
        let acquired = next_capture_message(&mut stream).await;
        let _ = model.update(acquired);
        model.config.burst_mode_setting = crate::config::BurstModeSetting::Frames4;
        let _hdr = model.capture_photo();
        assert!(model.burst_mode.is_collecting_frames());
        assert_eq!(model.hardware_photo_id, Some(2));
        let stage = model.burst_mode.stage;
        drop(held);
        let saved = next_capture_message(&mut stream).await;
        assert!(matches!(saved, super::Message::PhotoCaptureSaved(1, Ok(_))));
        let _ = model.update(saved);
        assert!(model.is_capturing);
        assert_eq!(model.hardware_photo_id, Some(2));
        assert_eq!(model.burst_mode.stage, stage);
        let next_id = model.next_photo_id;
        assert!(cosmic::iced::runtime::task::into_stream(model.capture_photo()).is_none());
        assert_eq!(model.next_photo_id, next_id);
        std::fs::remove_dir_all(save_dir).unwrap();
    }

    #[test]
    fn raw_queue_generated_bayer_saves_via_native_gpu_after_acquisition() {
        let Some((_gpu_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu(
                "raw_queue_generated_bayer_saves_via_native_gpu_after_acquisition",
            );
            return;
        };
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let mut model = raw_queue_model();
                model.config.demosaic_method = crate::shaders::DemosaicMethod::Bilinear;
                let save_dir = model.config.save_folder_name.clone();
                let held = model
                    .photo_processing_gate
                    .0
                    .clone()
                    .acquire_owned()
                    .await
                    .unwrap();
                let mut stream =
                    cosmic::iced::runtime::task::into_stream(model.capture_photo()).unwrap();
                let mut raw = model.current_frame.as_ref().unwrap().as_ref().clone();
                raw.width = 32;
                raw.height = 24;
                raw.format = crate::backends::camera::types::PixelFormat::BayerRGGB;
                raw.stride = raw.width;
                raw.data = crate::backends::camera::types::FrameData::from_owned_vec(
                    (0..raw.width * raw.height)
                        .map(|index| 40 + (index % 100) as u8)
                        .collect(),
                );
                assert!(
                    model
                        .still_capture
                        .publish(model.still_capture.pending().unwrap(), raw)
                );
                let acquired = next_capture_message(&mut stream).await;
                assert!(matches!(acquired, super::Message::PhotoCaptureAcquired(1)));
                let _ = model.update(acquired);
                let _second = model.capture_photo();
                assert_eq!(model.hardware_photo_id, Some(2));
                // Distinct sensor dimensions ensure preview fallback cannot conceal
                // a failure in the real Bayer processing path.
                drop(held);
                let saved = next_capture_message(&mut stream).await;
                let path = match &saved {
                    super::Message::PhotoCaptureSaved(1, Ok(path)) => path.clone(),
                    _ => panic!("generated Bayer frame must save"),
                };
                let image = image::open(path).unwrap().to_rgb8();
                assert_eq!(image.dimensions(), (32, 24));
                assert!(
                    image
                        .pixels()
                        .any(|pixel| pixel.0 != image.get_pixel(0, 0).0)
                );
                let _ = model.update(saved);
                assert_eq!(model.hardware_photo_id, Some(2));
                assert!(model.is_capturing);
                std::fs::remove_dir_all(save_dir).unwrap();
            });
    }

    async fn warm_live_model(model: &mut crate::app::state::AppModel) {
        warm_live_model_with_cadence(model, std::time::Duration::from_millis(100)).await;
    }

    async fn warm_live_model_with_cadence(
        model: &mut crate::app::state::AppModel,
        cadence: std::time::Duration,
    ) {
        model.preview_fit_to_view = true;
        model.photo_aspect_ratio = crate::app::state::PhotoAspectRatio::Native;
        model.sync_motion_photo_recorder();
        model.warm_motion_photo_recorder(cadence).await;
    }

    #[tokio::test]
    async fn motion_photo_warmup_observes_ready_output_between_sparse_preview_frames() {
        let mut model = live_model();
        // The producer can be descheduled beyond the encoded freshness budget.
        // It must observe ready output without waiting for its next frame tick.
        warm_live_model_with_cadence(&mut model, std::time::Duration::from_millis(300)).await;
        assert!(
            model
                .motion_photo_recorder
                .as_ref()
                .unwrap()
                .pre_roll_ready()
        );
        model
            .motion_photo_recorder
            .take()
            .unwrap()
            .stop_and_wait()
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn motion_photo_encodes_preroll_preview_frames_even_during_hdr_processing() {
        let mut model = live_model();
        warm_live_model(&mut model).await;
        model.start_motion_photo_capture().unwrap();
        // HDR must not starve the independent Motion Photo frame stream.
        model.burst_mode.stage = crate::app::state::BurstModeStage::Processing;
        for _ in 0..26 {
            let mut frame = model.current_frame.as_ref().unwrap().as_ref().clone();
            frame.captured_at = std::time::Instant::now();
            let _ = model.handle_camera_frame(Arc::new(frame));
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        let job = model
            .motion_photo_capture
            .as_mut()
            .unwrap()
            .result
            .take()
            .unwrap();
        let clip = job.await.unwrap().unwrap();
        let bytes = std::fs::read(&clip.0).unwrap();
        assert_eq!(&bytes[4..8], b"ftyp");
        assert!(bytes.windows(4).any(|x| x == b"avc1"));
    }

    #[tokio::test]
    async fn slow_still_watchdog_reports_without_abandoning_accepted_media() {
        let mut model = live_model();
        model.config.motion_photo = false;
        let _task = model.capture_photo();
        let _warning = model.handle_photo_encoding_watchdog(1);
        assert!(
            model
                .motion_photo_error
                .as_ref()
                .unwrap()
                .contains("watchdog")
        );
        assert_eq!(
            model.photo_jobs.len(),
            1,
            "warning must not drop a slow but still active encoder"
        );
    }

    #[tokio::test]
    async fn hardware_still_completion_releases_acquisition_before_live_embedding() {
        let mut model = live_model();
        model.start_motion_photo_capture().unwrap();
        model.hardware_photo_id = Some(1);
        model.is_capturing = true;
        let _task = model.handle_photo_saved(Ok("accepted-hardware-still.jpg".into()));
        assert!(
            !model.is_capturing,
            "only scarce acquisition may block, not post-roll/embedding"
        );
        assert!(model.hardware_photo_id.is_none());
        assert_eq!(model.photo_jobs.len(), 1);
        assert!(model.motion_photo_capture.is_none());
        assert!(model.photo_jobs[&1].live.as_ref().unwrap().result.is_none());
    }

    #[test]
    fn shutter_metadata_uses_cached_controls_without_opening_the_camera() {
        let mut model = live_model();
        model.exposure_settings = Some(crate::app::exposure_picker::ExposureSettings {
            exposure_time: Some(100),
            gain: Some(20),
            iso: Some(200),
            ..Default::default()
        });
        let metadata = model.build_camera_metadata();
        assert_eq!(metadata.gain, Some(20));
        assert_eq!(metadata.iso, Some(200));
        assert_eq!(metadata.exposure_time, Some(0.01));
    }

    #[test]
    fn close_during_ordinary_encoding_keeps_process_alive_until_completion() {
        const CHILD: &str = "CAMERA_TEST_CLOSE_ORDINARY";
        if std::env::var_os(CHILD).is_some() {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            let _entered = runtime.enter();
            let mut model = live_model();
            model.config.motion_photo = false;
            let _pending_task = model.capture_photo();
            let _close_task = model.handle_window_close();
            assert!(model.pending_close);
            assert_eq!(model.photo_jobs.len(), 1);
            println!("CLOSE_DRAIN_STILL_ALIVE");
            return;
        }
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "app::handlers::capture::tests::close_during_ordinary_encoding_keeps_process_alive_until_completion", "--nocapture"])
            .env(CHILD, "1").output().unwrap();
        assert!(
            child.status.success(),
            "{}",
            String::from_utf8_lossy(&child.stderr)
        );
        assert!(
            String::from_utf8_lossy(&child.stdout).contains("CLOSE_DRAIN_STILL_ALIVE"),
            "close must not _exit while ordinary still encoding is pending"
        );
    }

    // Run exit-sensitive regressions in a child: _exit(0) is not a test pass.
    fn assert_shutdown_child(test: &str, marker: &str, scenario: impl FnOnce()) {
        const CHILD: &str = "CAMERA_TEST_SHUTDOWN_SCENARIO";
        if std::env::var(CHILD).as_deref() == Ok(test) {
            scenario();
            println!("{marker}");
            return;
        }
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture"])
            .env(CHILD, test)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&child.stdout);
        assert!(
            child.status.success() && stdout.contains(marker),
            "child must reach all assertions, not _exit early: status={}\nstdout={stdout}\nstderr={}",
            child.status,
            String::from_utf8_lossy(&child.stderr)
        );
    }

    fn shutdown_model() -> crate::app::state::AppModel {
        let mut model = live_model();
        model.config.motion_photo = false;
        model
    }

    #[tokio::test]
    async fn shutdown_progress_counts_logical_saves_and_tracks_completions() {
        let mut model = shutdown_model();
        pending_photo(&mut model, 1);
        model.hardware_photo_id = Some(2);
        model.is_capturing = true;
        model.recording_jobs.insert(7);
        let (stop, _stopped) = tokio::sync::oneshot::channel();
        model.recording = super::RecordingState::start(7, "video.mp4".into(), stop, None);
        let _ = model.update(super::Message::WindowClose);
        assert_eq!(model.shutdown_pending_saves(), 3);
        let _ = model.update(super::Message::PhotoSaved(Ok("hardware.jpg".into())));
        assert_eq!(model.shutdown_pending_saves(), 2);
        let _ = model.update(super::Message::RecordingStopped {
            session: 7,
            result: Err("video failed".into()),
        });
        assert_eq!(model.shutdown_pending_saves(), 1);
        let _ = model.update(super::Message::PhotoCaptureFinished(
            1,
            Ok("photo.jpg".into()),
        ));
        assert_eq!(model.shutdown_pending_saves(), 0);
        assert!(model.shutdown_waiting_for_recorder);
    }

    #[tokio::test]
    async fn shutdown_hides_preview_and_all_interactive_surfaces_immediately() {
        let mut model = shutdown_model();
        pending_photo(&mut model, 1);
        model.core.window.show_context = true;
        model.tools_menu_visible = true;
        model.format_picker_visible = true;
        model.exposure_picker_visible = true;
        model.color_picker_visible = true;
        model.motor_picker_visible = true;
        let frame = model.current_frame.clone().unwrap();
        let _ = model.update(super::Message::WindowClose);
        assert!(
            model.current_frame.is_none(),
            "never present a live/frozen frame during drain"
        );
        assert!(!model.core.window.show_context);
        assert!(!model.tools_menu_visible);
        assert!(!model.format_picker_visible);
        assert!(!model.exposure_picker_visible);
        assert!(!model.color_picker_visible);
        assert!(!model.motor_picker_visible);
        let _ = model.update(super::Message::CameraFrame(frame));
        assert!(
            model.current_frame.is_none(),
            "queued source frames must not repopulate preview"
        );
        assert_eq!(model.shutdown_pending_saves(), 1);
    }

    #[tokio::test]
    async fn shutdown_stops_frame_producer_after_postroll_without_cancelling_slow_still() {
        let mut model = live_model();
        warm_live_model(&mut model).await;
        let mut source_frame = model.current_frame.as_ref().unwrap().as_ref().clone();
        model.start_motion_photo_capture().unwrap();
        model.is_capturing = true;
        model.hardware_photo_id = Some(1);
        let _ = model.update(super::Message::WindowClose);
        assert_eq!(
            model.shutdown_pending_saves(),
            1,
            "live still/clip/acquisition are one save"
        );
        assert!(
            !model
                .camera_cancel_flag
                .load(std::sync::atomic::Ordering::Acquire)
        );
        for _ in 0..22 {
            source_frame.captured_at = std::time::Instant::now();
            let _ = model.update(super::Message::CameraFrame(Arc::new(source_frame.clone())));
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        // Simulate raw acquisition having delivered its frame while its still
        // encoder is slow. No additional camera frames are needed for encoding.
        model.still_capture.reset();
        let _ = model.update(super::Message::Noop);
        assert!(
            model
                .camera_cancel_flag
                .load(std::sync::atomic::Ordering::Acquire),
            "post-roll finished: do not process prebuffer frames while slow still drains"
        );
        assert!(model.current_frame.is_none());
        assert_eq!(model.shutdown_pending_saves(), 1);
        assert!(
            !model
                .motion_photo_capture
                .as_ref()
                .unwrap()
                .cancel
                .load(std::sync::atomic::Ordering::Acquire)
        );
        let clip = model
            .motion_photo_capture
            .as_mut()
            .unwrap()
            .result
            .take()
            .unwrap()
            .await
            .unwrap()
            .expect("accepted clip survives preview producer shutdown");
        let bytes = std::fs::read(&clip.0).unwrap();
        assert_eq!(&bytes[4..8], b"ftyp");
        assert!(bytes.windows(4).any(|x| x == b"avc1"));
        let _ = model.update(super::Message::PhotoSaved(Err(
            "synthetic still error".into()
        )));
        assert_eq!(model.shutdown_pending_saves(), 0);
        assert!(model.shutdown_waiting_for_recorder);
    }

    #[tokio::test]
    async fn shutdown_source_removal_fails_partial_hdr_but_drains_dispatched_photos() {
        assert_source_loss_drains_photos(super::Message::HotplugDeviceRemoved(vec![
            "video-test".into(),
        ]))
        .await;
    }

    #[tokio::test]
    async fn shutdown_permanent_source_failure_fails_partial_hdr_but_drains_dispatched_photos() {
        assert_source_loss_drains_photos(super::Message::CameraRecoveryFailed(
            "permanent source failure".into(),
        ))
        .await;
    }

    #[tokio::test]
    async fn shutdown_source_loss_preserves_hdr_with_all_frames_already_owned() {
        for processing in [false, true] {
            for source_loss in [
                super::Message::HotplugDeviceRemoved(vec!["video-test".into()]),
                super::Message::CameraRecoveryFailed("permanent source failure".into()),
            ] {
                let mut model = shutdown_model();
                model
                    .available_cameras
                    .push(crate::backends::camera::types::CameraDevice {
                        device_info: Some(crate::backends::camera::types::DeviceInfo {
                            path: "/dev/video-test".into(),
                            ..Default::default()
                        }),
                        ..Default::default()
                    });
                model.config.burst_mode_setting = crate::config::BurstModeSetting::Frames4;
                let _ = model.capture_photo();
                for _ in 0..4 {
                    model
                        .burst_mode
                        .add_frame(model.current_frame.clone().unwrap());
                }
                if processing {
                    model.burst_mode.take_frames();
                    model.burst_mode.stage = crate::app::state::BurstModeStage::Processing;
                }
                let stage = model.burst_mode.stage;
                let _ = model.update(super::Message::WindowClose);
                let _ = model.update(source_loss);
                assert_eq!(
                    model.burst_mode.stage, stage,
                    "owned HDR bytes must finish processing/publication"
                );
                assert!(model.is_capturing);
                assert_eq!(model.hardware_photo_id, Some(1));
                assert_eq!(model.shutdown_pending_saves(), 1);
                assert!(!model.shutdown_waiting_for_recorder);
            }
        }
    }

    async fn assert_source_loss_drains_photos(source_loss: super::Message) {
        use crate::backends::camera::types::{CameraDevice, DeviceInfo};
        use std::sync::atomic::Ordering;

        let mut model = live_model();
        model.available_cameras.push(CameraDevice {
            device_info: Some(DeviceInfo {
                path: "/dev/video-test".into(),
                ..Default::default()
            }),
            ..Default::default()
        });
        // An independent, already-dispatched Motion Photo/mux and still encoder.
        model.start_motion_photo_capture().unwrap();
        let mut live = model.motion_photo_capture.take().unwrap();
        let dispatched_cancel = live.cancel.clone();
        let (complete, completion) = tokio::sync::oneshot::channel();
        live.result = Some(tokio::spawn(async move { completion.await.unwrap() }));
        model.photo_jobs.insert(
            90,
            crate::app::handlers::motion_photo::PhotoJob {
                live: Some(live),
                still_path: None,
            },
        );
        pending_photo(&mut model, 91);
        model.config.burst_mode_setting = crate::config::BurstModeSetting::Frames4;
        let _shutter = model.capture_photo();
        let acquisition_cancel = model.motion_photo_capture.as_ref().unwrap().cancel.clone();
        assert!(model.burst_mode.is_collecting_frames());
        assert!(
            !model
                .burst_mode
                .add_frame(model.current_frame.clone().unwrap())
        );
        assert_eq!(model.burst_mode.frames_captured(), 1);
        let _ = model.update(super::Message::WindowClose);
        assert_eq!(model.shutdown_pending_saves(), 3);
        let _ = model.update(source_loss);
        assert!(
            !model.is_capturing,
            "lost source must release unfinished HDR acquisition"
        );
        assert!(model.hardware_photo_id.is_none());
        assert_eq!(
            model.burst_mode.stage,
            crate::app::state::BurstModeStage::Error
        );
        assert_eq!(model.burst_mode.frames_captured(), 0);
        assert!(model.motion_photo_error.is_some());
        assert!(model.motion_photo_capture.is_none());
        assert!(acquisition_cancel.load(Ordering::Acquire));
        assert!(model.camera_cancel_flag.load(Ordering::Acquire));
        assert_eq!(model.shutdown_pending_saves(), 2);
        assert!(!dispatched_cancel.load(Ordering::Acquire));
        assert!(
            !model.photo_jobs[&90]
                .live
                .as_ref()
                .unwrap()
                .result
                .as_ref()
                .unwrap()
                .is_finished()
        );
        assert!(!model.shutdown_waiting_for_recorder);
        // Resolve independent work normally; never deliver ShutdownRecorderStopped
        // here because that intentionally calls _exit in production.
        assert!(
            complete
                .send(Err("synthetic mux completion".into()))
                .is_ok()
        );
        let clip = model
            .photo_jobs
            .get_mut(&90)
            .unwrap()
            .live
            .as_mut()
            .unwrap()
            .result
            .take()
            .unwrap()
            .await
            .unwrap();
        assert!(clip.is_err());
        let _ = model.update(super::Message::PhotoCaptureFinished(
            90,
            Ok("live.jpg".into()),
        ));
        assert_eq!(model.shutdown_pending_saves(), 1);
        let _ = model.update(super::Message::PhotoCaptureFinished(
            91,
            Ok("still.jpg".into()),
        ));
        assert_eq!(model.shutdown_pending_saves(), 0);
        assert!(
            model.shutdown_waiting_for_recorder,
            "all remaining encodes drained: shutdown eligible"
        );
    }

    #[tokio::test]
    async fn shutdown_hdr_completion_keeps_save_count_until_publication_message() {
        let mut model = shutdown_model();
        model.hardware_photo_id = Some(3);
        model.is_capturing = true;
        model.burst_mode.stage = crate::app::state::BurstModeStage::Processing;
        let generation = model.camera_cancel_flag.clone();
        let _ = model.update(super::Message::WindowClose);
        assert_eq!(model.shutdown_pending_saves(), 1);
        let _ = model.update(super::Message::BurstModeComplete(Ok("hdr.jpg".into())));
        assert_eq!(
            model.shutdown_pending_saves(),
            1,
            "HDR complete still queues PhotoSaved; native close must not race publication"
        );
        assert!(!model.shutdown_waiting_for_recorder);
        assert!(Arc::ptr_eq(&generation, &model.camera_cancel_flag));
        assert!(
            model
                .camera_cancel_flag
                .load(std::sync::atomic::Ordering::Acquire)
        );
        let _ = model.update(super::Message::PhotoSaved(Ok("hdr.jpg".into())));
        assert_eq!(model.shutdown_pending_saves(), 0);
        assert_eq!(model.last_media_path.as_deref(), Some("hdr.jpg"));
        assert!(model.shutdown_waiting_for_recorder);
    }

    fn pending_photo(model: &mut crate::app::state::AppModel, id: u64) {
        model.photo_jobs.insert(
            id,
            crate::app::handlers::motion_photo::PhotoJob {
                live: None,
                still_path: None,
            },
        );
    }

    fn assert_drain_blocked(model: &mut crate::app::state::AppModel) {
        assert!(model.pending_close);
        assert!(model.shutdown_has_pending_media());
        assert!(!model.shutdown_waiting_for_recorder);
        assert!(cosmic::iced::runtime::task::into_stream(model.maybe_finish_shutdown()).is_none());
        let _ = model.update(super::Message::ShutdownDrainWatchdog);
        assert!(
            model
                .motion_photo_error
                .as_deref()
                .unwrap()
                .contains("watchdog")
        );
        assert!(
            model.shutdown_has_pending_media(),
            "watchdog must not abandon jobs"
        );
    }

    async fn assert_recorder_drain_scheduled(task: cosmic::Task<cosmic::Action<super::Message>>) {
        use futures::StreamExt;
        let mut stream = cosmic::iced::runtime::task::into_stream(task).unwrap();
        let drain = async {
            while let Some(action) = stream.next().await {
                if matches!(
                    action,
                    cosmic::iced::runtime::Action::Output(cosmic::Action::App(
                        super::Message::ShutdownRecorderStopped(Ok(()))
                    ))
                ) {
                    return;
                }
            }
            panic!("missing shutdown recorder drain completion");
        };
        tokio::time::timeout(std::time::Duration::from_secs(1), drain)
            .await
            .unwrap();
        // Do not dispatch this terminal message: it legitimately calls _exit.
    }

    #[tokio::test]
    async fn shutdown_video_stop_then_close_waits_for_finalization() {
        for result in [
            Ok("video.mp4".into()),
            Err("synthetic video failure".into()),
        ] {
            let mut model = shutdown_model();
            let (stop, mut stopped) = tokio::sync::oneshot::channel();
            model.recording = super::RecordingState::start(7, "video.mp4".into(), stop, None);
            model.recording_jobs.insert(7);
            let _ = model.update(super::Message::ToggleRecording);
            assert_eq!(stopped.try_recv(), Ok(()));
            assert!(!model.recording.is_recording());
            assert!(model.recording_jobs.contains(&7));
            let _ = model.update(super::Message::WindowClose);
            assert_drain_blocked(&mut model);
            let task = model.update(super::Message::RecordingStopped { session: 7, result });
            assert!(model.recording_jobs.is_empty());
            assert!(!model.shutdown_has_pending_media());
            assert!(model.shutdown_waiting_for_recorder);
            assert_recorder_drain_scheduled(task).await;
        }
    }

    #[tokio::test]
    async fn accepted_video_stop_starts_deadline_before_worker_acknowledges() {
        use crate::app::recording_error::EncoderOperation;
        use crate::pipelines::video::{RecordingMonitor, RecordingPhase};

        for message in [
            super::Message::ToggleRecording,
            super::Message::CaptureButtonReleased,
            super::Message::WindowClose,
            super::Message::SetMode(super::CameraMode::Photo),
        ] {
            let mut model = shutdown_model();
            model.mode = super::CameraMode::Video;
            if matches!(message, super::Message::CaptureButtonReleased) {
                model.mode = super::CameraMode::Photo;
                model.quick_record = crate::app::state::QuickRecordState::Recording;
            }
            let (stop, mut stopped) = tokio::sync::oneshot::channel();
            model.recording = super::RecordingState::start(7, "video.mp4".into(), stop, None);
            model.recording_jobs.insert(7);
            let monitor = RecordingMonitor::new("synthetic".into());
            monitor.set_phase(RecordingPhase::Recording);
            model.register_encoder_operation(EncoderOperation::Video(7), monitor.clone());

            let _ = model.update(message);

            assert_eq!(stopped.try_recv(), Ok(()));
            assert_eq!(
                monitor.subscribe().borrow().phase,
                RecordingPhase::Finalizing,
                "Stop must arm a deadline without depending on the worker or new input"
            );
            assert!(
                model.recording_jobs.contains(&7),
                "Stop does not complete cleanup"
            );
            assert!(
                model
                    .encoder_operations
                    .contains_key(&EncoderOperation::Video(7))
            );
        }
    }

    #[tokio::test]
    async fn shutdown_close_signals_active_video_without_dropping_finalization() {
        let mut model = shutdown_model();
        let (stop, mut stopped) = tokio::sync::oneshot::channel();
        model.recording = super::RecordingState::start(7, "video.mp4".into(), stop, None);
        model.recording_jobs.insert(7);
        let _ = model.update(super::Message::WindowClose);
        assert_eq!(stopped.try_recv(), Ok(()));
        assert!(model.recording.is_recording());
        assert!(model.recording_jobs.contains(&7));
        assert_drain_blocked(&mut model);
        let _ = model.update(super::Message::WindowClose);
        assert_drain_blocked(&mut model);
    }

    #[tokio::test]
    async fn shutdown_timelapse_close_and_stop_then_close_preserve_finalization() {
        for stop_first in [false, true] {
            let mut model = shutdown_model();
            let (frame_sender, mut frames) = tokio::sync::mpsc::unbounded_channel();
            let start_time = std::time::Instant::now();
            model.timelapse = super::TimelapseState::Running {
                start_time,
                shots_taken: 2,
                interval_ms: 1000,
                frame_sender,
            };
            if stop_first {
                let _ = model.update(super::Message::ToggleTimelapse);
            }
            let _ = model.update(super::Message::WindowClose);
            assert!(
                matches!(model.timelapse, super::TimelapseState::Finalising {
                start_time: actual, shots_taken: 2,
            } if actual == start_time)
            );
            assert!(matches!(
                frames.try_recv(),
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
            ));
            assert_drain_blocked(&mut model);
            let _ = model.update(super::Message::TimelapseTick);
            let _ = model.update(super::Message::WindowClose);
            assert!(model.timelapse.is_finalising());
            assert_drain_blocked(&mut model);
        }
    }

    #[tokio::test]
    async fn shutdown_combined_ordinary_photo_video_waits_in_either_completion_order() {
        for photo_first in [false, true] {
            let mut model = shutdown_model();
            pending_photo(&mut model, 1);
            model.recording_jobs.insert(7);
            let _ = model.update(super::Message::WindowClose);
            assert_drain_blocked(&mut model);
            let photo = super::Message::PhotoCaptureSaved(1, Ok("photo.jpg".into()));
            let video = super::Message::RecordingStopped {
                session: 7,
                result: Ok("video.mp4".into()),
            };
            let (first, last) = if photo_first {
                (photo, video)
            } else {
                (video, photo)
            };
            let _ = model.update(first);
            assert_eq!(model.photo_jobs.len(), usize::from(!photo_first));
            assert_eq!(model.recording_jobs.len(), usize::from(photo_first));
            assert_drain_blocked(&mut model);
            // Extra accepted-photo sentinel keeps terminal transition safe to inspect.
            pending_photo(&mut model, 2);
            let _ = model.update(last);
            assert_eq!(model.photo_jobs.len(), 1);
            assert!(model.photo_jobs.contains_key(&2));
            assert!(model.recording_jobs.is_empty());
            assert_drain_blocked(&mut model);
            let task = model.update(super::Message::PhotoCaptureFinished(
                2,
                Err("synthetic sentinel completion".into()),
            ));
            assert!(!model.shutdown_has_pending_media());
            assert!(model.shutdown_waiting_for_recorder);
            assert_recorder_drain_scheduled(task).await;
        }
    }

    #[test]
    fn shutdown_timelapse_failed_completion_waits_for_pending_video() {
        assert_shutdown_child(
            "app::handlers::capture::tests::shutdown_timelapse_failed_completion_waits_for_pending_video",
            "FAILED_TIMELAPSE_STILL_DRAINING",
            || {
                let mut model = shutdown_model();
                model.recording_jobs.insert(7);
                model.timelapse = super::TimelapseState::Finalising {
                    start_time: std::time::Instant::now(),
                    shots_taken: 2,
                };
                let _ = model.update(super::Message::WindowClose);
                let _ = model.update(super::Message::TimelapseAssemblyComplete(Err(
                    "synthetic timelapse failure".into(),
                )));
                assert!(!model.timelapse.is_active());
                assert!(model.recording_jobs.contains(&7));
                assert_drain_blocked(&mut model);
            },
        );
    }

    #[tokio::test]
    async fn motion_photo_press_respects_timer_flash_and_incompatible_format() {
        for delayed in 0..3 {
            let mut model = live_model();
            match delayed {
                0 => model.photo_timer_setting = crate::app::state::PhotoTimerSetting::Sec3,
                1 => model.flash.enabled = true,
                _ => model.config.photo_output_format = crate::config::PhotoOutputFormat::Png,
            }
            let _ = model.handle_capture_button_pressed();
            assert!(model.photo_jobs.is_empty());
            assert!(model.quick_record.is_pressed());
        }
    }

    #[test]
    fn shutdown_timelapse_completion_waits_for_pending_ordinary_photo() {
        assert_shutdown_child(
            "app::handlers::capture::tests::shutdown_timelapse_completion_waits_for_pending_ordinary_photo",
            "TIMELAPSE_COMPLETION_STILL_DRAINING",
            || {
                let mut model = live_model();
                model.config.motion_photo = false;
                model.photo_jobs.insert(
                    1,
                    crate::app::handlers::motion_photo::PhotoJob {
                        live: None,
                        still_path: None,
                    },
                );
                model.timelapse = crate::app::state::TimelapseState::Finalising {
                    start_time: std::time::Instant::now(),
                    shots_taken: 2,
                };
                let _ = model.update(super::Message::WindowClose);
                let _ = model.update(super::Message::TimelapseAssemblyComplete(Ok(
                    "timelapse.mp4".into(),
                )));
                assert!(!model.timelapse.is_active());
                assert_eq!(model.photo_jobs.len(), 1);
                assert_eq!(model.last_media_path.as_deref(), Some("timelapse.mp4"));
                assert!(!model.shutdown_waiting_for_recorder);
                assert!(model.shutdown_has_pending_media());
            },
        );
    }

    #[tokio::test]
    async fn motion_photo_press_dispatches_once_and_close_before_release_drains() {
        let mut model = live_model();
        let press = model.handle_capture_button_pressed();
        assert_eq!(
            model.photo_jobs.len(),
            1,
            "accepted Motion Photo starts on press, not release"
        );
        assert!(cosmic::iced::runtime::task::into_stream(press).is_some());
        let _ = model.update(super::Message::WindowClose);
        assert!(model.pending_close);
        assert!(model.shutdown_has_pending_media());
        assert!(!model.shutdown_waiting_for_recorder);
        let _ = model.update(super::Message::CaptureButtonReleased);
        assert_eq!(
            model.photo_jobs.len(),
            1,
            "release must not duplicate the press shutter"
        );
    }

    #[tokio::test]
    async fn motion_photo_press_release_does_not_dispatch_a_second_shutter() {
        let mut model = live_model();
        let _ = model.handle_capture_button_pressed();
        assert_eq!(model.photo_jobs.len(), 1);
        let _ = model.handle_capture_button_released();
        assert_eq!(model.photo_jobs.len(), 1);
        assert!(matches!(
            model.quick_record,
            crate::app::state::QuickRecordState::Idle
        ));
    }

    #[tokio::test]
    async fn stale_quick_record_timer_cannot_promote_a_new_press() {
        let mut model = live_model();
        model.config.motion_photo = false;
        let _ = model.handle_capture_button_pressed();
        let _ = model.handle_capture_button_released();
        let _ = model.handle_capture_button_pressed();
        let _ = model.handle_quick_record_threshold();
        assert!(
            model.quick_record.is_pressed(),
            "old press timer cannot promote a fresh <300ms press"
        );
    }

    #[tokio::test]
    async fn three_rapid_motion_photos_save_distinct_centered_windows_while_close_drains() {
        use futures::StreamExt;
        let root = std::env::temp_dir().join(format!("three-live-{}", uuid::Uuid::new_v4()));
        let mut model = live_model();
        model.config.save_folder_name = root.display().to_string();
        let mut red = model.current_frame.as_ref().unwrap().as_ref().clone();
        red.data = crate::backends::camera::types::FrameData::from_copied(
            [220u8, 10, 10, 255].repeat(64 * 48).into(),
        );
        model.current_frame = Some(Arc::new(red));
        warm_live_model(&mut model).await;
        let mut streams = futures::stream::SelectAll::new();
        for _ in 0..3 {
            streams.push(
                cosmic::iced::runtime::task::into_stream(model.handle_capture_button_pressed())
                    .unwrap(),
            );
            let _ = model.handle_capture_button_released();
            let mut frame = model.current_frame.as_ref().unwrap().as_ref().clone();
            frame.captured_at = std::time::Instant::now();
            frame.data = crate::backends::camera::types::FrameData::from_copied(
                [10u8, 10, 220, 255].repeat(64 * 48).into(),
            );
            model.current_frame = Some(Arc::new(frame));
            model.forward_motion_photo_frame(&model.current_frame.clone().unwrap());
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        assert_eq!(model.photo_jobs.len(), 3);
        let source_frame = model.current_frame.as_ref().unwrap().as_ref().clone();
        let _ = model.update(super::Message::WindowClose);
        assert_eq!(model.shutdown_pending_saves(), 3);
        assert!(model.pending_close);
        assert!(
            model.motion_photo_recorder.is_some(),
            "close must preserve accepted post-roll"
        );
        let mut completed = Vec::new();
        let mut tick = tokio::time::interval(std::time::Duration::from_millis(100));
        let drain = async {
            while !model.photo_jobs.is_empty() {
                tokio::select! {
                    _ = tick.tick() => {
                        let mut frame = source_frame.clone();
                        frame.captured_at = std::time::Instant::now();
                        let _ = model.update(super::Message::CameraFrame(Arc::new(frame)));
                        assert!(model.current_frame.is_none());
                    }
                    action = streams.next() => {
                        if let Some(cosmic::iced::runtime::Action::Output(cosmic::Action::App(message))) = action {
                            let task = match message {
                                super::Message::PhotoCaptureSaved(id, result) => model.handle_photo_capture_saved(id, result),
                                super::Message::PhotoCaptureFinished(id, result) => {
                                    completed.push((id, result.as_ref().unwrap().clone()));
                                    model.handle_photo_capture_finished(id, result)
                                }
                                _ => cosmic::Task::none(),
                            };
                            assert_eq!(model.shutdown_pending_saves(), model.photo_jobs.len(),
                                "still completion and embedding must not double-count a Motion Photo");
                            if let Some(stream) = cosmic::iced::runtime::task::into_stream(task) { streams.push(stream); }
                        }
                    }
                }
            }
        };
        tokio::time::timeout(std::time::Duration::from_secs(15), drain)
            .await
            .unwrap();
        completed.sort_by_key(|(id, _)| *id);
        assert_eq!(
            completed.iter().map(|p| p.0).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        let paths: std::collections::BTreeSet<_> = completed.iter().map(|p| p.1.clone()).collect();
        assert_eq!(paths.len(), 3);
        for (id, path) in &completed {
            let bytes = std::fs::read(path).unwrap();
            let xmp = String::from_utf8_lossy(&bytes);
            assert!(xmp.contains("MotionPhotoPresentationTimestampUs=\"-1\""));
            let item = xmp
                .split("<Container:Item ")
                .find(|s| {
                    s.starts_with("Item:Mime=\"video/mp4\"")
                        || (s
                            .split("/>")
                            .next()
                            .unwrap()
                            .contains("Item:Semantic=\"MotionPhoto\""))
                })
                .unwrap();
            let length: usize = item
                .split("Item:Length=\"")
                .nth(1)
                .unwrap()
                .split('"')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            let extracted = root.join(format!("extracted-{id}.mp4"));
            std::fs::write(&extracted, &bytes[bytes.len() - length..]).unwrap();
            let decode = std::process::Command::new("ffmpeg")
                .args(["-v", "error", "-xerror", "-i"])
                .arg(&extracted)
                .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
                .output()
                .unwrap();
            assert!(
                decode.status.success(),
                "{}",
                String::from_utf8_lossy(&decode.stderr)
            );
            let frames: Vec<_> = decode
                .stdout
                .as_chunks::<{ 64 * 48 * 3 }>()
                .0
                .iter()
                .collect();
            assert!(
                frames.len() >= 23,
                "10fps synthetic source should span the nominal 2.5s window, got {} frames",
                frames.len()
            );
            let probe = std::process::Command::new("ffprobe")
                .args([
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                ])
                .arg(&extracted)
                .output()
                .unwrap();
            let duration: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap();
            let seconds: f64 = duration["format"]["duration"]
                .as_str()
                .unwrap()
                .parse()
                .unwrap();
            assert!(
                (2.3..=2.65).contains(&seconds),
                "capture {id} duration={seconds}"
            );
            assert!(frames.first().unwrap()[0] > 180);
            assert!(frames.last().unwrap()[2] > 180);
            let boundary =
                frames.iter().position(|f| f[2] > 180).unwrap() as f64 / frames.len() as f64;
            assert!(
                (0.32..0.65).contains(&boundary),
                "warmed shutter {id} not near midpoint: red/blue boundary {boundary}"
            );
            eprintln!(
                "capture {id}: {} decoded frames, BEFORE/AFTER boundary {boundary}, {}",
                frames.len(),
                path
            );
        }
        assert_eq!(model.last_published_photo_id, 3);
        assert_eq!(
            model.last_media_path.as_deref(),
            Some(completed[2].1.as_str())
        );
        if let Some(recorder) = model.motion_photo_recorder.take() {
            recorder.stop_and_wait().await.unwrap();
        }
        assert!(!model.shutdown_has_pending_media());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn three_rapid_ordinary_shutters_are_all_dispatched() {
        let mut model = live_model();
        for _ in 0..3 {
            let task = model.capture_photo();
            assert!(
                cosmic::iced::runtime::task::into_stream(task).is_some(),
                "ordinary shutter must not be dropped while prior post-roll/encoding is pending"
            );
        }
    }

    #[tokio::test]
    async fn motion_photo_top_bar_disable_saves_an_ordinary_still() {
        let mut model = live_model();
        let _ = model.update(crate::app::Message::ToggleMotionPhotoCapture);
        let _ = model.capture_photo();
        assert!(!model.config.motion_photo);
        assert_eq!(model.photo_jobs.len(), 1);
        assert!(!model.is_capturing);
        assert!(model.photo_jobs[&1].live.is_none());
        assert!(model.motion_photo_error.is_none());
    }

    #[test]
    fn flash_toggle_schedules_transient_confirmation() {
        let mut model = crate::app::state::AppModel::default();
        let task = model.handle_toggle_flash();
        assert!(model.flash.enabled);
        assert!(
            cosmic::iced::runtime::task::into_stream(task).is_some(),
            "successful toggle must schedule confirmation dismissal"
        );
    }

    #[tokio::test]
    async fn motion_photo_still_completion_waits_for_embedding_before_releasing_capture() {
        let mut model = live_model();
        let _ = model.capture_photo();
        let _ = model.handle_photo_capture_saved(1, Ok("still.jpg".into()));
        assert_eq!(
            model.photo_jobs.len(),
            1,
            "job is retained through embedding"
        );
        assert!(
            !model.is_capturing,
            "post-roll must not block another ordinary shutter"
        );
        assert!(model.photo_jobs[&1].live.as_ref().unwrap().result.is_none());
    }

    #[tokio::test]
    async fn motion_photo_hdr_processing_keeps_preview_source_running() {
        let mut model = live_model();
        model.start_motion_photo_capture().unwrap();
        // Insufficient burst frames ends before the expensive HDR worker starts.
        let _ = model.handle_burst_mode_frames_collected();
        assert!(
            !model
                .camera_cancel_flag
                .load(std::sync::atomic::Ordering::Acquire),
            "HDR processing must keep the live clip source streaming"
        );
    }

    #[tokio::test]
    async fn motion_photo_hdr_completion_does_not_replace_running_source_generation() {
        let mut model = live_model();
        model.start_motion_photo_capture().unwrap();
        let source = model.camera_cancel_flag.clone();
        let _ = model.handle_burst_mode_complete(Err("synthetic HDR failure".into()));
        assert!(Arc::ptr_eq(&source, &model.camera_cancel_flag));
        assert!(
            model.motion_photo_capture.is_none(),
            "A failed still must cancel its clip"
        );
    }

    #[tokio::test]
    async fn motion_photo_busy_lock_survives_settings_toggle() {
        let mut model = live_model();
        let _ = model.capture_photo();
        model.config.motion_photo = false;
        model.is_capturing = false; // completion animation cannot authorize overlap
        let _ = model.capture_photo();
        assert!(
            !model.is_capturing,
            "A pending clip must reject a second still even after the toggle changes"
        );
    }

    #[tokio::test]
    async fn motion_photo_close_defers_exit_until_the_capture_completion() {
        let mut model = live_model();
        model.start_motion_photo_capture().unwrap();
        let cancel = model.motion_photo_capture.as_ref().unwrap().cancel.clone();
        let _ = model.handle_window_close();
        assert!(model.pending_close);
        assert!(
            !cancel.load(std::sync::atomic::Ordering::Acquire),
            "closing drains accepted clips instead of cancelling them"
        );
    }

    #[tokio::test]
    async fn motion_photo_raw_burst_failure_releases_pending_clip() {
        let mut model = live_model();
        model.start_motion_photo_capture().unwrap();
        let _ = model.handle_burst_mode_raw_frames_captured(Err("raw frame timeout".into()));
        assert!(model.motion_photo_capture.is_none());
        assert!(model.motion_photo_error.is_some());
    }

    #[test]
    fn timelapse_start_rejects_pending_motion_photo() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let mut model = live_model();
        {
            let _entered = runtime.enter();
            model.start_motion_photo_capture().unwrap();
        }
        let _ = model.handle_toggle_timelapse();
        assert!(
            !model.timelapse.is_active(),
            "timelapse must not overlap an unfinished Motion Photo"
        );
    }

    #[test]
    fn timelapse_start_rejects_pending_still_capture() {
        let mut model = live_model();
        model.is_capturing = true;
        let _ = model.handle_toggle_timelapse();
        assert!(
            !model.timelapse.is_active(),
            "timelapse must not overlap an unfinished still capture"
        );
    }

    #[test]
    fn timelapse_stop_remains_allowed_during_capture() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let mut model = live_model();
        {
            let _entered = runtime.enter();
            model.start_motion_photo_capture().unwrap();
        }
        model.is_capturing = true;
        let (frame_sender, _frames) = tokio::sync::mpsc::unbounded_channel();
        model.timelapse = crate::app::state::TimelapseState::Running {
            start_time: std::time::Instant::now(),
            shots_taken: 2,
            interval_ms: 1000,
            frame_sender,
        };
        let _ = model.handle_toggle_timelapse();
        assert!(model.timelapse.is_finalising());
    }

    #[tokio::test]
    async fn motion_photo_pending_clip_rejects_quick_record_press() {
        let mut model = live_model();
        model.start_motion_photo_capture().unwrap();
        assert!(!model.can_accept_photo());
        let next_id = model.next_photo_id;
        model.motion_photo_error = Some("existing diagnostic".into());
        for task in [
            model.handle_capture_button_pressed(),
            model.capture_photo(),
            model.handle_capture(),
        ] {
            assert!(cosmic::iced::runtime::task::into_stream(task).is_none());
        }
        assert_eq!(model.next_photo_id, next_id);
        assert_eq!(
            model.motion_photo_error.as_deref(),
            Some("existing diagnostic")
        );
        assert!(!model.quick_record.is_pressed());
        model.motion_photo_capture = None;
        assert!(model.can_accept_photo());
    }

    #[tokio::test]
    async fn motion_photo_starts_for_raw_multistream_and_hdr_shutters() {
        for burst in [
            crate::config::BurstModeSetting::Off,
            crate::config::BurstModeSetting::Frames4,
        ] {
            let mut model = live_model();
            model
                .available_cameras
                .push(crate::backends::camera::types::CameraDevice {
                    supports_multistream: true,
                    ..Default::default()
                });
            model.config.burst_mode_setting = burst;
            let _ = model.capture_photo();
            assert!(model.motion_photo_capture.is_some());
        }
    }

    #[tokio::test]
    async fn motion_photo_source_change_cancels_clip_and_removes_temporary_file() {
        let mut model = live_model();
        model.start_motion_photo_capture().unwrap();
        model
            .camera_cancel_flag
            .store(true, std::sync::atomic::Ordering::Release);
        let live = model.motion_photo_capture.as_mut().unwrap();
        let path = live.temporary_path.clone();
        let job = live.result.take().unwrap();
        let result = tokio::time::timeout(std::time::Duration::from_secs(1), job)
            .await
            .unwrap()
            .unwrap();
        assert!(result.is_err());
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn motion_photo_file_source_and_missing_microphone_fail_explicitly() {
        let mut model = live_model();
        model.current_frame_is_file_source = true;
        let _ = model.capture_photo();
        assert!(
            model
                .motion_photo_error
                .as_deref()
                .unwrap()
                .contains("file preview")
        );
        assert!(model.motion_photo_capture.is_none());
        assert_eq!(
            model.photo_jobs.len(),
            1,
            "a video error must not reject the still shutter"
        );
        model.is_capturing = false;
        model.current_frame_is_file_source = false;
        model.config.record_audio = true;
        let _ = model.capture_photo();
        assert!(
            model
                .motion_photo_error
                .as_deref()
                .unwrap()
                .contains("no microphone")
        );
        assert!(model.motion_photo_capture.is_none());
    }

    #[tokio::test]
    async fn motion_photo_saved_handler_embeds_real_clip_and_extracted_video_decodes() {
        use futures::StreamExt;
        let root = std::env::temp_dir().join(format!(
            "motion-photo-runtime-probe-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let mut model = live_model();
        warm_live_model(&mut model).await;
        let _ = model.capture_photo();
        let still = root.join("synthetic.jpg");
        image::RgbImage::from_pixel(64, 48, image::Rgb([80, 80, 80]))
            .save(&still)
            .unwrap();
        let task = model.handle_photo_capture_saved(1, Ok(still.display().to_string()));
        let mut completion = cosmic::iced::runtime::task::into_stream(task).unwrap();
        for _ in 0..26 {
            let mut frame = model.current_frame.as_ref().unwrap().as_ref().clone();
            frame.captured_at = std::time::Instant::now();
            model.forward_motion_photo_frame(&Arc::new(frame));
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        let clip_path = model.photo_jobs[&1]
            .live
            .as_ref()
            .unwrap()
            .temporary_path
            .clone();
        let clip_bytes = std::fs::read(&clip_path).unwrap();
        let Some(cosmic::iced::runtime::Action::Output(cosmic::Action::App(
            super::Message::PhotoCaptureFinished(1, result),
        ))) = completion.next().await
        else {
            panic!("missing Motion Photo completion");
        };
        let output = result.as_ref().unwrap().clone();
        let _ = model.handle_photo_capture_finished(1, result);
        assert!(model.photo_jobs.is_empty());
        assert!(!model.is_capturing);
        assert!(model.motion_photo_capture.is_none());
        assert!(!clip_path.exists());
        assert!(
            !still.exists(),
            "A successful Motion Photo must not leave a duplicate still"
        );
        let packaged = std::fs::read(&output).unwrap();
        let extracted = root.join("extracted.mp4");
        let video = &packaged[packaged.len() - clip_bytes.len()..];
        assert_eq!(video, clip_bytes);
        std::fs::write(&extracted, video).unwrap();
        let decode = std::process::Command::new("timeout")
            .args(["10s", "ffmpeg", "-v", "error", "-i"])
            .arg(&extracted)
            .args(["-f", "null", "-"])
            .output()
            .unwrap();
        assert!(
            decode.status.success(),
            "{}",
            String::from_utf8_lossy(&decode.stderr)
        );
        image::open(&output).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn motion_photo_hdr_crop_retains_shutter_zoom_after_controls_change() {
        let mut model = live_model();
        model.preview_fit_to_view = true;
        model.photo_aspect_ratio = crate::app::state::PhotoAspectRatio::Native;
        model.zoom_level = 2.0;
        model.start_motion_photo_capture().unwrap();
        model.zoom_level = 1.0;
        assert_eq!(
            model.motion_photo_burst_crop(64, 48),
            Some((16, 12, 32, 24))
        );
    }

    #[tokio::test]
    async fn motion_photo_hdr_fractional_zoom_matches_still_postprocessor_rounding() {
        let mut model = live_model();
        model.preview_fit_to_view = true;
        model.photo_aspect_ratio = crate::app::state::PhotoAspectRatio::Native;
        model.zoom_level = 1.7;
        model.start_motion_photo_capture().unwrap();
        assert_eq!(
            model.motion_photo_burst_crop(64, 48),
            Some((13, 10, 38, 28))
        );
    }

    #[tokio::test]
    async fn motion_photo_shutter_starts_bounded_clip_before_still_dispatch() {
        let mut model = live_model();
        let _ = model.capture_photo();
        assert!(
            model.photo_jobs[&1].live.is_some(),
            "Motion Photo shutter must snapshot its own independently identified clip"
        );
    }

    #[tokio::test]
    async fn motion_photo_without_selected_encoder_still_dispatches_immediately() {
        let mut model = crate::app::state::AppModel::default();
        model.config.motion_photo = true;
        model.config.burst_mode_setting = crate::config::BurstModeSetting::Off;
        model.current_frame = Some(Arc::new(crate::backends::camera::types::CameraFrame {
            width: 64,
            height: 48,
            data: crate::backends::camera::types::FrameData::from_copied(
                vec![0; 64 * 48 * 4].into(),
            ),
            format: crate::backends::camera::types::PixelFormat::RGBA,
            stride: 256,
            yuv_planes: None,
            captured_at: std::time::Instant::now(),
            sensor_timestamp_ns: None,
            libcamera_metadata: None,
        }));
        let _ = model.capture_photo();
        assert!(
            model.photo_jobs.len() == 1,
            "Missing prebuffer/encoder must never delay or reject the still shutter"
        );
    }

    #[tokio::test]
    async fn photo_save_does_not_block_the_async_executor() {
        use crate::app::state::{AppModel, Message};
        use crate::backends::camera::types::{CameraFrame, FrameData, PixelFormat};
        use futures::StreamExt;
        use std::io::Write;
        use std::sync::{Arc, Mutex};

        struct UiWriter(Arc<Mutex<Vec<u8>>>);
        impl Write for UiWriter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let save_dir =
            std::env::temp_dir().join(format!("camera-ui-photo-test-{}", uuid::Uuid::new_v4()));
        let mut model = AppModel {
            current_frame: Some(Arc::new(CameraFrame {
                width: 16,
                height: 16,
                data: FrameData::from_owned_vec(vec![128; 16 * 16 * 4]),
                format: PixelFormat::RGBA,
                stride: 16 * 4,
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: None,
            })),
            hdr_override_disabled: true,
            ..AppModel::default()
        };
        model.config.save_folder_name = save_dir.to_string_lossy().into_owned();
        let mut stream = cosmic::iced::runtime::task::into_stream(model.capture_photo()).unwrap();
        let events = Arc::new(Mutex::new(Vec::new()));
        let writer = Arc::clone(&events);
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .without_time()
            .with_max_level(tracing::Level::INFO)
            .with_writer(move || UiWriter(Arc::clone(&writer)))
            .finish();
        {
            // The first poll is iced's initial yield. The next runs the save
            // job on its single async worker; heavy processing must not run
            // inline there and starve preview subscriptions and animations.
            let _guard = tracing::subscriber::set_default(subscriber);
            let waker = futures::task::noop_waker();
            let mut context = std::task::Context::from_waker(&waker);
            assert!(stream.poll_next_unpin(&mut context).is_pending());
            assert!(stream.poll_next_unpin(&mut context).is_pending());
        }
        let ui_events = String::from_utf8(events.lock().unwrap().clone()).unwrap();
        assert!(
            !ui_events.contains("Starting post-processing"),
            "photo processing ran inline on the async executor: {ui_events}"
        );

        let mut saved = None;
        while let Some(action) = stream.next().await {
            if let cosmic::iced::runtime::Action::Output(cosmic::Action::App(
                Message::PhotoCaptureSaved(_, result),
            )) = action
            {
                saved = Some(result.unwrap());
                // Encoding completion is the readiness signal; do not wait
                // for the independent animation/watchdog timers to drain.
                break;
            }
        }
        let image = image::open(saved.expect("save task must still finish")).unwrap();
        std::fs::remove_dir_all(save_dir).unwrap();
        assert_eq!((image.width(), image.height()), (16, 16));
    }

    #[test]
    fn recording_snapshot_keeps_viewfinder_dimensions_and_gates_audio_rate_with_device() {
        use crate::backends::camera::types::CameraFormat;
        gstreamer::init().unwrap();
        let mut model = live_model();
        model
            .available_audio_devices
            .push(crate::backends::audio::AudioDevice {
                name: "Test microphone".into(),
                serial: "test".into(),
                node_name: "test.microphone".into(),
                is_default: true,
                channels: Vec::new(),
                sample_format: "S16LE".into(),
                sample_rate: 44_100,
            });
        let format = CameraFormat {
            width: 4000,
            height: 3000,
            framerate: None,
            hardware_accelerated: false,
            pixel_format: "RGGB".into(),
        };
        for record_audio in [false, true] {
            model.config.record_audio = record_audio;
            let snapshot = model.appsrc_recording_config(&format, "recording.mp4".into());
            assert_eq!((snapshot.width, snapshot.height), (64, 48));
            assert_eq!(snapshot.framerate, 30);
            assert_eq!(snapshot.format, format);
            assert_eq!(snapshot.output_path, std::path::Path::new("recording.mp4"));
            assert_eq!(snapshot.selected_encoder.unwrap().element_name, "x264enc");
            assert_eq!(
                snapshot.audio_device.as_deref(),
                record_audio.then_some("test.microphone")
            );
            assert_eq!(
                snapshot.audio_source_rate_hz,
                if record_audio { 44_100 } else { 0 }
            );
        }
        model.current_frame = None;
        model.current_audio_device_index = 99;
        model.current_video_encoder_index = 99;
        let snapshot = model.appsrc_recording_config(&format, "fallback.mp4".into());
        assert_eq!((snapshot.width, snapshot.height), (4000, 3000));
        assert!(snapshot.audio_device.is_none());
        assert_eq!(snapshot.audio_source_rate_hz, 0);
        assert!(snapshot.selected_encoder.is_none());
    }

    #[test]
    fn photo_jobs_snapshot_demosaicing_at_capture_start() {
        use crate::app::state::AppModel;
        use crate::shaders::DemosaicMethod;
        gstreamer::init().unwrap();
        let mut app = AppModel::default();
        for method in [
            DemosaicMethod::Rcd,
            DemosaicMethod::Lmmse,
            DemosaicMethod::Bilinear,
        ] {
            app.config.demosaic_method = method;
            let photo_config = app.photo_processing_config();
            drop(app.capture_burst_mode_photo());
            app.config.demosaic_method = DemosaicMethod::Rcd;
            assert_eq!(photo_config.demosaic_method, method);
            assert_eq!(app.burst_mode.demosaic_method, method);
            app.burst_mode.reset();
        }
    }

    #[tokio::test]
    async fn still_routes_preserve_crop_pixels_across_fit_cover_and_sensor_rotation() {
        use crate::app::state::PhotoAspectRatio;
        use crate::backends::camera::types::FrameData;

        // Independent pixel oracle: square fit and square cover both keep
        // sensor columns 16..48 of a 64x32 frame, regardless of quarter-turn.
        // Power-of-two geometry avoids the existing fractional-pixel rounding;
        // there are no UI bars in this fixture.
        for fit in [false, true] {
            for rotation in [
                SensorRotation::None,
                SensorRotation::Rotate90,
                SensorRotation::Rotate180,
                SensorRotation::Rotate270,
            ] {
                for route in 0..3 {
                    let mut model = raw_queue_model();
                    let save_dir = model.config.save_folder_name.clone();
                    model.available_cameras[0].supports_multistream = route != 0;
                    model.available_cameras[0].rotation = rotation;
                    model.zoom_level = 1.0;
                    model.ui_hidden = true;
                    model.screen_width = 400.0;
                    model.screen_height = 400.0;
                    model.preview_fit_to_view = fit;
                    model.photo_aspect_ratio = if fit {
                        PhotoAspectRatio::Ratio1x1
                    } else {
                        PhotoAspectRatio::Native
                    };
                    let sensor = image::RgbaImage::from_fn(64, 32, |x, y| {
                        image::Rgba([x as u8, y as u8, (x + y) as u8, 255])
                    });
                    let mut frame = model.current_frame.as_ref().unwrap().as_ref().clone();
                    frame.width = 64;
                    frame.height = 32;
                    frame.stride = 64 * 4;
                    frame.data = FrameData::from_owned_vec(sensor.clone().into_raw());
                    model.current_frame = Some(Arc::new(frame.clone()));
                    let mut stream =
                        cosmic::iced::runtime::task::into_stream(model.capture_photo()).unwrap();
                    if route != 0 {
                        if route == 2 {
                            frame.stride = 1; // Exercise the real preview fallback.
                        }
                        assert!(
                            model
                                .still_capture
                                .publish(model.still_capture.pending().unwrap(), frame,)
                        );
                        let acquired = next_capture_message(&mut stream).await;
                        assert!(matches!(acquired, super::Message::PhotoCaptureAcquired(1)));
                        let _ = model.update(acquired);
                    }
                    let saved = next_capture_message(&mut stream).await;
                    let super::Message::PhotoCaptureSaved(1, Ok(path)) = &saved else {
                        panic!("route {route}, fit={fit}, rotation={rotation} must save");
                    };
                    let expected = image::imageops::crop_imm(&sensor, 16, 0, 32, 32).to_image();
                    let actual = image::open(path).unwrap().to_rgba8();
                    assert!(
                        actual == expected,
                        "route {route}, fit={fit}, rotation={rotation}: crop pixels differ"
                    );
                    let _ = model.update(saved);
                    assert!(model.photo_jobs.is_empty());
                    std::fs::remove_dir_all(save_dir).unwrap();
                }
            }
        }
    }

    #[tokio::test]
    async fn fit_photo_capture_keeps_the_preview_aspect_in_a_tall_desktop_window() {
        use crate::app::state::{AppModel, Message, PhotoAspectRatio};
        use crate::backends::camera::types::{CameraFrame, FrameData, PixelFormat};
        use futures::StreamExt;
        use std::sync::Arc;

        let save_dir =
            std::env::temp_dir().join(format!("camera-fit-aspect-test-{}", uuid::Uuid::new_v4()));
        let mut model = AppModel {
            current_frame: Some(Arc::new(CameraFrame {
                width: 160,
                height: 120,
                data: FrameData::from_owned_vec(vec![128; 160 * 120 * 4]),
                format: PixelFormat::RGBA,
                stride: 160 * 4,
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: None,
            })),
            preview_fit_to_view: true,
            photo_aspect_ratio: PhotoAspectRatio::Ratio2x1,
            screen_width: 600.0,
            screen_height: 900.0,
            zoom_level: 1.0,
            hdr_override_disabled: true,
            ..AppModel::default()
        };
        model.config.motion_photo = false;
        model.config.save_folder_name = save_dir.to_string_lossy().into_owned();

        let task = model.capture_photo();
        let mut stream = cosmic::iced::runtime::task::into_stream(task).unwrap();
        // Follow the per-capture completion path used by the other still tests,
        // without draining the unrelated encoding-watchdog timer.
        let saved_path = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while let Some(action) = stream.next().await {
                if let cosmic::iced::runtime::Action::Output(cosmic::Action::App(
                    Message::PhotoCaptureSaved(id, result),
                )) = action
                {
                    let path = result.as_ref().expect("photo must save").clone();
                    let _ = model.handle_photo_capture_saved(id, result);
                    return path;
                }
            }
            panic!("capture must report the saved photo");
        })
        .await
        .expect("photo capture task timed out");
        assert!(model.photo_jobs.is_empty());
        assert_eq!(model.last_media_path.as_deref(), Some(saved_path.as_str()));
        let image = image::open(&saved_path).unwrap();
        std::fs::remove_dir_all(save_dir).unwrap();
        assert_eq!(
            (image.width(), image.height()),
            (160, 80),
            "Fit capture must retain the preview's sensor-space 2:1 crop, regardless of window shape"
        );
    }

    #[tokio::test]
    async fn raw_capture_failure_uses_preview_fallback() {
        let fallback_called = Cell::new(false);

        let result = capture_with_preview_fallback(
            || async { Err::<&'static str, _>("raw GPU unavailable".to_string()) },
            || async {
                fallback_called.set(true);
                Ok::<_, String>("preview.jpg")
            },
        )
        .await;

        assert_eq!(result, Ok("preview.jpg"));
        assert!(fallback_called.get());
    }

    #[tokio::test]
    async fn raw_capture_success_does_not_use_preview_fallback() {
        let fallback_called = Cell::new(false);

        let result = capture_with_preview_fallback(
            || async { Ok::<_, String>("raw.jpg") },
            || async {
                fallback_called.set(true);
                Ok::<_, String>("preview.jpg")
            },
        )
        .await;

        assert_eq!(result, Ok("raw.jpg"));
        assert!(!fallback_called.get());
    }

    #[test]
    fn app_recording_uses_decoded_frames_and_a_bounded_channel_for_live_filters() {
        use crate::backends::camera::recording::RecordingBridge;

        let mut model = live_model();
        let bridge = RecordingBridge::default();

        model.recording_bridge = Some(bridge.clone());
        let config = super::AppsrcRecordingConfig {
            width: 64,
            height: 48,
            framerate: 30,
            format: crate::backends::camera::types::CameraFormat {
                width: 64,
                height: 48,
                framerate: Some(Default::default()),
                hardware_accelerated: true,
                pixel_format: "MJPEG".into(),
            },
            output_path: "unused-recording.mp4".into(),
            sensor_rotation: SensorRotation::None,
            audio_device: None,
            audio_source_rate_hz: 0,
            selected_encoder: None,
            bitrate_kbps: 1000,
        };

        // Construction must configure the real producer bridge without polling
        // the task, starting an encoder, or opening a camera/microphone.
        let task = model.start_appsrc_recording(config);

        let sender = bridge.recording_sender();
        assert_eq!(sender.lock().unwrap().as_ref().unwrap().max_capacity(), 15);
        assert_eq!(model.recording_jobs.len(), 1);
        drop(task);
    }
}
