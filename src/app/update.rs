// SPDX-License-Identifier: GPL-3.0-only

//! Message update handling
//!
//! This module handles all application messages by routing them to focused handler methods.
//! The main `update()` function acts as a dispatcher, while specific handlers are implemented
//! in the `handlers` submodules organized by functional domain.
//!
//! # Handler Modules
//!
//! - `handlers::ui`: UI navigation, pickers, tools menu
//! - `handlers::exposure`: Exposure and camera control settings
//! - `handlers::color`: Color adjustment controls
//! - `handlers::camera`: Camera selection, frame handling, transitions
//! - `handlers::format`: Resolution, framerate, codec selection
//! - `handlers::capture`: Photo capture, video recording, zoom
//! - `handlers::virtual_camera`: Virtual camera streaming
//! - `handlers::system`: Gallery, filters, settings, recovery, QR codes

use crate::app::state::{AppModel, ContextPage, Message};
use cosmic::Application;
use cosmic::Task;
use tracing::{debug, info, warn};

impl AppModel {
    /// Fail closed for queued/synthetic user actions during drain. Keep source
    /// and accepted-job completions alive; they are not user interactions.
    fn accepts_during_shutdown(message: &Message) -> bool {
        matches!(
            message,
            Message::WindowClose
                | Message::WindowFocusChanged(_)
                | Message::CameraFrame(_)
                | Message::Noop
                | Message::FileSourcePreviewLoaded(..)
                | Message::VideoSeekPreviewLoaded(_)
                | Message::VideoPreviewPlaybackUpdate(..)
                | Message::VideoPreviewPlaybackStopped
                | Message::VideoFileProgress(..)
                | Message::DisplayOrientationChanged(_)
                | Message::WindowControlsChanged(_)
                | Message::PortalColorSchemeChanged(_)
                | Message::CosmicThemeChanged
                | Message::CamerasInitialized(..)
                | Message::CameraListChanged(_)
                | Message::HotplugDeviceRemoved(_)
                | Message::HotplugDeviceAdded(_)
                | Message::AudioListChanged(_)
                | Message::BrokenEncodersDetected(_)
                | Message::ExposureControlsQueried(..)
                | Message::ExposureControlApplied
                | Message::ExposureControlFailed(_)
                | Message::WhiteBalanceToggled(_)
                | Message::PhotoSaved(_)
                | Message::PhotoCaptureAcquired(_)
                | Message::PhotoCaptureSaved(..)
                | Message::PhotoCaptureFinished(..)
                | Message::PhotoEncodingWatchdog(_)
                | Message::ShutdownRecorderStopped(_)
                | Message::ShutdownDrainWatchdog
                | Message::BurstModeFramesCollected
                | Message::BurstModeRawFramesCaptured(_)
                | Message::BurstModeComplete(_)
                | Message::PollBurstModeProgress
                | Message::ResetBurstModeState
                | Message::RecordingStarted(_)
                | Message::RecordingStopped { .. }
                | Message::EncoderFailed { .. }
                | Message::EncoderOperationFinished(_)
                | Message::DismissEncoderFailure
                | Message::TimelapseAssemblyComplete(_)
                | Message::VirtualCameraStarted
                | Message::VirtualCameraStopped(_)
                | Message::GalleryPortalOpenCompleted(..)
                | Message::GalleryThumbnailLoaded(_)
                | Message::CameraRecoveryStarted { .. }
                | Message::CameraRecoverySucceeded
                | Message::CameraRecoveryFailed(_)
                | Message::AudioRecoveryStarted { .. }
                | Message::AudioRecoverySucceeded
                | Message::AudioRecoveryFailed(_)
                | Message::PrivacyCoverStatusChanged(_)
                | Message::GpuPipelinesWarmed(_)
                | Message::BugReportGenerated(_)
                | Message::InsightsCaptureComplete(_)
                | Message::QrDetectionsUpdated { .. }
        )
    }

    /// Main message handler - routes messages to appropriate handler methods.
    ///
    /// This dispatcher pattern keeps the main update function clean and makes
    /// it easy to find the handling code for any message type.
    pub fn update(&mut self, message: Message) -> Task<cosmic::Action<Message>> {
        if self.pending_close && !Self::accepts_during_shutdown(&message) {
            return Task::none();
        }
        // Async file-preview results are source completions, but may never
        // reintroduce a frame into the non-interactive shutdown screen.
        let message = if self.pending_close
            && matches!(
                message,
                Message::FileSourcePreviewLoaded(..)
                    | Message::VideoSeekPreviewLoaded(_)
                    | Message::VideoPreviewPlaybackUpdate(..)
            ) {
            Message::Noop
        } else {
            message
        };
        let frame_message = matches!(message, Message::CameraFrame(_));
        self.config_persistence.prepare(&self.config);
        let task = match message {
            Message::AudioLevelTick => self.handle_audio_level_tick(),
            // ===== UI Navigation =====
            Message::LaunchUrl(url) => self.handle_launch_url(url),
            Message::ToggleContextPage(page) => self.handle_toggle_context_page(page),
            Message::OpenSettingsPage(page) => self.handle_open_settings_page(page),
            Message::ToggleFormatPicker => self.handle_toggle_format_picker(),
            Message::CloseFormatPicker => self.handle_close_format_picker(),
            Message::ToggleDeviceInfo => self.handle_toggle_device_info(),

            // ===== Tools Menu =====
            Message::ToggleToolsMenu => self.handle_toggle_tools_menu(),
            Message::CloseToolsMenu => self.handle_close_tools_menu(),

            // ===== Motor/PTZ Controls =====
            Message::ToggleMotorPicker => {
                self.motor_picker_visible = !self.motor_picker_visible;
                // Close other pickers when opening motor picker
                if self.motor_picker_visible {
                    self.exposure_picker_visible = false;
                    self.color_picker_visible = false;
                    self.tools_menu_visible = false;
                }
                Task::none()
            }
            Message::CloseMotorPicker => {
                self.motor_picker_visible = false;
                Task::none()
            }
            Message::SetPanAbsolute(value) => {
                self.set_v4l2_pan(value);
                Task::none()
            }
            Message::SetTiltAbsolute(value) => {
                self.set_v4l2_tilt(value);
                Task::none()
            }
            Message::ResetPanTilt => {
                self.reset_pan_tilt();
                Task::none()
            }

            // ===== Exposure Controls =====
            Message::ToggleExposurePicker => self.handle_toggle_exposure_picker(),
            Message::CloseExposurePicker => self.handle_close_exposure_picker(),
            Message::SetExposureMode(mode) => self.handle_set_exposure_mode(mode),
            Message::SetExposureCompensation(value) => self.handle_set_exposure_compensation(value),
            Message::ResetExposureCompensation => self.handle_reset_exposure_compensation(),
            Message::SetExposureTime(value) => self.handle_set_exposure_time(value),
            Message::SetGain(value) => self.handle_set_gain(value),
            Message::SetIsoSensitivity(value) => self.handle_set_iso_sensitivity(value),
            Message::SetMeteringMode(mode) => self.handle_set_metering_mode(mode),
            Message::ToggleAutoExposurePriority => self.handle_toggle_auto_exposure_priority(),
            Message::ExposureControlsQueried(controls, settings, color_settings) => {
                self.handle_exposure_controls_queried(controls, settings, color_settings)
            }
            Message::ExposureControlApplied => Task::none(),
            Message::WhiteBalanceToggled(temp) => {
                if let Some(temp_value) = temp {
                    if let Some(ref mut settings) = self.color_settings {
                        settings.white_balance_temperature = Some(temp_value);
                    }
                    info!(
                        temperature = temp_value,
                        "White balance switched to manual, preserved auto temperature"
                    );
                }
                Task::none()
            }
            Message::ExposureControlFailed(error) => {
                warn!(error = %error, "Exposure control failed");
                Task::none()
            }
            Message::SetBacklightCompensation(value) => {
                self.handle_set_backlight_compensation(value)
            }
            Message::SetFocusAbsolute(value) => self.handle_set_focus_absolute(value),
            Message::ToggleFocusAuto => self.handle_toggle_focus_auto(),
            Message::ResetExposureSettings => self.handle_reset_exposure_settings(),
            Message::ExposureModeSelected(entity) => self.handle_exposure_mode_selected(entity),

            // ===== Color Controls =====
            Message::ToggleColorPicker => self.handle_toggle_color_picker(),
            Message::CloseColorPicker => self.handle_close_color_picker(),
            Message::SetContrast(value) => self.handle_set_contrast(value),
            Message::SetSaturation(value) => self.handle_set_saturation(value),
            Message::SetSharpness(value) => self.handle_set_sharpness(value),
            Message::SetHue(value) => self.handle_set_hue(value),
            Message::ToggleAutoWhiteBalance => self.handle_toggle_auto_white_balance(),
            Message::SetWhiteBalanceTemperature(value) => {
                self.handle_set_white_balance_temperature(value)
            }
            Message::ResetColorSettings => self.handle_reset_color_settings(),

            // ===== Camera Control =====
            Message::SwitchCamera => self.handle_switch_camera(),
            Message::SelectCamera(index) => self.handle_select_camera(index),
            Message::CameraFrame(frame) => self.handle_camera_frame(frame),
            Message::CamerasInitialized(cameras, index, formats) => {
                self.handle_cameras_initialized(cameras, index, formats)
            }
            Message::BrokenEncodersDetected(broken) => self.handle_broken_encoders_detected(broken),
            Message::CameraListChanged(cameras) => self.handle_camera_list_changed(cameras),
            Message::HotplugDeviceRemoved(removed) => self.handle_hotplug_device_removed(removed),
            Message::HotplugDeviceAdded(new_devices) => {
                self.handle_hotplug_device_added(new_devices)
            }
            Message::AudioListChanged(devices) => self.handle_audio_list_changed(devices),
            Message::StartCameraTransition => self.handle_start_camera_transition(),
            Message::ClearTransitionBlur => self.handle_clear_transition_blur(),
            Message::ToggleMirrorPreview => self.handle_toggle_mirror_preview(),
            Message::ToggleMirrorCaptures => self.handle_toggle_mirror_captures(),
            Message::ToggleHapticFeedback => self.handle_toggle_haptic_feedback(),
            Message::ToggleVirtualCameraEnabled => self.handle_toggle_virtual_camera_enabled(),

            // ===== Format Selection =====
            Message::SetMode(mode) => self.handle_set_mode(mode),
            Message::PickerSelectResolution(width) => self.handle_picker_select_resolution(width),
            Message::PickerSelectFormat(index) => self.handle_picker_select_format(index),
            Message::SelectBitratePreset(index) => self.handle_select_bitrate_preset(index),

            // ===== Capture Operations =====
            Message::Capture => self.handle_capture(),
            Message::ToggleFlash => self.handle_toggle_flash(),
            Message::DismissToggleFeedback(shown_at) => {
                self.dismiss_toggle_feedback(shown_at);
                Task::none()
            }
            Message::DismissFlashError => self.handle_dismiss_flash_error(),
            Message::ToggleBurstMode => self.handle_toggle_burst_mode(),
            Message::ToggleMotionPhotoCapture => self.handle_toggle_motion_photo_capture(),
            Message::SetBurstModeFrameCount(index) => self.handle_set_burst_mode_frame_count(index),
            Message::BurstModeFramesCollected => self.handle_burst_mode_frames_collected(),
            Message::BurstModeRawFramesCaptured(result) => {
                self.handle_burst_mode_raw_frames_captured(result)
            }
            Message::BurstModeComplete(result) => self.handle_burst_mode_complete(result),
            Message::PollBurstModeProgress => self.handle_poll_burst_mode_progress(),
            Message::ResetBurstModeState => {
                self.burst_mode.reset();
                // Ensure flash is turned off when burst mode resets (safety measure)
                self.turn_off_flash_hardware();
                self.flash.active = false;
                debug!("Burst mode state reset");
                Task::none()
            }
            Message::BrightnessEvaluationTick => self.handle_brightness_evaluation_tick(),
            Message::CyclePhotoAspectRatio => self.handle_cycle_photo_aspect_ratio(),
            Message::FlashComplete => self.handle_flash_complete(),
            Message::CyclePhotoTimer => self.handle_cycle_photo_timer(),
            Message::PhotoTimerTick => self.handle_photo_timer_tick(),
            Message::PhotoTimerAnimationFrame => Task::none(),
            Message::AbortPhotoTimer => self.handle_abort_photo_timer(),
            Message::ZoomIn => self.handle_zoom_in(),
            Message::ZoomOut => self.handle_zoom_out(),
            Message::ResetZoom => self.handle_reset_zoom(),
            Message::TogglePreviewFit => {
                // Snapshot the animated values before toggling so a
                // mid-animation reversal starts from where the eye currently
                // is, not from the prior endpoint.
                let from = self.capture_fit_state();
                self.preview_fit_to_view = !self.preview_fit_to_view;
                self.config.preview_fit_to_view = self.preview_fit_to_view;
                self.persist_config_async();
                self.start_fit_animation(from)
            }
            Message::ToggleUiChrome => self.handle_toggle_ui_chrome(),
            Message::FitAnimationTick => self.tick_animation_until(
                crate::app::view::FIT_ANIMATION_DURATION,
                Message::FitAnimationTick,
                self.fit_animation.map(|a| a.start.elapsed()),
                |s| s.fit_animation = None,
            ),
            Message::ZoomAnimationTick => self.tick_animation_until(
                crate::app::view::ZOOM_ANIMATION_DURATION,
                Message::ZoomAnimationTick,
                self.zoom_animation.map(|a| a.start.elapsed()),
                |s| s.zoom_animation = None,
            ),
            Message::WindowFocusChanged(focused) => {
                info!(focused, "WindowFocusChanged");
                crate::backends::volume_keys::set_focused(focused && !self.pending_close);
                Task::none()
            }
            Message::DisplayOrientationChanged(orientation) => {
                // Backend already logs at info on the dedupe boundary; keep
                // the update path silent to avoid double-logging.
                self.display_orientation = orientation;
                Task::none()
            }
            Message::WindowClose => self.handle_window_close(),
            Message::WindowControlsChanged(controls) => {
                self.window_controls = controls;
                Task::none()
            }
            Message::WindowMinimize => self.core.minimize(None),
            Message::WindowToggleMaximize => self.core.toggle_maximize(None),
            Message::WindowDrag => self.core.drag(None),
            Message::PinchZoom(level) => self.handle_pinch_zoom(level),
            Message::PhotoSaved(result) => self.handle_photo_saved(result),
            Message::PhotoCaptureAcquired(id) => self.handle_photo_capture_acquired(id),
            Message::PhotoCaptureSaved(id, result) => self.handle_photo_capture_saved(id, result),
            Message::PhotoEncodingWatchdog(id) => self.handle_photo_encoding_watchdog(id),
            Message::PhotoCaptureFinished(id, result) => {
                self.handle_photo_capture_finished(id, result)
            }
            Message::ShutdownRecorderStopped(result) => {
                self.handle_shutdown_recorder_stopped(result)
            }
            Message::ShutdownDrainWatchdog => self.handle_shutdown_drain_watchdog(),
            Message::DismissMotionPhotoError => self.handle_dismiss_motion_photo_error(),
            Message::ClearCaptureAnimation => self.handle_clear_capture_animation(),
            Message::ToggleRecording => self.handle_toggle_recording(),
            Message::RecordingStarted(path) => self.handle_recording_started(path),
            Message::EncoderFailed { operation, failure } => {
                self.handle_encoder_failure(operation, failure)
            }
            Message::EncoderOperationFinished(operation) => {
                self.finish_encoder_operation(operation, None);
                Task::none()
            }
            Message::DismissEncoderFailure => self.dismiss_encoder_failure(),
            Message::RecordingStopped { session, result } => {
                self.handle_recording_stopped(session, result)
            }
            Message::UpdateRecordingDuration => self.handle_update_recording_duration(),
            Message::StartRecordingAfterDelay => self.handle_start_recording_after_delay(),
            Message::CaptureButtonPressed => self.handle_capture_button_pressed(),
            Message::CaptureButtonReleased => self.handle_capture_button_released(),
            Message::QuickRecordThreshold => self.handle_quick_record_threshold(),

            // ===== Timelapse =====
            Message::NextMode => self.handle_cycle_mode(true),
            Message::PrevMode => self.handle_cycle_mode(false),
            Message::ToggleTimelapse => self.handle_toggle_timelapse(),
            Message::TimelapseTick => self.handle_timelapse_tick(),
            Message::SetTimelapseInterval(i) => self.handle_set_timelapse_interval(i),
            Message::TimelapseAssemblyComplete(result) => {
                self.handle_timelapse_assembly_complete(result)
            }

            // ===== Virtual Camera =====
            Message::ToggleVirtualCamera => self.handle_toggle_virtual_camera(),
            Message::VirtualCameraStarted => self.handle_virtual_camera_started(),
            Message::VirtualCameraStopped(result) => self.handle_virtual_camera_stopped(result),
            Message::UpdateVirtualCameraDuration => self.handle_update_virtual_camera_duration(),
            Message::OpenVirtualCameraFile => self.handle_open_virtual_camera_file(),
            Message::VirtualCameraFileSelected(file_source) => {
                self.handle_virtual_camera_file_selected(file_source)
            }
            Message::ClearVirtualCameraFile => self.handle_clear_virtual_camera_file(),
            Message::FileSourcePreviewLoaded(frame, duration) => {
                self.handle_file_source_preview_loaded(frame, duration)
            }
            Message::VideoFileProgress(position, duration, progress) => {
                self.handle_video_file_progress(position, duration, progress)
            }
            Message::VideoFileSeek(position) => self.handle_video_file_seek(position),
            Message::VideoSeekPreviewLoaded(frame) => self.handle_video_seek_preview_loaded(frame),
            Message::VideoPreviewPlaybackUpdate(frame, pos, dur, progress) => {
                self.handle_video_preview_playback_update(frame, pos, dur, progress)
            }
            Message::VideoPreviewPlaybackStopped => self.handle_video_preview_playback_stopped(),
            Message::ToggleVideoPlayPause => self.handle_toggle_video_play_pause(),
            Message::StartVideoPreviewPlayback => self.start_video_preview_playback(),

            // ===== Gallery =====
            Message::OpenGallery => self.handle_open_gallery(),
            Message::GalleryPortalOpenCompleted(path, result) => {
                self.handle_gallery_portal_open_completed(path, result)
            }
            Message::RefreshGalleryThumbnail => self.handle_refresh_gallery_thumbnail(),
            Message::GalleryThumbnailLoaded(data) => self.handle_gallery_thumbnail_loaded(data),

            // ===== Filters =====
            Message::SelectFilter(filter) => self.handle_select_filter(filter),

            // ===== Settings =====
            Message::UpdateConfig(config) => self.handle_update_config(config),
            Message::ConfigReloaded(revision, result) => {
                self.handle_config_reloaded(revision, result)
            }
            Message::SetAppTheme(index) => self.handle_set_app_theme(index),
            Message::SetOverlayEffect(index) => self.handle_set_overlay_effect(index),
            Message::SetControlsPosition(index) => self.handle_set_controls_position(index),
            Message::SelectDefaultMode(index) => self.handle_select_default_mode(index),
            Message::PortalColorSchemeChanged(is_dark) => {
                self.handle_portal_color_scheme_changed(is_dark)
            }
            Message::CosmicThemeChanged => self.handle_cosmic_theme_changed(),
            Message::SelectAudioDevice(index) => self.handle_select_audio_device(index),
            Message::SelectVideoEncoder(index) => self.handle_select_video_encoder(index),
            Message::SelectPhotoOutputFormat(index) => {
                self.handle_select_photo_output_format(index)
            }
            Message::SetDemosaicMethod(method) => self.handle_set_demosaic_method(method),
            Message::ToggleMotionPhoto(enabled) => self.handle_toggle_motion_photo(enabled),
            Message::ToggleRecordAudio => self.handle_toggle_record_audio(),
            Message::SelectAudioEncoder(index) => self.handle_select_audio_encoder(index),
            Message::ToggleSaveBurstRaw => self.handle_toggle_save_burst_raw(),
            Message::SelectCompositionGuide(index) => self.handle_select_composition_guide(index),
            Message::ResetAllSettings => self.handle_reset_all_settings(),

            // ===== System & Recovery =====
            Message::CameraRecoveryStarted {
                attempt,
                max_attempts,
            } => self.handle_camera_recovery_started(attempt, max_attempts),
            Message::CameraRecoverySucceeded => self.handle_camera_recovery_succeeded(),
            Message::CameraRecoveryFailed(error) => self.handle_camera_recovery_failed(error),
            Message::AudioRecoveryStarted {
                attempt,
                max_attempts,
            } => self.handle_audio_recovery_started(attempt, max_attempts),
            Message::AudioRecoverySucceeded => self.handle_audio_recovery_succeeded(),
            Message::AudioRecoveryFailed(error) => self.handle_audio_recovery_failed(error),
            Message::GenerateBugReport => self.handle_generate_bug_report(),
            Message::BugReportGenerated(result) => self.handle_bug_report_generated(result),
            Message::ShowBugReport => self.handle_show_bug_report(),

            // ===== QR Code Detection =====
            Message::ToggleQrDetection => self.handle_toggle_qr_detection(),
            Message::QrDetectionsUpdated {
                generation,
                detections,
            } => self.handle_qr_detections_updated(generation, detections),
            Message::QrOpenUrl(url) => self.handle_qr_open_url(url),
            Message::QrConnectWifi {
                ssid,
                password,
                security,
                hidden,
            } => self.handle_qr_connect_wifi(ssid, password, security, hidden),
            Message::QrCopyText(text) => self.handle_qr_copy_text(text),

            // ===== Privacy Cover Detection =====
            Message::PrivacyCoverStatusChanged(is_closed) => {
                self.handle_privacy_cover_status_changed(is_closed)
            }

            // ===== Insights Drawer =====
            Message::UpdateInsightsMetrics => self.handle_update_insights_metrics(),
            Message::CopyPipelineString => self.handle_copy_pipeline_string(),
            Message::InsightsCaptureFrames => self.handle_insights_capture(1),
            Message::InsightsCaptureBurst => self.handle_insights_capture(6),
            Message::InsightsCaptureComplete(result) => {
                match &result {
                    Ok(paths) => info!(count = paths.len(), "Insights capture saved"),
                    Err(e) => warn!(error = %e, "Insights capture failed"),
                }
                Task::none()
            }

            Message::GpuPipelinesWarmed(result) => {
                if let Err(e) = result {
                    warn!("GPU pipeline warmup failed: {e}");
                }
                Task::none()
            }

            // ===== Keyboard shortcuts =====
            Message::OpenKeyBindingsPage => {
                self.context_page = ContextPage::KeyBindings;
                self.core.window.show_context = true;
                Task::none()
            }
            Message::Escape => self.on_escape(),
            Message::StartRecordingKeyBind(action) => {
                self.recording_keybind = Some(crate::app::state::RecordingKeyBindState {
                    action,
                    captured: None,
                    conflict_with: None,
                });
                Task::none()
            }
            Message::KeyBindRecordingCaptured(kb) => {
                if let Some(action) = self.recording_keybind.as_ref().map(|r| r.action) {
                    let conflict = self.bindings.conflict_for(&kb, action);
                    if let Some(rec) = self.recording_keybind.as_mut() {
                        rec.conflict_with = conflict;
                        rec.captured = Some(kb);
                    }
                }
                Task::none()
            }
            Message::CancelKeyBindRecording => {
                self.recording_keybind = None;
                Task::none()
            }
            Message::KeyBindSave => {
                if let Some(rec) = self.recording_keybind.take()
                    && let Some(combo) = rec.captured
                {
                    if let Some(other) = rec.conflict_with {
                        self.config
                            .key_bindings
                            .insert(other, crate::app::keybind::SerializedKeyBind::default());
                        self.bindings.set(other, None);
                    }
                    self.bindings.set(rec.action, Some(combo.clone()));
                    self.config.key_bindings.insert(
                        rec.action,
                        crate::app::keybind::SerializedKeyBind::from(&combo),
                    );
                    self.persist_config_async();
                }
                Task::none()
            }
            Message::ResetKeyBindToDefault(action) => {
                self.config.key_bindings.remove(&action);
                self.bindings.reset_to_default(action);
                self.persist_config_async();
                Task::none()
            }
            Message::ResetAllKeyBindings => {
                self.config.key_bindings.clear();
                self.bindings = crate::app::keybind::Bindings::defaults();
                self.persist_config_async();
                Task::none()
            }

            Message::Noop => Task::none(),
        };
        // Covers camera/format/settings resets as well as every animation
        // tick. Zoom never changes the capture subscription identity.
        self.sync_zoom_control();
        if !frame_message {
            self.sync_motion_photo_recorder();
        }
        self.reconcile_shutdown_preview();
        Task::batch([task, self.maybe_finish_shutdown()])
    }
}

#[cfg(test)]
mod shutdown_tests {
    use super::*;

    #[test]
    fn shutdown_source_events_do_not_restart_camera_or_restore_preview() {
        let mut model = AppModel::default();
        model
            .available_cameras
            .push(crate::backends::camera::types::CameraDevice {
                name: "accepted source".into(),
                path: "accepted-source".into(),
                ..Default::default()
            });
        model.pending_close = true;
        model.photo_jobs.insert(
            1,
            crate::app::handlers::motion_photo::PhotoJob {
                live: None,
                still_path: None,
            },
        );
        let generation = model.camera_cancel_flag.clone();
        for message in [
            Message::CamerasInitialized(Vec::new(), 0, Vec::new()),
            Message::CameraListChanged(Vec::new()),
            Message::HotplugDeviceAdded(vec![("/dev/new-camera".into(), "new camera".into())]),
        ] {
            let _ = model.update(message);
            assert_eq!(
                model.available_cameras.len(),
                1,
                "freeze subscription identity during drain"
            );
            assert_eq!(model.available_cameras[0].path, "accepted-source");
            assert!(std::sync::Arc::ptr_eq(
                &generation,
                &model.camera_cancel_flag
            ));
            assert!(model.current_frame.is_none());
        }
        assert!(AppModel::accepts_during_shutdown(
            &Message::CameraRecoverySucceeded
        ));
        assert!(AppModel::accepts_during_shutdown(
            &Message::BurstModeRawFramesCaptured(Err("source error".into()))
        ));
        assert!(AppModel::accepts_during_shutdown(
            &Message::RecordingStopped {
                session: 1,
                result: Err("source error".into()),
            }
        ));
    }

    #[test]
    fn shutdown_rejects_queued_ui_keyboard_and_config_interactions() {
        gstreamer::init().unwrap();
        let mut model = AppModel {
            pending_close: true,
            ..Default::default()
        };
        model.photo_jobs.insert(
            1,
            crate::app::handlers::motion_photo::PhotoJob {
                live: None,
                still_path: None,
            },
        );
        let mode = model.mode;
        let mirror = model.config.mirror_preview;
        let zoom = model.zoom_level;
        for message in [
            Message::ToggleContextPage(ContextPage::Settings),
            Message::ToggleToolsMenu,
            Message::ToggleFormatPicker,
            Message::OpenKeyBindingsPage,
            Message::ToggleMirrorPreview,
            Message::ZoomIn,
            Message::PinchZoom(3.0),
            Message::NextMode,
            Message::Capture,
            Message::CaptureButtonReleased,
            Message::WindowMinimize,
            Message::WindowToggleMaximize,
            Message::WindowDrag,
            Message::Escape,
            Message::ResetAllSettings,
            Message::UpdateConfig(crate::config::Config::default()),
        ] {
            assert!(cosmic::iced::runtime::task::into_stream(model.update(message)).is_none());
        }
        assert!(!model.core.window.show_context);
        assert!(!model.tools_menu_visible);
        assert!(!model.format_picker_visible);
        assert_eq!(model.mode, mode);
        assert_eq!(model.config.mirror_preview, mirror);
        assert_eq!(model.zoom_level, zoom);
        assert_eq!(model.shutdown_pending_saves(), 1);
        let _ = model.update(Message::PhotoCaptureFinished(
            1,
            Err("synthetic completion".into()),
        ));
        assert_eq!(model.shutdown_pending_saves(), 0);
        assert!(model.shutdown_waiting_for_recorder);
    }
}
