// SPDX-License-Identifier: GPL-3.0-only

//! System handlers
//!
//! Handles gallery operations, filter selection, settings, recovery, bug reports,
//! and QR code detection.

use crate::app::state::{AppModel, FilterType, Message, RecordingState};
use cosmic::Task;
use cosmic::cosmic_config::CosmicConfigEntry;
use tracing::{error, info};

impl AppModel {
    // =========================================================================
    // Gallery Handlers
    // =========================================================================

    pub(crate) fn handle_open_gallery(&self) -> Task<cosmic::Action<Message>> {
        // If we have a last media path, open the file manager with that file pre-selected
        if let Some(ref path) = self.last_media_path {
            info!(path = %path, "Opening gallery with file pre-selected");
            if Self::show_in_file_manager(path).is_ok() {
                return Task::none();
            }
            info!("show_in_file_manager failed, falling back to directory open");
        }

        let photo_dir = crate::app::get_photo_directory(&self.config.save_folder_name);
        info!(path = %photo_dir.display(), "Opening gallery directory");

        if let Err(e) = open::that(&photo_dir) {
            error!(error = %e, path = %photo_dir.display(), "Failed to open gallery directory");
        } else {
            info!("Gallery opened successfully");
        }
        Task::none()
    }

    pub(crate) fn handle_refresh_gallery_thumbnail(&self) -> Task<cosmic::Action<Message>> {
        let photos_dir = crate::app::get_photo_directory(&self.config.save_folder_name);
        let videos_dir = crate::app::get_video_directory(&self.config.save_folder_name);
        Task::perform(
            async move { crate::storage::load_latest_thumbnail(photos_dir, videos_dir).await },
            |handle| cosmic::Action::App(Message::GalleryThumbnailLoaded(handle)),
        )
    }

    pub(crate) fn handle_gallery_thumbnail_loaded(
        &mut self,
        data: Option<crate::storage::GalleryThumbnailData>,
    ) -> Task<cosmic::Action<Message>> {
        if let Some((handle, rgba, width, height, path)) = data {
            self.gallery_thumbnail = Some(handle);
            self.gallery_thumbnail_rgba = Some((rgba, width, height));
            self.last_media_path = Some(path.display().to_string());
        } else {
            self.gallery_thumbnail = None;
            self.gallery_thumbnail_rgba = None;
        }
        Task::none()
    }

    // =========================================================================
    // Filter Handlers
    // =========================================================================

    pub(crate) fn handle_select_filter(
        &mut self,
        filter: FilterType,
    ) -> Task<cosmic::Action<Message>> {
        self.selected_filter = filter;
        // Update the shared atomic so the recording pusher picks up the change
        self.recording_filter_code.store(
            filter.gpu_filter_code(),
            std::sync::atomic::Ordering::Relaxed,
        );
        info!("Filter selected: {:?}", filter);

        // Update virtual camera filter if streaming
        if self.virtual_camera.is_streaming() {
            self.virtual_camera.set_filter(filter);
        }

        // Close the filter drawer after selection
        self.core.window.show_context = false;

        Task::none()
    }

    // =========================================================================
    // Settings Handlers
    // =========================================================================

    pub(crate) fn handle_update_config(
        &mut self,
        config: crate::config::Config,
    ) -> Task<cosmic::Action<Message>> {
        info!("UpdateConfig received");
        self.config = config;
        // Keep the draw-time global in step with an externally-changed config.
        crate::app::overlay_style::init_overlay_effect(self.config.overlay_effect);
        Task::none()
    }

    pub(crate) fn handle_set_app_theme(&mut self, index: usize) -> Task<cosmic::Action<Message>> {
        use crate::config::AppTheme;

        let app_theme = match index {
            0 => AppTheme::System,
            1 => AppTheme::Dark,
            2 => AppTheme::Light,
            _ => return Task::none(),
        };

        info!(?app_theme, "Setting application theme");
        self.config.app_theme = app_theme;

        if let Some(handler) = self.config_handler.as_ref()
            && let Err(err) = self.config.write_entry(handler)
        {
            error!(?err, "Failed to save app theme setting");
        }

        cosmic::command::set_theme(app_theme.theme())
    }

    pub(crate) fn handle_set_overlay_effect(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
        use crate::config::OverlayEffect;

        // Indexes `available()` — the same slice the dropdown was built from,
        // which drops `System` off-COSMIC. Indexing `ALL` here would silently
        // shift every choice by one on those desktops.
        let effect = match OverlayEffect::from_dropdown_index(index) {
            Some(e) => e,
            None => return Task::none(),
        };

        info!(?effect, "Setting overlay effect");
        self.config.overlay_effect = effect;

        // The draw-time colour roots read the global, not `self.config`.
        crate::app::overlay_style::init_overlay_effect(effect);

        if let Some(handler) = self.config_handler.as_ref()
            && let Err(err) = self.config.write_entry(handler)
        {
            error!(?err, "Failed to save overlay effect setting");
        }

        Task::none()
    }

    pub(crate) fn handle_set_controls_position(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
        let Some(&position) = crate::config::ControlsPosition::ALL.get(index) else {
            return Task::none();
        };

        info!(?position, "Setting controls panel position");
        self.config.controls_position = position;

        if let Some(handler) = self.config_handler.as_ref()
            && let Err(err) = self.config.write_entry(handler)
        {
            error!(?err, "Failed to save controls panel position");
        }

        Task::none()
    }

    pub(crate) fn handle_select_default_mode(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
        use crate::app::state::CameraMode;

        // Build the same visible modes list as the dropdown
        let mut visible_modes = vec![CameraMode::Photo, CameraMode::Video, CameraMode::Timelapse];
        if self.config.virtual_camera_enabled {
            visible_modes.push(CameraMode::Virtual);
        }

        let mode = match visible_modes.get(index) {
            Some(&m) => m,
            None => return Task::none(),
        };

        info!(?mode, "Setting default launch mode");
        self.config.default_mode = mode;

        if let Some(handler) = self.config_handler.as_ref()
            && let Err(err) = self.config.write_entry(handler)
        {
            error!(?err, "Failed to save default mode setting");
        }

        Task::none()
    }

    /// Re-apply the active theme after a COSMIC theme config edit (accent
    /// color, palette, light/dark mode switch from outside the app, etc.).
    ///
    /// `AppTheme::theme()` re-reads from `cosmic_config` each call, so
    /// running the same selector through `set_theme` again gives the app
    /// the just-edited palette. Previously the app only re-applied on
    /// light/dark *toggles* (via the portal subscription on non-COSMIC, or
    /// implicitly via libcosmic's own restart heuristic on COSMIC), which
    /// is why an accent change in the *current* mode looked stuck until
    /// the user toggled to the other mode and back (issue #290).
    pub(crate) fn handle_cosmic_theme_changed(&mut self) -> Task<cosmic::Action<Message>> {
        info!("COSMIC theme config changed — re-applying active theme");
        cosmic::command::set_theme(self.config.app_theme.theme())
    }

    pub(crate) fn handle_portal_color_scheme_changed(
        &mut self,
        is_dark: bool,
    ) -> Task<cosmic::Action<Message>> {
        use crate::config::AppTheme;

        // Only apply if user wants to follow system theme
        if self.config.app_theme != AppTheme::System {
            return Task::none();
        }

        info!(is_dark, "Portal color scheme changed, updating theme");
        let theme = if is_dark {
            cosmic::Theme::dark()
        } else {
            cosmic::Theme::light()
        };
        cosmic::command::set_theme(theme)
    }

    pub(crate) fn handle_select_audio_device(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
        if index < self.available_audio_devices.len() {
            info!(index, "Selected audio device index");
            self.current_audio_device_index = index;
        }
        self.sync_audio_probe();
        Task::none()
    }

    pub(crate) fn handle_audio_level_tick(&mut self) -> Task<cosmic::Action<Message>> {
        // Recorder wins over probe — they never coexist by design.
        let source = if self.recording.is_recording() {
            self.recording.audio_levels().cloned()
        } else {
            self.probe_audio_levels.clone()
        };

        self.audio_levels_snapshot =
            source.and_then(|arc| arc.lock().ok().map(|guard| guard.clone()));
        Task::none()
    }

    pub(crate) fn handle_audio_list_changed(
        &mut self,
        new_devices: Vec<crate::backends::audio::AudioDevice>,
    ) -> Task<cosmic::Action<Message>> {
        info!(
            old_count = self.available_audio_devices.len(),
            new_count = new_devices.len(),
            "Audio device list changed (hotplug event)"
        );

        // If enumeration returned empty but we had devices before, treat it as
        // an enumeration failure (e.g. pw-dump failed) rather than all devices
        // being removed. Preserve the existing list to avoid killing recordings.
        if new_devices.is_empty() && !self.available_audio_devices.is_empty() {
            info!("Audio enumeration returned empty list, keeping existing devices");
            return Task::none();
        }

        // Try to keep the current device selected if it's still available
        let current_still_available = if let Some(current) = self
            .available_audio_devices
            .get(self.current_audio_device_index)
        {
            new_devices
                .iter()
                .position(|d| d.serial == current.serial && d.name == current.name)
        } else {
            None
        };

        // Stop recording if the audio input used for recording was disconnected
        // Only do this when we have a non-empty new device list (confirmed enumeration)
        // and the specific device is genuinely missing from it
        if current_still_available.is_none()
            && !new_devices.is_empty()
            && self.config.record_audio
            && self.recording.is_recording()
        {
            info!("Audio input disconnected during recording, stopping recording gracefully");
            if let Some(sender) = self.recording.take_stop_sender() {
                let _ = sender.send(());
            }
            self.recording = RecordingState::Idle;
        }

        self.available_audio_devices = new_devices;
        self.audio_dropdown_options = self
            .available_audio_devices
            .iter()
            .map(|dev| {
                if dev.is_default {
                    format!("{} (Default)", dev.name)
                } else {
                    dev.name.clone()
                }
            })
            .collect();

        if let Some(new_index) = current_still_available {
            self.current_audio_device_index = new_index;
        } else if !self.available_audio_devices.is_empty() {
            // Reset to first device (default is sorted first)
            self.current_audio_device_index = 0;
        }

        self.sync_audio_probe();
        Task::none()
    }

    pub(crate) fn handle_select_video_encoder(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
        if index < self.available_video_encoders.len() {
            info!(index, encoder = %self.available_video_encoders[index].display_name, "Selected video encoder");
            self.current_video_encoder_index = index;

            self.config.last_video_encoder_index = Some(index);
            if let Some(handler) = self.config_handler.as_ref()
                && let Err(err) = self.config.write_entry(handler)
            {
                error!(?err, "Failed to save encoder selection");
            }
        }
        Task::none()
    }

    pub(crate) fn handle_select_photo_output_format(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
        use crate::config::PhotoOutputFormat;

        if index < PhotoOutputFormat::ALL.len() {
            let format = PhotoOutputFormat::ALL[index];
            info!(?format, "Selected photo output format");
            self.config.photo_output_format = format;

            if let Some(handler) = self.config_handler.as_ref()
                && let Err(err) = self.config.write_entry(handler)
            {
                error!(?err, "Failed to save photo output format selection");
            }
        }
        Task::none()
    }

    pub(crate) fn handle_toggle_record_audio(&mut self) -> Task<cosmic::Action<Message>> {
        if self.recording.is_recording() {
            return Task::none();
        }

        use cosmic::cosmic_config::CosmicConfigEntry;

        self.config.record_audio = !self.config.record_audio;
        info!(
            record_audio = self.config.record_audio,
            "Toggled record audio"
        );

        if let Some(handler) = self.config_handler.as_ref()
            && let Err(err) = self.config.write_entry(handler)
        {
            error!(?err, "Failed to save record audio setting");
        }
        self.sync_audio_probe();
        Task::none()
    }

    pub(crate) fn handle_select_audio_encoder(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
        use crate::config::AudioEncoder;
        use cosmic::cosmic_config::CosmicConfigEntry;

        if index < AudioEncoder::ALL.len() {
            let encoder = AudioEncoder::ALL[index];
            info!(?encoder, "Selected audio encoder");
            self.config.audio_encoder = encoder;

            if let Some(handler) = self.config_handler.as_ref()
                && let Err(err) = self.config.write_entry(handler)
            {
                error!(?err, "Failed to save audio encoder selection");
            }
        }
        Task::none()
    }

    pub(crate) fn handle_toggle_save_burst_raw(&mut self) -> Task<cosmic::Action<Message>> {
        self.config.save_burst_raw = !self.config.save_burst_raw;
        info!(
            save_burst_raw = self.config.save_burst_raw,
            "Toggled save burst raw frames"
        );

        if let Some(handler) = self.config_handler.as_ref()
            && let Err(err) = self.config.write_entry(handler)
        {
            error!(?err, "Failed to save burst raw setting");
        }
        Task::none()
    }

    pub(crate) fn handle_select_composition_guide(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
        use crate::config::CompositionGuide;
        if let Some(&guide) = CompositionGuide::ALL.get(index) {
            self.config.composition_guide = guide;
            info!(?guide, "Selected composition guide");

            if let Some(handler) = self.config_handler.as_ref()
                && let Err(err) = self.config.write_entry(handler)
            {
                error!(?err, "Failed to save composition guide setting");
            }
        }
        Task::none()
    }

    pub(crate) fn handle_reset_all_settings(&mut self) -> Task<cosmic::Action<Message>> {
        info!("Resetting all settings to defaults");
        // Snapshot the pre-reset animated values so the fit/fill change
        // animates rather than snaps when reset flips preview_fit_to_view
        // (or drops out of View mode).
        let from_fit = self.capture_fit_state();

        self.config = crate::config::Config::default();

        if let Some(handler) = self.config_handler.as_ref()
            && let Err(err) = self.config.write_entry(handler)
        {
            error!(?err, "Failed to save reset settings");
        }

        // Reset UI state that shadows config values
        self.current_video_encoder_index = 0;
        self.selected_filter = FilterType::default();
        self.recording_filter_code.store(
            self.selected_filter.gpu_filter_code(),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.photo_aspect_ratio = self.config.photo_aspect_ratio;
        self.preview_fit_to_view = self.config.preview_fit_to_view;
        self.zoom_level = 1.0;
        // View is a passive UI mode (no capture controls). Reset shouldn't
        // strand the user there — drop back to the configured default mode
        // so the rest of the reset (capture controls, scrim, etc.) is
        // actually visible.
        if self.mode == crate::app::state::CameraMode::View {
            self.mode = self.config.default_mode;
        }

        // Reset camera exposure and color controls to defaults
        let exposure_task = self.handle_reset_exposure_settings();
        let color_task = self.handle_reset_color_settings();
        let fit_anim_task = self.start_fit_animation(from_fit);

        self.update_all_dropdowns();
        Task::batch([
            exposure_task,
            color_task,
            fit_anim_task,
            cosmic::command::set_theme(self.config.app_theme.theme()),
        ])
    }

    // =========================================================================
    // System & Recovery Handlers
    // =========================================================================

    pub(crate) fn handle_camera_recovery_started(
        &self,
        attempt: u32,
        max_attempts: u32,
    ) -> Task<cosmic::Action<Message>> {
        info!(attempt, max_attempts, "Camera backend recovery started");
        Task::none()
    }

    pub(crate) fn handle_camera_recovery_succeeded(&self) -> Task<cosmic::Action<Message>> {
        info!("Camera backend recovery succeeded");
        Task::none()
    }

    pub(crate) fn handle_camera_recovery_failed(
        &self,
        error: String,
    ) -> Task<cosmic::Action<Message>> {
        error!(error = %error, "Camera backend recovery failed");
        Task::none()
    }

    pub(crate) fn handle_audio_recovery_started(
        &self,
        attempt: u32,
        max_attempts: u32,
    ) -> Task<cosmic::Action<Message>> {
        info!(attempt, max_attempts, "Audio backend recovery started");
        Task::none()
    }

    pub(crate) fn handle_audio_recovery_succeeded(&self) -> Task<cosmic::Action<Message>> {
        info!("Audio backend recovery succeeded");
        Task::none()
    }

    pub(crate) fn handle_audio_recovery_failed(
        &self,
        error: String,
    ) -> Task<cosmic::Action<Message>> {
        error!(error = %error, "Audio backend recovery failed");
        Task::none()
    }

    pub(crate) fn handle_generate_bug_report(&mut self) -> Task<cosmic::Action<Message>> {
        info!("Generating bug report...");

        // Ensure insights are up-to-date (normally only refreshed when the
        // Insights drawer is open, so they may be stale/empty here).
        let _ = self.handle_update_insights_metrics();

        let video_devices = self.available_cameras.clone();
        let audio_devices = self.available_audio_devices.clone();
        let video_encoders = self.available_video_encoders.clone();
        let save_folder_name = self.config.save_folder_name.clone();
        let insights = self.insights.clone();
        let config = self.config.clone();
        let snapshot = crate::bug_report::AppStateSnapshot {
            current_camera_index: self.current_camera_index,
            current_audio_device_index: self.current_audio_device_index,
            current_video_encoder_index: self.current_video_encoder_index,
        };

        Task::perform(
            async move {
                crate::bug_report::BugReportGenerator::generate(
                    &video_devices,
                    &audio_devices,
                    &video_encoders,
                    &save_folder_name,
                    &insights,
                    &config,
                    &snapshot,
                )
                .await
                .map(|p| p.display().to_string())
            },
            |result| cosmic::Action::App(Message::BugReportGenerated(result)),
        )
    }

    pub(crate) fn handle_bug_report_generated(
        &mut self,
        result: Result<String, String>,
    ) -> Task<cosmic::Action<Message>> {
        match result {
            Ok(path) => {
                info!(path = %path, "Bug report generated successfully");
                self.last_bug_report_path = Some(path);

                let url = &self.config.bug_report_url;
                if let Err(e) = open::that(url) {
                    error!(error = %e, url = %url, "Failed to open bug report URL");
                } else {
                    info!(url = %url, "Opened bug report URL");
                }
            }
            Err(err) => {
                error!(error = %err, "Failed to generate bug report");
            }
        }
        Task::none()
    }

    pub(crate) fn handle_show_bug_report(&self) -> Task<cosmic::Action<Message>> {
        if let Some(report_path) = &self.last_bug_report_path {
            info!(path = %report_path, "Showing bug report in file manager");
            if let Err(e) = Self::show_in_file_manager(report_path) {
                error!(error = %e, path = %report_path, "Failed to show bug report in file manager");
            }
        }
        Task::none()
    }

    // =========================================================================
    // Helper Functions
    // =========================================================================

    /// Show a file in the file manager with pre-selection
    pub(crate) fn show_in_file_manager(file_path: &str) -> Result<(), String> {
        use std::process::Command;

        let path = std::path::Path::new(file_path);
        let file_uri = format!("file://{}", path.display());

        // Method 1: Try D-Bus FileManager1.ShowItems
        let dbus_result = Command::new("dbus-send")
            .args([
                "--session",
                "--dest=org.freedesktop.FileManager1",
                "--type=method_call",
                "/org/freedesktop/FileManager1",
                "org.freedesktop.FileManager1.ShowItems",
                &format!("array:string:{}", file_uri),
                "string:",
            ])
            .output();

        if let Ok(output) = dbus_result
            && output.status.success()
        {
            info!("Opened file manager via D-Bus");
            return Ok(());
        }

        // Method 2: Try file manager-specific commands
        let file_managers = [
            ("nautilus", vec!["--select", file_path]),
            ("dolphin", vec!["--select", file_path]),
            ("nemo", vec![file_path]),
            ("caja", vec![file_path]),
            ("thunar", vec![file_path]),
        ];

        for (fm_name, args) in &file_managers {
            if let Ok(output) = Command::new(fm_name).args(args).spawn() {
                info!(file_manager = fm_name, "Opened file manager");
                drop(output);
                return Ok(());
            }
        }

        // Method 3: Fallback to opening the parent directory
        if let Some(parent) = path.parent()
            && let Ok(child) = Command::new("xdg-open").arg(parent).spawn()
        {
            info!("Opened parent directory as fallback");
            drop(child);
            return Ok(());
        }

        Err("Failed to open file manager".to_string())
    }

    // =========================================================================
    // QR Code Detection Handlers
    // =========================================================================

    pub(crate) fn handle_toggle_qr_detection(&mut self) -> Task<cosmic::Action<Message>> {
        self.qr_detection_enabled = !self.qr_detection_enabled;
        info!(enabled = self.qr_detection_enabled, "QR detection toggled");

        // Clear detections when disabling
        if !self.qr_detection_enabled {
            self.qr_detections.clear();
        }

        Task::none()
    }

    pub(crate) fn handle_qr_detections_updated(
        &mut self,
        detections: Vec<crate::app::frame_processor::QrDetection>,
    ) -> Task<cosmic::Action<Message>> {
        let count = detections.len();
        self.qr_detections = detections;
        self.last_qr_detection_time = Some(std::time::Instant::now());

        if count > 0 {
            info!(count, "QR detections updated");
        }

        Task::none()
    }

    pub(crate) fn handle_qr_open_url(&self, url: String) -> Task<cosmic::Action<Message>> {
        info!(url = %url, "Opening URL from QR code");
        match open::that_detached(&url) {
            Ok(()) => {
                info!("URL opened successfully");
            }
            Err(err) => {
                error!(url = %url, error = %err, "Failed to open URL");
            }
        }
        Task::none()
    }

    pub(crate) fn handle_qr_connect_wifi(
        &self,
        ssid: String,
        password: Option<String>,
        security: String,
        hidden: bool,
    ) -> Task<cosmic::Action<Message>> {
        // Use NetworkManager D-Bus API - works in both native and flatpak
        Task::perform(
            crate::network_manager::connect_wifi(ssid, password, security, hidden),
            |_| cosmic::Action::App(Message::Noop),
        )
    }

    pub(crate) fn handle_qr_copy_text(&self, text: String) -> Task<cosmic::Action<Message>> {
        info!(
            text_length = text.len(),
            "Copying text from QR code to clipboard"
        );

        // Use iced/cosmic clipboard API - works in both native and flatpak
        cosmic::iced::clipboard::write(text).map(|_: ()| cosmic::Action::App(Message::Noop))
    }

    // =========================================================================
    // Insights Handlers
    // =========================================================================

    pub(crate) fn handle_update_insights_metrics(&mut self) -> Task<cosmic::Action<Message>> {
        self.update_insights_pipeline();
        self.update_insights_backend();
        self.update_insights_measured_fps();
        self.update_insights_format_chain();
        self.update_insights_performance();
        self.update_insights_frame_metadata();
        self.update_insights_recording();

        Task::none()
    }

    fn update_insights_pipeline(&mut self) {
        use crate::app::insights::InsightsState;
        use crate::backends::camera::libcamera::native::diagnostics as diag;

        let new_pipeline = diag::get_pipeline_string();

        let pixel_format = self.active_format.as_ref().map(|f| f.pixel_format.as_str());
        if new_pipeline != self.insights.full_pipeline_string {
            let mjpeg_decoder = diag::get_mjpeg_decoder();
            if mjpeg_decoder.is_some() {
                self.insights.decoder_chain = vec![crate::app::insights::types::DecoderStatus {
                    name: "turbojpeg",
                    description: "libjpeg-turbo MJPEG decoder",
                    state: crate::app::insights::types::FallbackState::Selected,
                }];
            } else {
                self.insights.decoder_chain =
                    InsightsState::build_decoder_chain(pixel_format, new_pipeline.as_deref());
            }
            self.insights.full_pipeline_string = new_pipeline;
        }
    }

    fn update_insights_backend(&mut self) {
        use crate::app::insights::types::StreamInfo;
        use crate::backends::camera::libcamera::native::diagnostics as diag;

        self.insights.backend_type = "libcamera";
        if let Some(camera) = self.available_cameras.get(self.current_camera_index) {
            self.insights.pipeline_handler = camera.pipeline_handler.clone();
            self.insights.libcamera_version = camera.libcamera_version.clone();
            self.insights.sensor_model = camera.sensor_model.clone();
            self.insights.libcamera_multistream_capable = camera.pipeline_handler.is_some();

            // Populate V4L2 format list (refresh when camera changes)
            let current_v4l2_path = camera.v4l2_path().unwrap_or("").to_string();
            if self.insights.v4l2_formats_device != current_v4l2_path {
                self.insights.v4l2_formats_device = current_v4l2_path.clone();
                self.insights.v4l2_formats = if !current_v4l2_path.is_empty() {
                    crate::backends::camera::v4l2_utils::enumerate_v4l2_formats(&current_v4l2_path)
                } else {
                    Vec::new()
                };
                self.insights.libcamera_formats = self.available_formats.clone();
            }
        }

        self.insights.mjpeg_decoder = diag::get_mjpeg_decoder();
        self.insights.is_multistream = diag::get_is_multistream();

        if let Some((resolution, pixel_fmt, role, frame_count)) = diag::get_preview_stream_info() {
            self.insights.preview_stream = Some(StreamInfo {
                role,
                resolution,
                pixel_format: pixel_fmt,
                frame_count,
                source: "libcamera (native)".to_string(),
                ..Default::default()
            });
        }

        if let Some((resolution, pixel_fmt, role, frame_count, stride, height)) =
            diag::get_capture_stream_info()
        {
            let frame_size_bytes = stride as usize * height as usize;
            let gpu_processing = if pixel_fmt.contains("CSI2P") {
                format!(
                    "{} \u{2192} unpack (compute) \u{2192} demosaic (compute) \u{2192} RGBA",
                    pixel_fmt
                )
            } else if pixel_fmt.starts_with('S') && pixel_fmt.contains("GG") {
                // Unpacked Bayer (e.g. SRGGB10)
                format!("{} \u{2192} demosaic (compute) \u{2192} RGBA", pixel_fmt)
            } else {
                String::new()
            };
            self.insights.capture_stream = Some(StreamInfo {
                role,
                resolution,
                pixel_format: pixel_fmt,
                frame_count,
                source: "libcamera (native)".to_string(),
                gpu_processing,
                frame_size_bytes,
            });
        } else if !self.insights.is_multistream {
            self.insights.capture_stream = None;
        }
    }

    fn update_insights_measured_fps(&mut self) {
        let current_count = self
            .insights
            .preview_stream
            .as_ref()
            .map(|s| s.frame_count)
            .unwrap_or(0);
        let now = std::time::Instant::now();

        if let Some(prev_instant) = self.insights.prev_fps_instant {
            let elapsed = now.duration_since(prev_instant).as_secs_f64();
            let delta = current_count.saturating_sub(self.insights.prev_frame_count);
            if elapsed > 0.1 && delta > 0 {
                let instantaneous = delta as f64 / elapsed;
                // Exponential moving average for stability (alpha=0.3)
                self.insights.measured_fps = self.insights.measured_fps * 0.7 + instantaneous * 0.3;
            }
        }

        self.insights.prev_frame_count = current_count;
        self.insights.prev_fps_instant = Some(now);
    }

    fn update_insights_format_chain(&mut self) {
        use crate::backends::camera::libcamera::native::diagnostics as diag;

        let Some(format) = &self.active_format else {
            return;
        };

        let source = "libcamera (native)".to_string();

        let mjpeg_decoder = self.insights.mjpeg_decoder.as_ref();

        let mjpeg_decoded_fmt = if mjpeg_decoder.is_some() {
            diag::get_mjpeg_decoded_format()
        } else {
            None
        };

        // Use the actual negotiated format from the preview stream, not the
        // user-selected format. libcamera may negotiate a different pixel format
        // than what was selected (e.g., user selects YUYV but libcamera uses MJPEG
        // or NV12 for the ViewFinder stream).
        let actual_stream_format = self
            .insights
            .preview_stream
            .as_ref()
            .map(|s| s.pixel_format.as_str());

        let gpu_input_format = if let Some(ref fmt) = mjpeg_decoded_fmt {
            fmt.as_str()
        } else if mjpeg_decoder.is_some() {
            "I420"
        } else {
            actual_stream_format.unwrap_or(&format.pixel_format)
        };

        let wgpu_processing = match gpu_input_format {
            "I420" => "I420 (YUV 4:2:0) \u{2192} RGBA (compute shader)".to_string(),
            "I422" => "I422 (YUV 4:2:2) \u{2192} RGBA (compute shader)".to_string(),
            "I444" => "I444 (YUV 4:4:4) \u{2192} RGBA (compute shader)".to_string(),
            "Nv12" | "NV12" => "NV12 \u{2192} RGBA (compute shader)".to_string(),
            "YUYV" | "YUY2" | "Yuy2" => "YUYV \u{2192} RGBA (compute shader)".to_string(),
            "Rgba" | "RGBA" => "Passthrough".to_string(),
            other => format!("{} \u{2192} RGBA (compute shader)", other),
        };

        self.insights.format_chain.source = source;
        self.insights.format_chain.resolution = self
            .insights
            .preview_stream
            .as_ref()
            .map(|s| s.resolution.clone())
            .unwrap_or_else(|| format!("{}x{}", format.width, format.height));
        self.insights.format_chain.framerate = format
            .framerate
            .map(|fps| format!("{} fps", fps))
            .unwrap_or_else(|| "N/A".to_string());
        // Use actual negotiated format from stream, not user-selected format
        self.insights.format_chain.native_format = actual_stream_format
            .unwrap_or(&format.pixel_format)
            .to_string();
        self.insights.format_chain.wgpu_processing = wgpu_processing;

        if let Some(ref decoded) = mjpeg_decoded_fmt {
            let yuv_label = match decoded.as_str() {
                "I420" => "YUV 4:2:0",
                "I422" => "YUV 4:2:2",
                "I444" => "YUV 4:4:4",
                "I440" => "YUV 4:4:0",
                "Y800" => "Grayscale",
                other => other,
            };
            // Non-baseline entropy coding is the usual explanation for a high
            // CPU decode time, so name it right next to the decode step.
            let mjpeg_label = match diag::get_mjpeg_entropy_note() {
                Some(note) => format!("MJPEG ({note})"),
                None => "MJPEG".to_string(),
            };
            self.insights.cpu_processing = Some(format!(
                "{} \u{2192} {} (turbojpeg)",
                mjpeg_label, yuv_label
            ));
        } else if mjpeg_decoder.is_some() {
            self.insights.cpu_processing = Some("MJPEG \u{2192} YUV (turbojpeg)".to_string());
        } else {
            self.insights.cpu_processing = None;
        }
    }

    fn update_insights_performance(&mut self) {
        use crate::app::video_primitive;
        use crate::backends::camera::libcamera::native::diagnostics as diag;

        self.insights.cpu_decode_time_us = diag::get_mjpeg_decode_time_us();

        if let Some(stream) = &self.insights.preview_stream
            && let Some((w, h)) = stream.resolution.split_once('x')
            && let (Ok(w), Ok(h)) = (w.parse::<usize>(), h.parse::<usize>())
        {
            let bpp = if stream.pixel_format.contains("I420")
                || stream.pixel_format.contains("NV12")
            {
                1.5_f64
            } else if stream.pixel_format.contains("RGBA") || stream.pixel_format.contains("BGRA") {
                4.0
            } else if stream.pixel_format.contains("YUYV") {
                2.0
            } else if stream.pixel_format.contains("RGB24") {
                3.0
            } else {
                1.5
            };
            self.insights.frame_size_decoded = (w as f64 * h as f64 * bpp) as usize;
        }

        self.insights.gpu_conversion_time_us = video_primitive::get_gpu_upload_time_us();
        let gpu_frame_size = video_primitive::get_gpu_frame_size() as usize;

        if gpu_frame_size > 0 && self.insights.gpu_conversion_time_us > 10 {
            let bytes_per_sec = (gpu_frame_size as f64)
                / (self.insights.gpu_conversion_time_us as f64 / 1_000_000.0);
            self.insights.copy_bandwidth_mbps = bytes_per_sec / (1024.0 * 1024.0);
        } else {
            self.insights.copy_bandwidth_mbps = 0.0;
        }
    }

    fn update_insights_frame_metadata(&mut self) {
        self.insights.audio_levels = self
            .recording
            .audio_levels()
            .and_then(|levels| levels.lock().ok())
            .map(|l| l.clone());

        if let Some(frame) = &self.current_frame {
            self.insights.frame_latency_us = frame.captured_at.elapsed().as_micros() as u64;

            if let Some(meta) = &frame.libcamera_metadata {
                self.insights.has_libcamera_metadata = true;
                self.insights.meta_exposure_us = meta.exposure_time;
                self.insights.meta_analogue_gain = meta.analogue_gain;
                self.insights.meta_digital_gain = meta.digital_gain;
                self.insights.meta_colour_temperature = meta.colour_temperature;
                self.insights.meta_sequence = meta.sequence;
                self.insights.meta_colour_gains = meta.colour_gains;
                self.insights.meta_black_level = meta.black_level;
                self.insights.meta_lens_position = meta.lens_position;
                self.insights.meta_lux = meta.lux;
                self.insights.meta_focus_fom = meta.focus_fom;
                self.insights.frame_metadata = Some(meta.clone());
            }
        }
    }

    fn update_insights_recording(&mut self) {
        self.insights.recording_diag = crate::pipelines::video::get_recording_diagnostics();
        self.insights.recording_stats = if self.insights.recording_diag.is_some() {
            Some(crate::pipelines::video::get_recording_stats())
        } else {
            None
        };
    }

    pub(crate) fn handle_copy_pipeline_string(&self) -> Task<cosmic::Action<Message>> {
        if let Some(pipeline) = &self.insights.full_pipeline_string {
            info!("Copying pipeline string to clipboard");
            cosmic::iced::clipboard::write(pipeline.clone())
                .map(|_: ()| cosmic::Action::App(Message::Noop))
        } else {
            Task::none()
        }
    }

    // =========================================================================
    // Insights Capture (raw frame dump)
    // =========================================================================

    /// Capture raw binary frames from all running streams plus metadata as JSON.
    ///
    /// `frame_count`: 1 for single capture, 6 for burst.
    /// Saves to `{photos_dir}/insights/capture_{timestamp}_*.{bin,json}`.
    pub(crate) fn handle_insights_capture(
        &mut self,
        frame_count: usize,
    ) -> Task<cosmic::Action<Message>> {
        // Snapshot the insights state as a JSON-serialisable value
        let insights_json = self.build_insights_json();

        // Grab the current preview frame (viewfinder stream) immediately
        let preview_frame = self.current_frame.clone();

        // For the raw/capture stream we use the same still-capture mechanism as burst mode
        let is_multistream = self.is_current_camera_multistream();
        let still_requested = std::sync::Arc::clone(&self.still_capture_requested);
        let still_frame = std::sync::Arc::clone(&self.latest_still_frame);
        let still_frame_notify = std::sync::Arc::clone(&self.still_frame_notify);

        let save_dir =
            crate::app::get_photo_directory(&self.config.save_folder_name).join("insights");

        info!(frame_count, is_multistream, "Starting insights capture");

        Task::perform(
            async move {
                // Ensure output directory exists
                tokio::fs::create_dir_all(&save_dir)
                    .await
                    .map_err(|e| format!("Failed to create insights dir: {e}"))?;

                let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S_%3f");
                let mut saved_paths = Vec::new();

                // Save insights state JSON (once per capture session)
                let insights_path = save_dir.join(format!("capture_{timestamp}_insights.json"));
                tokio::fs::write(&insights_path, &insights_json)
                    .await
                    .map_err(|e| format!("Failed to write insights JSON: {e}"))?;
                saved_paths.push(insights_path.display().to_string());

                for i in 0..frame_count {
                    // e.g. "_1_of_6" for burst, "" for single
                    let suffix = if frame_count > 1 {
                        format!("_{}_of_{}", i + 1, frame_count)
                    } else {
                        String::new()
                    };

                    // --- Preview frame ---
                    // Snapshot from when capture was triggered (can't poll live
                    // frames from async task, but the preview frame is useful
                    // as a reference for each burst iteration's metadata).
                    if let Some(ref frame) = preview_frame {
                        let bin_path =
                            save_dir.join(format!("capture_{timestamp}_preview{suffix}.bin"));
                        tokio::fs::write(&bin_path, frame.data.as_ref())
                            .await
                            .map_err(|e| format!("Failed to write preview bin: {e}"))?;
                        saved_paths.push(bin_path.display().to_string());

                        let meta_path =
                            save_dir.join(format!("capture_{timestamp}_preview{suffix}_meta.json"));
                        let meta_json = frame_metadata_to_json(frame, "preview");
                        tokio::fs::write(&meta_path, meta_json)
                            .await
                            .map_err(|e| format!("Failed to write preview meta: {e}"))?;
                        saved_paths.push(meta_path.display().to_string());
                    }

                    // --- Raw/capture stream frame ---
                    if is_multistream {
                        still_requested.store(true, std::sync::atomic::Ordering::Release);

                        let timeout = std::time::Duration::from_secs(2);
                        let raw_frame = super::capture::wait_for_still_frame(
                            &still_frame,
                            &still_frame_notify,
                            timeout,
                        )
                        .await;
                        if raw_frame.is_none() {
                            info!(frame = i + 1, "Timeout waiting for raw frame, skipping");
                        }

                        if let Some(ref frame) = raw_frame {
                            let bin_path =
                                save_dir.join(format!("capture_{timestamp}_raw{suffix}.bin"));
                            tokio::fs::write(&bin_path, frame.data.as_ref())
                                .await
                                .map_err(|e| format!("Failed to write raw bin: {e}"))?;
                            saved_paths.push(bin_path.display().to_string());

                            let meta_path =
                                save_dir.join(format!("capture_{timestamp}_raw{suffix}_meta.json"));
                            let meta_json = frame_metadata_to_json(frame, "raw");
                            tokio::fs::write(&meta_path, meta_json)
                                .await
                                .map_err(|e| format!("Failed to write raw meta: {e}"))?;
                            saved_paths.push(meta_path.display().to_string());
                        }
                    }
                }

                info!(
                    count = saved_paths.len(),
                    dir = %save_dir.display(),
                    "Insights capture complete"
                );
                Ok(saved_paths)
            },
            |result| cosmic::Action::App(Message::InsightsCaptureComplete(result)),
        )
    }

    /// Build a JSON string capturing the entire current insights state.
    fn build_insights_json(&self) -> String {
        let ins = &self.insights;
        let mut map = serde_json::Map::new();

        // Pipeline
        if let Some(ref p) = ins.full_pipeline_string {
            map.insert("pipeline".into(), serde_json::Value::String(p.clone()));
        }
        map.insert(
            "decoder_chain".into(),
            serde_json::Value::Array(
                ins.decoder_chain
                    .iter()
                    .map(|d| {
                        serde_json::json!({
                            "name": d.name,
                            "description": d.description,
                            "state": format!("{:?}", d.state),
                        })
                    })
                    .collect(),
            ),
        );

        // Format chain
        map.insert(
            "format_chain".into(),
            serde_json::json!({
                "source": ins.format_chain.source,
                "resolution": ins.format_chain.resolution,
                "framerate": ins.format_chain.framerate,
                "native_format": ins.format_chain.native_format,
                "wgpu_processing": ins.format_chain.wgpu_processing,
            }),
        );

        // Performance
        map.insert(
            "performance".into(),
            serde_json::json!({
                "frame_latency_us": ins.frame_latency_us,
                "dropped_frames": ins.dropped_frames,
                "frame_size_decoded": ins.frame_size_decoded,
                "gpu_conversion_time_us": ins.gpu_conversion_time_us,
                "copy_time_us": ins.copy_time_us,
                "cpu_decode_time_us": ins.cpu_decode_time_us,
                "cpu_processing": ins.cpu_processing,
                "copy_bandwidth_mbps": ins.copy_bandwidth_mbps,
            }),
        );

        // Backend
        map.insert(
            "backend".into(),
            serde_json::json!({
                "type": ins.backend_type,
                "pipeline_handler": ins.pipeline_handler,
                "libcamera_version": ins.libcamera_version,
                "sensor_model": ins.sensor_model,
                "mjpeg_decoder": ins.mjpeg_decoder,
                "is_multistream": ins.is_multistream,
                "libcamera_multistream_capable": ins.libcamera_multistream_capable,
            }),
        );

        // Streams
        if let Some(ref s) = ins.preview_stream {
            map.insert(
                "preview_stream".into(),
                serde_json::json!({
                    "role": s.role,
                    "resolution": s.resolution,
                    "pixel_format": s.pixel_format,
                    "frame_count": s.frame_count,
                }),
            );
        }
        if let Some(ref s) = ins.capture_stream {
            map.insert(
                "capture_stream".into(),
                serde_json::json!({
                    "role": s.role,
                    "resolution": s.resolution,
                    "pixel_format": s.pixel_format,
                    "frame_count": s.frame_count,
                }),
            );
        }

        // Recording pipeline
        if let Some(ref rec) = ins.recording_diag {
            let mut rec_json = serde_json::json!({
                "mode": rec.mode,
                "encoder": rec.encoder,
                "resolution": rec.resolution,
                "framerate": rec.framerate,
                "pipeline_string": rec.pipeline_string,
            });
            if let Some(ref stats) = ins.recording_stats {
                rec_json["stats"] = serde_json::json!({
                    "capture_sent": stats.capture_sent,
                    "capture_dropped": stats.capture_dropped,
                    "pusher_pushed": stats.pusher_pushed,
                    "pusher_skipped": stats.pusher_skipped,
                    "channel_backlog": stats.channel_backlog,
                    "effective_fps": stats.effective_fps,
                    "last_pts_ms": stats.last_pts_ms,
                    "last_processing_delay_us": stats.last_processing_delay_us,
                    "last_convert_time_us": stats.last_convert_time_us,
                });
            }
            map.insert("recording_pipeline".into(), rec_json);
        }

        // Frame metadata
        map.insert(
            "frame_metadata".into(),
            serde_json::json!({
                "has_libcamera_metadata": ins.has_libcamera_metadata,
                "exposure_us": ins.meta_exposure_us,
                "analogue_gain": ins.meta_analogue_gain,
                "digital_gain": ins.meta_digital_gain,
                "colour_temperature": ins.meta_colour_temperature,
                "sequence": ins.meta_sequence,
                "colour_gains": ins.meta_colour_gains,
                "black_level": ins.meta_black_level,
                "lens_position": ins.meta_lens_position,
                "lux": ins.meta_lux,
                "focus_fom": ins.meta_focus_fom,
            }),
        );

        serde_json::to_string_pretty(&serde_json::Value::Object(map)).unwrap_or_default()
    }

    pub(crate) fn persist_config(&mut self) {
        if let Some(handler) = self.config_handler.as_ref()
            && let Err(err) = self.config.write_entry(handler)
        {
            tracing::warn!("Failed to persist config: {err}");
        }
    }
}

/// Serialize a single CameraFrame's metadata to JSON.
fn frame_metadata_to_json(
    frame: &crate::backends::camera::types::CameraFrame,
    stream_name: &str,
) -> String {
    let mut map = serde_json::Map::new();

    map.insert(
        "stream".into(),
        serde_json::Value::String(stream_name.to_string()),
    );
    map.insert("width".into(), serde_json::json!(frame.width));
    map.insert("height".into(), serde_json::json!(frame.height));
    map.insert(
        "format".into(),
        serde_json::Value::String(format!("{:?}", frame.format)),
    );
    map.insert("stride".into(), serde_json::json!(frame.stride));
    map.insert("data_size".into(), serde_json::json!(frame.data.len()));
    map.insert(
        "sensor_timestamp_ns".into(),
        serde_json::json!(frame.sensor_timestamp_ns),
    );

    if let Some(ref planes) = frame.yuv_planes {
        map.insert(
            "yuv_planes".into(),
            serde_json::json!({
                "y_offset": planes.y_offset,
                "y_size": planes.y_size,
                "uv_offset": planes.uv_offset,
                "uv_size": planes.uv_size,
                "uv_stride": planes.uv_stride,
                "v_offset": planes.v_offset,
                "v_size": planes.v_size,
                "v_stride": planes.v_stride,
                "uv_width": planes.uv_width,
                "uv_height": planes.uv_height,
            }),
        );
    }

    if let Some(ref meta) = frame.libcamera_metadata {
        map.insert(
            "libcamera_metadata".into(),
            serde_json::json!({
                "exposure_time_us": meta.exposure_time,
                "analogue_gain": meta.analogue_gain,
                "digital_gain": meta.digital_gain,
                "colour_temperature": meta.colour_temperature,
                "lens_position": meta.lens_position,
                "sensor_timestamp": meta.sensor_timestamp,
                "sequence": meta.sequence,
                "af_state": format!("{:?}", meta.af_state),
                "ae_state": format!("{:?}", meta.ae_state),
                "awb_state": format!("{:?}", meta.awb_state),
                "colour_gains": meta.colour_gains,
                "colour_correction_matrix": meta.colour_correction_matrix,
                "black_level": meta.black_level,
                "lux": meta.lux,
                "focus_fom": meta.focus_fom,
            }),
        );
    }

    serde_json::to_string_pretty(&serde_json::Value::Object(map)).unwrap_or_default()
}
