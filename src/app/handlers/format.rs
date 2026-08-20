// SPDX-License-Identifier: GPL-3.0-only

//! Format selection handlers
//!
//! Handles mode switching, resolution selection, framerate selection,
//! codec/pixel format selection, and format picker interactions.

use crate::app::state::{AppModel, CameraMode, FileSource, Message, RecordingState};
use crate::app::utils::{parse_codec, parse_resolution};
use cosmic::Task;
use cosmic::cosmic_config::CosmicConfigEntry;
use std::sync::Arc;
use tracing::{error, info, warn};

impl AppModel {
    // =========================================================================
    // Format Selection Handlers
    // =========================================================================

    /// Get the current camera's sensor rotation
    pub(crate) fn current_camera_rotation(&self) -> crate::backends::camera::types::SensorRotation {
        self.available_cameras
            .get(self.current_camera_index)
            .map(|c| c.rotation)
            .unwrap_or_default()
    }

    /// Compose sensor and display orientation for preview sampling. Mirroring
    /// reverses the apparent direction of a quarter-turn, so front-camera
    /// previews use the opposite display sign from unmirrored back cameras.
    pub(crate) fn preview_adjusted_rotation(
        &self,
        sensor_rotation: crate::backends::camera::types::SensorRotation,
        mirrored: bool,
    ) -> crate::backends::camera::types::SensorRotation {
        let display_degrees = self.display_orientation.degrees() as i32;
        let combined_degrees = if mirrored {
            sensor_rotation.degrees() as i32 + display_degrees
        } else {
            sensor_rotation.degrees() as i32 - display_degrees
        };
        crate::backends::camera::types::SensorRotation::from_degrees_int(combined_degrees)
    }

    pub(crate) fn preview_rotation(&self) -> crate::backends::camera::types::SensorRotation {
        self.preview_adjusted_rotation(self.current_camera_rotation(), self.should_mirror_preview())
    }

    /// Rotation baked into media so it remains upright in the physical device
    /// orientation in which capture started.
    pub(crate) fn capture_media_rotation(&self) -> crate::backends::camera::types::SensorRotation {
        let sensor_rotation = self.current_camera_rotation();
        let display_degrees = self.display_orientation.degrees() as i32;
        let is_back = self
            .available_cameras
            .get(self.current_camera_index)
            .and_then(|camera| camera.camera_location.as_deref())
            == Some("back");
        let degrees = if is_back {
            sensor_rotation.degrees() as i32 - display_degrees
        } else {
            sensor_rotation.degrees() as i32 + display_degrees
        };
        crate::backends::camera::types::SensorRotation::from_degrees_int(degrees)
    }

    /// The orientation used only to map on-screen capture geometry back into
    /// sensor coordinates.
    ///
    /// This mirrors exactly how `preview_video_config` (in
    /// `camera_preview/widget.rs`) composes the two rotations for the live
    /// preview. Crop geometry must use the SAME combined value - not just
    /// the sensor rotation - because `cover_capture_crop` inverse-maps an
    /// on-screen rect (whose axes reflect the sensor rotated by the extra
    /// display quarter-turn) through a Cover scale computed from the frame
    /// dimensions this rotation swaps. Using the sensor-only rotation there
    /// would swap dimensions inconsistently with what's on screen whenever
    /// `display_orientation` is non-zero, producing a wrongly cropped and
    /// wrongly cropped photo. This value must not be baked into saved media:
    /// the compositor transform is transient presentation state, while saved
    /// media uses only the camera's intrinsic sensor correction.
    ///
    /// At `Rotate0` (`display_orientation.degrees() == 0`) this is identical
    /// to `current_camera_rotation()`, so portrait capture is unaffected.
    ///
    /// Centralising the composition here (rather than repeating it at each
    /// capture call site) stops the sites from drifting out of sync, the
    /// same way `AppModel::map_bar_insets` centralises the bar-edge mapping.
    pub(crate) fn capture_geometry_rotation(
        &self,
    ) -> crate::backends::camera::types::SensorRotation {
        self.preview_rotation()
    }

    /// Cycle to the next or previous mode in the ordered mode list.
    pub(crate) fn handle_cycle_mode(&mut self, forward: bool) -> Task<cosmic::Action<Message>> {
        let modes = self.available_modes();
        let current_idx = modes.iter().position(|&m| m == self.mode).unwrap_or(0);
        let new_idx = if forward {
            if current_idx + 1 < modes.len() {
                current_idx + 1
            } else {
                return Task::none(); // Already at last mode
            }
        } else if current_idx > 0 {
            current_idx - 1
        } else {
            return Task::none(); // Already at first mode
        };
        self.handle_set_mode(modes[new_idx])
    }

    pub(crate) fn handle_set_mode(&mut self, mode: CameraMode) -> Task<cosmic::Action<Message>> {
        // Tapping the active mode chip a second time is the gesture that
        // opens the tools menu. The carousel sends `SetMode(self.mode)` in
        // that case rather than a separate message, so this branch is the
        // entry point for the double-tap shortcut — not a no-op.
        if self.mode == mode {
            return self.handle_toggle_tools_menu();
        }

        // Snapshot the animated values before mutating self.mode so the
        // cross-mode-boundary transition animates smoothly instead of
        // snapping. The View mode boundary changes the top scrim, bottom
        // scrim and capture-area placeholder; the Photo↔non-Photo boundary
        // changes the bottom scrim. Zoom is captured separately because
        // it's driven by an independent animation.
        let from_fit = self.capture_fit_state();
        let from_zoom = self.current_zoom_level();

        self.haptic_tap();

        // When switching away from Virtual mode with a playing video, pause it first
        if self.mode == CameraMode::Virtual
            && matches!(self.virtual_camera_file_source, Some(FileSource::Video(_)))
            && !self.video_file_paused
        {
            info!("Pausing video preview before mode switch");
            self.stop_video_preview_playback();
            self.video_file_paused = true;
        }

        if self.recording.is_recording() {
            if let Some(sender) = self.recording.take_stop_sender() {
                let _ = sender.send(());
            }
            self.recording = RecordingState::Idle;
        }

        // Stop timelapse if active (dropping sender closes encoder channel)
        if self.timelapse.is_active() {
            info!("Stopping timelapse due to mode switch");
            self.timelapse = crate::app::state::TimelapseState::Idle;
        }

        // Skip blur transition and camera restart when a file source is active
        // (no camera stream to restart, blur would never resolve)
        let file_source_active = self.virtual_camera_file_source.is_some();

        // Reset filter when switching to Virtual mode (filters supported in Photo and Video)
        if mode == CameraMode::Virtual
            && self.selected_filter != crate::app::state::FilterType::Standard
        {
            self.selected_filter = crate::app::state::FilterType::Standard;
        }
        // Close the filter drawer if switching to a mode that doesn't support it
        if mode == CameraMode::Virtual
            && self.context_page == crate::app::state::ContextPage::Filters
            && self.core.window.show_context
        {
            self.core.window.show_context = false;
        }

        // Select format from cached data — never blocks on hardware.
        // If the format changes, format_id in the subscription key changes,
        // causing the subscription to restart automatically (with blur if needed).
        let old_format = self.active_format.clone();
        self.mode = mode;
        self.zoom_level = 1.0; // Reset zoom when switching modes
        self.select_format_from_cache(mode);

        // Kick off a fit/fill animation if any animated value differs from
        // where the eye currently is. start_fit_animation handles the no-op
        // case and reuses any in-flight tick chain.
        let fit_anim_task = self.start_fit_animation(from_fit);

        // If we just left Photo mode while zoomed, animate the zoom back to
        // 1× so the preview eases out instead of snapping. zoom_level was
        // already set to 1.0 above; we just need to install the animation
        // and schedule a tick.
        let zoom_anim_task = if (from_zoom - 1.0).abs() > 0.001 {
            let was_idle = self.zoom_animation.is_none();
            self.zoom_animation = Some(crate::app::state::ZoomAnimation {
                start: std::time::Instant::now(),
                from: from_zoom,
            });
            if was_idle {
                Self::delay_task(16, Message::ZoomAnimationTick)
            } else {
                Task::none()
            }
        } else {
            Task::none()
        };

        // All modes use the same stream layout (ViewFinder+Raw), so no pipeline
        // restart is needed unless the format itself changed.
        if !file_source_active && self.active_format != old_format {
            info!("Mode switch: format changed — restarting pipeline");
            self.start_blur_transition();
        } else {
            info!("Mode switch: no pipeline restart needed — keeping stream");
        }

        // When switching to Virtual mode with a file source, restore the file source preview
        if mode == CameraMode::Virtual
            && let Some(ref source) = self.virtual_camera_file_source
        {
            let path = match source {
                FileSource::Image(p) | FileSource::Video(p) => p.clone(),
            };
            let is_video = matches!(source, FileSource::Video(_));
            // For video files, use the stored seek position to restore at the correct frame
            let seek_position = if is_video {
                self.video_preview_seek_position
            } else {
                0.0
            };
            info!(
                is_video,
                seek_position, "Restoring file source preview after mode switch"
            );

            let preview_task = Task::perform(
                async move {
                    use crate::backends::virtual_camera::{
                        get_video_duration, load_preview_frame, load_video_frame_at_position,
                    };

                    // For video files with a seek position, load frame at that position
                    // Otherwise load the first frame
                    let frame = if is_video && seek_position > 0.0 {
                        match load_video_frame_at_position(&path, seek_position) {
                            Ok(frame) => Some(Arc::new(frame)),
                            Err(e) => {
                                warn!(?e, "Failed to load video frame at position");
                                // Fall back to first frame
                                load_preview_frame(&path).ok().map(Arc::new)
                            }
                        }
                    } else {
                        match load_preview_frame(&path) {
                            Ok(frame) => Some(Arc::new(frame)),
                            Err(e) => {
                                warn!(?e, "Failed to load preview frame");
                                None
                            }
                        }
                    };

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
            return Task::batch([preview_task, fit_anim_task, zoom_anim_task]);
        }

        // Note: we don't call save_settings() here to avoid blocking the UI
        // on eMMC writes (~150ms on phone). Settings are saved when the user
        // explicitly changes format/resolution, or on app exit.

        // Re-query exposure controls when pipeline restarts
        if self.active_format != old_format {
            return Task::batch([
                self.query_exposure_controls_task(),
                fit_anim_task,
                zoom_anim_task,
            ]);
        }

        Task::batch([fit_anim_task, zoom_anim_task])
    }

    pub(crate) fn handle_select_mode(&mut self, index: usize) -> Task<cosmic::Action<Message>> {
        if let Some(format) = self.mode_list.get(index).cloned() {
            info!(
                width = format.width,
                height = format.height,
                framerate = ?format.framerate,
                pixel_format = %format.pixel_format,
                "Switching to mode from consolidated dropdown"
            );
            self.change_format(format);
            self.start_blur_transition();

            // Re-query exposure controls to reset to defaults for new format
            return self.query_exposure_controls_task();
        }
        Task::none()
    }

    pub(crate) fn handle_select_pixel_format(
        &mut self,
        pixel_format: String,
    ) -> Task<cosmic::Action<Message>> {
        info!(pixel_format = %pixel_format, "Switching to pixel format");
        self.change_pixel_format(pixel_format);
        self.start_blur_transition();

        // Re-query exposure controls to get fresh defaults for new format
        self.query_exposure_controls_task()
    }

    pub(crate) fn handle_select_resolution(
        &mut self,
        resolution_str: String,
    ) -> Task<cosmic::Action<Message>> {
        if let Some((width, height)) = parse_resolution(&resolution_str) {
            info!(width, height, "Switching to resolution");
            self.change_resolution(width, height);
            self.zoom_level = 1.0; // Reset zoom when changing resolution
            self.start_blur_transition();

            // Re-query exposure controls to get fresh defaults for new resolution
            return self.query_exposure_controls_task();
        }
        Task::none()
    }

    pub(crate) fn handle_select_framerate(
        &mut self,
        framerate_str: String,
    ) -> Task<cosmic::Action<Message>> {
        // Handle "Auto" for VFR (variable framerate) - libcamera manages dynamically
        if framerate_str == "Auto" {
            info!("Switching to VFR (Auto framerate - libcamera managed)");
            self.change_framerate_optional(None);
            self.start_blur_transition();
            return self.query_exposure_controls_task();
        }

        if let Ok(fps) = framerate_str.parse::<u32>() {
            info!(fps, "Switching to framerate");
            self.change_framerate_optional(Some(fps));
            self.start_blur_transition();

            // Re-query exposure controls to get fresh defaults for new framerate
            return self.query_exposure_controls_task();
        }
        Task::none()
    }

    pub(crate) fn handle_select_codec(
        &mut self,
        codec_str: String,
    ) -> Task<cosmic::Action<Message>> {
        let pixel_format = parse_codec(&codec_str);
        info!(pixel_format = %pixel_format, "Switching to codec");
        self.change_pixel_format(pixel_format);

        // Re-query exposure controls to get fresh defaults for new codec
        self.query_exposure_controls_task()
    }

    pub(crate) fn handle_picker_select_resolution(
        &mut self,
        width: u32,
    ) -> Task<cosmic::Action<Message>> {
        self.picker_selected_resolution = Some(width);
        let current_fps = self.active_format.as_ref().and_then(|f| f.framerate);

        let matching_formats: Vec<(usize, &crate::backends::camera::types::CameraFormat)> = self
            .available_formats
            .iter()
            .enumerate()
            .filter(|(_, fmt)| fmt.width == width)
            .collect();

        if !matching_formats.is_empty() {
            let format_to_apply = if let Some(target_fps) = current_fps {
                let target_int = target_fps.as_int() as i32;
                matching_formats
                    .iter()
                    .find(|(_, fmt)| fmt.framerate == Some(target_fps))
                    .or_else(|| {
                        matching_formats
                            .iter()
                            .filter(|(_, fmt)| fmt.framerate.is_some())
                            .min_by_key(|(_, fmt)| {
                                let fps = fmt.framerate.unwrap().as_int() as i32;
                                (fps - target_int).abs()
                            })
                    })
                    .or_else(|| matching_formats.first())
            } else {
                matching_formats.first()
            };

            if let Some(&(index, _)) = format_to_apply {
                self.active_format = self.available_formats.get(index).cloned();

                if let Some(fmt) = &self.active_format {
                    info!(width, format = %fmt, "Applied resolution with framerate preservation");
                    self.photo_aspect_ratio = self.config.photo_aspect_ratio;
                }
                self.zoom_level = 1.0; // Reset zoom when changing resolution
                self.save_settings();
                self.start_blur_transition();
            }
        }
        Task::none()
    }

    pub(crate) fn handle_picker_select_format(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
        if index < self.available_formats.len() {
            self.active_format = self.available_formats.get(index).cloned();
            self.format_picker_visible = false;

            if let Some(fmt) = &self.active_format {
                info!(format = %fmt, "Selected format from picker");
                self.photo_aspect_ratio = self.config.photo_aspect_ratio;
            }
            self.zoom_level = 1.0; // Reset zoom when changing format
            self.save_settings();
            self.start_blur_transition();

            // Re-query exposure controls to reset to defaults for new format
            return self.query_exposure_controls_task();
        }
        Task::none()
    }

    pub(crate) fn handle_select_bitrate_preset(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
        if index < crate::constants::BitratePreset::ALL.len() {
            let preset = crate::constants::BitratePreset::ALL[index];
            info!(preset = ?preset, "Selected bitrate preset");
            self.config.bitrate_preset = preset;

            if let Some(handler) = self.config_handler.as_ref()
                && let Err(err) = self.config.write_entry(handler)
            {
                error!(?err, "Failed to save bitrate preset setting");
            }
        }
        Task::none()
    }
}

#[cfg(test)]
mod orientation_tests {
    use super::*;
    use crate::backends::camera::types::{CameraDevice, SensorRotation};
    use crate::backends::display_orientation::DisplayOrientation;

    fn landscape_model(sensor_rotation: SensorRotation, location: &str) -> AppModel {
        AppModel {
            available_cameras: vec![CameraDevice {
                rotation: sensor_rotation,
                camera_location: Some(location.to_string()),
                ..CameraDevice::default()
            }],
            display_orientation: DisplayOrientation::Rotate270,
            ..AppModel::default()
        }
    }

    #[test]
    fn landscape_preview_rotation_accounts_for_front_camera_mirroring() {
        let mut front = landscape_model(SensorRotation::Rotate90, "front");
        front.config.mirror_preview = true;
        assert_eq!(front.preview_rotation(), SensorRotation::None);

        let back = landscape_model(SensorRotation::Rotate270, "back");
        assert_eq!(back.preview_rotation(), SensorRotation::None);
    }

    #[test]
    fn landscape_capture_rotation_matches_the_physical_device_orientation() {
        let front = landscape_model(SensorRotation::Rotate90, "front");
        assert_eq!(front.capture_media_rotation(), SensorRotation::None);

        let back = landscape_model(SensorRotation::Rotate270, "back");
        assert_eq!(back.capture_media_rotation(), SensorRotation::None);
    }

    #[test]
    fn portrait_keeps_each_cameras_sensor_correction() {
        let mut front = landscape_model(SensorRotation::Rotate90, "front");
        front.display_orientation = DisplayOrientation::Rotate0;
        front.config.mirror_preview = true;
        assert_eq!(front.preview_rotation(), SensorRotation::Rotate90);
        assert_eq!(front.capture_geometry_rotation(), SensorRotation::Rotate90);
        assert_eq!(front.capture_media_rotation(), SensorRotation::Rotate90);

        let mut back = landscape_model(SensorRotation::Rotate270, "back");
        back.display_orientation = DisplayOrientation::Rotate0;
        assert_eq!(back.preview_rotation(), SensorRotation::Rotate270);
        assert_eq!(back.capture_geometry_rotation(), SensorRotation::Rotate270);
        assert_eq!(back.capture_media_rotation(), SensorRotation::Rotate270);
    }

    #[test]
    fn opposite_landscape_direction_is_upright_for_front_and_back() {
        let mut front = landscape_model(SensorRotation::Rotate90, "front");
        front.display_orientation = DisplayOrientation::Rotate90;
        front.config.mirror_preview = true;
        assert_eq!(front.preview_rotation(), SensorRotation::Rotate180);
        assert_eq!(front.capture_media_rotation(), SensorRotation::Rotate180);

        let mut back = landscape_model(SensorRotation::Rotate270, "back");
        back.display_orientation = DisplayOrientation::Rotate90;
        assert_eq!(back.preview_rotation(), SensorRotation::Rotate180);
        assert_eq!(back.capture_media_rotation(), SensorRotation::Rotate180);
    }
}
