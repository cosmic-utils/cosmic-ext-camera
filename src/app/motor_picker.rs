// SPDX-License-Identifier: GPL-3.0-only

//! Motor/PTZ controls picker overlay
//!
//! Provides a floating overlay for camera motor controls including:
//! - V4L2 pan/tilt/zoom controls (for PTZ cameras)

use crate::app::overlay_style::PICKER_PANEL;
use crate::app::state::{AppModel, Message};
use crate::backends::camera::v4l2_controls;
use crate::fl;
use cosmic::Element;
use cosmic::iced::Length;
use cosmic::widget;
use tracing::warn;

/// Width of the motor picker panel
const PICKER_PANEL_WIDTH: f32 = 280.0;

impl AppModel {
    /// Check if any motor controls are available (V4L2 PTZ)
    pub fn has_motor_controls(&self) -> bool {
        self.available_exposure_controls.has_any_ptz()
    }

    /// Build the motor controls picker overlay
    pub fn build_motor_picker(&self) -> Element<'_, Message> {
        let spacing = cosmic::theme::spacing();

        let mut column = widget::Column::new()
            .spacing(spacing.space_s)
            .padding(spacing.space_s)
            .width(Length::Shrink);

        // Title with reset button
        let title_row = widget::Row::new()
            .push(widget::text::heading(fl!("ptz-title")).width(Length::Fill))
            .push(
                widget::button::icon(widget::icon::from_name("edit-undo-symbolic"))
                    .on_press(Message::ResetPanTilt)
                    .class(cosmic::theme::Button::Text)
                    .padding(4),
            )
            .align_y(cosmic::iced::Alignment::Center);
        column = column.push(title_row);

        // V4L2 Pan control
        if self.available_exposure_controls.pan_absolute.available {
            let range = &self.available_exposure_controls.pan_absolute;
            let current = self.get_v4l2_pan_value().unwrap_or(range.default);
            let pan_row = widget::Row::new()
                .push(widget::text::body(fl!("ptz-pan")).width(Length::Fixed(60.0)))
                .push(
                    widget::slider(range.min..=range.max, current, Message::SetPanAbsolute)
                        .width(Length::Fixed(140.0)),
                )
                .push(widget::text::body(format_arc_seconds(current)).width(Length::Fixed(40.0)))
                .spacing(spacing.space_xs)
                .align_y(cosmic::iced::Alignment::Center);
            column = column.push(pan_row);
        }

        // V4L2 Tilt control
        if self.available_exposure_controls.tilt_absolute.available {
            let range = &self.available_exposure_controls.tilt_absolute;
            let current = self.get_v4l2_tilt_value().unwrap_or(range.default);
            let tilt_row = widget::Row::new()
                .push(widget::text::body(fl!("ptz-tilt")).width(Length::Fixed(60.0)))
                .push(
                    widget::slider(range.min..=range.max, current, Message::SetTiltAbsolute)
                        .width(Length::Fixed(140.0)),
                )
                .push(widget::text::body(format_arc_seconds(current)).width(Length::Fixed(40.0)))
                .spacing(spacing.space_xs)
                .align_y(cosmic::iced::Alignment::Center);
            column = column.push(tilt_row);
        }

        // Note: Zoom control is handled separately via normal camera zoom
        // (hardware zoom is preferred, shader zoom is fallback)

        // Build picker panel with semi-transparent themed background. When
        // frosted glass is active, `frosted_panel` stacks a live-blurred preview
        // backdrop behind the translucent panel.
        let picker_content = widget::scrollable(column).height(
            crate::app::view::picker_panel_height(self.controls_are_sideways(), self.screen_height),
        );
        let picker_panel = widget::mouse_area(
            widget::container(self.frosted_panel(picker_content.into(), PICKER_PANEL))
                .width(Length::Fixed(PICKER_PANEL_WIDTH)),
        )
        .on_press(Message::Noop);

        // Anchor to the top bar's inner edge (below it in portrait, off the side
        // bar in landscape), sharing the same bar-anchor helper as the tools menu
        // and the other pickers.
        self.anchor_bar_popup(picker_panel.into(), Message::CloseMotorPicker)
    }

    /// Get current V4L2 pan value
    fn get_v4l2_pan_value(&self) -> Option<i32> {
        let path = self.get_v4l2_device_path()?;
        v4l2_controls::get_control(&path, v4l2_controls::V4L2_CID_PAN_ABSOLUTE)
    }

    /// Get current V4L2 tilt value
    fn get_v4l2_tilt_value(&self) -> Option<i32> {
        let path = self.get_v4l2_device_path()?;
        v4l2_controls::get_control(&path, v4l2_controls::V4L2_CID_TILT_ABSOLUTE)
    }

    /// Get current V4L2 zoom value
    pub fn get_v4l2_zoom_value(&self) -> Option<i32> {
        let path = self.get_v4l2_device_path()?;
        v4l2_controls::get_control(&path, v4l2_controls::V4L2_CID_ZOOM_ABSOLUTE)
    }

    /// Set V4L2 pan absolute position
    pub fn set_v4l2_pan(&self, value: i32) {
        if let Some(path) = self.get_v4l2_device_path()
            && let Err(e) =
                v4l2_controls::set_control(&path, v4l2_controls::V4L2_CID_PAN_ABSOLUTE, value)
        {
            warn!("Failed to set pan: {}", e);
        }
    }

    /// Set V4L2 tilt absolute position
    pub fn set_v4l2_tilt(&self, value: i32) {
        if let Some(path) = self.get_v4l2_device_path()
            && let Err(e) =
                v4l2_controls::set_control(&path, v4l2_controls::V4L2_CID_TILT_ABSOLUTE, value)
        {
            warn!("Failed to set tilt: {}", e);
        }
    }

    /// Set V4L2 zoom absolute position
    pub fn set_v4l2_zoom(&self, value: i32) {
        if let Some(path) = self.get_v4l2_device_path()
            && let Err(e) =
                v4l2_controls::set_control(&path, v4l2_controls::V4L2_CID_ZOOM_ABSOLUTE, value)
        {
            warn!("Failed to set zoom: {}", e);
        }
    }

    /// Reset pan/tilt to default positions
    pub fn reset_pan_tilt(&self) {
        // Reset V4L2 pan/tilt if available
        // Note: Some cameras (like OBSBOT) only process one motor command at a time.
        // We spawn a thread to send commands with a small delay between them.
        if let Some(path) = self.get_v4l2_device_path() {
            let path = path.to_string();
            let pan_available = self.available_exposure_controls.pan_absolute.available;
            let pan_default = self.available_exposure_controls.pan_absolute.default;
            let tilt_available = self.available_exposure_controls.tilt_absolute.available;
            let tilt_default = self.available_exposure_controls.tilt_absolute.default;

            // Spawn thread to avoid blocking UI
            // Some cameras (like OBSBOT) drop commands sent in quick succession,
            // so we send each command twice with small delays to ensure both are processed
            std::thread::spawn(move || {
                for _ in 0..2 {
                    if tilt_available {
                        let _ = v4l2_controls::set_control(
                            &path,
                            v4l2_controls::V4L2_CID_TILT_ABSOLUTE,
                            tilt_default,
                        );
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    if pan_available {
                        let _ = v4l2_controls::set_control(
                            &path,
                            v4l2_controls::V4L2_CID_PAN_ABSOLUTE,
                            pan_default,
                        );
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            });
        }
    }
}

/// Format arc seconds value to degrees for display
fn format_arc_seconds(arc_seconds: i32) -> String {
    let degrees = arc_seconds as f32 / 3600.0;
    format!("{:.1}°", degrees)
}
