// SPDX-License-Identifier: GPL-3.0-only

//! Exposure picker UI view
//!
//! Floating overlay for exposure controls with semi-transparent background.

use super::ControlRange;
use crate::app::overlay_style::PICKER_PANEL;
use crate::app::state::{AppModel, Message};
use crate::fl;
use cosmic::Element;
use cosmic::iced::{Alignment, Length};
use cosmic::widget;

// UI Constants
const PICKER_PANEL_WIDTH: f32 = 260.0;
const COLOR_PICKER_WIDTH: f32 = 280.0;
const LABEL_WIDTH: f32 = 70.0;
const SLIDER_WIDTH_EXPOSURE: f32 = 100.0;
const SLIDER_WIDTH_COLOR: f32 = 120.0;
const VALUE_WIDTH_EXPOSURE: f32 = 50.0;
const VALUE_WIDTH_COLOR: f32 = 40.0;
const CONTROL_SPACING: u16 = 8;

impl AppModel {
    /// Build the exposure picker overlay
    pub fn build_exposure_picker(&self) -> Element<'_, Message> {
        let spacing = cosmic::theme::spacing();
        let settings_data = self.exposure_settings.as_ref();
        let current_mode = settings_data.map(|s| s.mode).unwrap_or_default();
        let is_manual = current_mode == super::ExposureMode::Manual;

        let mut column = widget::Column::new()
            .spacing(spacing.space_xs)
            .padding(spacing.space_s)
            .width(Length::Shrink);

        // Mode toggle row (segmented button + reset icon)
        column = column.push(self.build_mode_toggle());

        // Add controls based on mode
        if is_manual {
            column = self.add_manual_controls(column, settings_data);
        } else {
            column = self.add_auto_controls(column, settings_data);
        }

        // Build picker panel with semi-transparent themed background
        let picker_content = widget::scrollable(column).height(
            crate::app::view::picker_panel_height(self.controls_are_sideways(), self.screen_height),
        );
        let picker_panel = widget::mouse_area(
            widget::container(self.frosted_panel(picker_content.into(), PICKER_PANEL))
                .width(Length::Fixed(PICKER_PANEL_WIDTH)),
        )
        .on_press(Message::Noop);

        // Anchor to the top bar's inner edge (below it in portrait, off the side
        // bar in landscape), shared with the tools menu and the color picker.
        self.anchor_bar_popup(picker_panel.into(), Message::CloseExposurePicker)
    }

    /// Build mode toggle row (Auto/Manual)
    fn build_mode_toggle(&self) -> Element<'_, Message> {
        let toggle = widget::segmented_button::horizontal(&self.exposure_mode_model)
            .button_alignment(Alignment::Center)
            .width(Length::Fill)
            .on_activate(Message::ExposureModeSelected);

        let reset_btn = widget::button::icon(widget::icon::from_name("edit-undo-symbolic"))
            .on_press(Message::ResetExposureSettings)
            .class(cosmic::theme::Button::Text)
            .padding(4);

        widget::Row::new()
            .push(toggle)
            .push(reset_btn)
            .spacing(CONTROL_SPACING)
            .align_y(Alignment::Center)
            .into()
    }

    /// Add auto mode controls to column
    fn add_auto_controls<'a>(
        &'a self,
        mut column: widget::Column<'a, Message, cosmic::Theme>,
        settings_data: Option<&'a super::ExposureSettings>,
    ) -> widget::Column<'a, Message, cosmic::Theme> {
        let controls = &self.available_exposure_controls;

        if controls.exposure_bias.available {
            column = column.push(self.build_ev_row(settings_data));
        }
        if controls.backlight_compensation.available {
            column = column.push(self.build_backlight_row(settings_data));
        }

        // Focus controls (available in both auto and manual modes)
        column = self.add_focus_controls(column, settings_data);

        column
    }

    /// Add manual mode controls to column (shows all controls, disabled if unsupported)
    fn add_manual_controls<'a>(
        &'a self,
        mut column: widget::Column<'a, Message, cosmic::Theme>,
        settings_data: Option<&'a super::ExposureSettings>,
    ) -> widget::Column<'a, Message, cosmic::Theme> {
        let controls = &self.available_exposure_controls;

        // Exposure Time
        if controls.exposure_time.available {
            column = column.push(self.build_exposure_time_row(settings_data));
        } else {
            column = column.push(Self::build_unsupported_row(fl!("exposure-time")));
        }

        // Gain
        if controls.gain.available {
            column = column.push(self.build_gain_row(settings_data));
        } else {
            column = column.push(Self::build_unsupported_row(fl!("exposure-gain")));
        }

        // ISO
        if controls.iso.available {
            column = column.push(self.build_iso_row(settings_data));
        } else {
            column = column.push(Self::build_unsupported_row(fl!("exposure-iso")));
        }

        // EV Compensation
        if controls.exposure_bias.available {
            column = column.push(self.build_ev_row(settings_data));
        } else {
            column = column.push(Self::build_unsupported_row(fl!("exposure-ev")));
        }

        // Metering Mode
        if controls.has_metering && !controls.metering_modes.is_empty() {
            column = column.push(self.build_metering_row(settings_data));
        } else {
            column = column.push(Self::build_unsupported_row(fl!("exposure-metering")));
        }

        // Auto Priority
        if controls.has_auto_priority {
            column = column.push(self.build_auto_priority_row(settings_data));
        } else {
            column = column.push(Self::build_unsupported_row(fl!("exposure-auto-priority")));
        }

        // Focus controls (available in both auto and manual modes)
        column = self.add_focus_controls(column, settings_data);

        column
    }

    // =========================================================================
    // Control row builders
    // =========================================================================

    /// Build a control row with label, slider, and value display
    fn build_control_row<'a>(
        label: String,
        value_text: String,
        value_width: f32,
        slider: impl Into<Element<'a, Message>>,
    ) -> Element<'a, Message> {
        widget::row::with_capacity(3)
            .align_y(Alignment::Center)
            .spacing(CONTROL_SPACING)
            .width(Length::Shrink)
            .push(
                widget::text(label)
                    .size(13)
                    .width(Length::Fixed(LABEL_WIDTH)),
            )
            .push(slider.into())
            .push(
                widget::text::body(value_text)
                    .width(Length::Fixed(value_width))
                    .align_x(Alignment::End),
            )
            .into()
    }

    /// Build a generic slider row using ControlRange
    fn build_slider_row<'a, F>(
        label: String,
        current: i32,
        range: &ControlRange,
        slider_width: f32,
        value_width: f32,
        format_value: impl Fn(i32) -> String,
        message_fn: F,
    ) -> Element<'a, Message>
    where
        F: 'a + Fn(i32) -> Message,
    {
        let min = range.min as f32;
        let max = range.max as f32;
        let clamped = (current as f32).clamp(min, max);

        let slider = widget::slider(min..=max, clamped, move |v| message_fn(v as i32))
            .width(Length::Fixed(slider_width));

        Self::build_control_row(label, format_value(current), value_width, slider)
    }

    /// Build a row for unsupported controls
    fn build_unsupported_row(label: String) -> Element<'static, Message> {
        widget::row::with_capacity(2)
            .align_y(Alignment::Center)
            .spacing(CONTROL_SPACING)
            .width(Length::Shrink)
            .push(
                widget::text(label)
                    .size(13)
                    .width(Length::Fixed(LABEL_WIDTH)),
            )
            .push(widget::text(fl!("exposure-not-supported")).size(13))
            .into()
    }

    /// Build EV compensation row
    fn build_ev_row(
        &self,
        settings_data: Option<&super::ExposureSettings>,
    ) -> Element<'_, Message> {
        let controls = &self.available_exposure_controls;
        let current = settings_data.map(|s| s.exposure_compensation).unwrap_or(0);
        let range = &controls.exposure_bias;

        let min = range.min as f32;
        let max = range.max as f32;
        let clamped = (current as f32).clamp(min, max);

        let slider = widget::slider(min..=max, clamped, |v| {
            Message::SetExposureCompensation(v as i32)
        })
        .width(Length::Fixed(SLIDER_WIDTH_EXPOSURE))
        .breakpoints(&[0.0]);

        Self::build_control_row(
            fl!("exposure-ev"),
            format!("{:+.1}", current as f32 / 1000.0),
            VALUE_WIDTH_EXPOSURE,
            slider,
        )
    }

    /// Build backlight compensation row
    fn build_backlight_row(
        &self,
        settings_data: Option<&super::ExposureSettings>,
    ) -> Element<'_, Message> {
        let range = &self.available_exposure_controls.backlight_compensation;
        let current = settings_data
            .and_then(|s| s.backlight_compensation)
            .unwrap_or(range.default);

        Self::build_slider_row(
            fl!("exposure-backlight"),
            current,
            range,
            SLIDER_WIDTH_EXPOSURE,
            VALUE_WIDTH_EXPOSURE,
            |v| format!("{}", v),
            Message::SetBacklightCompensation,
        )
    }

    /// Build exposure time row
    fn build_exposure_time_row(
        &self,
        settings_data: Option<&super::ExposureSettings>,
    ) -> Element<'_, Message> {
        let range = &self.available_exposure_controls.exposure_time;
        let current = settings_data
            .and_then(|s| s.exposure_time)
            .unwrap_or(range.default);

        Self::build_slider_row(
            fl!("exposure-time"),
            current,
            range,
            SLIDER_WIDTH_EXPOSURE,
            VALUE_WIDTH_EXPOSURE,
            format_exposure_time,
            Message::SetExposureTime,
        )
    }

    /// Build gain row
    fn build_gain_row(
        &self,
        settings_data: Option<&super::ExposureSettings>,
    ) -> Element<'_, Message> {
        let range = &self.available_exposure_controls.gain;
        let current = settings_data.and_then(|s| s.gain).unwrap_or(range.default);

        Self::build_slider_row(
            fl!("exposure-gain"),
            current,
            range,
            SLIDER_WIDTH_EXPOSURE,
            VALUE_WIDTH_EXPOSURE,
            |v| format!("{}", v),
            Message::SetGain,
        )
    }

    /// Build ISO row
    fn build_iso_row(
        &self,
        settings_data: Option<&super::ExposureSettings>,
    ) -> Element<'_, Message> {
        let range = &self.available_exposure_controls.iso;
        let current = settings_data.and_then(|s| s.iso).unwrap_or(range.default);

        Self::build_slider_row(
            fl!("exposure-iso"),
            current,
            range,
            SLIDER_WIDTH_EXPOSURE,
            VALUE_WIDTH_EXPOSURE,
            |v| format!("{}", v),
            Message::SetIsoSensitivity,
        )
    }

    /// Build metering mode row
    fn build_metering_row(
        &self,
        settings_data: Option<&super::ExposureSettings>,
    ) -> Element<'_, Message> {
        let controls = &self.available_exposure_controls;
        let current_mode = settings_data
            .and_then(|s| s.metering_mode)
            .unwrap_or_default();

        let mut row = widget::Row::new()
            .push(
                widget::text(fl!("exposure-metering"))
                    .size(13)
                    .width(Length::Fixed(LABEL_WIDTH)),
            )
            .spacing(4)
            .align_y(Alignment::Center)
            .width(Length::Shrink);

        for mode in &controls.metering_modes {
            let is_active = *mode == current_mode;
            let mode_copy = *mode;

            let btn = widget::button::text(mode.display_name())
                .on_press(Message::SetMeteringMode(mode_copy))
                .class(if is_active {
                    cosmic::theme::Button::Suggested
                } else {
                    cosmic::theme::Button::Text
                });

            row = row.push(btn);
        }

        row.into()
    }

    /// Build auto priority row (frame rate variation toggle)
    fn build_auto_priority_row(
        &self,
        settings_data: Option<&super::ExposureSettings>,
    ) -> Element<'_, Message> {
        let enabled = settings_data.and_then(|s| s.auto_priority).unwrap_or(false);

        widget::Row::new()
            .push(
                widget::text(fl!("exposure-auto-priority"))
                    .size(13)
                    .width(Length::Fixed(LABEL_WIDTH)),
            )
            .push(widget::toggler(enabled).on_toggle(|_| Message::ToggleAutoExposurePriority))
            .spacing(CONTROL_SPACING)
            .align_y(Alignment::Center)
            .width(Length::Shrink)
            .into()
    }

    // =========================================================================
    // Focus Controls
    // =========================================================================

    /// Add focus controls (auto toggle + manual slider)
    fn add_focus_controls<'a>(
        &'a self,
        mut column: widget::Column<'a, Message, cosmic::Theme>,
        settings_data: Option<&'a super::ExposureSettings>,
    ) -> widget::Column<'a, Message, cosmic::Theme> {
        let controls = &self.available_exposure_controls;

        if controls.focus.available {
            if controls.has_focus_auto {
                column = column.push(self.build_focus_auto_row(settings_data));
            }

            // Show manual slider when auto focus is off (or unavailable)
            let is_auto = settings_data.and_then(|s| s.focus_auto).unwrap_or(false);
            if !is_auto || !controls.has_focus_auto {
                column = column.push(self.build_focus_slider_row(settings_data));
            }
        }

        column
    }

    /// Build auto focus toggle row
    fn build_focus_auto_row(
        &self,
        settings_data: Option<&super::ExposureSettings>,
    ) -> Element<'_, Message> {
        let enabled = settings_data.and_then(|s| s.focus_auto).unwrap_or(false);

        widget::Row::new()
            .push(
                widget::text(fl!("focus-auto"))
                    .size(13)
                    .width(Length::Fixed(LABEL_WIDTH)),
            )
            .push(widget::toggler(enabled).on_toggle(|_| Message::ToggleFocusAuto))
            .push(
                widget::text(if enabled {
                    fl!("color-auto")
                } else {
                    fl!("color-manual")
                })
                .size(12),
            )
            .spacing(CONTROL_SPACING)
            .align_y(Alignment::Center)
            .width(Length::Shrink)
            .into()
    }

    /// Build manual focus slider row
    fn build_focus_slider_row(
        &self,
        settings_data: Option<&super::ExposureSettings>,
    ) -> Element<'_, Message> {
        let range = &self.available_exposure_controls.focus;
        let current = settings_data
            .and_then(|s| s.focus_absolute)
            .unwrap_or(range.default);

        Self::build_slider_row(
            fl!("focus-position"),
            current,
            range,
            SLIDER_WIDTH_EXPOSURE,
            VALUE_WIDTH_EXPOSURE,
            |v| format!("{}", v),
            Message::SetFocusAbsolute,
        )
    }

    // =========================================================================
    // Color Picker
    // =========================================================================

    /// Build the color picker overlay
    pub fn build_color_picker(&self) -> Element<'_, Message> {
        let spacing = cosmic::theme::spacing();
        let color_data = self.color_settings.as_ref();

        let mut column = widget::Column::new()
            .spacing(spacing.space_xs)
            .padding(spacing.space_s)
            .width(Length::Shrink);

        column = column.push(self.build_color_header());
        column = self.add_image_controls(column, color_data);
        column = self.add_white_balance_controls(column, color_data);

        let picker_content = widget::scrollable(column).height(
            crate::app::view::picker_panel_height(self.controls_are_sideways(), self.screen_height),
        );
        let picker_panel = widget::mouse_area(
            widget::container(self.frosted_panel(picker_content.into(), PICKER_PANEL))
                .width(Length::Fixed(COLOR_PICKER_WIDTH)),
        )
        .on_press(Message::Noop);

        // Anchor to the top bar's inner edge (below it in portrait, off the side
        // bar in landscape), shared with the tools menu and the exposure picker.
        self.anchor_bar_popup(picker_panel.into(), Message::CloseColorPicker)
    }

    /// Build color picker header with title and reset button
    fn build_color_header(&self) -> Element<'_, Message> {
        let reset_btn = widget::button::icon(widget::icon::from_name("edit-undo-symbolic"))
            .on_press(Message::ResetColorSettings)
            .class(cosmic::theme::Button::Text)
            .padding(4);

        widget::Row::new()
            .push(
                widget::text(fl!("color-title"))
                    .size(14)
                    .width(Length::Fill),
            )
            .push(reset_btn)
            .spacing(CONTROL_SPACING)
            .align_y(Alignment::Center)
            .into()
    }

    /// Add image adjustment controls (contrast, saturation, sharpness, hue)
    fn add_image_controls<'a>(
        &'a self,
        mut column: widget::Column<'a, Message, cosmic::Theme>,
        settings_data: Option<&'a super::ColorSettings>,
    ) -> widget::Column<'a, Message, cosmic::Theme> {
        let controls = &self.available_exposure_controls;

        if controls.contrast.available {
            column = column.push(self.build_color_slider_row(
                fl!("color-contrast"),
                settings_data.and_then(|s| s.contrast),
                &controls.contrast,
                Message::SetContrast,
            ));
        }
        if controls.saturation.available {
            column = column.push(self.build_color_slider_row(
                fl!("color-saturation"),
                settings_data.and_then(|s| s.saturation),
                &controls.saturation,
                Message::SetSaturation,
            ));
        }
        if controls.sharpness.available {
            column = column.push(self.build_color_slider_row(
                fl!("color-sharpness"),
                settings_data.and_then(|s| s.sharpness),
                &controls.sharpness,
                Message::SetSharpness,
            ));
        }
        if controls.hue.available {
            column = column.push(self.build_color_slider_row(
                fl!("color-hue"),
                settings_data.and_then(|s| s.hue),
                &controls.hue,
                Message::SetHue,
            ));
        }

        column
    }

    /// Build a color control slider row
    fn build_color_slider_row<'a, F>(
        &self,
        label: String,
        current: Option<i32>,
        range: &ControlRange,
        message_fn: F,
    ) -> Element<'a, Message>
    where
        F: 'a + Fn(i32) -> Message,
    {
        Self::build_slider_row(
            label,
            current.unwrap_or(range.default),
            range,
            SLIDER_WIDTH_COLOR,
            VALUE_WIDTH_COLOR,
            |v| format!("{}", v),
            message_fn,
        )
    }

    /// Add white balance controls
    fn add_white_balance_controls<'a>(
        &'a self,
        mut column: widget::Column<'a, Message, cosmic::Theme>,
        settings_data: Option<&'a super::ColorSettings>,
    ) -> widget::Column<'a, Message, cosmic::Theme> {
        let controls = &self.available_exposure_controls;

        if controls.has_white_balance_auto {
            column = column.push(self.build_auto_white_balance_row(settings_data));
        }

        if controls.white_balance_temperature.available {
            let is_auto = settings_data
                .and_then(|s| s.white_balance_auto)
                .unwrap_or(true);
            if !is_auto || !controls.has_white_balance_auto {
                column = column.push(self.build_white_balance_temp_row(settings_data));
            }
        }

        column
    }

    /// Build auto white balance toggle row
    fn build_auto_white_balance_row(
        &self,
        settings_data: Option<&super::ColorSettings>,
    ) -> Element<'_, Message> {
        let enabled = settings_data
            .and_then(|s| s.white_balance_auto)
            .unwrap_or(true);

        widget::Row::new()
            .push(
                widget::text(fl!("color-white-balance"))
                    .size(13)
                    .width(Length::Fixed(LABEL_WIDTH)),
            )
            .push(widget::toggler(enabled).on_toggle(|_| Message::ToggleAutoWhiteBalance))
            .push(
                widget::text(if enabled {
                    fl!("color-auto")
                } else {
                    fl!("color-manual")
                })
                .size(12),
            )
            .spacing(CONTROL_SPACING)
            .align_y(Alignment::Center)
            .width(Length::Shrink)
            .into()
    }

    /// Build white balance temperature row
    fn build_white_balance_temp_row(
        &self,
        settings_data: Option<&super::ColorSettings>,
    ) -> Element<'_, Message> {
        let range = &self.available_exposure_controls.white_balance_temperature;
        let current = settings_data
            .and_then(|s| s.white_balance_temperature)
            .unwrap_or(range.default);

        Self::build_slider_row(
            fl!("color-temperature"),
            current,
            range,
            SLIDER_WIDTH_COLOR,
            VALUE_WIDTH_EXPOSURE,
            |v| format!("{}K", v),
            Message::SetWhiteBalanceTemperature,
        )
    }
}

/// Format exposure time in 100µs units for display
fn format_exposure_time(time_100us: i32) -> String {
    let seconds = time_100us as f64 / 10000.0;
    if seconds < 0.5 {
        let denominator = (1.0 / seconds).round() as i32;
        format!("1/{}s", denominator)
    } else {
        format!("{:.1}s", seconds)
    }
}
