// SPDX-License-Identifier: GPL-3.0-only

//! Format picker UI view

use crate::app::overlay_style::{OVERLAY_CONTAINER, PICKER_PANEL};
use crate::app::state::{AppModel, Message};
use crate::constants::{formats, ui};
use crate::fl;
use cosmic::Element;
use cosmic::iced::{Alignment, Length};
use cosmic::widget;

impl AppModel {
    /// Build the iOS-style format picker overlay
    ///
    /// Shows resolution and framerate selection buttons in a semi-transparent overlay.
    /// Click outside the picker to close.
    pub fn build_format_picker(&self) -> Element<'_, Message> {
        let spacing = cosmic::theme::spacing();

        // Group formats by resolution label
        let (unique_resolutions, resolution_groups) = self.group_formats_by_label();

        // Determine currently selected resolution for picker
        let selected_res = self
            .picker_selected_resolution
            .or_else(|| self.active_format.as_ref().map(|f| f.width))
            .unwrap_or(formats::DEFAULT_PICKER_RESOLUTION);

        const BUTTON_WIDTH: f32 = ui::PICKER_BUTTON_WIDTH;

        // Build resolution row
        let mut res_row = widget::Row::new()
            .spacing(spacing.space_xxs)
            .align_y(Alignment::Center)
            .push(
                widget::text(fl!("format-resolution"))
                    .size(ui::PICKER_LABEL_TEXT_SIZE)
                    .width(Length::Fixed(ui::PICKER_LABEL_WIDTH)),
            );

        for &(res_label, width) in &unique_resolutions {
            let centered_text = widget::container(widget::text(res_label))
                .width(Length::Fill)
                .align_x(cosmic::iced::alignment::Horizontal::Center);

            let is_selected = width == selected_res;

            // Use Suggested for selected, Text for unselected - COSMIC's native button highlighting
            let button = widget::button::custom(centered_text)
                .on_press(Message::PickerSelectResolution(width))
                .class(if is_selected {
                    cosmic::theme::Button::Suggested
                } else {
                    cosmic::theme::Button::Text
                })
                .width(Length::Fill);

            // Wrap in styled container with overlay background
            let styled_button = widget::container(button)
                .style(OVERLAY_CONTAINER.style())
                .width(Length::Fixed(BUTTON_WIDTH));

            res_row = res_row.push(styled_button);
        }

        // Build framerate row
        let mut fps_row = widget::Row::new()
            .spacing(spacing.space_xxs)
            .align_y(Alignment::Center)
            .push(
                widget::text(fl!("format-framerate"))
                    .size(ui::PICKER_LABEL_TEXT_SIZE)
                    .width(Length::Fixed(ui::PICKER_LABEL_WIDTH)),
            );

        if let Some(formats) = resolution_groups.get(&selected_res) {
            use crate::backends::camera::types::Framerate;
            use std::collections::HashSet;
            let mut seen_fps: HashSet<Framerate> = HashSet::new();
            let mut fps_buttons: Vec<(Framerate, usize)> = Vec::new();

            // Collect unique framerates
            for &(idx, fmt) in formats {
                if let Some(fps) = fmt.framerate
                    && seen_fps.insert(fps)
                {
                    fps_buttons.push((fps, idx));
                }
            }

            fps_buttons.sort_by(|(a, _), (b, _)| {
                a.as_f64()
                    .partial_cmp(&b.as_f64())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Create framerate buttons
            for (fps, idx) in fps_buttons {
                let centered_text = widget::container(widget::text(fps.to_string()))
                    .width(Length::Fill)
                    .align_x(cosmic::iced::alignment::Horizontal::Center);

                // Check if this framerate matches the active format's framerate
                // (not exact format match, since pixel format may differ)
                let is_selected = self
                    .active_format
                    .as_ref()
                    .and_then(|active| active.framerate)
                    .map(|active_fps| active_fps == fps)
                    .unwrap_or(false);

                // Use Suggested for selected, Text for unselected - COSMIC's native button highlighting
                let button = widget::button::custom(centered_text)
                    .on_press(Message::PickerSelectFormat(idx))
                    .class(if is_selected {
                        cosmic::theme::Button::Suggested
                    } else {
                        cosmic::theme::Button::Text
                    })
                    .width(Length::Fill);

                // Wrap in styled container with overlay background
                let styled_button = widget::container(button)
                    .style(OVERLAY_CONTAINER.style())
                    .width(Length::Fixed(BUTTON_WIDTH));

                fps_row = fps_row.push(styled_button);
            }
        }

        // Keep long resolution/FPS rows reachable on compact landscape
        // windows instead of allowing fixed-width choices to clip off-screen.
        let picker_content = widget::scrollable(
            widget::Column::new()
                .push(res_row)
                .push(widget::space::vertical().height(spacing.space_s))
                .push(fps_row)
                .padding(spacing.space_xs),
        )
        .horizontal()
        .height(crate::app::view::picker_panel_height(
            self.controls_are_sideways(),
            self.screen_height,
        ));

        // Build picker panel with semi-transparent themed background.
        let picker_panel = self.frosted_panel(picker_content.into(), PICKER_PANEL);

        // Anchor to the top bar's inner edge at its leading (format-button) end -
        // below it in portrait, off the side bar in landscape - sharing the same
        // bar-anchor helper as the tools menu and the other pickers.
        self.anchor_bar_popup_leading(picker_panel, Message::CloseFormatPicker)
    }
}
