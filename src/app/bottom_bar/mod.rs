// SPDX-License-Identifier: GPL-3.0-only

//! Bottom bar module
//!
//! This module handles the bottom control bar UI components:
//! - Gallery button (with thumbnail)
//! - Mode switcher (Photo/Video toggle)
//! - Camera switcher (flip cameras)

pub mod camera_switcher;
pub mod fade_primitive;

pub mod gallery_button;
pub mod mode_carousel;
pub mod mode_switcher;
pub mod slide_h;

// Re-export for convenience

use crate::app::bar_layout::{bar_cross_lengths, sideways_column_reverses};
use crate::app::state::{AppModel, Message};
use cosmic::Element;
use cosmic::iced::alignment::Horizontal;
use cosmic::iced::{Alignment, Background, Color, Length};
use cosmic::widget;

use slide_h::SlideH;

fn sideways_button_slide_signs(quarter: crate::app::bar_layout::Quarter) -> (f32, f32) {
    if sideways_column_reverses(quarter) {
        (-1.0, 1.0)
    } else {
        (1.0, -1.0)
    }
}

/// Fixed height for bottom bar to match filter picker
pub const BOTTOM_BAR_HEIGHT: f32 = 74.0;

/// Shared horizontal layout used by the bottom bar (gallery / mode-carousel /
/// camera-switcher) and the recording-state capture row (spacer / stop circle
/// / photo button). The shape `[left] [Fill] [center] [Fill] [right]` keeps
/// the two layouts visually aligned column-for-column.
pub fn three_col_row<'a>(
    left: Element<'a, Message>,
    center: Element<'a, Message>,
    right: Element<'a, Message>,
    padding: impl Into<cosmic::iced::Padding>,
) -> Element<'a, Message> {
    let fill = || {
        widget::Space::new()
            .width(Length::Fill)
            .height(Length::Shrink)
    };
    widget::Row::new()
        .push(left)
        .push(fill())
        .push(center)
        .push(fill())
        .push(right)
        .padding(padding)
        .align_y(Alignment::Center)
        .width(Length::Fill)
        .into()
}

impl AppModel {
    /// Build the complete bottom bar widget
    ///
    /// Assembles gallery button, mode switcher, and camera switcher
    /// into a centered horizontal layout. The carousel visually extends
    /// beyond its layout bounds during expansion; SlideH slides the
    /// buttons outward in sync (reading from a shared atomic every frame).
    /// During recording, timelapse, quick-record, virtual-camera streaming,
    /// or a photo-timer countdown, the children are replaced with empty
    /// space of the same fixed height so the surrounding layout stays put.
    pub fn build_bottom_bar(&self) -> Element<'_, Message> {
        let bar_hidden = self.recording.is_recording()
            || self.quick_record.is_recording()
            || self.timelapse.is_active()
            || self.virtual_camera.is_streaming()
            || self.photo_timer_countdown.is_some();

        let sideways = self.controls_are_sideways();

        let inner: Element<'_, Message> = if bar_hidden {
            widget::Space::new()
                .width(Length::Fill)
                .height(Length::Fixed(BOTTOM_BAR_HEIGHT))
                .into()
        } else if self.mode.is_view_only() {
            // View mode: just the mode carousel — no gallery, no camera
            // switcher. Portrait keeps the same three-column shape so the
            // carousel sits at the same horizontal position as in the other
            // modes; sideways has nothing to balance against, so it's just
            // the carousel alone in the strip's Column.
            if sideways {
                widget::Column::new()
                    .push(self.build_mode_switcher())
                    .align_x(Horizontal::Center)
                    .width(Length::Fill)
                    .into()
            } else {
                let spacing = cosmic::theme::spacing();
                let side = || -> Element<'_, Message> {
                    widget::Space::new()
                        .width(Length::Fixed(
                            crate::constants::ui::PLACEHOLDER_BUTTON_WIDTH,
                        ))
                        .height(Length::Shrink)
                        .into()
                };
                three_col_row(
                    side(),
                    self.build_mode_switcher(),
                    side(),
                    [0, spacing.space_m],
                )
            }
        } else {
            let spacing = cosmic::theme::spacing();
            let slide = std::sync::Arc::clone(&self.carousel_button_slide);
            // The carousel extends visually beyond its layout via render_bounds,
            // and SlideH slides the side buttons in sync with the expansion.
            if sideways {
                // Same three widgets as portrait, stacked instead of laid out
                // side by side to fit the vertical strip. The carousel rotates
                // its own content along the strip (it reads the display
                // orientation via `build_mode_switcher`); here it is just placed
                // in the Column alongside the gallery and camera-switcher buttons.
                //
                // Portrait leading->trailing is `gallery · carousel · switcher`.
                // A rotate-left (270deg / Ccw90) turn reverses the strip so the
                // column reads `switcher · carousel · gallery` (matching the
                // carousel's own labels, which already flip bottom->top there);
                // rotate-right (90deg) keeps portrait order. See
                // `bar_layout::sideways_column_reverses`.
                let quarter = self.controls_bar_layout().quarter;
                let sibling_spacing = mode_carousel::sideways_sibling_spacing_for_modes(
                    &self.available_modes(),
                    self.screen_height,
                );
                // Turn the carousel's sibling motion onto the strip's physical
                // vertical axis, with signs selected after the 270° reversal.
                let (gallery_sign, switcher_sign) = sideways_button_slide_signs(quarter);
                let mut children: Vec<Element<'_, Message>> = vec![
                    SlideH::new_vertical(self.build_gallery_button(), slide.clone(), gallery_sign)
                        .into(),
                    self.build_mode_switcher(),
                    SlideH::new_vertical(self.build_camera_switcher(), slide, switcher_sign).into(),
                ];
                if sideways_column_reverses(quarter) {
                    children.reverse();
                }
                widget::Column::with_children(children)
                    .align_x(Horizontal::Center)
                    .spacing(sibling_spacing)
                    .width(Length::Fill)
                    .into()
            } else {
                three_col_row(
                    SlideH::new(self.build_gallery_button(), slide.clone(), 1.0).into(),
                    self.build_mode_switcher(),
                    SlideH::new(self.build_camera_switcher(), slide, -1.0).into(),
                    [0, spacing.space_m],
                )
            }
        };

        let mut container = widget::container(inner).style(|_theme| widget::container::Style {
            background: Some(Background::Color(Color::TRANSPARENT)),
            ..Default::default()
        });

        if sideways {
            // No longer the thing pinned directly to the window edge - since
            // Task 1, `bottom_section` (in view.rs) is the right-pinned strip
            // and this bar is nested content within it. `bar_cross_lengths`
            // sizes a bar that spans the long axis at a fixed cross-axis
            // thickness, which is the wrong shape here: reusing it with
            // BOTTOM_BAR_HEIGHT as the strip's width was defect #1 (it clips
            // the carousel). This just fills the strip's width and hugs its
            // own content's height.
            container = container.width(Length::Fill).height(Length::Shrink);
        } else {
            let (width, height) = bar_cross_lengths(sideways, BOTTOM_BAR_HEIGHT);
            container = container
                .width(width)
                .height(height)
                .center_y(BOTTOM_BAR_HEIGHT);
        }

        container.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::bar_layout::Quarter;

    #[test]
    fn sideways_buttons_slide_away_from_the_carousel() {
        assert_eq!(sideways_button_slide_signs(Quarter::Cw90), (1.0, -1.0));
        assert_eq!(sideways_button_slide_signs(Quarter::Ccw90), (-1.0, 1.0));
    }
}
