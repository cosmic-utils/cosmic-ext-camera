// SPDX-License-Identifier: GPL-3.0-only

//! Composition guide overlay module
//!
//! Renders composition guide lines (Rule of Thirds, Phi Grid, etc.)
//! on top of the camera preview using a canvas widget.

mod widget;

use crate::app::state::{AppModel, CameraMode, Message};
use crate::config::CompositionGuide;
use cosmic::Element;
use cosmic::iced::Length;

/// Full-size invisible spacer (used when no overlay is needed).
fn empty_overlay<'a>() -> Element<'a, Message> {
    cosmic::widget::Space::new()
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

impl AppModel {
    fn composition_preview_geometry(
        &self,
        frame_width: u32,
        frame_height: u32,
    ) -> (f32, f32, crate::app::preview_geometry::BarInsets) {
        let rotation = self
            .preview_adjusted_rotation(self.current_frame_rotation, self.should_mirror_preview());
        let (width, height) = if rotation.swaps_dimensions() {
            (frame_height as f32, frame_width as f32)
        } else {
            (frame_width as f32, frame_height as f32)
        };
        (width, height, self.bar_insets())
    }

    /// Build the composition guide overlay element.
    ///
    /// Passes the live state needed to compute the visible-video rectangle
    /// at draw time so the guide tracks fit/fill state, the photo aspect-
    /// ratio crop, and the bottom-bar scrim height (which differs by mode
    /// and animates across mode switches).
    pub fn build_composition_overlay(&self) -> Element<'_, Message> {
        if self.config.composition_guide == CompositionGuide::None {
            return empty_overlay();
        }

        let Some(frame) = &self.current_frame else {
            return empty_overlay();
        };

        let (rotated_w, rotated_h, insets) =
            self.composition_preview_geometry(frame.width, frame.height);
        if rotated_w < 1.0 || rotated_h < 1.0 {
            return empty_overlay();
        }

        // Aspect-ratio crop applies in Photo mode only; non-Native ratios
        // produce a sub-rect in Cover and a different letterbox in Contain.
        // Use the *display*-oriented ratio so the guide aligns with the
        // rotated preview on portrait windows.
        let aspect_crop_ratio =
            if self.mode == CameraMode::Photo && !self.current_frame_is_file_source {
                self.photo_aspect_ratio
                    .display_ratio(self.screen_is_portrait())
            } else {
                None
            };

        widget::composition_canvas(
            self.config.composition_guide,
            rotated_w,
            rotated_h,
            aspect_crop_ratio,
            self.cover_blend(),
            insets,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::camera::types::SensorRotation;
    use crate::backends::display_orientation::DisplayOrientation;
    use crate::config::{Config, ControlsPosition};

    #[test]
    fn composition_geometry_matches_preview_rotation_and_side_insets() {
        let model = AppModel {
            config: Config {
                controls_position: ControlsPosition::Left,
                ..Config::default()
            },
            display_orientation: DisplayOrientation::Rotate270,
            current_frame_rotation: SensorRotation::Rotate270,
            screen_width: 733.0,
            screen_height: 360.0,
            ..AppModel::default()
        };

        let (width, height, insets) = model.composition_preview_geometry(1452, 1080);
        assert_eq!((width, height), (1452.0, 1080.0));
        assert_eq!(insets.top, 0.0);
        assert_eq!(insets.bottom, 0.0);
        assert!(insets.left > 0.0 || insets.right > 0.0);
    }
}
