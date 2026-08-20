// SPDX-License-Identifier: GPL-3.0-only

//! Mode switcher — builds a ModeCarousel widget for mode selection.

use crate::app::bottom_bar::mode_carousel::ModeCarousel;
use crate::app::state::{AppModel, Message};
use cosmic::Element;

impl AppModel {
    /// Build the mode switcher widget using the custom ModeCarousel.
    pub fn build_mode_switcher(&self) -> Element<'_, Message> {
        // Allow mode switching during blur transitions (camera restart) —
        // only disable during recording, streaming, or timelapse.
        let is_disabled = self.recording.is_recording()
            || self.virtual_camera.is_streaming()
            || self.timelapse.is_active();

        let modes = self.available_modes();

        // Quarter turn for the carousel content, derived from the display
        // orientation. `Quarter::None` in portrait keeps the carousel exactly
        // as it was; a quarter turn rotates its content along the side strip.
        let quarter = self.controls_bar_layout().quarter;

        ModeCarousel::new(
            modes,
            self.mode,
            Message::SetMode,
            is_disabled,
            std::sync::Arc::clone(&self.carousel_button_slide),
            // In View mode the carousel stands alone (no gallery / camera-
            // switcher buttons next to it), so collapse the rounded chip
            // to the active "View" pill when fully settled.
            self.mode.is_view_only(),
            quarter,
        )
        .into()
    }
}
