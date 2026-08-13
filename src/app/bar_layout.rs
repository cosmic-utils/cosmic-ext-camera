// SPDX-License-Identifier: GPL-3.0-only

//! Which screen edge each UI bar occupies, derived from the display transform.
//!
//! The compositor *does* re-lay the window out when the device is held sideways:
//! at a quarter turn the window itself becomes landscape and renders upright.
//! The bars still need to move to the window's side edges for ergonomics (a
//! camera app held sideways wants its controls under the thumbs, not stretched
//! above and below the preview) - see `bar_layout`'s doc for why 90 and 270 are
//! mirrored. This module is the single place that mapping lives; nothing else
//! may hardcode "left" or "right".

use crate::backends::display_orientation::DisplayOrientation;
use crate::config::ControlsPosition;
use cosmic::iced::Length;

/// A screen edge, in window coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    Top,
    Bottom,
    Left,
    Right,
}

/// Quarter turn applied to bar content so it reads upright to the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quarter {
    None,
    Cw90,
    Ccw90,
}

impl Quarter {
    /// Angle (radians) to pass to a canvas `Frame::rotate` when turning
    /// portrait bar content into the sideways strip.
    ///
    /// The signs are pinned by the carousel's `remap_pointer` hit-test
    /// contract (a tap near the strip TOP must map to the carousel RIGHT for
    /// `Ccw90`), not by the enum names - see the tests in `mode_carousel`.
    /// In iced's y-down space a positive angle turns the content clockwise on
    /// screen, so `Ccw90` maps to `-FRAC_PI_2` here.
    pub fn radians(self) -> f32 {
        match self {
            Quarter::None => 0.0,
            Quarter::Ccw90 => -std::f32::consts::FRAC_PI_2,
            Quarter::Cw90 => std::f32::consts::FRAC_PI_2,
        }
    }
}

/// Which edge each bar occupies, and how its content is turned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BarLayout {
    pub top_bar: Edge,
    pub bottom_bar: Edge,
    pub quarter: Quarter,
}

/// Which edge each bar occupies for `o`.
///
/// `Rotate180` is deliberately treated as portrait - out of scope for now.
/// Flip is ignored: `degrees()` already collapses `Flipped90` onto 90.
///
/// 90 and 270 are mirror images of each other: 270 keeps the top bar on the
/// LEFT edge and turns bar content CCW; 90 swaps the top bar to the RIGHT edge
/// and turns bar content CW. This is a deliberate ergonomic override, not a
/// consequence of how the compositor renders the window - the compositor
/// re-lays the window out to landscape and compensates in both directions, so
/// 90 and 270 render byte-identically to the user today. Mirroring the bars
/// anyway means the *strip* - not just the preview - tracks which physical
/// edge is now the "bottom" of the device, so controls land under the thumbs
/// the same way regardless of which way the device is rotated into landscape.
pub fn bar_layout(o: DisplayOrientation) -> BarLayout {
    match o.degrees() {
        270 => BarLayout {
            top_bar: Edge::Left,
            bottom_bar: Edge::Right,
            quarter: Quarter::Ccw90,
        },
        90 => BarLayout {
            top_bar: Edge::Right,
            bottom_bar: Edge::Left,
            quarter: Quarter::Cw90,
        },
        // 0 and 180 both lay out as portrait.
        _ => BarLayout {
            top_bar: Edge::Top,
            bottom_bar: Edge::Bottom,
            quarter: Quarter::None,
        },
    }
}

/// Resolve bar placement from compositor orientation and user preference.
///
/// Bottom preserves automatic orientation. Explicit side positions override
/// layout only; preview and capture rotation continue to consume the real
/// [`DisplayOrientation`] directly.
pub fn effective_bar_layout(o: DisplayOrientation, position: ControlsPosition) -> BarLayout {
    match position {
        ControlsPosition::Bottom => bar_layout(o),
        ControlsPosition::Left => BarLayout {
            top_bar: Edge::Right,
            bottom_bar: Edge::Left,
            quarter: Quarter::Cw90,
        },
        ControlsPosition::Right => BarLayout {
            top_bar: Edge::Left,
            bottom_bar: Edge::Right,
            quarter: Quarter::Ccw90,
        },
    }
}

impl crate::app::state::AppModel {
    /// Effective UI-bar layout. This is deliberately separate from
    /// `display_orientation`, which remains the sole input to preview and
    /// capture rotation.
    pub(crate) fn controls_bar_layout(&self) -> BarLayout {
        effective_bar_layout(self.display_orientation, self.config.controls_position)
    }

    pub(crate) fn controls_are_sideways(&self) -> bool {
        self.controls_bar_layout().quarter != Quarter::None
    }
}

/// Whether the device is held sideways, i.e. the bars move to the side edges.
pub fn is_sideways(o: DisplayOrientation) -> bool {
    matches!(o.degrees(), 90 | 270)
}

/// When a portrait bar is re-laid out as a sideways `Column`, does its portrait
/// leading→trailing (left→right) order map onto the column top→bottom directly,
/// or reversed?
///
/// The bars emulate a rigid quarter turn of the whole window. Turning **left**
/// (270°, `Ccw90`) sends the portrait row's *trailing* (right) end to the TOP
/// of the side strip, so the column is the reverse of the portrait order - for
/// the top bar that puts the window controls at the strip top; for the bottom
/// bar it makes the mode column read `switcher · carousel · gallery`. Turning
/// **right** (90°, `Cw90`) is the mirror image: the portrait *leading* (left)
/// end goes to the top, so the column keeps portrait order (window controls at
/// the bottom; mode column `gallery · carousel · switcher`). This matches the
/// spec's orientation matrix and the carousel's own label direction, which
/// already flips CW/CCW between the two turns.
///
/// Portrait (`None`) still lays out as a `Row`, so the value is unused there
/// and returns `false`.
pub fn sideways_column_reverses(quarter: Quarter) -> bool {
    matches!(quarter, Quarter::Ccw90)
}

/// Width of the right-hand control strip when the device is held sideways.
///
/// Derived from the portrait bottom band height: capture area (76 + 2·12 = 100)
/// plus bar height (74) = 174. Ensures same-axis correctness and matches the
/// visual height of stacked controls in portrait mode.
pub const SIDEWAYS_STRIP_WIDTH: f32 = 174.0;

/// Cross-axis sizing for a bar strip: `thickness` on the short axis and
/// `Fill` on the long axis, swapped between portrait and sideways.
///
/// Portrait bars run the width of the window and are `thickness` tall
/// (`Fill` width / `Fixed` height); sideways bars run the height of the
/// window and are `thickness` wide (`Fixed` width / `Fill` height). Shared
/// by every bar-strip container (`build_bottom_bar`, `build_top_bar`) so the
/// swap logic lives in exactly one place.
pub fn bar_cross_lengths(sideways: bool, thickness: f32) -> (Length, Length) {
    if sideways {
        (Length::Fixed(thickness), Length::Fill)
    } else {
        (Length::Fill, Length::Fixed(thickness))
    }
}

/// Padding and horizontal alignment for a top popup (the tools menu, the
/// exposure/color pickers, the format and motor pickers) anchored `gap` off the
/// top bar's inner edge, keyed on which edge the top bar occupies and on which
/// end of the bar the popup's button lives.
///
/// `leading` picks which end of the *portrait* top bar the popup hangs under:
/// `false` (trailing) is the right end - the tools/motor/exposure/color popups,
/// whose buttons sit at the bar's trailing edge; `true` (leading) is the left
/// end - the format picker, whose button sits at the bar's leading edge. The
/// distinction only affects the portrait (`Edge::Top`) arm, where it flips the
/// horizontal alignment; on the side strips both ends hang off the same strip
/// edge, so `leading` is inert there.
///
/// Returns `([top, right, bottom, left], align_right)`, where `align_right`
/// selects the leading (`true`) vs trailing (`false`) `Space(Fill)` spring the
/// caller uses to push the panel to that side:
///
/// - `Edge::Top` (portrait): below the full-width top bar -
///   `[thickness + gap, gap, 0, gap]`; right-aligned for a trailing popup
///   (byte-identical to the historical tools-menu hardcode) and left-aligned for
///   a leading popup (byte-identical to the historical format-picker hardcode).
/// - `Edge::Right` (90°): the top bar is a vertical strip on the right edge;
///   hang the popup `thickness + gap` in from it, near the top -
///   `[gap, thickness + gap, 0, gap]`, right-aligned.
/// - `Edge::Left` (270°): the mirror image - `[gap, gap, 0, thickness + gap]`,
///   left-aligned.
///
/// `Edge::Bottom` never hosts a top popup and falls through to the portrait
/// arm as a safe default.
pub fn bar_anchored_popup_padding(
    top_bar: Edge,
    thickness: u16,
    gap: u16,
    leading: bool,
) -> ([u16; 4], bool) {
    let inner = thickness + gap;
    match top_bar {
        // Side bar on the left: anchor just inside it, near the top, left-aligned.
        Edge::Left => ([gap, gap, 0, inner], false),
        // Side bar on the right: mirror - anchor just inside it, right-aligned.
        Edge::Right => ([gap, inner, 0, gap], true),
        // Portrait (Top) - and Bottom as a fallback: below the full-width top
        // bar. A trailing popup hugs the right end, a leading popup the left.
        Edge::Top | Edge::Bottom => ([inner, gap, 0, gap], !leading),
    }
}

/// Whether the capture area keeps its fixed (portrait) height, or lets its
/// content size it.
///
/// Portrait always pins the capture area to `capture_area_height()` so the
/// View↔Photo transition can animate the slot's collapse. When the device is
/// held sideways *and* the recording/streaming three-slot layout is showing, the
/// three controls stack into a `Column` that is taller than the ~100px portrait
/// lane, so the area must size to its content instead of clipping the stack.
/// Every other sideways case (idle capture button, video-file controls) still
/// wants the fixed lane height.
pub fn capture_area_height_is_fixed(sideways: bool, recording_layout: bool) -> bool {
    !(sideways && recording_layout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bottom_position_preserves_compositor_orientation() {
        let portrait = effective_bar_layout(
            DisplayOrientation::Rotate0,
            crate::config::ControlsPosition::Bottom,
        );
        let rotated = effective_bar_layout(
            DisplayOrientation::Rotate270,
            crate::config::ControlsPosition::Bottom,
        );

        assert_eq!(portrait.bottom_bar, Edge::Bottom);
        assert_eq!(portrait.top_bar, Edge::Top);
        assert_eq!(rotated.bottom_bar, Edge::Right);
        assert_eq!(rotated.top_bar, Edge::Left);
    }

    #[test]
    fn manual_side_positions_override_layout_only() {
        let left = effective_bar_layout(
            DisplayOrientation::Rotate0,
            crate::config::ControlsPosition::Left,
        );
        let right = effective_bar_layout(
            DisplayOrientation::Rotate0,
            crate::config::ControlsPosition::Right,
        );

        assert_eq!(left.bottom_bar, Edge::Left);
        assert_eq!(left.top_bar, Edge::Right);
        assert_eq!(left.quarter, Quarter::Cw90);
        assert_eq!(right.bottom_bar, Edge::Right);
        assert_eq!(right.top_bar, Edge::Left);
        assert_eq!(right.quarter, Quarter::Ccw90);
    }

    #[test]
    fn portrait_keeps_bars_on_top_and_bottom() {
        let l = bar_layout(DisplayOrientation::Rotate0);
        assert_eq!(l.top_bar, Edge::Top);
        assert_eq!(l.bottom_bar, Edge::Bottom);
        assert_eq!(l.quarter, Quarter::None);
    }

    #[test]
    fn rotate180_is_treated_as_portrait() {
        // Explicitly out of scope: 180 renders exactly like portrait.
        let l = bar_layout(DisplayOrientation::Rotate180);
        assert_eq!(l.top_bar, Edge::Top);
        assert_eq!(l.bottom_bar, Edge::Bottom);
        assert_eq!(l.quarter, Quarter::None);
    }

    #[test]
    fn rotate270_is_left_edge_ccw() {
        let l = bar_layout(DisplayOrientation::Rotate270);
        assert_eq!(l.top_bar, Edge::Left);
        assert_eq!(l.bottom_bar, Edge::Right);
        assert_eq!(l.quarter, Quarter::Ccw90);
    }

    #[test]
    fn rotate90_mirrors_270_right_edge_cw() {
        let l = bar_layout(DisplayOrientation::Rotate90);
        assert_eq!(l.top_bar, Edge::Right);
        assert_eq!(l.bottom_bar, Edge::Left);
        assert_eq!(l.quarter, Quarter::Cw90);
        // genuinely mirrored, not identical:
        assert_ne!(
            bar_layout(DisplayOrientation::Rotate90),
            bar_layout(DisplayOrientation::Rotate270)
        );
    }

    #[test]
    fn flipped_variants_follow_their_unflipped_quarter_turn() {
        // `degrees()` ignores flip, so Flipped90 lays out like Rotate90.
        assert_eq!(
            bar_layout(DisplayOrientation::Flipped90),
            bar_layout(DisplayOrientation::Rotate90)
        );
        assert_eq!(
            bar_layout(DisplayOrientation::Flipped0),
            bar_layout(DisplayOrientation::Rotate0)
        );
    }

    #[test]
    fn rotate270_reverses_column_putting_window_controls_at_top() {
        // 270° (Ccw90): the portrait row's trailing end (window controls) lands
        // at the TOP of the side strip, so the sideways Column reverses the
        // portrait order.
        let q = bar_layout(DisplayOrientation::Rotate270).quarter;
        assert!(
            sideways_column_reverses(q),
            "270 must put window controls at the strip top"
        );
    }

    #[test]
    fn rotate90_mirrors_270_window_controls_at_bottom() {
        // 90° (Cw90) is the mirror image: window controls at the strip BOTTOM,
        // so the Column keeps portrait order (no reverse).
        let q90 = bar_layout(DisplayOrientation::Rotate90).quarter;
        let q270 = bar_layout(DisplayOrientation::Rotate270).quarter;
        assert!(
            !sideways_column_reverses(q90),
            "90 must put window controls at the strip bottom"
        );
        assert_ne!(
            sideways_column_reverses(q90),
            sideways_column_reverses(q270),
            "90 and 270 must be mirror images of each other"
        );
    }

    #[test]
    fn portrait_column_order_is_never_reversed() {
        // Portrait stays a Row; the reverse decision must be inert there.
        assert!(!sideways_column_reverses(
            bar_layout(DisplayOrientation::Rotate0).quarter
        ));
    }

    #[test]
    fn is_sideways_only_for_quarter_turns() {
        assert!(!is_sideways(DisplayOrientation::Rotate0));
        assert!(!is_sideways(DisplayOrientation::Rotate180));
        assert!(is_sideways(DisplayOrientation::Rotate90));
        assert!(is_sideways(DisplayOrientation::Rotate270));
    }

    #[test]
    fn bar_cross_lengths_portrait_is_fill_width_fixed_height() {
        assert_eq!(
            bar_cross_lengths(false, 74.0),
            (Length::Fill, Length::Fixed(74.0))
        );
    }

    #[test]
    fn bar_cross_lengths_sideways_is_fixed_width_fill_height() {
        assert_eq!(
            bar_cross_lengths(true, 74.0),
            (Length::Fixed(74.0), Length::Fill)
        );
    }

    #[test]
    fn sideways_strip_thickness_is_not_the_portrait_bar_height() {
        // Regression guard for the carousel-clipping defect: a sideways strip
        // sized by BOTTOM_BAR_HEIGHT (~74px) clips the carousel out of
        // existence. The strip must be wide enough for its rotated content.
        let (w, h) = bar_cross_lengths(true, SIDEWAYS_STRIP_WIDTH);
        assert_eq!(w, Length::Fixed(SIDEWAYS_STRIP_WIDTH));
        assert_eq!(h, Length::Fill);
        assert!(
            std::hint::black_box(SIDEWAYS_STRIP_WIDTH) > crate::app::bottom_bar::BOTTOM_BAR_HEIGHT,
            "strip must be wider than the portrait bar is tall, or the carousel clips"
        );
    }

    #[test]
    fn popup_anchor_portrait_trailing_is_below_bar_top_right() {
        // Portrait trailing (tools/motor/exposure/color): top = thickness + gap
        // (clear the full-width top bar), right = gap, right-aligned. Matches the
        // historical hardcode exactly.
        assert_eq!(
            bar_anchored_popup_padding(Edge::Top, 47, 12, false),
            ([59, 12, 0, 12], true)
        );
    }

    #[test]
    fn popup_anchor_portrait_leading_is_below_bar_top_left() {
        // Portrait leading (format picker): same insets, but the panel hugs the
        // LEFT end (left-aligned), matching the historical format-picker hardcode.
        assert_eq!(
            bar_anchored_popup_padding(Edge::Top, 47, 12, true),
            ([59, 12, 0, 12], false)
        );
    }

    #[test]
    fn popup_anchor_leading_only_flips_alignment_in_portrait() {
        // On the side strips the `leading` flag is inert: both ends hang off the
        // same strip edge, so padding and alignment are unchanged.
        assert_eq!(
            bar_anchored_popup_padding(Edge::Left, 47, 12, true),
            bar_anchored_popup_padding(Edge::Left, 47, 12, false)
        );
        assert_eq!(
            bar_anchored_popup_padding(Edge::Right, 47, 12, true),
            bar_anchored_popup_padding(Edge::Right, 47, 12, false)
        );
    }

    #[test]
    fn popup_anchor_270_hangs_off_left_bar_near_top() {
        // 270° (top bar on the LEFT): left = thickness + gap = 59 (just inside
        // the left strip), top = gap = 12 (near the top), left-aligned.
        assert_eq!(
            bar_anchored_popup_padding(Edge::Left, 47, 12, false),
            ([12, 12, 0, 59], false)
        );
    }

    #[test]
    fn popup_anchor_90_mirrors_270_off_right_bar() {
        // 90° (top bar on the RIGHT): the mirror - right = thickness + gap = 59,
        // top = gap = 12, right-aligned.
        assert_eq!(
            bar_anchored_popup_padding(Edge::Right, 47, 12, false),
            ([12, 59, 0, 12], true)
        );
    }

    #[test]
    fn popup_anchor_90_and_270_are_mirror_images() {
        let (p90, a90) = bar_anchored_popup_padding(Edge::Right, 47, 12, false);
        let (p270, a270) = bar_anchored_popup_padding(Edge::Left, 47, 12, false);
        // Same top/bottom, swapped right/left insets, opposite alignment.
        assert_eq!(p90[0], p270[0]);
        assert_eq!(p90[2], p270[2]);
        assert_eq!(p90[1], p270[3]);
        assert_eq!(p90[3], p270[1]);
        assert_ne!(a90, a270);
    }

    #[test]
    fn capture_area_fixed_height_except_sideways_recording() {
        // Portrait always keeps the fixed lane height (drives the View↔Photo
        // animation), in every layout state.
        assert!(capture_area_height_is_fixed(false, false));
        assert!(capture_area_height_is_fixed(false, true));
        // Sideways idle capture button still uses the fixed lane.
        assert!(capture_area_height_is_fixed(true, false));
        // Only the sideways recording/streaming stack sizes to its content.
        assert!(!capture_area_height_is_fixed(true, true));
    }

    #[test]
    fn sideways_strip_width_equals_portrait_bottom_band_height() {
        // capture area (CAPTURE_BUTTON_OUTER_SIZE + 2*space_xs = 76 + 24 = 100)
        // plus BOTTOM_BAR_HEIGHT (74) = 174. NOT the ~150 label-derived value.
        assert_eq!(SIDEWAYS_STRIP_WIDTH, 174.0);
    }
}
