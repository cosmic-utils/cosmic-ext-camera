// SPDX-License-Identifier: GPL-3.0-only

//! Main application view
//!
//! This module composes the main UI from modularized components:
//! - Camera preview (camera_preview module)
//! - Top bar with format picker (inline)
//! - Capture button (controls module)
//! - Bottom bar (bottom_bar module)
//! - Format picker overlay (format_picker module)

use crate::app::bar_layout::{Edge, bar_cross_lengths, sideways_column_reverses};
use crate::app::bottom_bar::slide_h::SlideH;
use crate::app::overlay_style::{
    OVERLAY_CONTAINER, PICKER_PANEL, POPUP_PANEL, overlay_chip_button_class, window_bg_style,
};
use crate::app::preview_geometry::TOP_BAR_HEIGHT;
use crate::app::qr_overlay::build_qr_overlay;
use crate::app::state::{AppModel, BurstModeStage, CameraMode, FilterType, Message};
use crate::constants::resolution_thresholds;
use crate::constants::ui;
use crate::fl;
use cosmic::Element;
use cosmic::iced::{Alignment, Background, Color, Length};
use cosmic::widget::{self, icon};
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::info;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FitZoomLayout {
    Inline,
    FloatUpright,
}

fn fit_zoom_layout(sideways: bool) -> FitZoomLayout {
    if sideways {
        FitZoomLayout::FloatUpright
    } else {
        FitZoomLayout::Inline
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VideoProgressPlacement {
    AboveCapture,
    AboveSideLanes,
}

fn video_progress_placement(sideways: bool) -> VideoProgressPlacement {
    if sideways {
        VideoProgressPlacement::AboveSideLanes
    } else {
        VideoProgressPlacement::AboveCapture
    }
}

pub(crate) fn picker_panel_height(sideways: bool, screen_height: f32) -> Length {
    const COMPACT_SIDE_HEIGHT_MAX: f32 = 480.0;
    if sideways && screen_height > 0.0 && screen_height <= COMPACT_SIDE_HEIGHT_MAX {
        let gap = f32::from(cosmic::theme::spacing().space_xs) * 2.0;
        Length::Fixed((screen_height - gap).max(0.0))
    } else {
        Length::Shrink
    }
}

fn overlay_popup_padding(model: &AppModel) -> [f32; 4] {
    let insets = AppModel::map_bar_insets(
        model.controls_bar_layout(),
        model.top_ui_height(),
        model.bottom_ui_height(),
    );
    if model.controls_are_sideways() {
        let chip_extent = model.zoom_chip_strip_height();
        let (right, left) = match model.controls_bar_layout().bottom_bar {
            Edge::Right => (insets.right + chip_extent, insets.left),
            Edge::Left => (insets.right, insets.left + chip_extent),
            _ => (insets.right, insets.left),
        };
        [insets.top, right, insets.bottom, left]
    } else {
        [
            insets.top,
            insets.right,
            insets.bottom + model.zoom_chip_strip_height(),
            insets.left,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FitZoomAxis {
    Horizontal,
    Vertical,
}

fn fit_zoom_axis(sideways: bool) -> FitZoomAxis {
    if sideways {
        FitZoomAxis::Vertical
    } else {
        FitZoomAxis::Horizontal
    }
}

/// Flash icon SVG (lightning bolt)
const FLASH_ICON: &[u8] = include_bytes!("../../resources/button_icons/flash.svg");
/// Flash off icon SVG (lightning bolt with strike-through)
const FLASH_OFF_ICON: &[u8] = include_bytes!("../../resources/button_icons/flash-off.svg");
/// Timer off icon SVG
const TIMER_OFF_ICON: &[u8] = include_bytes!("../../resources/button_icons/timer-off.svg");
/// Timer 3s icon SVG
const TIMER_3_ICON: &[u8] = include_bytes!("../../resources/button_icons/timer-3.svg");
/// Timer 5s icon SVG
const TIMER_5_ICON: &[u8] = include_bytes!("../../resources/button_icons/timer-5.svg");
/// Timer 10s icon SVG
const TIMER_10_ICON: &[u8] = include_bytes!("../../resources/button_icons/timer-10.svg");
/// Aspect ratio native icon SVG
const ASPECT_NATIVE_ICON: &[u8] = include_bytes!("../../resources/button_icons/aspect-native.svg");
/// Aspect ratio 4:3 icon SVG (landscape)
const ASPECT_4_3_ICON: &[u8] = include_bytes!("../../resources/button_icons/aspect-4-3.svg");
/// Aspect ratio 3:4 icon SVG (portrait companion of 4:3)
const ASPECT_3_4_ICON: &[u8] = include_bytes!("../../resources/button_icons/aspect-3-4.svg");
/// Aspect ratio 16:9 icon SVG (landscape)
const ASPECT_16_9_ICON: &[u8] = include_bytes!("../../resources/button_icons/aspect-16-9.svg");
/// Aspect ratio 9:16 icon SVG (portrait companion of 16:9)
const ASPECT_9_16_ICON: &[u8] = include_bytes!("../../resources/button_icons/aspect-9-16.svg");
/// Aspect ratio 2:1 (18:9) icon SVG (landscape)
const ASPECT_2_1_ICON: &[u8] = include_bytes!("../../resources/button_icons/aspect-2-1.svg");
/// Aspect ratio 1:2 icon SVG (portrait companion of 2:1)
const ASPECT_1_2_ICON: &[u8] = include_bytes!("../../resources/button_icons/aspect-1-2.svg");
/// Aspect ratio 1:1 icon SVG
const ASPECT_1_1_ICON: &[u8] = include_bytes!("../../resources/button_icons/aspect-1-1.svg");
/// Exposure icon SVG
const EXPOSURE_ICON: &[u8] = include_bytes!("../../resources/button_icons/exposure.svg");
const TOOLS_GRID_ICON: &[u8] = include_bytes!("../../resources/button_icons/tools-grid.svg");
const FILTER_ICON: &[u8] = include_bytes!("../../resources/button_icons/image-filter.svg");
/// Moon icon SVG (burst mode)
const MOON_ICON: &[u8] = include_bytes!("../../resources/button_icons/moon.svg");
/// Moon off icon SVG (burst mode disabled, with strike-through)
const MOON_OFF_ICON: &[u8] = include_bytes!("../../resources/button_icons/moon-off.svg");
/// Camera tilt/motor control icon SVG
const CAMERA_TILT_ICON: &[u8] = include_bytes!("../../resources/button_icons/camera-tilt.svg");

/// Gap left between the context drawer and the window's right/bottom edges.
/// Matches the 8 px inset libcosmic uses for its own context-drawer overlay.
const CONTEXT_DRAWER_INSET: u16 = 8;

/// Burst mode progress bar dimensions
const BURST_MODE_PROGRESS_BAR_WIDTH: f32 = 200.0;
const BURST_MODE_PROGRESS_BAR_HEIGHT: f32 = 8.0;

/// Fallback aspect ratio used before the first window-resize event arrives.
const FALLBACK_ASPECT_RATIO: f32 = 16.0 / 9.0;

impl AppModel {
    /// Current window aspect ratio, populated from `on_window_resize`. Returns
    /// 16:9 as a fallback before the first resize event.
    pub fn screen_aspect_ratio(&self) -> f32 {
        if self.screen_width > 0.0 && self.screen_height > 0.0 {
            self.screen_width / self.screen_height
        } else {
            FALLBACK_ASPECT_RATIO
        }
    }

    /// `true` when the window is taller than wide. Drives the orientation
    /// flip applied to the aspect-ratio crop, the canvas overlay bars, the
    /// composition guide and the aspect-icon selection so all four agree
    /// with what the rotated preview shows.
    pub fn screen_is_portrait(&self) -> bool {
        self.screen_width > 0.0
            && self.screen_height > 0.0
            && self.screen_height > self.screen_width
    }

    /// Settled top-bar scrim / shader bar height. 0 in View mode and while
    /// the chrome is hidden (the preview takes the full window in fit/fill);
    /// `TOP_BAR_HEIGHT` otherwise.
    pub fn settled_top_ui_height(&self) -> f32 {
        if self.mode.is_view_only() || self.ui_hidden {
            0.0
        } else {
            TOP_BAR_HEIGHT
        }
    }

    /// Animated top-bar scrim height. Interpolates between snapshots through
    /// `fit_animation` so the Photo↔View transition slides smoothly.
    pub fn top_ui_height(&self) -> f32 {
        let target = self.settled_top_ui_height();
        let Some(anim) = self.fit_animation else {
            return target;
        };
        anim.from.top_ui_height + (target - anim.from.top_ui_height) * self.fit_animation_eased()
    }

    /// Settled pixel height of the bottom UI scrim. The top edge sits at:
    ///
    /// - **View mode / hidden chrome**: 0. The preview extends to the
    ///   window's bottom edge in fit/fill (the carousel renders on top of the
    ///   live preview without a scrim, and with the chrome hidden there is no
    ///   carousel at all).
    /// - **Photo mode**: the top of the capture-button area. By construction
    ///   the symmetric `space_xs` paddings (`build_capture_button`'s top
    ///   padding and the zoom row's `control_spacing` bottom padding) make
    ///   that line coincide with the midpoint between the capture circle
    ///   and the zoom/fit row above it.
    /// - **Other modes**: a quarter of the capture button's bottom padding
    ///   (`space_xs / 4`) above the carousel's top edge — close to the
    ///   carousel but with a small visual gap so the bar doesn't appear
    ///   to swallow the bottom controls.
    ///
    /// Photo capture math (`cover_capture_crop`) reads this through the
    /// settled value so a shot taken mid-animation isn't cropped against
    /// an in-flight value.
    pub fn settled_bottom_ui_height(&self) -> f32 {
        if self.mode.is_view_only() || self.ui_hidden {
            return 0.0;
        }
        let spacing = cosmic::theme::spacing();
        let bottom_bar_h = crate::app::bottom_bar::BOTTOM_BAR_HEIGHT;
        if self.mode == CameraMode::Photo {
            let capture_h = crate::app::controls::capture_button::CAPTURE_BUTTON_OUTER_SIZE
                + 2.0 * f32::from(spacing.space_xs);
            bottom_bar_h + capture_h
        } else {
            bottom_bar_h + f32::from(spacing.space_xs) / 4.0
        }
    }

    /// Animated bottom-bar scrim height. During an in-flight fit animation,
    /// interpolates from the captured starting height toward
    /// `settled_bottom_ui_height()` using the same eased progress as
    /// `cover_blend`. Drives the canvas scrim and the video shader's
    /// `bar_bottom_px`, so the preview's centre slides with the scrim during
    /// a Photo↔non-Photo transition.
    pub fn bottom_ui_height(&self) -> f32 {
        let target = self.settled_bottom_ui_height();
        let Some(anim) = self.fit_animation else {
            return target;
        };
        anim.from.bottom_ui_height
            + (target - anim.from.bottom_ui_height) * self.fit_animation_eased()
    }

    /// Space the UI bars reserve on each edge, honouring the display transform.
    ///
    /// The bar *extents* are unchanged by rotation - a bar is the same thickness
    /// whichever edge it sits on - so this reuses the existing animated heights
    /// and only decides which edges they land on.
    pub fn bar_insets(&self) -> crate::app::preview_geometry::BarInsets {
        let layout = self.controls_bar_layout();
        let base = Self::map_bar_insets(layout, self.top_ui_height(), self.bottom_ui_height());
        // When an aspect crop letterboxes onto a bar, grow that bar out to the
        // kept rect so the scrim doesn't leave a control-free tinted band
        // beside the bar. Uses the SAME animated ratio the scrim/frost/preview
        // frame with, so all three keep describing one rectangle. `None`
        // (Native, or non-Photo) leaves `base` untouched → portrait unchanged.
        crate::app::preview_geometry::expanded_insets_for_ratio(
            base,
            self.screen_width,
            self.screen_height,
            self.crop_target_ratio(),
        )
    }

    /// [`Self::bar_insets`], but from the SETTLED bar heights rather than the
    /// animated ones.
    ///
    /// Used by the capture path: a shot taken mid-animation must crop against
    /// where the bars will land, not an in-flight value (see
    /// `settled_bottom_ui_height`'s docs).
    pub fn settled_bar_insets(&self) -> crate::app::preview_geometry::BarInsets {
        let layout = self.controls_bar_layout();
        let base = Self::map_bar_insets(
            layout,
            self.settled_top_ui_height(),
            self.settled_bottom_ui_height(),
        );
        // Match the capture path's crop: `cover_capture_crop` is fed
        // `display_ratio(portrait)`, so expand against the same ratio. Because
        // expanding then re-framing with that ratio reproduces the identical
        // kept rect, the saved crop is unchanged - only the reserved-bar
        // bookkeeping grows (guarded by
        // `cover_capture_crop_follows_the_screen_not_the_sensor_centre`, which
        // calls the crop with base insets directly).
        crate::app::preview_geometry::expanded_insets_for_ratio(
            base,
            self.screen_width,
            self.screen_height,
            self.settled_crop_target_ratio(),
        )
    }

    /// The aspect ratio the SETTLED capture path crops against, or `None` when
    /// no crop applies. This is EXACTLY the ratio every `cover_capture_crop`
    /// call site in `capture.rs` is handed: `display_ratio(portrait)` in the
    /// Cover branch, and nothing at all in the fit-to-view branch (where the
    /// crop is sensor-space and the insets are irrelevant). Keeping the two in
    /// lockstep is what makes the expansion crop-preserving - expand with ratio
    /// R, then `cover_capture_crop` re-frames with the same R and lands the
    /// identical kept rect.
    fn settled_crop_target_ratio(&self) -> Option<f32> {
        if self.preview_fit_to_view {
            None
        } else {
            self.photo_aspect_ratio
                .display_ratio(self.screen_is_portrait())
        }
    }

    /// Places the top/bottom bar extents onto whichever edges `layout` puts
    /// them on. Single source of truth shared by [`Self::bar_insets`] and
    /// [`Self::settled_bar_insets`] so the edge mapping cannot drift between
    /// the animated and settled variants.
    fn map_bar_insets(
        layout: crate::app::bar_layout::BarLayout,
        top: f32,
        bottom: f32,
    ) -> crate::app::preview_geometry::BarInsets {
        use crate::app::bar_layout::Edge;
        use crate::app::preview_geometry::BarInsets;

        match (layout.top_bar, layout.bottom_bar) {
            (Edge::Left, Edge::Right) => BarInsets::vertical(top, bottom),
            (Edge::Right, Edge::Left) => BarInsets::vertical(bottom, top),
            _ => BarInsets::horizontal(top, bottom),
        }
    }

    /// Wrap `bar` in a container pinned to `edge`.
    ///
    /// Fills the whole stack layer, then aligns the (already sized) bar
    /// against the requested edge. The bar's own container decides its
    /// extent on the cross axis (e.g. `build_top_bar` / `build_bottom_bar`
    /// switch between `Fixed` and `Fill` based on
    /// `bar_layout::is_sideways`); this helper only decides *where* that
    /// extent lands.
    fn pin_to_edge<'a>(
        &self,
        bar: Element<'a, Message>,
        edge: crate::app::bar_layout::Edge,
    ) -> Element<'a, Message> {
        use crate::app::bar_layout::Edge;
        use cosmic::iced::alignment::{Horizontal, Vertical};

        let c = widget::container(bar)
            .width(Length::Fill)
            .height(Length::Fill);

        match edge {
            Edge::Top => c.align_y(Vertical::Top).align_x(Horizontal::Center),
            Edge::Bottom => c.align_y(Vertical::Bottom).align_x(Horizontal::Center),
            Edge::Left => c.align_x(Horizontal::Left).align_y(Vertical::Center),
            Edge::Right => c.align_x(Horizontal::Right).align_y(Vertical::Center),
        }
        .into()
    }

    /// Settled height of the empty placeholder above the bottom bar. 0 in
    /// View (no capture button — fit/zoom row sits flush above the
    /// carousel) and while the chrome is hidden; the capture button area
    /// otherwise.
    pub fn settled_capture_area_height(&self) -> f32 {
        if self.mode.is_view_only() || self.ui_hidden {
            0.0
        } else {
            let spacing = cosmic::theme::spacing();
            crate::app::controls::capture_button::CAPTURE_BUTTON_OUTER_SIZE
                + 2.0 * f32::from(spacing.space_xs)
        }
    }

    /// Animated capture-area placeholder height. Interpolates through
    /// `fit_animation` so Photo↔View glides the fit/zoom row toward the
    /// carousel instead of snapping.
    pub fn capture_area_height(&self) -> f32 {
        let target = self.settled_capture_area_height();
        let Some(anim) = self.fit_animation else {
            return target;
        };
        anim.from.capture_area_height
            + (target - anim.from.capture_area_height) * self.fit_animation_eased()
    }

    /// Height the floating fit/zoom chip row occupies at the BOTTOM of the
    /// preview content area, i.e. *inside* `frame_rect_on_screen` rather than
    /// below it — the chips deliberately float over the live preview, so
    /// `bottom_ui_height()` stops above them.
    ///
    /// **Invariant**: must match the row `view()` actually builds — the fit
    /// chip's fixed `space_l` height plus the `control_spacing` (`space_xs`)
    /// bottom padding of its centring container, under the same visibility
    /// condition. The zoom chip is a `button::text` of the same standard
    /// height, and the row is `align_y(Center)`, so `space_l` is the row's
    /// height. Grep `show_zoom_label` if that row changes shape.
    ///
    /// Used by `build_overlay_popup` to keep popups out of the chip strip.
    fn zoom_chip_strip_height(&self) -> f32 {
        if self.mode.supports_fit_and_zoom() && !self.tools_menu_visible && !self.ui_hidden {
            let spacing = cosmic::theme::spacing();
            f32::from(spacing.space_l) + f32::from(spacing.space_xs)
        } else {
            0.0
        }
    }
}

/// Build a centered overlay popup dialog with icon, title, body text, and optional button
///
/// Used for modal-style popups (privacy warning, flash error). Frosted like the
/// rest of the overlay chrome: a live-blurred preview backdrop behind the theme's
/// translucent surface when frosting is on, opaque when it's off. The blur is
/// what keeps the text legible, so this no longer needs the near-opaque hardcoded
/// alpha it used to carry.
///
/// Takes `model` (rather than standing alone) because the backdrop needs the
/// current frame and preview transforms, which `frosted_panel` reads off it.
fn build_overlay_popup<'a>(
    model: &'a AppModel,
    icon: Element<'a, Message>,
    title: &str,
    body: &str,
    button: Option<Element<'a, Message>>,
) -> Element<'a, Message> {
    let spacing = cosmic::theme::spacing();

    let mut content = widget::Column::new()
        .push(icon)
        .push(
            widget::text(title.to_string())
                .size(20)
                .font(cosmic::font::bold()),
        )
        .push(widget::text(body.to_string()).size(14))
        .spacing(spacing.space_s)
        .align_x(Alignment::Center);

    if let Some(btn) = button {
        content = content.push(btn);
    }

    // Padding goes on an inner container: `frosted_panel` wraps `content` in the
    // styled container itself, so the tint sizes to the padded content.
    let popup_box = model.frosted_panel(
        widget::container(content).padding(spacing.space_l).into(),
        POPUP_PANEL,
    );

    // Centre the popup in the CAMERA PREVIEW, not the window. The popup layer
    // is a full-window stack child, so window-centring dropped it below the
    // preview's middle (the bottom UI is much taller than the top bar) and let
    // it collide with the fit/zoom chips floating at the preview's bottom edge.
    //
    // Padding the layer by the UI bar heights reproduces `frame_rect_on_screen`
    // without re-deriving it: that helper's content rect spans
    // `top_ui_height()..H - bottom_ui_height()`, and both of its aspect-ratio
    // branches keep the framed rect *concentric* with that content rect. So a
    // Fill container inset by the two bar heights and centred lands exactly on
    // the frame rect's centre — for every ratio, and animating in step with the
    // bars during a Photo↔View transition.
    //
    // The bottom inset additionally reserves the chip strip, which lives INSIDE
    // the content rect (`bottom_ui_height()` stops above the chips). Centring on
    // the bare preview centre would clear today's popups by ~65 px but is not
    // collision-proof — a long flash-error message grows the popup vertically
    // and would reach the chips again. Reserving the strip makes the clearance
    // structural, and costs only ~22 px of upward shift (half the strip), so the
    // popup still reads as centred on the preview.
    widget::container(popup_box)
        .width(Length::Fill)
        .height(Length::Fill)
        .padding(overlay_popup_padding(model))
        .align_x(cosmic::iced::alignment::Horizontal::Center)
        .align_y(cosmic::iced::alignment::Vertical::Center)
        .into()
}

/// Create an icon button with a themed background for use on camera preview overlays.
/// `highlighted = true` switches to the accent (Suggested) class so toggle-state
/// buttons (flash, HDR, tools menu) show their active state visually.
fn overlay_icon_button<'a, M: Clone + 'static>(
    handle: impl Into<widget::icon::Handle>,
    message: Option<M>,
    highlighted: bool,
) -> Element<'a, M> {
    let mut button = widget::button::icon(handle).extra_small();
    if highlighted {
        button = button.class(cosmic::theme::Button::Suggested);
    }
    if let Some(msg) = message {
        button = button.on_press(msg);
    }
    button.into()
}

/// Dim a top-bar icon that is inert while the UI is disabled mid-transition.
///
/// Symbolic icons read `icon_color`, not `text_color`; the top bar now publishes
/// an ambient `icon_color` of its own (accent while focused), so the dimming has
/// to override that channel or the disabled icons come out fully accented.
fn disabled_top_bar_icon_style(_theme: &cosmic::Theme) -> widget::container::Style {
    let dimmed = Color::from_rgba(1.0, 1.0, 1.0, 0.3);
    widget::container::Style {
        icon_color: Some(dimmed),
        text_color: Some(dimmed),
        ..Default::default()
    }
}

/// Animation duration for fit/fill transition.
pub const FIT_ANIMATION_DURATION: std::time::Duration = std::time::Duration::from_millis(300);

/// Animation duration for the zoom-reset transition.
pub const ZOOM_ANIMATION_DURATION: std::time::Duration = std::time::Duration::from_millis(300);

impl AppModel {
    /// Whether the main window currently holds keyboard focus.
    ///
    /// Mirrors the check libcosmic runs before styling its own header bar, so
    /// our custom title bar switches between focused and unfocused colours at
    /// the same moment a native COSMIC header bar would.
    pub fn window_is_focused(&self) -> bool {
        let main_window = self.core.main_window_id();
        self.core
            .focus_chain()
            .iter()
            .any(|id| Some(*id) == main_window)
    }

    /// Settled cover blend: 0.0 (Contain) when fit-to-view is enabled in a
    /// mode that supports it (Photo, View), 1.0 (Cover) everywhere else.
    /// The single source of truth for the preview's geometry target.
    ///
    /// Virtual mode is forced to Contain regardless of the toggle — the
    /// fit/fill chip is hidden there anyway (it's gated on
    /// `supports_fit_and_zoom`), and Cover would silently crop edges off
    /// what's being streamed to consumer apps, which doesn't match the
    /// "what you see is what you send" expectation.
    pub fn settled_blend(&self) -> f32 {
        if matches!(self.mode, crate::app::state::CameraMode::Virtual) {
            return 0.0;
        }
        if self.preview_fit_to_view && self.mode.supports_fit_and_zoom() {
            0.0
        } else {
            1.0
        }
    }

    /// Animated zoom level. During an in-flight zoom-reset transition,
    /// interpolates from the captured starting zoom toward `self.zoom_level`
    /// using the same ease-out cubic shape as the fit/fill animation.
    /// Pinch and step zoom clear `zoom_animation`, so they remain real-time.
    pub fn current_zoom_level(&self) -> f32 {
        let target = self.zoom_level;
        let Some(anim) = self.zoom_animation else {
            return target;
        };
        let t =
            (anim.start.elapsed().as_secs_f32() / ZOOM_ANIMATION_DURATION.as_secs_f32()).min(1.0);
        let eased = 1.0 - (1.0 - t).powi(3);
        anim.from + (target - anim.from) * eased
    }

    /// Eased progress of the in-flight fit animation, in [0, 1]. Returns 1.0
    /// when no animation is running (i.e. fully settled).
    fn fit_animation_eased(&self) -> f32 {
        let Some(anim) = self.fit_animation else {
            return 1.0;
        };
        let t =
            (anim.start.elapsed().as_secs_f32() / FIT_ANIMATION_DURATION.as_secs_f32()).min(1.0);
        // Ease-out cubic: 1 - (1-t)^3
        1.0 - (1.0 - t).powi(3)
    }

    /// Returns the current cover blend value (0.0 = contain/fit, 1.0 = cover/fill).
    /// During animation, returns an ease-out interpolation toward `settled_blend()`.
    pub fn cover_blend(&self) -> f32 {
        let target = self.settled_blend();
        let Some(anim) = self.fit_animation else {
            return target;
        };
        anim.from.blend + (target - anim.from.blend) * self.fit_animation_eased()
    }

    /// Snapshot every value that animates through a fit/fill transition.
    /// Callers take this *before* mutating `self.mode` or
    /// `self.preview_fit_to_view`, then pass the snapshot to
    /// `start_fit_animation`. Centralising the read here means a new
    /// animated channel only needs to be added once (struct + this method
    /// + the matching settled getter).
    pub fn capture_fit_state(&self) -> crate::app::state::FitFrom {
        crate::app::state::FitFrom {
            blend: self.cover_blend(),
            top_ui_height: self.top_ui_height(),
            bottom_ui_height: self.bottom_ui_height(),
            capture_area_height: self.capture_area_height(),
        }
    }

    /// Install a fit/fill animation if any of the animated values differ
    /// from where the eye currently is, returning the tick task that drives
    /// it (or `Task::none` when no animation is needed). Callers must mutate
    /// `self.mode` and/or `self.preview_fit_to_view` before calling so the
    /// settled values reflect the new state. If a tick chain is already in
    /// flight (i.e. `fit_animation` was already `Some`), no new chain is
    /// spawned — the existing one picks up the replaced animation on its
    /// next fire, so re-triggers don't double the tick rate.
    pub fn start_fit_animation(
        &mut self,
        from: crate::app::state::FitFrom,
    ) -> cosmic::Task<cosmic::Action<Message>> {
        let target_blend = self.settled_blend();
        let target_top = self.settled_top_ui_height();
        let target_bottom = self.settled_bottom_ui_height();
        let target_capture = self.settled_capture_area_height();
        let differs = (target_blend - from.blend).abs() > f32::EPSILON
            || (target_top - from.top_ui_height).abs() > f32::EPSILON
            || (target_bottom - from.bottom_ui_height).abs() > f32::EPSILON
            || (target_capture - from.capture_area_height).abs() > f32::EPSILON;
        if !differs {
            return cosmic::Task::none();
        }
        // The animation crosses the View-mode boundary whenever the source
        // or destination has zero capture-area height (View's signature).
        // Storing this explicitly means downstream rendering paths don't
        // have to infer it from a height comparison.
        let is_view_boundary =
            from.capture_area_height <= f32::EPSILON || target_capture <= f32::EPSILON;
        let was_idle = self.fit_animation.is_none();
        self.fit_animation = Some(crate::app::state::FitAnimation {
            start: std::time::Instant::now(),
            from,
            is_view_boundary,
        });
        if was_idle {
            Self::delay_task(16, Message::FitAnimationTick)
        } else {
            cosmic::Task::none()
        }
    }

    /// Build the main application view
    ///
    /// Composes all UI components into a layered layout with overlays.
    pub fn view(&self) -> Element<'_, Message> {
        static HAS_RENDERED: AtomicBool = AtomicBool::new(false);
        if !HAS_RENDERED.swap(true, Ordering::Relaxed) {
            info!("first UI render");
        }

        // Camera preview from camera_preview module
        let camera_preview = self.build_camera_preview();

        // Flash mode - show only preview with white overlay, no UI
        // Only show screen flash overlay for front cameras (back cameras use hardware LED)
        if self.flash.active && !self.use_hardware_flash() {
            let flash_overlay = widget::container(
                widget::Space::new()
                    .width(Length::Fill)
                    .height(Length::Fill),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .style(|_theme| widget::container::Style {
                background: Some(Background::Color(Color::WHITE)),
                ..Default::default()
            });

            return widget::container(
                cosmic::iced::widget::stack![camera_preview, flash_overlay]
                    .width(Length::Fill)
                    .height(Length::Fill),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .style(window_bg_style)
            .into();
        }

        // Burst mode capture/processing - show progress overlay
        if self.burst_mode.is_active() {
            let burst_mode_overlay = self.build_burst_mode_overlay();
            return widget::container(
                cosmic::iced::widget::stack![camera_preview, burst_mode_overlay]
                    .width(Length::Fill)
                    .height(Length::Fill),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .style(window_bg_style)
            .into();
        }

        // Build top bar
        let top_bar = self.build_top_bar();

        // Zoom/fit row is shown in modes that allow manual zoom and the
        // fit-to-view toggle (Photo, View), and never while the chrome is
        // hidden.
        let show_zoom_label = self.mode.supports_fit_and_zoom() && !self.ui_hidden;

        // Capture button area - changes based on recording/streaming state and video file selection
        // Check if we have video file controls (play/pause button for video file sources)
        let play_pause_button = self.build_video_play_pause_button();
        let has_video_controls = play_pause_button.is_some();

        // Held sideways the bars move to the side edges; the capture area is a
        // narrow (~100px) vertical lane, so the recording three-slot layout
        // stacks into a Column instead of the portrait Row.
        let sideways = self.controls_are_sideways();
        // The recording/streaming state that replaces the single capture button
        // with the [play·stop·photo] three-slot layout.
        let recording_layout = (self.recording.is_recording() && !self.quick_record.is_recording())
            || self.virtual_camera.is_streaming();

        let capture_button_only = if recording_layout {
            // Mirror the bottom bar's three-column layout so the stop circle
            // sits where the carousel does and the photo button lines up with
            // the camera-switch position. Portrait uses `three_col_row` (the
            // shared shape); sideways stacks the same three slots into a Column
            // that fits the narrow strip. The side spacer width and center
            // container width match the bottom bar's gallery/switch buttons and
            // carousel width.
            let stop_circle = self.build_capture_circle();
            let photo_button = self.build_photo_during_recording_button();
            let slide = std::sync::Arc::clone(&self.carousel_button_slide);

            let spacing = cosmic::theme::spacing();
            let side_width = ui::PLACEHOLDER_BUTTON_WIDTH;
            let center_width = crate::app::bottom_bar::mode_carousel::carousel_width_for_modes(
                &self.available_modes(),
            );

            // While the virtual camera is streaming a video file source, keep
            // the play/pause control reachable in the left slot (it's hidden
            // by the streaming layout otherwise). For the camera-source
            // streaming case `play_pause_button` is `None` and we fall back
            // to the original spacer.
            let left_slot: Element<'_, Message> = if let Some(pp_button) = play_pause_button {
                widget::container(pp_button)
                    .width(Length::Fixed(side_width))
                    .center_x(side_width)
                    .into()
            } else {
                widget::Space::new()
                    .width(Length::Fixed(side_width))
                    .height(Length::Shrink)
                    .into()
            };
            let center_slot: Element<'_, Message> = widget::container(stop_circle)
                .width(Length::Fixed(center_width))
                .center_x(center_width)
                .into();

            if sideways {
                // Same three slots as portrait, stacked to fit the vertical
                // strip. Mirror-aware: a rotate-left (270°/Ccw90) turn reverses
                // the column so the stop circle keeps mirroring the carousel
                // (whose labels already flip bottom↔top there); rotate-right
                // (90°) keeps portrait order. See `sideways_column_reverses`.
                let photo_slot: Element<'_, Message> = widget::container(photo_button)
                    .width(Length::Fixed(side_width))
                    .center_x(side_width)
                    .into();
                let mut slots: Vec<Element<'_, Message>> = vec![left_slot, center_slot, photo_slot];
                if sideways_column_reverses(self.controls_bar_layout().quarter) {
                    slots.reverse();
                }
                widget::Column::with_children(slots)
                    .align_x(cosmic::iced::alignment::Horizontal::Center)
                    .spacing(spacing.space_m)
                    .width(Length::Fill)
                    .into()
            } else {
                // Vertical padding matches build_capture_button so the circle
                // doesn't shift when the layout flips between idle and recording.
                crate::app::bottom_bar::three_col_row(
                    left_slot,
                    // Wrap the center container in a SlidePrimer so it publishes the
                    // resting carousel slide (from its own bounds) into the shared
                    // atomic. During recording the carousel isn't in the tree, so
                    // without this the photo button's SlideH would read a stale/zero
                    // offset - the misplacement seen in spoofed preview screenshots.
                    crate::app::bottom_bar::slide_h::SlidePrimer::new(
                        center_slot,
                        std::sync::Arc::clone(&self.carousel_button_slide),
                    )
                    .into(),
                    SlideH::new(photo_button, slide, -1.0).into(),
                    [spacing.space_xs, spacing.space_m],
                )
            }
        } else if has_video_controls {
            // Video file selected but not streaming: show play button + capture button
            let capture_button = self.build_capture_button();
            let icon_button_width = crate::constants::ui::ICON_BUTTON_WIDTH;

            // Layout: [Fill] [Play container] [Capture] [Spacer matching Play] [Fill]
            // Use fixed-width container for play button to ensure centering
            let mut row = widget::Row::new().push(
                widget::Space::new()
                    .width(Length::Fill)
                    .height(Length::Shrink),
            );

            if let Some(pp_button) = play_pause_button {
                // Wrap play/pause button in fixed-width container for consistent centering
                row = row.push(
                    widget::container(pp_button)
                        .width(Length::Fixed(icon_button_width))
                        .align_x(cosmic::iced::alignment::Horizontal::Center),
                );
            }

            row = row
                .push(capture_button)
                // Spacer matches play/pause button width for centering
                .push(
                    widget::Space::new()
                        .width(Length::Fixed(icon_button_width))
                        .height(Length::Shrink),
                )
                .push(
                    widget::Space::new()
                        .width(Length::Fill)
                        .height(Length::Shrink),
                )
                .align_y(Alignment::Center)
                .width(Length::Fill);

            row.into()
        } else {
            // Normal single capture button
            self.build_capture_button()
        };

        // Capture button area (filter name label is now an overlay on the
        // preview). Wrap in a fixed-height container driven by the animated
        // `capture_area_height` so the slot collapses to 0 when entering
        // View and expands back when leaving — the fit/zoom row above
        // glides toward / away from the carousel. The capture button
        // itself, however, pops in/out instead of being gradually clipped:
        // we render an empty Space whenever a View transition is in flight
        // and only swap to the real button once it's at its settled height.
        let capture_h = self.capture_area_height();
        let view_transition_in_flight = self.fit_animation.is_some_and(|a| a.is_view_boundary);
        let inner: Element<'_, Message> = if self.mode.is_view_only() || view_transition_in_flight {
            widget::Space::new()
                .width(Length::Fill)
                .height(Length::Shrink)
                .into()
        } else {
            capture_button_only
        };
        // Portrait always pins the lane to `capture_h` (drives the View↔Photo
        // collapse animation). Sideways, the taller stacked recording Column
        // needs to size to its own content instead of being clipped to the
        // ~100px lane - see `capture_area_height_is_fixed`.
        let mut capture_area = widget::container(inner).width(Length::Fill).clip(true);
        if crate::app::bar_layout::capture_area_height_is_fixed(sideways, recording_layout) {
            capture_area = capture_area.height(Length::Fixed(capture_h.max(0.0)));
        }
        let capture_button_area: Element<'_, Message> = capture_area.into();

        // Bottom area: always show bottom bar (filter picker is now a sidebar
        // overlay). Skipped entirely while the chrome is hidden — the column
        // below drops it, so building the carousel would be wasted work.
        let bottom_area: Element<'_, Message> = if self.ui_hidden {
            widget::Space::new()
                .width(Length::Fill)
                .height(Length::Shrink)
                .into()
        } else {
            self.build_bottom_bar()
        };

        // Immersive layout: camera preview fills the screen, all UI overlaid on top.
        // Aspect ratio crop is shown as translucent top/bottom bars (canvas overlay).
        let content: Element<'_, Message> = {
            let spacing = cosmic::theme::spacing();
            let control_spacing = spacing.space_xs;

            // Which edge each bar lands on for the current display transform.
            // The window itself never rotates, so this is the single place
            // that decides "top bar" and "bottom bar" mean physically-left,
            // -right, -top or -bottom.
            let layout = self.controls_bar_layout();

            // Hidden chrome: the whole bottom column collapses — no progress
            // bar, no capture button, no carousel. `show_zoom_label` above
            // already dropped the fit/zoom chips, so the bottom stack child
            // becomes an empty layer over the preview.
            //
            // Portrait: the capture button sits above the
            // [gallery · carousel · switcher] group, which is a row below it.
            // Sideways: the same relationship rotated 90° - the capture button
            // and the group sit SIDE BY SIDE across the strip's width (each
            // centred along the strip's length), rather than stacked one after
            // the other down the strip. The capture stays on the preview-facing
            // (inner) side; which physical side that is depends on the strip
            // edge, so the two lanes mirror between 270° and 90°.
            let video_progress_bar = self.build_video_progress_bar();
            let bottom_controls: Element<'_, Message> = if self.ui_hidden {
                widget::Space::new()
                    .width(Length::Fill)
                    .height(Length::Shrink)
                    .into()
            } else if sideways {
                use crate::app::bar_layout::Edge;
                // Capture lane takes the strip width minus the bar-lane
                // (SIDEWAYS_STRIP_WIDTH - BOTTOM_BAR_HEIGHT); the group lane
                // fills the rest. Both run the full strip length and centre
                // their content along it.
                //
                // The row carries `align_y(Center)`. Without it the row's
                // cross-axis defaults to `Start`, so a group lane that resolves
                // to its compact content height (the gallery / carousel /
                // switcher stack ≈ its own height, not the full strip) is
                // pinned to the TOP of the strip, while the capture lane - whose
                // content is a fixed ~100px area that its own `center_y` already
                // parks in the middle - still reads as centred. That mismatch is
                // the "capture centred, group jammed to the top" bug. Centring
                // the row's cross axis parks the compact group in the middle of
                // the strip alongside the capture button, and is applied in the
                // flex alignment pass regardless of how each lane's height
                // resolves (it is a no-op for a lane that already fills the
                // strip, so the capture lane is unaffected).
                // Both lanes are Shrink (their compact content height); the row
                // height resolves to the taller lane (the group), and
                // `align_y(Center)` parks the shorter capture lane at its centre.
                // Keeping the whole cluster COMPACT (not Fill) lets
                // `pin_to_edge`'s own vertical centring place it at the window's
                // middle. Filling the strip height here instead hit iced's flex
                // `compression` fallback and settled the cluster ~70px low.
                let capture_lane = widget::container(capture_button_area)
                    .width(Length::Fixed(
                        crate::app::bar_layout::SIDEWAYS_STRIP_WIDTH
                            - crate::app::bottom_bar::BOTTOM_BAR_HEIGHT,
                    ))
                    .center_x(Length::Fill);
                let group_lane = widget::container(bottom_area).width(Length::Fill);
                let row = widget::Row::new()
                    .width(Length::Fill)
                    .align_y(cosmic::iced::alignment::Vertical::Center);
                let lanes: Element<'_, Message> = if layout.bottom_bar == Edge::Right {
                    // Strip on the right: inner (preview) edge is the left →
                    // capture on the left, group toward the screen edge.
                    row.push(capture_lane).push(group_lane)
                } else {
                    // Strip on the left (mirror): capture on the right (inner).
                    row.push(group_lane).push(capture_lane)
                }
                .into();

                let mut side_controls = widget::Column::new().width(Length::Fill);
                if video_progress_placement(sideways) == VideoProgressPlacement::AboveSideLanes
                    && let Some(progress_bar) = video_progress_bar
                {
                    side_controls = side_controls.push(progress_bar);
                }
                side_controls.push(lanes).into()
            } else {
                let mut col = widget::Column::new().width(Length::Fill);
                if video_progress_placement(sideways) == VideoProgressPlacement::AboveCapture
                    && let Some(progress_bar) = video_progress_bar
                {
                    col = col.push(progress_bar);
                }
                col.push(capture_button_area).push(bottom_area).into()
            };

            // Bottom section: zoom label + bottom controls.
            //
            // Sideways, this is the whole right-pinned control strip: it
            // moves from window-width to a fixed strip width
            // (`SIDEWAYS_STRIP_WIDTH`) and carries the `Fill` height itself -
            // the strip runs the full length of its edge - while its
            // children (the fit/zoom chips, capture button, and bottom bar)
            // stay `Shrink` and stack along it in the same order as portrait.
            let mut bottom_section = widget::Column::new().width(Length::Fill);
            if sideways {
                use cosmic::iced::alignment::Horizontal;
                bottom_section = bottom_section
                    .width(Length::Fixed(crate::app::bar_layout::SIDEWAYS_STRIP_WIDTH))
                    .height(Length::Shrink)
                    .align_x(Horizontal::Center);
            }

            // Fit/zoom chips. Hidden while the tools menu is open so the two
            // don't visually compete — the menu itself is shown as an overlay.
            //
            // Portrait: stacked into the strip (`bottom_section`) just above the
            // capture button, byte-identical to before. Sideways (decision
            // D4-B): pulled OUT of the strip and floated over the preview
            // against the strip's INNER edge, vertically centred - returned as a
            // separate stack layer (`sideways_chip_layer`) added over the
            // preview below, mirrored per rotation.
            let sideways_chip_layer: Option<Element<'_, Message>> = if show_zoom_label
                && !self.tools_menu_visible
            {
                match fit_zoom_layout(sideways) {
                    FitZoomLayout::FloatUpright => Some(self.float_chips_over_preview(
                        self.build_fit_zoom_group(fit_zoom_axis(sideways)),
                        layout.bottom_bar,
                    )),
                    FitZoomLayout::Inline => {
                        bottom_section = bottom_section.push(
                            widget::container(self.build_fit_zoom_group(fit_zoom_axis(sideways)))
                                .width(Length::Fill)
                                .center_x(Length::Fill)
                                .padding([0, 0, control_spacing, 0]),
                        );
                        None
                    }
                }
            } else {
                None
            };

            // Sideways, `bottom_controls` is itself a Fill-height Row whose two
            // lanes centre their content along the strip, so pushing it directly
            // into the Fill-height strip already centres the capture+group.
            // Portrait pushes the Shrink Column as before.
            // Sideways, the centred cluster settles noticeably LOW: the region
            // pin_to_edge centres within sits below the visible strip's middle
            // (a compositor/window vertical-inset interaction the widget layout
            // can't see). Lift the cluster with a trailing spacer so it reads
            // centred on the strip. Tuned against on-device screenshots; may need
            // adjusting if the panel insets differ on another device.
            if sideways {
                bottom_section = bottom_section
                    .push(bottom_controls)
                    .push(widget::Space::new().height(Length::Fixed(0.0)));
            } else {
                bottom_section = bottom_section.push(bottom_controls);
            }

            // The shader handles the Cover/Contain blend via cover_blend(), so
            // the preview always uses Cover layout (fills the window).  The shader
            // zooms out to show the full frame in Contain mode, with transparent
            // letterbox areas.
            let camera_layer: Element<'_, Message> = camera_preview;

            let top_bar_layer = self.pin_to_edge(top_bar, layout.top_bar);
            let bottom_bar_layer = self.pin_to_edge(bottom_section.into(), layout.bottom_bar);

            let mut main_stack = cosmic::iced::widget::stack![
                camera_layer,
                self.frosted_bars(),
                self.build_crop_overlay(),
                self.build_composition_overlay(),
                self.build_qr_overlay(),
                self.build_privacy_warning(),
                top_bar_layer,
                bottom_bar_layer
            ];

            // Sideways only: the fit/zoom chips float over the preview against
            // the strip's inner edge (see `sideways_chip_layer` above). Layered
            // over the bars but under the modal popups pushed below.
            if let Some(chip_layer) = sideways_chip_layer {
                main_stack = main_stack.push(chip_layer);
            }

            if self.flash.error_popup.is_some() {
                main_stack = main_stack.push(self.build_flash_error_popup());
            }

            if let Some(remaining) = self.photo_timer_countdown {
                main_stack = main_stack.push(self.build_timer_overlay(remaining));
            }

            main_stack.width(Length::Fill).height(Length::Fill).into()
        };

        // Wrap content in a stack so we can overlay the picker
        let mut main_stack = cosmic::iced::widget::stack![content];

        // Add format picker overlay if visible
        // Hide with libcamera backend in photo/video modes (resolution is handled automatically)
        if self.format_picker_visible && !self.is_format_picker_hidden() {
            main_stack = main_stack.push(self.build_format_picker());
        }

        // Add exposure picker overlay if visible
        if self.exposure_picker_visible {
            main_stack = main_stack.push(self.build_exposure_picker());
        }

        // Add color picker overlay if visible
        if self.color_picker_visible {
            main_stack = main_stack.push(self.build_color_picker());
        }

        // Add motor/PTZ controls picker overlay if visible
        if self.motor_picker_visible {
            main_stack = main_stack.push(self.build_motor_picker());
        }

        // Add tools menu overlay if visible
        if self.tools_menu_visible {
            main_stack = main_stack.push(self.build_tools_menu());
        }

        // Context drawer (Settings, Filters, Insights, Shortcuts) last, so it
        // sits above every other overlay.
        if self.core.window.show_context {
            main_stack = main_stack.push(self.build_context_drawer());
        }

        // Wrap everything in the window background container. This is the single
        // layer that paints the app's window background — see [`window_bg_style`]
        // — so with COSMIC's window frosting on it is translucent and the
        // compositor's blurred wallpaper reads through everywhere the camera
        // image does not cover (issue #569).
        widget::container(main_stack)
            .width(Length::Fill)
            .height(Length::Fill)
            .style(window_bg_style)
            .into()
    }

    /// Width of the context drawer pane.
    ///
    /// Mirrors libcosmic's `Core::context_width`, which is crate-private: it
    /// reserves 360 px of content (plus 8 px padding) for the main view and
    /// clamps the drawer to 344..=480 px.
    fn context_drawer_width(&self) -> f32 {
        (self.screen_width - (360.0 + 8.0)).clamp(344.0, 480.0)
    }

    /// Reserve vertical title-bar space only while the title bar is horizontal.
    fn context_drawer_top_inset(layout: crate::app::bar_layout::BarLayout) -> f32 {
        if layout.quarter != crate::app::bar_layout::Quarter::None {
            0.0
        } else {
            TOP_BAR_HEIGHT
        }
    }

    /// Keep drawers on their standard right edge unless automatic display
    /// orientation moved the vertical title bar there. A manual control-bar
    /// position must not move drawers along with it.
    fn context_drawer_aligns_right(&self, layout: crate::app::bar_layout::BarLayout) -> bool {
        self.config.controls_position != crate::config::ControlsPosition::Bottom
            || layout.top_bar != crate::app::bar_layout::Edge::Right
    }

    /// Build the context drawer (Settings, Filters, Insights, Shortcuts) as a
    /// view-level overlay.
    ///
    /// libcosmic would normally render this for us from
    /// `Application::context_drawer`, but it pins the drawer to the top of the
    /// main view as an iced overlay. This app draws its title bar *inside* that
    /// view, so the drawer swallowed the window controls whenever it opened
    /// (issue #565). Building it here lets the pane start below
    /// [`TOP_BAR_HEIGHT`], leaving the title bar visible and clickable.
    fn build_context_drawer(&self) -> Element<'_, Message> {
        use crate::app::state::ContextPage;

        let drawer = match self.context_page {
            ContextPage::Settings => self.settings_view(),
            ContextPage::Filters => self.filters_view(),
            ContextPage::Insights => self.insights_view(),
            ContextPage::KeyBindings => crate::app::keybind::key_bindings_page::view(self),
        };

        let width = self.context_drawer_width();
        let pane = widget::ContextDrawer::new_inner_overlay(
            drawer.title,
            drawer.actions,
            drawer.header,
            drawer.footer,
            drawer.content,
            drawer.on_close,
            width,
            // Opaque pane: it floats over the live preview, exactly as
            // libcosmic's own overlay drawer does.
            true,
        );

        // Swallow presses that land on the pane so they don't reach the preview
        // underneath — the same guard the picker overlays use.
        let pane = widget::mouse_area(
            widget::container(pane)
                .width(Length::Fixed(width))
                .height(Length::Fill),
        )
        .on_press(Message::Noop);

        let spring = widget::space::horizontal().width(Length::Fill);
        let layout = self.controls_bar_layout();
        let row = if self.context_drawer_aligns_right(layout) {
            widget::Row::new().push(spring).push(pane)
        } else {
            widget::Row::new().push(pane).push(spring)
        };

        widget::container(row.padding([
            Self::context_drawer_top_inset(layout) as u16,
            CONTEXT_DRAWER_INSET,
            CONTEXT_DRAWER_INSET,
            0,
        ]))
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
    }

    /// Build the top bar with recording indicator and format button
    fn build_top_bar(&self) -> Element<'_, Message> {
        // View mode and hidden chrome strip every top-bar button (and the
        // title-bar window controls) but keep the draggable row so the user
        // can still move / double-click-to-maximize the window. Without it a
        // hidden-chrome window would be unmovable on compositors that have no
        // titlebar of their own.
        if self.mode.is_view_only() || self.ui_hidden {
            let sideways = self.controls_are_sideways();
            let space = widget::Space::new()
                .width(Length::Fill)
                .height(Length::Fill);
            let mut empty = widget::container(space).style(|_theme| widget::container::Style {
                background: Some(Background::Color(Color::TRANSPARENT)),
                ..Default::default()
            });
            let (width, height) = bar_cross_lengths(sideways, TOP_BAR_HEIGHT);
            empty = empty.width(width).height(height);
            return widget::mouse_area(empty)
                .on_drag(Message::WindowDrag)
                .on_double_press(Message::WindowToggleMaximize)
                .into();
        }

        let spacing = cosmic::theme::spacing();
        let is_disabled = self.transition_state.ui_disabled;
        let sideways = self.controls_are_sideways();

        // Axis-aware spacers. Portrait lays the bar out as a `Row`, so gaps run
        // horizontally; sideways it becomes a `Column`, so the same gaps run
        // vertically. In the portrait branch these produce `Space` widgets
        // byte-identical to the old inline constructions.
        let axis_gap = move |px: f32| -> Element<'static, Message> {
            let s = widget::Space::new();
            if sideways {
                s.height(Length::Fixed(px)).into()
            } else {
                s.width(Length::Fixed(px)).into()
            }
        };
        let fill_gap = move || -> Element<'static, Message> {
            let s = widget::Space::new();
            if sideways {
                s.height(Length::Fill).into()
            } else {
                s.width(Length::Fill).into()
            }
        };

        // Children in portrait order (leading -> trailing). Sideways, this same
        // sequence is folded into a `Column`; a rotate-left (270deg / Ccw90)
        // turn reverses it so the trailing window controls land at the strip's
        // TOP, and rotate-right (90deg) mirrors that - see
        // `bar_layout::sideways_column_reverses`.
        let mut cells: Vec<Element<'_, Message>> = Vec::new();

        // Show recording indicator when recording (from controls module)
        if let Some(indicator) = self.build_recording_indicator() {
            cells.push(indicator);
            cells.push(axis_gap(f32::from(spacing.space_s)));
        }

        // Show streaming indicator when streaming virtual camera
        if let Some(indicator) = self.build_streaming_indicator() {
            cells.push(indicator);
            cells.push(axis_gap(f32::from(spacing.space_s)));
        }

        // Show timelapse indicator when timelapse is running
        if let Some(indicator) = self.build_timelapse_indicator() {
            cells.push(indicator);
            cells.push(axis_gap(f32::from(spacing.space_s)));
        }

        // Show format/resolution button in both photo and video modes
        // Hide button when:
        // - Format picker is visible
        // - Recording in video mode
        // - Streaming virtual camera (resolution cannot be changed during streaming)
        // - File source is set in Virtual mode (show file resolution instead)
        let has_file_source =
            self.mode == CameraMode::Virtual && self.virtual_camera_file_source.is_some();
        let show_format_button = !self.format_picker_visible
            && (self.mode == CameraMode::Photo
                || self.mode == CameraMode::Timelapse
                || !self.recording.is_recording())
            && !self.virtual_camera.is_streaming()
            && !has_file_source
            && !self.is_format_picker_hidden();

        if show_format_button {
            cells.push(self.build_format_button());
        } else if has_file_source {
            // Show file source resolution (non-clickable)
            cells.push(self.build_file_source_resolution_label());
        }

        // Right side buttons
        cells.push(fill_gap());

        // Top-bar toggle buttons (flash, HDR, file, tools) are always
        // shown. Picker overlays appear on top of them but never replace them.
        // Flash toggle button (Photo mode, or Video/Timelapse mode with hardware flash for torch)
        if self.mode == CameraMode::Photo
            || ((self.mode == CameraMode::Video || self.mode == CameraMode::Timelapse)
                && self.use_hardware_flash())
        {
            let flash_icon_bytes = if self.flash.enabled {
                FLASH_ICON
            } else {
                FLASH_OFF_ICON
            };
            let flash_icon = widget::icon::from_svg_bytes(flash_icon_bytes).symbolic(true);

            if is_disabled {
                cells.push(
                    widget::container(widget::icon(flash_icon).size(20))
                        .style(disabled_top_bar_icon_style)
                        .padding([4, 8])
                        .into(),
                );
            } else {
                cells.push(overlay_icon_button(
                    flash_icon,
                    Some(Message::ToggleFlash),
                    self.flash.enabled,
                ));
            }

            // 5px spacing
            cells.push(axis_gap(5.0));

            if self.should_show_burst_button() {
                // Show moon-off icon when HDR+ is disabled (by override or setting)
                let is_hdr_active = self.would_use_burst_mode();
                let moon_icon_bytes = if is_hdr_active {
                    MOON_ICON
                } else {
                    MOON_OFF_ICON
                };
                let moon_icon = widget::icon::from_svg_bytes(moon_icon_bytes).symbolic(true);

                if is_disabled {
                    cells.push(
                        widget::container(widget::icon(moon_icon).size(20))
                            .style(disabled_top_bar_icon_style)
                            .padding([4, 8])
                            .into(),
                    );
                } else {
                    cells.push(overlay_icon_button(
                        moon_icon,
                        Some(Message::ToggleBurstMode),
                        is_hdr_active,
                    ));
                }

                // 5px spacing
                cells.push(axis_gap(5.0));
            }
        }

        // File open button (only in Virtual mode, hidden when streaming)
        if self.mode == CameraMode::Virtual && !self.virtual_camera.is_streaming() {
            let has_file = self.virtual_camera_file_source.is_some();
            if is_disabled {
                let file_button =
                    widget::button::icon(icon::from_name("document-open-symbolic").symbolic(true));
                cells.push(
                    widget::container(file_button)
                        .style(disabled_top_bar_icon_style)
                        .into(),
                );
            } else {
                let message = if has_file {
                    Message::ClearVirtualCameraFile
                } else {
                    Message::OpenVirtualCameraFile
                };
                cells.push(overlay_icon_button(
                    icon::from_name("document-open-symbolic").symbolic(true),
                    Some(message),
                    has_file,
                ));
            }

            // 5px spacing
            cells.push(axis_gap(5.0));
        }

        // Tools menu button (opens overlay with timer, aspect ratio, exposure, filter, motor)
        // Highlight when tools menu is open or any tool setting is non-default
        let tools_active = self.tools_menu_visible || self.has_non_default_tool_settings();
        let tools_icon = widget::icon::from_svg_bytes(TOOLS_GRID_ICON).symbolic(true);

        if is_disabled {
            cells.push(
                widget::container(widget::icon(tools_icon).size(20))
                    .style(disabled_top_bar_icon_style)
                    .padding([4, 8])
                    .into(),
            );
        } else {
            cells.push(overlay_icon_button(
                tools_icon,
                Some(Message::ToggleToolsMenu),
                tools_active,
            ));
        }

        // Settings button (normally in header_end)
        if !is_disabled {
            cells.push(
                widget::button::icon(icon::from_name("preferences-system-symbolic").symbolic(true))
                    .extra_small()
                    .on_press(Message::ToggleContextPage(
                        crate::app::state::ContextPage::Settings,
                    ))
                    .into(),
            );
        }

        // Window control buttons
        cells.push(axis_gap(5.0));
        cells.push(
            widget::button::icon(icon::from_name("window-minimize-symbolic").symbolic(true))
                .extra_small()
                .on_press(Message::WindowMinimize)
                .into(),
        );
        cells.push(
            widget::button::icon(icon::from_name("window-maximize-symbolic").symbolic(true))
                .extra_small()
                .on_press(Message::WindowToggleMaximize)
                .into(),
        );
        cells.push(
            widget::button::icon(icon::from_name("window-close-symbolic").symbolic(true))
                .extra_small()
                .on_press(Message::WindowClose)
                .into(),
        );

        // Portrait keeps the historical `Row`; sideways folds the same cells
        // into a `Column`. The top bar's window controls belong at the strip's
        // TOP in BOTH rotations - a horizontal mirror preserves vertical order,
        // so unlike the bottom bar (which mirrors via `sideways_column_reverses`)
        // the portrait cells, whose window controls are trailing, are reversed
        // whenever sideways. Not doing this at 90° left them at the bottom and
        // pushed the cluster partly off-screen.
        let content: Element<'_, Message> = if sideways {
            cells.reverse();
            widget::Column::with_children(cells)
                .padding([7, 7, 8, 7])
                .align_x(Alignment::Center)
                .into()
        } else {
            widget::Row::with_children(cells)
                .padding([7, 7, 8, 7])
                .align_y(Alignment::Center)
                .into()
        };

        // This bar *is* the window's title bar, so it adopts the native COSMIC
        // header bar's colours: accent icons while the window holds focus,
        // dimmed while it doesn't, which is how a COSMIC window signals that
        // it's the active one. `transparent: true` keeps the header bar's
        // background off so the live preview still shows through (issue #565).
        let focused = self.window_is_focused();
        let mut top_bar_widget =
            widget::container(content).class(cosmic::theme::Container::HeaderBar {
                focused,
                sharp_corners: true,
                transparent: true,
            });
        // Width always follows the sideways/portrait swap. Height is only
        // pinned in the sideways case - portrait leaves it at the
        // container's default (sized from the row's own content/padding), same
        // as before this was factored through the shared helper.
        let (width, height) = bar_cross_lengths(sideways, TOP_BAR_HEIGHT);
        top_bar_widget = top_bar_widget.width(width);
        if sideways {
            top_bar_widget = top_bar_widget.height(height);
        }

        // Make the top bar draggable for window movement
        widget::mouse_area(top_bar_widget)
            .on_drag(Message::WindowDrag)
            .on_double_press(Message::WindowToggleMaximize)
            .into()
    }

    /// Build the format button (resolution/FPS display)
    fn build_format_button(&self) -> Element<'_, Message> {
        let spacing = cosmic::theme::spacing();
        let is_disabled = self.transition_state.ui_disabled;

        // Format label with superscript-style RES and FPS
        let (res_label, fps_label) = if let Some(fmt) = &self.active_format {
            let res = if fmt.width >= resolution_thresholds::THRESHOLD_4K {
                fl!("indicator-4k")
            } else if fmt.width >= resolution_thresholds::THRESHOLD_HD {
                fl!("indicator-hd")
            } else if fmt.width >= resolution_thresholds::THRESHOLD_720P {
                fl!("indicator-720p")
            } else {
                fl!("indicator-sd")
            };

            let fps = if let Some(fps) = fmt.framerate {
                fps.to_string()
            } else {
                ui::DEFAULT_FPS_DISPLAY.to_string()
            };

            (res, fps)
        } else {
            (fl!("indicator-hd"), ui::DEFAULT_FPS_DISPLAY.to_string())
        };

        // Create button with resolution^RES framerate^FPS layout
        let res_superscript =
            widget::container(widget::text(fl!("indicator-res")).size(ui::SUPERSCRIPT_TEXT_SIZE))
                .padding(ui::SUPERSCRIPT_PADDING);
        let fps_superscript =
            widget::container(widget::text(fl!("indicator-fps")).size(ui::SUPERSCRIPT_TEXT_SIZE))
                .padding(ui::SUPERSCRIPT_PADDING);

        let button_content = widget::Row::new()
            .push(widget::text(res_label).size(ui::RES_LABEL_TEXT_SIZE))
            .push(res_superscript)
            .push(widget::space::horizontal().width(spacing.space_xxs))
            .push(widget::text(fps_label).size(ui::RES_LABEL_TEXT_SIZE))
            .push(fps_superscript)
            .spacing(ui::RES_LABEL_SPACING)
            .align_y(Alignment::Center);

        let button = if is_disabled {
            widget::button::custom(button_content).class(cosmic::theme::Button::Text)
        } else {
            widget::button::custom(button_content)
                .on_press(Message::ToggleFormatPicker)
                .class(cosmic::theme::Button::Text)
        };

        // Wrap in container with themed semi-transparent background for visibility on camera preview
        widget::container(button)
            .style(move |theme| {
                let mut style = OVERLAY_CONTAINER.container_style(theme);
                if is_disabled {
                    style.text_color = Some(Color::from_rgba(1.0, 1.0, 1.0, 0.3));
                }
                style
            })
            .into()
    }

    /// Build file source resolution label (non-clickable)
    ///
    /// Shows the resolution of the selected file source (image or video).
    /// Displayed instead of the format picker when a file source is selected.
    fn build_file_source_resolution_label(&self) -> Element<'_, Message> {
        // Get resolution from current_frame (which contains the file preview)
        let (width, height) = if let Some(ref frame) = self.current_frame {
            (frame.width, frame.height)
        } else {
            (0, 0)
        };

        // Show actual resolution (e.g., "1280×720")
        let dimensions = if width > 0 && height > 0 {
            format!("{}×{}", width, height)
        } else {
            "---".to_string()
        };

        let label_content = widget::Row::new()
            .push(
                widget::text(dimensions)
                    .size(ui::RES_LABEL_TEXT_SIZE)
                    .class(cosmic::theme::style::Text::Accent),
            )
            .align_y(Alignment::Center);

        // Non-clickable container with same styling as format button
        widget::container(label_content).padding([4, 8]).into()
    }

    /// Build zoom level button for display above capture button
    ///
    /// Shows current zoom level (1x, 1.3x, 2x, etc.) in Photo mode.
    /// Click to reset zoom to 1.0.
    /// Rendered zoom level, e.g. "1x", "1.3x", "10x". Shared by the portrait
    /// zoom-label button and the sideways rotated chip widget.
    fn zoom_label_text(&self) -> String {
        if self.zoom_level >= 10.0 {
            "10x".to_string()
        } else if (self.zoom_level - self.zoom_level.round()).abs() < 0.05 {
            format!("{}x", self.zoom_level.round() as u32)
        } else {
            format!("{:.1}x", self.zoom_level)
        }
    }

    /// Whether the preview is zoomed past 1x (drives the zoom chip's accent
    /// fill).
    fn is_zoomed(&self) -> bool {
        (self.zoom_level - 1.0).abs() > 0.01
    }

    fn build_zoom_label(&self) -> Element<'_, Message> {
        let zoom_text = self.zoom_label_text();
        let is_zoomed = self.is_zoomed();

        // Suggested (accent fill) when zoomed; otherwise a frosted Text button
        // so the resting background matches the top/bottom bars.
        let button = widget::button::text(zoom_text)
            .on_press(Message::ResetZoom)
            .class(if is_zoomed {
                cosmic::theme::Button::Suggested
            } else {
                overlay_chip_button_class()
            });
        if is_zoomed {
            button.into()
        } else {
            self.frosted_panel(button.into(), OVERLAY_CONTAINER)
        }
    }

    /// Build the fit/zoom controls with their normal button styling. Portrait
    /// places them side by side; side layouts stack the same upright buttons.
    fn build_fit_zoom_group(&self, axis: FitZoomAxis) -> Element<'_, Message> {
        let spacing = cosmic::theme::spacing();
        let fit_icon_name = if self.preview_fit_to_view {
            "view-fullscreen-symbolic"
        } else {
            "view-restore-symbolic"
        };
        let fit_button_inner = widget::button::custom(
            widget::Row::new()
                .push(
                    widget::icon::from_name(fit_icon_name)
                        .symbolic(true)
                        .size(16),
                )
                .padding([0, spacing.space_s])
                .height(Length::Fixed(spacing.space_l.into()))
                .align_y(Alignment::Center),
        )
        .padding(0)
        .on_press(Message::TogglePreviewFit)
        .class(if self.preview_fit_to_view {
            cosmic::theme::Button::Suggested
        } else {
            overlay_chip_button_class()
        });
        // Inactive: frosted like the top/bottom bars so the button sits on a
        // matching surface. Active: keep the Suggested (accent) fill so toggle
        // state stays visible.
        let fit_button: Element<'_, Message> = if self.preview_fit_to_view {
            fit_button_inner.into()
        } else {
            self.frosted_panel(fit_button_inner.into(), OVERLAY_CONTAINER)
        };

        match axis {
            FitZoomAxis::Horizontal => widget::Row::new()
                .push(fit_button)
                .push(widget::space::horizontal().width(Length::Fixed(8.0)))
                .push(self.build_zoom_label())
                .align_y(Alignment::Center)
                .into(),
            FitZoomAxis::Vertical => widget::Column::new()
                .push(fit_button)
                .push(widget::space::vertical().height(Length::Fixed(8.0)))
                .push(self.build_zoom_label())
                .align_x(cosmic::iced::alignment::Horizontal::Center)
                .into(),
        }
    }

    /// Distance from the window edge to the fit/zoom chips in a side strip.
    /// View mode has no capture lane, so its visible strip is only the bottom
    /// bar rather than the full portrait-band-derived strip width.
    fn sideways_chip_inset(&self, gap: f32) -> f32 {
        let strip_width = if self.mode.is_view_only() {
            crate::app::bottom_bar::BOTTOM_BAR_HEIGHT
        } else {
            crate::app::bar_layout::SIDEWAYS_STRIP_WIDTH
        };
        strip_width + gap
    }

    /// Float the fit/zoom chip row over the preview against the sideways
    /// strip's INNER edge (the side nearest the preview), vertically centred
    /// and mirrored per rotation.
    ///
    /// The strip sits on `bottom_edge` - `Edge::Right` at 270°, `Edge::Left` at
    /// 90° - so the chips hug that edge from the preview side, offset inward by
    /// the strip's width so the gap to the strip is real. Portrait never calls
    /// this: there the row is stacked inside the strip instead.
    fn float_chips_over_preview<'a>(
        &self,
        chips: Element<'a, Message>,
        bottom_edge: crate::app::bar_layout::Edge,
    ) -> Element<'a, Message> {
        use crate::app::bar_layout::Edge;
        use cosmic::iced::Padding;
        use cosmic::iced::alignment::{Horizontal, Vertical};

        // Offset inward by the strip width PLUS a gap, so the chips sit a real
        // gap clear of the strip's inner edge (matching the portrait spacing
        // between the chips and the bottom bar) instead of touching it.
        let gap = f32::from(cosmic::theme::spacing().space_xs);
        let inset = self.sideways_chip_inset(gap);
        let c = widget::container(chips)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_y(Vertical::Center);
        match bottom_edge {
            // Strip on the right (270°): chips hug its left (inner) edge.
            Edge::Right => c.align_x(Horizontal::Right).padding(Padding {
                top: 0.0,
                right: inset,
                bottom: 0.0,
                left: 0.0,
            }),
            // Strip on the left (90°): mirror - chips hug its right (inner) edge.
            Edge::Left => c.align_x(Horizontal::Left).padding(Padding {
                top: 0.0,
                right: 0.0,
                bottom: 0.0,
                left: inset,
            }),
            // Sideways is only ever Left/Right; centre as a safe fallback.
            _ => c.align_x(Horizontal::Center),
        }
        .into()
    }

    /// Build the QR code overlay layer
    ///
    /// This creates an overlay that shows detected QR codes with bounding boxes
    /// and action buttons. The overlay widget handles coordinate transformation
    /// at render time to correctly position elements over the video content.
    fn build_qr_overlay(&self) -> Element<'_, Message> {
        // Only show overlay if QR detection is enabled and we have detections
        if !self.qr_detection_enabled || self.qr_detections.is_empty() {
            return widget::Space::new()
                .width(Length::Fill)
                .height(Length::Fill)
                .into();
        }

        // Get frame dimensions
        let Some(frame) = &self.current_frame else {
            return widget::Space::new()
                .width(Length::Fill)
                .height(Length::Fill)
                .into();
        };

        let should_mirror = self.should_mirror_preview();

        // Reuse the preview's transform and four-sided content insets so the
        // boxes stay registered through rotation and fit/fill animation.
        build_qr_overlay(
            &self.qr_detections,
            frame.width,
            frame.height,
            self.cover_blend(),
            self.bar_insets(),
            self.preview_adjusted_rotation(self.current_frame_rotation, should_mirror),
            should_mirror,
        )
    }

    /// Build the tools menu overlay
    ///
    /// Shows timer, aspect ratio, exposure, filter buttons
    /// in a floating panel aligned to the top-right with large icon buttons in a 2-row grid.
    fn build_tools_menu(&self) -> Element<'_, Message> {
        let spacing = cosmic::theme::spacing();
        let is_photo_mode = self.mode == CameraMode::Photo;

        // Collect all tool buttons for the grid
        let mut buttons: Vec<Element<'_, Message>> = Vec::new();

        // Timer button (Photo mode only)
        if is_photo_mode {
            let timer_active =
                self.photo_timer_setting != crate::app::state::PhotoTimerSetting::Off;
            let timer_icon_bytes = match self.photo_timer_setting {
                crate::app::state::PhotoTimerSetting::Off => TIMER_OFF_ICON,
                crate::app::state::PhotoTimerSetting::Sec3 => TIMER_3_ICON,
                crate::app::state::PhotoTimerSetting::Sec5 => TIMER_5_ICON,
                crate::app::state::PhotoTimerSetting::Sec10 => TIMER_10_ICON,
            };
            let timer_icon = widget::icon::from_svg_bytes(timer_icon_bytes).symbolic(true);
            buttons.push(self.build_tools_grid_button(
                timer_icon,
                fl!("tools-timer"),
                Message::CyclePhotoTimer,
                timer_active,
            ));

            // Aspect ratio button (Photo mode only). Square ratios (Native,
            // 1:1) are orientation-agnostic; the others swap to their
            // portrait companion icon when the window is taller than wide
            // so the label matches the rotated preview.
            let aspect_active = self.is_aspect_ratio_changed();
            let portrait = self.screen_is_portrait();
            let aspect_icon_bytes = match self.photo_aspect_ratio {
                crate::app::state::PhotoAspectRatio::Native => ASPECT_NATIVE_ICON,
                crate::app::state::PhotoAspectRatio::Ratio1x1 => ASPECT_1_1_ICON,
                crate::app::state::PhotoAspectRatio::Ratio4x3 if portrait => ASPECT_3_4_ICON,
                crate::app::state::PhotoAspectRatio::Ratio4x3 => ASPECT_4_3_ICON,
                crate::app::state::PhotoAspectRatio::Ratio16x9 if portrait => ASPECT_9_16_ICON,
                crate::app::state::PhotoAspectRatio::Ratio16x9 => ASPECT_16_9_ICON,
                crate::app::state::PhotoAspectRatio::Ratio2x1 if portrait => ASPECT_1_2_ICON,
                crate::app::state::PhotoAspectRatio::Ratio2x1 => ASPECT_2_1_ICON,
            };
            let aspect_icon = widget::icon::from_svg_bytes(aspect_icon_bytes).symbolic(true);
            buttons.push(self.build_tools_grid_button(
                aspect_icon,
                fl!("tools-aspect"),
                Message::CyclePhotoAspectRatio,
                aspect_active,
            ));
        }

        // Exposure button
        if self.available_exposure_controls.has_any_essential() {
            let exposure_icon = widget::icon::from_svg_bytes(EXPOSURE_ICON).symbolic(true);
            buttons.push(self.build_tools_grid_button(
                exposure_icon,
                fl!("tools-exposure"),
                Message::ToggleExposurePicker,
                self.is_exposure_changed(),
            ));
        }

        // Color button (for contrast, saturation, white balance, etc.)
        if self.available_exposure_controls.has_any_image_controls()
            || self.available_exposure_controls.has_any_white_balance()
        {
            buttons.push(self.build_tools_grid_button(
                icon::from_name("applications-graphics-symbolic").symbolic(true),
                fl!("tools-color"),
                Message::ToggleColorPicker,
                self.is_color_changed(),
            ));
        }

        // Filter button (photo, video, timelapse, and virtual-camera modes)
        if self.mode == CameraMode::Photo
            || self.mode == CameraMode::Video
            || self.mode == CameraMode::Timelapse
            || self.mode == CameraMode::Virtual
        {
            let filter_active = self.selected_filter != FilterType::Standard;
            buttons.push(self.build_tools_grid_button(
                widget::icon::from_svg_bytes(FILTER_ICON).symbolic(true),
                fl!("tools-filter"),
                Message::ToggleContextPage(crate::app::state::ContextPage::Filters),
                filter_active,
            ));
        }

        // Motor/PTZ button (shows when camera has motor controls)
        if self.has_motor_controls() {
            buttons.push(self.build_tools_grid_button(
                widget::icon::from_svg_bytes(CAMERA_TILT_ICON).symbolic(true),
                fl!("tools-motor"),
                Message::ToggleMotorPicker,
                self.motor_picker_visible,
            ));
        }

        // Distribute buttons into 2 rows
        let items_per_row = buttons.len().div_ceil(2); // Ceiling division
        let mut rows: Vec<Element<'_, Message>> = Vec::new();
        let mut current_row: Vec<Element<'_, Message>> = Vec::new();

        for (i, button) in buttons.into_iter().enumerate() {
            current_row.push(button);
            if current_row.len() >= items_per_row || i == items_per_row * 2 - 1 {
                let row = widget::row::with_children(std::mem::take(&mut current_row))
                    .spacing(spacing.space_s)
                    .align_y(Alignment::Start);
                rows.push(row.into());
            }
        }
        if !current_row.is_empty() {
            let row = widget::row::with_children(current_row)
                .spacing(spacing.space_s)
                .align_y(Alignment::Start);
            rows.push(row.into());
        }

        // Build column from rows
        let column = widget::column::with_children(rows)
            .spacing(spacing.space_s)
            .padding(spacing.space_s);

        // Build panel with semi-transparent themed background
        let panel = self.frosted_panel(column.into(), PICKER_PANEL);

        // Anchor to the top bar's inner edge (below it in portrait, off the side
        // bar in landscape), tapping outside to close.
        self.anchor_bar_popup(panel, Message::CloseToolsMenu)
    }

    /// Anchor a top popup (tools menu, exposure/color pickers) `space_xs` off the
    /// top bar's inner edge at the bar's trailing (top) end, in every
    /// orientation, and close it when the surrounding area is tapped.
    ///
    /// Portrait hangs the panel below the full-width top bar, top-right (unchanged
    /// from when this was a hardcoded `Row` + spring). Held sideways, the top bar
    /// is a vertical strip on one edge, so the panel hangs `TOP_BAR_HEIGHT +
    /// space_xs` in from that strip near the top, clear of the control strip /
    /// capture button on the opposite edge. The exact insets and which side the
    /// panel hugs come from [`bar_anchored_popup_padding`].
    pub(crate) fn anchor_bar_popup<'a>(
        &self,
        panel: Element<'a, Message>,
        close_msg: Message,
    ) -> Element<'a, Message> {
        // Trailing (right-end) popups: tools menu, exposure/color/motor pickers.
        self.anchor_bar_popup_at(panel, close_msg, false)
    }

    /// Like [`anchor_bar_popup`], but for a popup whose button sits at the top
    /// bar's *leading* (left) end - currently just the format picker. Portrait
    /// hangs the panel below the bar's left end; the sideways side-strip anchor
    /// is identical to the trailing case (both hang off the same strip edge).
    pub(crate) fn anchor_bar_popup_leading<'a>(
        &self,
        panel: Element<'a, Message>,
        close_msg: Message,
    ) -> Element<'a, Message> {
        self.anchor_bar_popup_at(panel, close_msg, true)
    }

    fn anchor_bar_popup_at<'a>(
        &self,
        panel: Element<'a, Message>,
        close_msg: Message,
        leading: bool,
    ) -> Element<'a, Message> {
        let spacing = cosmic::theme::spacing();
        let top_bar = self.controls_bar_layout().top_bar;
        let (padding, align_right) = crate::app::bar_layout::bar_anchored_popup_padding(
            top_bar,
            TOP_BAR_HEIGHT as u16,
            spacing.space_xs,
            leading,
        );

        let spring = widget::Space::new()
            .width(Length::Fill)
            .height(Length::Shrink);
        // `align_right` picks the leading vs trailing spring: a leading Fill
        // pushes the panel to the right edge, a trailing Fill to the left.
        let positioned = if align_right {
            widget::Row::new().push(spring).push(panel)
        } else {
            widget::Row::new().push(panel).push(spring)
        }
        .padding(padding);

        widget::mouse_area(
            widget::container(positioned)
                .width(Length::Fill)
                .height(Length::Fill),
        )
        .on_press(close_msg)
        .into()
    }

    /// Build a grid button with large icon and text label below (outside the button)
    fn build_tools_grid_button<'a>(
        &self,
        icon_handle: impl Into<widget::icon::Handle>,
        label: String,
        message: Message,
        is_active: bool,
    ) -> Element<'a, Message> {
        self.build_tools_grid_button_with_enabled(icon_handle, label, message, is_active, true)
    }

    /// Build a grid button with large icon and text label below, with optional enabled state
    fn build_tools_grid_button_with_enabled<'a>(
        &self,
        icon_handle: impl Into<widget::icon::Handle>,
        label: String,
        message: Message,
        is_active: bool,
        enabled: bool,
    ) -> Element<'a, Message> {
        // Icon button with appropriate styling
        let mut button = widget::button::custom(widget::icon(icon_handle.into()).size(32))
            .class(if is_active {
                cosmic::theme::Button::Suggested
            } else {
                cosmic::theme::Button::Text
            })
            .padding(12);

        // Only add on_press handler if enabled
        if enabled {
            button = button.on_press(message);
        }

        // Wrap inactive buttons in a container with visible background
        let button_element: Element<'_, Message> = if is_active {
            button.into()
        } else {
            widget::container(button)
                .style(OVERLAY_CONTAINER.style())
                .into()
        };

        // Button with label below
        widget::Column::new()
            .push(button_element)
            .push(widget::text(label).size(11))
            .spacing(4)
            .align_x(Alignment::Center)
            .into()
    }

    /// Check if any tool settings are non-default (for highlighting tools button).
    /// Photo-only settings (timer, aspect ratio) are only counted while the
    /// app is in Photo mode — they don't take effect elsewhere, so they
    /// shouldn't drive the highlight in Video / Timelapse / Virtual.
    fn has_non_default_tool_settings(&self) -> bool {
        let in_photo = self.mode == CameraMode::Photo;
        let timer_active =
            in_photo && self.photo_timer_setting != crate::app::state::PhotoTimerSetting::Off;
        let aspect_active = in_photo && self.is_aspect_ratio_changed();
        let exposure_active = self.is_exposure_changed();
        let color_active = self.is_color_changed();
        let filter_active = self.selected_filter != FilterType::Standard;

        timer_active || aspect_active || exposure_active || color_active || filter_active
    }

    /// Check if aspect ratio is cropped (not using native ratio)
    fn is_aspect_ratio_changed(&self) -> bool {
        self.photo_aspect_ratio != crate::app::state::PhotoAspectRatio::Native
    }

    /// Check if exposure settings differ from defaults
    fn is_exposure_changed(&self) -> bool {
        let controls = &self.available_exposure_controls;
        self.exposure_settings
            .as_ref()
            .map(|s| {
                let mode_changed = controls.has_exposure_auto
                    && s.mode != crate::app::exposure_picker::ExposureMode::AperturePriority;
                let ev_changed = controls.exposure_bias.available
                    && s.exposure_compensation != controls.exposure_bias.default;
                let backlight_changed = controls.backlight_compensation.available
                    && s.backlight_compensation
                        .map(|v| v != controls.backlight_compensation.default)
                        .unwrap_or(false);
                mode_changed || ev_changed || backlight_changed
            })
            .unwrap_or(false)
    }

    /// Check if color settings differ from defaults
    fn is_color_changed(&self) -> bool {
        let controls = &self.available_exposure_controls;
        self.color_settings
            .as_ref()
            .map(|s| {
                let image_changed = (controls.contrast.available
                    && s.contrast
                        .map(|v| v != controls.contrast.default)
                        .unwrap_or(false))
                    || (controls.saturation.available
                        && s.saturation
                            .map(|v| v != controls.saturation.default)
                            .unwrap_or(false))
                    || (controls.sharpness.available
                        && s.sharpness
                            .map(|v| v != controls.sharpness.default)
                            .unwrap_or(false))
                    || (controls.hue.available
                        && s.hue.map(|v| v != controls.hue.default).unwrap_or(false));
                let wb_auto_off = controls.has_white_balance_auto
                    && s.white_balance_auto.map(|v| !v).unwrap_or(false);
                image_changed || wb_auto_off
            })
            .unwrap_or(false)
    }

    /// Build the privacy cover warning overlay
    ///
    /// Shows a centered warning when the camera's privacy cover is closed.
    fn build_privacy_warning(&self) -> Element<'_, Message> {
        if !self.privacy_cover_closed {
            return widget::Space::new()
                .width(Length::Fill)
                .height(Length::Fill)
                .into();
        }

        build_overlay_popup(
            self,
            widget::text("\u{26A0}").size(48).into(),
            &fl!("privacy-cover-closed"),
            &fl!("privacy-cover-hint"),
            None,
        )
    }

    /// Build the burst mode progress overlay
    ///
    /// Shows status text, frame count, and progress bar during burst mode capture/processing.
    fn build_burst_mode_overlay(&self) -> Element<'_, Message> {
        let (status_text, detail_text) = match self.burst_mode.stage {
            BurstModeStage::Capturing => (
                fl!("burst-mode-hold-steady"),
                fl!(
                    "burst-mode-frames",
                    captured = self.burst_mode.frames_captured(),
                    total = self.burst_mode.target_frame_count
                ),
            ),
            BurstModeStage::Processing => (fl!("burst-mode-processing"), String::new()),
            _ => (String::new(), String::new()),
        };

        // Progress percentage
        let progress_percent = (self.burst_mode.progress() * 100.0) as u32;

        // Build progress bar (simple filled bar)
        let progress_width = BURST_MODE_PROGRESS_BAR_WIDTH;
        let progress_height = BURST_MODE_PROGRESS_BAR_HEIGHT;
        let filled_width = progress_width * self.burst_mode.progress();

        let progress_bar = widget::container(
            widget::Row::new()
                .push(
                    widget::container(
                        widget::Space::new()
                            .width(Length::Fixed(filled_width))
                            .height(Length::Fixed(progress_height)),
                    )
                    .style(|theme: &cosmic::Theme| {
                        let accent = theme.cosmic().accent_color();
                        widget::container::Style {
                            background: Some(Background::Color(Color::from_rgb(
                                accent.red,
                                accent.green,
                                accent.blue,
                            ))),
                            ..Default::default()
                        }
                    }),
                )
                .push(
                    widget::container(
                        widget::Space::new()
                            .width(Length::Fixed(progress_width - filled_width))
                            .height(Length::Fixed(progress_height)),
                    )
                    .style(|_theme| widget::container::Style {
                        background: Some(Background::Color(Color::from_rgba(1.0, 1.0, 1.0, 0.3))),
                        ..Default::default()
                    }),
                ),
        )
        .style(|theme: &cosmic::Theme| widget::container::Style {
            border: cosmic::iced::Border {
                radius: theme.cosmic().corner_radii.radius_xs.into(),
                ..Default::default()
            },
            ..Default::default()
        });

        // Build the overlay content
        let overlay_content = widget::Column::new()
            .push(
                widget::text(status_text)
                    .size(32)
                    .font(cosmic::font::bold()),
            )
            .push(
                widget::Space::new()
                    .width(Length::Shrink)
                    .height(Length::Fixed(8.0)),
            )
            .push(widget::text(detail_text).size(18))
            .push(
                widget::Space::new()
                    .width(Length::Shrink)
                    .height(Length::Fixed(16.0)),
            )
            .push(progress_bar)
            .push(
                widget::Space::new()
                    .width(Length::Shrink)
                    .height(Length::Fixed(8.0)),
            )
            .push(widget::text(format!("{}%", progress_percent)).size(14))
            .align_x(Alignment::Center);

        // Semi-transparent background panel
        let overlay_panel = self.frosted_panel(
            widget::container(overlay_content).padding(24).into(),
            POPUP_PANEL,
        );

        widget::container(overlay_panel)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(cosmic::iced::alignment::Horizontal::Center)
            .align_y(cosmic::iced::alignment::Vertical::Center)
            .into()
    }

    /// Build the flash permission error popup dialog
    ///
    /// Shows a centered modal with warning icon, error message, and OK button
    /// when flash hardware was detected but cannot be controlled.
    fn build_flash_error_popup(&self) -> Element<'_, Message> {
        let error_msg = self
            .flash
            .error_popup
            .as_deref()
            .unwrap_or("Flash permission error");

        build_overlay_popup(
            self,
            widget::text("\u{26A0}").size(48).into(),
            "Flash Permission Error",
            error_msg,
            Some(
                widget::button::suggested("OK")
                    .on_press(Message::DismissFlashError)
                    .into(),
            ),
        )
    }

    /// Build the timer countdown overlay
    ///
    /// Shows large countdown number with fade effect during photo timer countdown.
    fn build_timer_overlay(&self, remaining: u8) -> Element<'_, Message> {
        // Calculate fade opacity based on elapsed time since tick start
        // Opacity starts at 1.0 and fades to 0.0 over the second
        let opacity = if let Some(tick_start) = self.photo_timer_tick_start {
            let elapsed_ms = tick_start.elapsed().as_millis() as f32;
            // Fade out over 900ms (leave 100ms fully transparent before next number)
            (1.0 - (elapsed_ms / 900.0)).max(0.0)
        } else {
            1.0
        };

        // Large countdown number with fade effect
        let countdown_text = widget::container(
            widget::text(remaining.to_string())
                .size(400) // Very large to fill preview
                .font(cosmic::font::bold()),
        )
        .style(move |_theme| widget::container::Style {
            text_color: Some(Color::from_rgba(1.0, 1.0, 1.0, opacity)),
            ..Default::default()
        });

        widget::container(countdown_text)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(cosmic::iced::alignment::Horizontal::Center)
            .align_y(cosmic::iced::alignment::Vertical::Center)
            .into()
    }
}

#[cfg(test)]
mod bar_insets_tests {
    use super::*;

    #[test]
    fn compact_side_picker_height_stays_inside_the_window() {
        let spacing = cosmic::theme::spacing();
        assert_eq!(
            picker_panel_height(true, 360.0),
            Length::Fixed(360.0 - f32::from(spacing.space_xs) * 2.0)
        );
        assert_eq!(picker_panel_height(false, 360.0), Length::Shrink);
    }

    #[test]
    fn sideways_video_progress_is_kept_with_the_side_controls() {
        assert_eq!(
            video_progress_placement(true),
            VideoProgressPlacement::AboveSideLanes
        );
        assert_eq!(
            video_progress_placement(false),
            VideoProgressPlacement::AboveCapture
        );
    }

    #[test]
    fn sideways_modal_padding_reserves_side_bars_not_portrait_bars() {
        let left = AppModel {
            config: crate::config::Config {
                controls_position: crate::config::ControlsPosition::Left,
                ..Default::default()
            },
            screen_width: 733.0,
            screen_height: 360.0,
            ..Default::default()
        };
        let padding = overlay_popup_padding(&left);

        assert_eq!(padding[0], 0.0);
        assert_eq!(padding[2], 0.0);
        assert!(padding[1] > 0.0 || padding[3] > 0.0);
    }

    #[test]
    fn sideways_fit_zoom_uses_the_standard_upright_row() {
        assert_eq!(fit_zoom_layout(true), FitZoomLayout::FloatUpright);
        assert_eq!(fit_zoom_layout(false), FitZoomLayout::Inline);
    }

    #[test]
    fn sideways_fit_zoom_stacks_buttons_vertically() {
        assert_eq!(fit_zoom_axis(true), FitZoomAxis::Vertical);
        assert_eq!(fit_zoom_axis(false), FitZoomAxis::Horizontal);
    }
    use crate::app::bar_layout::bar_layout;
    use crate::backends::display_orientation::DisplayOrientation;

    #[test]
    fn context_drawer_only_reserves_the_horizontal_title_bar() {
        assert_eq!(
            AppModel::context_drawer_top_inset(bar_layout(DisplayOrientation::Rotate0)),
            TOP_BAR_HEIGHT
        );
        assert_eq!(
            AppModel::context_drawer_top_inset(bar_layout(DisplayOrientation::Rotate180)),
            TOP_BAR_HEIGHT
        );
        assert_eq!(
            AppModel::context_drawer_top_inset(bar_layout(DisplayOrientation::Rotate90)),
            0.0
        );
        assert_eq!(
            AppModel::context_drawer_top_inset(bar_layout(DisplayOrientation::Rotate270)),
            0.0
        );
    }

    #[test]
    fn context_drawer_stays_opposite_a_side_title_bar() {
        let m = AppModel::default();
        assert!(m.context_drawer_aligns_right(bar_layout(DisplayOrientation::Rotate270)));
        assert!(!m.context_drawer_aligns_right(bar_layout(DisplayOrientation::Rotate90)));
    }

    #[test]
    fn manual_left_controls_keep_context_drawers_on_the_right() {
        let m = AppModel {
            config: crate::config::Config {
                controls_position: crate::config::ControlsPosition::Left,
                ..Default::default()
            },
            ..Default::default()
        };

        assert!(m.context_drawer_aligns_right(m.controls_bar_layout()));
    }

    #[test]
    fn view_mode_chips_collapse_the_missing_capture_lane() {
        let gap = 12.0;
        let photo = AppModel {
            mode: CameraMode::Photo,
            ..Default::default()
        };
        let view = AppModel {
            mode: CameraMode::View,
            ..Default::default()
        };

        assert_eq!(
            photo.sideways_chip_inset(gap),
            crate::app::bar_layout::SIDEWAYS_STRIP_WIDTH + gap
        );
        assert_eq!(
            view.sideways_chip_inset(gap),
            crate::app::bottom_bar::BOTTOM_BAR_HEIGHT + gap
        );
    }

    /// Deliberately different top/bottom values so a swapped mapping fails
    /// loudly instead of passing by coincidence (e.g. if both were 100.0).
    const TOP: f32 = 47.0;
    const BOTTOM: f32 = 174.0;

    #[test]
    fn rotate0_keeps_bars_on_top_and_bottom() {
        let layout = bar_layout(DisplayOrientation::Rotate0);
        let insets = AppModel::map_bar_insets(layout, TOP, BOTTOM);

        assert_eq!(insets.top, TOP);
        assert_eq!(insets.bottom, BOTTOM);
        assert_eq!(insets.left, 0.0);
        assert_eq!(insets.right, 0.0);
    }

    #[test]
    fn rotate180_is_treated_as_portrait() {
        let layout = bar_layout(DisplayOrientation::Rotate180);
        let insets = AppModel::map_bar_insets(layout, TOP, BOTTOM);

        assert_eq!(insets.top, TOP);
        assert_eq!(insets.bottom, BOTTOM);
        assert_eq!(insets.left, 0.0);
        assert_eq!(insets.right, 0.0);
    }

    /// `Rotate270` keeps the top bar on the LEFT edge, bottom bar on the RIGHT.
    #[test]
    fn rotate270_puts_the_top_bar_on_the_left() {
        let layout = bar_layout(DisplayOrientation::Rotate270);
        let insets = AppModel::map_bar_insets(layout, TOP, BOTTOM);

        assert_eq!(insets.left, TOP);
        assert_eq!(insets.right, BOTTOM);
        assert_eq!(insets.top, 0.0);
        assert_eq!(insets.bottom, 0.0);
    }

    /// `Rotate90` is a mirror of `Rotate270`, not identical to it: the top bar
    /// lands on the RIGHT edge instead of the left. This is a deliberate
    /// ergonomic override (see `bar_layout`'s doc) - the compositor renders 90
    /// and 270 byte-identically, but the bars mirror anyway so controls track
    /// which physical edge is now the device's "bottom".
    #[test]
    fn rotate90_mirrors_rotate270_top_bar_on_the_right() {
        let insets =
            AppModel::map_bar_insets(bar_layout(DisplayOrientation::Rotate90), TOP, BOTTOM);

        assert_eq!(insets.left, BOTTOM);
        assert_eq!(insets.right, TOP);
        assert_eq!(insets.top, 0.0);
        assert_eq!(insets.bottom, 0.0);

        assert_ne!(
            insets,
            AppModel::map_bar_insets(bar_layout(DisplayOrientation::Rotate270), TOP, BOTTOM)
        );
    }

    /// End-to-end through `AppModel::bar_insets`, not just the pure helper -
    /// pins that the display-orientation field actually reaches the mapping.
    #[test]
    fn bar_insets_reads_the_models_display_orientation() {
        let m = AppModel {
            display_orientation: DisplayOrientation::Rotate90,
            ..AppModel::default()
        };

        let insets = m.bar_insets();

        // Rotate90's top bar is on the RIGHT edge, so the top bar's extent
        // (top_ui_height) lands in insets.right, not insets.left.
        assert_eq!(insets.left, m.bottom_ui_height());
        assert_eq!(insets.right, m.top_ui_height());
        assert_eq!(insets.top, 0.0);
        assert_eq!(insets.bottom, 0.0);
    }

    #[test]
    fn manual_controls_position_overrides_bar_edges() {
        let m = AppModel {
            display_orientation: DisplayOrientation::Rotate0,
            config: crate::config::Config {
                controls_position: crate::config::ControlsPosition::Left,
                ..Default::default()
            },
            ..Default::default()
        };

        let layout = m.controls_bar_layout();
        let insets = m.bar_insets();

        assert_eq!(layout.bottom_bar, crate::app::bar_layout::Edge::Left);
        assert_eq!(layout.top_bar, crate::app::bar_layout::Edge::Right);
        assert_eq!(insets.left, m.bottom_ui_height());
        assert_eq!(insets.right, m.top_ui_height());
    }
}
