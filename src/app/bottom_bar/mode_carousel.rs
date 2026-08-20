// SPDX-License-Identifier: GPL-3.0-only

//! ModeCarousel — horizontally scrollable mode selector with snap.
//!
//! A custom iced Widget that handles touch and mouse events directly.
//! Labels slide continuously with the finger during drag, then snap
//! to the nearest mode center on release with an ease-out animation.

use crate::app::bar_layout::Quarter;
use crate::app::state::{CameraMode, Message};
use crate::fl;
use cosmic::iced::advanced::widget::Tree;
use cosmic::iced::advanced::widget::tree;
use cosmic::iced::advanced::{Clipboard, Shell, Widget, layout, renderer};
use cosmic::iced::{
    Background, Border, Color, Element, Event, Length, Pixels, Point, Rectangle, Size, Vector,
};
use cosmic::iced::{alignment, mouse, touch};
use cosmic::widget::canvas::{Frame, Text as CanvasText};
use cosmic::{Renderer, Theme};
use iced_core::text::{self as iced_text, Text as IcedText};
use std::time::Instant;
/// Font size for mode labels
const FONT_SIZE: f32 = 16.0;
/// Widget height (matches gallery button height)
const CAROUSEL_HEIGHT: f32 = 40.0;
/// Snap animation duration in milliseconds
const SNAP_DURATION_MS: f32 = 600.0;
/// Animation duration for expansion stages (ms)
const EXPAND_TRANSITION_MS: f32 = 300.0;
/// Hold threshold before triggering stage 2 expansion (ms)
const HOLD_TO_FULL_EXPAND_MS: f32 = 600.0;
/// Delay after cursor leaves before collapsing (ms)
const HOVER_LEAVE_COLLAPSE_DELAY_MS: f32 = 600.0;
/// Delay after stage 2 collapse before stage 1 collapses (ms)
const STAGE1_COLLAPSE_AFTER_STAGE2_MS: f32 = 600.0;
/// Minimum drag distance to switch modes (pixels)
const SWITCH_THRESHOLD: f32 = 40.0;
/// Opacity for inactive labels
const INACTIVE_OPACITY: f32 = 0.7;
/// Opacity for disabled labels
const DISABLED_OPACITY: f32 = 0.25;
/// Gap between labels in the strip
const LABEL_GAP: f32 = 2.0;
/// Width of gallery/switch buttons (used for slide calculations)
const BUTTON_WIDTH: f32 = 40.0;
/// Side-axis length below which the compact phone layout is used.
const COMPACT_SIDE_AXIS_MAX: f32 = 480.0;
/// Maximum movement to still count as a tap (pixels)
const TAP_THRESHOLD: f32 = 5.0;

/// Internal state persisted across frames via the iced widget tree.
#[derive(Debug)]
struct CarouselState {
    /// Touch finger currently driving the drag
    active_finger: Option<touch::Finger>,
    /// Cursor/finger x at drag start (absolute coordinates)
    drag_start_x: Option<f32>,
    /// Current drag offset in pixels (positive = dragged right)
    drag_offset: f32,
    /// Drag offset at the moment the drag started (to support animation interruption)
    drag_base_offset: f32,
    /// Press position relative to widget bounds (for tap detection)
    press_widget_x: f32,

    /// Snap animation: offset value when animation started
    snap_from: f32,
    /// Snap animation: target offset to animate towards
    snap_target: f32,
    /// Snap animation: timestamp when snap started
    snap_start: Option<Instant>,

    /// Last selected mode index — used to detect external mode changes and reset offset
    last_selected_idx: Option<usize>,

    /// Set when we initiate a mode change so layout() can distinguish
    /// our own anchor shift from an external mode change.
    pending_mode: Option<CameraMode>,

    /// Stage 1 expansion: 0.0 = collapsed, 1.0 = fade zones fully visible
    expand_t: f32,
    /// Stage 1 animation start time
    expand_start: Option<Instant>,
    /// Whether expanding (true) or collapsing (false)
    expanding: bool,

    /// Stage 2 full expansion: 0.0 = normal, 1.0 = all labels visible
    full_expand_t: f32,
    /// Stage 2 animation start time
    full_expand_start: Option<Instant>,
    /// Whether fully expanding (true) or collapsing (false)
    full_expanding: bool,
    /// Timestamp when press started (for hold detection)
    press_time: Option<Instant>,
    /// Timestamp when stage 2 collapse completed (for delayed stage 1 collapse)
    stage2_collapse_done: Option<Instant>,
    /// Whether the mouse cursor is hovering over the wider carousel area (includes buttons)
    hovered: bool,
    /// Whether the mouse cursor is specifically over the carousel visual bounds
    hovered_carousel: bool,
    /// Timestamp when cursor left the hover area (delayed collapse)
    hover_leave_time: Option<Instant>,

    /// Animated pill center position (in strip coordinates, spring toward target)
    pill_center: f32,
    /// Animated pill width (spring toward target label width)
    pill_width: f32,
    /// Spring velocity for pill center
    pill_vel_center: f32,
    /// Spring velocity for pill width
    pill_vel_width: f32,
    /// Whether pill animation has been initialized
    pill_initialized: bool,
    /// Timestamp of last animation tick (for frame-rate independent spring)
    last_tick: Option<Instant>,

    /// Cached label widths (including padding)
    label_widths: Vec<f32>,
    /// Cached center-x positions for each label in the virtual strip
    label_centers: Vec<f32>,
    /// Cached label strings (avoids fl!() allocations on every frame)
    label_strings: Vec<String>,
    /// Number of modes when labels were last cached (dirty flag)
    cached_mode_count: usize,
    /// Nearest mode index during drag (for haptic feedback on boundary crossing)
    drag_nearest_idx: Option<usize>,
}

impl Default for CarouselState {
    fn default() -> Self {
        Self {
            active_finger: None,
            drag_start_x: None,
            drag_offset: 0.0,
            drag_base_offset: 0.0,
            press_widget_x: 0.0,
            last_selected_idx: None,
            pending_mode: None,
            expand_t: 0.0,
            expand_start: None,
            expanding: false,
            full_expand_t: 0.0,
            full_expand_start: None,
            full_expanding: false,
            press_time: None,
            stage2_collapse_done: None,
            hovered: false,
            hovered_carousel: false,
            hover_leave_time: None,
            pill_center: 0.0,
            pill_width: 0.0,
            pill_vel_center: 0.0,
            pill_vel_width: 0.0,
            pill_initialized: false,
            last_tick: None,
            snap_from: 0.0,
            snap_target: 0.0,
            snap_start: None,
            label_widths: Vec::new(),
            label_centers: Vec::new(),
            label_strings: Vec::new(),
            cached_mode_count: 0,
            drag_nearest_idx: None,
        }
    }
}

/// Horizontally scrollable mode carousel with snap-to-center.
pub struct ModeCarousel<'a> {
    modes: Vec<CameraMode>,
    selected: CameraMode,
    on_select: Box<dyn Fn(CameraMode) -> Message + 'a>,
    disabled: bool,
    /// Shared button inward slide (written here every frame, read by SlideH draw)
    slide_shared: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// When true and the carousel is fully collapsed, the rounded background
    /// shrinks to the active pill's width so the visible chip is exactly the
    /// pill — no extra padding around it. Used in View mode where the
    /// carousel stands alone without sibling buttons.
    fit_to_pill_when_collapsed: bool,
    /// Quarter turn for the whole block when the device is held sideways.
    /// `Quarter::None` in portrait - every code path is byte-identical to the
    /// unrotated carousel then.
    quarter: Quarter,
}

impl<'a> ModeCarousel<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        modes: Vec<CameraMode>,
        selected: CameraMode,
        on_select: impl Fn(CameraMode) -> Message + 'a,
        disabled: bool,
        slide_shared: std::sync::Arc<std::sync::atomic::AtomicU32>,
        fit_to_pill_when_collapsed: bool,
        quarter: Quarter,
    ) -> Self {
        Self {
            modes,
            selected,
            on_select: Box::new(on_select),
            disabled,
            slide_shared,
            fit_to_pill_when_collapsed,
            quarter,
        }
    }

    fn selected_index(&self) -> usize {
        self.modes
            .iter()
            .position(|m| *m == self.selected)
            .unwrap_or(0)
    }
}

impl<'a> Widget<Message, Theme, Renderer> for ModeCarousel<'a> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<CarouselState>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(CarouselState::default())
    }

    fn size(&self) -> Size<Length> {
        if self.quarter == Quarter::None {
            Size::new(Length::Shrink, Length::Fixed(CAROUSEL_HEIGHT))
        } else {
            // Sideways: the block's short axis (its "height" in portrait) is
            // now the strip's width; its long axis runs down the strip.
            Size::new(Length::Fixed(CAROUSEL_HEIGHT), Length::Shrink)
        }
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        _renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let state = tree.state.downcast_mut::<CarouselState>();

        // Always recalculate label metrics (cheap operation)
        recalculate_label_metrics(&self.modes, state);

        // Reset drag offset when selected mode changes externally
        // Skip reset if we initiated the change (pending_mode is set during our snap)
        let selected_idx = self.selected_index();
        if state.last_selected_idx != Some(selected_idx) {
            if state.pending_mode.is_some() {
                // We initiated this mode change — compensate for anchor shift.
                // Keep the animation clock running to preserve velocity continuity.
                let old_idx = state.last_selected_idx.unwrap_or(selected_idx);
                let old_anchor = anchor_center(state, &self.modes, old_idx);
                let new_anchor = anchor_center(state, &self.modes, selected_idx);
                let current = current_visual_offset(state);
                let shifted = current + new_anchor - old_anchor;

                // Compute new_from so interpolation equals shifted at current t:
                // new_from + (0 - new_from) * eased(t) = shifted
                // new_from * (1 - eased(t)) = shifted
                // new_from = shifted / (1 - eased(t))
                if let Some(snap_start) = state.snap_start {
                    let elapsed = snap_start.elapsed().as_secs_f32() * 1000.0;
                    let t = (elapsed / SNAP_DURATION_MS).min(1.0);
                    let eased = ease_out_cubic(t);
                    let denom = 1.0 - eased;
                    if denom > 0.05 {
                        // Enough animation time left — adjust without restarting
                        state.snap_from = shifted / denom;
                        state.snap_target = 0.0;
                    } else {
                        // Almost done — just restart a short animation
                        state.snap_from = shifted;
                        state.snap_target = 0.0;
                        state.snap_start = Some(Instant::now());
                    }
                } else {
                    state.snap_from = shifted;
                    state.snap_target = 0.0;
                    state.snap_start = Some(Instant::now());
                }
            } else if state.drag_start_x.is_some() || state.active_finger.is_some() {
                // User is actively pressing/dragging — do an anchor shift
                // instead of a hard reset to preserve drag continuity.
                let old_idx = state.last_selected_idx.unwrap_or(selected_idx);
                let old_anchor = anchor_center(state, &self.modes, old_idx);
                let new_anchor = anchor_center(state, &self.modes, selected_idx);
                let shift = new_anchor - old_anchor;
                state.drag_offset += shift;
                state.drag_base_offset += shift;
            } else {
                state.drag_offset = 0.0;
                state.drag_base_offset = 0.0;
                state.snap_start = None;
                state.snap_from = 0.0;
                state.snap_target = 0.0;
            }
            state.last_selected_idx = Some(selected_idx);
        }

        // Layout width fits 2 labels tightly, consistent regardless of selection.
        // With 2 modes this fits all; with 3+ the expand animation reveals the rest.
        // Use the 2 smallest labels so the size stays stable across mode switches.
        // The long axis of the block. In portrait it's the node width; when
        // rotated it runs down the strip and becomes the node height.
        let long = content_long(&state.label_widths);
        layout::Node::new(carousel_node_size(self.quarter, long, limits))
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        _style: &renderer::Style,
        layout: layout::Layout<'_>,
        _cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        // The layout node is the on-screen (strip) rectangle. All the carousel
        // geometry below is computed in *portrait* space - a horizontal block
        // with the same origin but width/height un-swapped - and the emitted
        // quads/text are turned back into strip space at the end. In portrait
        // (`Quarter::None`) `portrait_*` are the identity, so this is
        // byte-identical to the unrotated carousel.
        let strip_bounds = layout.bounds();
        let quarter = self.quarter;
        let bounds = portrait_bounds(quarter, strip_bounds);
        let viewport_p = portrait_viewport(quarter, *viewport);
        let state = tree.state.downcast_ref::<CarouselState>();

        if state.label_centers.is_empty() {
            return;
        }

        let selected_idx = self.selected_index();
        let viewport_center = bounds.width / 2.0;

        // Calculate visual offset: base (centers selected mode) + drag/animation
        let anchor = anchor_center(state, &self.modes, selected_idx);
        let base_offset = viewport_center - anchor;
        let vis_off = current_visual_offset(state);
        let visual_offset = base_offset + vis_off;

        // Determine which mode is closest to viewport center (for live pill highlight).
        let highlighted_idx = compute_highlighted_idx(state, &self.modes, selected_idx, vis_off);

        // Theme values
        let cosmic = theme.cosmic();
        let accent = cosmic.accent_color();
        let accent_alpha = if self.disabled { 0.3 } else { 0.85 };
        let inactive_alpha = if self.disabled {
            DISABLED_OPACITY
        } else {
            INACTIVE_OPACITY
        };

        // Background color: window bg blended with control component tint at its theme alpha.
        // At full expansion (stage 2), lerp toward window bg so carousel merges with background.
        let win_bg = cosmic.bg_color();
        let btn_bg = cosmic.button_bg_color();
        let a = btn_bg.alpha;
        let base_bg = Color::from_rgba(
            win_bg.red * (1.0 - a) + btn_bg.red * a,
            win_bg.green * (1.0 - a) + btn_bg.green * a,
            win_bg.blue * (1.0 - a) + btn_bg.blue * a,
            1.0,
        );
        // Compute expand/full_expand from timestamps for smooth rendering
        // on every draw frame, even when update() hasn't ticked animations.
        let expand = if let Some(start) = state.expand_start {
            let t = (start.elapsed().as_secs_f32() * 1000.0 / EXPAND_TRANSITION_MS).min(1.0);
            let eased = ease_out_cubic(t);
            if state.expanding { eased } else { 1.0 - eased }
        } else {
            state.expand_t
        };
        let full_expand = if let Some(start) = state.full_expand_start {
            let t = (start.elapsed().as_secs_f32() * 1000.0 / EXPAND_TRANSITION_MS).min(1.0);
            let eased = ease_out_cubic(t);
            if state.full_expanding {
                eased
            } else {
                1.0 - eased
            }
        } else {
            state.full_expand_t
        };
        let bg_color = Color::from_rgba(
            base_bg.r * (1.0 - full_expand) + win_bg.red * full_expand,
            base_bg.g * (1.0 - full_expand) + win_bg.green * full_expand,
            base_bg.b * (1.0 - full_expand) + win_bg.blue * full_expand,
            1.0,
        );
        let fade_w_max = bounds.width / 4.0;

        // Stage 1 fade zone width
        let fade_w = fade_w_max * expand;

        // Resolve expansion against the carousel's physical long axis. For a
        // side strip that is screen Y, not the synthetic portrait-space X.
        let axis_bounds = button_axis_bounds(quarter, strip_bounds, bounds);
        let extend = carousel_extension_for_progress(
            axis_bounds,
            viewport_p,
            state,
            expand,
            full_expand,
            quarter != Quarter::None,
        );
        // Button slide is computed in update() via compute_button_slide()
        // to avoid side effects in draw().

        // Expand rendering area proportionally to expansion animations.
        // When `fit_to_pill_when_collapsed` is set (View mode), the
        // background shrinks toward the active pill as the carousel
        // collapses, so the visible chip matches the pill exactly when
        // settled and grows to the full layout bounds while expanded.
        // Use the same `visual_offset` the pill itself uses so the chip
        // sits exactly where the pill is drawn.
        let collapsed_bounds = if self.fit_to_pill_when_collapsed && state.pill_initialized {
            let pill_x = bounds.x + state.pill_center + visual_offset - state.pill_width / 2.0;
            Rectangle {
                x: pill_x,
                y: bounds.y,
                width: state.pill_width,
                height: bounds.height,
            }
        } else {
            bounds
        };
        let render_bounds = if extend > 0.0 {
            // Lerp from the collapsed (pill-sized) bounds to the full layout
            // bounds as `expand` (stage 1) progresses, then stretch outward
            // by `extend` so labels and fade zones get room.
            let t = expand.clamp(0.0, 1.0);
            let base_x = collapsed_bounds.x + (bounds.x - collapsed_bounds.x) * t;
            let base_w = collapsed_bounds.width + (bounds.width - collapsed_bounds.width) * t;
            Rectangle {
                x: base_x - extend,
                y: bounds.y,
                width: base_w + extend * 2.0,
                height: bounds.height,
            }
        } else {
            collapsed_bounds
        };

        // Compute corner radius once — shared by background and fade overlays.
        // render_bounds == bounds when extend == 0, so this is always correct.
        let max_radius = (render_bounds.width / 2.0).min(render_bounds.height / 2.0);
        let bg_radius: [f32; 4] = cosmic.corner_radii.radius_xl.map(|cr| cr.min(max_radius));

        use cosmic::iced::advanced::Renderer as _;
        renderer.with_layer(
            strip_rect(quarter, strip_bounds, render_bounds),
            |renderer| {
                renderer.fill_quad(
                    renderer::Quad {
                        bounds: strip_rect(quarter, strip_bounds, render_bounds),
                        border: Border {
                            color: Color::TRANSPARENT,
                            width: 0.0,
                            radius: permute_radii(quarter, bg_radius).into(),
                        },
                        shadow: Default::default(),
                        snap: true,
                    },
                    Background::Color(bg_color),
                );

                // Draw animated pill background (smoothly slides between modes)
                if state.pill_initialized {
                    let pill_pad = 2.0;
                    let pill_x =
                        bounds.x + state.pill_center + visual_offset - state.pill_width / 2.0;
                    let pill_y = bounds.y + pill_pad;
                    let pill_h = bounds.height - pill_pad * 2.0;

                    // Clamp pill to stay within the background bounds so it doesn't
                    // extend past the carousel's rounded corners.
                    let clamped_left = pill_x.max(render_bounds.x);
                    let clamped_right =
                        (pill_x + state.pill_width).min(render_bounds.x + render_bounds.width);
                    let clamped_w = (clamped_right - clamped_left).max(0.0);

                    if clamped_w > 0.0 {
                        let pill_bounds = Rectangle {
                            x: clamped_left,
                            y: pill_y,
                            width: clamped_w,
                            height: pill_h,
                        };

                        // Use background radius on sides that touch the carousel edge,
                        // pill radius on interior sides.
                        let pill_max_r = pill_h / 2.0;
                        let pill_r: [f32; 4] =
                            cosmic.corner_radii.radius_xl.map(|cr| cr.min(pill_max_r));
                        let touches_left = pill_x < render_bounds.x + 0.5;
                        let touches_right =
                            pill_x + state.pill_width > render_bounds.x + render_bounds.width - 0.5;

                        // CSS order: TL, TR, BR, BL
                        let final_r: [f32; 4] = [
                            if touches_left {
                                bg_radius[0]
                            } else {
                                pill_r[0]
                            },
                            if touches_right {
                                bg_radius[1]
                            } else {
                                pill_r[1]
                            },
                            if touches_right {
                                bg_radius[2]
                            } else {
                                pill_r[2]
                            },
                            if touches_left {
                                bg_radius[3]
                            } else {
                                pill_r[3]
                            },
                        ];

                        renderer.fill_quad(
                            renderer::Quad {
                                bounds: strip_rect(quarter, strip_bounds, pill_bounds),
                                border: Border {
                                    color: Color::TRANSPARENT,
                                    width: 0.0,
                                    radius: permute_radii(quarter, final_r).into(),
                                },
                                shadow: Default::default(),
                                snap: true,
                            },
                            Background::Color(Color::from_rgba(
                                accent.red,
                                accent.green,
                                accent.blue,
                                accent_alpha,
                            )),
                        );
                    }
                }

                // When rotated, labels are the only thing that needs a real
                // rotation (quads stay axis-aligned). Collect them into a canvas
                // Frame whose transform turns the whole strip-anchored block: draw
                // each glyph run at its portrait-local position, let the frame
                // rotate them, then emit the tessellated geometry once. In portrait
                // this frame is never created and the direct `fill_text` path runs.
                let mut text_frame = if quarter != Quarter::None {
                    let mut frame = Frame::new(renderer, viewport.size());
                    let (tx, ty) = match quarter {
                        Quarter::Ccw90 => (strip_bounds.x, strip_bounds.y + strip_bounds.height),
                        Quarter::Cw90 => (strip_bounds.x + strip_bounds.width, strip_bounds.y),
                        Quarter::None => unreachable!(),
                    };
                    frame.translate(Vector::new(tx, ty));
                    frame.rotate(quarter.radians());
                    Some(frame)
                } else {
                    None
                };

                // Render all labels (in the expanded area during drag)
                for (i, _mode) in self.modes.iter().enumerate() {
                    let label_w = state.label_widths[i];
                    let label_center = state.label_centers[i];
                    let label_x = bounds.x + label_center + visual_offset - label_w / 2.0;

                    // Skip labels fully outside the render area
                    if label_x + label_w < render_bounds.x
                        || label_x > render_bounds.x + render_bounds.width
                    {
                        continue;
                    }

                    let label_bounds = Rectangle {
                        x: label_x,
                        y: bounds.y,
                        width: label_w,
                        height: bounds.height,
                    };

                    let is_highlighted = i == highlighted_idx;

                    // Use cached label strings to avoid fl!() allocations in the draw path
                    let label_text = state.label_strings.get(i).cloned().unwrap_or_default();
                    let on_bg: Color = cosmic.on_bg_color().into();
                    let text_color = if is_highlighted {
                        on_bg
                    } else {
                        Color::from_rgba(on_bg.r, on_bg.g, on_bg.b, inactive_alpha)
                    };

                    let font = if is_highlighted {
                        cosmic::font::bold()
                    } else {
                        cosmic::font::default()
                    };

                    // With Alignment::Center, position is the CENTER point of the text
                    if let Some(frame) = text_frame.as_mut() {
                        // Position is the label centre expressed in portrait-local
                        // coordinates (relative to the block origin); the frame's
                        // translate+rotate turns it into the strip. Same size and
                        // centre-alignment as the portrait path, so glyph metrics
                        // are unchanged - only the whole run is rotated.
                        let centre = label_bounds.center();
                        let mut text = CanvasText::from(label_text);
                        text.position = Point::new(centre.x - bounds.x, centre.y - bounds.y);
                        text.color = text_color;
                        text.size = Pixels(FONT_SIZE);
                        text.font = font;
                        text.align_x = iced_text::Alignment::Center;
                        text.align_y = alignment::Vertical::Center;
                        text.shaping = iced_text::Shaping::Advanced;
                        frame.fill_text(text);
                    } else {
                        iced_text::Renderer::fill_text(
                            renderer,
                            IcedText {
                                content: label_text,
                                bounds: label_bounds.size(),
                                size: Pixels(FONT_SIZE),
                                line_height: iced_text::LineHeight::default(),
                                font,
                                align_x: iced_text::Alignment::Center,
                                align_y: alignment::Vertical::Center,
                                shaping: iced_text::Shaping::Advanced,
                                wrapping: iced_text::Wrapping::default(),
                                ellipsize: iced_text::Ellipsize::default(),
                            },
                            label_bounds.center(),
                            text_color,
                            render_bounds,
                        );
                    }
                }

                // Emit the rotated label geometry once, on top of the quads.
                if let Some(frame) = text_frame {
                    use iced_wgpu::graphics::geometry::Renderer as _;
                    renderer.draw_geometry(frame.into_geometry());
                }
            },
        );

        // Draw fade overlays in a SEPARATE layer so they composite
        // on top of BOTH text and quads from the previous layer.
        // Skip when fade_w is too small to produce a visible gradient (avoids 1px artifacts).
        // The edge fades run along the block's long axis; rotating them onto the
        // strip is a later task, so they are disabled while rotated (the labels
        // and pill carry the interaction; only the soft gradient is missing).
        if fade_w > 2.0 && quarter == Quarter::None {
            renderer.with_layer(render_bounds, |renderer| {
                use iced_wgpu::primitive::Renderer as PrimitiveRenderer;

                // Fade overlays at the outer edges of the expanded area.
                // Use the same clamped corner radius as the background.
                let fade_render_w = fade_w;
                let corner_r = bg_radius[0].max(bg_radius[3]);

                // Left fade: at the left edge of render_bounds
                let left_bounds = Rectangle {
                    x: render_bounds.x,
                    y: bounds.y,
                    width: fade_render_w,
                    height: bounds.height,
                };
                renderer.draw_primitive(
                    left_bounds,
                    super::fade_primitive::FadePrimitive {
                        color: [bg_color.r, bg_color.g, bg_color.b, 1.0],
                        direction: 0,
                        corner_radius: corner_r,
                    },
                );

                // Right fade: at the right edge of render_bounds
                let right_bounds = Rectangle {
                    x: render_bounds.x + render_bounds.width - fade_render_w,
                    y: bounds.y,
                    width: fade_render_w,
                    height: bounds.height,
                };
                renderer.draw_primitive(
                    right_bounds,
                    super::fade_primitive::FadePrimitive {
                        color: [bg_color.r, bg_color.g, bg_color.b, 1.0],
                        direction: 1,
                        corner_radius: corner_r,
                    },
                );
            });
        }
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: layout::Layout<'_>,
        cursor: mouse::Cursor,
        _renderer: &Renderer,
        _clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        _viewport: &Rectangle,
    ) {
        // Work in portrait space (see `draw`): all hit-testing below runs
        // against the un-rotated block, and every incoming pointer is mapped
        // back into that space first via `remap`. In portrait `remap` and
        // `portrait_*` are the identity, so behaviour is unchanged.
        let strip_bounds = layout.bounds();
        let quarter = self.quarter;
        let bounds = portrait_bounds(quarter, strip_bounds);
        let viewport_p = portrait_viewport(quarter, *_viewport);
        let remap = |p: Point| remap_pointer(quarter, strip_bounds, p);
        let state = tree.state.downcast_mut::<CarouselState>();

        // Tick animations here — update() runs on events including redraw requests.
        if tick_animations(state, &self.modes, self.selected_index(), shell) {
            shell.request_redraw();
        }

        // Update button slide position from update() (not draw()) to avoid
        // side effects in the rendering pass.
        let button_bounds = button_axis_bounds(quarter, strip_bounds, bounds);
        let inward = button_slide_for_orientation(quarter, button_bounds, viewport_p, state);
        self.slide_shared
            .store(inward.to_bits(), std::sync::atomic::Ordering::Relaxed);

        if self.disabled {
            return;
        }

        match event {
            // === Touch events ===
            Event::Touch(touch::Event::FingerPressed { id, position })
                if visual_bounds(bounds, state).contains(remap(*position))
                    && state.active_finger.is_none() =>
            {
                let position = remap(*position);
                interrupt_animation(state);
                start_expand(state);
                // Interrupt any running stage 2 collapse so it doesn't
                // fight with the new interaction
                if state.full_expand_start.is_some() && !state.full_expanding {
                    start_full_expand(state);
                }
                state.press_time = Some(Instant::now());
                state.stage2_collapse_done = None;
                state.hover_leave_time = None;
                state.active_finger = Some(*id);
                state.drag_start_x = Some(position.x);
                state.drag_base_offset = state.drag_offset;
                state.press_widget_x = position.x - bounds.x;
                shell.capture_event();
                shell.request_redraw(); // for hold detection timer
            }
            Event::Touch(touch::Event::FingerMoved { id, position })
                if state.active_finger == Some(*id) =>
            {
                let position = remap(*position);
                if let Some(start_x) = state.drag_start_x {
                    let raw = state.drag_base_offset + (position.x - start_x);
                    state.drag_offset =
                        rubber_band_offset(raw, state, &self.modes, self.selected_index());
                    check_drag_haptic(state, &self.modes, self.selected_index());
                    // Trigger stage 2 on drag movement (or reverse a running collapse)
                    if !state.full_expanding || state.full_expand_start.is_none() {
                        start_full_expand(state);
                    }
                }
                shell.capture_event();
                shell.request_redraw();
            }
            Event::Touch(touch::Event::FingerLifted { id, .. })
                if state.active_finger == Some(*id) =>
            {
                reset_interaction_state(state);
                handle_drag_end(
                    state,
                    &self.modes,
                    self.selected_index(),
                    bounds.width,
                    shell,
                    &self.on_select,
                );
                shell.request_redraw();
                shell.capture_event();
            }
            Event::Touch(touch::Event::FingerLost { id, .. })
                if state.active_finger == Some(*id) =>
            {
                reset_interaction_state(state);
                start_snap_animation(state, 0.0);
                shell.request_redraw();
                shell.capture_event();
            }

            // === Mouse events ===
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                if let Some(pos) = cursor.position() {
                    let pos = remap(pos);
                    let vb = visual_bounds(bounds, state);
                    if vb.contains(pos) && state.active_finger.is_none() {
                        interrupt_animation(state);
                        start_expand(state);
                        // Interrupt any running stage 2 collapse
                        if state.full_expand_start.is_some() && !state.full_expanding {
                            start_full_expand(state);
                        }
                        state.press_time = Some(Instant::now());
                        state.stage2_collapse_done = None;
                        state.hover_leave_time = None;
                        state.drag_start_x = Some(pos.x);
                        state.drag_base_offset = state.drag_offset;
                        state.press_widget_x = pos.x - bounds.x;
                        shell.capture_event();
                        shell.request_redraw(); // for hold detection timer
                    }
                }
            }
            Event::Mouse(mouse::Event::CursorMoved { position }) => {
                let position = remap(*position);
                if let Some(start_x) = state.drag_start_x
                    && state.active_finger.is_none()
                {
                    let raw = state.drag_base_offset + (position.x - start_x);
                    state.drag_offset =
                        rubber_band_offset(raw, state, &self.modes, self.selected_index());
                    check_drag_haptic(state, &self.modes, self.selected_index());
                    // Trigger stage 2 on drag movement (or reverse a running collapse)
                    if !state.full_expanding || state.full_expand_start.is_none() {
                        start_full_expand(state);
                    }
                    shell.request_redraw();
                }

                // Hover detection: expand only when over the carousel itself,
                // but stay expanded (and abort collapse) when over adjacent buttons.
                let over_carousel = visual_bounds(bounds, state).contains(position);
                let over_area = hover_bounds(bounds, state).contains(position);
                if over_carousel && !state.hovered_carousel {
                    // Entering the carousel — trigger expansion + hold timer
                    state.hovered = true;
                    state.hovered_carousel = true;
                    state.hover_leave_time = None;
                    start_expand(state);
                    state.press_time = Some(Instant::now());
                    state.stage2_collapse_done = None;
                    // Reverse any running stage 2 collapse
                    if state.full_expand_start.is_some() && !state.full_expanding {
                        start_full_expand(state);
                    }
                    shell.request_redraw();
                } else if over_carousel && state.hovered_carousel {
                    // Already over carousel — just reverse any running collapse
                    if state.full_expand_start.is_some() && !state.full_expanding {
                        start_full_expand(state);
                        shell.request_redraw();
                    }
                    if state.expand_start.is_some() && !state.expanding {
                        start_expand(state);
                        shell.request_redraw();
                    }
                } else if !over_carousel && state.hovered_carousel {
                    // Left the carousel (might still be over buttons)
                    state.hovered_carousel = false;
                    state.press_time = None;
                }

                if over_area
                    && !state.hovered
                    && (state.expand_t > 0.01 || state.expand_start.is_some())
                {
                    // Entering button/gap area while carousel is expanded —
                    // don't trigger expansion, but abort any running collapse.
                    state.hovered = true;
                    state.hover_leave_time = None;
                    state.stage2_collapse_done = None;
                    if state.full_expand_start.is_some() && !state.full_expanding {
                        start_full_expand(state);
                    }
                    if state.expand_start.is_some() && !state.expanding {
                        start_expand(state);
                    }
                    shell.request_redraw();
                } else if !over_area && state.hovered {
                    // Left the entire area (carousel + buttons) — start delayed collapse
                    state.hovered = false;
                    state.hovered_carousel = false;
                    state.press_time = None;
                    if state.drag_start_x.is_none() {
                        state.hover_leave_time = Some(Instant::now());
                    }
                    shell.request_redraw();
                }
            }
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left))
                if state.drag_start_x.is_some() && state.active_finger.is_none() =>
            {
                state.drag_start_x = None;
                // Check if cursor is still over carousel or adjacent buttons
                let over_carousel = cursor
                    .position()
                    .is_some_and(|p| visual_bounds(bounds, state).contains(remap(p)));
                let over_area = cursor
                    .position()
                    .is_some_and(|p| hover_bounds(bounds, state).contains(remap(p)));
                state.hovered = over_area;
                if over_carousel {
                    // Over carousel — stay expanded, reset hold timer
                    state.press_time = Some(Instant::now());
                    state.stage2_collapse_done = None;
                } else if over_area {
                    // Over button area — keep expanded, no new hold timer
                    state.stage2_collapse_done = None;
                    state.press_time = None;
                } else {
                    state.press_time = None;
                    start_full_collapse(state);
                    if state.full_expand_t < 0.01 {
                        start_collapse(state);
                    }
                }
                handle_drag_end(
                    state,
                    &self.modes,
                    self.selected_index(),
                    bounds.width,
                    shell,
                    &self.on_select,
                );
            }

            // === Scroll wheel ===
            Event::Mouse(mouse::Event::WheelScrolled { delta })
                if cursor
                    .position()
                    .is_some_and(|p| visual_bounds(bounds, state).contains(remap(p))) =>
            {
                let scroll = wheel_scroll(quarter, delta);
                let idx = self.selected_index();
                if scroll < -0.5 && idx + 1 < self.modes.len() {
                    shell.publish((self.on_select)(self.modes[idx + 1]));
                } else if scroll > 0.5 && idx > 0 {
                    shell.publish((self.on_select)(self.modes[idx - 1]));
                }
                shell.capture_event();
            }

            _ => {}
        }
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: layout::Layout<'_>,
        cursor: mouse::Cursor,
        _viewport: &Rectangle,
        _renderer: &Renderer,
    ) -> mouse::Interaction {
        let state = tree.state.downcast_ref::<CarouselState>();
        let strip_bounds = layout.bounds();
        let bounds = portrait_bounds(self.quarter, strip_bounds);
        if state.active_finger.is_some() || state.drag_start_x.is_some() {
            mouse::Interaction::Grabbing
        } else if !self.disabled
            && cursor.position().is_some_and(|p| {
                visual_bounds(bounds, state).contains(remap_pointer(self.quarter, strip_bounds, p))
            })
        {
            mouse::Interaction::Pointer
        } else {
            mouse::Interaction::default()
        }
    }
}

impl<'a> From<ModeCarousel<'a>> for Element<'a, Message, Theme, Renderer> {
    fn from(widget: ModeCarousel<'a>) -> Self {
        Element::new(widget)
    }
}

// ============================================================================
// Rotation geometry (sideways strip)
// ============================================================================
//
// The carousel is authored horizontally ("portrait" space). When the device is
// held sideways the block turns a quarter along the strip. We keep all the
// internal layout/spacing/drag math horizontal and only remap coordinates:
//
//   * `portrait_bounds` / `portrait_viewport` un-swap the axes so the math runs
//     against the block as if it were still horizontal, anchored at the strip's
//     origin.
//   * `strip_point` / `strip_rect` turn a portrait rectangle (a quad) back into
//     the strip - a 90° turn of an axis-aligned rect stays axis-aligned.
//   * `remap_pointer` is the inverse: it maps an incoming pointer in strip space
//     back into portrait space so the existing hit-testing is untouched.
//
// Every function is the identity for `Quarter::None`, so portrait is unchanged.

/// The block laid out horizontally: same origin, width/height un-swapped.
pub(super) fn portrait_bounds(quarter: Quarter, bounds: Rectangle) -> Rectangle {
    match quarter {
        Quarter::None => bounds,
        Quarter::Cw90 | Quarter::Ccw90 => Rectangle {
            x: bounds.x,
            y: bounds.y,
            width: bounds.height,
            height: bounds.width,
        },
    }
}

/// The viewport with its axes un-swapped, matching `portrait_bounds`.
fn portrait_viewport(quarter: Quarter, viewport: Rectangle) -> Rectangle {
    match quarter {
        Quarter::None => viewport,
        Quarter::Cw90 | Quarter::Ccw90 => Rectangle {
            x: viewport.y,
            y: viewport.x,
            width: viewport.height,
            height: viewport.width,
        },
    }
}

/// Map a point from portrait space into the on-screen strip space.
/// `bounds` is the strip (layout) rectangle. Inverse of [`remap_pointer`].
fn strip_point(quarter: Quarter, bounds: Rectangle, p: Point) -> Point {
    let a = p.x - bounds.x; // along the row (portrait X)
    let b = p.y - bounds.y; // across the row (portrait Y)
    match quarter {
        Quarter::None => p,
        Quarter::Ccw90 => Point::new(bounds.x + b, bounds.y + bounds.height - a),
        Quarter::Cw90 => Point::new(bounds.x + bounds.width - b, bounds.y + a),
    }
}

/// Turn a portrait-space axis-aligned rectangle into its strip-space rectangle.
pub(super) fn strip_rect(quarter: Quarter, bounds: Rectangle, r: Rectangle) -> Rectangle {
    if quarter == Quarter::None {
        return r;
    }
    let a = strip_point(quarter, bounds, Point::new(r.x, r.y));
    let c = strip_point(quarter, bounds, Point::new(r.x + r.width, r.y + r.height));
    Rectangle {
        x: a.x.min(c.x),
        y: a.y.min(c.y),
        width: (a.x - c.x).abs(),
        height: (a.y - c.y).abs(),
    }
}

/// Permute CSS corner radii `[TL, TR, BR, BL]` to follow the quarter turn so a
/// rounded quad keeps its rounded corners on the same physical edges.
pub(super) fn permute_radii(quarter: Quarter, r: [f32; 4]) -> [f32; 4] {
    match quarter {
        Quarter::None => r,
        Quarter::Ccw90 => [r[1], r[2], r[3], r[0]],
        Quarter::Cw90 => [r[3], r[0], r[1], r[2]],
    }
}

/// Map an incoming pointer in strip space back into the carousel's portrait
/// space so the existing hit-testing (drag / snap / tap thresholds) is unchanged.
/// Inverse of [`strip_point`]. Identity for `Quarter::None`.
pub(super) fn remap_pointer(quarter: Quarter, bounds: Rectangle, p: Point) -> Point {
    let u = p.x - bounds.x;
    let v = p.y - bounds.y;
    match quarter {
        Quarter::None => p,
        Quarter::Ccw90 => Point::new(bounds.x + bounds.height - v, bounds.y + u),
        Quarter::Cw90 => Point::new(bounds.x + v, bounds.y + bounds.width - u),
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn wheel_scroll(quarter: Quarter, delta: &mouse::ScrollDelta) -> f32 {
    match (quarter, delta) {
        (Quarter::None, mouse::ScrollDelta::Lines { x, .. }) => *x,
        (Quarter::None, mouse::ScrollDelta::Pixels { x, .. }) => *x / 50.0,
        (Quarter::Cw90, mouse::ScrollDelta::Lines { y, .. }) => *y,
        (Quarter::Cw90, mouse::ScrollDelta::Pixels { y, .. }) => *y / 50.0,
        (Quarter::Ccw90, mouse::ScrollDelta::Lines { y, .. }) => -*y,
        (Quarter::Ccw90, mouse::ScrollDelta::Pixels { y, .. }) => -*y / 50.0,
    }
}

/// Get the localized display label for a camera mode.
fn mode_label(mode: CameraMode) -> String {
    match mode {
        CameraMode::Photo => fl!("mode-photo"),
        CameraMode::Video => fl!("mode-video"),
        CameraMode::Timelapse => fl!("mode-timelapse"),
        CameraMode::Virtual => fl!("mode-virtual"),
        CameraMode::View => fl!("mode-view"),
    }
}

/// Estimate label width based on character count and font size.
fn estimate_label_width(label: &str, pad_h: f32) -> f32 {
    let char_width = FONT_SIZE * 0.50;
    label.chars().count() as f32 * char_width + (pad_h + 4.0) * 2.0
}

/// Compute the carousel layout width for a given set of modes.
/// Used by the recording row layout to match the carousel's center width.
pub fn carousel_width_for_modes(modes: &[CameraMode]) -> f32 {
    let spacing = cosmic::theme::spacing();
    let pad_h = spacing.space_xs as f32;

    let mut widths: Vec<f32> = modes
        .iter()
        .map(|m| estimate_label_width(&mode_label(*m), pad_h))
        .collect();

    if widths.len() <= 2 {
        widths.iter().sum::<f32>() + (widths.len().saturating_sub(1)) as f32 * LABEL_GAP
    } else {
        widths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        widths[0] + widths[1] + LABEL_GAP
    }
}

/// Collapsed space from a rotated carousel to each sibling button. Spacious
/// windows reserve medium expansion; compact windows keep the normal gap and
/// let the expansion animation move the buttons out of its way.
pub fn sideways_sibling_spacing_for_modes(modes: &[CameraMode], viewport_long_axis: f32) -> f32 {
    let spacing = cosmic::theme::spacing();
    if viewport_long_axis > 0.0 && viewport_long_axis <= COMPACT_SIDE_AXIS_MAX {
        spacing.space_m as f32
    } else {
        carousel_width_for_modes(modes) / 4.0 + spacing.space_s as f32
    }
}

fn sideways_reserved_extension(carousel_long: f32, viewport_long_axis: f32) -> f32 {
    let spacing = cosmic::theme::spacing();
    (if viewport_long_axis > 0.0 && viewport_long_axis <= COMPACT_SIDE_AXIS_MAX {
        spacing.space_m as f32
    } else {
        carousel_long / 4.0 + spacing.space_s as f32
    } - spacing.space_s as f32)
        .max(0.0)
}

/// Recalculate label widths, center positions, and cached label strings.
/// Skips work if mode count hasn't changed (labels are stable across frames).
fn recalculate_label_metrics(modes: &[CameraMode], state: &mut CarouselState) {
    if state.cached_mode_count == modes.len() && state.label_widths.len() == modes.len() {
        return; // Labels haven't changed
    }

    state.label_widths.clear();
    state.label_centers.clear();
    state.label_strings.clear();

    let spacing = cosmic::theme::spacing();
    let pad_h = spacing.space_xs as f32;
    let mut x = 0.0;
    for mode in modes {
        let label = mode_label(*mode);
        let width = estimate_label_width(&label, pad_h);
        state.label_widths.push(width);
        state.label_centers.push(x + width / 2.0);
        state.label_strings.push(label);
        x += width + LABEL_GAP;
    }
    state.cached_mode_count = modes.len();
}

/// Hover detection bounds: extends visual bounds to cover the gap + adjacent buttons.
/// Moving the cursor from the carousel to the gallery/switcher button doesn't trigger leave.
fn hover_bounds(bounds: Rectangle, state: &CarouselState) -> Rectangle {
    let mut vb = visual_bounds(bounds, state);
    let spacing = cosmic::theme::spacing();
    // Extend by button_spacing (gap between carousel edge and button) + button width
    let margin = spacing.space_s as f32 + BUTTON_WIDTH;
    vb.x -= margin;
    vb.width += margin * 2.0;
    vb
}

/// Compute the carousel's visual bounds including expansion beyond layout bounds.
/// Note: this may be slightly wider than the rendered area (draw applies a
/// max_extend clamp) — intentional so hit-testing is generous.
fn visual_bounds(bounds: Rectangle, state: &CarouselState) -> Rectangle {
    let fade_w_max = bounds.width / 4.0;
    let needed = compute_needed_per_side(bounds.width, state);
    let stage2_extra = (needed - fade_w_max).max(0.0) * state.full_expand_t;
    let ext = fade_w_max * state.expand_t + stage2_extra;
    Rectangle {
        x: bounds.x - ext,
        y: bounds.y,
        width: bounds.width + ext * 2.0,
        height: bounds.height,
    }
}

/// The carousel's natural along-block content length (its "long" axis): the
/// full label strip for ≤2 modes, otherwise the two narrowest labels (so the
/// size is stable across mode switches). This is the compact *collapsed* size -
/// the expansion animation is drawn BEYOND the layout node (see `render_bounds`
/// / `with_layer`), so the node itself never needs the whole lane length.
fn content_long(label_widths: &[f32]) -> f32 {
    if label_widths.len() <= 2 {
        let total: f32 = label_widths.iter().sum();
        total + label_widths.len().saturating_sub(1) as f32 * LABEL_GAP
    } else {
        let mut sorted: Vec<f32> = label_widths.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[0] + sorted[1] + LABEL_GAP
    }
}

/// The on-screen layout node size for the carousel given its content length and
/// the available `limits`. Rotated (`quarter != None`) the block's fixed
/// thickness (`CAROUSEL_HEIGHT`) crosses the strip while the content length runs
/// DOWN it - bounded to the compact content, so a `Fill`-height group lane never
/// inflates the node to the whole strip (which would push the gallery/switcher
/// far from the pill and defeat the compact, centred stack). Portrait is the
/// axis-swap of the same rule and is byte-identical to before.
fn carousel_node_size(quarter: Quarter, long: f32, limits: &layout::Limits) -> Size<f32> {
    if quarter == Quarter::None {
        Size::new(long.min(limits.max().width), CAROUSEL_HEIGHT)
    } else {
        Size::new(CAROUSEL_HEIGHT, long.min(limits.max().height))
    }
}

/// Total width of the label strip (all labels + gaps between them).
fn total_strip_width(state: &CarouselState) -> f32 {
    if state.label_widths.is_empty() {
        return 0.0;
    }
    let total: f32 = state.label_widths.iter().sum();
    total + (state.label_widths.len() - 1) as f32 * LABEL_GAP
}

/// Tick all animations forward. Called from update() (runs on events,
/// including each requested redraw frame). Returns `true` if a next-frame
/// redraw is required for an active animation; for idle delay timers it
/// instead schedules a single one-shot wake-up on `shell` so we don't
/// repaint at vsync rate while only counting down a timer.
fn tick_animations(
    state: &mut CarouselState,
    modes: &[CameraMode],
    selected_idx: usize,
    shell: &mut Shell<'_, Message>,
) -> bool {
    let mut needs_redraw = false;

    // Compute delta time for frame-rate independent spring.
    // Normalized to 60fps: dt=1.0 at 60fps, dt=2.0 at 30fps, etc.
    let now = Instant::now();
    let dt = state
        .last_tick
        .map(|last| now.duration_since(last).as_secs_f32() * 60.0)
        .unwrap_or(1.0)
        .min(4.0);
    state.last_tick = Some(now);

    // Snap animation
    if state.snap_start.is_some() {
        let done = tick_snap_animation(state);
        if done {
            state.pending_mode = None;
        } else {
            needs_redraw = true;
        }
    }

    // Stage 1 expansion
    if let Some(expand_start) = state.expand_start {
        let elapsed = expand_start.elapsed().as_secs_f32() * 1000.0;
        let t = (elapsed / EXPAND_TRANSITION_MS).min(1.0);
        let eased = ease_out_cubic(t);
        if state.expanding {
            state.expand_t = eased;
        } else {
            state.expand_t = 1.0 - eased;
        }
        if t >= 1.0 {
            state.expand_start = None;
        } else {
            needs_redraw = true;
        }
    }

    // Stage 2 full expansion
    if let Some(full_start) = state.full_expand_start {
        let elapsed = full_start.elapsed().as_secs_f32() * 1000.0;
        let t = (elapsed / EXPAND_TRANSITION_MS).min(1.0);
        let eased = ease_out_cubic(t);
        let new_t = if state.full_expanding {
            eased
        } else {
            1.0 - eased
        };
        state.full_expand_t = new_t;
        if t >= 1.0 {
            state.full_expand_start = None;
            if !state.full_expanding {
                state.stage2_collapse_done = Some(Instant::now());
                needs_redraw = true;
            }
        } else {
            needs_redraw = true;
        }
    }

    // Hold threshold for stage 2
    if let Some(press_time) = state.press_time {
        let hold_ms = press_time.elapsed().as_secs_f32() * 1000.0;
        if hold_ms >= HOLD_TO_FULL_EXPAND_MS
            && state.full_expand_t < 0.01
            && state.full_expand_start.is_none()
        {
            start_full_expand(state);
            needs_redraw = true;
        } else if state.full_expand_t < 0.01 && state.full_expand_start.is_none() {
            // Idle wait for the hold threshold: schedule a single wake-up at
            // the deadline instead of polling every frame.
            shell.request_redraw_at(
                press_time + std::time::Duration::from_millis(HOLD_TO_FULL_EXPAND_MS as u64),
            );
        }
    }

    // Delayed hover-leave collapse
    if let Some(leave_time) = state.hover_leave_time {
        if state.hovered {
            state.hover_leave_time = None;
        } else if leave_time.elapsed().as_secs_f32() * 1000.0 >= HOVER_LEAVE_COLLAPSE_DELAY_MS {
            state.hover_leave_time = None;
            start_full_collapse(state);
            if state.full_expand_t < 0.01 {
                start_collapse(state);
            }
            needs_redraw = true;
        } else {
            // Idle wait — schedule a single redraw at the deadline.
            shell.request_redraw_at(
                leave_time + std::time::Duration::from_millis(HOVER_LEAVE_COLLAPSE_DELAY_MS as u64),
            );
        }
    }

    // Delayed stage 1 collapse
    let is_pressing =
        state.active_finger.is_some() || state.drag_start_x.is_some() || state.hovered;
    if let Some(done_time) = state.stage2_collapse_done {
        if is_pressing {
            state.stage2_collapse_done = None;
        } else if done_time.elapsed().as_secs_f32() * 1000.0 >= STAGE1_COLLAPSE_AFTER_STAGE2_MS {
            state.stage2_collapse_done = None;
            start_collapse(state);
            needs_redraw = true;
        } else {
            // Idle wait — schedule a single redraw at the deadline.
            shell.request_redraw_at(
                done_time
                    + std::time::Duration::from_millis(STAGE1_COLLAPSE_AFTER_STAGE2_MS as u64),
            );
        }
    }

    // Pill spring animation
    if !state.label_centers.is_empty() {
        let vis_off = current_visual_offset(state);
        let highlighted_idx = compute_highlighted_idx(state, modes, selected_idx, vis_off);
        let target_center = state.label_centers[highlighted_idx];
        let target_width = state.label_widths[highlighted_idx];
        if !state.pill_initialized {
            state.pill_center = target_center;
            state.pill_width = target_width;
            state.pill_initialized = true;
        } else {
            // Frame-rate independent damped spring: scale forces by dt
            // so behavior is consistent at 30fps, 60fps, 120fps, etc.
            const STIFFNESS: f32 = 0.08;
            const DAMPING: f32 = 0.65;
            let damping = DAMPING.powf(dt);
            state.pill_vel_center = state.pill_vel_center * damping
                + (target_center - state.pill_center) * STIFFNESS * dt;
            state.pill_center += state.pill_vel_center * dt;
            state.pill_vel_width =
                state.pill_vel_width * damping + (target_width - state.pill_width) * STIFFNESS * dt;
            state.pill_width += state.pill_vel_width * dt;
            if state.pill_vel_center.abs() > 0.1
                || state.pill_vel_width.abs() > 0.1
                || (state.pill_center - target_center).abs() > 0.5
                || (state.pill_width - target_width).abs() > 0.5
            {
                needs_redraw = true;
            }
        }
    }

    needs_redraw
}

/// Determine which mode label should be highlighted (pill target).
/// Used by both `draw()` and `tick_animations()`.
fn compute_highlighted_idx(
    state: &CarouselState,
    modes: &[CameraMode],
    selected_idx: usize,
    vis_off: f32,
) -> usize {
    if let Some(pending) = state.pending_mode {
        // Tap-initiated mode change: pill goes directly to the target mode
        // without highlighting intermediate modes during the scroll.
        modes
            .iter()
            .position(|m| *m == pending)
            .unwrap_or(selected_idx)
    } else if state.drag_start_x.is_some() || state.snap_start.is_some() {
        let anchor = anchor_center(state, modes, selected_idx);
        nearest_mode_to_viewport_center(state, anchor, selected_idx, vis_off)
    } else {
        selected_idx
    }
}

/// Compute the per-side extension needed to show all labels plus fade zones.
fn compute_needed_per_side(bounds_width: f32, state: &CarouselState) -> f32 {
    let fade_w_max = bounds_width / 4.0;
    let strip_w = total_strip_width(state);
    let spacing = cosmic::theme::spacing();
    let label_padding = spacing.space_xs as f32;
    ((strip_w + label_padding * 2.0 - bounds_width) / 2.0 + fade_w_max).max(0.0)
}

fn carousel_extension_for_progress(
    bounds: Rectangle,
    viewport: Rectangle,
    state: &CarouselState,
    expand: f32,
    full_expand: f32,
    reserve_medium: bool,
) -> f32 {
    let fade_w_max = bounds.width / 4.0;
    let fade_w = fade_w_max * expand;
    let needed_per_side = compute_needed_per_side(bounds.width, state);
    let stage2_extra = (needed_per_side - fade_w_max).max(0.0) * full_expand;
    let spacing = cosmic::theme::spacing();
    let button_spacing = spacing.space_s as f32;
    let padding = spacing.space_m as f32;
    let gap = (bounds.x - padding - BUTTON_WIDTH).max(0.0);
    let avail = (gap - button_spacing).max(0.0);
    let viewport_half = (viewport.width - bounds.width) / 2.0;

    if needed_per_side <= avail {
        (fade_w + stage2_extra).min(needed_per_side)
    } else {
        // Side layouts reserve stage one in their sibling spacing. Portrait
        // retains its established behavior of fitting stage one in the actual
        // leading gap.
        let medium = if reserve_medium {
            fade_w.min(fade_w_max)
        } else {
            fade_w.min(avail)
        };
        let target = viewport_half.min(needed_per_side);
        medium + (target - medium).max(0.0) * full_expand
    }
}

#[cfg(test)]
fn carousel_extension(
    quarter: Quarter,
    strip_bounds: Rectangle,
    viewport: Rectangle,
    state: &CarouselState,
) -> f32 {
    let portrait = portrait_bounds(quarter, strip_bounds);
    let viewport = portrait_viewport(quarter, viewport);
    let axis_bounds = button_axis_bounds(quarter, strip_bounds, portrait);
    carousel_extension_for_progress(
        axis_bounds,
        viewport,
        state,
        state.expand_t,
        state.full_expand_t,
        quarter != Quarter::None,
    )
}

/// Get the current visual offset accounting for drag or snap animation.
fn current_visual_offset(state: &CarouselState) -> f32 {
    if let Some(snap_start) = state.snap_start {
        let elapsed = snap_start.elapsed().as_secs_f32() * 1000.0;
        let t = (elapsed / SNAP_DURATION_MS).min(1.0);
        let eased = ease_out_cubic(t);
        state.snap_from + (state.snap_target - state.snap_from) * eased
    } else {
        state.drag_offset
    }
}

/// Advance snap animation, returning true if complete.
fn tick_snap_animation(state: &mut CarouselState) -> bool {
    if let Some(snap_start) = state.snap_start {
        let elapsed = snap_start.elapsed().as_secs_f32() * 1000.0;
        if elapsed >= SNAP_DURATION_MS {
            state.drag_offset = state.snap_target;
            state.snap_start = None;
            state.snap_from = 0.0;
            state.snap_target = 0.0;
            return true;
        }
    }
    false
}

/// Inverse of ease_out_cubic: given an eased value y, return the linear t.
/// ease_out_cubic(t) = 1-(1-t)^3  →  inverse(y) = 1-(1-y)^(1/3)
fn inverse_ease_out_cubic(y: f32) -> f32 {
    1.0 - (1.0 - y.clamp(0.0, 1.0)).cbrt()
}

/// Start or resume an expand/collapse animation, backdating the start time
/// to maintain continuity from the current `current_t` value.
/// `expanding`: true to animate toward 1.0, false toward 0.0.
fn start_animation(
    current_t: f32,
    expanding: bool,
    anim_start: &mut Option<Instant>,
    anim_expanding: &mut bool,
) {
    // Already at target and no animation running — nothing to do.
    if expanding && current_t >= 1.0 && anim_start.is_none() {
        return;
    }
    if !expanding && current_t <= 0.0 && anim_start.is_none() {
        return;
    }
    *anim_expanding = expanding;
    // For expand: skip to where ease_out_cubic(t) == current_t.
    // For collapse: skip to where 1 - ease_out_cubic(t) == current_t,
    // i.e. ease_out_cubic(t) == 1 - current_t.
    let eased_progress = if expanding {
        current_t
    } else {
        1.0 - current_t
    };
    let skip_ms = inverse_ease_out_cubic(eased_progress) * EXPAND_TRANSITION_MS;
    *anim_start = Some(Instant::now() - std::time::Duration::from_millis(skip_ms as u64));
}

/// Start expanding the carousel (fade zones animate in).
fn start_expand(state: &mut CarouselState) {
    start_animation(
        state.expand_t,
        true,
        &mut state.expand_start,
        &mut state.expanding,
    );
}

/// Start collapsing the carousel (fade zones animate out).
fn start_collapse(state: &mut CarouselState) {
    start_animation(
        state.expand_t,
        false,
        &mut state.expand_start,
        &mut state.expanding,
    );
}

/// Start stage 2 full expansion (show all labels).
fn start_full_expand(state: &mut CarouselState) {
    start_animation(
        state.full_expand_t,
        true,
        &mut state.full_expand_start,
        &mut state.full_expanding,
    );
}

/// Start stage 2 collapse (back to normal width).
fn start_full_collapse(state: &mut CarouselState) {
    start_animation(
        state.full_expand_t,
        false,
        &mut state.full_expand_start,
        &mut state.full_expanding,
    );
}

/// Reset touch/mouse state after finger lift or loss.
fn reset_interaction_state(state: &mut CarouselState) {
    state.active_finger = None;
    state.drag_start_x = None;
    state.press_time = None;
    state.hovered = false;
    state.hovered_carousel = false;
    state.drag_nearest_idx = None;
    // Drop any in-flight tap-initiated mode change; otherwise the next layout
    // pass would compute snap math from a stale pending_mode after the
    // gesture has already been cancelled (e.g. FingerLost / palm rejection).
    state.pending_mode = None;
    start_full_collapse(state);
    if state.full_expand_t < 0.01 {
        start_collapse(state);
    }
}

/// Interrupt a running snap animation, freezing at current position.
fn interrupt_animation(state: &mut CarouselState) {
    if state.snap_start.is_some() {
        state.drag_offset = current_visual_offset(state);
        state.snap_start = None;
        state.pending_mode = None;
    }
}

/// Check if the nearest mode changed during drag and fire haptic feedback.
fn check_drag_haptic(state: &mut CarouselState, modes: &[CameraMode], selected_idx: usize) {
    let vis_off = current_visual_offset(state);
    let anchor = anchor_center(state, modes, selected_idx);
    let nearest = nearest_mode_to_viewport_center(state, anchor, selected_idx, vis_off);

    if state.drag_nearest_idx != Some(nearest) {
        if state.drag_nearest_idx.is_some() {
            // Only fire haptic after the first update (not on drag start)
            crate::backends::haptic::vibrate(10);
        }
        state.drag_nearest_idx = Some(nearest);
    }
}

/// Start a snap animation from current drag_offset to target.
fn start_snap_animation(state: &mut CarouselState, target: f32) {
    state.snap_from = state.drag_offset;
    state.snap_target = target;
    state.snap_start = Some(Instant::now());
}

/// Handle end of drag: determine nearest mode and start snap animation.
fn handle_drag_end(
    state: &mut CarouselState,
    modes: &[CameraMode],
    selected_idx: usize,
    viewport_width: f32,
    shell: &mut Shell<'_, Message>,
    on_select: &dyn Fn(CameraMode) -> Message,
) {
    let offset = state.drag_offset;

    // Tap detection: if finger barely moved, find which label was tapped
    if offset.abs() < TAP_THRESHOLD && !modes.is_empty() {
        let viewport_center = viewport_width / 2.0;
        let anchor = anchor_center(state, modes, selected_idx);
        let base_offset = viewport_center - anchor;
        let strip_x = state.press_widget_x - base_offset;
        if let Some(tapped_idx) = find_label_at_x(state, strip_x)
            && tapped_idx != selected_idx
        {
            // If both modes share the same anchor (Photo↔Video), the
            // carousel doesn't need to scroll — only the pill moves.
            let new_anchor = anchor_center(state, modes, tapped_idx);
            let delta = if (anchor - new_anchor).abs() < 0.1 {
                0.0 // same anchor, no scroll needed
            } else {
                state.label_centers[selected_idx] - state.label_centers[tapped_idx]
            };
            start_snap_animation(state, delta);
            state.pending_mode = Some(modes[tapped_idx]);
            shell.publish(on_select(modes[tapped_idx]));
            shell.request_redraw();
            return;
        }
        // Tapped the selected label — publish to toggle tools menu, then snap back
        shell.publish(on_select(modes[selected_idx]));
        start_snap_animation(state, 0.0);
        shell.request_redraw();
        return;
    }

    if offset.abs() > SWITCH_THRESHOLD && !modes.is_empty() {
        // Find which mode center is closest to the viewport center.
        // Use anchor (not label center) so Photo/Video shared anchor
        // doesn't skew the nearest-mode calculation.
        let anchor = anchor_center(state, modes, selected_idx);
        let target_center = anchor - offset;
        let nearest_idx = state
            .label_centers
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let da = (*a - target_center).abs();
                let db = (*b - target_center).abs();
                da.partial_cmp(&db).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(selected_idx);

        if nearest_idx != selected_idx {
            let delta = state.label_centers[selected_idx] - state.label_centers[nearest_idx];
            start_snap_animation(state, delta);
            state.pending_mode = Some(modes[nearest_idx]);
            shell.publish(on_select(modes[nearest_idx]));
            shell.request_redraw();
            return;
        }
    }

    // Snap back to current selection
    start_snap_animation(state, 0.0);
    shell.request_redraw();
}

/// Find which label index contains the given x position in strip coordinates.
fn find_label_at_x(state: &CarouselState, strip_x: f32) -> Option<usize> {
    for (i, (&center, &width)) in state
        .label_centers
        .iter()
        .zip(state.label_widths.iter())
        .enumerate()
    {
        let left = center - width / 2.0;
        let right = center + width / 2.0;
        if strip_x >= left && strip_x <= right {
            return Some(i);
        }
    }
    None
}

/// Calculate the anchor point in strip coordinates that should align with
/// viewport center. For Photo or Video, this is the midpoint between their
/// centers (so both are equally visible). For other modes, it's that mode's center.
fn anchor_center(state: &CarouselState, modes: &[CameraMode], selected_idx: usize) -> f32 {
    let mode = modes[selected_idx];
    if matches!(mode, CameraMode::Photo | CameraMode::Video) {
        // Find the indices of Photo and Video
        let photo_idx = modes.iter().position(|m| *m == CameraMode::Photo);
        let video_idx = modes.iter().position(|m| *m == CameraMode::Video);
        if let (Some(pi), Some(vi)) = (photo_idx, video_idx) {
            // Midpoint between Photo and Video centers
            return (state.label_centers[pi] + state.label_centers[vi]) / 2.0;
        }
    }
    state.label_centers[selected_idx]
}

/// Find which mode label is closest to the viewport center.
/// For the first snap (pill still on selected mode), uses the selected
/// label's own center so Photo↔Video snapping feels natural.
/// After the first snap, uses the anchor point so subsequent snaps
/// align with what's visually below the capture button.
fn nearest_mode_to_viewport_center(
    state: &CarouselState,
    anchor: f32,
    selected_idx: usize,
    vis_offset: f32,
) -> usize {
    if state.label_centers.is_empty() {
        return 0;
    }
    // Before the first snap: use the selected label's center as reference.
    // After the first snap (pill moved to a different label): use the anchor
    // so the viewport-center calculation matches what's below the capture button.
    let pill_on_selected = state.pill_initialized
        && (state.pill_center - state.label_centers[selected_idx]).abs() < 1.0;
    let reference = if pill_on_selected {
        state.label_centers[selected_idx]
    } else {
        anchor
    };
    let center_in_strip = reference - vis_offset;

    // If the carousel isn't scrolled (vis_offset near 0), the selected mode
    // is at (or near) the viewport center. Return it directly to avoid
    // ties between equidistant modes (e.g. Photo/Video shared anchor).
    if vis_offset.abs() < 1.0 {
        return selected_idx;
    }

    state
        .label_centers
        .iter()
        .enumerate()
        .min_by(|(i_a, a), (i_b, b)| {
            let da = (*a - center_in_strip).abs();
            let db = (*b - center_in_strip).abs();
            // On ties, prefer the selected mode
            da.partial_cmp(&db).unwrap().then_with(|| {
                let a_sel = if *i_a == selected_idx { 0 } else { 1 };
                let b_sel = if *i_b == selected_idx { 0 } else { 1 };
                a_sel.cmp(&b_sel)
            })
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// The resting (collapsed) inward slide for the side buttons, derived purely
/// from the carousel's on-screen bounds — the value `compute_button_slide`
/// returns when `full_expand_t == 0`. Exposed so the recording-state capture
/// row can reproduce the same offset without a live carousel in the tree
/// (during recording the carousel isn't rendered, so nothing else writes the
/// shared slide atomic). See `SlidePrimer`.
pub fn resting_button_slide(bounds: Rectangle) -> f32 {
    let spacing = cosmic::theme::spacing();
    let fade_w_max = bounds.width / 4.0;
    let button_spacing = spacing.space_s as f32;
    let padding = spacing.space_m as f32;
    let gap = (bounds.x - padding - BUTTON_WIDTH).max(0.0);
    gap - fade_w_max - button_spacing
}

/// Portrait buttons need absolute placement. Sideways siblings already reserve
/// stage-one growth in layout, so they move only for extra stage-two growth.
fn button_slide_for_orientation(
    quarter: Quarter,
    bounds: Rectangle,
    viewport: Rectangle,
    state: &CarouselState,
) -> f32 {
    if quarter == Quarter::None {
        compute_button_slide(bounds, &viewport, state)
    } else {
        let reserved = sideways_reserved_extension(bounds.width, viewport.width);
        let extension = carousel_extension_for_progress(
            bounds,
            viewport,
            state,
            state.expand_t,
            state.full_expand_t,
            true,
        );
        -(extension - reserved).max(0.0)
    }
}

/// Geometry for sibling-button clearance along the carousel's physical axis.
/// The carousel renderer works in synthetic portrait space, but landscape
/// siblings move along screen Y, so their edge distances must use Y/height.
fn button_axis_bounds(quarter: Quarter, strip_bounds: Rectangle, portrait: Rectangle) -> Rectangle {
    if quarter == Quarter::None {
        portrait
    } else {
        Rectangle {
            x: strip_bounds.y,
            y: strip_bounds.x,
            width: strip_bounds.height,
            height: strip_bounds.width,
        }
    }
}

/// Compute the button inward slide distance for the current animation state.
/// Called from `update()` so the value is fresh each event cycle without
/// mutating state inside `draw()`.
fn compute_button_slide(bounds: Rectangle, viewport: &Rectangle, state: &CarouselState) -> f32 {
    let spacing = cosmic::theme::spacing();
    let button_spacing = spacing.space_s as f32;
    let padding = spacing.space_m as f32;
    let gap = (bounds.x - padding - BUTTON_WIDTH).max(0.0);

    let medium_pos = resting_button_slide(bounds);

    let needed_per_side = compute_needed_per_side(bounds.width, state);
    let viewport_half = (viewport.width - bounds.width) / 2.0;
    let max_extend = if needed_per_side <= gap - button_spacing {
        needed_per_side
    } else {
        viewport_half.min(needed_per_side)
    };
    let carousel_left_at_full = bounds.x - max_extend;
    let natural_btn_right = padding + BUTTON_WIDTH + button_spacing;

    let target = if natural_btn_right > carousel_left_at_full {
        -(padding + BUTTON_WIDTH + button_spacing)
    } else {
        (carousel_left_at_full - natural_btn_right).min(medium_pos)
    };
    // Only move during stage 2 (full expansion) — stay at medium_pos during stage 1
    medium_pos + (target - medium_pos) * state.full_expand_t
}

/// Ease-out cubic: fast start, smooth deceleration, no bounce.
fn ease_out_cubic(t: f32) -> f32 {
    1.0 - (1.0 - t).powi(3)
}

/// Apply rubber-band resistance to drag offset.
/// Dampens the offset when dragging past the first or last mode.
fn rubber_band_offset(
    raw_offset: f32,
    state: &CarouselState,
    modes: &[CameraMode],
    selected_idx: usize,
) -> f32 {
    if state.label_centers.is_empty() || modes.is_empty() {
        return raw_offset;
    }

    let anchor = anchor_center(state, modes, selected_idx);
    let first_center = state.label_centers[0];
    let last_center = *state.label_centers.last().unwrap();

    // Maximum offsets that keep the first/last mode at the anchor point
    let max_right = anchor - first_center; // dragging right past first
    let max_left = anchor - last_center; // dragging left past last (negative)

    if raw_offset > max_right {
        // Dragging past first mode — diminishing return
        let overshoot = raw_offset - max_right;
        let dampened = 10.0 * (1.0 + overshoot / 10.0).ln();
        max_right + dampened
    } else if raw_offset < max_left {
        // Dragging past last mode — diminishing return
        let overshoot = max_left - raw_offset;
        let dampened = 10.0 * (1.0 + overshoot / 10.0).ln();
        max_left - dampened
    } else {
        raw_offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn side_carousel_wheel_uses_vertical_scroll_delta() {
        let vertical_lines = mouse::ScrollDelta::Lines { x: 0.0, y: -1.0 };
        let horizontal_lines = mouse::ScrollDelta::Lines { x: -1.0, y: 0.0 };
        let vertical_pixels = mouse::ScrollDelta::Pixels { x: 0.0, y: 75.0 };

        assert_eq!(wheel_scroll(Quarter::Cw90, &vertical_lines), -1.0);
        assert_eq!(wheel_scroll(Quarter::Cw90, &horizontal_lines), 0.0);
        assert_eq!(wheel_scroll(Quarter::Ccw90, &vertical_pixels), -1.5);
        assert_eq!(wheel_scroll(Quarter::None, &horizontal_lines), -1.0);
    }

    #[test]
    fn mirrored_side_carousels_reverse_wheel_direction() {
        let down = mouse::ScrollDelta::Lines { x: 0.0, y: -1.0 };
        assert_eq!(wheel_scroll(Quarter::Cw90, &down), -1.0);
        assert_eq!(wheel_scroll(Quarter::Ccw90, &down), 1.0);
    }

    #[test]
    fn manual_side_layout_uses_the_strip_y_axis_for_expansion_room() {
        let strip = Rectangle {
            x: 0.0,
            y: 240.0,
            width: CAROUSEL_HEIGHT,
            height: 160.0,
        };
        let viewport = Rectangle {
            x: 0.0,
            y: 0.0,
            width: 360.0,
            height: 733.0,
        };
        let mut state = CarouselState {
            label_widths: vec![64.0, 72.0, 72.0, 104.0, 88.0],
            expand_t: 1.0,
            ..Default::default()
        };

        let medium = carousel_extension(Quarter::Cw90, strip, viewport, &state);
        state.full_expand_t = 1.0;
        let full = carousel_extension(Quarter::Cw90, strip, viewport, &state);

        assert!((medium - strip.height / 4.0).abs() < 0.001);
        assert!(full > medium);
    }

    #[test]
    fn sideways_layout_reserves_medium_expansion_plus_button_gap() {
        let modes = [CameraMode::Photo, CameraMode::Video, CameraMode::Timelapse];
        let spacing = cosmic::theme::spacing();

        assert_eq!(
            sideways_sibling_spacing_for_modes(&modes, 733.0),
            carousel_width_for_modes(&modes) / 4.0 + spacing.space_s as f32
        );
    }

    #[test]
    fn compact_side_layout_uses_normal_collapsed_spacing() {
        let modes = [CameraMode::Photo, CameraMode::Video, CameraMode::Timelapse];

        assert_eq!(
            sideways_sibling_spacing_for_modes(&modes, 360.0),
            cosmic::theme::spacing().space_m as f32
        );
    }

    #[test]
    fn compact_side_buttons_move_during_medium_expansion() {
        let bounds = Rectangle {
            x: 100.0,
            y: 20.0,
            width: 160.0,
            height: 40.0,
        };
        let viewport = Rectangle {
            x: 0.0,
            y: 0.0,
            width: 360.0,
            height: 313.0,
        };
        let mut state = CarouselState {
            label_widths: vec![64.0, 72.0, 72.0, 104.0, 88.0],
            ..Default::default()
        };

        assert_eq!(
            button_slide_for_orientation(Quarter::Cw90, bounds, viewport, &state),
            0.0
        );
        state.expand_t = 1.0;
        assert!(button_slide_for_orientation(Quarter::Cw90, bounds, viewport, &state) < 0.0);
    }

    #[test]
    fn sideways_button_slide_starts_at_zero() {
        let bounds = Rectangle {
            x: 100.0,
            y: 20.0,
            width: 160.0,
            height: 40.0,
        };
        let viewport = Rectangle {
            x: 0.0,
            y: 0.0,
            width: 780.0,
            height: 313.0,
        };
        let state = CarouselState::default();

        assert_eq!(
            button_slide_for_orientation(Quarter::None, bounds, viewport, &state),
            resting_button_slide(bounds)
        );
        assert_eq!(
            button_slide_for_orientation(Quarter::Cw90, bounds, viewport, &state),
            0.0
        );
        assert_eq!(
            button_slide_for_orientation(Quarter::Ccw90, bounds, viewport, &state),
            0.0
        );
    }

    #[test]
    fn sideways_button_slide_tracks_only_stage_two_expansion() {
        let bounds = Rectangle {
            x: 100.0,
            y: 20.0,
            width: 160.0,
            height: 40.0,
        };
        let viewport = Rectangle {
            x: 0.0,
            y: 0.0,
            width: 780.0,
            height: 313.0,
        };
        let mut state = CarouselState {
            label_widths: vec![64.0, 72.0, 72.0, 104.0, 88.0],
            ..Default::default()
        };

        // Stage 1 only reveals the fade zones. Sibling controls stay put.
        state.expand_t = 1.0;
        assert_eq!(
            button_slide_for_orientation(Quarter::Cw90, bounds, viewport, &state),
            0.0
        );

        // Stage 2 moves them smoothly away from the growing carousel.
        state.full_expand_t = 0.5;
        let halfway = button_slide_for_orientation(Quarter::Cw90, bounds, viewport, &state);
        state.full_expand_t = 1.0;
        let expanded = button_slide_for_orientation(Quarter::Cw90, bounds, viewport, &state);

        assert!(halfway < 0.0);
        assert!((halfway * 2.0 - expanded).abs() < 0.001);
        assert!(expanded < halfway);

        // Collapse follows the same position curve in reverse; drag/snap does
        // not add unrelated sibling motion beyond the stage-2 extent.
        state.drag_offset = 80.0;
        state.snap_from = -40.0;
        assert_eq!(
            button_slide_for_orientation(Quarter::Ccw90, bounds, viewport, &state,),
            expanded
        );
        state.full_expand_t = 0.0;
        assert_eq!(
            button_slide_for_orientation(Quarter::Ccw90, bounds, viewport, &state,),
            0.0
        );
    }

    #[test]
    fn landscape_button_clearance_uses_y_axis_not_side_strip_x() {
        let portrait = Rectangle {
            x: 640.0,
            y: 80.0,
            width: 160.0,
            height: 40.0,
        };
        let left_strip = Rectangle {
            x: 0.0,
            y: 80.0,
            width: 40.0,
            height: 160.0,
        };
        let right_strip = Rectangle {
            x: 740.0,
            ..left_strip
        };
        let moved_down = Rectangle {
            y: 120.0,
            ..left_strip
        };

        let left = button_axis_bounds(Quarter::Cw90, left_strip, portrait);
        let right = button_axis_bounds(Quarter::Cw90, right_strip, portrait);
        let down = button_axis_bounds(Quarter::Cw90, moved_down, portrait);

        assert_eq!(left.x, right.x);
        assert_eq!(left.width, right.width);
        assert_eq!(down.x - left.x, 40.0);
    }

    #[test]
    fn ccw_remap_sends_strip_top_to_carousel_right() {
        // A 200x40 carousel rotated CCW into a 40x200 strip: a tap near the
        // strip's TOP maps to the carousel's RIGHT (end of the row). The block
        // reads the other way round after the 180° flip.
        let b = Rectangle {
            x: 0.0,
            y: 0.0,
            width: 40.0,
            height: 200.0,
        };
        let p = remap_pointer(Quarter::Ccw90, b, Point::new(20.0, 10.0));
        // portrait width (long axis) is b.height = 200; end of row is near 200.
        assert!(p.x > 180.0 && (p.y - 20.0).abs() < 1.0);
    }

    #[test]
    fn portrait_remap_is_identity() {
        // Portrait must be byte-identical: the remap changes nothing.
        let b = Rectangle {
            x: 5.0,
            y: 7.0,
            width: 200.0,
            height: 40.0,
        };
        let p = Point::new(42.0, 19.0);
        assert_eq!(remap_pointer(Quarter::None, b, p), p);
        assert_eq!(strip_point(Quarter::None, b, p), p);
        let r = Rectangle {
            x: 10.0,
            y: 11.0,
            width: 30.0,
            height: 12.0,
        };
        assert_eq!(strip_rect(Quarter::None, b, r), r);
    }

    #[test]
    fn remap_pointer_inverts_strip_point() {
        // The hit-test remap must be the exact inverse of the draw transform,
        // or taps land on the wrong label.
        let b = Rectangle {
            x: 12.0,
            y: 8.0,
            width: 40.0,
            height: 200.0,
        };
        for quarter in [Quarter::Ccw90, Quarter::Cw90] {
            // A point inside the portrait block (origin b, dims swapped).
            let portrait = Point::new(b.x + 130.0, b.y + 25.0);
            let strip = strip_point(quarter, b, portrait);
            let back = remap_pointer(quarter, b, strip);
            assert!((back.x - portrait.x).abs() < 0.001, "{quarter:?} x");
            assert!((back.y - portrait.y).abs() < 0.001, "{quarter:?} y");
        }
    }

    #[test]
    fn cw_remap_sends_strip_top_to_carousel_left() {
        // Cw90 mirrors Ccw90: a tap near the strip TOP maps to the carousel's
        // LEFT (start of the row) after the 180° flip.
        let b = Rectangle {
            x: 0.0,
            y: 0.0,
            width: 40.0,
            height: 200.0,
        };
        let p = remap_pointer(Quarter::Cw90, b, Point::new(20.0, 10.0));
        assert!(p.x < 20.0 && (p.y - 20.0).abs() < 1.0);
    }

    #[test]
    fn content_long_uses_two_narrowest_for_three_plus_modes() {
        // View, Photo, Video, Timelapse - the two narrowest (View + Photo) set
        // the stable compact length, not the total of all four.
        let widths = vec![64.0_f32, 72.0, 72.0, 104.0];
        let long = content_long(&widths);
        assert!((long - (64.0 + 72.0 + LABEL_GAP)).abs() < 0.001);
        // ≤2 modes fall back to the full strip so both labels stay visible.
        assert!((content_long(&[64.0, 72.0]) - (64.0 + 72.0 + LABEL_GAP)).abs() < 0.001);
    }

    #[test]
    fn rotated_node_is_compact_content_not_full_lane() {
        // Four modes → a compact ~138px content length down the strip.
        let long = content_long(&[64.0_f32, 72.0, 72.0, 104.0]);

        // A tall `Fill` group lane must NOT stretch the rotated node down the
        // strip: the node equals the compact content, never the lane length.
        let tall_lane = layout::Limits::new(Size::ZERO, Size::new(74.0, 2000.0));
        for quarter in [Quarter::Cw90, Quarter::Ccw90] {
            let sz = carousel_node_size(quarter, long, &tall_lane);
            assert_eq!(
                sz.width, CAROUSEL_HEIGHT,
                "{quarter:?} thickness crosses strip"
            );
            assert!(
                (sz.height - long).abs() < 0.001,
                "{quarter:?} rotated node must equal compact content ({long}), got {}",
                sz.height
            );
            assert!(
                sz.height < 200.0,
                "{quarter:?} must not inflate to the lane"
            );
        }

        // A short lane still clamps so the node can never overflow the strip.
        let short_lane = layout::Limits::new(Size::ZERO, Size::new(74.0, 100.0));
        assert!(
            (carousel_node_size(Quarter::Cw90, long, &short_lane).height - 100.0).abs() < 0.001
        );

        // Portrait is the axis-swap of the same rule: content on the width,
        // fixed thickness on the height (byte-identical to before).
        let wide_lane = layout::Limits::new(Size::ZERO, Size::new(2000.0, 74.0));
        let p = carousel_node_size(Quarter::None, long, &wide_lane);
        assert_eq!(p.height, CAROUSEL_HEIGHT);
        assert!((p.width - long).abs() < 0.001);
    }

    #[test]
    fn strip_rect_turns_the_block_axis_aligned() {
        // The full block rectangle must remap onto the strip exactly.
        let b = Rectangle {
            x: 100.0,
            y: 0.0,
            width: 40.0,
            height: 200.0,
        };
        let portrait = portrait_bounds(Quarter::Ccw90, b); // 200 wide, 40 tall
        let mapped = strip_rect(Quarter::Ccw90, b, portrait);
        assert!((mapped.x - b.x).abs() < 0.001);
        assert!((mapped.y - b.y).abs() < 0.001);
        assert!((mapped.width - b.width).abs() < 0.001);
        assert!((mapped.height - b.height).abs() < 0.001);
    }
}
