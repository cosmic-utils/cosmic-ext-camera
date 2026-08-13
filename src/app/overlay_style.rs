// SPDX-License-Identifier: GPL-3.0-only

//! Styling and backing surfaces for the overlay chrome that floats over the
//! live preview: the scrim bars, the panel tints, and the live-blurred
//! ("frosted glass") backdrops painted behind them.
//!
//! Sits below `view` and beside `frosted_backdrop`, which owns the widgets that
//! actually paint the blur; this module decides *what* colour and *which*
//! rectangles they get.

use crate::app::preview_geometry::{BarInsets, TOP_BAR_HEIGHT, frame_rect_on_screen, scrim_bars};
use crate::app::state::{AppModel, CameraMode, Message};
use crate::config::OverlayEffect;
use crate::constants::ui::{OVERLAY_BACKGROUND_ALPHA, POPUP_BACKGROUND_ALPHA};
use cosmic::Element;
use cosmic::iced::{Background, Color, Length};
use cosmic::widget;
use std::sync::atomic::{AtomicU8, Ordering};

/// The user's [`OverlayEffect`] as an index into [`OverlayEffect::ALL`].
///
/// A process global, which is a real cost — it is not reachable from a test's
/// `AppModel`, and two tests that write it cannot run concurrently. It is also
/// unavoidable: the colour roots below ([`overlay_surface`] and its callers,
/// [`PanelStyle::container_style`] and [`overlay_chip_button_class`]) are free
/// functions that iced calls at DRAW time with nothing but a `&cosmic::Theme`.
/// There is no
/// `&self` on that path, so `AppModel::config` cannot be threaded in without
/// re-plumbing every widget style closure in the app to carry the setting.
///
/// Written by `handle_set_overlay_effect` and by [`init_overlay_effect`] at
/// startup; read on every draw. `Relaxed` is sufficient — a draw that races a
/// settings change renders either the old or the new surface, and the next
/// frame (which the settings change itself triggers) settles it.
static OVERLAY_EFFECT: AtomicU8 = AtomicU8::new(0);

/// Publish `effect` to the draw-time colour roots. Call on every change to
/// `Config::overlay_effect`, including the initial load.
pub fn init_overlay_effect(effect: OverlayEffect) {
    let index = OverlayEffect::ALL
        .iter()
        .position(|e| *e == effect)
        .unwrap_or(0);
    OVERLAY_EFFECT.store(index as u8, Ordering::Relaxed);
}

/// The user's configured overlay effect.
fn overlay_effect() -> OverlayEffect {
    OverlayEffect::ALL
        .get(OVERLAY_EFFECT.load(Ordering::Relaxed) as usize)
        .copied()
        .unwrap_or_default()
}

/// How overlay chrome paints itself over the live preview.
///
/// Three states, not a bool: "translucent" has to be distinguishable from
/// "frosted" (backdrop or not) *and* from "opaque" (tint or not), and no single
/// flag says both.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OverlaySurface {
    /// Translucent panels over a live-blurred copy of the preview.
    Frosted,
    /// Flat translucent tint straight over the sharp preview. No blur.
    Translucent,
    /// Solid panels. No blur.
    Opaque,
}

impl OverlaySurface {
    /// Whether this surface wants the blurred backdrop painted behind it.
    /// The two blur-free surfaces take the early return in [`AppModel::frosted_panel`]
    /// and [`AppModel::frosted_bars`], so they never build a primitive and the
    /// dual-Kawase chain never schedules.
    fn is_frosted(self) -> bool {
        self == Self::Frosted
    }

    /// Background colour for overlay chrome that floats over the live preview
    /// (bottom-bar scrim, chips, pickers, indicators, dropdowns).
    ///
    /// - Frosted → the theme's *transparent* container, whose alpha tracks the
    ///   blur-strength `alpha_map`, so the panel reveals the blurred preview
    ///   backdrop and matches the desktop's frosted design language.
    /// - Translucent → the theme's *opaque* container with `translucent_alpha`
    ///   bolted on. Deliberately NOT the transparent container: its `alpha_map`
    ///   alpha is tuned to sit over a blur, and over a sharp preview it reads as
    ///   a smear. This hardcoded alpha is what the app shipped before the
    ///   frosted backdrop existed.
    /// - Opaque → the theme's *opaque* container, so panels are solid instead of
    ///   permanently see-through.
    fn bg_color(self, theme: &cosmic::Theme, translucent_alpha: f32) -> Color {
        let cosmic = theme.cosmic();
        match self {
            Self::Frosted => Color::from(cosmic.background(true).base),
            Self::Translucent => {
                let bg = cosmic.bg_color();
                Color::from_rgba(bg.red, bg.green, bg.blue, translucent_alpha)
            }
            Self::Opaque => Color::from(cosmic.background(false).base),
        }
    }

    /// Legible foreground colour for text/icons drawn on the overlay scrim.
    ///
    /// The container `on` colour is contrast-boosted by the theme for the
    /// frosted case, so text stays readable over the blurred preview.
    fn on_color(self, theme: &cosmic::Theme) -> Color {
        Color::from(theme.cosmic().background(self.is_frosted()).on)
    }
}

/// Which surface the overlay chrome should render as.
///
/// The user's [`OverlayEffect`] wins outright; only `System` consults the
/// desktop, following COSMIC's window-frosting setting via the widget theme's
/// `transparent` flag (set by the runtime when window frosting is active — the
/// toggle the user flips in Appearance settings).
///
/// Off-COSMIC there is no such flag, and `OverlayEffect::effective_for` has
/// already rewritten a stored `System` to `Translucent` by the time we match on
/// it — the flat off-COSMIC baseline: those desktops have no frosting setting to
/// read and no compositor blur to lean on, so a plain translucent tint is the
/// sensible default there.
///
/// Note the off-COSMIC case diverges from libcosmic's own widgets: they resolve
/// `Layer::Background` through `Theme::transparent`, which is `false` there, so
/// a cosmic widget nested in one of our panels styles itself opaque while the
/// panel around it is translucent. Harmless in practice — our panels are the
/// only surface that actually paints a background over the preview — but it is
/// why we do not simply read `theme.transparent` everywhere.
pub fn overlay_surface(theme: &cosmic::Theme) -> OverlaySurface {
    resolve_surface(
        overlay_effect(),
        crate::config::is_cosmic_desktop(),
        theme.transparent,
    )
}

/// Whether the app's own WINDOW background should be see-through, so that the
/// compositor's frosted glass — the blurred wallpaper cosmic-comp already paints
/// behind us — reads through the parts of the window the camera image does not
/// cover.
///
/// This is a different question from [`overlay_surface`], and the two must not be
/// conflated. That one asks how our chrome should paint itself over the *live
/// preview*, and the blur behind it is one we render ourselves. This one asks
/// whether there is a blur behind the *window* to reveal at all — a blur only
/// cosmic-comp can draw, and only for a surface it has been asked to blur.
///
/// `Theme::transparent` is exactly that signal: libcosmic sets it to
/// `blur_enabled && Core::frosted(theme)` (see `app/cosmic.rs`), i.e. "this
/// window really has compositor blur attached". Nothing else will do — deciding
/// from [`OverlayEffect`] alone would punch a see-through hole in the window on
/// every desktop that has no blur to put behind it, showing the raw wallpaper
/// (and whatever windows are stacked below) instead of frosted glass.
///
/// The one veto is [`OverlayEffect::Off`], the app's "no see-through chrome"
/// escape hatch. A user who has turned the effect off and still gets a
/// see-through *window* would rightly call that a bug, so `Off` keeps the whole
/// window opaque. Every other effect leaves the window to the compositor: the
/// window background is the desktop's business, not the overlay chrome's.
pub fn window_is_frosted(theme: &cosmic::Theme) -> bool {
    resolve_window_frosted(overlay_effect().effective(), theme.transparent)
}

/// [`window_is_frosted`] as a pure function of its two inputs — `theme.transparent`
/// comes from the compositor and cannot be varied from a test.
fn resolve_window_frosted(effect: OverlayEffect, theme_transparent: bool) -> bool {
    theme_transparent && effect != OverlayEffect::Off
}

/// The colour of the app's own window background: the letterbox around the
/// camera image, the placeholders before the first frame, and the root container
/// behind everything.
///
/// Translucent when [`window_is_frosted`] — the theme's own transparent-container
/// colour, whose alpha tracks COSMIC's frost-thickness `alpha_map`, so our window
/// is exactly as see-through as every other COSMIC window at that setting.
/// Opaque otherwise. Sourced through `Theme::background(bool)`, which is the same
/// call libcosmic's own `ApplicationExt::view_main` makes for its content
/// container, so we frost by joining the desktop's convention rather than by
/// inventing an alpha.
///
/// This is the ONLY layer that paints the window background. Everything that
/// used to paint its own opaque copy in the letterbox now stands aside for it —
/// see `content_rect_uv` in `video_primitive`, which cuts the letterbox out of
/// the frosted backdrop, and the `cover_blend` factor on the scrim tint in
/// [`AppModel::build_crop_overlay`]. Two translucent copies of the same colour
/// stacked would compose to nearly opaque and undo the effect.
pub fn window_bg_color(theme: &cosmic::Theme) -> Color {
    Color::from(theme.cosmic().background(window_is_frosted(theme)).base)
}

/// [`window_bg_color`] as a `widget::container::style` argument.
pub fn window_bg_style(theme: &cosmic::Theme) -> widget::container::Style {
    widget::container::Style {
        background: Some(Background::Color(window_bg_color(theme))),
        ..Default::default()
    }
}

/// The [`overlay_surface`] decision as a pure function of its three inputs.
///
/// Split out because `is_cosmic_desktop()` caches in a `LazyLock` and
/// `theme.transparent` comes from the compositor: neither can be varied from a
/// test, so the matrix is only reachable here.
fn resolve_surface(
    effect: OverlayEffect,
    is_cosmic: bool,
    theme_transparent: bool,
) -> OverlaySurface {
    match effect.effective_for(is_cosmic) {
        OverlayEffect::Frosted => OverlaySurface::Frosted,
        OverlayEffect::Translucent => OverlaySurface::Translucent,
        OverlayEffect::Off => OverlaySurface::Opaque,
        // Only reachable on COSMIC: `effective_for` rewrote it otherwise.
        // Frosting on → frosted; frosting off → translucent (never opaque, which
        // the user must pick deliberately via `Off`).
        OverlayEffect::System => {
            if theme_transparent {
                OverlaySurface::Frosted
            } else {
                OverlaySurface::Translucent
            }
        }
    }
}

/// An overlay panel's container style together with the corner radius that
/// style rounds its tint to.
///
/// The pairing is the whole point. [`AppModel::frosted_panel`] needs the radius
/// as a concrete number at build time — the blur chain cuts the rounded
/// silhouette itself, from a radius the primitive carries — and a blur rounded
/// LESS than the tint spills a hard-edged blurred wedge outside the panel.
/// Passing the style and the radius as two independent arguments let a call
/// site pair one panel's style with another panel's radius and ship exactly
/// that artifact, with nothing but a naming convention against it. Here a
/// single `radii` selector feeds both [`Self::container_style`] and
/// [`Self::corner_radius`], so there is no second value left to mispair.
#[derive(Clone, Copy)]
pub struct PanelStyle {
    /// The theme corner-radius family the tint rounds to.
    radii: fn(&cosmic::cosmic_theme::CornerRadii) -> [f32; 4],
    /// Whether the panel sets `text_color` to the contrast-boosted `on` colour.
    tinted_text: bool,
    /// Tint alpha under [`OverlaySurface::Translucent`]. Ignored by the other
    /// two surfaces, which take their alpha from the theme's container.
    translucent_alpha: f32,
}

/// Style for a floating picker panel's background (exposure, motor/PTZ, format
/// pickers, the tools menu).
///
/// Uses `radius_s` (slightly rounded) as the maximum roundness, so panels stay
/// square or slightly rounded even when the theme is set to "round".
/// Does not set text_color, to let buttons use their native COSMIC theme colors.
pub const PICKER_PANEL: PanelStyle = PanelStyle {
    radii: |radii| radii.radius_s,
    tinted_text: false,
    translucent_alpha: OVERLAY_BACKGROUND_ALPHA,
};

/// Style for a centred popup panel (permission popups, the HDR+ progress
/// overlay).
///
/// Uses `radius_m`, and unlike [`PICKER_PANEL`] sets `text_color`: these panels
/// are labels rather than button hosts, and their text sits directly on frosted
/// glass, where the contrast-boosted `on` colour is what keeps it readable
/// through the blurred preview.
pub const POPUP_PANEL: PanelStyle = PanelStyle {
    radii: |radii| radii.radius_m,
    tinted_text: true,
    translucent_alpha: POPUP_BACKGROUND_ALPHA,
};

/// Style with the frosted-glass overlay background for overlay elements.
///
/// Uses `radius_xl` to match COSMIC button corner radius (follows round/slightly
/// round/square theme setting). Does not set text_color to allow buttons to use
/// their native COSMIC theme colors.
pub const OVERLAY_CONTAINER: PanelStyle = PanelStyle {
    radii: |radii| radii.radius_xl,
    tinted_text: false,
    translucent_alpha: OVERLAY_BACKGROUND_ALPHA,
};

impl PanelStyle {
    /// Resolve the container style against `theme`. Called at DRAW time with the
    /// theme in force then.
    pub fn container_style(self, theme: &cosmic::Theme) -> widget::container::Style {
        let cosmic = theme.cosmic();
        let surface = overlay_surface(theme);
        widget::container::Style {
            background: Some(Background::Color(
                surface.bg_color(theme, self.translucent_alpha),
            )),
            border: cosmic::iced::Border {
                radius: (self.radii)(&cosmic.corner_radii).into(),
                ..Default::default()
            },
            text_color: self.tinted_text.then(|| surface.on_color(theme)),
            // Symbolic icons inside a panel read this, so every picker, chip and
            // indicator floating over the preview gets the same accent glyphs as
            // the title bar. Buttons that set their own `icon_color` (the accent
            // *fill* of `Button::Suggested`, the dimmed disabled states) still
            // win over it, so toggle states stay legible.
            icon_color: Some(chrome_accent_color(theme)),
            ..Default::default()
        }
    }

    /// This panel's style as a `widget::container::style` argument.
    pub fn style(self) -> impl Fn(&cosmic::Theme) -> widget::container::Style + 'static {
        move |theme| self.container_style(theme)
    }

    /// The single radius the frosted backdrop must round to, taken from the same
    /// `corner_radii` family [`Self::container_style`] writes into
    /// `border.radius`.
    ///
    /// The frosted primitive only supports ONE uniform radius, while a
    /// container style can specify four. Every style here is symmetric, so the
    /// four values normally collapse to one. If a theme ever hands us asymmetric
    /// corners we cannot render them faithfully, so we pick the MAXIMUM of the
    /// four and accept a bounded artifact:
    ///
    /// - Too LARGE a radius rounds the blur more tightly than the tint, so a
    ///   small crescent inside a squarer corner is tinted but unblurred.
    ///   The tint still covers it, so it stays within the panel outline.
    /// - Too SMALL a radius rounds the blur less than the tint, so the blur
    ///   spills OUTSIDE the tint's rounded corner: a hard-edged blurred wedge
    ///   floating over the sharp preview with no tint on top of it.
    ///
    /// The spill is a visible hard edge outside the panel; the crescent is a
    /// subtle softness change inside it. The max is therefore the safer pick.
    fn corner_radius(self) -> f32 {
        let radii = (self.radii)(&cosmic::theme::active().cosmic().corner_radii);
        let max = radii.iter().copied().fold(f32::MIN, f32::max);
        debug_assert!(
            radii.iter().all(|r| (r - radii[0]).abs() < 0.01),
            "frosted_panel: asymmetric corner radii {radii:?} cannot be rendered \
             faithfully by the frosted backdrop (single uniform radius); \
             falling back to the maximum ({max})"
        );
        max
    }
}

/// Foreground colour for the app's own window chrome — the custom title bar and
/// the chips, panels and pickers that float over the preview.
///
/// COSMIC paints header-bar icons in the accent colour, and this app draws its
/// entire chrome itself: the title bar row, the fit/fill and zoom chips, the
/// tools menu and the pickers all sit on the same floating surfaces, so they all
/// take the same colour rather than each picking its own (issue #565).
///
/// `accent_text_color` rather than the raw accent: the theme contrast-adjusts it
/// for foreground use, which is what keeps it legible on the overlay scrim.
pub fn chrome_accent_color(theme: &cosmic::Theme) -> Color {
    Color::from(theme.cosmic().accent_text_color())
}

/// Button class for chips that sit on the translucent overlay scrim:
/// transparent background (the surrounding [`OVERLAY_CONTAINER`] provides
/// the colour) with [`chrome_accent_color`] text/icon, matching the title bar
/// above them. Avoids `Button::Text`, whose own background and hover tint don't
/// match the scrim.
pub fn overlay_chip_button_class() -> cosmic::theme::Button {
    use cosmic::widget::button::Style;
    let plain = |theme: &cosmic::Theme| -> Style {
        let accent = chrome_accent_color(theme);
        Style {
            text_color: Some(accent),
            icon_color: Some(accent),
            ..Style::new()
        }
    };
    let with_overlay = |theme: &cosmic::Theme, alpha: f32| -> Style {
        let cosmic = theme.cosmic();
        // Match the scrim: source the hover/press tint from the same container
        // variant (opaque vs. transparent) the surrounding panel uses. The tint
        // stays `on`-derived — a wash of the accent behind accent glyphs would
        // wipe out the contrast the hover state is meant to add.
        let on = overlay_surface(theme).on_color(theme);
        let accent = chrome_accent_color(theme);
        Style {
            background: Some(Background::Color(Color::from_rgba(on.r, on.g, on.b, alpha))),
            // Match the wrapper container's corner radius so the hover/press
            // overlay rounds with the chip instead of drawing a sharp box.
            border_radius: cosmic.corner_radii.radius_xl.into(),
            text_color: Some(accent),
            icon_color: Some(accent),
            ..Style::new()
        }
    };
    cosmic::theme::Button::Custom {
        active: Box::new(move |_focused, theme| plain(theme)),
        disabled: Box::new(move |theme| {
            let mut s = plain(theme);
            if let Some(ref mut c) = s.text_color {
                c.a *= 0.5;
            }
            if let Some(ref mut c) = s.icon_color {
                c.a *= 0.5;
            }
            s
        }),
        hovered: Box::new(move |_focused, theme| with_overlay(theme, 0.08)),
        pressed: Box::new(move |_focused, theme| with_overlay(theme, 0.16)),
    }
}

impl AppModel {
    /// Wrap an overlay panel's `content` so that, when frosted glass is active,
    /// a live-blurred copy of the preview is painted BEHIND the panel's
    /// translucent tint (real frosted glass). When frosting is off, the panel is
    /// returned as a plain styled (opaque) container with no backdrop.
    ///
    /// The backdrop uses the SAME preview transforms and is scissored to the
    /// panel rect, so the blurred slice lines up with the sharp preview behind
    /// the panel. If no frame is available the backdrop is skipped and the
    /// translucent-only panel is returned (never crashes).
    ///
    /// `panel` carries both the tint and the radius the blur must round to (see
    /// [`PanelStyle`]), so the two cannot disagree.
    pub(crate) fn frosted_panel<'a>(
        &self,
        content: Element<'a, Message>,
        panel: PanelStyle,
    ) -> Element<'a, Message> {
        if !overlay_surface(&cosmic::theme::active()).is_frosted() {
            return widget::container(content).style(panel.style()).into();
        }

        let styled = widget::container(content).style(panel.style());

        // Backdrop needs a live frame + matching preview config; fall back to the
        // translucent-only panel otherwise (never crashes).
        match (
            self.current_frame.as_ref(),
            self.preview_video_config(
                crate::app::video_primitive::VIDEO_ID_FROSTED,
                self.preview_transforms(),
            ),
        ) {
            (Some(frame), Some(config)) => crate::app::frosted_backdrop::FrostedContainer::new(
                styled.into(),
                frame,
                &config,
                panel.corner_radius(),
            )
            .into(),
            _ => styled.into(),
        }
    }

    /// Aspect-ratio crop target for the scrim bars (Photo mode only), shared by
    /// the scrim tint (`build_crop_overlay`) and the frosted backdrop
    /// (`frosted_bars`) so both derive the exact same four bars.
    ///
    /// Keyed on the **animated** `cover_blend()`, not on the settled
    /// `preview_fit_to_view` target. The preview's own crop region eases toward
    /// the full texture with `cover_blend` (see `video_shader.wgsl`), so gating
    /// on the target flag would drop the bars the instant the user taps
    /// fit/fill while the preview underneath was still animating — the bars
    /// vanished at the *start* of a Fill→Fit transition instead of its end.
    /// `cover_blend()` only reaches exactly 0.0 on the animation's final tick
    /// (ease-out cubic hits 1.0 at t = 1) and stays there once settled, so the
    /// bars now persist for the whole Fill→Fit animation and disappear as it
    /// completes. Fit→Fill is unchanged in spirit: the blend leaves 0.0 on the
    /// first tick, so the bars come back as the transition starts.
    pub(in crate::app) fn crop_target_ratio(&self) -> Option<f32> {
        if self.cover_blend() > 0.0
            && self.mode == CameraMode::Photo
            && !self.current_frame_is_file_source
        {
            // Display-oriented ratio so the crop bars match the rotated preview
            // on portrait windows (e.g. a "2:1" pref → 1:2 portrait region).
            self.photo_aspect_ratio
                .display_ratio(self.screen_is_portrait())
        } else {
            None
        }
    }

    /// Live frosted backdrop for the crop/letterbox scrim bars.
    ///
    /// Returns a full-window overlay layer that paints a live-blurred copy of the
    /// preview clipped to the SAME four bars the scrim tints (top, bottom, and —
    /// on aspect-cropped ratios — left/right). Meant to be stacked directly ABOVE
    /// the sharp `camera_layer` and BELOW the scrim (`build_crop_overlay`), so the
    /// scrim's translucent tint composites over the blur. Returns an empty
    /// `Space` when frosting is off or no frame is available (so the caller can
    /// unconditionally stack it).
    pub(in crate::app) fn frosted_bars(&self) -> Element<'_, Message> {
        let empty = || -> Element<'_, Message> {
            widget::Space::new()
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        };

        if !overlay_surface(&cosmic::theme::active()).is_frosted() {
            return empty();
        }

        let (Some(frame), Some(config)) = (
            self.current_frame.as_ref(),
            self.preview_video_config(
                crate::app::video_primitive::VIDEO_ID_FROSTED,
                self.preview_transforms(),
            ),
        ) else {
            return empty();
        };

        crate::app::frosted_backdrop::FrostedScrim::new(
            frame,
            &config,
            self.crop_target_ratio(),
            self.bar_insets(),
        )
        .into()
    }

    /// Build the translucent overlay background canvas.
    /// Draws crop framing bars when an aspect ratio is selected (Photo mode only).
    pub(in crate::app) fn build_crop_overlay(&self) -> Element<'_, Message> {
        // In fit-to-view mode, the frame is letterboxed — no crop bars needed, just default UI bars.
        // In Cover mode with an aspect ratio, draw crop bars. Shared with the
        // frosted backdrop so both cover exactly the same bars.
        let target_ratio = self.crop_target_ratio();

        let theme = cosmic::theme::active();
        // Scrim colour follows the overlay effect, on the same alpha scale as
        // the chrome it sits under (hence OVERLAY_BACKGROUND_ALPHA, not the
        // popup's).
        let base = overlay_surface(&theme).bg_color(&theme, OVERLAY_BACKGROUND_ALPHA);
        // Derive the scrim alpha from the animated top-bar height so it
        // fades in/out with the Photo↔View transition without needing its
        // own animation channel. **Invariant**: this only behaves
        // correctly because `settled_top_ui_height()` is binary today —
        // either 0 (View) or `TOP_BAR_HEIGHT` (every other mode). If a
        // future mode picks an intermediate top height, the alpha will
        // settle at a fractional value and look permanently dimmed. In
        // that case promote `scrim_alpha` to its own `FitFrom` channel.
        //
        // Deliberately reads `top_ui_height()` - the raw bar height, still
        // binary 0/`TOP_BAR_HEIGHT` - NOT `bar_insets().top`, which an aspect
        // crop grows past `TOP_BAR_HEIGHT` (see `expanded_insets_for_ratio`).
        // The fade is a chrome signal, the inset is a geometry one; routing the
        // grown inset here would push `alpha_t` above 1 and clamp the fade to a
        // hard on/off, so the two channels stay separate.
        let top_h = self.top_ui_height();
        let alpha_t = (top_h / TOP_BAR_HEIGHT).clamp(0.0, 1.0);
        // ...and fade it out with the fit blend, because in **Fit** the bars are
        // by construction exactly the letterbox (see `frosted_backdrop`'s module
        // docs: Contain shrinks the image into `viewport.y - bar_top -
        // bar_bottom`, so the bar regions are precisely where the image is not).
        // A scrim exists to separate chrome from the live preview; over the
        // letterbox there is no preview to separate from, only the window
        // background — and painting a second translucent copy of that colour on
        // top of [`window_bg_color`] composes the two to nearly opaque, so the
        // bars would read as solid slabs against frosted side letterbox.
        //
        // This is invisible while the window background is opaque, which is why
        // it is unconditional: the tint and the background carry the SAME RGB
        // (both `bg_color()`), so over an opaque background dropping the tint
        // changes nothing at all. Pinned by
        // `scrim_tint_over_an_opaque_window_is_a_no_op`.
        let fit_t = self.cover_blend().clamp(0.0, 1.0);
        let overlay_color = Color::from_rgba(base.r, base.g, base.b, base.a * alpha_t * fit_t);

        cosmic::widget::canvas(OverlayBackgroundProgram {
            target_ratio,
            overlay_color,
            insets: self.bar_insets(),
        })
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
    }
}

/// Canvas program that draws translucent top/bottom bars for UI backgrounds and crop framing.
/// This is the single source of truth for all translucent overlays — the top bar and bottom
/// controls containers have transparent backgrounds and rely on this canvas.
struct OverlayBackgroundProgram {
    /// Target aspect ratio (width / height), or None for no crop framing
    target_ratio: Option<f32>,
    /// Translucent overlay color
    overlay_color: Color,
    /// Space the UI bars reserve on each edge, honouring the display
    /// transform (see `AppModel::bar_insets`).
    insets: BarInsets,
}

impl cosmic::widget::canvas::Program<Message, cosmic::Theme> for OverlayBackgroundProgram {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &cosmic::Renderer,
        _theme: &cosmic::Theme,
        bounds: cosmic::iced::Rectangle,
        _cursor: cosmic::iced::mouse::Cursor,
    ) -> Vec<cosmic::widget::canvas::Geometry<cosmic::Renderer>> {
        let mut frame = cosmic::widget::canvas::Frame::new(renderer, bounds.size());

        // The framed rect is shared with the capture path
        // (`cover_capture_crop`) so the saved photo matches what's
        // visible inside the crop bars — including when the UI bars are
        // asymmetric and a sensor-centered crop would diverge from the
        // on-screen content area.
        let frame_rect =
            frame_rect_on_screen(bounds.width, bounds.height, self.insets, self.target_ratio);

        for bar in scrim_bars(bounds.size(), frame_rect) {
            if bar.width <= 0.0 || bar.height <= 0.0 {
                continue;
            }
            frame.fill_rectangle(bar.position(), bar.size(), self.overlay_color);
        }

        vec![frame.into_geometry()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The full override matrix: 4 effects x COSMIC on/off x `theme.transparent`.
    ///
    /// Pins two things at once — that the three explicit effects ignore the
    /// environment entirely, and that `System` alone reads it (COSMIC → frosting
    /// on/off maps to frosted/translucent; off-COSMIC → translucent, the flat
    /// baseline).
    #[test]
    fn resolve_surface_covers_every_effect_and_environment() {
        use OverlaySurface::{Frosted, Opaque, Translucent};
        // (effect, is_cosmic, theme.transparent) -> surface
        let cases = [
            // System is the only row that reads the environment: on COSMIC,
            // frosting on → frosted, frosting off → translucent (not opaque).
            (OverlayEffect::System, true, true, Frosted),
            (OverlayEffect::System, true, false, Translucent),
            // ...and off-COSMIC it is translucent either way: there is no desktop
            // flag to follow, so `transparent` must not leak in.
            (OverlayEffect::System, false, true, Translucent),
            (OverlayEffect::System, false, false, Translucent),
            // The other three pin one surface regardless.
            (OverlayEffect::Frosted, true, true, Frosted),
            (OverlayEffect::Frosted, true, false, Frosted),
            (OverlayEffect::Frosted, false, true, Frosted),
            (OverlayEffect::Frosted, false, false, Frosted),
            (OverlayEffect::Translucent, true, true, Translucent),
            (OverlayEffect::Translucent, true, false, Translucent),
            (OverlayEffect::Translucent, false, true, Translucent),
            (OverlayEffect::Translucent, false, false, Translucent),
            (OverlayEffect::Off, true, true, Opaque),
            (OverlayEffect::Off, true, false, Opaque),
            (OverlayEffect::Off, false, true, Opaque),
            (OverlayEffect::Off, false, false, Opaque),
        ];

        // Exhaustive over ALL: a new variant must be added here or this fails.
        for effect in OverlayEffect::ALL {
            assert!(
                cases.iter().any(|(e, ..)| *e == effect),
                "{effect:?} is not covered by the matrix"
            );
        }

        for (effect, is_cosmic, transparent, want) in cases {
            assert_eq!(
                resolve_surface(effect, is_cosmic, transparent),
                want,
                "{effect:?} on is_cosmic={is_cosmic} transparent={transparent}"
            );
        }
    }

    /// The window background follows the COMPOSITOR, not the overlay effect.
    ///
    /// `theme.transparent` is the only signal that cosmic-comp has actually
    /// attached blur to our surface, and it is a hard requirement: without a blur
    /// behind the window, a see-through background is not frosted glass, it is a
    /// hole showing the raw wallpaper and whatever is stacked below. So no effect
    /// — not even `Frosted`, which the user can force on any desktop — may turn
    /// the window see-through on its own.
    ///
    /// `Off` is the one veto in the other direction: it is the app's "no
    /// see-through chrome" switch, and a see-through window would contradict it.
    #[test]
    fn the_window_frosts_only_when_the_compositor_says_so() {
        for effect in OverlayEffect::ALL {
            assert!(
                !resolve_window_frosted(effect, false),
                "{effect:?} made the window see-through with no compositor blur \
                 behind it — that is a hole in the window, not frosted glass"
            );
        }
        for effect in [
            OverlayEffect::System,
            OverlayEffect::Frosted,
            OverlayEffect::Translucent,
        ] {
            assert!(
                resolve_window_frosted(effect, true),
                "{effect:?} must leave the window background to the compositor"
            );
        }
        assert!(
            !resolve_window_frosted(OverlayEffect::Off, true),
            "'Off' is the app's no-see-through-chrome switch; it must keep the \
             window opaque too"
        );
    }

    /// The window background is the theme's own container colour, picked through
    /// the same `Theme::background(bool)` call libcosmic makes for its content
    /// container — so our frosting is the desktop's, at the desktop's own
    /// frost-thickness alpha, rather than an alpha we invented.
    #[test]
    fn the_frosted_window_background_is_the_themes_transparent_container() {
        for theme in [cosmic::Theme::dark(), cosmic::Theme::light()] {
            let opaque = Color::from(theme.cosmic().background(false).base);
            let frosted = Color::from(theme.cosmic().background(true).base);
            assert_eq!(
                opaque.a, 1.0,
                "the un-frosted window background must be fully opaque"
            );
            // The two carry the same RGB and differ only in alpha — which is what
            // makes dropping the scrim tint over the letterbox invisible while the
            // window is opaque (see the test below).
            assert_eq!(
                (opaque.r, opaque.g, opaque.b),
                (frosted.r, frosted.g, frosted.b)
            );
        }
    }

    /// Fading the scrim tint out with the fit blend is a no-op while the window
    /// background is opaque — which is why it can be unconditional.
    ///
    /// In Fit the scrim bars ARE the letterbox, so the tint sits on the window
    /// background rather than on live preview. Both carry `bg_color()`'s RGB, so
    /// over an opaque background any alpha composites to the same colour and the
    /// fade cannot be seen. Over a *translucent* one it is the difference between
    /// a frosted bar and a nearly solid slab, since two stacked copies of the same
    /// translucent colour compose to `1-(1-a)^2`.
    #[test]
    fn scrim_tint_over_an_opaque_window_is_a_no_op() {
        for theme in [cosmic::Theme::dark(), cosmic::Theme::light()] {
            let window = Color::from(theme.cosmic().background(false).base);
            for surface in [
                OverlaySurface::Frosted,
                OverlaySurface::Translucent,
                OverlaySurface::Opaque,
            ] {
                let tint = surface.bg_color(&theme, OVERLAY_BACKGROUND_ALPHA);
                assert_eq!(
                    (tint.r, tint.g, tint.b),
                    (window.r, window.g, window.b),
                    "{surface:?}: the scrim tint and the window background must \
                     share an RGB, or fading the tint out over the letterbox would \
                     be a visible colour change rather than the no-op it is meant \
                     to be"
                );
            }
            // And the stacking really is the problem worth avoiding: two copies of
            // the frosted alpha compose to nearly opaque.
            let a = Color::from(theme.cosmic().background(true).base).a;
            assert!(
                1.0 - (1.0 - a).powi(2) > a + 0.1,
                "stacking two translucent copies at alpha {a} must be materially \
                 more opaque than one, otherwise this fade has no reason to exist"
            );
        }
    }

    /// A config written on COSMIC, then read on a desktop that never offers
    /// `System`. It must land on Translucent — the off-COSMIC default — and must
    /// never fall through to opaque or blank chrome.
    #[test]
    fn system_effect_stored_off_cosmic_resolves_to_translucent() {
        for transparent in [true, false] {
            assert_eq!(
                resolve_surface(OverlayEffect::System, false, transparent),
                OverlaySurface::Translucent,
                "System off-COSMIC (transparent={transparent})"
            );
        }
    }

    /// Only `Frosted` may schedule the blur. `is_frosted()` is the exact gate on
    /// the early returns in `frosted_panel` / `frosted_bars`, so a false here is
    /// a `FrostedContainer`/`FrostedScrim` that never gets built and a
    /// dual-Kawase chain that never runs.
    #[test]
    fn only_the_frosted_surface_paints_a_backdrop() {
        assert!(OverlaySurface::Frosted.is_frosted());
        assert!(!OverlaySurface::Translucent.is_frosted());
        assert!(!OverlaySurface::Opaque.is_frosted());
    }

    /// Every effect that resolves away from `Frosted` costs zero GPU, in every
    /// environment.
    #[test]
    fn translucent_and_off_never_blur_in_any_environment() {
        for is_cosmic in [true, false] {
            for transparent in [true, false] {
                for effect in [OverlayEffect::Translucent, OverlayEffect::Off] {
                    assert!(
                        !resolve_surface(effect, is_cosmic, transparent).is_frosted(),
                        "{effect:?} scheduled a blur on is_cosmic={is_cosmic} \
                         transparent={transparent}"
                    );
                }
            }
        }
    }

    /// Translucent must reproduce what the app shipped before the frosted
    /// backdrop: the OPAQUE background colour with a hardcoded alpha bolted on.
    ///
    /// What this actually pins is the ALPHA. In the stock themes
    /// `background.base` and `transparent_background.base` carry the same RGB
    /// and differ only in alpha, so sourcing the RGB from the wrong container is
    /// not observable here — the regression that IS observable, and the one that
    /// matters, is inheriting the transparent container's `alpha_map` alpha,
    /// which is tuned to sit over a blur and smears over a sharp preview.
    #[test]
    fn translucent_uses_the_opaque_bg_color_with_a_hardcoded_alpha() {
        for theme in [cosmic::Theme::dark(), cosmic::Theme::light()] {
            let bg = theme.cosmic().bg_color();

            let picker = OverlaySurface::Translucent.bg_color(&theme, OVERLAY_BACKGROUND_ALPHA);
            assert_eq!(
                picker,
                Color::from_rgba(bg.red, bg.green, bg.blue, 0.7),
                "translucent chrome must be bg_color() @ 0.7"
            );

            let popup = OverlaySurface::Translucent.bg_color(&theme, POPUP_BACKGROUND_ALPHA);
            assert_eq!(
                popup,
                Color::from_rgba(bg.red, bg.green, bg.blue, 0.95),
                "translucent popups must be bg_color() @ 0.95"
            );

            // The theme's own alphas must not reach the translucent surface:
            // neither the frosted container's, nor the opaque one's.
            let frosted = OverlaySurface::Frosted.bg_color(&theme, OVERLAY_BACKGROUND_ALPHA);
            let opaque = OverlaySurface::Opaque.bg_color(&theme, OVERLAY_BACKGROUND_ALPHA);
            assert_ne!(
                picker.a, frosted.a,
                "translucent inherited the alpha_map alpha"
            );
            assert_ne!(picker.a, opaque.a, "translucent is not translucent");
        }
    }

    /// The alphas each panel hands `bg_color`, matching what the panels were
    /// before the frosted backdrop landed: popups near-opaque, chrome at 0.7.
    #[test]
    fn panel_styles_carry_the_pre_frosted_translucent_alphas() {
        assert_eq!(PICKER_PANEL.translucent_alpha, OVERLAY_BACKGROUND_ALPHA);
        assert_eq!(
            OVERLAY_CONTAINER.translucent_alpha,
            OVERLAY_BACKGROUND_ALPHA
        );
        assert_eq!(POPUP_PANEL.translucent_alpha, POPUP_BACKGROUND_ALPHA);
    }

    /// Frosted/Opaque ignore `translucent_alpha` — only the Translucent surface
    /// reads it, so a popup and a chip agree everywhere else.
    #[test]
    fn only_translucent_reads_the_panel_alpha() {
        let theme = cosmic::Theme::dark();
        for surface in [OverlaySurface::Frosted, OverlaySurface::Opaque] {
            assert_eq!(
                surface.bg_color(&theme, OVERLAY_BACKGROUND_ALPHA),
                surface.bg_color(&theme, POPUP_BACKGROUND_ALPHA),
                "{surface:?} must not vary with translucent_alpha"
            );
        }
    }

    /// The global is index-based in both directions; a mismatch would silently
    /// map the dropdown's selection onto the wrong effect.
    #[test]
    fn overlay_effect_global_round_trips_every_variant() {
        // Serialised against the other global-touching test: this is a process
        // global, so two writers cannot interleave.
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());

        for effect in OverlayEffect::ALL {
            init_overlay_effect(effect);
            assert_eq!(overlay_effect(), effect);
        }

        init_overlay_effect(OverlayEffect::default());
    }

    /// The default follows COSMIC where there is a flag to follow, and is a flat
    /// translucent tint elsewhere.
    ///
    /// `is_cosmic_desktop()` caches in a `LazyLock`, so the test can only assert
    /// against whichever environment it happens to run in; the branch it does
    /// not take is covered by
    /// `default_effect_is_system_on_cosmic_translucent_elsewhere` in `config`,
    /// which tests the pure form.
    #[test]
    fn default_effect_is_system_on_cosmic_translucent_elsewhere() {
        let want = if crate::config::is_cosmic_desktop() {
            OverlayEffect::System
        } else {
            OverlayEffect::Translucent
        };
        assert_eq!(OverlayEffect::default(), want);
        assert_eq!(crate::config::Config::default().overlay_effect, want);
    }

    /// Whatever the default is, it must resolve to the intended surface:
    /// translucent off-COSMIC, and on COSMIC frosted when frosting is on and
    /// translucent when it is off. Opaque is never a default — only `Off` reaches
    /// it, and only by the user picking it.
    #[test]
    fn default_effect_resolves_to_the_intended_surface() {
        for transparent in [true, false] {
            // Off-COSMIC the default is translucent, regardless of `transparent`.
            assert_eq!(
                resolve_surface(OverlayEffect::Translucent, false, transparent),
                OverlaySurface::Translucent
            );
            // On COSMIC `System` maps frosting on/off to frosted/translucent.
            assert_eq!(
                resolve_surface(OverlayEffect::System, true, transparent),
                if transparent {
                    OverlaySurface::Frosted
                } else {
                    OverlaySurface::Translucent
                }
            );
        }
    }
}
