// SPDX-License-Identifier: GPL-3.0-only

//! On-screen geometry shared by the preview, the scrim and the capture path.
//!
//! Everything here is a pure function of the window size, the UI bar heights and
//! the selected aspect ratio. It deliberately sits BELOW `view` and
//! `frosted_backdrop`: the scrim tint, the frosted backdrop and the saved
//! photo's crop must all describe the same rectangles, and they do so by
//! deriving them from this module rather than from each other.

use cosmic::iced::{Rectangle, Size};

/// Fixed pixel height for the top UI bar overlay (matches native COSMIC header bar).
pub const TOP_BAR_HEIGHT: f32 = 47.0;

/// Space reserved by the UI bars on each edge, in logical px.
///
/// Portrait fills `top`/`bottom`; a sideways device fills `left`/`right`. Keeping
/// all four means the preview, the scrim, the frosted backdrop and the saved
/// photo's crop all describe the same rectangle in every orientation.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BarInsets {
    pub top: f32,
    pub bottom: f32,
    pub left: f32,
    pub right: f32,
}

impl BarInsets {
    /// Bars on the top and bottom edges (portrait).
    pub fn horizontal(top: f32, bottom: f32) -> Self {
        Self {
            top,
            bottom,
            left: 0.0,
            right: 0.0,
        }
    }

    /// Bars on the left and right edges (device held sideways).
    pub fn vertical(left: f32, right: f32) -> Self {
        Self {
            top: 0.0,
            bottom: 0.0,
            left,
            right,
        }
    }
}

/// On-screen "framed" rectangle that the canvas crop overlay highlights and
/// that the captured photo's Cover-mode crop maps to. Sharing this helper
/// between the canvas and the capture path guarantees the saved image
/// matches what the user sees inside the translucent crop bars — including
/// when the UI bars are asymmetric (top 47 px vs bottom ~174 px) and a
/// sensor-centered crop would diverge from the on-screen content area.
pub fn frame_rect_on_screen(
    screen_w: f32,
    screen_h: f32,
    insets: BarInsets,
    target_ratio: Option<f32>,
) -> Rectangle {
    let content_x = insets.left;
    let content_top = insets.top;
    let content_w = (screen_w - insets.left - insets.right).max(0.0);
    let content_h = (screen_h - insets.top - insets.bottom).max(0.0);
    let content_rect = Rectangle {
        x: content_x,
        y: content_top,
        width: content_w,
        height: content_h,
    };
    match target_ratio {
        None => content_rect,
        Some(ratio) if content_h > 0.0 && content_w > 0.0 => {
            let content_aspect = content_w / content_h;
            if ratio > content_aspect {
                let h = content_w / ratio;
                Rectangle {
                    x: content_x,
                    y: content_top + (content_h - h) / 2.0,
                    width: content_w,
                    height: h,
                }
            } else {
                let w = content_h * ratio;
                Rectangle {
                    x: content_x + (content_w - w) / 2.0,
                    y: content_top,
                    width: w,
                    height: content_h,
                }
            }
        }
        Some(_) => content_rect,
    }
}

/// Smallest bar extent [`scrim_bars`] will emit, in logical px.
///
/// The blur cannot honour a thinner bar — its chain downsamples the scissored
/// region, so a sub-pixel target is degenerate — and the tint painting half a
/// pixel the blur skipped is exactly the kind of divergence `scrim_bars` exists
/// to rule out. Dropping the bar on both surfaces costs at most half a pixel of
/// tint and keeps them describing the same set of bars.
const MIN_BAR_EXTENT: f32 = 0.5;

/// The four scrim bars — top, bottom, left, right, in that order — covering
/// everything *outside* `frame_rect`, so the framed area is exactly the target
/// ratio. Coordinates are relative to `bounds`'s own origin.
///
/// The single source of the bar geometry for BOTH surfaces that paint it: the
/// scrim's translucent tint (`OverlayBackgroundProgram`) and the live-blurred
/// backdrop underneath it (`FrostedScrim`). They are stacked one on the other,
/// so any disagreement about a bar's rectangle shows up directly as tint
/// without blur (or the reverse).
///
/// Bars below [`MIN_BAR_EXTENT`] in either dimension come back zero-sized;
/// callers skip anything with a zero width or height.
pub fn scrim_bars(bounds: Size, frame_rect: Rectangle) -> [Rectangle; 4] {
    let top_bar = frame_rect.y;
    let bottom_bar = (bounds.height - (frame_rect.y + frame_rect.height)).max(0.0);
    let left_bar = frame_rect.x;
    let right_bar = (bounds.width - (frame_rect.x + frame_rect.width)).max(0.0);
    // Side bars run between the top and bottom bars, not the full height.
    let mid_h = (bounds.height - top_bar - bottom_bar).max(0.0);

    [
        Rectangle {
            x: 0.0,
            y: 0.0,
            width: bounds.width,
            height: top_bar,
        },
        Rectangle {
            x: 0.0,
            y: bounds.height - bottom_bar,
            width: bounds.width,
            height: bottom_bar,
        },
        Rectangle {
            x: 0.0,
            y: top_bar,
            width: left_bar,
            height: mid_h,
        },
        Rectangle {
            x: bounds.width - right_bar,
            y: top_bar,
            width: right_bar,
            height: mid_h,
        },
    ]
    .map(|bar| {
        if bar.width > MIN_BAR_EXTENT && bar.height > MIN_BAR_EXTENT {
            bar
        } else {
            Rectangle {
                width: 0.0,
                height: 0.0,
                ..bar
            }
        }
    })
}

/// Map the on-screen [`frame_rect_on_screen`] to sensor coordinates via the
/// preview's Cover scaling. The result is the sensor sub-rect the user
/// actually sees in the framed area on screen — which is *not* a sensor-
/// centered crop when the UI bars are asymmetric. Capture-mode crop logic
/// uses this to keep the saved photo aligned with the on-screen framing.
///
/// `frame_w` / `frame_h` are display-oriented (rotation-swapped by the
/// caller); the returned coords are in the same space.
pub fn cover_capture_crop(
    frame_w: u32,
    frame_h: u32,
    screen_w: f32,
    screen_h: f32,
    insets: BarInsets,
    target_ratio: Option<f32>,
) -> (u32, u32, u32, u32) {
    let fw = frame_w as f32;
    let fh = frame_h as f32;
    if fw <= 0.0 || fh <= 0.0 || screen_w <= 0.0 || screen_h <= 0.0 {
        // No screen geometry yet (window hasn't reported size). Fall back
        // to no crop so we save *something* sensible.
        return (0, 0, frame_w, frame_h);
    }
    // Cover scale: scale the frame so the wider dimension just covers the
    // screen, the other overflows.
    let scale = (screen_w / fw).max(screen_h / fh);
    let scaled_x_off = (screen_w - fw * scale) / 2.0;
    let scaled_y_off = (screen_h - fh * scale) / 2.0;
    let frame_rect = frame_rect_on_screen(screen_w, screen_h, insets, target_ratio);
    // Inverse-map the on-screen frame rect back to sensor coords.
    let sx = ((frame_rect.x - scaled_x_off) / scale).max(0.0);
    let sy = ((frame_rect.y - scaled_y_off) / scale).max(0.0);
    let scw = (frame_rect.width / scale).min(fw - sx);
    let sch = (frame_rect.height / scale).min(fh - sy);
    (sx as u32, sy as u32, scw as u32, sch as u32)
}

/// Grow whichever reserved UI bar the aspect crop's letterbox lands on out to
/// the kept rect's edge, so the framed rect abuts the bar and the control-free
/// scrim slack between them collapses.
///
/// Two-pass. Pass 1 computes the letterbox `frame_rect` from the BASE insets
/// (the un-grown bar heights). Pass 2 grows each edge THAT ALREADY HAS A BAR
/// (`base extent > 0`) out to that frame edge; a bar-free edge is left
/// untouched, so a crop that letterboxes onto it keeps its scrim (portrait
/// left/right, landscape top/bottom). Asymmetric bars (top 47 vs bottom 174 -
/// or, sideways, the two unequal side bars) each grow to their OWN crop edge,
/// which keeps the kept rect centred in the content area.
///
/// Recomputing [`frame_rect_on_screen`] with the returned insets and the SAME
/// ratio yields the IDENTICAL kept rect - the expanded content area equals it
/// exactly - so every consumer (scrim, frosted backdrop, preview centring,
/// capture crop) still describes the same crop. Only the reserved-bar
/// bookkeeping changes: the bar now owns the whole region between the window
/// edge and the kept rect, instead of leaving a control-free tinted band.
///
/// `None` ratio returns `base` untouched, so the no-crop (Native) path is
/// byte-identical to the pre-expansion behaviour.
pub fn expanded_insets_for_ratio(
    base: BarInsets,
    screen_w: f32,
    screen_h: f32,
    target_ratio: Option<f32>,
) -> BarInsets {
    if target_ratio.is_none() {
        return base;
    }
    let fr = frame_rect_on_screen(screen_w, screen_h, base, target_ratio);
    // Only edges that already reserve a bar grow; bar-free edges keep their
    // scrim. `max(base)` is a floor: a bar never shrinks below its reserved
    // extent (the letterbox can only push the frame edge away from the bar,
    // never toward it, so this is defensive rather than load-bearing).
    let grow = |base_extent: f32, grown_edge: f32| {
        if base_extent > 0.0 {
            grown_edge.max(base_extent)
        } else {
            base_extent
        }
    };
    BarInsets {
        top: grow(base.top, fr.y),
        bottom: grow(base.bottom, (screen_h - (fr.y + fr.height)).max(0.0)),
        left: grow(base.left, fr.x),
        right: grow(base.right, (screen_w - (fr.x + fr.width)).max(0.0)),
    }
}

/// Map a rectangle from display-oriented frame coordinates back into the
/// original sensor frame. `rotation` is the clockwise orientation applied to
/// the sensor frame before the rectangle was calculated.
pub(crate) fn inverse_oriented_crop(
    rect: (u32, u32, u32, u32),
    sensor_w: u32,
    sensor_h: u32,
    rotation: crate::backends::camera::types::SensorRotation,
) -> (u32, u32, u32, u32) {
    use crate::backends::camera::types::SensorRotation;

    let (x, y, w, h) = rect;
    match rotation {
        SensorRotation::None => (x, y, w, h),
        SensorRotation::Rotate90 => (sensor_w.saturating_sub(y + h), x, h, w),
        SensorRotation::Rotate180 => (
            sensor_w.saturating_sub(x + w),
            sensor_h.saturating_sub(y + h),
            w,
            h,
        ),
        SensorRotation::Rotate270 => (y, sensor_h.saturating_sub(x + w), h, w),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The phone's real chrome: a 47 px header and a ~174 px control bar. The
    /// asymmetry is the whole reason this module exists, so it is the default
    /// geometry for every test below that does not say otherwise.
    const TOP: f32 = 47.0;
    const BOTTOM: f32 = 174.0;
    const W: f32 = 1080.0;
    const H: f32 = 2340.0;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    /// With no target ratio the framed rect is exactly the content area between
    /// the bars — not the window, and not a centred anything.
    #[test]
    fn frame_rect_without_a_ratio_is_the_content_area() {
        let r = frame_rect_on_screen(W, H, BarInsets::horizontal(TOP, BOTTOM), None);
        assert!(approx(r.x, 0.0) && approx(r.width, W));
        assert!(approx(r.y, TOP));
        assert!(approx(r.height, H - TOP - BOTTOM));
    }

    /// A ratio WIDER than the content area letterboxes inside it: full width,
    /// reduced height, centred vertically *within the content area* — which,
    /// with asymmetric bars, is NOT the centre of the window.
    ///
    /// The window-centre confusion is exactly the bug class here: at 16:9 the
    /// content centre sits at y = 47 + 2119/2 = 1106.5 while the window centre is
    /// 1170, so a rect centred on the wrong one lands ~63 px off — visible, and
    /// silently mis-crops the saved photo via `cover_capture_crop`.
    #[test]
    fn frame_rect_letterboxes_a_wide_ratio_inside_the_content_area() {
        let content_h = H - TOP - BOTTOM;
        let r = frame_rect_on_screen(W, H, BarInsets::horizontal(TOP, BOTTOM), Some(16.0 / 9.0));
        assert!(approx(r.width, W), "a wide ratio must keep full width");
        assert!(approx(r.height, W / (16.0 / 9.0)));
        // Centred in the CONTENT area, so equal slack above and below it.
        let slack_above = r.y - TOP;
        let slack_below = (TOP + content_h) - (r.y + r.height);
        assert!(
            approx(slack_above, slack_below),
            "the framed rect must centre in the content area (slack {slack_above:.1} \
             above vs {slack_below:.1} below), not in the window"
        );
        assert!(r.y > TOP);
    }

    /// A ratio NARROWER than the content area pillarboxes: full content height,
    /// reduced width, centred horizontally, flush against the bars.
    ///
    /// This needs a LANDSCAPE window, and that is worth stating: the phone's
    /// portrait content area is aspect 1080/2119 = 0.51, so every ratio the app
    /// actually offers (1:1, 4:3, 3:4, 16:9, 9:16 — all >= 0.5625) is *wider*
    /// than it and takes the letterbox branch above. The pillarbox branch only
    /// runs on a landscape window, so this is the test that keeps it honest.
    #[test]
    fn frame_rect_pillarboxes_a_narrow_ratio() {
        const LW: f32 = 2340.0;
        const LH: f32 = 1080.0;
        let content_h = LH - TOP - BOTTOM;
        // Content aspect here is 2340/859 = 2.72, so all of these are narrower.
        for ratio in [1.0f32, 4.0 / 3.0, 3.0 / 4.0, 16.0 / 9.0] {
            let r = frame_rect_on_screen(LW, LH, BarInsets::horizontal(TOP, BOTTOM), Some(ratio));
            assert!(approx(r.height, content_h), "ratio {ratio}");
            assert!(approx(r.width, content_h * ratio), "ratio {ratio}");
            assert!(approx(r.x, (LW - r.width) / 2.0), "ratio {ratio}");
            assert!(approx(r.y, TOP), "ratio {ratio}");
            assert!(r.width < LW, "ratio {ratio} should not span the window");
        }
    }

    /// The branch actually taken on the phone: at every ratio the app ships, on
    /// the portrait window, the framed rect spans the FULL width and letterboxes.
    ///
    /// Pins the boundary itself. The content aspect is 0.51, and 9:16 = 0.5625 is
    /// the narrowest ratio offered — only ~10% clear of the pillarbox branch. A
    /// change to the bar heights that pushed the content aspect past 0.5625 would
    /// silently flip 9:16 to pillarboxing and put crop bars on the sides of the
    /// screen where the design says they go top and bottom.
    #[test]
    fn every_shipping_ratio_letterboxes_on_the_phone() {
        for ratio in [1.0f32, 4.0 / 3.0, 3.0 / 4.0, 16.0 / 9.0, 9.0 / 16.0] {
            let r = frame_rect_on_screen(W, H, BarInsets::horizontal(TOP, BOTTOM), Some(ratio));
            assert!(
                approx(r.width, W) && approx(r.x, 0.0),
                "ratio {ratio} must span the full width on the phone's portrait \
                 window (content aspect {:.3}), got {r:?} — if this pillarboxes now, \
                 the crop bars moved to the sides of the screen",
                W / (H - TOP - BOTTOM)
            );
            assert!(r.height < H - TOP - BOTTOM);
        }
    }

    /// The framed rect never escapes the content area, on any ratio — and never
    /// hides under the chrome bars.
    #[test]
    fn frame_rect_stays_inside_the_content_area() {
        for ratio in [
            None,
            Some(1.0),
            Some(4.0 / 3.0),
            Some(3.0 / 4.0),
            Some(16.0 / 9.0),
            Some(9.0 / 16.0),
            Some(2.35),
        ] {
            let r = frame_rect_on_screen(W, H, BarInsets::horizontal(TOP, BOTTOM), ratio);
            assert!(r.x >= -0.01 && r.x + r.width <= W + 0.01, "{ratio:?}");
            assert!(
                r.y >= TOP - 0.01 && r.y + r.height <= H - BOTTOM + 0.01,
                "{ratio:?} gave {r:?}, which spills into the chrome bars"
            );
        }
    }

    /// A window with no room between the bars (or no window at all) must not
    /// produce a NaN or a negative rect — `frame_rect_on_screen` runs on the very
    /// first view build, before the window has reported a size.
    #[test]
    fn frame_rect_survives_degenerate_windows() {
        for (w, h) in [(0.0, 0.0), (W, 0.0), (0.0, H), (W, TOP + BOTTOM)] {
            for ratio in [None, Some(1.0), Some(16.0 / 9.0)] {
                let r = frame_rect_on_screen(w, h, BarInsets::horizontal(TOP, BOTTOM), ratio);
                assert!(
                    r.width >= 0.0
                        && r.height >= 0.0
                        && r.width.is_finite()
                        && r.height.is_finite(),
                    "{w}x{h} at {ratio:?} gave {r:?}"
                );
            }
        }
    }

    /// THE invariant: the four bars plus the framed rect tile `bounds` exactly —
    /// they cover every pixel outside the frame, no pixel inside it, and never
    /// each other.
    ///
    /// This is what makes the tint and the blur describable as "the same bars".
    /// A bar that overlapped the frame would double-tint (and blur) content the
    /// user is supposed to see sharp; a gap would leave a sharp, untinted seam.
    /// Checked by rasterising, because that catches an off-by-one the arithmetic
    /// would let through.
    #[test]
    fn scrim_bars_tile_everything_outside_the_frame() {
        for (w, h) in [(1080.0f32, 2340.0f32), (948.0, 586.0), (600.0, 600.0)] {
            for ratio in [None, Some(1.0), Some(4.0 / 3.0), Some(16.0 / 9.0)] {
                let bounds = Size::new(w, h);
                let frame_rect =
                    frame_rect_on_screen(w, h, BarInsets::horizontal(TOP, BOTTOM), ratio);
                let bars = scrim_bars(bounds, frame_rect);

                // Sample pixel centres over the whole window.
                let (gw, gh) = (w as u32, h as u32);
                for gy in (0..gh).step_by(7) {
                    for gx in (0..gw).step_by(7) {
                        let (px, py) = (gx as f32 + 0.5, gy as f32 + 0.5);
                        let covered = bars
                            .iter()
                            .filter(|b| b.width > 0.0 && b.height > 0.0)
                            .filter(|b| {
                                px >= b.x && px < b.x + b.width && py >= b.y && py < b.y + b.height
                            })
                            .count();
                        let in_frame = px >= frame_rect.x
                            && px < frame_rect.x + frame_rect.width
                            && py >= frame_rect.y
                            && py < frame_rect.y + frame_rect.height;
                        if in_frame {
                            assert_eq!(
                                covered, 0,
                                "{w}x{h} at {ratio:?}: ({px},{py}) is INSIDE the framed \
                                 rect but {covered} scrim bar(s) cover it — the scrim \
                                 would tint and blur content meant to stay sharp"
                            );
                        } else {
                            assert_eq!(
                                covered, 1,
                                "{w}x{h} at {ratio:?}: ({px},{py}) is outside the framed \
                                 rect and {covered} scrim bar(s) cover it — expected \
                                 exactly one (0 = a sharp untinted seam, 2 = double tint)"
                            );
                        }
                    }
                }
            }
        }
    }

    /// A sub-pixel bar comes back zero-sized, on both axes, so both surfaces drop
    /// it together.
    ///
    /// The blur chain downsamples its scissored region, so it cannot honour a
    /// half-pixel bar; the tint can. If `scrim_bars` handed the bar back, the
    /// tint would paint half a pixel the blur skipped — the exact divergence this
    /// module exists to rule out. 0.4 is below `MIN_BAR_EXTENT`, 0.6 is above.
    #[test]
    fn scrim_bars_drop_sub_pixel_bars() {
        let bounds = Size::new(100.0, 100.0);
        // Top bar 0.4 px, bottom bar 0.4 px: both below the guard.
        let bars = scrim_bars(
            bounds,
            Rectangle {
                x: 0.0,
                y: 0.4,
                width: 100.0,
                height: 99.2,
            },
        );
        assert_eq!((bars[0].width, bars[0].height), (0.0, 0.0), "top");
        assert_eq!((bars[1].width, bars[1].height), (0.0, 0.0), "bottom");

        // 0.6 px clears the guard and must survive.
        let bars = scrim_bars(
            bounds,
            Rectangle {
                x: 0.0,
                y: 0.6,
                width: 100.0,
                height: 98.8,
            },
        );
        assert!(bars[0].height > 0.0 && bars[1].height > 0.0);

        // Side bars are guarded on width, not just height.
        let bars = scrim_bars(
            bounds,
            Rectangle {
                x: 0.4,
                y: 0.0,
                width: 99.2,
                height: 100.0,
            },
        );
        assert_eq!((bars[2].width, bars[2].height), (0.0, 0.0), "left");
        assert_eq!((bars[3].width, bars[3].height), (0.0, 0.0), "right");
    }

    /// A frame that exactly fills its bounds produces no bars at all.
    #[test]
    fn scrim_bars_are_empty_when_the_frame_fills_the_window() {
        let bounds = Size::new(W, H);
        let bars = scrim_bars(
            bounds,
            Rectangle {
                x: 0.0,
                y: 0.0,
                width: W,
                height: H,
            },
        );
        for bar in bars {
            assert_eq!((bar.width, bar.height), (0.0, 0.0));
        }
    }

    /// With ASYMMETRIC chrome bars the capture crop is deliberately NOT
    /// sensor-centred — it follows what is on screen.
    ///
    /// This is the reason `cover_capture_crop` exists rather than a plain centred
    /// crop. The content area's centre sits above the window's (47 px of chrome
    /// above vs 174 below), so the sensor rect the user sees is above the sensor's
    /// own centre. A centred crop would save a photo shifted DOWN against the
    /// preview — the classic "the photo isn't what I framed" bug.
    #[test]
    fn cover_capture_crop_follows_the_screen_not_the_sensor_centre() {
        const FW: u32 = 1280;
        const FH: u32 = 960;
        let (_, y, cw, ch) =
            cover_capture_crop(FW, FH, W, H, BarInsets::horizontal(TOP, BOTTOM), Some(1.0));
        assert!(cw > 0 && ch > 0);

        let crop_centre_y = y as f32 + ch as f32 / 2.0;
        let sensor_centre_y = FH as f32 / 2.0;
        assert!(
            crop_centre_y < sensor_centre_y - 1.0,
            "with 47 px of chrome above and 174 below, the framed area sits ABOVE \
             the window centre, so the captured crop must sit above the sensor \
             centre too — got crop centre {crop_centre_y:.1} vs sensor centre \
             {sensor_centre_y:.1}. Equal means the crop went back to being \
             sensor-centred and the saved photo no longer matches the preview."
        );
    }

    /// With SYMMETRIC bars the same call *is* sensor-centred — so the asymmetry
    /// above is really the bars talking, not an unconditional offset.
    #[test]
    fn cover_capture_crop_is_centred_when_the_bars_are() {
        const FW: u32 = 1280;
        const FH: u32 = 960;
        let (_, y, _, ch) =
            cover_capture_crop(FW, FH, W, H, BarInsets::horizontal(100.0, 100.0), Some(1.0));
        let crop_centre_y = y as f32 + ch as f32 / 2.0;
        assert!(
            (crop_centre_y - FH as f32 / 2.0).abs() < 1.5,
            "symmetric bars must give a sensor-centred crop, got centre \
             {crop_centre_y:.1} vs {:.1}",
            FH as f32 / 2.0
        );
    }

    /// The crop never leaves the sensor, on any ratio or window — the result
    /// indexes a real buffer, so an out-of-bounds rect is a crash or a garbled
    /// save rather than a wrong framing.
    #[test]
    fn cover_capture_crop_stays_inside_the_sensor() {
        for (fw, fh) in [(1280u32, 960u32), (2592, 1940), (960, 1280)] {
            for (sw, sh) in [(W, H), (586.0, 948.0), (1000.0, 1000.0)] {
                for ratio in [None, Some(1.0), Some(4.0 / 3.0), Some(16.0 / 9.0)] {
                    let (x, y, cw, ch) = cover_capture_crop(
                        fw,
                        fh,
                        sw,
                        sh,
                        BarInsets::horizontal(TOP, BOTTOM),
                        ratio,
                    );
                    assert!(
                        x + cw <= fw && y + ch <= fh,
                        "{fw}x{fh} sensor, {sw}x{sh} screen, {ratio:?}: crop \
                         ({x},{y},{cw},{ch}) runs off the sensor"
                    );
                }
            }
        }
    }

    /// No screen geometry yet (the window has not reported a size) must fall back
    /// to the whole frame, not to a zero-sized or NaN crop — this runs on the
    /// capture path, so the fallback decides whether a photo saves at all.
    #[test]
    fn cover_capture_crop_falls_back_when_there_is_no_geometry() {
        for (fw, fh, sw, sh) in [
            (1280u32, 960u32, 0.0f32, 0.0f32),
            (1280, 960, W, 0.0),
            (0, 0, W, H),
        ] {
            assert_eq!(
                cover_capture_crop(
                    fw,
                    fh,
                    sw,
                    sh,
                    BarInsets::horizontal(TOP, BOTTOM),
                    Some(1.0)
                ),
                (0, 0, fw, fh),
                "{fw}x{fh} frame with a {sw}x{sh} screen must fall back to no crop"
            );
        }
    }

    /// Pins FINDING C1: the capture path must use the SENSOR rotation
    /// composed with the DISPLAY orientation - exactly what
    /// `AppModel::capture_rotation` computes and what
    /// `preview_video_config` (`camera_preview/widget.rs`) already does for
    /// the live preview - not the sensor rotation alone.
    ///
    /// Scenario: an unrotated sensor (1280x960, `SensorRotation::None`)
    /// behind a phone the user is holding sideways
    /// (`DisplayOrientation::Rotate90`). The window is re-laid out to landscape
    /// when held sideways; the capture-rotation composition (sensor + display
    /// orientation) keeps the saved crop matching the on-screen framing,
    /// independent of what the compositor does to the window.
    ///
    /// Every capture call site does: swap `(fw, fh)` by
    /// `rotation.swaps_dimensions()`, feed them to `cover_capture_crop`
    /// together with the on-screen insets, then swap the result back by the
    /// same `rotation` on the way out. If `rotation` is sensor-only here
    /// (`None`), that whole pipeline runs UNSWAPPED - wrong Cover scale
    /// *and* no swap-back - which transposes the final sensor-space crop:
    /// its aspect ratio comes out as the exact RECIPROCAL of what the
    /// display-orientation-aware (combined) rotation produces. That
    /// reciprocal relationship is what this test pins.
    #[test]
    fn capture_geometry_rotation_composes_sensor_and_display_orientation_for_the_crop() {
        use crate::app::state::AppModel;
        use crate::backends::camera::types::SensorRotation;
        use crate::backends::display_orientation::DisplayOrientation;

        let m = AppModel {
            display_orientation: DisplayOrientation::Rotate90,
            screen_width: 1080.0,
            screen_height: 2340.0,
            ..AppModel::default()
        };

        // Sensor has no rotation of its own - isolates the bug to the
        // display-orientation term rather than mixing in a sensor mount
        // angle too.
        assert_eq!(m.current_camera_rotation(), SensorRotation::None);

        // The combined rotation must fold in the display transform: 0
        // (sensor) + 90 (display) = 90, not 0.
        let combined = m.capture_geometry_rotation();
        assert_eq!(
            combined,
            SensorRotation::Rotate90,
            "capture_geometry_rotation() must compose sensor (None) with display \
             (Rotate90); got {combined:?} - if this reads None, capture geometry \
             regressed to sensor-only rotation"
        );

        const SENSOR_W: u32 = 1280;
        const SENSOR_H: u32 = 960;
        let target_ratio = Some(16.0f32 / 9.0);
        let insets = BarInsets::default();

        // Reproduces exactly the capture-site pattern in `capture.rs`
        // (`capture_photo_with_frame`, `capture_photo_from_raw_stream`,
        // `handle_burst_mode_frames_collected`): swap dims by whichever
        // rotation is passed in, crop, then swap the result back.
        let final_crop_wh = |rotation: SensorRotation| -> (u32, u32) {
            let (fw, fh) = if rotation.swaps_dimensions() {
                (SENSOR_H, SENSOR_W)
            } else {
                (SENSOR_W, SENSOR_H)
            };
            let (x, y, w, h) = cover_capture_crop(
                fw,
                fh,
                m.screen_width,
                m.screen_height,
                insets,
                target_ratio,
            );
            if rotation.swaps_dimensions() {
                let _ = (x, y); // position not under test here, only shape
                (h, w)
            } else {
                (w, h)
            }
        };

        let (correct_w, correct_h) = final_crop_wh(combined);
        let (buggy_w, buggy_h) = final_crop_wh(m.current_camera_rotation());

        let correct_ratio = correct_w as f32 / correct_h as f32;
        let buggy_ratio = buggy_w as f32 / buggy_h as f32;

        assert!(
            (correct_ratio - 9.0 / 16.0).abs() < 0.01,
            "the display-orientation-aware crop must come out ~9:16 in sensor \
             coordinates (the 16:9 target rotated 90°), got {correct_w}x{correct_h} \
             = {correct_ratio:.3}"
        );
        assert!(
            (buggy_ratio - correct_ratio).abs() > 0.5,
            "sensor-only rotation must NOT reproduce the display-orientation-aware \
             crop shape - got buggy {buggy_w}x{buggy_h} ({buggy_ratio:.3}) vs correct \
             {correct_w}x{correct_h} ({correct_ratio:.3}). If these match (or the \
             buggy ratio also comes out ~9:16), a capture call site regressed to \
             sensor-only rotation (FINDING C1) and this test should have caught it."
        );
    }

    #[test]
    fn inverse_oriented_crop_restores_asymmetric_sensor_rect_for_every_rotation() {
        use crate::backends::camera::types::SensorRotation;

        const SENSOR_W: u32 = 1280;
        const SENSOR_H: u32 = 960;

        assert_eq!(
            inverse_oriented_crop(
                (100, 20, 700, 1100),
                SENSOR_W,
                SENSOR_H,
                SensorRotation::Rotate90,
            ),
            (160, 100, 1100, 700),
        );
        assert_eq!(
            inverse_oriented_crop(
                (100, 20, 700, 1100),
                SENSOR_W,
                SENSOR_H,
                SensorRotation::Rotate270,
            ),
            (20, 160, 1100, 700),
        );
        assert_eq!(
            inverse_oriented_crop(
                (100, 20, 700, 600),
                SENSOR_W,
                SENSOR_H,
                SensorRotation::Rotate180,
            ),
            (480, 340, 700, 600),
        );
        assert_eq!(
            inverse_oriented_crop(
                (100, 20, 700, 600),
                SENSOR_W,
                SENSOR_H,
                SensorRotation::None,
            ),
            (100, 20, 700, 600),
        );
    }

    #[test]
    fn horizontal_insets_match_the_old_two_height_behaviour() {
        // Regression guard: portrait must be identical to the pre-refactor result.
        const W: f32 = 1080.0;
        const H: f32 = 2280.0;
        const TOP: f32 = 47.0;
        const BOTTOM: f32 = 174.0;
        let r = frame_rect_on_screen(W, H, BarInsets::horizontal(TOP, BOTTOM), None);
        assert_eq!(r.x, 0.0);
        assert_eq!(r.y, TOP);
        assert_eq!(r.width, W);
        assert_eq!(r.height, H - TOP - BOTTOM);
    }

    #[test]
    fn vertical_insets_reserve_left_and_right() {
        const W: f32 = 1080.0;
        const H: f32 = 2280.0;
        const LEFT: f32 = 47.0;
        const RIGHT: f32 = 174.0;
        let r = frame_rect_on_screen(W, H, BarInsets::vertical(LEFT, RIGHT), None);
        assert_eq!(r.x, LEFT);
        assert_eq!(r.y, 0.0);
        assert_eq!(r.width, W - LEFT - RIGHT);
        assert_eq!(r.height, H);
    }

    #[test]
    fn vertical_insets_centre_a_ratio_fit_within_the_content_rect() {
        // Content box is 859x2280; a 16:9 target is far wider than that box is,
        // so the fit is width-limited and centred vertically inside the content.
        const W: f32 = 1080.0;
        const H: f32 = 2280.0;
        const LEFT: f32 = 47.0;
        const RIGHT: f32 = 174.0;
        let r = frame_rect_on_screen(W, H, BarInsets::vertical(LEFT, RIGHT), Some(16.0 / 9.0));
        let content_w = W - LEFT - RIGHT;
        assert_eq!(r.x, LEFT);
        assert_eq!(r.width, content_w);
        let expected_h = content_w / (16.0 / 9.0);
        assert!((r.height - expected_h).abs() < 0.01, "height {}", r.height);
        // Centred in the full height because top/bottom insets are zero.
        assert!((r.y - (H - expected_h) / 2.0).abs() < 0.01, "y {}", r.y);
    }

    #[test]
    fn zero_insets_give_the_whole_screen() {
        let r = frame_rect_on_screen(100.0, 200.0, BarInsets::default(), None);
        assert_eq!(r.x, 0.0);
        assert_eq!(r.y, 0.0);
        assert_eq!(r.width, 100.0);
        assert_eq!(r.height, 200.0);
    }

    /// THE task's invariant, portrait: a 4:3 crop letterboxes onto the top and
    /// bottom bars, so those bars grow to the kept rect and no control-free
    /// scrim band is left between the bar and the frame.
    ///
    /// Adapted from the brief's sketch: that sketch expected `scrim_bars` to
    /// come back zero-height, but `scrim_bars` deliberately tiles the FULL
    /// window from y = 0 (the transparent chrome containers rely on it for
    /// their tinted background - see `overlay_style`). So the collapse we
    /// assert is the meaningful one: the scrim's top/bottom bars now coincide
    /// with the reserved insets, i.e. NOTHING tinted extends BEYOND the bar.
    #[test]
    fn letterbox_expands_the_bar_and_drops_the_separate_scrim() {
        let base = BarInsets::horizontal(47.0, 174.0);
        let ratio = Some(0.75);
        let (sw, sh) = (360.0, 780.0);

        // The un-expanded letterbox: full width, centred in the content area,
        // leaving control-free slack between the 47 px bar and the kept rect.
        let base_fr = frame_rect_on_screen(sw, sh, base, ratio);
        assert!(
            base_fr.y > 47.0 + 1.0,
            "sanity: the letterbox must leave slack above the 47 px bar, got y {}",
            base_fr.y
        );

        let insets = expanded_insets_for_ratio(base, sw, sh, ratio);
        let fr = frame_rect_on_screen(sw, sh, insets, ratio);

        // The bar grew exactly to the kept rect's edge, on both bar edges.
        assert!(
            (insets.top - base_fr.y).abs() < 0.01,
            "top grew to letterbox edge"
        );
        assert!(
            (insets.top - fr.y).abs() < 0.01,
            "kept rect abuts the grown bar"
        );
        let base_bottom_edge = sh - (base_fr.y + base_fr.height);
        assert!(
            (insets.bottom - base_bottom_edge).abs() < 0.01,
            "bottom grew too"
        );
        // Bar-free side edges are untouched.
        assert_eq!(insets.left, 0.0);
        assert_eq!(insets.right, 0.0);

        // The crop RESULT is unchanged: same kept rect as the base pass.
        assert!((fr.x - base_fr.x).abs() < 0.01 && (fr.width - base_fr.width).abs() < 0.01);
        assert!((fr.y - base_fr.y).abs() < 0.01 && (fr.height - base_fr.height).abs() < 0.01);

        // The separate scrim collapses: the scrim's top/bottom bars coincide
        // with the reserved insets, so no tint extends beyond the bar.
        let bars = scrim_bars(Size::new(sw, sh), fr);
        let top_beyond_bar = bars[0].height - insets.top;
        let bottom_beyond_bar = bars[1].height - insets.bottom;
        assert!(
            top_beyond_bar.abs() < MIN_BAR_EXTENT,
            "scrim beyond top bar must collapse, got {top_beyond_bar}"
        );
        assert!(
            bottom_beyond_bar.abs() < MIN_BAR_EXTENT,
            "scrim beyond bottom bar must collapse, got {bottom_beyond_bar}"
        );
    }

    /// Landscape pillarbox: the bars sit on the two SIDES (47 vs 174, unequal),
    /// and a narrower-than-content crop pillarboxes onto them. Each side bar
    /// grows to its OWN crop edge - asymmetrically - and the kept rect stays
    /// centred in the content area (and unchanged).
    #[test]
    fn pillarbox_expands_the_side_bars_asymmetrically() {
        const LEFT: f32 = 47.0;
        const RIGHT: f32 = 174.0;
        let base = BarInsets::vertical(LEFT, RIGHT);
        let (sw, sh) = (2340.0, 1080.0);
        let ratio = Some(1.0); // 1:1 is far narrower than the wide content area

        let base_fr = frame_rect_on_screen(sw, sh, base, ratio);
        assert!(
            base_fr.x > LEFT + 1.0,
            "sanity: pillarbox leaves slack past the left bar"
        );

        let insets = expanded_insets_for_ratio(base, sw, sh, ratio);
        let fr = frame_rect_on_screen(sw, sh, insets, ratio);

        // Side bars grew to their frame edges...
        assert!(
            (insets.left - base_fr.x).abs() < 0.01,
            "left grew to frame edge"
        );
        let base_right_edge = sw - (base_fr.x + base_fr.width);
        assert!(
            (insets.right - base_right_edge).abs() < 0.01,
            "right grew to frame edge"
        );
        // ...and unequally, preserving the 174 - 47 asymmetry.
        assert!(
            (insets.right - insets.left - (RIGHT - LEFT)).abs() < 0.01,
            "side bars keep their base asymmetry: left {}, right {}",
            insets.left,
            insets.right
        );
        // Top/bottom stay bar-free.
        assert_eq!(insets.top, 0.0);
        assert_eq!(insets.bottom, 0.0);

        // Kept rect unchanged and still centred in the content area.
        assert!((fr.x - base_fr.x).abs() < 0.01 && (fr.width - base_fr.width).abs() < 0.01);
        let content_left = insets.left;
        let content_right = sw - insets.right;
        let slack_l = fr.x - content_left;
        let slack_r = content_right - (fr.x + fr.width);
        assert!(
            (slack_l - slack_r).abs() < 0.01,
            "kept rect centred in content area"
        );
    }

    /// Native (no crop) is a no-op: the insets come back exactly as given, so
    /// every downstream path stays byte-identical to today. Also pins that a
    /// crop landing on a BAR-FREE edge does not grow a bar there.
    #[test]
    fn expansion_is_a_no_op_without_a_crop_and_on_bar_free_edges() {
        let base = BarInsets::horizontal(47.0, 174.0);
        // No ratio → untouched.
        assert_eq!(expanded_insets_for_ratio(base, 360.0, 780.0, None), base);

        // A crop that letterboxes on the top/bottom bars must NOT invent side
        // bars where there were none.
        let insets = expanded_insets_for_ratio(base, 360.0, 780.0, Some(0.75));
        assert_eq!(insets.left, 0.0);
        assert_eq!(insets.right, 0.0);

        // Portrait window, a pillarbox-narrow crop lands on the bar-free
        // left/right edges: those keep their scrim (stay 0), and the top/bottom
        // bars - which the full-height pillarbox abuts exactly - are unchanged.
        let narrow = expanded_insets_for_ratio(base, 780.0, 360.0, Some(0.3));
        let fr = frame_rect_on_screen(780.0, 360.0, base, Some(0.3));
        assert_eq!(narrow.left, 0.0, "no bar on the left edge to grow");
        assert_eq!(narrow.right, 0.0, "no bar on the right edge to grow");
        assert!((narrow.top - fr.y).abs() < 0.01);
        assert!((narrow.bottom - (360.0 - (fr.y + fr.height))).abs() < 0.01);
    }
}
