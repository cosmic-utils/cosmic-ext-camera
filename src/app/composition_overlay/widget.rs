// SPDX-License-Identifier: GPL-3.0-only

//! Canvas-based composition guide overlay widget

use crate::app::preview_geometry::BarInsets;
use crate::app::state::Message;
use crate::config::CompositionGuide;
use cosmic::iced::{Color, Length, Point, Rectangle};
use cosmic::widget::canvas;

/// Semi-transparent white, 40% opacity
const LINE_COLOR: Color = Color::from_rgba(1.0, 1.0, 1.0, 0.4);
const LINE_WIDTH: f32 = 1.5;
const PHI: f32 = 1.618_034;

/// Inputs needed to compute the visible-video rectangle at draw time. The
/// rect depends on the canvas bounds (only known in `draw`), so we pass the
/// state-dependent values through and resolve everything once we have them.
struct GuideProgram {
    guide: CompositionGuide,
    /// Sensor frame dimensions after applying rotation (so they describe the
    /// preview's *displayed* orientation).
    rotated_sensor_w: f32,
    rotated_sensor_h: f32,
    /// Photo aspect-ratio crop, if any (e.g. 4:3, 1:1). `None` means Native
    /// — no crop, the visible area follows the screen / sensor aspect.
    aspect_crop_ratio: Option<f32>,
    /// Cover/Contain blend: 0.0 = Contain (letterboxed inside content area),
    /// 1.0 = Cover (fills the canvas), interpolated during fit-animation.
    cover_blend: f32,
    /// Four-sided UI reservations shared with the live preview.
    insets: BarInsets,
}

impl GuideProgram {
    /// Visible-video rectangle at the Cover endpoint, matching the area
    /// the capture path actually crops to:
    ///
    /// - **Native** aspect: the content area between the top UI bar and the
    ///   bottom-bar scrim. The video texture covers the full screen, but the
    ///   bits behind the UI bars aren't captured.
    /// - **Non-Native** aspect (e.g. 4:3, 1:1): the aspect ratio inscribed in
    ///   the content area between the UI bars — the same rectangle the
    ///   canvas crop overlay (`OverlayBackgroundProgram`) frames, so the
    ///   guide lines align with the translucent crop bars exactly.
    fn cover_rect(&self, bounds: Rectangle) -> Rectangle {
        let content_x = bounds.x + self.insets.left;
        let content_y = bounds.y + self.insets.top;
        let content_h = (bounds.height - self.insets.top - self.insets.bottom).max(0.0);
        let content_w = (bounds.width - self.insets.left - self.insets.right).max(0.0);
        let content_rect = Rectangle {
            x: content_x,
            y: content_y,
            width: content_w,
            height: content_h,
        };
        match self.aspect_crop_ratio {
            None => content_rect,
            Some(ratio) if content_h > 0.0 && content_w > 0.0 => {
                let content_aspect = content_w / content_h;
                if ratio > content_aspect {
                    // Letterbox top/bottom inside the content area.
                    let h = content_w / ratio;
                    Rectangle {
                        x: content_x,
                        y: content_y + (content_h - h) / 2.0,
                        width: content_w,
                        height: h,
                    }
                } else {
                    // Pillarbox the sides inside the content area.
                    let w = content_h * ratio;
                    Rectangle {
                        x: content_x + (content_w - w) / 2.0,
                        y: content_y,
                        width: w,
                        height: content_h,
                    }
                }
            }
            Some(_) => content_rect,
        }
    }

    /// Visible-video rectangle at the Contain endpoint. The image is
    /// letterboxed inside the content area between the top UI bar and the
    /// bottom-bar scrim. The aspect-crop ratio (when set) drives the
    /// letterbox; otherwise the rotated sensor aspect does.
    fn contain_rect(&self, bounds: Rectangle) -> Rectangle {
        let content_x = bounds.x + self.insets.left;
        let content_y = bounds.y + self.insets.top;
        let content_h = (bounds.height - self.insets.top - self.insets.bottom).max(0.0);
        let content_w = (bounds.width - self.insets.left - self.insets.right).max(0.0);
        if content_h <= 0.0 || content_w <= 0.0 || self.rotated_sensor_h <= 0.0 {
            return Rectangle {
                x: content_x,
                y: content_y,
                width: 0.0,
                height: 0.0,
            };
        }
        let frame_aspect = self
            .aspect_crop_ratio
            .unwrap_or(self.rotated_sensor_w / self.rotated_sensor_h);
        let content_aspect = content_w / content_h;
        if frame_aspect > content_aspect {
            let h = content_w / frame_aspect;
            Rectangle {
                x: content_x,
                y: content_y + (content_h - h) / 2.0,
                width: content_w,
                height: h,
            }
        } else {
            let w = content_h * frame_aspect;
            Rectangle {
                x: content_x + (content_w - w) / 2.0,
                y: content_y,
                width: w,
                height: content_h,
            }
        }
    }

    /// Linearly interpolate the visible-video rectangle by `cover_blend` so
    /// the guide tracks the preview through a fit/fill animation.
    fn visible_rect(&self, bounds: Rectangle) -> Rectangle {
        let cover = self.cover_rect(bounds);
        let contain = self.contain_rect(bounds);
        let t = self.cover_blend.clamp(0.0, 1.0);
        Rectangle {
            x: contain.x + (cover.x - contain.x) * t,
            y: contain.y + (cover.y - contain.y) * t,
            width: contain.width + (cover.width - contain.width) * t,
            height: contain.height + (cover.height - contain.height) * t,
        }
    }
}

impl canvas::Program<Message, cosmic::Theme> for GuideProgram {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &cosmic::Renderer,
        _theme: &cosmic::Theme,
        bounds: Rectangle,
        _cursor: cosmic::iced::mouse::Cursor,
    ) -> Vec<canvas::Geometry<cosmic::Renderer>> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());

        let stroke = canvas::Stroke::default()
            .with_color(LINE_COLOR)
            .with_width(LINE_WIDTH);

        let vb = self.visible_rect(Rectangle {
            x: 0.0,
            y: 0.0,
            width: bounds.width,
            height: bounds.height,
        });
        if vb.width <= 0.0 || vb.height <= 0.0 {
            return vec![frame.into_geometry()];
        }

        frame.with_clip(vb, |frame| match self.guide {
            CompositionGuide::RuleOfThirds => {
                draw_grid_lines(frame, vb, 1.0 / 3.0, 2.0 / 3.0, stroke);
            }
            CompositionGuide::PhiGrid => {
                draw_grid_lines(frame, vb, 0.382, 0.618, stroke);
            }
            CompositionGuide::SpiralTopLeft => {
                draw_fibonacci_spiral(frame, vb, false, false, stroke);
            }
            CompositionGuide::SpiralTopRight => {
                draw_fibonacci_spiral(frame, vb, true, false, stroke);
            }
            CompositionGuide::SpiralBottomLeft => {
                draw_fibonacci_spiral(frame, vb, false, true, stroke);
            }
            CompositionGuide::SpiralBottomRight => {
                draw_fibonacci_spiral(frame, vb, true, true, stroke);
            }
            CompositionGuide::Diagonals => {
                stroke_line(frame, vb.x, vb.y, vb.x + vb.width, vb.y + vb.height, stroke);
                stroke_line(frame, vb.x + vb.width, vb.y, vb.x, vb.y + vb.height, stroke);
            }
            CompositionGuide::Crosshair => {
                let mx = vb.x + vb.width / 2.0;
                let my = vb.y + vb.height / 2.0;
                stroke_line(frame, vb.x, my, vb.x + vb.width, my, stroke);
                stroke_line(frame, mx, vb.y, mx, vb.y + vb.height, stroke);
            }
            CompositionGuide::None => {}
        });

        vec![frame.into_geometry()]
    }
}

/// Stroke a single line segment.
fn stroke_line(
    frame: &mut canvas::Frame<cosmic::Renderer>,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    stroke: canvas::Stroke<'_>,
) {
    let path = canvas::path::Path::line(Point::new(x1, y1), Point::new(x2, y2));
    frame.stroke(&path, stroke);
}

/// Draw 2 horizontal + 2 vertical grid lines at the given fractions.
fn draw_grid_lines(
    frame: &mut canvas::Frame<cosmic::Renderer>,
    vb: Rectangle,
    frac1: f32,
    frac2: f32,
    stroke: canvas::Stroke<'_>,
) {
    for frac in [frac1, frac2] {
        let y = vb.y + vb.height * frac;
        let x = vb.x + vb.width * frac;
        stroke_line(frame, vb.x, y, vb.x + vb.width, y, stroke);
        stroke_line(frame, x, vb.y, x, vb.y + vb.height, stroke);
    }
}

/// Draw a golden spiral with subdivision lines.
///
/// Computed inside a true golden rectangle (aspect ratio φ) inscribed within the
/// video bounds, so the subdivision never degenerates for arbitrary aspect ratios.
///
/// `flip_x`/`flip_y` mirror around the video center to target any corner.
///
/// Screen angles: 0 = right, π/2 = down, π = left, 3π/2 = up.
fn draw_fibonacci_spiral(
    frame: &mut canvas::Frame<cosmic::Renderer>,
    vb: Rectangle,
    flip_x: bool,
    flip_y: bool,
    stroke: canvas::Stroke<'_>,
) {
    use std::f32::consts::{FRAC_PI_2, PI, TAU};

    // Transform a point, optionally mirroring around the video center.
    let pt = |px: f32, py: f32| -> Point {
        Point::new(
            if flip_x {
                vb.x + vb.width - (px - vb.x)
            } else {
                px
            },
            if flip_y {
                vb.y + vb.height - (py - vb.y)
            } else {
                py
            },
        )
    };

    // Mirror an angle for flipped axes.
    let flip_angle = |a: f32| -> f32 {
        match (flip_x, flip_y) {
            (false, false) => a,
            (true, false) => PI - a,
            (false, true) => TAU - a,
            (true, true) => a + PI,
        }
    };

    // Inscribe a golden rectangle (aspect ratio φ) within the video bounds.
    let (mut rx, mut ry, mut rw, mut rh) = if vb.width / vb.height >= PHI {
        let gw = vb.height * PHI;
        (vb.x + (vb.width - gw) / 2.0, vb.y, gw, vb.height)
    } else {
        let gh = vb.width / PHI;
        (vb.x, vb.y + (vb.height - gh) / 2.0, vb.width, gh)
    };

    // Direction cycle: right → bottom → left → top.
    // Each step cuts a square, draws the subdivision line and a quarter-circle arc.
    // Arc center sits at the corner of the square adjacent to the remainder.
    for i in 0..16 {
        let sq = rw.min(rh);
        if sq < 1.0 {
            break;
        }

        let (lx1, ly1, lx2, ly2, cx, cy, angle) = match i % 4 {
            0 => {
                // Cut from RIGHT — vertical line
                let x = rx + rw - sq;
                rw -= sq;
                (x, ry, x, ry + rh, x, ry, 0.0_f32)
            }
            1 => {
                // Cut from BOTTOM — horizontal line
                let y = ry + rh - sq;
                rh -= sq;
                (rx, y, rx + rw, y, rx + sq, y, FRAC_PI_2)
            }
            2 => {
                // Cut from LEFT — vertical line
                let x = rx + sq;
                let cy = ry + sq;
                rx += sq;
                rw -= sq;
                (x, ry, x, ry + rh, x, cy, PI)
            }
            3 => {
                // Cut from TOP — horizontal line
                let y = ry + sq;
                let cx = rx;
                ry += sq;
                rh -= sq;
                (rx, y, rx + rw, y, cx, y, 3.0 * FRAC_PI_2)
            }
            _ => unreachable!(),
        };

        // Subdivision line
        let p1 = pt(lx1, ly1);
        let p2 = pt(lx2, ly2);
        stroke_line(frame, p1.x, p1.y, p2.x, p2.y, stroke);

        // Quarter-circle arc
        let center = pt(cx, cy);
        let mut sa = flip_angle(angle);
        let mut ea = flip_angle(angle + FRAC_PI_2);
        // Ensure positive sweep ≤ π (the short quarter-circle path)
        if ea < sa {
            std::mem::swap(&mut sa, &mut ea);
        }
        if ea - sa > PI {
            ea -= TAU;
            std::mem::swap(&mut sa, &mut ea);
        }

        let path = canvas::path::Path::new(|b| {
            b.arc(canvas::path::Arc {
                center,
                radius: sq,
                start_angle: cosmic::iced::Radians(sa),
                end_angle: cosmic::iced::Radians(ea),
            });
        });
        frame.stroke(&path, stroke);
    }
}

/// Create a composition guide canvas element.
#[allow(clippy::too_many_arguments)]
pub fn composition_canvas<'a>(
    guide: CompositionGuide,
    rotated_sensor_w: f32,
    rotated_sensor_h: f32,
    aspect_crop_ratio: Option<f32>,
    cover_blend: f32,
    insets: BarInsets,
) -> cosmic::Element<'a, Message> {
    cosmic::widget::Canvas::new(GuideProgram {
        guide,
        rotated_sensor_w,
        rotated_sensor_h,
        aspect_crop_ratio,
        cover_blend,
        insets,
    })
    .width(Length::Fill)
    .height(Length::Fill)
    .into()
}
