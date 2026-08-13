// SPDX-License-Identifier: GPL-3.0-only

//! Live "frosted glass" backdrop for overlay panels.
//!
//! Paints a LIVE-BLURRED copy of the camera preview behind a translucent overlay
//! panel, so the panel reads as real frosted glass rather than a sharp preview
//! seen through a translucent tint.
//!
//! # Why a custom container
//!
//! The blur is produced by the shared [`VideoPipeline`] (the same one that draws
//! the live preview) keyed by [`VIDEO_ID_FROSTED`]; that pipeline owns the camera
//! frame GPU texture privately, so the backdrop MUST reach it through
//! `VideoPrimitive` / `VideoPipeline`.
//!
//! iced's `stack` sizes itself to its first child, which makes it awkward to lay
//! a full-`Fill` backdrop *beneath* a `Shrink`-sized panel without inflating the
//! stack to the whole window. [`FrostedContainer`] instead lays out the panel
//! first, sizes itself to the panel, then draws the backdrop (clipped to the
//! panel bounds, positioned at full-preview geometry) BEFORE the panel — real
//! frosted glass with pixel-perfect sizing.
//!
//! Geometry vs. clipping are decoupled inside the primitive:
//! * the widget's **layout bounds** (the panel rect) become the scissor
//!   (`set_scissor_rect`), so the blur is painted only where the panel is;
//! * the **full preview geometry** — the whole window, which is the rect the
//!   preview is laid out into in BOTH fit modes — is derived inside `prepare()`
//!   from the RENDER TARGET and used for `set_viewport`. It is deliberately not
//!   sourced from either widget's bounds: one blur chain is shared by every
//!   frosted consumer, and the consumers report rects that differ by a pixel
//!   (see [`VideoPrimitive`]'s `BlurTargets` docs).
//!
//! Because the blur uses the exact same transforms (cover/contain blend, crop,
//! zoom, mirror, rotation, letterbox, colour filter) as the live preview and is
//! merely clipped to the panel, the blurred slice lines up with the sharp
//! preview behind it.
//!
//! # The window is the preview's LAYOUT rect, not always its IMAGE
//!
//! "Full preview geometry" means the rect the preview is laid out into, not the
//! rect it fills with pixels. In **Fill** (`cover_blend` = 1) the two coincide:
//! Cover covers the window, and every panel and bar sits over live image. In
//! **Fit** (`cover_blend` = 0) they do not — Contain shrinks the image into the
//! window MINUS the UI bars (`content_height = viewport.y - bar_top -
//! bar_bottom`, see `video_shader_blur.wgsl`), so the bar regions are *by
//! construction* exactly where the image is not, and the backdrop resolves them
//! to `letterbox_color` there.
//!
//! That is correct, not a bug to route around: in Fit there is no preview under
//! the bars to blur, and the sharp preview paints the same `letterbox_color`
//! underneath. It follows that the panels and the scrim must share ONE fit —
//! forcing the scrim to Cover would fill the bars with a cover-cropped image
//! that appears nowhere else on screen, which is precisely the alignment this
//! module exists to guarantee. Pinned by `frosted_scrim_bars_are_letterbox_in_fit`
//! and `frosted_scrim_bars_are_not_flat_in_fill` in `video_primitive.rs`.
//!
//! Rounded corners are cut by the blur shader's antialiased SDF, from the radius
//! the primitive carries — NOT by the scissor. A scissor rect is integer-pixel
//! binary coverage, so shaping the silhouette with it (as this once did, via
//! libcosmic's `rounded_rect_strips`) can only ever produce a staircase, which
//! is glaringly visible next to the panel tint's own antialiased edge.

use crate::app::preview_geometry::{BarInsets, frame_rect_on_screen, scrim_bars};
use crate::app::state::Message;
use crate::app::video_primitive::{
    VIDEO_ID_FROSTED, VideoFrame, VideoPrimitive, compositor_blur_params,
};
use crate::app::video_widget::VideoWidgetConfig;
use crate::backends::camera::types::{CameraFrame, PixelFormat};
use cosmic::iced::advanced::widget::{Operation, Tree};
use cosmic::iced::advanced::{Clipboard, Layout, Shell, Widget, layout};
use cosmic::iced::mouse;
use cosmic::iced::{Element, Event, Length, Rectangle, Size, Vector};
use cosmic::{Renderer, Theme};
use iced_wgpu::primitive::Renderer as PrimitiveRenderer;
use std::sync::Arc;

/// The dual-Kawase parameters cosmic-comp would use behind its OWN frosted
/// surfaces at the theme's current `frosted` (`BlurStrength`, ordinal 0..=13,
/// Medium = 6) — so the COSMIC "Frost thickness" setting drives our app-drawn
/// blur to the same on-screen thickness as the desktop's.
///
/// This is REAL parity, and it is now parity by CONSTRUCTION rather than by
/// numerical agreement. `BlurStrength`'s own docs say "Actual blur radius is
/// decided by cosmic-comp" — so we ask cosmic-comp's own table what it decided,
/// and hand the answer to a transcription of cosmic-comp's own kernels
/// (`video_shader_kawase.wgsl`), which run in cosmic-comp's own unit (physical
/// screen px). There is no conversion left to get wrong.
///
/// It did not use to be like this. The backdrop asked for a target SIGMA in
/// screen px, and `prepare()` solved a 37-tap ring radius to hit it, undoing two
/// changes of unit on the way. That worked to within ~15% and cost several
/// hundred lines — and it still banded on device at high frost, because a sparse
/// rosette over sharp sensor data bands no matter what radius you feed it.
fn frost_blur_params() -> crate::app::video_primitive::CompositorBlurParams {
    compositor_blur_params(cosmic::theme::active().cosmic().frosted as u8)
}

/// Build the `VideoPrimitive` for the frosted backdrop from the current preview
/// frame and the SAME config the live preview uses, so the blur transforms match
/// exactly. The `frame` `Arc` is shared (refcount clone) — no pixel copy.
fn make_primitive(
    frame: &Arc<CameraFrame>,
    config: &VideoWidgetConfig,
    corner_radius: f32,
) -> VideoPrimitive {
    let mut primitive = VideoPrimitive::new(VIDEO_ID_FROSTED);
    // Match the preview's transforms exactly — the colour filter among them. The
    // filter is not a transform the tint can stand in for: with Sketch active the
    // preview is a black-and-white pencil drawing, and a Standard backdrop blurs
    // the raw colour scene, so the scrim bars read as a colour smear butted
    // against a monochrome preview (found on device). `video_shader_blur.wgsl`
    // applies it in pass 0, the one pass that samples the source frame.
    //
    // The radius is in logical px; `prepare()` scales it into physical px and the
    // blur shader cuts the rounded silhouette with an antialiased SDF.
    primitive.corner_radius = corner_radius;
    primitive.filter_type = config.filter_type;
    // Run the compositor's own blur, at the compositor's own parameters for the
    // theme's frost setting (see `frost_blur_params`).
    primitive.blur_params = frost_blur_params();
    primitive.mirror_horizontal = config.mirror_horizontal;
    primitive.rotation = config.rotation;
    primitive.crop_uv = config.crop_uv;
    primitive.zoom_level = config.zoom_level;
    primitive.letterbox_color = config.letterbox_color;

    if frame.width > 0 && frame.height > 0 {
        let stride = if frame.stride > 0 {
            frame.stride
        } else {
            match frame.format {
                PixelFormat::RGBA | PixelFormat::ABGR | PixelFormat::BGRA => frame.width * 4,
                PixelFormat::RGB24 => frame.width * 3,
                PixelFormat::YUYV | PixelFormat::UYVY | PixelFormat::YVYU | PixelFormat::VYUY => {
                    frame.width * 2
                }
                _ => frame.width,
            }
        };

        // Must be the same `current_frame` the preview mints its own VideoFrame
        // from: the pipeline maps VIDEO_ID_FROSTED onto VIDEO_ID_NORMAL's source
        // texture and dedups the second upload by frame-data pointer, so feeding
        // the backdrop a *different* frame would leave whichever consumer uploads
        // first to win and the other to render the wrong pixels, silently. Give
        // the backdrop its own frame only by giving it its own video_id.
        let video_frame = VideoFrame {
            id: VIDEO_ID_FROSTED,
            width: frame.width,
            height: frame.height,
            data: frame.data.clone(), // refcount increment, no pixel copy
            format: frame.format,
            stride,
            yuv_planes: frame.yuv_planes,
        };
        primitive.update_frame(video_frame);
    }
    primitive
}

/// A container that draws a live-blurred preview backdrop behind `panel`,
/// scissored to the panel's own on-screen rectangle.
pub struct FrostedContainer<'a> {
    panel: Element<'a, Message, Theme, Renderer>,
    primitive: VideoPrimitive,
    cover_blend: f32,
    insets: BarInsets,
}

impl<'a> FrostedContainer<'a> {
    pub fn new(
        panel: Element<'a, Message, Theme, Renderer>,
        frame: &Arc<CameraFrame>,
        config: &VideoWidgetConfig,
        corner_radius: f32,
    ) -> Self {
        Self {
            panel,
            primitive: make_primitive(frame, config, corner_radius),
            cover_blend: config.cover_blend.unwrap_or(1.0),
            insets: config.insets,
        }
    }
}

impl<'a> Widget<Message, Theme, Renderer> for FrostedContainer<'a> {
    fn children(&self) -> Vec<Tree> {
        vec![Tree::new(&self.panel)]
    }

    fn diff(&mut self, tree: &mut Tree) {
        tree.diff_children(std::slice::from_mut(&mut self.panel));
    }

    fn size(&self) -> Size<Length> {
        // Follow the panel exactly, so the container (and thus the backdrop
        // scissor rect) matches the panel's on-screen footprint.
        self.panel.as_widget().size()
    }

    fn size_hint(&self) -> Size<Length> {
        self.panel.as_widget().size_hint()
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        // Lay out the panel; the container adopts its size so the backdrop is
        // clipped to exactly the panel rectangle.
        let node = self
            .panel
            .as_widget_mut()
            .layout(&mut tree.children[0], renderer, limits);
        let size = node.size();
        layout::Node::with_children(size, vec![node])
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        style: &cosmic::iced::advanced::renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        // `layout.bounds()` = panel rect → drives the scissor (clip_bounds).
        let bounds = layout.bounds();

        // The shader's Cover/Contain math uses the FULL preview dimensions, not
        // the panel's, so pass the full viewport size here. The blur chain's own
        // geometry is NOT taken from here — `prepare()` derives it from the
        // render target, because this rect and the scrim's disagree (see
        // `BlurTargets`).
        self.primitive.update_viewport(
            viewport.width,
            viewport.height,
            self.cover_blend,
            self.insets,
        );

        // Draw the blurred backdrop FIRST, in the CURRENT layer, as ONE draw
        // scissored to the panel rect. The rounded silhouette is cut by the blur
        // chain's antialiased SDF (see `video_shader_frosted.wgsl`), not by the
        // scissor: a scissor is integer-pixel binary coverage, so rounding with
        // it could only ever produce a staircase. The primitive carries the
        // radius and overrides the viewport to the full-preview geometry.
        //
        // iced_wgpu renders custom primitives AFTER quads/text within a layer,
        // so drawing the panel in the same layer would put its translucent
        // background quad *under* this primitive. To keep the blur BEHIND the
        // panel we push the panel into a NEW layer (rendered above this one) —
        // exactly how `stack` layers its children.
        use cosmic::iced::advanced::Renderer as _;
        renderer.draw_primitive(bounds, self.primitive.clone());

        // Then draw the translucent panel on top; its tint composites over the
        // blur to complete the frosted-glass look.
        let panel_layout = layout.children().next().unwrap_or(layout);
        renderer.with_layer(*viewport, |renderer| {
            self.panel.as_widget().draw(
                &tree.children[0],
                renderer,
                theme,
                style,
                panel_layout,
                cursor,
                viewport,
            );
        });
    }

    fn tag(&self) -> cosmic::iced::advanced::widget::tree::Tag {
        cosmic::iced::advanced::widget::tree::Tag::stateless()
    }

    fn state(&self) -> cosmic::iced::advanced::widget::tree::State {
        cosmic::iced::advanced::widget::tree::State::None
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        operation: &mut dyn Operation<()>,
    ) {
        let panel_layout = layout.children().next().unwrap_or(layout);
        self.panel.as_widget_mut().operate(
            &mut tree.children[0],
            panel_layout,
            renderer,
            operation,
        );
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        // The backdrop is purely decorative; forward all events to the panel so
        // its buttons/sliders keep working.
        let panel_layout = layout.children().next().unwrap_or(layout);
        self.panel.as_widget_mut().update(
            &mut tree.children[0],
            event,
            panel_layout,
            cursor,
            renderer,
            clipboard,
            shell,
            viewport,
        );
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        let panel_layout = layout.children().next().unwrap_or(layout);
        self.panel.as_widget().mouse_interaction(
            &tree.children[0],
            panel_layout,
            cursor,
            viewport,
            renderer,
        )
    }

    fn overlay<'b>(
        &'b mut self,
        tree: &'b mut Tree,
        layout: Layout<'b>,
        renderer: &Renderer,
        viewport: &Rectangle,
        translation: Vector,
    ) -> Option<cosmic::iced::advanced::overlay::Element<'b, Message, Theme, Renderer>> {
        let panel_layout = layout.children().next().unwrap_or(layout);
        self.panel.as_widget_mut().overlay(
            &mut tree.children[0],
            panel_layout,
            renderer,
            viewport,
            translation,
        )
    }
}

impl<'a> From<FrostedContainer<'a>> for Element<'a, Message, Theme, Renderer> {
    fn from(widget: FrostedContainer<'a>) -> Self {
        Element::new(widget)
    }
}

/// Full-window, non-interactive frosted backdrop for the crop/letterbox scrim
/// bars (top, bottom, left, right). It takes its bars from
/// [`scrim_bars`][crate::app::preview_geometry::scrim_bars], the same call
/// `OverlayBackgroundProgram` tints, and paints the live-blurred preview clipped
/// to each bar, positioned at full-preview geometry. Stacked directly above the
/// sharp preview and below the scrim tint, so on every aspect ratio the frosted
/// glass covers the same region the scrim tints (including the side bars that
/// only appear on some ratios).
pub struct FrostedScrim {
    primitive: VideoPrimitive,
    target_ratio: Option<f32>,
    insets: BarInsets,
    cover_blend: f32,
}

impl FrostedScrim {
    pub fn new(
        frame: &Arc<CameraFrame>,
        config: &VideoWidgetConfig,
        target_ratio: Option<f32>,
        insets: BarInsets,
    ) -> Self {
        Self {
            // Scrim bars are plain rectangles — no corner rounding.
            primitive: make_primitive(frame, config, 0.0),
            target_ratio,
            insets,
            cover_blend: config.cover_blend.unwrap_or(1.0),
        }
    }
}

impl Widget<Message, Theme, Renderer> for FrostedScrim {
    fn size(&self) -> Size<Length> {
        Size::new(Length::Fill, Length::Fill)
    }

    fn layout(
        &mut self,
        _tree: &mut Tree,
        _renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        layout::Node::new(limits.max())
    }

    fn draw(
        &self,
        _tree: &Tree,
        renderer: &mut Renderer,
        _theme: &Theme,
        _style: &cosmic::iced::advanced::renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        if bounds.width <= 0.0 || bounds.height <= 0.0 {
            return;
        }

        // The blur samples the whole preview and is merely scissored per bar, so
        // pass the exact preview transforms. In Fit the bars land in the
        // preview's letterbox and so resolve to `letterbox_color` — see the
        // module docs on why that is the intended image rather than something to
        // correct for.
        self.primitive
            .update_viewport(bounds.width, bounds.height, self.cover_blend, self.insets);

        let frame_rect =
            frame_rect_on_screen(bounds.width, bounds.height, self.insets, self.target_ratio);

        // Each `draw_primitive` registers a draw scissored to `bar` (clip_bounds)
        // while the primitive internally overrides the viewport to full-preview
        // geometry — so the blurred slice lines up with the sharp preview.
        for bar in scrim_bars(bounds.size(), frame_rect) {
            if bar.width <= 0.0 || bar.height <= 0.0 {
                continue;
            }
            renderer.draw_primitive(
                bar + Vector::new(bounds.x, bounds.y),
                self.primitive.clone(),
            );
        }
    }
}

impl<'a> From<FrostedScrim> for Element<'a, Message, Theme, Renderer> {
    fn from(widget: FrostedScrim) -> Self {
        Element::new(widget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::state::FilterType;
    use crate::app::video_primitive::compositor_blur_params;
    use crate::app::video_widget::VideoContentFit;
    use crate::backends::camera::types::FrameData;

    fn frame(format: PixelFormat, stride: u32) -> Arc<CameraFrame> {
        Arc::new(CameraFrame {
            width: 64,
            height: 48,
            data: FrameData::Copied(vec![7u8; 64 * 48 * 4].into()),
            format,
            stride,
            captured_at: std::time::Instant::now(),
            yuv_planes: None,
            sensor_timestamp_ns: None,
            libcamera_metadata: None,
        })
    }

    fn config() -> VideoWidgetConfig {
        VideoWidgetConfig {
            video_id: crate::app::video_primitive::VIDEO_ID_NORMAL,
            content_fit: VideoContentFit::Cover,
            filter_type: FilterType::Sepia,
            corner_radius: 0.0,
            mirror_horizontal: true,
            rotation: 3,
            crop_uv: Some((0.125, 0.0, 0.875, 1.0)),
            zoom_level: 2.5,
            scroll_zoom_enabled: true,
            cover_blend: Some(0.5),
            insets: BarInsets::horizontal(47.0, 174.0),
            letterbox_color: [0.1, 0.2, 0.3, 1.0],
        }
    }

    /// The backdrop must copy the preview's transforms VERBATIM.
    ///
    /// This is the second half of the agreement `preview_video_config` exists to
    /// guarantee (see `the_preview_and_the_backdrop_always_agree` in
    /// `camera_preview::widget`): sharing one config is worth nothing if
    /// `make_primitive` then fails to apply part of it. A dropped field here — a
    /// zoom that stays 1.0, a rotation that stays 0 — slides the blurred slice
    /// against the sharp preview exactly as a disagreeing config would.
    #[test]
    fn make_primitive_copies_every_preview_transform() {
        let cfg = config();
        let p = make_primitive(&frame(PixelFormat::RGBA, 256), &cfg, 12.0);

        assert_eq!(p.video_id, VIDEO_ID_FROSTED);
        assert_eq!(p.mirror_horizontal, cfg.mirror_horizontal);
        assert_eq!(p.rotation, cfg.rotation);
        assert_eq!(p.crop_uv, cfg.crop_uv);
        assert_eq!(p.zoom_level, cfg.zoom_level);
        assert_eq!(p.letterbox_color, cfg.letterbox_color);
        assert_eq!(p.corner_radius, 12.0);
        // The filter is a transform like any other, and this line is the one that
        // used to pin the opposite. Leaving it Standard shipped: with Sketch on,
        // the preview was a pencil drawing and the scrim bars blurred the raw
        // colour scene right up against it.
        assert_eq!(p.filter_type, cfg.filter_type);
        // And the blur really is parameterized from the theme, not left at the
        // transition blur's default.
        assert_eq!(p.blur_params, frost_blur_params());
    }

    /// The backdrop shares the preview's frame by REFCOUNT, never by copy.
    ///
    /// `make_primitive` runs on every view build for every frosted panel — six to
    /// ten of them at 30 fps. Copying a 1280x960 RGBA frame each time would be
    /// ~150 MB/s of memcpy for a decorative backdrop.
    #[test]
    fn make_primitive_shares_the_frame_without_copying_it() {
        let f = frame(PixelFormat::RGBA, 256);
        let before = f.data.as_ptr();
        let p = make_primitive(&f, &config(), 0.0);
        let guard = p.data.lock().unwrap();
        let vf = guard
            .frame
            .as_ref()
            .expect("the primitive must carry the frame");
        assert_eq!(
            vf.data.as_ptr(),
            before,
            "the backdrop must share the preview's pixels by refcount — and it must \
             be the SAME frame the preview uploads, because the pipeline dedups the \
             second upload by data pointer"
        );
        assert_eq!(vf.id, VIDEO_ID_FROSTED);
    }

    /// A frame that reports no stride gets one derived from its pixel format.
    ///
    /// Cameras do report `stride: 0`. Passing that on gives the upload a zero row
    /// pitch — a garbled backdrop, silently — so `make_primitive` fills it in.
    /// The mapping is per-format and easy to get wrong by a factor of the pixel
    /// size, which is exactly what this pins.
    #[test]
    fn make_primitive_derives_a_missing_stride_from_the_format() {
        for (format, want) in [
            (PixelFormat::RGBA, 64 * 4),
            (PixelFormat::ABGR, 64 * 4),
            (PixelFormat::BGRA, 64 * 4),
            (PixelFormat::RGB24, 64 * 3),
            (PixelFormat::YUYV, 64 * 2),
            (PixelFormat::UYVY, 64 * 2),
            (PixelFormat::YVYU, 64 * 2),
            (PixelFormat::VYUY, 64 * 2),
            // Planar / greyscale: one byte per pixel on the first plane.
            (PixelFormat::NV12, 64),
            (PixelFormat::I420, 64),
        ] {
            let p = make_primitive(&frame(format, 0), &config(), 0.0);
            let guard = p.data.lock().unwrap();
            let got = guard.frame.as_ref().unwrap().stride;
            assert_eq!(got, want, "{format:?} with no reported stride");
        }
    }

    /// A stride the camera DID report always wins over the derived one.
    #[test]
    fn make_primitive_keeps_a_reported_stride() {
        // Padded rows: 64 px of RGBA would be 256, but this camera pads to 320.
        let p = make_primitive(&frame(PixelFormat::RGBA, 320), &config(), 0.0);
        let guard = p.data.lock().unwrap();
        assert_eq!(
            guard.frame.as_ref().unwrap().stride,
            320,
            "a reported stride must not be overwritten by the format default — \
             padded rows would be read at the wrong pitch and skew the image"
        );
    }

    /// A zero-sized frame produces a primitive with no frame at all, rather than
    /// a zero-dimension upload.
    #[test]
    fn make_primitive_skips_an_empty_frame() {
        let mut f = frame(PixelFormat::RGBA, 0);
        Arc::get_mut(&mut f).unwrap().width = 0;
        let p = make_primitive(&f, &config(), 0.0);
        assert!(p.data.lock().unwrap().frame.is_none());
    }

    /// `frost_blur_params` must hand back cosmic-comp's own entry for the theme's
    /// current frost setting — that mapping IS the parity claim.
    #[test]
    fn frost_blur_params_follows_the_theme() {
        let level = cosmic::theme::active().cosmic().frosted as u8;
        assert_eq!(frost_blur_params(), compositor_blur_params(level));
        // And it is a real entry, not a degenerate one.
        assert!(frost_blur_params().passes >= 1);
        assert!(frost_blur_params().offset > 0.0);
    }
}
