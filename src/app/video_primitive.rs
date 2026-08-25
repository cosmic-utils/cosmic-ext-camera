// SPDX-License-Identifier: GPL-3.0-only

//! Custom video rendering primitive with direct GPU texture updates
//!
//! This module implements iced_video_player-style optimizations:
//! - Direct GPU texture updates (no Handle recreation)
//! - RGBA textures for native RGB processing
//! - Persistent textures across frames

use crate::app::state::FilterType;
use crate::backends::camera::types::{FrameData, PixelFormat, YuvPlanes};
use cosmic::iced::Rectangle;

/// Video ID for the normal camera preview (no blur).
pub const VIDEO_ID_NORMAL: u64 = 0;
/// Video ID for the blurred background preview (used during transitions/HDR+ processing).
pub const VIDEO_ID_BLUR: u64 = 1;
/// Video ID for the live frosted-glass backdrop drawn behind translucent overlay
/// panels. Renders the same blur chain as `VIDEO_ID_BLUR` but every frame (no
/// cache) and scissored to the panel rectangle while positioned at full-preview
/// geometry, so the blurred slice lines up with the sharp preview behind it.
pub const VIDEO_ID_FROSTED: u64 = 2;
/// Video ID for the filter picker's thumbnail grid: one id for all fifteen
/// swatches, because they are the same frame under fifteen filters and a filter
/// is a property of the *binding*, not of the texture (see [`source_texture_id`]
/// and `VideoPipeline::bindings`).
pub const VIDEO_ID_FILTER_PREVIEW: u64 = 99;
use iced_wgpu::graphics::Viewport;
use iced_wgpu::primitive::{Pipeline as PipelineTrait, Primitive as PrimitiveTrait};
use iced_wgpu::wgpu;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

// ---------------------------------------------------------------------------
// Compositor blur parity
//
// The frosted backdrop must land on the SAME on-screen blur as cosmic-comp's own
// frosted surfaces at the same "Frost thickness" setting. It does, and not by
// approximation: we run THEIR algorithm — the dual-Kawase in
// `video_shader_kawase.wgsl`, transcribed from their `blur_downsample.frag` and
// `blur_upsample.frag` — with THEIR (passes, offset) out of the table below, in
// THEIR unit (physical screen px). Parity is exact by construction.
//
// That is worth spelling out, because it used to be an elaborate model. Both
// kernels were reduced to a Gaussian sigma and our ring radius was solved to hit
// theirs, through two changes of unit (sensor texel -> intermediate texel ->
// screen px) that depended on the window size and the live fit blend. It worked,
// roughly, and it cost `compositor_blur_sigma`, `frosted_blur_radius_for_sigma`,
// `RING_KERNEL_SIGMA_PER_RADIUS`, `FROST_MAX_BLUR_RADIUS` and a clamp that made
// high frost read thin in Fit mode — and it still banded on device, because a
// 37-tap rosette over sharp sensor data is a sparse lattice no radius fixes.
// Running the real algorithm deleted all of it. The table is what survived, and
// it is now the whole of the parity story rather than an input to a model.
// ---------------------------------------------------------------------------

// Portions Copyright (C) System76, Inc. — derived from cosmic-comp (GPL-3.0-only)

/// Mirror of cosmic-comp `backend/render/wayland/blur_effect.rs`'s `MAX_STEPS`.
const BLUR_MAX_STEPS: usize = 15;

/// One entry of cosmic-comp's `BLUR_PARAMS`: a dual-Kawase pass count and the
/// base sample offset.
///
/// Upstream's third field, `extended_radius`, is deliberately omitted: it only
/// sizes the off-surface region the compositor captures so the blur has content
/// to pull in from outside the surface. It does not enter the kernel, so it has
/// no bearing on the resulting sigma.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompositorBlurParams {
    /// Dual-Kawase pass count: this many downsamples then this many upsamples.
    pub passes: u32,
    /// Base sample offset in PHYSICAL SCREEN px. Pass `i` uses `offset / 2^i`.
    pub offset: f64,
}

/// Re-derivation of cosmic-comp's `BLUR_PARAMS` table.
///
/// This mirrors the upstream `LazyLock` in `blur_effect.rs` line-for-line rather
/// than hardcoding the 15 resulting floats, so a reader can diff it against
/// upstream and see at a glance whether it has drifted. In particular the
/// `remaining_steps` saturation is NOT incidental: the last band asks for 8
/// steps but only 6 remain, so its offsets are spaced `5/6` apart, not `5/8`.
///
/// The table is also non-monotonic in `offset` (index 3 is 3.0, index 4 drops
/// back to 2.6) because the pass count steps up at each band boundary — the
/// *sigma* is what increases monotonically, not the offset.
static COMPOSITOR_BLUR_PARAMS: LazyLock<Vec<CompositorBlurParams>> = LazyLock::new(|| {
    let mut params = Vec::new();

    let mut remaining_steps = BLUR_MAX_STEPS as isize;
    // min offset, max offset (upstream's `extended radius` column is dropped).
    let offsets = [(1.0, 2.0), (2.0, 3.0), (2.0, 5.0), (3.0, 8.0)];

    let sum = offsets.iter().map(|(min, max)| *max - *min).sum::<f64>();
    for (i, (min, max)) in offsets.into_iter().enumerate() {
        let mut iter_num = f64::ceil((max - min) / sum * (BLUR_MAX_STEPS as f64)) as usize;
        remaining_steps -= iter_num as isize;

        if remaining_steps < 0 {
            iter_num = iter_num.saturating_add_signed(remaining_steps);
        }

        let diff = max - min;
        for j in 1..=iter_num {
            params.push(CompositorBlurParams {
                passes: i as u32 + 1,
                offset: min + (diff / iter_num as f64) * j as f64,
            });
        }
    }

    params
});

/// cosmic-comp's own dual-Kawase parameters for theme `frosted` level `level`
/// (`BlurStrength`, ordinal 0..=13, Medium = 6).
///
/// This is the ENTIRE parity mechanism. Upstream indexes `BLUR_PARAMS` with
/// `frosted as u8 + 1`, clamped to `MAX_STEPS - 1` (`backend/render/mod.rs`'s
/// `blur_strength`, and `blur_effect.rs`'s `.min(MAX_STEPS - 1)`), so level
/// 0..=13 selects table index 1..=14 and Medium (6) selects index 7 — 3 passes
/// at offset 4.4. We select the same entry and feed it to the same kernels, so
/// there is nothing left to convert or approximate.
///
/// The offsets are in PHYSICAL, not logical, px: upstream never scales `offset`
/// by the output scale factor, so its blur is genuinely half as wide (in logical
/// px) on a 2x display. Our chain works in physical px for exactly that reason —
/// see `video_shader_blur.wgsl`'s header on why pass 0 renders at screen
/// resolution.
pub fn compositor_blur_params(level: u8) -> CompositorBlurParams {
    COMPOSITOR_BLUR_PARAMS[(level as usize + 1).min(BLUR_MAX_STEPS - 1)]
}

/// Most Kawase passes any level asks for (`BLUR_PARAMS`'s top band is 4), and
/// therefore how many ping-pong steps [`BlurTargets`] preallocates: `2 * this`.
const MAX_KAWASE_PASSES: u32 = 4;

/// Clamp a pass count to what a `width` x `height` target can actually carry.
///
/// Pass `i` reads the sub-rect `[0, W>>i]`, so `W >> passes` must stay at least
/// one texel or the deepest level degenerates to nothing and the up-chain
/// reconstructs from a single texel. Upstream never needs this — a compositor's
/// framebuffer is always at least a screen — but our targets follow the preview
/// widget, which tests and transient layouts can make tiny.
fn effective_kawase_passes(width: u32, height: u32, passes: u32) -> u32 {
    let smallest = width.min(height).max(1);
    // `smallest >> p >= 1` <=> `p <= floor(log2(smallest))`.
    let max_passes = smallest.ilog2();
    passes.clamp(1, max_passes.max(1))
}

// Static for GPU upload time tracking (insights)
static GPU_UPLOAD_TIME_US: AtomicU64 = AtomicU64::new(0);
static GPU_FRAME_SIZE: AtomicU64 = AtomicU64::new(0);

/// Get the last GPU upload time in microseconds
pub fn get_gpu_upload_time_us() -> u64 {
    GPU_UPLOAD_TIME_US.load(Ordering::Relaxed)
}

/// Get the last GPU frame size in bytes
pub fn get_gpu_frame_size() -> u64 {
    GPU_FRAME_SIZE.load(Ordering::Relaxed)
}

/// Default UV texture dimensions when yuv_planes is not available
fn default_uv_size(format: PixelFormat, width: u32, height: u32) -> (u32, u32) {
    match format {
        PixelFormat::NV12 | PixelFormat::NV21 | PixelFormat::I420 => (width / 2, height / 2),
        PixelFormat::YUYV | PixelFormat::UYVY | PixelFormat::YVYU | PixelFormat::VYUY => {
            (width / 2, height)
        }
        _ => (1, 1),
    }
}

/// Video frame data for GPU upload
///
/// Supports both RGBA and YUV formats. For YUV formats, the data is converted
/// to RGBA by a GPU compute shader before rendering.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub id: u64,
    pub width: u32,
    pub height: u32,
    /// Frame data: RGBA pixels, Y plane (NV12/I420), or packed YUYV
    pub data: FrameData,
    /// Pixel format (RGBA, NV12, I420, YUYV)
    pub format: PixelFormat,
    /// Row stride for main data (bytes per row including padding)
    pub stride: u32,
    /// Additional YUV planes (for NV12/I420 formats)
    pub yuv_planes: Option<YuvPlanes>,
}

impl VideoFrame {
    /// Get data slice for the main plane
    #[inline]
    pub fn data_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get RGBA data slice (only valid for RGBA format)
    /// For YUV formats, use the YUV conversion pipeline first
    #[inline]
    pub fn rgba_data(&self) -> &[u8] {
        debug_assert!(
            self.format == PixelFormat::RGBA,
            "rgba_data() called on YUV frame"
        );
        &self.data
    }

    /// Check if this frame needs GPU conversion (YUV, ABGR, BGRA, etc.)
    #[inline]
    pub fn needs_gpu_conversion(&self) -> bool {
        self.format.needs_gpu_conversion()
    }
}

/// Viewport and content fit data for Cover mode
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ViewportUniform {
    /// Viewport width and height (full widget size)
    viewport_size: [f32; 2],
    /// Content fit blend: 0.0 = Contain, 1.0 = Cover (interpolated during animation)
    content_fit_mode: f32,
    /// Filter mode: 0 = None, 1 = Black & White
    filter_mode: u32,
    /// Corner radius in pixels (0 = no rounding)
    corner_radius: f32,
    /// Mirror horizontally: 0 = normal, 1 = mirrored
    mirror_horizontal: u32,
    /// UV offset for scroll clipping (normalized 0-1, where visible area starts)
    uv_offset: [f32; 2],
    /// UV scale for scroll clipping (normalized, size of visible area relative to full widget)
    uv_scale: [f32; 2],
    /// Crop UV min (u_min, v_min) - normalized 0-1
    crop_uv_min: [f32; 2],
    /// Crop UV max (u_max, v_max) - normalized 0-1
    crop_uv_max: [f32; 2],
    /// Zoom level (1.0 = no zoom, 2.0 = 2x zoom, etc.)
    ///
    /// Read by every shader that samples the SOURCE frame: the preview, the
    /// filter pre-blur, and the blur chain's pass 1. The blur chain's passes 2
    /// and 3 sample an intermediate that pass 1 has already zoomed, so they must
    /// leave this at 1.0 — exactly as they do for `rotation`, `mirror_horizontal`
    /// and the crop.
    zoom_level: f32,
    /// Sensor rotation: 0=None, 1=90CW, 2=180, 3=270CW
    rotation: u32,
    /// Top bar height in pixels (for contain-mode centering between UI bars)
    bar_top_height: f32,
    /// Bottom bar height in pixels
    bar_bottom_height: f32,
    /// Dual-Kawase sample offset for THIS pass, in PHYSICAL SCREEN px — i.e.
    /// upstream's `offset / 2^i` (down) or `offset / 2^(passes-i)` (up), already
    /// divided down by `prepare()`. Read only by `video_shader_kawase.wgsl`;
    /// every other shader declares the field and ignores it.
    ///
    /// This and `dim_factor` occupy what used to be two pure padding floats,
    /// which the layout needs anyway to align `letterbox_color` (vec4) to
    /// offset 80 (see `viewport_uniform_layout_is_stable`).
    kawase_offset: f32,
    /// RGB multiplier applied by the final composite. The transition blur uses
    /// < 1.0 to darken the frozen frame as a "something is happening" cue; the
    /// frosted backdrop uses 1.0, since its tint comes from the theme's
    /// container alpha rather than from dimming the wallpaper behind it.
    ///
    /// Applied ONCE, at the composite. It used to be applied per pass and
    /// compound over exactly three of them; the Kawase runs a level-dependent
    /// 2..8 passes, so "per pass" no longer names a fixed total — see
    /// [`TRANSITION_BLUR_DIM`].
    dim_factor: f32,
    /// Theme background color (RGBA, sRGB straight) used by the transform pass
    /// to fill letterbox areas instead of returning transparent (which would
    /// let the COSMIC window background show through during Fit-mode blur
    /// transitions). Other shaders accept the field but ignore it.
    letterbox_color: [f32; 4],
    /// The rect (x, y, w, h) the corners are cut from, in PHYSICAL px of the
    /// render target — the space `@builtin(position)` is in — so the shader can
    /// round an antialiased SDF against it. Written by every pass that rounds:
    /// the frosted backdrop's composite and the sharp preview alike, both via
    /// [`corner_sdf_params`]. Zeroed (with `corner_radius` = 0) for the passes
    /// that render into an intermediate, which keeps them square.
    ///
    /// This is the extent the corners are cut from; `viewport_size` is the size
    /// the content-fit math works in. They are the same rect for the preview and
    /// nothing else, which is why they are separate fields — see the pre-blur
    /// pass 2 uniform, which deliberately lies about `viewport_size`.
    ///
    /// Lands at offset 96 (a vec4 boundary, since `letterbox_color` ends there)
    /// and leaves every existing field's offset untouched, so the WGSL structs
    /// that don't declare it stay valid against the same buffer. Only
    /// `video_shader_frosted.wgsl` and `video_shader.wgsl` declare it.
    panel_rect: [f32; 4],
    /// Final composite only: film-grain amplitude, ported from cosmic-comp's
    /// `clipped_surface.frag` (their `NOISE = 0.03`). See [`FROSTED_NOISE`].
    ///
    /// Appended after `panel_rect` for the same reason `panel_rect` was appended
    /// after `letterbox_color`: every earlier offset is untouched, so the four
    /// shaders that stop short of it stay valid against the same, larger buffer.
    /// Only `video_shader_frosted.wgsl` declares it.
    noise: f32,
    /// Pads to the 16-byte boundary `content_rect` (a vec4) has to start on. A
    /// WGSL uniform's size must round up to a 16-byte multiple too, and `noise`
    /// alone would leave the struct at 116.
    _pad: [f32; 3],
    /// Final composite only: the rect the preview IMAGE occupies, as
    /// `(min_x, min_y, max_x, max_y)` normalized over the blur chain's own rect —
    /// the full preview — which is exactly the space the composite's
    /// `tex_coords` spans. Everything outside it is letterbox, and the composite
    /// paints nothing there. See [`content_rect_uv`] for why, and for the fit
    /// math it mirrors.
    ///
    /// Defaults to the whole unit square, which the shader reads as "no letterbox
    /// to cut" and skips entirely — so every pass that does not set it (the
    /// Kawase steps, the pre-blur, the sharp preview) is bit-for-bit unaffected.
    ///
    /// Appended last for the same reason `noise` and `panel_rect` were: every
    /// earlier offset is untouched, so the five shaders that stop short of it stay
    /// valid against the same, larger buffer.
    content_rect: [f32; 4],
    /// Left/right bar widths in pixels. Appended to preserve every existing
    /// uniform offset while extending Contain/Fit to side controls.
    bar_left_width: f32,
    bar_right_width: f32,
    _side_pad: [f32; 2],
}

impl Default for ViewportUniform {
    /// An identity-ish pass: no transforms, no filter, square corners, no
    /// dimming, no grain, no Kawase offset. Every construction site starts here
    /// and overrides only the fields it actually cares about, so adding a field to
    /// this struct no longer means editing every literal.
    fn default() -> Self {
        Self {
            viewport_size: [0.0, 0.0],
            content_fit_mode: 0.0,
            filter_mode: 0,
            corner_radius: 0.0,
            mirror_horizontal: 0,
            uv_offset: [0.0, 0.0],
            uv_scale: [1.0, 1.0],
            crop_uv_min: [0.0, 0.0],
            crop_uv_max: [1.0, 1.0],
            zoom_level: 1.0,
            rotation: 0,
            bar_top_height: 0.0,
            bar_bottom_height: 0.0,
            kawase_offset: 0.0,
            dim_factor: 1.0,
            letterbox_color: [0.0, 0.0, 0.0, 1.0],
            panel_rect: [0.0; 4],
            noise: 0.0,
            _pad: [0.0; 3],
            content_rect: FULL_CONTENT_RECT,
            bar_left_width: 0.0,
            bar_right_width: 0.0,
            _side_pad: [0.0; 2],
        }
    }
}

/// "The image covers the whole preview rect" — no letterbox, nothing for the
/// composite to cut. The default, and the fallback whenever the fit math has no
/// meaningful answer (no frame yet, a degenerate window). Erring this way keeps
/// the frosted glass painting; erring the other way would make it vanish.
const FULL_CONTENT_RECT: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

/// The rect the preview IMAGE occupies inside the preview rect, normalized to it
/// as `(min_x, min_y, max_x, max_y)`. Its complement is the letterbox.
///
/// # Why this exists
///
/// The blur chain fills the letterbox with an opaque `letterbox_color` (see
/// `video_shader_blur.wgsl`), and it has to: the Kawase kernels normalize by
/// `sum.a` and read alpha as a region mask, so a transparent letterbox would make
/// the image bleed outward past its own edge instead of blending toward the
/// background. That fill is correct *inside the chain* and wrong *on screen* —
/// composited at alpha 1 over a translucent window background, it stacks a second
/// opaque copy of the background colour over the one layer that is supposed to be
/// see-through, and the frosted glass disappears exactly where there is no camera
/// image to justify it (issue #569).
///
/// So the chain keeps its opaque fill and the composite cuts the letterbox back
/// out, using this rect. What is left is the honest picture: blurred preview
/// where there is preview, and the window background — the compositor's frosted
/// glass — where there is not.
///
/// This is deliberately geometric rather than per-pixel. Nothing in the blurred
/// texture distinguishes "letterbox" from "a scene that happens to be that
/// colour": alpha comes out of the Kawase as exactly 1 or 0 (it is `sum / sum.a`),
/// so there is no channel left to carry the distinction. The fit is the only thing
/// that knows, and it is knowable in closed form.
///
/// # Why it is safe to duplicate the fit math
///
/// Every line below mirrors the `{}` block of `video_shader_blur.wgsl`'s
/// `fs_main`, from the same uniform values, and the two are pinned together on
/// the GPU by `content_rect_matches_where_the_shader_stops_drawing`: it renders
/// pass 0 with a magenta letterbox and checks that the boundary this computes is
/// where the magenta actually starts. A drift in either is a test failure, not a
/// silent misalignment.
///
/// `tex_size` is the SOURCE texture's dimensions in sensor orientation — what
/// `textureDimensions(texture_blur)` returns in that shader. Digital zoom is
/// deliberately absent: pass 0 applies it *after* its letterbox early-out, so it
/// changes which texels are sampled, never how much of the screen the image
/// covers.
#[allow(clippy::too_many_arguments)]
fn content_rect_uv_with_insets(
    viewport_size: [f32; 2],
    tex_size: (u32, u32),
    rotation: u32,
    crop_min: [f32; 2],
    crop_max: [f32; 2],
    cover_blend: f32,
    insets: crate::app::preview_geometry::BarInsets,
) -> [f32; 4] {
    let [w, h] = viewport_size;
    let content_w = w - insets.left - insets.right;
    let content_h = h - insets.top - insets.bottom;
    // A window with no room between its own chrome has no meaningful contain fit
    // (the shader's `contain_zoom` goes non-positive there). Paint everything.
    if w <= 0.0
        || h <= 0.0
        || content_w <= 0.0
        || content_h <= 0.0
        || tex_size.0 == 0
        || tex_size.1 == 0
    {
        return FULL_CONTENT_RECT;
    }

    let blend = cover_blend.clamp(0.0, 1.0);
    let lerp = |a: f32, b: f32| a + (b - a) * blend;
    // 90/270 rotations swap the sensor's axes into display orientation. Both
    // `tex_size` and the crop range are swapped, exactly as the shader does.
    let rotated = rotation == 1 || rotation == 3;
    let (tex_w, tex_h) = if rotated {
        (tex_size.1 as f32, tex_size.0 as f32)
    } else {
        (tex_size.0 as f32, tex_size.1 as f32)
    };
    let (range_x, range_y) = {
        let x = lerp(crop_max[0], 1.0) - lerp(crop_min[0], 0.0);
        let y = lerp(crop_max[1], 1.0) - lerp(crop_min[1], 0.0);
        if rotated { (y, x) } else { (x, y) }
    };
    let (eff_w, eff_h) = (tex_w * range_x, tex_h * range_y);
    if eff_w <= 0.0 || eff_h <= 0.0 {
        return FULL_CONTENT_RECT;
    }

    let contain_zoom = (content_w / eff_w).min(content_h / eff_h);
    let cover_zoom = (w / eff_w).max(h / eff_h);
    let zoom = lerp(contain_zoom, cover_zoom);
    // Contain centres the image between the UI bars, Cover on the window.
    let center_x = lerp((insets.left + content_w * 0.5) / w, 0.5);
    let center_y = lerp((insets.top + content_h * 0.5) / h, 0.5);

    // `scale = viewport / (effective_tex * zoom)` is the shader's UV scale, so
    // the image's on-screen extent is `effective_tex * zoom` — in the same
    // (logical px) unit as the viewport, which is all the normalization needs.
    let half_w = eff_w * zoom * 0.5 / w;
    let half_h = eff_h * zoom * 0.5 / h;
    [
        center_x - half_w,
        center_y - half_h,
        center_x + half_w,
        center_y + half_h,
    ]
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn content_rect_uv(
    viewport_size: [f32; 2],
    tex_size: (u32, u32),
    rotation: u32,
    crop_min: [f32; 2],
    crop_max: [f32; 2],
    cover_blend: f32,
    bar_top: f32,
    bar_bottom: f32,
) -> [f32; 4] {
    content_rect_uv_with_insets(
        viewport_size,
        tex_size,
        rotation,
        crop_min,
        crop_max,
        cover_blend,
        crate::app::preview_geometry::BarInsets {
            top: bar_top,
            bottom: bar_bottom,
            ..Default::default()
        },
    )
}

/// The `(panel_rect, corner_radius)` an antialiased corner SDF is cut with, in
/// the PHYSICAL px of the render target that `@builtin(position)` is in.
///
/// Every shader that rounds a corner takes its silhouette from these two, so
/// there is one answer to "which rect are the corners cut from" instead of one
/// per pass — the passes disagree about `viewport_size` (the pre-blur's second
/// pass matches it to its intermediate; the frosted backdrop spans the whole
/// preview), and a silhouette derived from it inherits that disagreement.
///
/// Takes the RAW physical bounds, not the clamped ones. Clamping is a property
/// of the render target, not of the widget: one that runs off a window edge
/// still has its corner out there, and rounding at the clamped edge instead
/// would curve the silhouette away mid-screen-edge. The clamp stays where it
/// belongs — on the scissor and the viewport, which wgpu does validate.
///
/// `radius` arrives in LOGICAL px (it comes from a container's style), so it is
/// scaled into physical space and clamped to half the rect the way iced clamps
/// its own quad corners, so a blurred panel matches the tint drawn over it.
fn corner_sdf_params(
    raw_physical_bounds: (f32, f32, f32, f32),
    radius: f32,
    scale: f32,
) -> ([f32; 4], f32) {
    let (px, py, pw, ph) = raw_physical_bounds;
    let radius_px = (radius * scale).min(pw * 0.5).min(ph * 0.5).max(0.0);
    ([px, py, pw, ph], radius_px)
}

/// RGB dim applied to the frozen frame during a camera/mode transition, as a
/// subtle "something is happening" cue.
///
/// It is `0.85³`, and that cube is the whole story: the old chain applied `0.85`
/// in each of its three passes, so the frame landed at ~0.61 of its original
/// brightness. The Kawase chain has no fixed pass count to compound over — it
/// runs `2 * passes` kernels, and `passes` depends on the frost level — so "per
/// pass" stopped naming a total. The dim therefore moved to the final composite,
/// which applies it exactly once, and the constant absorbed the exponent so the
/// on-screen darkness is unchanged. (Doing it at the composite also makes it
/// per-panel for free, which is where it wants to live anyway.)
///
/// The multiply is linear in both worlds — the shader reads decoded values,
/// scales, and writes back through the same encoding on every hop — so `0.85`
/// three times and `0.614125` once are the same number, not an approximation.
const TRANSITION_BLUR_DIM: f32 = 0.85 * 0.85 * 0.85;

/// Film-grain amplitude for the frosted chrome: cosmic-comp's `NOISE`
/// (`backend/render/wayland/blur_effect.rs:37`), applied where they apply it —
/// the final composite over the blurred backdrop (`clipped_surface.frag:87-92`).
///
/// It is not decoration. Grain is part of the material COSMIC's glass is made
/// of, so omitting it was a parity gap; and it dithers the quantisation steps a
/// heavy blur of a low-contrast scene would otherwise show as contouring.
///
/// The TRANSITION blur deliberately passes 0.0 instead. It is not glass — it is
/// a dimmed veil over a frozen frame, with no tint on top to read the grain
/// through — and its look is user-verified, so it is left alone.
const FROSTED_NOISE: f32 = 0.03;

/// Dual-Kawase parameters for the transition blur (`VIDEO_ID_BLUR`), which —
/// unlike the frosted backdrop — answers to no compositor setting and picks its
/// own.
///
/// # How this was chosen
///
/// The transition blur's look is user-verified and had to survive the rewrite,
/// but it was authored as a ring radius (`22.292178`) in intermediate texels,
/// which the Kawase has no notion of. So it was preserved the only way that is
/// checkable: hold the ON-SCREEN thickness fixed on the device it was tuned on.
///
/// The old chain gave `k · 22.292178 · sqrt(1 + 1/16) = 7.54` intermediate texels
/// of sigma (`k = 0.32813`, the ring kernel's sigma per unit radius; the `1/16`
/// is pass 1's contribution at its 4x-finer scale). On the target phone in Cover
/// — a 2592x1940 sensor blurred at 648x485, covered onto a 1080x2340 window, so
/// `int_scale = max(1080/648, 2340/485) = 4.825` screen px per texel — that is
/// `36.4` physical screen px.
///
/// Inverting the Kawase variance identity `σ² = offset²·(4^p − 1)·35/72` (derived
/// in `kawase_sigma_matches_the_variance_model`) at `p = 4` gives
/// `offset = sqrt(36.4² / (255 · 35/72)) = 3.268`.
///
/// `p = 4` rather than `p = 3` on upstream's own advice: `p = 3` would need
/// `offset = 6.58`, and `BLUR_PARAMS` caps its 3-pass band at 5.0 and steps to 4
/// passes rather than push the offset higher. 3.268 sits at the low end of their
/// 4-pass band (3.0..8.0), i.e. inside the range they validated.
///
/// Note this makes the transition blur's thickness DEVICE-INDEPENDENT, where the
/// old radius drifted with window size and sensor resolution. That is a change,
/// and an intended one — it is the same defect the frosted path was fixed for.
/// The number above pins the phone it was tuned on.
const TRANSITION_BLUR_PARAMS: CompositorBlurParams = CompositorBlurParams {
    passes: 4,
    offset: 3.268,
};

/// Combined frame and viewport data to reduce mutex contention
/// Single lock acquisition instead of two separate locks per frame
#[derive(Debug)]
pub struct FrameViewportData {
    pub frame: Option<VideoFrame>,
    /// (width, height, cover_blend, four-sided bar insets)
    pub viewport: (f32, f32, f32, crate::app::preview_geometry::BarInsets),
    /// Physical widget bounds (x, y, width, height) clamped to render target
    /// Stored during prepare() and used in render() for valid viewport rect
    pub physical_bounds: Option<(f32, f32, f32, f32)>,
    /// UV offset for scroll/render-target clipping (normalized 0-1)
    pub uv_offset: (f32, f32),
    /// UV scale for scroll/render-target clipping (normalized 0-1)
    pub uv_scale: (f32, f32),
    /// For the frosted backdrop only: the FULL preview geometry in physical px,
    /// derived in `prepare()` from the RENDER TARGET — never from a widget's
    /// bounds, because one blur chain is shared by consumers that disagree about
    /// the rect (see `BlurTargets`). Sizes the Kawase targets in `prepare()` and
    /// is the `set_viewport` rect in `render()`. `None` for every other
    /// video_id.
    pub frosted_full_physical: Option<(f32, f32, f32, f32)>,
    /// Frosted backdrop only: this primitive's OWN binding for the blur's final
    /// pass, built in `prepare()` and used by `render()` instead of the shared
    /// `intermediate_2.bind_group`.
    ///
    /// Why it can't be shared: iced_wgpu runs every `prepare()` before any
    /// `render()`, so a uniform owned by the pipeline is clobbered by the last
    /// panel to prepare — there'd be no way to give each panel its own rect and
    /// corner radius. Each panel's primitive owning its buffer sidesteps that
    /// without a slot allocator. The blur intermediates themselves stay shared,
    /// so passes 1+2 still run once per frame; only this cheap final blit is
    /// per-panel.
    ///
    /// # The per-frame allocation is deliberate — please don't "optimise" it
    ///
    /// Because `VideoPrimitive::new` mints a fresh `Arc<Mutex<FrameViewportData>>`
    /// on every view build, this binding cannot survive a frame: `prepare()`
    /// creates a `wgpu::Buffer` + `wgpu::BindGroup` for every frosted panel, every
    /// frame (~6-10 panels at 30fps ≈ 200-300 of each per second). That churn is a
    /// known, accepted cost. The alternatives were worked through and rejected:
    ///
    /// * **It is not the bottleneck.** This same feature runs a 3-pass Gaussian
    ///   blur every frame; on the target phone (Adreno 630) that is milliseconds
    ///   of GPU. Ten 112-byte uniform buffers plus ten descriptor sets are an
    ///   estimated 0.1–0.3 ms of CPU — order of 1% of a 33 ms budget. If frosted
    ///   glass ever needs to get cheaper, cut the panel count or the blur passes;
    ///   this is noise beside them.
    /// * **Pooling needs an identity this design does not have.** A pool could live
    ///   on `VideoPipeline` (which does persist across frames), but the only
    ///   channel from `prepare()` to `render()` is `self.data`, and that is new for
    ///   every drawn instance — a fresh one per view build, and a fresh one per
    ///   clone (see `VideoPrimitive`'s `Clone`), so it carries nothing a pool could
    ///   be keyed on. A pool would therefore have to hand out slots
    ///   round-robin during `prepare()` and reset the counter from
    ///   `Pipeline::trim()`. iced_wgpu does call `trim()` at the end of each
    ///   `draw()` today, and its render loop only ever renders primitives it also
    ///   prepared — but `trim`'s own docs promise merely "normally called at the
    ///   end of a frame". Betting the panel↔rect mapping on an undocumented
    ///   ordering invariant of an external crate, where the failure mode is *panel
    ///   A rendering with panel B's corner radius*, is a bad trade for ~1% CPU.
    /// * **Pooling only the buffers doesn't help.** A bind group pins the specific
    ///   buffer it was built over, so reusing a buffer still means knowing which
    ///   buffer belongs to this panel — the identical identity problem — and a bind
    ///   group would still be built per frame. All of the complexity, half of the
    ///   win. (The reverse, pooling only bind groups, is impossible for the same
    ///   reason.)
    /// * **Dynamic offsets** — one persistent buffer + one bind group, with a
    ///   per-panel offset passed to `set_bind_group` — would remove both
    ///   allocations, but need `has_dynamic_offset: true`, hence a separate bind
    ///   group layout, hence a second copy of the blur render pipeline, plus
    ///   256-byte-aligned padding per panel, *and* they still leave unanswered
    ///   which offset belongs to which panel.
    ///
    /// So: correct churn beats a subtle pooling bug. If this ever does show up in a
    /// profile, fix the identity problem first — give panels a stable key at the
    /// widget level, so a pool can be keyed on something real — rather than bolting
    /// a round-robin allocator onto `trim()`.
    pub frosted_final_binding: Option<FilterBinding>,
    /// Pre-blur filters only (today just `Pencil`): this primitive's OWN binding
    /// for the pre-blur's second pass, built in `prepare()` and used by
    /// `render()` instead of a pipeline-owned one.
    ///
    /// Per-primitive for exactly the reason [`Self::frosted_final_binding`] is,
    /// and this field exists because that lesson was not applied here the first
    /// time. The uniform carries `panel_rect` + `corner_radius`, which differ per
    /// consumer, so one buffer on the pipeline is clobbered by the last consumer
    /// to `prepare()`. That is not hypothetical: the live preview and the filter
    /// picker's own Pencil swatch are both pre-blur consumers whenever Pencil is
    /// selected *and* the picker is open. The swatch prepares last, so the
    /// preview drew with the swatch's rect and radius — the SDF, which the
    /// preview disables with `corner_radius = 0`, switched on against a
    /// thumbnail-sized box and rejected every one of the preview's fragments,
    /// blanking it to whatever was underneath.
    ///
    /// The pre-blur *intermediate* stays shared, which is safe for a different
    /// reason: each consumer renders pass 1 into it and samples it in pass 2
    /// back-to-back within its own `render()`, so the reuse is serial within the
    /// encoder. Only the uniform outlives that window, because every
    /// `queue.write_buffer` in `prepare()` lands before any `render()` runs.
    pub preblur_binding: Option<FilterBinding>,
}

impl Default for FrameViewportData {
    fn default() -> Self {
        Self {
            frame: None,
            viewport: (
                0.0,
                0.0,
                0.0,
                crate::app::preview_geometry::BarInsets::default(),
            ),
            physical_bounds: None,
            uv_offset: (0.0, 0.0),
            uv_scale: (1.0, 1.0),
            frosted_full_physical: None,
            frosted_final_binding: None,
            preblur_binding: None,
        }
    }
}

/// Custom primitive for video rendering
#[derive(Debug)]
pub struct VideoPrimitive {
    pub video_id: u64,
    /// Combined frame and viewport data - single mutex for both
    pub data: Arc<Mutex<FrameViewportData>>,
    /// Filter type to apply
    pub filter_type: FilterType,
    /// Corner radius in pixels (0 = no rounding)
    pub corner_radius: f32,
    /// Mirror horizontally (selfie mode)
    pub mirror_horizontal: bool,
    /// Sensor rotation: 0=None, 1=90CW, 2=180, 3=270CW
    pub rotation: u32,
    /// Crop UV coordinates (u_min, v_min, u_max, v_max) - None means no cropping
    pub crop_uv: Option<(f32, f32, f32, f32)>,
    /// Zoom level (1.0 = no zoom, 2.0 = 2x zoom, etc.)
    pub zoom_level: f32,
    /// Theme background color (sRGB straight, RGBA) — passed to the blur
    /// shader so the letterbox in Contain / Fit mode is painted with the
    /// app background instead of leaking through to the COSMIC window bg.
    pub letterbox_color: [f32; 4],
    /// Dual-Kawase parameters for `VIDEO_ID_BLUR` / `VIDEO_ID_FROSTED`; ignored
    /// by every other `video_id`.
    ///
    /// The frosted backdrop sets this to `compositor_blur_params(theme.frosted)`
    /// — literally the entry cosmic-comp would pick for the same setting — and
    /// the transition blur leaves the default, [`TRANSITION_BLUR_PARAMS`].
    ///
    /// There is no conversion step and nothing to solve: the offset is in
    /// physical screen px, and the chain runs in physical screen px. This field
    /// replaces a `blur_radius` (in texels of whichever texture a pass happened
    /// to sample — two different sizes, neither a screen size) plus a
    /// `blur_target_sigma` that `prepare()` had to invert into one using the
    /// live frame-to-screen scale.
    pub blur_params: CompositorBlurParams,
}

impl Clone for VideoPrimitive {
    /// Copies the DATA, rather than sharing a handle to it: the clone gets its
    /// own `Arc<Mutex<FrameViewportData>>`, holding this primitive's frame and
    /// viewport as of now.
    ///
    /// This is what makes one primitive drawable as several independent
    /// instances, which is exactly what every caller wants: a widget clones its
    /// primitive once per `draw_primitive`, and `prepare()` writes that draw's
    /// own rect, radius and composite binding into the data behind it. Sharing
    /// the `Arc` made those writes collide — iced_wgpu runs every `prepare()`
    /// before any `render()`, so the last clone to prepare decided the panel rect
    /// for all of them, and `FrostedScrim`'s four bars would have rendered with
    /// the fourth bar's silhouette the moment the scrim was given a corner
    /// radius.
    ///
    /// Cloning is cheap and stays cheap: `VideoFrame` holds its pixels behind a
    /// `FrameData` (an `Arc`), so this copies a refcount, and the duplicate
    /// upload each clone would otherwise drive is deduplicated by
    /// `last_frame_ptr` — the same mechanism that already absorbs the filter
    /// picker's fifteen widgets.
    ///
    /// `frosted_final_binding` and `preblur_binding` are deliberately not carried
    /// across: each is the product of one draw's `prepare()`, so a clone taken
    /// before `prepare()` (which is every clone — widgets clone in `draw()`) must
    /// not inherit another draw's GPU state.
    fn clone(&self) -> Self {
        let data = match self.data.lock() {
            Ok(guard) => FrameViewportData {
                frame: guard.frame.clone(),
                viewport: guard.viewport,
                physical_bounds: guard.physical_bounds,
                uv_offset: guard.uv_offset,
                uv_scale: guard.uv_scale,
                frosted_full_physical: guard.frosted_full_physical,
                frosted_final_binding: None,
                preblur_binding: None,
            },
            Err(_) => FrameViewportData::default(),
        };

        Self {
            video_id: self.video_id,
            data: Arc::new(Mutex::new(data)),
            filter_type: self.filter_type,
            corner_radius: self.corner_radius,
            mirror_horizontal: self.mirror_horizontal,
            rotation: self.rotation,
            crop_uv: self.crop_uv,
            zoom_level: self.zoom_level,
            letterbox_color: self.letterbox_color,
            blur_params: self.blur_params,
        }
    }
}

impl VideoPrimitive {
    /// RGB dim for the final composite.
    ///
    /// Only the transition blur dims: it is a deliberate "the camera is
    /// switching" cue over a frozen frame. The frosted backdrop must not — it
    /// sits behind live UI chrome whose translucency already comes from the
    /// theme's container alpha, so dimming it too would just make every panel
    /// read as a grey smudge rather than as glass.
    fn dim_factor(&self) -> f32 {
        if self.video_id == VIDEO_ID_FROSTED {
            1.0
        } else {
            TRANSITION_BLUR_DIM
        }
    }

    /// Film-grain amplitude for the final composite. Only the frosted chrome is
    /// glass; see [`FROSTED_NOISE`].
    fn noise(&self) -> f32 {
        if self.video_id == VIDEO_ID_FROSTED {
            FROSTED_NOISE
        } else {
            0.0
        }
    }
}

/// The `VideoPipeline::textures` key a `video_id` uploads through.
///
/// It is NOT the identity: [`VIDEO_ID_FROSTED`] and [`VIDEO_ID_FILTER_PREVIEW`]
/// upload through [`VIDEO_ID_NORMAL`]'s entry, because all three are the same
/// pixels. Every frosted primitive is minted from the preview's own
/// `current_frame` `Arc` (see `frosted_backdrop::make_primitive`), and so is
/// every filter swatch (`filter_picker::view`), so keying the source texture by
/// `video_id` meant uploading that `Arc` two or three times per frame — ~20 MB of
/// bus traffic per extra copy, and an extra full-res texture each, on the target
/// phone's 2592x1940 back camera, for identical texels. The per-texture
/// `last_frame_ptr` dedup could not catch it: three keys, three
/// `last_frame_ptr`s.
///
/// The mapping is sound ONLY while the mapped ids genuinely carry the same
/// `Arc`. Feeding one of them a *different* frame would leave whichever consumer
/// uploads first to win and the other to render the wrong pixels, silently —
/// give a consumer its own frame only by giving it its own `video_id`.
///
/// Only the SOURCE texture is shared. The bindings, the blur targets, the
/// uniforms and the transforms all stay keyed by `video_id`, because those
/// genuinely differ per consumer (see [`BlurTargets`]). That is what makes the
/// picker's fifteen filters free of this: `bindings` is keyed by
/// `(video_id, filter_mode)`, so each swatch keeps its own filter uniform over
/// the one shared texture, exactly as it already did under its own id.
///
/// [`VIDEO_ID_BLUR`] keeps its own entry. It is the one consumer whose frame is
/// meant to be *frozen* — it survives the camera being torn down and restarted —
/// so it must not be re-pointed at whatever the live preview last uploaded.
fn source_texture_id(video_id: u64) -> u64 {
    match video_id {
        VIDEO_ID_FROSTED | VIDEO_ID_FILTER_PREVIEW => VIDEO_ID_NORMAL,
        other => other,
    }
}

/// Video texture (shared across filter variations, and across the `video_id`s
/// that [`source_texture_id`] maps together)
struct VideoTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    /// Pointer to last uploaded frame data (for deduplication)
    /// Multiple widgets with same video_id share an Arc, so same pointer = same frame
    last_frame_ptr: usize,
    /// How long this source's last upload took, for the "GPU is behind" skip in
    /// [`VideoPipeline::upload`].
    ///
    /// Per SOURCE, not per pipeline: the sources on screen at once (the frozen
    /// transition frame and the live preview) are different sizes and different
    /// formats, so one pipeline-wide figure had one consumer's cost decide the
    /// other's skip — they took turns being skipped, each dropping to half rate.
    last_upload_duration: std::time::Duration,
}

/// Filter-specific binding (viewport buffer + bind group)
/// Created per (video_id, filter_mode) combination to allow shared texture with different filters
#[derive(Debug)]
pub(crate) struct FilterBinding {
    bind_group: wgpu::BindGroup,
    viewport_buffer: wgpu::Buffer,
}

/// YUV conversion parameters uniform (must match shader struct)
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct YuvConvertParams {
    width: u32,
    height: u32,
    format: u32,
    y_stride: u32,
    uv_stride: u32,
    v_stride: u32,
    _pad: [u32; 2],
}

/// YUV textures for a video source (for YUV→RGBA conversion)
struct YuvTextures {
    tex_y: wgpu::Texture,
    tex_y_view: wgpu::TextureView,
    tex_uv: wgpu::Texture,
    tex_uv_view: wgpu::TextureView,
    tex_v: wgpu::Texture,
    tex_v_view: wgpu::TextureView,
    width: u32,
    height: u32,
    uv_width: u32,
    uv_height: u32,
    format: PixelFormat,
    /// Cached bind group for the YUV→RGBA compute shader.
    /// Invalidated when textures are recreated (dimension/format change).
    convert_bind_group: Option<wgpu::BindGroup>,
}

/// Custom pipeline for efficient video rendering
pub struct VideoPipeline {
    pipeline_rgba: wgpu::RenderPipeline,
    /// Blur chain pass 0: the transform into screen space (`video_shader_blur.wgsl`).
    pipeline_rgb_blur: wgpu::RenderPipeline,
    /// Dual-Kawase downsample / upsample (`video_shader_kawase.wgsl`).
    pipeline_kawase_down: wgpu::RenderPipeline,
    pipeline_kawase_up: wgpu::RenderPipeline,
    /// Per-panel composite: one tap + SDF + dim + grain (`video_shader_frosted.wgsl`).
    pipeline_frosted_composite: wgpu::RenderPipeline,
    pipeline_preblur: wgpu::RenderPipeline, // Lightweight blur for filter pre-processing
    bind_group_layout_rgba: wgpu::BindGroupLayout,
    bind_group_layout_rgb: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    // Shared textures by video_id (single upload per source)
    textures: std::collections::HashMap<u64, VideoTexture>,
    // Per-filter bindings keyed by (video_id, filter_mode)
    // Allows shared texture with different filter uniforms
    bindings: std::collections::HashMap<(u64, u32), FilterBinding>,
    /// Multi-pass blur state, keyed by `video_id` — see [`BlurTargets`].
    ///
    /// Using RwLock for interior mutability (Sync-safe) since render() takes &self.
    blur_targets: std::sync::RwLock<std::collections::HashMap<u64, BlurTargets>>,
    // Filter pre-blur intermediate texture for multi-pass filters (e.g. Pencil)
    // Full resolution — used to store the pre-blurred frame for spatial filters
    filter_preblur_intermediate: std::sync::RwLock<Option<PreblurIntermediate>>,
    // GPU timing tracking to detect and handle stalls (per source: see
    // `VideoTexture::last_upload_duration`)
    frames_skipped: std::sync::atomic::AtomicU32,
    // YUV→RGBA conversion compute pipeline
    yuv_compute_pipeline: Option<wgpu::ComputePipeline>,
    yuv_bind_group_layout: Option<wgpu::BindGroupLayout>,
    yuv_uniform_buffer: Option<wgpu::Buffer>,
    // YUV textures per video_id
    yuv_textures: std::collections::HashMap<u64, YuvTextures>,
    // Store the texture format for use in prepare
    output_format: wgpu::TextureFormat,
}

/// Intermediate texture for the filter pre-blur (full resolution).
///
/// It carries no bind group of its own: the preblur pass renders INTO it through
/// the source's `FilterBinding`, and the pass that samples it back out uses the
/// consuming primitive's own [`FrameViewportData::preblur_binding`]. (It used to
/// hold a bind group + uniform buffer, which were dead — they only looked live
/// because the blur chain's intermediates shared this type. They no longer do.)
///
/// The texture itself is shared by every pre-blur consumer, which is safe only
/// because each renders into it and samples it back within its own `render()`:
/// the reuse is serial within the encoder. See
/// [`FrameViewportData::preblur_binding`] for what could NOT be shared, and why.
struct PreblurIntermediate {
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

/// One step of the Kawase ping-pong: the bind group it samples through, and the
/// uniform buffer carrying that step's sub-rect size and offset.
///
/// Step `k` samples `view_a` when `k` is even and `view_b` when it is odd, and
/// renders into the other one. Since a run is always `2 * passes` steps — an even
/// number, whatever the frost level — the result always lands back in `view_a`,
/// which is what makes `composite_binding` a single fixed binding rather than
/// something that has to be re-pointed per level.
struct KawaseStep {
    bind_group: wgpu::BindGroup,
    viewport_buffer: wgpu::Buffer,
}

/// The whole blur state for ONE `video_id`: the two screen-resolution ping-pong
/// textures, the per-step bindings, the final composite's binding, and the
/// "the chain already ran this frame" flag.
///
/// # Why the textures are SCREEN resolution
///
/// They used to be frame/4, because the kernel worked in frame space and the
/// final pass did the cover/contain fit. The chain now does the opposite: pass 0
/// applies every transform and lands in SCREEN space at SCREEN resolution, the
/// Kawase runs there, and the composite is an identity blit. That is what makes
/// `BLUR_PARAMS`'s offsets — which are authored in physical screen px — usable
/// verbatim, with no scale conversion anywhere; the conversion apparatus was
/// both the complexity and the source of the banding.
///
/// It costs memory: on a 1080x2340 phone the pair is ~20 MB against ~2.5 MB at
/// frame/4. That is the deliberate trade. The compute does not scale the same way
/// — Kawase halves after the first downsample, so the extra work is roughly one
/// full-res 5-tap and one full-res 8-tap — and the alternative (Kawase on
/// quarter-res intermediates) would put the offsets back into a converted unit
/// and re-open exactly the question this design closes.
///
/// # Why this is per-`video_id` and not a singleton
///
/// Two consumers drive the chain with DIFFERENT parameterizations:
///
/// * [`VIDEO_ID_BLUR`] — the transition / HDR+ blur: [`TRANSITION_BLUR_PARAMS`]
///   and [`TRANSITION_BLUR_DIM`], no grain.
/// * [`VIDEO_ID_FROSTED`] — the live frosted chrome: the compositor's own
///   (passes, offset) for the theme's frost level, no dim, [`FROSTED_NOISE`].
///
/// Every step here owns the *uniform buffer* it is configured with, so one shared
/// set meant both consumers wrote the same buffers. iced_wgpu runs EVERY
/// `prepare()` before ANY `render()`, so whenever both were on screen in the same
/// frame (open a picker, then switch camera or let an HDR+ burst finish) the last
/// `prepare()` won, whichever primitive rendered first ran the chain with those
/// uniforms and set the shared `cached` flag, and the second primitive reused
/// that result wholesale. Both got one parameterization — the wrong one for at
/// least one of them. Keying by `video_id` gives each its own textures, its own
/// uniforms and — the point — its own `cached` flag.
///
/// # Why exactly one entry per `video_id` is enough
///
/// All [`VIDEO_ID_FROSTED`] primitives legitimately share one parameterization:
/// the frost level is global (theme-derived), every frosted primitive is built
/// from the same preview config (same rotation / crop / zoom / fit blend / bars),
/// and the chain spans the FULL preview rather than any one panel rect — so
/// pass 0 and the Kawase are identical for all of them. Their per-panel data
/// (rect + corner radius + this frame's `noise`/`dim`) lives in a separate
/// per-primitive composite binding, not in these buffers. So the map holds at
/// most 2 entries, and within a frame the chain still runs exactly ONCE per
/// `video_id` no matter how many panels and scrim bars are drawn.
///
/// That sharing puts a hard constraint on the chain's geometry: it is a property
/// of the `video_id`, so it must NOT be sourced from any one consumer's widget
/// bounds. It used to be, and the consumers did not agree — a `FrostedContainer` reports its layer rect, a `FrostedScrim` its
/// own inset layout bounds. Each `prepare()` then re-sized these textures out
/// from under every consumer that had already prepared, orphaning their
/// composite bind groups onto textures the chain never rendered into: the black
/// scrim bars. `prepare()` now takes the RENDER TARGET, the one rect every
/// consumer sees identically, so the invariant holds by construction. See
/// `frosted_consumers_that_disagree_do_not_blank_the_scrim`.
///
/// Nor can the two ids share one PAIR of textures while keeping their own
/// uniforms: `view_a` is not just scratch, it is where a run's result waits to be
/// composited, and the composites do not follow their own chain — a frosted
/// panel whose `cached` is already set does nothing but blit `view_a`. One shared
/// pair would make that blit read whichever chain ran last. It happens to work
/// out under today's draw order (preview first, chrome after), but nothing
/// enforces that order, and the failure is silent wrong pixels — the same shape
/// as the black scrim bars below.
///
/// Entries are never evicted, so both ids resident costs ~40 MB at 1080x2340.
/// There are at most two of them, both belong to effects that recur constantly,
/// and dropping one would only buy that memory back until the next transition —
/// at the cost of a realloc and a cold `cached` flag on the frame that needs it
/// most. There is also nowhere honest to do it from: `prepare()` only runs for
/// primitives that exist, so an idle `VIDEO_ID_BLUR` never gets a turn, and
/// evicting it from ANOTHER id's `prepare()` on a staleness counter would couple
/// two consumers that share nothing else.
struct BlurTargets {
    /// Pass 0 renders here, and the ping-pong ends here. The composite samples it.
    view_a: wgpu::TextureView,
    view_b: wgpu::TextureView,
    width: u32,
    height: u32,
    /// `2 * MAX_KAWASE_PASSES` entries; a run uses the first `2 * passes`.
    steps: Vec<KawaseStep>,
    /// Over `view_a`, for consumers with no per-panel data (the transition blur).
    composite_binding: FilterBinding,
    /// Set after this `video_id`'s chain runs; cleared by `prepare()` when a new
    /// frame arrives. Within one frame it is what keeps the expensive passes from
    /// re-running per panel / per scrim bar.
    cached: std::sync::atomic::AtomicBool,
}

impl VideoPrimitive {
    pub fn new(video_id: u64) -> Self {
        Self {
            video_id,
            data: Arc::new(Mutex::new(FrameViewportData::default())),
            filter_type: FilterType::Standard,
            corner_radius: 0.0,
            mirror_horizontal: false,
            rotation: 0,
            crop_uv: None,
            zoom_level: 1.0,
            // Black is a sensible default if no widget overrides it (e.g.
            // headless tests). The real bg color is plumbed in via
            // `VideoWidgetConfig::letterbox_color` from the active theme.
            letterbox_color: [0.0, 0.0, 0.0, 1.0],
            // The transition blur's tuned look, carried across the rewrite at
            // equal on-screen thickness; the frosted backdrop overrides it with
            // the compositor's own entry.
            blur_params: TRANSITION_BLUR_PARAMS,
        }
    }

    pub fn update_frame(&self, frame: VideoFrame) {
        if let Ok(mut guard) = self.data.lock() {
            guard.frame = Some(frame);
        }
    }

    pub fn update_viewport(
        &self,
        width: f32,
        height: f32,
        cover_blend: f32,
        insets: crate::app::preview_geometry::BarInsets,
    ) {
        if let Ok(mut guard) = self.data.lock() {
            guard.viewport = (width, height, cover_blend, insets);
        }
    }
}

impl PipelineTrait for VideoPipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        VideoPipeline::new(device, format)
    }

    fn trim(&mut self) {
        // No-op: we manage texture lifecycle ourselves via video_id keying.
        // Clearing here would destroy live textures and cause flickering.
    }
}

impl PrimitiveTrait for VideoPrimitive {
    type Pipeline = VideoPipeline;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        viewport: &Viewport,
    ) {
        use std::time::Instant;
        let prepare_start = Instant::now();

        // Seed the compute-pipeline singleton with the renderer's own device
        // and queue on the very first prepare. After this, burst-mode and
        // virtual-camera filter work share the same GPU context as the iced
        // renderer instead of opening a separate `wgpu::Instance`.
        // Subsequent calls are cheap no-ops (the OnceCell guards them).
        crate::gpu::try_seed_shared_gpu_from_renderer(
            std::sync::Arc::new(device.clone()),
            std::sync::Arc::new(queue.clone()),
        );

        // Calculate physical bounds from logical bounds using scale factor
        // Then clamp to render target to ensure valid viewport rect
        let scale = viewport.scale_factor() as f32;
        let render_target = viewport.physical_size();

        let raw_physical_bounds = (
            bounds.x * scale,
            bounds.y * scale,
            bounds.width * scale,
            bounds.height * scale,
        );

        // Clamp physical bounds to render target to avoid wgpu validation errors
        let clamped_x = raw_physical_bounds.0.max(0.0);
        let clamped_y = raw_physical_bounds.1.max(0.0);
        let clamped_w = ((raw_physical_bounds.0 + raw_physical_bounds.2)
            .min(render_target.width as f32)
            - clamped_x)
            .max(0.0);
        let clamped_h = ((raw_physical_bounds.1 + raw_physical_bounds.3)
            .min(render_target.height as f32)
            - clamped_y)
            .max(0.0);

        let clamped_physical_bounds = (clamped_x, clamped_y, clamped_w, clamped_h);

        // Calculate UV offset/scale to compensate for clamping
        // This ensures the visible portion maps to correct texture coordinates
        let (uv_offset, uv_scale) = if raw_physical_bounds.2 > 0.0 && raw_physical_bounds.3 > 0.0 {
            let uv_offset_x = (clamped_x - raw_physical_bounds.0) / raw_physical_bounds.2;
            let uv_offset_y = (clamped_y - raw_physical_bounds.1) / raw_physical_bounds.3;
            let uv_scale_x = clamped_w / raw_physical_bounds.2;
            let uv_scale_y = clamped_h / raw_physical_bounds.3;
            ((uv_offset_x, uv_offset_y), (uv_scale_x, uv_scale_y))
        } else {
            ((0.0, 0.0), (1.0, 1.0))
        };

        // Take frame and viewport data with brief lock, then release before GPU ops
        // Also store clamped physical bounds and UV adjustment for use in render()
        #[allow(clippy::type_complexity)]
        let (frame_opt, viewport_data, stored_uv_offset, stored_uv_scale, blur_rect) = {
            if let Ok(mut data_guard) = self.data.lock() {
                data_guard.physical_bounds = Some(clamped_physical_bounds);
                data_guard.uv_offset = uv_offset;
                data_guard.uv_scale = uv_scale;
                // For the frosted backdrop: the rect render() positions the blur
                // viewport over while scissoring to the panel.
                //
                // This is the RENDER TARGET, not a rect reported by the widget,
                // and that is load-bearing. ONE chain
                // is shared by every `VIDEO_ID_FROSTED` consumer (see
                // `BlurTargets`), so its geometry has to be a property of the
                // video_id — but the consumers do not agree on it. A
                // `FrostedContainer` reports its layer rect (the whole window);
                // a `FrostedScrim` reports its own layout bounds, which its
                // parent had inset by a pixel. Different rects round to different
                // ping-pong texture sizes, so each consumer's prepare() re-sized
                // the targets out from under every consumer that had already
                // prepared, leaving their composite bind groups pointing at
                // orphaned textures the chain never rendered into — i.e. at
                // zeroes. That is the black scrim bar: the LAST consumer to
                // prepare won the textures, so the toolbox and chips (prepared
                // late) blurred correctly while the bars (prepared early)
                // composited a blank texture. See
                // `frosted_consumers_that_disagree_do_not_blank_the_scrim`.
                //
                // The render target is the one rect every consumer sees
                // identically, so sourcing it here makes the shared chain's
                // geometry agree BY CONSTRUCTION rather than by hoping two
                // widgets in different layers derive the same rect. It is also
                // what the working consumer already reported, so the panels'
                // blur is unchanged; the bars simply join them on it.
                //
                // Both fit modes lay the preview out into the full window, so the
                // whole window is exactly the rect the blur has to span.
                if self.video_id == VIDEO_ID_FROSTED {
                    data_guard.frosted_full_physical = Some((
                        0.0,
                        0.0,
                        render_target.width as f32,
                        render_target.height as f32,
                    ));
                }
                // The rect the blur chain lives in: the FULL preview, in
                // physical px. It sizes the Kawase targets and is the viewport
                // the composite blits through, so the two agree by construction
                // and the composite's texture mapping is a plain identity.
                let blur_rect = if self.video_id == VIDEO_ID_FROSTED {
                    data_guard
                        .frosted_full_physical
                        .unwrap_or(clamped_physical_bounds)
                } else {
                    clamped_physical_bounds
                };
                (
                    data_guard.frame.take(),
                    data_guard.viewport,
                    data_guard.uv_offset,
                    data_guard.uv_scale,
                    blur_rect,
                )
            } else {
                return;
            }
        };
        // Mutex released here - GPU operations won't block other threads

        let lock_time = prepare_start.elapsed();

        {
            // The Kawase ping-pong targets follow the on-screen preview rect, so
            // they are ensured on every prepare rather than only when a frame
            // arrives — a window resize must resize them even while the
            // transition blur is sitting on a frozen frame. Cheap: it early-outs
            // unless the size actually changed.
            if self.video_id == VIDEO_ID_BLUR || self.video_id == VIDEO_ID_FROSTED {
                pipeline.ensure_blur_targets(
                    self.video_id,
                    device,
                    blur_rect.2.round().max(1.0) as u32,
                    blur_rect.3.round().max(1.0) as u32,
                    pipeline.output_format,
                );
            }

            // Upload frame if available
            if let Some(frame) = frame_opt {
                let upload_start = Instant::now();

                if self.video_id == VIDEO_ID_BLUR || self.video_id == VIDEO_ID_FROSTED {
                    // Invalidate THIS video_id's blur cache so the freshly
                    // uploaded frame gets re-blurred; the other consumer's cached
                    // blur is untouched and still valid (see `BlurTargets`).
                    //
                    // Unconditional, and deliberately so. A view build hands every
                    // primitive a frame, so this fires on every redraw — including
                    // one with no NEW camera frame behind it, where the chain then
                    // re-runs over texels `upload` just dedup'd away. Gating it on
                    // frame identity alone would be WRONG: `cover_blend` and zoom
                    // animate the TRANSFORMS, and pass 0 bakes those into `view_a`,
                    // so a fit or zoom animation must re-blur an unchanged frame or
                    // the backdrop freezes mid-animation while the sharp preview
                    // moves under it. A correct gate keys on frame identity AND
                    // every input to pass 0 (rotation, mirror, crop, zoom, blend,
                    // bar heights, letterbox, `blur_params`, `blur_rect`).
                    //
                    // Not worth it: the camera subscription pushes a frame at ~30fps
                    // and each one drives a redraw, so redraws without a new frame
                    // are the rare interleaved ones (a timer tick, a cursor move) —
                    // at most a handful of extra chains per second, against a
                    // multi-field cache key whose every field is a chance to freeze
                    // the backdrop the way this comment exists to prevent.
                    pipeline.invalidate_blur_cache(self.video_id);
                }
                // For filters that need pre-blur, ensure the intermediate texture exists
                if self.filter_type.needs_preblur()
                    && self.video_id != VIDEO_ID_BLUR
                    && self.video_id != VIDEO_ID_FROSTED
                {
                    pipeline.ensure_filter_preblur_intermediate(
                        device,
                        frame.width,
                        frame.height,
                        pipeline.output_format,
                    );
                }

                pipeline.upload(device, queue, frame);

                let upload_time = upload_start.elapsed();
                if upload_time.as_millis() > 16 {
                    tracing::warn!(
                        upload_ms = upload_time.as_millis(),
                        lock_ms = lock_time.as_millis(),
                        "GPU upload took longer than frame period - causing stutter"
                    );
                }
            }

            // Update viewport uniform data (using viewport_data captured before releasing lock)
            let (mut width, mut height, cover_blend, insets) = viewport_data;

            // Same reason `frosted_full_physical` is the render target above: the
            // pass-0 uniform lives in the binding keyed by `(video_id,
            // filter_mode)`, so every `VIDEO_ID_FROSTED` consumer shares ONE copy
            // of it and the last prepare() of the frame decides the fit for all of
            // them. The widgets report sizes that differ by a pixel, so leaving
            // this widget-sourced made the blur's fit depend on draw order. Take
            // the render target, which they all see identically.
            if self.video_id == VIDEO_ID_FROSTED {
                width = render_target.width as f32 / scale;
                height = render_target.height as f32 / scale;
            }

            // Cover blend: 0.0 = Contain, 1.0 = Cover (may be intermediate during animation)
            let content_fit_mode = cover_blend;

            let filter_mode = self.filter_type.gpu_filter_code();

            let (panel_rect, corner_radius_px) =
                corner_sdf_params(raw_physical_bounds, self.corner_radius, scale);

            // Get or create binding for this (video_id, filter_mode) combination
            // This allows sharing the source texture while having per-filter uniforms
            pipeline.get_or_create_binding(device, self.video_id, filter_mode);

            // Crop UV values (default to the full image if not set).
            let (crop_min, crop_max) = self.crop_uv.map_or(
                ([0.0f32, 0.0], [1.0f32, 1.0]),
                |(u_min, v_min, u_max, v_max)| ([u_min, v_min], [u_max, v_max]),
            );

            // Update viewport buffer for this specific filter binding
            let binding_key = (self.video_id, filter_mode);
            if let Some(binding) = pipeline.bindings.get(&binding_key) {
                if self.video_id == VIDEO_ID_BLUR || self.video_id == VIDEO_ID_FROSTED {
                    // PASS 0 — the transform pass. It is now configured exactly
                    // like the sharp preview's own uniform (`video_shader.wgsl`),
                    // because it now does exactly the preview's job: the real
                    // viewport, the live cover/contain blend, the bars, the crop,
                    // the zoom, the rotation, the mirror, the filter, the
                    // letterbox — all of it, into a screen-resolution target.
                    //
                    // This is the reversal at the heart of the rewrite. Pass 1
                    // used to be told `viewport_size = the texture's own dims` and
                    // `content_fit_mode = 0` (Contain), i.e. "render the frame
                    // into a frame-shaped box"; the fit onto the screen was the
                    // LAST pass's job, after the blurring. Both halves of that
                    // were wrong: it made the blur happen in frame texels (a unit
                    // no compositor parameter is written in), and it made pass 1
                    // sample sharp sensor data through a wide kernel (the banding).
                    //
                    // `viewport_size` is LOGICAL px while the target is physical;
                    // that is not a bug. The fit math only ever produces a
                    // dimensionless UV scale — the px unit cancels between the
                    // viewport and the zoom — so it is scale-invariant, and the
                    // bars arrive in logical px from the widget.
                    let pass0_uniform = ViewportUniform {
                        viewport_size: [width, height],
                        content_fit_mode,
                        filter_mode,
                        mirror_horizontal: if self.mirror_horizontal { 1 } else { 0 },
                        crop_uv_min: crop_min,
                        crop_uv_max: crop_max,
                        // Digital zoom is baked in HERE, in the one pass that
                        // samples the source frame — exactly like mirror /
                        // rotation / crop above, and for the same reason: every
                        // pass downstream works on this pass's output, which
                        // already has it, so they leave `zoom_level` at the 1.0
                        // default and must keep doing so or the zoom compounds.
                        zoom_level: self.zoom_level,
                        rotation: self.rotation,
                        bar_top_height: insets.top,
                        bar_bottom_height: insets.bottom,
                        bar_left_width: insets.left,
                        bar_right_width: insets.right,
                        letterbox_color: self.letterbox_color,
                        ..Default::default()
                    };
                    queue.write_buffer(
                        &binding.viewport_buffer,
                        0,
                        bytemuck::cast_slice(&[pass0_uniform]),
                    );
                } else {
                    // Regular video: use requested mode with UV adjustment for clipping
                    let uniform_data = ViewportUniform {
                        viewport_size: [width, height],
                        content_fit_mode,
                        filter_mode,
                        corner_radius: corner_radius_px,
                        panel_rect,
                        mirror_horizontal: if self.mirror_horizontal { 1 } else { 0 },
                        uv_offset: [stored_uv_offset.0, stored_uv_offset.1],
                        uv_scale: [stored_uv_scale.0, stored_uv_scale.1],
                        crop_uv_min: crop_min,
                        crop_uv_max: crop_max,
                        zoom_level: self.zoom_level,
                        rotation: self.rotation,
                        bar_top_height: insets.top,
                        bar_bottom_height: insets.bottom,
                        bar_left_width: insets.left,
                        bar_right_width: insets.right,
                        letterbox_color: self.letterbox_color,
                        ..Default::default()
                    };
                    queue.write_buffer(
                        &binding.viewport_buffer,
                        0,
                        bytemuck::cast_slice(&[uniform_data]),
                    );
                }

                // Update filter pre-blur intermediate viewport + create binding if needed.
                // The preblur intermediate already has mirror/rotation/crop baked in
                // from the preblur pass, so the second pass uses identity transforms
                // but keeps the filter_mode and screen viewport settings.
                if self.filter_type.needs_preblur()
                    && self.video_id != VIDEO_ID_BLUR
                    && self.video_id != VIDEO_ID_FROSTED
                    && let Some(intermediate) = pipeline
                        .filter_preblur_intermediate
                        .read()
                        .unwrap()
                        .as_ref()
                {
                    // Update the preblur binding's viewport uniform.
                    // The intermediate already has cover-fit baked in (pass 1
                    // wrote the cover-fit-to-widget view into a frame-sized
                    // texture). Pass 2 must sample it 1:1 across the widget;
                    // we get scale = (1, 1) by telling the shader its
                    // "viewport" matches the intermediate's pixel dimensions,
                    // so the Contain math degenerates to identity instead of
                    // letterboxing the intermediate inside the widget.
                    //
                    // Which is why the corner SDF must not be derived from
                    // `viewport_size`: it would cut the corners from the
                    // intermediate's extent — a frame-sized box, not the widget —
                    // so a radius meant for a small swatch shrinks to a fraction
                    // of a pixel and Pencil (the one filter with a pre-blur) is
                    // the one swatch that renders square. It takes its rect from
                    // `panel_rect` instead, which this lie cannot reach.
                    // Mirror/rotation/crop/zoom are already baked into the
                    // intermediate by the preblur pass, so everything but
                    // the filter and the matched viewport stays at identity.
                    let pb_uniform = ViewportUniform {
                        viewport_size: [intermediate.width as f32, intermediate.height as f32],
                        content_fit_mode: 0.0, // Contain — identity given the matched viewport_size
                        filter_mode,
                        corner_radius: corner_radius_px,
                        panel_rect,
                        letterbox_color: self.letterbox_color,
                        ..Default::default()
                    };

                    if let Ok(mut data_guard) = self.data.lock() {
                        // Rebuilt each frame, per primitive: `panel_rect` and
                        // `corner_radius` are this consumer's own, and a buffer on
                        // the pipeline would be overwritten by the next consumer to
                        // prepare (see `FrameViewportData::preblur_binding`).
                        // Rebuilding also keeps the bind group's borrow of
                        // `intermediate.view` from going stale across a resize.
                        let pb_viewport_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("camera preblur filter viewport buffer"),
                            size: std::mem::size_of::<ViewportUniform>() as u64,
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        let pb_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("camera preblur filter bind group"),
                            layout: &pipeline.bind_group_layout_rgba,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        &intermediate.view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(&pipeline.sampler),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: pb_viewport_buffer.as_entire_binding(),
                                },
                            ],
                        });
                        queue.write_buffer(
                            &pb_viewport_buffer,
                            0,
                            bytemuck::cast_slice(&[pb_uniform]),
                        );
                        data_guard.preblur_binding = Some(FilterBinding {
                            bind_group: pb_bind_group,
                            viewport_buffer: pb_viewport_buffer,
                        });
                    }
                }

                // Configure THIS video_id's Kawase steps and its composite.
                //
                // The buffers below carry this primitive's (passes, offset), dim
                // and grain, and the other blur consumer's parameterization must
                // not land in them (see `BlurTargets`).
                //
                // This takes `blur_targets` and then `self.data`, where `render()`
                // takes them in the opposite order. That is a lock-order inversion
                // on paper, and it cannot deadlock in practice: both run on the one
                // render thread, so the two orders are never live at once. Should
                // either ever move off it, this is the site to normalize — take
                // `self.data` first here, as `render()` does.
                let blur_targets_guard = pipeline.blur_targets.read().unwrap();

                if let Some(targets) = blur_targets_guard.get(&self.video_id) {
                    let (tex_w, tex_h) = (targets.width, targets.height);
                    let passes = effective_kawase_passes(tex_w, tex_h, self.blur_params.passes)
                        .min(MAX_KAWASE_PASSES);
                    let base_offset = self.blur_params.offset;

                    // Upstream's `render_blur` (blur_effect.rs), transcribed.
                    // Down pass `i` reads the sub-rect `[0, W>>i]` with
                    // `half_pixel = 0.5/(W>>i)` and `offset/2^i`; the up passes are
                    // its mirror image, walking `k = passes..=1`. `sub_size` is
                    // upstream's `adjusted_tex_size`, and `uv_scale` is what turns
                    // our viewport-relative `tex_coords` back into upstream's
                    // full-texture-normalized `v_coords` (see the shader header).
                    let write_step = |step: usize, level: u32, offset: f64| {
                        let Some(step) = targets.steps.get(step) else {
                            return;
                        };
                        let sub_w = (tex_w >> level).max(1);
                        let sub_h = (tex_h >> level).max(1);
                        let uniform = ViewportUniform {
                            viewport_size: [sub_w as f32, sub_h as f32],
                            uv_scale: [sub_w as f32 / tex_w as f32, sub_h as f32 / tex_h as f32],
                            kawase_offset: offset as f32,
                            ..Default::default()
                        };
                        queue.write_buffer(
                            &step.viewport_buffer,
                            0,
                            bytemuck::cast_slice(&[uniform]),
                        );
                    };
                    for i in 0..passes {
                        write_step(i as usize, i, base_offset / f64::from(1u32 << i));
                    }
                    for j in 0..passes {
                        let level = passes - j;
                        write_step(
                            (passes + j) as usize,
                            level,
                            base_offset / f64::from(1u32 << level),
                        );
                    }

                    // Where the image ends and the letterbox begins, for the
                    // composite to cut the latter back out (see
                    // `content_rect_uv`). Derived from exactly the values pass 0
                    // was configured with just above — the same viewport, blend,
                    // bars, crop and rotation — plus the source texture's own
                    // dimensions, which is what pass 0 reads via
                    // `textureDimensions`. Taken from the uploaded texture rather
                    // than from the frame, so it is literally the extent the
                    // shader sampled.
                    let content_rect = pipeline
                        .textures
                        .get(&source_texture_id(self.video_id))
                        .map_or(FULL_CONTENT_RECT, |tex| {
                            content_rect_uv_with_insets(
                                [width, height],
                                (tex.width, tex.height),
                                self.rotation,
                                crop_min,
                                crop_max,
                                content_fit_mode,
                                insets,
                            )
                        });

                    // The final composite: one tap, plus dim, grain, the letterbox
                    // cut and (for the frosted chrome) the rounded silhouette. This
                    // is the shared binding, used by consumers with no per-panel
                    // data — i.e. the transition blur.
                    let mut composite_uniform = ViewportUniform {
                        viewport_size: [width, height],
                        dim_factor: self.dim_factor(),
                        noise: self.noise(),
                        content_rect,
                        ..Default::default()
                    };
                    queue.write_buffer(
                        &targets.composite_binding.viewport_buffer,
                        0,
                        bytemuck::cast_slice(&[composite_uniform]),
                    );

                    // The frosted backdrop rounds its own corners in the shader,
                    // which needs this panel's rect + radius — per-panel data the
                    // buffer above cannot carry, since every frosted panel shares
                    // it and the last prepare() would win. So give this primitive
                    // its own buffer + bind group over the frosted entry's blurred
                    // texture — `targets` here is keyed by `self.video_id`, so this
                    // never picks up the transition blur's result — and the chain
                    // still runs once per frame, only this blit differs.
                    if self.video_id == VIDEO_ID_FROSTED {
                        composite_uniform.panel_rect = panel_rect;
                        composite_uniform.corner_radius = corner_radius_px;

                        if let Ok(mut data_guard) = self.data.lock() {
                            // Rebuilt each frame: the primitive (and its data) is
                            // recreated on every view build, so the binding can
                            // never outlive a resize of the ping-pong textures —
                            // the bind group below borrows `targets.view_a`, and
                            // this is what keeps that reference from going stale.
                            //
                            // Yes, this allocates a buffer + bind group per panel
                            // per frame. That is a known, accepted cost, not an
                            // oversight: see `FrameViewportData::frosted_final_binding`
                            // for the pooling schemes that were considered and why
                            // each was rejected. Please read that before changing it.
                            let viewport_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("camera frosted composite viewport buffer"),
                                size: std::mem::size_of::<ViewportUniform>() as u64,
                                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            });
                            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("camera frosted composite bind group"),
                                layout: &pipeline.bind_group_layout_rgb,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(
                                            &targets.view_a,
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::Sampler(&pipeline.sampler),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: viewport_buffer.as_entire_binding(),
                                    },
                                ],
                            });
                            queue.write_buffer(
                                &viewport_buffer,
                                0,
                                bytemuck::cast_slice(&[composite_uniform]),
                            );
                            data_guard.frosted_final_binding = Some(FilterBinding {
                                bind_group,
                                viewport_buffer,
                            });
                        }
                    }
                }
            }
        }
    }

    fn render(
        &self,
        _pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        // Convert filter_type to filter_mode for binding lookup
        let filter_mode = self.filter_type.gpu_filter_code();

        // Hold the lock across the render call: it yields both the viewport rect
        // and (for the frosted backdrop) this primitive's own final-pass binding,
        // which is borrowed rather than cloned. `VideoPipeline::render` never
        // touches `self.data`, so there is nothing to deadlock against.
        let data_guard = self.data.lock().ok();

        // Use stored physical bounds for viewport (prevents distortion in scrollable contexts).
        // For the frosted backdrop, the viewport must span the FULL preview
        // geometry (so the blur aligns with the sharp preview) while the scissor
        // (clip_bounds) restricts drawing to the panel rect — so prefer the
        // stored full-preview physical bounds when present.
        let widget_bounds = data_guard
            .as_ref()
            .and_then(|guard| {
                if self.video_id == VIDEO_ID_FROSTED {
                    guard.frosted_full_physical.or(guard.physical_bounds)
                } else {
                    guard.physical_bounds
                }
            })
            .unwrap_or((
                clip_bounds.x as f32,
                clip_bounds.y as f32,
                clip_bounds.width as f32,
                clip_bounds.height as f32,
            ));

        let frosted_final_binding = data_guard
            .as_ref()
            .and_then(|guard| guard.frosted_final_binding.as_ref());

        let preblur_binding = data_guard
            .as_ref()
            .and_then(|guard| guard.preblur_binding.as_ref());

        _pipeline.render(
            self.video_id,
            filter_mode,
            self.filter_type.needs_preblur(),
            self.blur_params,
            encoder,
            target,
            clip_bounds,
            widget_bounds,
            frosted_final_binding,
            preblur_binding,
        );
    }
}

impl VideoPipeline {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        // ===== Video Pipeline =====
        // Shader for video rendering with shared filter functions
        let shader_source = format!(
            "{}\n{}\n{}\n{}",
            crate::shaders::FILTER_FUNCTIONS,
            crate::shaders::TEXTURE_FILTER_FUNCTIONS,
            crate::shaders::GEOMETRY_FUNCTIONS,
            include_str!("video_shader.wgsl")
        );
        let shader_rgba = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("camera video shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Bind group layout for video texture, sampler, and viewport
        let bind_group_layout_rgba =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera video bind group layout"),
                entries: &[
                    // RGBA texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Viewport uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout_rgba = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("camera video pipeline layout"),
            bind_group_layouts: &[&bind_group_layout_rgba],
            immediate_size: 0,
        });

        let pipeline_rgba = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("camera video pipeline"),
            layout: Some(&pipeline_layout_rgba),
            vertex: wgpu::VertexState {
                module: &shader_rgba,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_rgba,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        // ===== Blur Pipeline (for multi-pass blur) =====
        let shader_blur_source = format!(
            "{}\n{}\n{}",
            crate::shaders::FILTER_FUNCTIONS,
            crate::shaders::TEXTURE_FILTER_FUNCTIONS,
            include_str!("video_shader_blur.wgsl")
        );
        let shader_rgb_blur = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("camera blur shader"),
            source: wgpu::ShaderSource::Wgsl(shader_blur_source.into()),
        });

        // Bind group layout for blur texture, sampler, and viewport
        let bind_group_layout_rgb =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera blur bind group layout"),
                entries: &[
                    // RGB texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Viewport uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout_rgb = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("camera blur pipeline layout"),
            bind_group_layouts: &[&bind_group_layout_rgb],
            immediate_size: 0,
        });

        let pipeline_rgb_blur = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("camera blur pipeline"),
            layout: Some(&pipeline_layout_rgb),
            vertex: wgpu::VertexState {
                module: &shader_rgb_blur,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_rgb_blur,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    // Alpha blending (not REPLACE) so the frosted backdrop's
                    // antialiased corners composite over the sharp preview
                    // instead of overwriting it. Every other blur pass emits
                    // alpha = 1, where blending is identical to REPLACE, so the
                    // transition blur is unaffected.
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        // ===== Dual-Kawase Pipelines =====
        // A verbatim port of cosmic-comp's blur kernels; see the shader header.
        let shader_kawase = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("camera kawase shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("video_shader_kawase.wgsl").into()),
        });
        let make_kawase = |label: &str, entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout_rgb),
                vertex: wgpu::VertexState {
                    module: &shader_kawase,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_kawase,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        // REPLACE, not blending: these passes write the alpha
                        // channel as DATA (it is the region mask the kernels
                        // normalize by), so it must land in the target verbatim
                        // rather than be composited away. The target is cleared to
                        // transparent each step, so nothing is lost.
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline_kawase_down = make_kawase("camera kawase down pipeline", "fs_down");
        let pipeline_kawase_up = make_kawase("camera kawase up pipeline", "fs_up");

        // ===== Frosted Composite Pipeline =====
        let shader_frosted_source = format!(
            "{}\n{}",
            crate::shaders::GEOMETRY_FUNCTIONS,
            include_str!("video_shader_frosted.wgsl")
        );
        let shader_frosted = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("camera frosted composite shader"),
            source: wgpu::ShaderSource::Wgsl(shader_frosted_source.into()),
        });
        let pipeline_frosted_composite =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("camera frosted composite pipeline"),
                layout: Some(&pipeline_layout_rgb),
                vertex: wgpu::VertexState {
                    module: &shader_frosted,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_frosted,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        // Alpha blending (not REPLACE) so the frosted backdrop's
                        // antialiased corners composite over the sharp preview
                        // instead of overwriting it. The transition blur emits
                        // alpha = 1, where blending is identical to REPLACE, so it
                        // is unaffected.
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            });

        // ===== Filter Pre-blur Pipeline (lightweight Gaussian for multi-pass filters) =====
        // Reuses the same bind group layout as the main RGBA pipeline (texture + sampler + viewport)
        let shader_preblur = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("camera preblur shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("video_shader_preblur.wgsl").into()),
        });

        let pipeline_preblur = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("camera preblur pipeline"),
            layout: Some(&pipeline_layout_rgba),
            vertex: wgpu::VertexState {
                module: &shader_preblur,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_preblur,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        // Shared sampler for all pipelines
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("camera video sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // ===== YUV→RGBA Conversion Compute Pipeline =====
        let yuv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yuv_convert_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/yuv_convert.wgsl").into()),
        });

        let yuv_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("yuv_convert_bind_group_layout"),
                entries: &[
                    // tex_y: Y plane or packed YUYV
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // tex_uv: UV plane (NV12) or U plane (I420)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // tex_v: V plane (I420 only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // output: RGBA storage texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // params: uniform buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let yuv_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("yuv_convert_pipeline_layout"),
            bind_group_layouts: &[&yuv_bind_group_layout],
            immediate_size: 0,
        });

        let yuv_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("yuv_convert_compute_pipeline"),
                layout: Some(&yuv_pipeline_layout),
                module: &yuv_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let yuv_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yuv_convert_uniform_buffer"),
            size: std::mem::size_of::<YuvConvertParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline_rgba,
            pipeline_rgb_blur,
            pipeline_kawase_down,
            pipeline_kawase_up,
            pipeline_frosted_composite,
            pipeline_preblur,
            bind_group_layout_rgba,
            bind_group_layout_rgb,
            sampler,
            textures: std::collections::HashMap::new(),
            bindings: std::collections::HashMap::new(),
            blur_targets: std::sync::RwLock::new(std::collections::HashMap::new()),
            filter_preblur_intermediate: std::sync::RwLock::new(None),
            frames_skipped: std::sync::atomic::AtomicU32::new(0),
            yuv_compute_pipeline: Some(yuv_compute_pipeline),
            yuv_bind_group_layout: Some(yuv_bind_group_layout),
            yuv_uniform_buffer: Some(yuv_uniform_buffer),
            yuv_textures: std::collections::HashMap::new(),
            output_format: format,
        }
    }

    /// Upload frame data directly to GPU textures (texture only, bindings created separately)
    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, frame: VideoFrame) {
        use std::time::Instant;

        if frame.width == 0 || frame.height == 0 {
            return;
        }

        static FIRST_PREVIEW_UPLOAD: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        crate::startup::milestone_once("first_preview_gpu_upload", &FIRST_PREVIEW_UPLOAD);

        // The entry this frame's pixels live in — which is not `frame.id`, since
        // the frosted backdrop draws the preview's own frame (see
        // `source_texture_id`).
        let tex_id = source_texture_id(frame.id);

        // Get data pointer for deduplication (all filter picker widgets share the same Arc)
        //
        // Sound only because `AppModel::current_frame` keeps frame N's Arc alive
        // while frame N+1 is allocated, so the two can never share an address. If
        // a caller ever drops a frame before minting its successor, the allocator
        // may hand back the same block and this dedup will freeze the preview on
        // the stale texture.
        let frame_data_ptr = frame.data.as_ptr() as usize;

        // Check if texture exists and needs resizing
        let needs_creation = match self.textures.get(&tex_id) {
            Some(tex) => tex.width != frame.width || tex.height != frame.height,
            None => true,
        };

        // Check if this exact frame was already uploaded (same Arc pointer)
        // This prevents 15 redundant uploads when filter picker widgets share the same frame
        //
        // BEFORE the skip below, deliberately: an already-uploaded frame costs
        // nothing to "skip", so letting it reach the throttle only burned the
        // throttle's state on a no-op — the second consumer of a shared frame
        // consumed the skip its own upload never needed and reset the timer, so
        // a genuinely slow source was never actually throttled.
        if !needs_creation
            && let Some(tex) = self.textures.get(&tex_id)
            && tex.last_frame_ptr == frame_data_ptr
        {
            // Same frame data already uploaded, skip
            return;
        }

        // Skip frame if GPU is behind (last upload took > 32ms = 2 frame periods at 60fps)
        // This prevents the GPU command queue from backing up and causing UI hangs
        if let Some(tex) = self.textures.get_mut(&tex_id)
            && tex.last_upload_duration.as_millis() > 32
        {
            let last_upload_ms = tex.last_upload_duration.as_millis();
            // Reset timing to allow next frame through
            tex.last_upload_duration = std::time::Duration::ZERO;
            // Skipping a frame means skipping it for every consumer of this
            // source, not just the one that happened to ask first: record it as
            // handled so the dedup above turns the rest of this frame's asks into
            // no-ops. Otherwise the preview's skip merely handed the upload to the
            // frosted backdrop, and the frame was never actually skipped.
            tex.last_frame_ptr = frame_data_ptr;
            let skipped = self
                .frames_skipped
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            if skipped % 10 == 1 {
                tracing::warn!(
                    skipped_count = skipped,
                    last_upload_ms,
                    "Skipping frame - GPU behind, preventing UI hang"
                );
            }
            return;
        }

        let upload_start = Instant::now();

        // Create or resize texture if needed (invalidates all bindings for the
        // video_ids that share it)
        if needs_creation {
            let create_start = Instant::now();
            let new_tex = self.create_texture(device, frame.width, frame.height);
            self.textures.insert(tex_id, new_tex);
            // Remove all bindings over the old texture since it changed
            self.bindings
                .retain(|(vid, _), _| source_texture_id(*vid) != tex_id);
            // Invalidate cached YUV bind group (references old output view)
            if let Some(yuv) = self.yuv_textures.get_mut(&tex_id) {
                yuv.convert_bind_group = None;
            }
            let create_time = create_start.elapsed();
            if create_time.as_millis() > 5 {
                tracing::warn!(
                    create_ms = create_time.as_millis(),
                    width = frame.width,
                    height = frame.height,
                    "Texture creation took significant time - may cause stutter"
                );
            }
        }

        // Handle non-RGBA (YUV, ABGR, BGRA, etc.) or direct RGBA upload
        let gpu_copy_start = Instant::now();

        if frame.needs_gpu_conversion() {
            // GPU conversion path: Update last frame pointer, then run compute shader
            {
                let tex = self
                    .textures
                    .get_mut(&tex_id)
                    .expect("Texture should exist");
                tex.last_frame_ptr = frame_data_ptr;
            }
            // Now self.textures borrow is released, we can call upload_yuv_and_convert
            self.upload_yuv_and_convert(device, queue, tex_id, &frame);
        } else {
            // Direct RGBA texture upload (CPU to GPU copy)
            let tex = self
                .textures
                .get_mut(&tex_id)
                .expect("Texture should exist");
            tex.last_frame_ptr = frame_data_ptr;

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                frame.rgba_data(),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(frame.stride),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: frame.width,
                    height: frame.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        let gpu_copy_time = gpu_copy_start.elapsed();

        // Store GPU upload metrics for insights
        GPU_UPLOAD_TIME_US.store(gpu_copy_time.as_micros() as u64, Ordering::Relaxed);
        GPU_FRAME_SIZE.store(frame.data_slice().len() as u64, Ordering::Relaxed);

        // Track upload duration for frame skipping decisions
        let upload_duration = upload_start.elapsed();
        if let Some(tex) = self.textures.get_mut(&tex_id) {
            tex.last_upload_duration = upload_duration;
        }

        // Log GPU upload performance periodically (every ~30 frames based on frame.id)
        if frame.id.is_multiple_of(30) {
            let size_bytes = frame.data_slice().len();
            tracing::debug!(
                gpu_upload_us = gpu_copy_time.as_micros() as u64,
                total_prepare_us = upload_duration.as_micros() as u64,
                width = frame.width,
                height = frame.height,
                size_bytes,
                format = ?frame.format,
                "GPU texture upload"
            );
        }

        // Reset skip counter on successful upload
        let skipped = self
            .frames_skipped
            .load(std::sync::atomic::Ordering::Relaxed);
        if skipped > 0 {
            tracing::info!(
                frames_recovered = skipped,
                "GPU caught up, resuming normal frame rate"
            );
            self.frames_skipped
                .store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Create a texture for a video source (shared across filter variations)
    /// Includes STORAGE_BINDING usage for YUV→RGBA compute shader output
    fn create_texture(&self, device: &wgpu::Device, width: u32, height: u32) -> VideoTexture {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("camera RGBA texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            // Include STORAGE_BINDING for YUV→RGBA compute shader output
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        VideoTexture {
            texture,
            view,
            width,
            height,
            last_frame_ptr: 0, // Will be set on first upload
            last_upload_duration: std::time::Duration::ZERO,
        }
    }

    /// Upload YUV frame data and convert to RGBA using GPU compute shader
    ///
    /// This method:
    /// 1. Uploads YUV plane data to GPU textures
    /// 2. Runs compute shader to convert YUV→RGBA
    /// 3. Outputs directly to the RGBA texture used for rendering
    ///
    /// All processing stays on GPU - no CPU round-trip between YUV conversion and rendering.
    fn upload_yuv_and_convert(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tex_id: u64,
        frame: &VideoFrame,
    ) {
        use std::time::Instant;
        let convert_start = Instant::now();

        // Ensure YUV textures exist (UV dimensions from yuv_planes if available)
        let (uv_w, uv_h) = frame
            .yuv_planes
            .as_ref()
            .map(|p| (p.uv_width, p.uv_height))
            .unwrap_or_else(|| default_uv_size(frame.format, frame.width, frame.height));
        self.ensure_yuv_textures(
            device,
            tex_id,
            frame.width,
            frame.height,
            (uv_w, uv_h),
            frame.format,
        );

        // Get output texture view (already cached in VideoTexture)
        let output_view = match self.textures.get(&tex_id) {
            Some(tex) => &tex.view,
            None => {
                tracing::error!("Output texture not found for YUV conversion");
                return;
            }
        };

        let yuv_textures = match self.yuv_textures.get_mut(&tex_id) {
            Some(t) => t,
            None => {
                tracing::error!("YUV textures not found after ensure_yuv_textures");
                return;
            }
        };

        // Get the full buffer data (zero-copy from GStreamer)
        let buffer_data = frame.data_slice();

        // Upload planes using offsets (zero-copy: we slice from the mapped buffer)
        match frame.format {
            // Packed 4:2:2 formats: YUYV, UYVY, YVYU, VYUY
            // All packed as RGBA8 where each texel encodes 2 pixels
            PixelFormat::YUYV | PixelFormat::UYVY | PixelFormat::YVYU | PixelFormat::VYUY => {
                let packed_width = frame.width / 2;
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &yuv_textures.tex_y,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    buffer_data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(frame.stride),
                        rows_per_image: None,
                    },
                    wgpu::Extent3d {
                        width: packed_width,
                        height: frame.height,
                        depth_or_array_layers: 1,
                    },
                );
            }
            // Semi-planar 4:2:0 formats: NV12, NV21
            PixelFormat::NV12 | PixelFormat::NV21 => {
                // NV12: Use offsets to slice Y and UV planes from buffer
                if let Some(ref yuv_planes) = frame.yuv_planes {
                    let uv_width = frame.width / 2;
                    let uv_height = frame.height / 2;

                    // Y plane: full resolution, R8 format
                    let y_end = yuv_planes.y_offset + yuv_planes.y_size;
                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &yuv_textures.tex_y,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        &buffer_data[yuv_planes.y_offset..y_end],
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(frame.stride),
                            rows_per_image: None,
                        },
                        wgpu::Extent3d {
                            width: frame.width,
                            height: frame.height,
                            depth_or_array_layers: 1,
                        },
                    );

                    // UV plane: interleaved UV as RG8
                    let uv_end = yuv_planes.uv_offset + yuv_planes.uv_size;
                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &yuv_textures.tex_uv,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        &buffer_data[yuv_planes.uv_offset..uv_end],
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(yuv_planes.uv_stride),
                            rows_per_image: None,
                        },
                        wgpu::Extent3d {
                            width: uv_width,
                            height: uv_height,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
            PixelFormat::I420 => {
                // Planar YUV: Use offsets to slice Y, U, V planes from buffer
                // UV dimensions come from yuv_planes (supports 4:2:0, 4:2:2, 4:4:4)
                if let Some(ref yuv_planes) = frame.yuv_planes {
                    let uv_width = yuv_planes.uv_width;
                    let uv_height = yuv_planes.uv_height;

                    // Y plane: full resolution, R8 format
                    let y_end = yuv_planes.y_offset + yuv_planes.y_size;
                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &yuv_textures.tex_y,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        &buffer_data[yuv_planes.y_offset..y_end],
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(frame.stride),
                            rows_per_image: None,
                        },
                        wgpu::Extent3d {
                            width: frame.width,
                            height: frame.height,
                            depth_or_array_layers: 1,
                        },
                    );

                    // U plane: R8 format
                    let u_end = yuv_planes.uv_offset + yuv_planes.uv_size;
                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &yuv_textures.tex_uv,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        &buffer_data[yuv_planes.uv_offset..u_end],
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(yuv_planes.uv_stride),
                            rows_per_image: None,
                        },
                        wgpu::Extent3d {
                            width: uv_width,
                            height: uv_height,
                            depth_or_array_layers: 1,
                        },
                    );

                    // V plane: R8 format
                    if yuv_planes.v_size > 0 {
                        let v_end = yuv_planes.v_offset + yuv_planes.v_size;
                        queue.write_texture(
                            wgpu::TexelCopyTextureInfo {
                                texture: &yuv_textures.tex_v,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            &buffer_data[yuv_planes.v_offset..v_end],
                            wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(yuv_planes.v_stride),
                                rows_per_image: None,
                            },
                            wgpu::Extent3d {
                                width: uv_width,
                                height: uv_height,
                                depth_or_array_layers: 1,
                            },
                        );
                    }
                }
            }
            // Grayscale: single channel R8 format
            PixelFormat::Gray8 => {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &yuv_textures.tex_y,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    buffer_data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(frame.stride),
                        rows_per_image: None,
                    },
                    wgpu::Extent3d {
                        width: frame.width,
                        height: frame.height,
                        depth_or_array_layers: 1,
                    },
                );
            }
            // RGB24: Should have been converted to RGBA by GStreamer pipeline
            // If it arrives here, treat similarly to RGBA but with 3 bytes per pixel
            PixelFormat::RGB24 => {
                tracing::warn!(
                    "RGB24 format received - should have been converted to RGBA by pipeline"
                );
                return;
            }
            // ABGR/BGRA: Upload as RGBA8, shader will swizzle channels
            PixelFormat::ABGR | PixelFormat::BGRA => {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &yuv_textures.tex_y,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    buffer_data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(frame.stride),
                        rows_per_image: None,
                    },
                    wgpu::Extent3d {
                        width: frame.width,
                        height: frame.height,
                        depth_or_array_layers: 1,
                    },
                );
            }
            PixelFormat::RGBA => {
                // Should not reach here - RGBA is handled by direct upload path
                tracing::warn!("upload_yuv_and_convert called for RGBA frame");
                return;
            }
            // Bayer formats: Raw sensor data that requires debayering
            // This YUV convert path is not suitable - use dedicated debayer pipeline
            PixelFormat::BayerRGGB
            | PixelFormat::BayerBGGR
            | PixelFormat::BayerGRBG
            | PixelFormat::BayerGBRG => {
                tracing::warn!(
                    "Bayer format received in YUV pipeline - requires debayering, not supported here"
                );
                return;
            }
        }

        // Update uniform buffer with conversion parameters
        // Use the PixelFormat method to get format code
        let format_code = frame.format.gpu_format_code();

        let params = YuvConvertParams {
            width: frame.width,
            height: frame.height,
            format: format_code,
            y_stride: frame.stride,
            uv_stride: frame.yuv_planes.as_ref().map(|p| p.uv_stride).unwrap_or(0),
            v_stride: frame.yuv_planes.as_ref().map(|p| p.v_stride).unwrap_or(0),
            _pad: [0, 0],
        };

        if let Some(ref uniform_buffer) = self.yuv_uniform_buffer {
            queue.write_buffer(uniform_buffer, 0, bytemuck::cast_slice(&[params]));
        }

        // Create bind group lazily (reused across frames — only recreated when textures change)
        if yuv_textures.convert_bind_group.is_none() {
            let bind_group_layout = match &self.yuv_bind_group_layout {
                Some(layout) => layout,
                None => {
                    tracing::error!("YUV bind group layout not initialized");
                    return;
                }
            };

            yuv_textures.convert_bind_group = Some(
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("yuv_convert_bind_group"),
                    layout: bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&yuv_textures.tex_y_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&yuv_textures.tex_uv_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&yuv_textures.tex_v_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(output_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self
                                .yuv_uniform_buffer
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                    ],
                }),
            );
        }

        let bind_group = yuv_textures.convert_bind_group.as_ref().unwrap();

        // Dispatch compute shader
        let compute_pipeline = match &self.yuv_compute_pipeline {
            Some(pipeline) => pipeline,
            None => {
                tracing::error!("YUV compute pipeline not initialized");
                return;
            }
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("yuv_convert_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("yuv_convert_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(compute_pipeline);
            compute_pass.set_bind_group(0, Some(bind_group), &[]);

            // Dispatch: workgroup size is 16x16, so divide and round up
            let workgroup_x = frame.width.div_ceil(16);
            let workgroup_y = frame.height.div_ceil(16);
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        let convert_time = convert_start.elapsed();
        if frame.id.is_multiple_of(60) {
            tracing::debug!(
                format = ?frame.format,
                width = frame.width,
                height = frame.height,
                convert_us = convert_time.as_micros(),
                "YUV→RGBA GPU conversion"
            );
        }
    }

    /// Create or update YUV textures for a video source
    fn ensure_yuv_textures(
        &mut self,
        device: &wgpu::Device,
        video_id: u64,
        width: u32,
        height: u32,
        (uv_width, uv_height): (u32, u32),
        format: PixelFormat,
    ) {
        // Check if textures exist and match dimensions/format
        if let Some(yuv) = self.yuv_textures.get(&video_id)
            && yuv.width == width
            && yuv.height == height
            && yuv.uv_width == uv_width
            && yuv.uv_height == uv_height
            && yuv.format == format
        {
            return;
        }

        let (y_width, y_height) = (width, height);

        // Y plane texture format
        let y_format = match format {
            // Packed 4:2:2 formats: store as RGBA8 (4 bytes = 2 pixels)
            PixelFormat::YUYV | PixelFormat::UYVY | PixelFormat::YVYU | PixelFormat::VYUY => {
                wgpu::TextureFormat::Rgba8Unorm
            }
            // RGBA, RGB24, ABGR, BGRA: full RGBA texture
            PixelFormat::RGBA | PixelFormat::RGB24 | PixelFormat::ABGR | PixelFormat::BGRA => {
                wgpu::TextureFormat::Rgba8Unorm
            }
            // Y plane or grayscale: single channel
            _ => wgpu::TextureFormat::R8Unorm,
        };

        // UV plane texture format
        let uv_format = match format {
            // NV12/NV21: interleaved UV/VU as Rg8
            PixelFormat::NV12 | PixelFormat::NV21 => wgpu::TextureFormat::Rg8Unorm,
            // I420 and others: R8 for U/V planes
            _ => wgpu::TextureFormat::R8Unorm,
        };

        // Calculate Y texture width (packed formats store 2 pixels per texel)
        let y_tex_width = match format {
            PixelFormat::YUYV | PixelFormat::UYVY | PixelFormat::YVYU | PixelFormat::VYUY => {
                y_width / 2
            }
            _ => y_width,
        };

        // Create Y texture
        let tex_y = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("yuv_tex_y"),
            size: wgpu::Extent3d {
                width: y_tex_width,
                height: y_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: y_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let tex_y_view = tex_y.create_view(&wgpu::TextureViewDescriptor::default());

        // Create UV texture
        let tex_uv = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("yuv_tex_uv"),
            size: wgpu::Extent3d {
                width: uv_width.max(1),
                height: uv_height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: uv_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let tex_uv_view = tex_uv.create_view(&wgpu::TextureViewDescriptor::default());

        // Create V texture (I420 only, but always create for bind group consistency)
        let tex_v = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("yuv_tex_v"),
            size: wgpu::Extent3d {
                width: uv_width.max(1),
                height: uv_height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let tex_v_view = tex_v.create_view(&wgpu::TextureViewDescriptor::default());

        self.yuv_textures.insert(
            video_id,
            YuvTextures {
                tex_y,
                tex_y_view,
                tex_uv,
                tex_uv_view,
                tex_v,
                tex_v_view,
                width,
                height,
                uv_width,
                uv_height,
                format,
                convert_bind_group: None, // Created lazily on first use
            },
        );

        tracing::debug!(
            video_id,
            width,
            height,
            ?format,
            "Created YUV textures for GPU conversion"
        );
    }

    /// Get or create a filter-specific binding for a video
    /// Creates a unique binding per (video_id, filter_mode) combination
    /// This allows sharing the source texture while having different filter uniforms
    fn get_or_create_binding(
        &mut self,
        device: &wgpu::Device,
        video_id: u64,
        filter_mode: u32,
    ) -> Option<&FilterBinding> {
        let key = (video_id, filter_mode);

        // Check if binding already exists
        if self.bindings.contains_key(&key) {
            return self.bindings.get(&key);
        }

        // Need to create new binding - get the texture first. The binding is
        // per-(video_id, filter_mode) but the texture under it may be another
        // video_id's (see `source_texture_id`).
        let tex = self.textures.get(&source_texture_id(video_id))?;

        // Create viewport buffer for this filter
        let viewport_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera filter viewport buffer"),
            size: std::mem::size_of::<ViewportUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera filter bind group"),
            layout: &self.bind_group_layout_rgba,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: viewport_buffer.as_entire_binding(),
                },
            ],
        });

        self.bindings.insert(
            key,
            FilterBinding {
                bind_group,
                viewport_buffer,
            },
        );

        self.bindings.get(&key)
    }

    /// Mark `video_id`'s blur as stale, so the next `render()` re-runs its kernel
    /// passes. Touches ONLY that `video_id`'s flag — the other consumer's cached
    /// blur is still valid and must survive (see [`BlurTargets`]).
    fn invalidate_blur_cache(&self, video_id: u64) {
        if let Some(targets) = self.blur_targets.read().unwrap().get(&video_id) {
            targets
                .cached
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Create or update `video_id`'s Kawase ping-pong textures and their
    /// per-step bindings, sized to `width` x `height` PHYSICAL px — the on-screen
    /// preview rect (see [`BlurTargets`] for why screen resolution).
    ///
    /// Per `video_id`: each consumer parameterizes the chain differently and the
    /// steps carry those uniforms, so they must not be shared (see
    /// [`BlurTargets`]). Their sizes can differ too — the transition blur freezes
    /// a frame that may outlive a window resize the live frosted preview has
    /// already moved past.
    ///
    /// Called on EVERY `prepare()` for the blur ids, not only when a frame
    /// arrives: the size now follows the WINDOW, and the transition blur's frozen
    /// frame would otherwise keep rendering into a stale-sized target across a
    /// resize. It early-outs unless the size actually changed.
    fn ensure_blur_targets(
        &self,
        video_id: u64,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) {
        let width = width.max(1);
        let height = height.max(1);

        let needs_recreation = {
            let targets = self.blur_targets.read().unwrap();
            match targets.get(&video_id) {
                Some(t) => t.width != width || t.height != height,
                None => true,
            }
        };
        if !needs_recreation {
            return;
        }

        let make_texture = |label: &str| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let view_a = make_texture("camera blur kawase A");
        let view_b = make_texture("camera blur kawase B");

        let make_binding = |label: &str, view: &wgpu::TextureView| {
            let viewport_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<ViewportUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.bind_group_layout_rgb,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: viewport_buffer.as_entire_binding(),
                    },
                ],
            });
            FilterBinding {
                bind_group,
                viewport_buffer,
            }
        };

        // One step per ping-pong hop, at the deepest level any frost setting can
        // ask for. Allocating all of them up front (rather than per level) keeps
        // step `k`'s source texture a function of `k`'s parity alone, which is
        // what lets the composite always read `view_a` — see `KawaseStep`.
        let steps = (0..2 * MAX_KAWASE_PASSES)
            .map(|k| {
                let src = if k % 2 == 0 { &view_a } else { &view_b };
                let FilterBinding {
                    bind_group,
                    viewport_buffer,
                } = make_binding("camera blur kawase step", src);
                KawaseStep {
                    bind_group,
                    viewport_buffer,
                }
            })
            .collect();

        // One write, everything at once: a reader must never see a resized
        // `view_a` paired with the old steps. `cached` starts cold — fresh
        // textures hold nothing.
        self.blur_targets.write().unwrap().insert(
            video_id,
            BlurTargets {
                composite_binding: make_binding("camera blur composite", &view_a),
                view_a,
                view_b,
                width,
                height,
                steps,
                cached: std::sync::atomic::AtomicBool::new(false),
            },
        );
    }

    /// Create or update the filter pre-blur intermediate texture.
    ///
    /// Unlike the blur chain's targets (which follow the screen), this runs at
    /// the FRAME's full resolution to preserve detail for spatial filters like
    /// Pencil.
    fn ensure_filter_preblur_intermediate(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) {
        let needs_recreation = {
            let intermediate = self.filter_preblur_intermediate.read().unwrap();
            match intermediate.as_ref() {
                Some(i) => i.width != width || i.height != height,
                None => true,
            }
        };

        if needs_recreation {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("camera filter preblur intermediate"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

            *self.filter_preblur_intermediate.write().unwrap() = Some(PreblurIntermediate {
                view,
                width,
                height,
            });

            // No binding to invalidate: each primitive rebuilds its own
            // `preblur_binding` from this view every `prepare()`, which is what
            // keeps the bind group's borrow of it from outliving a resize.
        }
    }

    /// Render the video primitive.
    ///
    /// # Arguments
    /// * `video_id` - Unique identifier for the video source
    /// * `filter_mode` - Filter to apply (0 = none, 1+ = various filters)
    /// * `encoder` - GPU command encoder
    /// * `target` - Render target texture view
    /// * `clip_bounds` - Clipped bounds for scissor rect (visible portion after scroll clipping)
    /// * `widget_bounds` - Full widget bounds for viewport (x, y, width, height)
    /// * `needs_preblur` - Whether this filter needs a pre-blur pass
    /// * `blur_params` - The dual-Kawase (passes, offset) for the blur video_ids.
    ///   It has to arrive here rather than be read off the pipeline because it is
    ///   a property of the PRIMITIVE, and the two blur consumers differ (see
    ///   [`BlurTargets`]).
    /// * `frosted_final_binding` - The frosted backdrop's own composite binding,
    ///   carrying its panel rect + corner radius. `None` for every other video_id,
    ///   which falls back to that video_id's own shared composite binding.
    /// * `preblur_binding` - This primitive's own binding for a pre-blur filter's
    ///   second pass, carrying its panel rect + corner radius. Per-primitive for
    ///   the same reason as `frosted_final_binding`; `None` for every filter
    ///   without a pre-blur, and when it is `None` for one that has one the pass
    ///   falls back to sampling the source directly rather than drawing with
    ///   another consumer's rect.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        video_id: u64,
        filter_mode: u32,
        needs_preblur: bool,
        blur_params: CompositorBlurParams,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
        widget_bounds: (f32, f32, f32, f32),
        frosted_final_binding: Option<&FilterBinding>,
        preblur_binding: Option<&FilterBinding>,
    ) {
        // Look up binding for this (video_id, filter_mode) combination
        let binding_key = (video_id, filter_mode);
        if let Some(binding) = self.bindings.get(&binding_key) {
            // Skip rendering if clip bounds are empty
            if clip_bounds.width == 0 || clip_bounds.height == 0 {
                return;
            }

            // Blurred transition frames + the live frosted backdrop run the
            // dual-Kawase chain: ONE transform pass into a screen-resolution
            // target, `2 * passes` Kawase kernels ping-ponging over progressively
            // halved sub-rects of it, and a one-tap composite per panel/bar.
            //
            // The frosted backdrop stays live because it is fed a fresh frame
            // every view build, which invalidates its `BlurTargets::cached` in
            // prepare(); within one frame everything but the composite runs once
            // per video_id and every bar/panel/strip reuses the result via a
            // scissored blit, positioned at full preview geometry.
            //
            // What that costs, in screen areas S per frame per video_id, at
            // Medium frost (3 passes): pass 0 writes 1.00 S at one tap; the
            // downsamples write 1/4 + 1/16 + 1/64 = 0.33 S at five taps; the
            // upsamples write 1/16 + 1/4 + 1 = 1.31 S at eight taps. 2.64 S of
            // fragments (~6.6 M at 1080x2340) and ~13 S of texture taps (~33 M).
            // Raising frost to 13 adds a fourth pass and moves that to 2.66 S:
            // the cost is essentially independent of the setting, because the
            // final upsample back to level 0 dominates the whole chain.
            //
            // Two thirds of it is therefore full-resolution work, and neither
            // half is ours to remove. The level-0 upsample IS upstream's dual-
            // Kawase — `passes` down then `passes` up, back to full — and pass 0
            // is what buys the parity: it lands the frame in screen space at
            // screen resolution so `BLUR_PARAMS`'s screen-px offsets feed the
            // kernels verbatim. Folding pass 0's transform into the first
            // downsample would delete a full-res write, but the five taps would
            // then land on sensor texels through the transform instead of on
            // bilinear-filtered screen texels — different sampled data, so the
            // chain would no longer be upstream's and `kawase_kernels_match_
            // cosmic_comp` would be measuring a fiction. The cost is the price of
            // exact parity, and it is paid knowingly.
            if video_id == VIDEO_ID_BLUR || video_id == VIDEO_ID_FROSTED {
                // Strictly THIS video_id's targets: the transition blur and the
                // frosted chrome parameterize the chain differently and each owns
                // its own textures and cache flag (see `BlurTargets`).
                let blur_targets_guard = self.blur_targets.read().unwrap();
                let Some(targets) = blur_targets_guard.get(&video_id) else {
                    // Fallback to single-pass if the targets aren't ready
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("camera video render pass fallback"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: target,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });

                    // Use full widget bounds for viewport (prevents distortion in scrollables)
                    render_pass.set_viewport(
                        widget_bounds.0,
                        widget_bounds.1,
                        widget_bounds.2,
                        widget_bounds.3,
                        0.0,
                        1.0,
                    );

                    // Use clip bounds for scissor (clips to visible portion)
                    render_pass.set_scissor_rect(
                        clip_bounds.x,
                        clip_bounds.y,
                        clip_bounds.width,
                        clip_bounds.height,
                    );

                    render_pass.set_pipeline(&self.pipeline_rgb_blur);
                    render_pass.set_bind_group(0, Some(&binding.bind_group), &[]);
                    render_pass.draw(0..3, 0..1);
                    return;
                };

                // Run the expensive part only when THIS video_id's cache is cold:
                // once per new frame. The transition blur freezes a frame; the
                // frosted backdrop is fed a fresh frame each view build (which
                // resets its own flag in prepare()), so it re-blurs each frame
                // while multiple bars/panels/strips in the same frame reuse it.
                if !targets.cached.load(std::sync::atomic::Ordering::Relaxed) {
                    // Every target is cleared to TRANSPARENT, not to black. The
                    // Kawase kernels normalize by `sum.a` and so read a = 0 as
                    // "outside the live sub-rect" — clearing to opaque black would
                    // make those taps count and bleed a dark rim inward at every
                    // level. This is upstream's `frame.clear(0,0,0,0)`.
                    const CLEAR: wgpu::Color = wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    };

                    // PASS 0: transform the source frame into screen space, at
                    // screen resolution, in `view_a`. No kernel — see
                    // `video_shader_blur.wgsl`.
                    {
                        let mut render_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("camera blur pass 0 (transform)"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &targets.view_a,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(CLEAR),
                                        store: wgpu::StoreOp::Store,
                                    },
                                    depth_slice: None,
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                                multiview_mask: None,
                            });
                        render_pass.set_pipeline(&self.pipeline_rgb_blur);
                        render_pass.set_bind_group(0, Some(&binding.bind_group), &[]);
                        render_pass.draw(0..3, 0..1);
                    }

                    // The Kawase ping-pong, transcribed from upstream's
                    // `render_blur`: `passes` downsamples then `passes` upsamples,
                    // over sub-rects of two full-size textures rather than a mip
                    // chain. Step `k` reads `view_a` when `k` is even (which is why
                    // pass 0 wrote there) and `view_b` when odd; `2 * passes` is
                    // always even, so the result always lands back in `view_a`.
                    //
                    // The viewport is the DESTINATION sub-rect, which is what makes
                    // the shader's `tex_coords` span it; the source sub-rect
                    // reaches the shader as `uv_scale`. See the shader header.
                    let passes =
                        effective_kawase_passes(targets.width, targets.height, blur_params.passes)
                            .min(MAX_KAWASE_PASSES);
                    for k in 0..(2 * passes) {
                        let Some(step) = targets.steps.get(k as usize) else {
                            break;
                        };
                        // Down step i (k = i) shrinks to level i+1; up step j
                        // (k = passes + j) grows back to level passes-j-1.
                        let dst_level = if k < passes {
                            k + 1
                        } else {
                            2 * passes - k - 1
                        };
                        let dst_w = (targets.width >> dst_level).max(1);
                        let dst_h = (targets.height >> dst_level).max(1);
                        let dst_view = if k % 2 == 0 {
                            &targets.view_b
                        } else {
                            &targets.view_a
                        };

                        let mut render_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("camera blur kawase step"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: dst_view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(CLEAR),
                                        store: wgpu::StoreOp::Store,
                                    },
                                    depth_slice: None,
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                                multiview_mask: None,
                            });
                        render_pass.set_viewport(0.0, 0.0, dst_w as f32, dst_h as f32, 0.0, 1.0);
                        // The scissor as well as the viewport: `set_viewport` only
                        // sets the NDC transform, it does not clip, and the
                        // fullscreen triangle deliberately overshoots NDC 1.0.
                        // Without this the pass would also paint OUTSIDE its
                        // destination sub-rect, with extrapolated `tex_coords` and
                        // alpha = 1 — overwriting the transparent clear that the
                        // next pass's kernels rely on to know where the live region
                        // ends (see `video_shader_kawase.wgsl` on `sum.a`).
                        render_pass.set_scissor_rect(0, 0, dst_w, dst_h);
                        render_pass.set_pipeline(if k < passes {
                            &self.pipeline_kawase_down
                        } else {
                            &self.pipeline_kawase_up
                        });
                        render_pass.set_bind_group(0, Some(&step.bind_group), &[]);
                        render_pass.draw(0..3, 0..1);
                    }

                    targets
                        .cached
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                }

                // Final composite: blit `view_a` to the target — one sample per
                // fragment, plus dim, grain and the corner SDF. This is the only
                // part of the chain that runs per panel/bar at screen resolution,
                // which is exactly why it is the part that must stay cheap.
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("camera blur composite"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: target,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });

                    // The viewport is the preview rect the Kawase textures were
                    // sized from, so the composite's texture mapping is identity.
                    render_pass.set_viewport(
                        widget_bounds.0,
                        widget_bounds.1,
                        widget_bounds.2,
                        widget_bounds.3,
                        0.0,
                        1.0,
                    );

                    // Use clip bounds for scissor (clips to visible portion)
                    render_pass.set_scissor_rect(
                        clip_bounds.x,
                        clip_bounds.y,
                        clip_bounds.width,
                        clip_bounds.height,
                    );

                    render_pass.set_pipeline(&self.pipeline_frosted_composite);
                    // The frosted backdrop supplies its own binding so the shader
                    // can round THIS panel's corners; the transition blur has no
                    // per-panel data and uses the shared one.
                    let final_binding = frosted_final_binding
                        .map(|b| &b.bind_group)
                        .unwrap_or(&targets.composite_binding.bind_group);
                    render_pass.set_bind_group(0, Some(final_binding), &[]);
                    render_pass.draw(0..3, 0..1);
                }
            } else if needs_preblur {
                // Multi-pass rendering: pre-blur → filter
                let intermediate_opt = self.filter_preblur_intermediate.read().unwrap();
                if let Some(intermediate) = intermediate_opt.as_ref() {
                    // Pass 1: Render source through lightweight blur → intermediate
                    {
                        let mut render_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("camera filter preblur pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &intermediate.view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                        store: wgpu::StoreOp::Store,
                                    },
                                    depth_slice: None,
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                                multiview_mask: None,
                            });

                        render_pass.set_pipeline(&self.pipeline_preblur);
                        render_pass.set_bind_group(0, Some(&binding.bind_group), &[]);
                        render_pass.draw(0..3, 0..1);
                    }

                    // Pass 2: Render from pre-blurred intermediate with filter → screen
                    // Use the preblur bind group which samples from the intermediate texture.
                    // Fall back to single-pass from source if binding not ready yet.
                    let pass2_bind_group = preblur_binding
                        .map(|b| &b.bind_group)
                        .unwrap_or(&binding.bind_group);

                    {
                        let mut render_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("camera video render pass (from preblur)"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: target,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                    depth_slice: None,
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                                multiview_mask: None,
                            });

                        render_pass.set_viewport(
                            widget_bounds.0,
                            widget_bounds.1,
                            widget_bounds.2,
                            widget_bounds.3,
                            0.0,
                            1.0,
                        );

                        render_pass.set_scissor_rect(
                            clip_bounds.x,
                            clip_bounds.y,
                            clip_bounds.width,
                            clip_bounds.height,
                        );

                        render_pass.set_pipeline(&self.pipeline_rgba);
                        render_pass.set_bind_group(0, Some(pass2_bind_group), &[]);
                        render_pass.draw(0..3, 0..1);
                    }
                }
            } else {
                // Single-pass RGBA rendering for live preview
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("camera video render pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                // Use full widget bounds for viewport (prevents distortion in scrollables)
                render_pass.set_viewport(
                    widget_bounds.0,
                    widget_bounds.1,
                    widget_bounds.2,
                    widget_bounds.3,
                    0.0,
                    1.0,
                );

                // Use clip bounds for scissor (clips to visible portion)
                render_pass.set_scissor_rect(
                    clip_bounds.x,
                    clip_bounds.y,
                    clip_bounds.width,
                    clip_bounds.height,
                );

                render_pass.set_pipeline(&self.pipeline_rgba);
                render_pass.set_bind_group(0, Some(&binding.bind_group), &[]);
                render_pass.draw(0..3, 0..1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_gpu::{headless_device, skip_no_gpu};

    /// `ViewportUniform` is mirrored by hand in six WGSL files
    /// (`video_shader.wgsl`, `video_shader_blur.wgsl`, `video_shader_kawase.wgsl`,
    /// `video_shader_frosted.wgsl`, `video_shader_preblur.wgsl` and the gallery
    /// shader). Nothing but these offsets keeps the Rust struct
    /// and those declarations describing the same bytes, and a mismatch shows up
    /// as silently garbled rendering rather than a compile error — so pin them.
    ///
    /// `panel_rect`, `noise`, `content_rect`, and side insets are appended: only the
    /// composite shader declares them, and the other shaders stay valid against
    /// the same (larger) buffer precisely because every field before them keeps
    /// its offset.
    #[test]
    fn viewport_uniform_layout_is_stable() {
        use std::mem::{align_of, offset_of, size_of};

        assert_eq!(offset_of!(ViewportUniform, viewport_size), 0);
        assert_eq!(offset_of!(ViewportUniform, corner_radius), 16);
        assert_eq!(offset_of!(ViewportUniform, zoom_level), 56);
        // kawase_offset/dim_factor occupy the two slots that used to be pure
        // padding. letterbox_color is a vec4 and must sit on a 16-byte
        // boundary; these two are what push it there, so they must stay
        // exactly here and stay f32-sized.
        assert_eq!(offset_of!(ViewportUniform, kawase_offset), 72);
        assert_eq!(offset_of!(ViewportUniform, dim_factor), 76);
        assert_eq!(offset_of!(ViewportUniform, letterbox_color), 80);
        // panel_rect is also a vec4, landing exactly where letterbox_color ends.
        assert_eq!(offset_of!(ViewportUniform, panel_rect), 96);
        // `noise` is appended after every offset the other shaders rely on...
        assert_eq!(offset_of!(ViewportUniform, noise), 112);
        // ...and `content_rect` after that. It is a vec4 in WGSL, so it must land
        // on a 16-byte boundary: `_pad` is what carries it from 116 to 128, and
        // the shader gets there on its own because a vec4 aligns to 16. Declaring
        // a `vec3<f32>` pad in the WGSL to mirror `_pad` would align to 16 too
        // and silently push this to 144.
        assert_eq!(offset_of!(ViewportUniform, content_rect), 128);
        assert_eq!(offset_of!(ViewportUniform, bar_left_width), 144);
        assert_eq!(offset_of!(ViewportUniform, bar_right_width), 148);
        assert_eq!(size_of::<ViewportUniform>(), 160);
        assert_eq!(size_of::<ViewportUniform>() % 16, 0);
        assert_eq!(align_of::<ViewportUniform>(), 4);
    }

    /// The unit square is the default, so every pass that never heard of
    /// `content_rect` — the Kawase steps, the pre-blur, the sharp preview — keeps
    /// the composite's letterbox cut switched off.
    #[test]
    fn the_default_content_rect_cuts_nothing() {
        assert_eq!(ViewportUniform::default().content_rect, FULL_CONTENT_RECT);
        assert_eq!(FULL_CONTENT_RECT, [0.0, 0.0, 1.0, 1.0]);
    }

    /// Fill (`cover_blend` = 1) has no letterbox at all: Cover covers the window,
    /// so the image rect IS the preview rect and the composite must paint every
    /// pixel of every scrim bar with blurred preview.
    ///
    /// The `>=`/`<=` rather than exact equality is deliberate — Cover overflows on
    /// one axis by design, so the rect legitimately runs past the unit square, and
    /// what matters is that it never falls short and cuts live preview away.
    #[test]
    fn content_rect_covers_everything_in_fill() {
        for (tw, th) in [(1280u32, 960u32), (960, 1280), (1920, 1080)] {
            for rotation in 0..4u32 {
                let r = content_rect_uv(
                    [1080.0, 2340.0],
                    (tw, th),
                    rotation,
                    [0.0, 0.0],
                    [1.0, 1.0],
                    1.0,
                    47.0,
                    174.0,
                );
                assert!(
                    r[0] <= 0.0 && r[1] <= 0.0 && r[2] >= 1.0 && r[3] >= 1.0,
                    "{tw}x{th} rot {rotation} in Fill gave {r:?} — Cover fills the \
                     window, so a rect short of the unit square would cut live \
                     preview out of the frosted bars"
                );
            }
        }
    }

    /// Fit (`cover_blend` = 0) letterboxes into the area BETWEEN the UI bars — so
    /// the bar regions come out entirely letterbox, which is what lets the window
    /// background show through them.
    ///
    /// This is the geometric claim `frosted_backdrop`'s module docs make in prose
    /// ("the bar regions are by construction exactly where the image is not"), and
    /// the whole reason the cut is worth making: in Fit the scrim bars are pure
    /// letterbox, and painting them opaque is exactly the bug.
    #[test]
    fn content_rect_in_fit_stays_between_the_ui_bars() {
        const H: f32 = 2340.0;
        const TOP: f32 = 47.0;
        const BOTTOM: f32 = 174.0;
        let r = content_rect_uv(
            [1080.0, H],
            (1280, 960),
            0,
            [0.0, 0.0],
            [1.0, 1.0],
            0.0,
            TOP,
            BOTTOM,
        );
        assert!(
            r[1] >= TOP / H - 0.001,
            "the image must start below the top bar, got {r:?}"
        );
        assert!(
            r[3] <= 1.0 - BOTTOM / H + 0.001,
            "the image must end above the bottom bar, got {r:?}"
        );
        // And it is centred in the CONTENT area, not the window — the asymmetric
        // bars are the whole reason `content_center_y` exists in the shader.
        let centre = (r[1] + r[3]) / 2.0;
        let content_centre = (TOP + (H - TOP - BOTTOM) / 2.0) / H;
        assert!(
            (centre - content_centre).abs() < 0.001,
            "image centred at {centre:.4}, content area centres at \
             {content_centre:.4} — centring on the window instead would slide the \
             blurred slice against the sharp preview"
        );
    }

    #[test]
    fn content_rect_in_fit_uses_left_and_right_bar_insets() {
        let fit = |insets| {
            content_rect_uv_with_insets(
                [1000.0, 600.0],
                (800, 600),
                0,
                [0.0, 0.0],
                [1.0, 1.0],
                0.0,
                insets,
            )
        };
        let left = fit(crate::app::preview_geometry::BarInsets {
            left: 200.0,
            ..Default::default()
        });
        let right = fit(crate::app::preview_geometry::BarInsets {
            right: 200.0,
            ..Default::default()
        });

        for (actual, expected) in left.into_iter().zip([0.2, 0.0, 1.0, 1.0]) {
            assert!((actual - expected).abs() < 0.0001, "left fit was {left:?}");
        }
        for (actual, expected) in right.into_iter().zip([0.0, 0.0, 0.8, 1.0]) {
            assert!(
                (actual - expected).abs() < 0.0001,
                "right fit was {right:?}"
            );
        }
    }

    #[test]
    fn content_rect_in_fill_ignores_side_bar_insets() {
        for insets in [
            crate::app::preview_geometry::BarInsets {
                left: 200.0,
                ..Default::default()
            },
            crate::app::preview_geometry::BarInsets {
                right: 200.0,
                ..Default::default()
            },
        ] {
            let rect = content_rect_uv_with_insets(
                [1000.0, 600.0],
                (800, 600),
                0,
                [0.0, 0.0],
                [1.0, 1.0],
                1.0,
                insets,
            );
            assert!(
                rect[0] <= 0.0 && rect[1] <= 0.0 && rect[2] >= 1.0 && rect[3] >= 1.0,
                "Fill must still cover the full window, got {rect:?}"
            );
        }
    }

    #[test]
    fn every_transform_shader_mirrors_the_asymmetric_fit_pivot() {
        for (name, shader) in [
            ("preview", include_str!("video_shader.wgsl")),
            ("blur", include_str!("video_shader_blur.wgsl")),
            ("preblur", include_str!("video_shader_preblur.wgsl")),
        ] {
            assert!(
                shader
                    .contains("select(center_x, 1.0 - center_x, viewport.mirror_horizontal == 1u)"),
                "{name} must mirror the fit pivot with the sampled selfie image"
            );
        }
    }

    /// A 90/270 sensor rotation swaps the image's axes, so a landscape sensor
    /// letterboxes on the SIDES of a landscape window rather than above and below.
    ///
    /// Easy to lose: the shader swaps both `tex_size` and the crop range, and
    /// dropping either swap leaves a rect that looks plausible and is wrong by the
    /// sensor's aspect ratio.
    #[test]
    fn content_rect_follows_the_sensor_rotation() {
        let fit = |rotation| {
            content_rect_uv(
                [1920.0, 1080.0],
                (1280, 960),
                rotation,
                [0.0, 0.0],
                [1.0, 1.0],
                0.0,
                0.0,
                0.0,
            )
        };
        // Unrotated 4:3 into a 16:9 window: full height, pillarboxed.
        let up = fit(0);
        assert!(up[1] <= 0.001 && up[3] >= 0.999, "{up:?}");
        assert!(up[0] > 0.01 && up[2] < 0.99, "{up:?}");
        // Rotated 90 the image is 3:4 — narrower still, so more pillarbox.
        let side = fit(1);
        assert!(side[1] <= 0.001 && side[3] >= 0.999, "{side:?}");
        assert!(
            side[0] > up[0],
            "a 90 rotation makes a 4:3 sensor 3:4 on screen, which must pillarbox \
             MORE, got {side:?} against {up:?}"
        );
        assert_eq!(fit(1), fit(3), "90 and 270 differ only in which way up");
        assert_eq!(fit(0), fit(2), "180 keeps the extent, flips the content");
    }

    /// A crop narrows the image, so it letterboxes more — and the crop is in
    /// TEXTURE orientation, so its range is swapped by rotation alongside the
    /// texture's own dimensions.
    #[test]
    fn content_rect_accounts_for_the_aspect_crop() {
        let uncropped = content_rect_uv(
            [1080.0, 1080.0],
            (1280, 960),
            0,
            [0.0, 0.0],
            [1.0, 1.0],
            0.0,
            0.0,
            0.0,
        );
        // A 1:1 crop out of a 4:3 sensor: square, so it fills a square window.
        let cropped = content_rect_uv(
            [1080.0, 1080.0],
            (1280, 960),
            0,
            [0.125, 0.0],
            [0.875, 1.0],
            0.0,
            0.0,
            0.0,
        );
        assert!(
            uncropped[1] > 0.01,
            "4:3 in a square window must letterbox, got {uncropped:?}"
        );
        assert!(
            cropped[1] <= 0.001 && cropped[3] >= 0.999,
            "a 1:1 crop fills a square window edge to edge, got {cropped:?}"
        );
    }

    /// The crop only applies at Fit; Cover ignores it, blending it away to the
    /// full texture exactly as the shader's `effective_crop_*` does. A rect that
    /// kept the crop at Fill would cut the scrim bars short of the window edge.
    ///
    /// Note the mid-animation value is deliberately NOT asserted to interpolate
    /// between the ends. Two things blend at once — the crop region and the fit —
    /// so a 1:1 crop that exactly fills a square window at Fit passes through
    /// *wider-than-square* shapes on its way to Cover, and letterboxes slightly
    /// in the middle even though both endpoints are flush. That is the shader's
    /// own behaviour, faithfully reproduced; it is a transient of a ~250 ms
    /// animation and not something to "correct" here, because correcting it would
    /// put this function out of step with `video_shader_blur.wgsl`.
    #[test]
    fn content_rect_blends_the_crop_away_toward_fill() {
        let at = |blend| {
            content_rect_uv(
                [1080.0, 1080.0],
                (1280, 960),
                0,
                [0.125, 0.0],
                [0.875, 1.0],
                blend,
                0.0,
                0.0,
            )
        };
        // Fit: the 1:1 crop fills the square window exactly.
        let fit = at(0.0);
        assert!(
            fit[1] <= 0.001 && fit[3] >= 0.999,
            "a 1:1 crop must fill a square window at Fit, got {fit:?}"
        );
        // Fill: the crop is gone and Cover overflows, so nothing is cut.
        let fill = at(1.0);
        assert!(
            fill[0] <= 0.0 && fill[1] <= 0.0 && fill[2] >= 1.0 && fill[3] >= 1.0,
            "Cover must reach every edge with the crop blended away, got {fill:?}"
        );
        // Whatever happens in between stays bounded — a rect that ran away would
        // cut the backdrop to nothing mid-animation.
        for i in 1..10 {
            let r = at(i as f32 / 10.0);
            assert!(
                r[2] - r[0] > 0.5 && r[3] - r[1] > 0.5,
                "at blend {:.1} the image collapsed to {r:?}",
                i as f32 / 10.0
            );
        }
    }

    /// Degenerate geometry falls back to "no letterbox" rather than to an empty
    /// rect. Erring this way leaves the frosted glass painting; the other way
    /// makes it vanish, which is far worse than a letterbox left opaque for a
    /// frame.
    #[test]
    fn content_rect_falls_back_to_full_when_the_geometry_is_degenerate() {
        let cases = [
            // No window yet.
            ([0.0f32, 0.0], (1280u32, 960u32), 0.0f32, 0.0f32),
            // No frame yet.
            ([1080.0, 2340.0], (0, 0), 47.0, 174.0),
            // A window smaller than its own chrome: no content area to fit into.
            ([1080.0, 100.0], (1280, 960), 47.0, 174.0),
        ];
        for (viewport, tex, top, bottom) in cases {
            assert_eq!(
                content_rect_uv(viewport, tex, 0, [0.0, 0.0], [1.0, 1.0], 0.0, top, bottom),
                FULL_CONTENT_RECT,
                "{viewport:?} {tex:?} bars {top}/{bottom}"
            );
        }
        // A collapsed crop is degenerate too — a zero-extent image would cut the
        // whole backdrop away.
        assert_eq!(
            content_rect_uv(
                [1080.0, 2340.0],
                (1280, 960),
                0,
                [0.5, 0.5],
                [0.5, 0.5],
                0.0,
                0.0,
                0.0
            ),
            FULL_CONTENT_RECT
        );
    }

    /// The format the blur chain's ping-pong targets ACTUALLY have on device.
    ///
    /// `VideoPipeline::new` is handed iced's surface format, and
    /// `ensure_blur_targets` gives every Kawase target that same format (see
    /// `prepare()`, which passes `pipeline.output_format`). iced picks the
    /// surface format with `formats.find(wgpu::TextureFormat::is_srgb)` because
    /// `graphics::color::GAMMA_CORRECTION` is `true`, so on device that format is
    /// always sRGB — never the plain `Rgba8Unorm` these tests used to pass.
    ///
    /// The difference is not cosmetic and it is not nothing. On an sRGB target
    /// the hardware encodes on write and decodes on read, so each of the chain's
    /// `2 * passes` hops round-trips through the sRGB curve and requantises to 8
    /// bits on a NON-uniform grid — one that spends its codes on the shadows and
    /// starves the highlights, where the steps reach ~1 LSB. Since this branch
    /// exists to fix BANDING, a suite that only ever measured the uniform grid
    /// was blind to the one requantisation that ships.
    ///
    /// `Rgba8UnormSrgb` rather than the `Bgra8UnormSrgb` a Wayland surface
    /// typically reports: what matters here is the transfer function, and keeping
    /// RGBA channel order lets the readback indexing stay shared with the
    /// non-sRGB case.
    const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

    /// The camera source texture's format — genuinely `Rgba8Unorm` in production
    /// (see `upload`), so the frame's texels are read back exactly as uploaded.
    /// Only the blur TARGETS take the surface format.
    const SOURCE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    /// The formats the blur chain is exercised over: the linear one the suite has
    /// always used, and the sRGB one that actually ships.
    const BLUR_TARGET_FORMATS: [wgpu::TextureFormat; 2] = [SOURCE_FORMAT, SURFACE_FORMAT];

    /// Undo the hardware's sRGB encode, returning the linear value the shader
    /// wrote, on a 0..255 scale.
    ///
    /// Readback of an sRGB texture hands back ENCODED bytes, but every metric and
    /// threshold in this module is expressed in the shader's own (linear) space.
    /// Decoding here — rather than re-quantising to `u8` — keeps the sRGB and
    /// non-sRGB paths measurable against the identical threshold, so the only
    /// thing that moves between them is the extra quantisation each hop really
    /// took. That IS the quantity of interest.
    fn decode_srgb(b: u8) -> f32 {
        let c = f32::from(b) / 255.0;
        let linear = if c <= 0.040_45 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        };
        linear * 255.0
    }

    /// A readback byte in the shader's own space, whatever the target format.
    fn decode_channel(b: u8, format: wgpu::TextureFormat) -> f32 {
        if format.is_srgb() {
            decode_srgb(b)
        } else {
            f32::from(b)
        }
    }

    /// Render one frosted backdrop into a texture and read it back.
    ///
    /// The panel covers the whole 64x64 target with radius 32 (a circle), so the
    /// corners are the interesting part. The target is pre-cleared to opaque RED:
    /// wherever the backdrop's alpha is 0 the red survives, wherever it is 1 the
    /// blurred (white) frame wins, and any pixel in between is antialiasing.
    fn render_frosted_corner(corner_radius: f32) -> Option<Vec<[u8; 4]>> {
        const N: u32 = 64;
        let (_gpu_test, device, queue) = headless_device()?;
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut pipeline = VideoPipeline::new(&device, format);

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frosted corner test target"),
            size: wgpu::Extent3d {
                width: N,
                height: N,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        // An all-white frame: blurring white gives white, so the backdrop's
        // colour is uniform and any variation we find is the corner alpha.
        let mut primitive = VideoPrimitive::new(VIDEO_ID_FROSTED);
        primitive.corner_radius = corner_radius;
        primitive.update_frame(VideoFrame {
            id: VIDEO_ID_FROSTED,
            width: N,
            height: N,
            data: crate::backends::camera::types::FrameData::Copied(
                vec![255u8; (N * N * 4) as usize].into(),
            ),
            format: PixelFormat::RGBA,
            stride: N * 4,
            yuv_planes: None,
        });
        primitive.update_viewport(
            N as f32,
            N as f32,
            1.0,
            crate::app::preview_geometry::BarInsets::default(),
        );

        let bounds = Rectangle {
            x: 0.0,
            y: 0.0,
            width: N as f32,
            height: N as f32,
        };
        let viewport = Viewport::with_physical_size(cosmic::iced::Size::new(N, N), 1.0);
        primitive.prepare(&mut pipeline, &device, &queue, &bounds, &viewport);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // Pre-fill with red; the backdrop's final pass loads (not clears) it.
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("frosted corner test clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        primitive.render(
            &pipeline,
            &mut encoder,
            &view,
            &Rectangle {
                x: 0,
                y: 0,
                width: N,
                height: N,
            },
        );

        // 64 px * 4 bytes = 256, already the required row alignment.
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (N * N * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(N * 4),
                    rows_per_image: Some(N),
                },
            },
            wgpu::Extent3d {
                width: N,
                height: N,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let data = slice.get_mapped_range().to_vec();
        Some(
            data.as_chunks::<4>()
                .0
                .iter()
                .map(|c| [c[0], c[1], c[2], c[3]])
                .collect(),
        )
    }

    /// "Redness" of a pixel: 255 = untouched clear colour (backdrop alpha 0),
    /// 0 = fully covered by the blurred white backdrop (alpha 1).
    fn redness(px: [u8; 4]) -> i32 {
        px[0] as i32 - px[1] as i32
    }

    /// The frosted backdrop's rounded corners must be ANTIALIASED.
    ///
    /// This is the whole point of rounding in the shader: the old approach
    /// scissored the blur to a rounded-rect strip list, and a scissor is
    /// integer-pixel binary coverage — every pixel fully in or fully out, so a
    /// staircase was the only possible output. An SDF edge produces partial
    /// coverage, so we assert that pixels exist which are NEITHER clear colour
    /// NOR fully covered. That assertion fails on any scissor-based approach.
    #[test]
    fn frosted_corner_is_antialiased() {
        let Some(px) = render_frosted_corner(32.0) else {
            skip_no_gpu("frosted_corner_is_antialiased");
            return;
        };
        let at = |x: usize, y: usize| px[y * 64 + x];

        // Centre is fully covered by the backdrop; the extreme corner is outside
        // the circle entirely and keeps the red clear colour.
        assert!(
            redness(at(32, 32)) < 40,
            "centre should be covered by the backdrop, got {:?}",
            at(32, 32)
        );
        assert!(
            redness(at(0, 0)) > 200,
            "corner should be outside the rounded silhouette, got {:?}",
            at(0, 0)
        );

        // Walk the diagonal out of the corner: a scissor can only ever step from
        // "fully red" to "fully covered", so an intermediate pixel proves the SDF
        // is producing real partial coverage.
        let partial = (0..32)
            .map(|i| redness(at(i, i)))
            .filter(|r| (40..=200).contains(r))
            .count();
        assert!(
            partial > 0,
            "expected antialiased pixels along the corner diagonal, found none — \
             the edge is hard, so the SDF is not being applied"
        );
    }

    /// Render one filter-picker swatch — a small widget over a much larger frame
    /// — and read it back.
    ///
    /// The size mismatch is the point: it is what the filter picker really does
    /// (a 64 px swatch over a sensor-sized frame), and it is what separates the
    /// widget's rect from the pre-blur intermediate's. The target is pre-cleared
    /// to opaque RED, so a rounded-away corner stays red and a covered pixel does
    /// not; the frame is white, and every filter here maps flat white to a grey,
    /// so `redness` reads corner coverage and nothing else.
    fn render_swatch_corner(filter: FilterType, corner_radius: f32) -> Option<Vec<[u8; 4]>> {
        const N: u32 = 64;
        const FRAME_W: u32 = 640;
        const FRAME_H: u32 = 480;
        let (_gpu_test, device, queue) = headless_device()?;
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut pipeline = VideoPipeline::new(&device, format);

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("swatch corner test target"),
            size: wgpu::Extent3d {
                width: N,
                height: N,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        let mut primitive = VideoPrimitive::new(7);
        primitive.filter_type = filter;
        primitive.corner_radius = corner_radius;
        primitive.update_frame(VideoFrame {
            id: 7,
            width: FRAME_W,
            height: FRAME_H,
            data: crate::backends::camera::types::FrameData::Copied(
                vec![255u8; (FRAME_W * FRAME_H * 4) as usize].into(),
            ),
            format: PixelFormat::RGBA,
            stride: FRAME_W * 4,
            yuv_planes: None,
        });
        // Cover, as the swatches use: the image fills the widget edge to edge, so
        // the only thing that can clear a corner is the corner SDF.
        primitive.update_viewport(
            N as f32,
            N as f32,
            1.0,
            crate::app::preview_geometry::BarInsets::default(),
        );

        let bounds = Rectangle {
            x: 0.0,
            y: 0.0,
            width: N as f32,
            height: N as f32,
        };
        let viewport = Viewport::with_physical_size(cosmic::iced::Size::new(N, N), 1.0);
        primitive.prepare(&mut pipeline, &device, &queue, &bounds, &viewport);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("swatch corner test clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        primitive.render(
            &pipeline,
            &mut encoder,
            &view,
            &Rectangle {
                x: 0,
                y: 0,
                width: N,
                height: N,
            },
        );

        // 64 px * 4 bytes = 256, already the required row alignment.
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (N * N * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(N * 4),
                    rows_per_image: Some(N),
                },
            },
            wgpu::Extent3d {
                width: N,
                height: N,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let data = slice.get_mapped_range();
        Some(
            data.as_chunks::<4>()
                .0
                .iter()
                .map(|c| [c[0], c[1], c[2], c[3]])
                .collect(),
        )
    }

    /// A PRE-BLURRED filter's swatch must round its corners like every other.
    ///
    /// Pencil is the only filter with `needs_preblur()`, so it is the only one
    /// whose second pass samples an intermediate — and that pass matches
    /// `viewport_size` to the intermediate's dimensions to make its fit math
    /// degenerate to identity. A corner SDF taken from `viewport_size` therefore
    /// cut Pencil's corners from a 640x480 box inside a 64 px widget: a 16 px
    /// radius shrank to ~1.6 px of widget, and the swatch shipped square while
    /// all fourteen other filters rounded.
    ///
    /// Probed along the corner DIAGONAL, not at the extreme corner pixel: the
    /// two boxes share that corner, so it is cut either way and proves nothing —
    /// the size of the arc is the whole difference. At radius 16 the silhouette
    /// clears the diagonal out to ~4.7 px. Measured `redness` at (0,0)..(5,5):
    ///
    /// * from `panel_rect`:   [255, 255, 255, 255, 178, 5] — a 16 px arc.
    /// * from `viewport_size`: [210, 5, 6, 5, 5, 5] — the corner texel and
    ///   nothing else, the same 16 px having landed 18.75 px inside the
    ///   intermediate's edge.
    ///
    /// The centre reads 5 either way, so a test that only checked "the filter
    /// renders" could not have caught this.
    #[test]
    fn preblurred_swatch_rounds_its_corners() {
        assert!(
            FilterType::Pencil.needs_preblur(),
            "this test is about the pre-blur path; Pencil is its only user"
        );
        const RADIUS: f32 = 16.0;
        let Some(px) = render_swatch_corner(FilterType::Pencil, RADIUS) else {
            skip_no_gpu("preblurred_swatch_rounds_its_corners");
            return;
        };
        let at = |x: usize, y: usize| px[y * 64 + x];

        assert!(
            redness(at(32, 32)) < 40,
            "the swatch centre must be covered by the filtered image, got {:?}",
            at(32, 32)
        );
        for i in 0..4 {
            assert!(
                redness(at(i, i)) > 200,
                "({i}, {i}) is inside a {RADIUS} px corner radius and must be cut \
                 away, got {:?} — a covered pixel this far in means the SDF is \
                 being taken from the intermediate's extent instead of the \
                 widget's, which shrinks the radius by the ratio between them",
                at(i, i)
            );
        }

        // The same swatch with no radius covers those pixels, so the loop above
        // is measuring the radius and not some unrelated gap in the coverage.
        let Some(square) = render_swatch_corner(FilterType::Pencil, 0.0) else {
            return;
        };
        assert!(
            redness(square[2 * 64 + 2]) < 40,
            "an unrounded pre-blurred swatch must cover its own corner, got {:?}",
            square[2 * 64 + 2]
        );
    }

    /// The live preview and a filter swatch, both on the pre-blur path, exactly
    /// as the app arranges them: preview first (radius 0, full bounds), then the
    /// picker's swatch (radius 8, a small rect), then both rendered — iced_wgpu
    /// runs every `prepare()` before any `render()`, and that ordering is the
    /// whole bug. Returns the target's pixels.
    fn render_preview_beside_swatch(with_swatch: bool) -> Option<Vec<[u8; 4]>> {
        const N: u32 = 128;
        const FRAME_W: u32 = 640;
        const FRAME_H: u32 = 480;
        const SWATCH: f32 = 32.0;

        let (_gpu_test, device, queue) = headless_device()?;
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut pipeline = VideoPipeline::new(&device, format);

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("preview beside swatch target"),
            size: wgpu::Extent3d {
                width: N,
                height: N,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let viewport = Viewport::with_physical_size(cosmic::iced::Size::new(N, N), 1.0);

        let frame = |id: u64| VideoFrame {
            id,
            width: FRAME_W,
            height: FRAME_H,
            data: crate::backends::camera::types::FrameData::Copied(
                vec![255u8; (FRAME_W * FRAME_H * 4) as usize].into(),
            ),
            format: PixelFormat::RGBA,
            stride: FRAME_W * 4,
            yuv_planes: None,
        };

        // The live preview: Pencil selected, filling the window, and NOT rounded
        // — `corner_radius` 0 is what disables its SDF.
        let mut preview = VideoPrimitive::new(VIDEO_ID_NORMAL);
        preview.filter_type = FilterType::Pencil;
        preview.corner_radius = 0.0;
        preview.update_frame(frame(VIDEO_ID_NORMAL));
        preview.update_viewport(
            N as f32,
            N as f32,
            1.0,
            crate::app::preview_geometry::BarInsets::default(),
        );
        let preview_bounds = Rectangle {
            x: 0.0,
            y: 0.0,
            width: N as f32,
            height: N as f32,
        };
        preview.prepare(&mut pipeline, &device, &queue, &preview_bounds, &viewport);

        // The picker's Pencil swatch: same pre-blur path, its own small rounded
        // rect, and it prepares AFTER the preview because it is a later layer.
        let swatch = if with_swatch {
            let mut swatch = VideoPrimitive::new(VIDEO_ID_FILTER_PREVIEW);
            swatch.filter_type = FilterType::Pencil;
            swatch.corner_radius = 8.0;
            swatch.update_frame(frame(VIDEO_ID_FILTER_PREVIEW));
            swatch.update_viewport(
                SWATCH,
                SWATCH,
                1.0,
                crate::app::preview_geometry::BarInsets::default(),
            );
            let bounds = Rectangle {
                x: N as f32 - SWATCH,
                y: N as f32 - SWATCH,
                width: SWATCH,
                height: SWATCH,
            };
            swatch.prepare(&mut pipeline, &device, &queue, &bounds, &viewport);
            Some((swatch, bounds))
        } else {
            None
        };

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("preview beside swatch clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        preview.render(
            &pipeline,
            &mut encoder,
            &view,
            &Rectangle {
                x: 0,
                y: 0,
                width: N,
                height: N,
            },
        );
        if let Some((swatch, bounds)) = &swatch {
            swatch.render(
                &pipeline,
                &mut encoder,
                &view,
                &Rectangle {
                    x: bounds.x as u32,
                    y: bounds.y as u32,
                    width: bounds.width as u32,
                    height: bounds.height as u32,
                },
            );
        }

        // 128 px * 4 bytes = 512, already a multiple of the 256-byte row alignment.
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (N * N * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(N * 4),
                    rows_per_image: Some(N),
                },
            },
            wgpu::Extent3d {
                width: N,
                height: N,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let data = slice.get_mapped_range();
        Some(
            data.as_chunks::<4>()
                .0
                .iter()
                .map(|c| [c[0], c[1], c[2], c[3]])
                .collect(),
        )
    }

    /// Opening the filter picker while a PRE-BLURRED filter is selected must not
    /// blank the preview.
    ///
    /// Pencil is the only `needs_preblur()` filter, so selecting it and opening
    /// the picker is the one arrangement that puts TWO consumers on the pre-blur
    /// path at once: the preview and Pencil's own swatch. They disagree about
    /// `panel_rect` and `corner_radius`, and the pass that samples the
    /// intermediate used to read them from a single pipeline-owned uniform — so
    /// the swatch, preparing last, won. The preview then drew with a
    /// thumbnail-sized rect and a radius it had explicitly set to 0, and its SDF
    /// rejected every fragment: preview gone, frosted bars (excluded from the
    /// pre-blur path) untouched, which is exactly how it was reported.
    ///
    /// The two-consumer arrangement IS the test. A harness that renders only the
    /// preview passes against the broken code, because with nothing else to
    /// clobber the shared uniform the preview's own values are the last ones
    /// written — the same trap that let an earlier version of this bug survive a
    /// GPU test (see `frosted_consumers_that_disagree_do_not_blank_the_scrim`).
    #[test]
    fn opening_the_picker_does_not_blank_a_preblurred_preview() {
        assert!(
            FilterType::Pencil.needs_preblur(),
            "this test is about the pre-blur path; Pencil is its only user"
        );

        let Some(alone) = render_preview_beside_swatch(false) else {
            skip_no_gpu("opening_the_picker_does_not_blank_a_preblurred_preview");
            return;
        };
        let with_swatch = render_preview_beside_swatch(true).expect("adapter vanished");
        let at = |px: &Vec<[u8; 4]>, x: usize, y: usize| px[y * 128 + x];

        // Sanity: with no picker open the preview covers its own top-left, so the
        // probe below is measuring the swatch's interference and not a gap that
        // was always there.
        assert!(
            redness(at(&alone, 20, 20)) < 40,
            "a Pencil preview with no picker open must cover its own area, got {:?}",
            at(&alone, 20, 20)
        );

        // The real assertion: a coexisting swatch changes nothing about a pixel
        // far from it.
        for (x, y) in [(20, 20), (40, 60), (64, 64)] {
            let got = at(&with_swatch, x, y);
            assert!(
                redness(got) < 40,
                "({x}, {y}) is preview, far from the swatch, and must still be \
                 covered when the picker is open — got {got:?}, red clear \
                 showing through. The swatch prepared last and its panel_rect \
                 and corner_radius reached the preview's pre-blur pass, whose \
                 SDF then cut the whole preview away."
            );
        }

        // ...and the swatch still renders itself correctly alongside it.
        assert!(
            redness(at(&with_swatch, 112, 112)) < 40,
            "the swatch's own centre must be covered, got {:?}",
            at(&with_swatch, 112, 112)
        );
    }

    /// The re-derived `BLUR_PARAMS` must reproduce cosmic-comp's table exactly.
    ///
    /// These 15 pairs were computed BY HAND from
    /// `cosmic-comp/src/backend/render/wayland/blur_effect.rs` and are pinned
    /// here as an independent check on the mirrored construction above: if
    /// upstream's algorithm is edited into ours and both drift together, this
    /// test still notices.
    ///
    /// The interesting entries are the band boundaries. Index 3 -> 4 steps the
    /// offset DOWN (3.0 -> 2.6) while stepping the passes up, so the table is
    /// non-monotonic in offset. And the last band is the `remaining_steps`
    /// saturation: `ceil(5/10 * 15) = 8` steps are requested but only 6 remain,
    /// so its offsets are spaced 5/6 apart and the table totals exactly 15.
    #[test]
    fn compositor_blur_params_match_upstream() {
        let expected: [(u32, f64); 15] = [
            (1, 1.5),
            (1, 2.0),
            (2, 2.5),
            (2, 3.0),
            (3, 2.6),
            (3, 3.2),
            (3, 3.8),
            (3, 4.4),
            (3, 5.0),
            (4, 3.0 + 5.0 / 6.0),
            (4, 3.0 + 10.0 / 6.0),
            (4, 5.5),
            (4, 3.0 + 20.0 / 6.0),
            (4, 3.0 + 25.0 / 6.0),
            (4, 8.0),
        ];
        assert_eq!(COMPOSITOR_BLUR_PARAMS.len(), BLUR_MAX_STEPS);
        for (i, (passes, offset)) in expected.into_iter().enumerate() {
            let got = COMPOSITOR_BLUR_PARAMS[i];
            assert_eq!(got.passes, passes, "passes at index {i}");
            assert!(
                (got.offset - offset).abs() < 1e-9,
                "offset at index {i}: got {}, want {offset}",
                got.offset
            );
        }

        // And the level -> entry mapping, which is the half that actually ships:
        // upstream indexes with `frosted as u8 + 1` (`render/mod.rs`) and clamps
        // with `.min(MAX_STEPS - 1)` (`blur_effect.rs`). Every frost level 0..=13
        // must select the entry cosmic-comp would select, and these are the exact
        // (passes, offset) our shaders then run — so this IS the parity assertion,
        // not a proxy for one.
        for level in 0u8..=13 {
            let got = compositor_blur_params(level);
            let want = expected[level as usize + 1];
            assert_eq!(got.passes, want.0, "passes at frost level {level}");
            assert!(
                (got.offset - want.1).abs() < 1e-9,
                "offset at frost level {level}: got {}, want {}",
                got.offset,
                want.1
            );
        }
        // Medium, spelled out: level 6 -> index 7 -> 3 passes at offset 4.4.
        assert_eq!(compositor_blur_params(6).passes, 3);
        assert!((compositor_blur_params(6).offset - 4.4).abs() < 1e-9);
        // An out-of-range ordinal clamps rather than panicking, as upstream's
        // `.min(MAX_STEPS - 1)` does.
        assert_eq!(compositor_blur_params(200), compositor_blur_params(13));
    }

    /// A pass count must never outrun the target it runs on.
    #[test]
    fn kawase_passes_are_clamped_to_the_target() {
        // Roomy: whatever was asked for.
        assert_eq!(effective_kawase_passes(1080, 2340, 4), 4);
        assert_eq!(effective_kawase_passes(1080, 2340, 1), 1);
        // 16 px on the short side: `16 >> 4 = 1`, so 4 is exactly reachable.
        assert_eq!(effective_kawase_passes(1024, 16, 4), 4);
        // 8 px: the deepest level a 4-pass run needs (`8 >> 4`) is empty.
        assert_eq!(effective_kawase_passes(1024, 8, 4), 3);
        // Degenerate targets must still run one pass, not zero or a panic.
        assert_eq!(effective_kawase_passes(1, 1, 4), 1);
        assert_eq!(effective_kawase_passes(0, 0, 4), 1);
    }

    /// `MAX_KAWASE_PASSES` must cover the table, and `steps` must cover a run.
    ///
    /// Both are silent failures rather than loud ones, which is why they are
    /// pinned rather than trusted. `render()` walks `k` over `0..2 * passes` and
    /// does `let Some(step) = targets.steps.get(k) else { break }` — so a `steps`
    /// array one entry short does not panic and does not warn, it just stops the
    /// ping-pong early and leaves a half-reconstructed image on screen. Likewise
    /// a table entry asking for 5 passes would be clamped away by the `.min()` in
    /// `prepare()`/`render()` and simply blur less than cosmic-comp does.
    ///
    /// Derived from `COMPOSITOR_BLUR_PARAMS` rather than hardcoded: retuning the
    /// table is legitimate, and this should only fail if the table outgrows the
    /// allocation — which is exactly the off-by-one that bites at the top band.
    #[test]
    fn the_step_allocation_covers_the_whole_table() {
        let worst = (0u8..=13)
            .map(|l| compositor_blur_params(l).passes)
            .max()
            .unwrap();
        assert_eq!(
            worst, MAX_KAWASE_PASSES,
            "MAX_KAWASE_PASSES ({MAX_KAWASE_PASSES}) must equal the most passes \
             any frost level asks for ({worst}) — too low silently clamps the top \
             band and under-blurs against cosmic-comp; too high over-allocates"
        );
        // A run is `2 * passes` steps (down then up), so this is what `steps` must
        // hold for the worst level on a target big enough not to clamp.
        assert_eq!(effective_kawase_passes(1080, 2340, worst), worst);

        // And the ALLOCATION really is that big — read off the targets the
        // production path builds, not off the constant it was built from.
        let Some((_gpu_test, device, _queue)) = headless_device() else {
            skip_no_gpu("the_step_allocation_covers_the_whole_table");
            return;
        };
        let pipeline = VideoPipeline::new(&device, SURFACE_FORMAT);
        // Phone-shaped, but inside the 2048 `Limits::downlevel_defaults` cap; a
        // 1024 short side still carries all 4 passes (`1024 >> 4 = 64`).
        pipeline.ensure_blur_targets(VIDEO_ID_FROSTED, &device, 1024, 2048, SURFACE_FORMAT);
        let targets = pipeline.blur_targets.read().unwrap();
        let steps = targets.get(&VIDEO_ID_FROSTED).expect("targets").steps.len();
        assert_eq!(
            steps,
            2 * worst as usize,
            "`steps` must hold a full down+up run at the worst level ({worst} \
             passes = {} hops), or `render()`'s `steps.get(k)` breaks the loop \
             early and silently renders a half-finished chain",
            2 * worst
        );
    }

    /// The blur's cost must not quietly grow.
    ///
    /// `render()` documents the chain's fill rate in screen areas — 2.64 S at
    /// Medium, 2.66 S at max frost, "essentially independent of the setting". That
    /// accounting is load-bearing (it is the argument for why pass 0 is worth
    /// keeping) and it is a comment, so nothing stops a change from doubling the
    /// real cost while it goes on claiming 2.64.
    ///
    /// This recomputes it from the SAME recurrence `render()` walks — `dst_level`
    /// per step `k`, area `1 / 4^level` — so it tracks a retune of the table
    /// instead of fighting it, and only fires on a structural change: an extra
    /// full-resolution pass, a pass that stops halving, or a run that no longer
    /// mirrors. The bound is the structure's own claim (under 3 screen areas, and
    /// flat across the range), not a magic number.
    #[test]
    fn the_blur_chain_costs_what_its_docs_claim() {
        // Screen areas written by one chain at `passes`: pass 0's transform into
        // the full-resolution target, then the ping-pong's destinations.
        let fill = |passes: u32| -> f64 {
            let mut total = 1.0; // pass 0 writes 1.00 S
            for k in 0..(2 * passes) {
                let dst_level = if k < passes {
                    k + 1
                } else {
                    2 * passes - k - 1
                };
                total += 1.0 / f64::from(1u32 << (2 * dst_level));
            }
            total
        };

        // The documented figures, to two decimals.
        assert!(
            (fill(3) - 2.64).abs() < 0.005,
            "Medium frost (3 passes) should write 2.64 screen areas, got {:.3} — \
             `render()`'s fill-rate docs are now wrong",
            fill(3)
        );
        assert!(
            (fill(4) - 2.66).abs() < 0.005,
            "max frost (4 passes) should write 2.66 screen areas, got {:.3}",
            fill(4)
        );

        // The real claim: cost is essentially flat across the range, because the
        // level-0 upsample dominates. Measured 2.250 / 2.563 / 2.641 / 2.660 for
        // 1..=4 passes — a 0.41 S spread, and converging, since each extra pass
        // adds only its own `4^-level` share. Half a screen area of slack keeps a
        // retune of the table free while still failing on a chain that grew a
        // pass at, or near, full resolution.
        let (lo, hi) = (fill(1), fill(MAX_KAWASE_PASSES));
        assert!(
            hi - lo < 0.5,
            "the chain's cost must stay essentially flat across the frost range, \
             but 1 pass writes {lo:.3} S and {MAX_KAWASE_PASSES} passes {hi:.3} S \
             — `render()`'s docs argue the setting barely matters, and that is \
             what justifies the cost"
        );
        // And the ceiling. Two thirds of this is full-resolution work that is not
        // ours to remove; anything approaching double it is a new full-res pass
        // that IS.
        for passes in 1..=MAX_KAWASE_PASSES {
            assert!(
                fill(passes) < 3.0,
                "the blur chain at {passes} passes writes {:.2} screen areas, over \
                 the 3.0 ceiling — something added full-resolution work to a chain \
                 whose whole cost argument is that it has exactly one such pass \
                 (pass 0) plus upstream's own level-0 upsample",
                fill(passes)
            );
        }
    }

    /// Target size for the step-edge sigma fixture at `params`.
    ///
    /// Sized from the MODEL sigma rather than fixed, because the fixture has to
    /// hold the blur it is measuring: `sigma_from_step_image` reads its plateaus
    /// a sixteenth of the way in from each border, so a blur wide enough to reach
    /// them washes the plateaus together and the estimator bails out — returning
    /// `None`, which callers cannot tell from "no GPU". At max frost the model
    /// sigma is ~56 px, so the 256 px fixture this used to hardcode would SILENTLY
    /// skip rather than measure. 8 sigma across leaves ~4 sigma either side of the
    /// edge, and rounding to a multiple of 64 keeps `n * 4` bytes per row on the
    /// 256-byte alignment `copy_texture_to_buffer` demands.
    fn sigma_fixture_size(params: CompositorBlurParams) -> u32 {
        let want = (kawase_sigma_model(params) * 8.0).ceil() as u32;
        want.max(256).div_ceil(64) * 64
    }

    /// Render a step edge through the frosted path at `params` and return the
    /// actual blur sigma of the result, in physical screen px.
    ///
    /// The frosted chain works in screen px end to end, so there is no scale to
    /// arrange any more: the frame is 4x the target only so the transform pass has
    /// something to downscale, and the measured sigma is directly comparable to
    /// `params`'s own. The sigma is recovered by `sigma_from_step_image` from what
    /// the passes actually did on the GPU, not modelled.
    fn measure_frosted_sigma(params: CompositorBlurParams) -> Option<f32> {
        let n = sigma_fixture_size(params);
        // 4x the target so the transform pass genuinely downscales, but never past
        // the 2048 `Limits::downlevel_defaults` guarantees — at max frost the
        // fixture is 768 px and a 4x source would be 3072, which the device
        // rejects outright. Only max frost is clamped; every lower level still
        // gets its full 4x.
        let src = (n * 4).min(2048);
        let (_gpu_test, device, queue) = headless_device()?;
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut pipeline = VideoPipeline::new(&device, format);

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frosted sigma test target"),
            size: wgpu::Extent3d {
                width: n,
                height: n,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        // Black left of centre, white right of it: a clean step in x.
        let mut data = vec![0u8; (src * src * 4) as usize];
        for y in 0..src {
            for x in (src / 2)..src {
                let i = ((y * src + x) * 4) as usize;
                data[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }

        let mut primitive = VideoPrimitive::new(VIDEO_ID_FROSTED);
        primitive.blur_params = params;
        primitive.update_frame(VideoFrame {
            id: VIDEO_ID_FROSTED,
            width: src,
            height: src,
            data: crate::backends::camera::types::FrameData::Copied(data.into()),
            format: PixelFormat::RGBA,
            stride: src * 4,
            yuv_planes: None,
        });
        primitive.update_viewport(
            n as f32,
            n as f32,
            1.0,
            crate::app::preview_geometry::BarInsets::default(),
        );

        let bounds = Rectangle {
            x: 0.0,
            y: 0.0,
            width: n as f32,
            height: n as f32,
        };
        let viewport = Viewport::with_physical_size(cosmic::iced::Size::new(n, n), 1.0);
        primitive.prepare(&mut pipeline, &device, &queue, &bounds, &viewport);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        primitive.render(
            &pipeline,
            &mut encoder,
            &view,
            &Rectangle {
                x: 0,
                y: 0,
                width: n,
                height: n,
            },
        );

        // `sigma_fixture_size` keeps `n * 4` on the 256-byte row alignment.
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * n * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(n * 4),
                    rows_per_image: Some(n),
                },
            },
            wgpu::Extent3d {
                width: n,
                height: n,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let pixels = slice.get_mapped_range().to_vec();

        // `None` from here means the profile degenerated — the plateaus washed
        // together — NOT that there is no adapter, and the two must not share a
        // return value: every caller treats `None` as "skip, no GPU", so folding
        // them together turns a truncated blur chain into a PASS. That is not
        // hypothetical: undersizing `BlurTargets::steps` to `MAX_KAWASE_PASSES`
        // breaks the up-chain, washes the edge out, and used to make this test
        // report ok while silently measuring nothing at 3 and 4 passes.
        Some(sigma_from_step_image(&pixels, n).unwrap_or_else(|| {
            panic!(
                "the step edge washed out at {} passes / offset {:.3}: the profile \
                 has no measurable plateaus, so the chain did not produce the blur \
                 it was asked for (a truncated ping-pong run looks exactly like \
                 this), or the {n} px fixture is too small for it",
                params.passes, params.offset
            )
        }))
    }

    /// Recover the blur sigma, in target px, from an RGBA readback of a
    /// vertical step edge, via the profile's interquartile width.
    ///
    /// A step edge convolved with a kernel has that kernel's CDF as its profile,
    /// so the distance between the 25% and 75% crossings is `2·Φ⁻¹(0.75)·σ =
    /// 1.349·σ` for a Gaussian. The dual-Kawase is a cascade of box-ish kernels,
    /// so it is Gaussian to well within the 15% these tests allow.
    ///
    /// # Why not the second moment
    ///
    /// The obvious estimator — the second moment of the derivative about its own
    /// mean — is exact for ANY kernel and needs no Gaussian assumption, which is
    /// why it was used before. It cannot survive the film grain (see
    /// [`FROSTED_NOISE`]), which is real output the frosted path is supposed to
    /// have: a second moment weights each column by `x²`, so grain 100 px from the
    /// edge — where the true derivative is exactly zero — is amplified 10,000x.
    /// The grain is a deterministic hash, not fresh noise per run, so this is a
    /// systematic BIAS rather than scatter: it reported sigma 4.6 for a blur of
    /// 0.001, and no amount of row averaging fixes it (the bias only falls as
    /// `1/sqrt(rows)` while the leverage stays).
    ///
    /// The interquartile width has no such leverage. It reads the profile only
    /// where the profile is steep, so a ±3.8-level grain (already attenuated 16x
    /// by averaging every row) moves the crossings by a small fraction of a pixel.
    ///
    /// `lo`/`hi` are measured off the image rather than assumed to be 0 and 255,
    /// so the transition blur's dim does not register as a change in sigma.
    fn sigma_from_step_image(pixels: &[u8], n: u32) -> Option<f32> {
        let mut profile = vec![0.0f64; n as usize];
        for y in 0..n {
            for (x, acc) in profile.iter_mut().enumerate() {
                *acc += f64::from(pixels[((y * n + x as u32) * 4) as usize]) / f64::from(n);
            }
        }

        // The plateaus either side of the edge. A few px in from each border, so a
        // clamped sampler's edge behaviour is not what sets the scale.
        let edge = (n as usize / 16).max(1);
        let lo = profile[..edge].iter().sum::<f64>() / edge as f64;
        let hi = profile[profile.len() - edge..].iter().sum::<f64>() / edge as f64;
        if hi - lo < 8.0 {
            return None;
        }

        // First crossing of `frac`, linearly interpolated between samples.
        let crossing = |frac: f64| -> Option<f64> {
            let level = lo + (hi - lo) * frac;
            profile.windows(2).enumerate().find_map(|(i, w)| {
                (w[0] < level && level <= w[1])
                    .then(|| i as f64 + 0.5 + (level - w[0]) / (w[1] - w[0]))
            })
        };
        let (q1, q3) = (crossing(0.25)?, crossing(0.75)?);
        // 2 * Phi^-1(0.75) = 1.3490.
        Some(((q3 - q1) / 1.348_98) as f32)
    }

    /// Centroid of an off-centre white square rendered through the normal
    /// preview shader and through the blur shader's pass-1 configuration, at
    /// each of `zooms`, as `[(normal, blur); zooms]` in target pixels.
    ///
    /// Both paths get the SAME uniform, and `viewport_size` matches the frame
    /// dims, so the cover/contain math degenerates to identity in both and the
    /// only thing that can move the square is the UV transform chain itself.
    /// A symmetric blur kernel does not move a centroid, so the two paths are
    /// directly comparable even though one of them smears.
    ///
    /// Every probe shares ONE device and ONE pipeline, rewriting the uniform
    /// between draws. Spinning up a headless device per probe raced the other
    /// GPU tests badly enough to segfault the harness roughly one run in five.
    fn transform_probe_centroids(zooms: &[f32]) -> Option<Vec<[(f32, f32); 2]>> {
        const N: u32 = 256;
        // Well inside the frame, and offset from the centre so that zooming has
        // somewhere to move it to.
        const SQ_MIN: u32 = 80;
        const SQ_MAX: u32 = 112;

        let (_gpu_test, device, queue) = headless_device()?;
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut pipeline = VideoPipeline::new(&device, format);

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("zoom alignment target"),
            size: wgpu::Extent3d {
                width: N,
                height: N,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        // Opaque black frame with an opaque white square: the normal shader
        // passes the sampled alpha through, and the centroid weights by
        // luminance, so the background must contribute nothing but must not be
        // transparent.
        let mut data = vec![0u8; (N * N * 4) as usize];
        for y in 0..N {
            for x in 0..N {
                let i = ((y * N + x) * 4) as usize;
                let white = (SQ_MIN..SQ_MAX).contains(&x) && (SQ_MIN..SQ_MAX).contains(&y);
                let v = if white { 255 } else { 0 };
                data[i..i + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }
        pipeline.upload(
            &device,
            &queue,
            VideoFrame {
                id: VIDEO_ID_BLUR,
                width: N,
                height: N,
                data: crate::backends::camera::types::FrameData::Copied(data.into()),
                format: PixelFormat::RGBA,
                stride: N * 4,
                yuv_planes: None,
            },
        );
        pipeline.get_or_create_binding(&device, VIDEO_ID_BLUR, 0)?;
        let binding = pipeline.bindings.get(&(VIDEO_ID_BLUR, 0))?;

        let probe = |use_blur_pipeline: bool, zoom: f32| -> Option<(f32, f32)> {
            queue.write_buffer(
                &binding.viewport_buffer,
                0,
                bytemuck::cast_slice(&[ViewportUniform {
                    viewport_size: [N as f32, N as f32],
                    zoom_level: zoom,
                    ..Default::default()
                }]),
            );

            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("zoom alignment pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                if use_blur_pipeline {
                    render_pass.set_pipeline(&pipeline.pipeline_rgb_blur);
                } else {
                    render_pass.set_pipeline(&pipeline.pipeline_rgba);
                }
                render_pass.set_bind_group(0, Some(&binding.bind_group), &[]);
                render_pass.draw(0..3, 0..1);
            }

            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (N * N * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &target,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(N * 4),
                        rows_per_image: Some(N),
                    },
                },
                wgpu::Extent3d {
                    width: N,
                    height: N,
                    depth_or_array_layers: 1,
                },
            );
            queue.submit([encoder.finish()]);

            let slice = readback.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let pixels = slice.get_mapped_range().to_vec();

            let mut sum = 0.0f64;
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            for y in 0..N {
                for x in 0..N {
                    let i = ((y * N + x) * 4) as usize;
                    let lum = pixels[i] as f64 + pixels[i + 1] as f64 + pixels[i + 2] as f64;
                    sum += lum;
                    sx += lum * (x as f64 + 0.5);
                    sy += lum * (y as f64 + 0.5);
                }
            }
            if sum <= 1e-6 {
                return None;
            }
            Some(((sx / sum) as f32, (sy / sum) as f32))
        };

        zooms
            .iter()
            .map(|&z| Some([probe(false, z)?, probe(true, z)?]))
            .collect()
    }

    /// The frosted backdrop must blur the SAME view of the frame the sharp
    /// preview shows — including digital zoom.
    ///
    /// `video_shader_blur.wgsl` used to ignore `zoom_level` outright (its struct
    /// documented the field as "unused by this shader"), so zooming the preview
    /// moved the sharp image and left every frosted panel, bar and chip blurring
    /// the unzoomed frame behind it. The two visibly disagreed on device — which
    /// defeats the point of a backdrop whose blur is supposed to line up
    /// pixel-for-pixel with what is behind it.
    ///
    /// Agreement alone is not enough to pin this: two shaders that BOTH ignore
    /// zoom also agree. So this asserts the absolute displacement too — the
    /// square must actually have moved to where 2x zoom puts it — which is the
    /// half of the test that fails on the old shader.
    #[test]
    fn frosted_blur_zooms_with_the_preview() {
        let Some(probes) = transform_probe_centroids(&[1.0, 2.0]) else {
            skip_no_gpu("frosted_blur_zooms_with_the_preview");
            return;
        };
        let [normal_1x, blur_1x] = probes[0];
        let [normal_2x, blur_2x] = probes[1];

        // Sanity: unzoomed, the square sits at its authored centre (96, 96) in
        // both paths. This is the control — it passed before the fix too.
        for (name, c) in [("normal", normal_1x), ("blur", blur_1x)] {
            assert!(
                (c.0 - 96.0).abs() < 1.0 && (c.1 - 96.0).abs() < 1.0,
                "{name} at 1x should centre the square at (96, 96), got {c:?}"
            );
        }

        // At 2x the centre magnifies about (128, 128): 128 + 2*(96-128) = 64.
        // THIS is what the old blur shader got wrong — it stayed at 96.
        for (name, c) in [("normal", normal_2x), ("blur", blur_2x)] {
            assert!(
                (c.0 - 64.0).abs() < 1.5 && (c.1 - 64.0).abs() < 1.5,
                "{name} at 2x should magnify the square to (64, 64), got {c:?}"
            );
        }

        // And the two paths must agree with EACH OTHER, tightly: a blur smears
        // the square but cannot move its centroid.
        let dx = (normal_2x.0 - blur_2x.0).abs();
        let dy = (normal_2x.1 - blur_2x.1).abs();
        assert!(
            dx < 1.0 && dy < 1.0,
            "frosted blur must line up with the preview at 2x zoom: \
             normal {normal_2x:?} vs blur {blur_2x:?} (off by {dx:.2}, {dy:.2})"
        );
    }

    /// A vertical step edge (black left, white right) as an `n` x `n` RGBA frame.
    fn step_edge_frame(n: u32) -> Vec<u8> {
        let mut data = vec![0u8; (n * n * 4) as usize];
        for y in 0..n {
            for x in (n / 2)..n {
                let i = ((y * n + x) * 4) as usize;
                data[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        data
    }

    /// Whether `video_id`'s kernel passes have already run for the current frame.
    /// Reads the exact flag `VideoPipeline::render` gates passes 1+2 on.
    fn blur_is_cached(pipeline: &VideoPipeline, video_id: u64) -> bool {
        pipeline
            .blur_targets
            .read()
            .unwrap()
            .get(&video_id)
            .map(|t| t.cached.load(std::sync::atomic::Ordering::Relaxed))
            .unwrap_or(false)
    }

    /// Drive a `VIDEO_ID_BLUR` primitive and TWO `VIDEO_ID_FROSTED` primitives
    /// through ONE frame — every `prepare()` first, then every `render()`, which
    /// is exactly what iced_wgpu does and exactly what the shared-blur-state bug
    /// needed — and return each consumer's measured on-screen sigma.
    ///
    /// The geometry is `measure_frosted_sigma`'s: the frame is 4x the target on
    /// both axes, so the 1/4-res intermediates land at target size and Cover maps
    /// them 1:1 — `int_scale = 1`, so a measured px IS an intermediate texel.
    ///
    /// Returns `(blur_sigma, frosted_sigma)` in physical screen px.
    fn measure_blur_and_frosted_in_one_frame(
        frost_params: CompositorBlurParams,
    ) -> Option<(f32, f32)> {
        const N: u32 = 256;
        const SRC: u32 = N * 4;
        let (_gpu_test, device, queue) = headless_device()?;
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut pipeline = VideoPipeline::new(&device, format);

        let make_target = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: N,
                    height: N,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        // Separate targets only so the two results can be measured apart; both
        // are drawn from the same encoder, in one frame, sharing one pipeline.
        let blur_target = make_target("collision test transition target");
        let frosted_target = make_target("collision test frosted target");
        let blur_view = blur_target.create_view(&wgpu::TextureViewDescriptor::default());
        let frosted_view = frosted_target.create_view(&wgpu::TextureViewDescriptor::default());
        // The second frosted panel blits the cache; its pixels are not measured.
        let frosted_target_2 = make_target("collision test second frosted panel target");
        let frosted_view_2 = frosted_target_2.create_view(&wgpu::TextureViewDescriptor::default());

        let make_frame = |id: u64| VideoFrame {
            id,
            width: SRC,
            height: SRC,
            data: crate::backends::camera::types::FrameData::Copied(step_edge_frame(SRC).into()),
            format: PixelFormat::RGBA,
            stride: SRC * 4,
            yuv_planes: None,
        };

        // The transition blur: `TRANSITION_BLUR_PARAMS` + `TRANSITION_BLUR_DIM`,
        // straight off `VideoPrimitive::new`.
        let blur_primitive = VideoPrimitive::new(VIDEO_ID_BLUR);
        blur_primitive.update_frame(make_frame(VIDEO_ID_BLUR));
        blur_primitive.update_viewport(
            N as f32,
            N as f32,
            1.0,
            crate::app::preview_geometry::BarInsets::default(),
        );

        // The frosted chrome: the compositor's own params, no dim, plus grain.
        // Two panels, because a real frame draws several (picker + chips + bars).
        let make_frosted = || {
            let mut p = VideoPrimitive::new(VIDEO_ID_FROSTED);
            p.blur_params = frost_params;
            p.update_frame(make_frame(VIDEO_ID_FROSTED));
            p.update_viewport(
                N as f32,
                N as f32,
                1.0,
                crate::app::preview_geometry::BarInsets::default(),
            );
            p
        };
        let frosted_primitive = make_frosted();
        let frosted_primitive_2 = make_frosted();

        let bounds = Rectangle {
            x: 0.0,
            y: 0.0,
            width: N as f32,
            height: N as f32,
        };
        let viewport = Viewport::with_physical_size(cosmic::iced::Size::new(N, N), 1.0);

        // EVERY prepare, THEN every render. With one shared set of blur state this
        // is the moment the frosted solve clobbered the transition blur's uniforms.
        blur_primitive.prepare(&mut pipeline, &device, &queue, &bounds, &viewport);
        frosted_primitive.prepare(&mut pipeline, &device, &queue, &bounds, &viewport);
        frosted_primitive_2.prepare(&mut pipeline, &device, &queue, &bounds, &viewport);

        // A fresh frame arrived for both: neither has blurred anything yet.
        assert!(!blur_is_cached(&pipeline, VIDEO_ID_BLUR));
        assert!(!blur_is_cached(&pipeline, VIDEO_ID_FROSTED));

        let clip = Rectangle {
            x: 0,
            y: 0,
            width: N,
            height: N,
        };
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        blur_primitive.render(&pipeline, &mut encoder, &blur_view, &clip);
        // The transition blur's passes ran — and satisfied ONLY its own cache.
        // Before the fix this single flag was global, so the frosted backdrop
        // below skipped its passes and blitted the transition's blur instead.
        assert!(blur_is_cached(&pipeline, VIDEO_ID_BLUR));
        assert!(
            !blur_is_cached(&pipeline, VIDEO_ID_FROSTED),
            "the transition blur must not satisfy the frosted backdrop's cache"
        );

        frosted_primitive.render(&pipeline, &mut encoder, &frosted_view, &clip);
        assert!(blur_is_cached(&pipeline, VIDEO_ID_FROSTED));

        // Second frosted panel, same frame: the flag must still be set, i.e. it
        // takes the cheap blit branch and does NOT re-run passes 1+2. Keying the
        // cache per video_id must not have degraded it into per-primitive.
        frosted_primitive_2.render(&pipeline, &mut encoder, &frosted_view_2, &clip);
        assert!(
            blur_is_cached(&pipeline, VIDEO_ID_FROSTED),
            "a second frosted panel in the same frame must reuse the cached blur"
        );
        assert!(blur_is_cached(&pipeline, VIDEO_ID_BLUR));

        // At most one entry per consumer, ever.
        assert_eq!(pipeline.blur_targets.read().unwrap().len(), 2);

        let readback = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (N * N * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let blur_readback = readback("collision test transition readback");
        let frosted_readback = readback("collision test frosted readback");
        for (texture, buffer) in [
            (&blur_target, &blur_readback),
            (&frosted_target, &frosted_readback),
        ] {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        // 256 px * 4 bytes = 1024, already the required alignment.
                        bytes_per_row: Some(N * 4),
                        rows_per_image: Some(N),
                    },
                },
                wgpu::Extent3d {
                    width: N,
                    height: N,
                    depth_or_array_layers: 1,
                },
            );
        }
        queue.submit([encoder.finish()]);

        blur_readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});
        frosted_readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let blur_pixels = blur_readback.slice(..).get_mapped_range().to_vec();
        let frosted_pixels = frosted_readback.slice(..).get_mapped_range().to_vec();
        Some((
            sigma_from_step_image(&blur_pixels, N)?,
            sigma_from_step_image(&frosted_pixels, N)?,
        ))
    }

    /// The Gaussian sigma, in physical screen px, that a dual-Kawase run at
    /// `params` must produce — derived from upstream's two frag shaders.
    ///
    /// # Derivation
    ///
    /// At down pass `i` the shader gets `half_pixel = 0.5 / (W >> i)` and
    /// `offset / 2^i`, but `v_coords` is normalized over the FULL texture (the
    /// live region is a sub-rect of it), so the tap displacement is a constant
    /// `0.5 * offset` full-texture texels. The content living in that `W >> i`
    /// region is downscaled `2^i`, so in ORIGINAL screen px the tap distance is
    /// `u_i = 0.5 * offset * 2^i`. The upsample passes are the mirror image,
    /// giving `h_k = 0.5 * offset * 2^k` for `k = passes..=1`.
    ///
    /// `blur_downsample.frag` is a centre tap of weight 4 plus four diagonal taps
    /// of weight 1 at `(±u, ±u)`: per-axis variance `4u²/8 = u²/2`.
    ///
    /// `blur_upsample.frag` is 8 taps — weight 1 at `(±2h, 0)`, `(0, ±2h)` and
    /// weight 2 at `(±h, ±h)` — total weight 12: per-axis variance
    /// `(2·(2h)² + 4·2·h²)/12 = 4h²/3`.
    ///
    /// Variances add across independent passes, so
    ///
    /// ```text
    /// var = Σ_{i=0}^{p-1} (0.5·offset·2^i)²/2  +  Σ_{k=1}^{p} 4(0.5·offset·2^k)²/3
    ///     = offset²·(4^p − 1)·(1/24 + 4/9)
    ///     = offset²·(4^p − 1)·35/72
    /// ```
    ///
    /// The bilinear resampling between levels adds a little more variance; it is
    /// ignored here (it is on the order of a texel against a 20+ px sigma).
    ///
    /// This model is TEST-ONLY on purpose. It used to be shipping code, because
    /// the old chain had to solve a ring radius against it; nothing solves against
    /// it now — the shaders take `params` directly — so its only remaining job is
    /// to be an independent statement of what those shaders should do.
    fn kawase_sigma_model(params: CompositorBlurParams) -> f64 {
        (params.offset.powi(2) * (4f64.powi(params.passes as i32) - 1.0) * (35.0 / 72.0)).sqrt()
    }

    /// The transition blur and the frosted backdrop, on screen in the SAME frame,
    /// must each get THEIR OWN blur — not one shared one.
    ///
    /// This is the regression test for the shared-blur-state bug. `VideoPipeline`
    /// held ONE pair of intermediates and ONE `blur_cached` flag for both
    /// consumers, while they drive the chain with very different parameters
    /// (`TRANSITION_BLUR_PARAMS` vs the theme's frost entry; dim 0.61 vs 1.0;
    /// grain 0 vs 0.03). iced_wgpu runs every `prepare()` before any `render()`,
    /// so in a frame containing both — open a picker, then switch camera / change
    /// resolution, or let an HDR+ burst finish (`handle_burst_mode_complete` blurs
    /// for 200 ms without disabling the UI) — the last `prepare()` won the shared
    /// uniform buffers, the first `render()` ran the passes with them and set the
    /// shared flag, and the second reused that result wholesale. Both got one
    /// parameterization; at least one was wrong.
    ///
    /// The rewrite did not make this safe by accident — every Kawase step still
    /// owns a uniform buffer, so the collision is still there to be had if the
    /// keying is ever dropped.
    ///
    /// # What this proves
    ///
    /// The two sigmas are measured off real pixels the GPU produced, through the
    /// real `prepare()`/`render()` path, in one frame, with one shared
    /// `VideoPipeline` — so it is the actual collision under test, not a model of
    /// it. The flag assertions inside the helper cover the caching half.
    ///
    /// # What it does NOT prove
    ///
    /// * Not the compositing: each consumer renders to its own target, because two
    ///   step edges blitted over each other cannot be measured apart. Ordering and
    ///   shared state are faithful; overlap is not.
    /// * Not the dim factors or the grain, which a sigma measurement is blind to by
    ///   construction (it normalizes the derivative). They reach the same per-
    ///   `video_id` uniform buffers as the offsets, so they are fixed by the same
    ///   keying, but they ride on this test rather than being asserted.
    /// * Not that the chain runs exactly once — only that the flag gating it stays
    ///   set across a second frosted panel (asserted in the helper). There is no
    ///   pass counter to read.
    #[test]
    fn blur_and_frosted_do_not_share_one_blur() {
        // Deliberately far from the transition blur's own thickness, and small
        // enough that a 256 px target carries it comfortably.
        const FROST_PARAMS: CompositorBlurParams = CompositorBlurParams {
            passes: 2,
            offset: 2.5,
        };
        let Some((blur_sigma, frosted_sigma)) = measure_blur_and_frosted_in_one_frame(FROST_PARAMS)
        else {
            skip_no_gpu("blur_and_frosted_do_not_share_one_blur");
            return;
        };

        // The same one-sided band `kawase_sigma_matches_the_variance_model` uses,
        // and for the same reason: the model counts the kernels, the chain also
        // resamples between levels. What matters here is only that each consumer
        // lands on ITS OWN parameterization, and the bands do not overlap.
        for (name, got, params) in [
            ("transition blur", blur_sigma, TRANSITION_BLUR_PARAMS),
            ("frosted backdrop", frosted_sigma, FROST_PARAMS),
        ] {
            let model = kawase_sigma_model(params) as f32;
            let ratio = got / model;
            assert!(
                (1.0..=1.35).contains(&ratio),
                "{name} got sigma {got:.2} px against a model of {model:.2} \
                 (ratio {ratio:.3}) — it is being blurred with the other \
                 consumer's params"
            );
        }

        // And, plainly: they are not the same blur.
        assert!(
            blur_sigma > frosted_sigma * 1.5,
            "the two consumers collapsed onto one parameterization: \
             transition {blur_sigma:.2} px vs frosted {frosted_sigma:.2} px"
        );
    }

    /// The chain's on-screen thickness must follow the `(passes, offset)` law.
    ///
    /// `kawase_kernels_match_cosmic_comp` proves the KERNELS are upstream's, tap
    /// for tap. This proves the CHAIN around them is: that pass `i` really reads
    /// the `[0, W>>i]` sub-rect, that its offset really is `offset / 2^i`, that the
    /// up-chain really mirrors the down-chain, and that the whole thing really
    /// lands in physical screen px. Those are the things a correct kernel wired up
    /// wrongly would still get wrong — and they are exactly what
    /// `kawase_sigma_model`'s `offset²·(4^p − 1)` law encodes.
    ///
    /// Measured through the real `prepare()`/`render()` path, on real pixels.
    ///
    /// # Why the tolerance is one-sided
    ///
    /// The model is analytic and counts only the kernels. The real chain also
    /// RESAMPLES between levels — a bilinear downscale on the way down and a
    /// bilinear magnification on the way back up, at every level — and each of
    /// those is itself a small blur. So the measured sigma must be at least the
    /// model's and somewhat more, never less: variance adds, it does not cancel.
    ///
    /// Measured excess runs 13% (3 passes) to 24% (1 pass) — proportionally larger
    /// at low pass counts, where the kernels contribute least and the resampling is
    /// a bigger share of the total. cosmic-comp's chain resamples identically, so
    /// this excess is in THEIR output too: it is a gap in the model, not a gap in
    /// the parity.
    ///
    /// # The top band is the point
    ///
    /// Levels 9 and 13 are `MAX_KAWASE_PASSES` — the FOUR-pass chain, the only
    /// band where `BlurTargets::steps` (sized `2 * MAX_KAWASE_PASSES`) is used to
    /// its last entry. Nothing else renders it end to end: `high_frost_does_not_band`
    /// runs level 13 but only asserts the absence of banding, and a chain that
    /// silently truncated its ping-pong run would blur LESS and band less, sailing
    /// straight through. A sigma measurement cannot be fooled that way — it fails
    /// loudly instead (see the panic in `measure_frosted_sigma`).
    ///
    /// Measured on HEAD — level (passes): model -> rendered, ratio:
    /// 0 (1): 2.4 -> 3.0, 1.243; 2 (2): 8.1 -> 9.3, 1.147; 4 (3): 17.7 -> 20.1,
    /// 1.135; 9 (4): 52.0 -> 58.3, 1.122; 13 (4): 89.1 -> 100.2, 1.125. The excess
    /// falls monotonically as the pass count rises, exactly as the resampling
    /// account predicts, and the top band sits comfortably inside the band.
    #[test]
    fn kawase_sigma_matches_the_variance_model() {
        for level in [0u8, 2, 4, 9, 13] {
            let params = compositor_blur_params(level);
            let model = kawase_sigma_model(params) as f32;
            let Some(got) = measure_frosted_sigma(params) else {
                skip_no_gpu("kawase_sigma_matches_the_variance_model");
                return;
            };
            let ratio = got / model;
            assert!(
                (1.0..=1.35).contains(&ratio),
                "frost level {level} ({} passes, offset {:.3}): rendered sigma \
                 {got:.2} px vs model {model:.2} px (ratio {ratio:.3}) — outside \
                 the 1.00..1.35 the inter-level resampling accounts for",
                params.passes,
                params.offset,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Banding
    //
    // The defect this rewrite exists to fix. The old kernel was a fixed 37-tap
    // rosette — 3 rings x 12 taps at `r/3` spacing, plus a centre tap — run over
    // the SHARP, un-prefiltered sensor frame. At high frost the ring spacing
    // reached ~18 source texels, and a sparse 12-fold lattice sampling sharp data
    // does not approximate a Gaussian: it ghosts 12 displaced copies of whatever
    // it lands on. That is what the user saw on device.
    //
    // A rosette's fingerprint is ANGULAR, so the test is angular: walk a circle
    // around a blurred feature and look for 12-fold periodicity. An isotropic
    // blur is flat around that circle; a rosette is not.
    // -----------------------------------------------------------------------

    /// Strength of PERIODIC ANGULAR STRUCTURE in the luminance around a circle of
    /// radius `r` centred at (`cx`, `cy`): the largest relative amplitude among
    /// angular harmonics 2..=16, over `SAMPLES` evenly spaced bilinear taps.
    ///
    /// # Why harmonics rather than plain variance
    ///
    /// Plain `stddev / mean` is the obvious metric and it does separate the two
    /// kernels — but not by enough to be safe, because it counts the film grain
    /// (see [`FROSTED_NOISE`]) as structure. On this fixture the grain alone puts
    /// a clean Gaussian at ~0.09, against ~0.37 for the rosette: a 4x margin, with
    /// the pass/fail line inside the range grain can move on its own.
    ///
    /// A rosette's signature is not "variance", it is PERIODICITY — 12 ghosts
    /// evenly spaced around the turn. Projecting onto the angular harmonics reads
    /// that directly, and rejects grain almost entirely: white noise spreads its
    /// energy over every harmonic, so any single one keeps only `sqrt(2/SAMPLES)`
    /// of it. The same fixture then scores ~0.018 (Gaussian) against ~0.195
    /// (rosette) — an 11x margin that grain cannot close.
    ///
    /// `SAMPLES` is 360 rather than 96 for exactly that reason: the noise floor
    /// falls as `1/sqrt(SAMPLES)` while the real signal does not move.
    ///
    /// Harmonics 0 and 1 are excluded on purpose. 0 is the mean (that is the
    /// normalizer, not structure) and 1 is a centring error — a sub-pixel offset
    /// between the fixture's disc and the sampler's origin is pure first harmonic,
    /// and it is a property of the test rig, not of the kernel.
    ///
    /// Returns 0 where the mean is too dark to mean anything: the ratio diverges
    /// as the mean approaches zero, so far out in a blurred field's tail it would
    /// be reporting 8-bit quantisation noise as structure. Callers keep the circles
    /// on the blurred edge (see [`banding_fixture`]), where the mean is a healthy
    /// fraction of full scale; the floor is a backstop, not the plan.
    fn angular_variation(sample: &dyn Fn(f32, f32) -> f32, cx: f32, cy: f32, r: f32) -> f32 {
        const SAMPLES: usize = 360;
        /// Out of 255.
        const MIN_MEAN: f32 = 8.0;
        let vals: Vec<f32> = (0..SAMPLES)
            .map(|i| {
                let a = std::f32::consts::TAU * i as f32 / SAMPLES as f32;
                sample(cx + r * a.cos(), cy + r * a.sin())
            })
            .collect();
        let mean = vals.iter().sum::<f32>() / SAMPLES as f32;
        if mean <= MIN_MEAN {
            return 0.0;
        }
        (2..=16)
            .map(|k| {
                let (re, im) =
                    vals.iter()
                        .enumerate()
                        .fold((0.0f32, 0.0f32), |(re, im), (j, v)| {
                            let a = std::f32::consts::TAU * (k * j) as f32 / SAMPLES as f32;
                            (re + v * a.cos(), im + v * a.sin())
                        });
                2.0 * re.hypot(im) / SAMPLES as f32 / mean
            })
            .fold(0.0f32, f32::max)
    }

    /// The fixture both banding tests measure, sized from the blur's own sigma:
    /// `(field size, disc radius, radii to sample)`, all in px.
    ///
    /// A disc, not an impulse: a rosette convolved with an impulse IS the rosette,
    /// which any metric would flag — that would prove nothing about a real image.
    /// A disc is a feature with area and a hard edge, i.e. the kind of thing a
    /// camera frame is full of and the kind of thing that actually ghosted.
    ///
    /// Everything scales with sigma so the same fixture works at any frost level:
    /// a disc fixed at 24 px would vanish into the tail at level 13 (sigma ~89 px)
    /// and the metric would pass on an empty image. The circles sit across the
    /// blurred edge, from `1.3σ - 0.5σ` out to `1.3σ + 1.0σ`, which is where the
    /// gradient — and therefore any structure in the kernel — is strongest. The
    /// field is `8σ` so the outermost circle stays well clear of the border, where
    /// the Kawase's region normalization would have its own say.
    fn banding_fixture(sigma: f32) -> (u32, f32, [f32; 4]) {
        let n = ((sigma * 8.0).ceil() as u32).max(128);
        let disc_r = sigma * 1.3;
        (
            n,
            disc_r,
            [
                disc_r - 0.5 * sigma,
                disc_r,
                disc_r + 0.5 * sigma,
                disc_r + 1.0 * sigma,
            ],
        )
    }

    /// The disc's centre, in the INDEX space every sampler here works in: pixel
    /// `i` sits at coordinate `i`, so an `n`-wide field centres on `(n-1)/2`.
    /// Everything — `disc_field`, both CPU convolutions, the GPU readback — must
    /// agree on this, or a half-pixel offset shows up as angular variation and
    /// the metric measures the fixture instead of the kernel.
    fn disc_centre(n: u32) -> f32 {
        (n as f32 - 1.0) / 2.0
    }

    /// Bilinear sampler over a plain `n` x `n` luminance buffer.
    fn bilinear_field(buf: &[f32], n: u32) -> impl Fn(f32, f32) -> f32 + '_ {
        move |x: f32, y: f32| {
            let at = |ix: i32, iy: i32| -> f32 {
                let ix = ix.clamp(0, n as i32 - 1) as usize;
                let iy = iy.clamp(0, n as i32 - 1) as usize;
                buf[iy * n as usize + ix]
            };
            let (x0, y0) = (x.floor(), y.floor());
            let (fx, fy) = (x - x0, y - y0);
            let (x0, y0) = (x0 as i32, y0 as i32);
            let top = at(x0, y0) * (1.0 - fx) + at(x0 + 1, y0) * fx;
            let bot = at(x0, y0 + 1) * (1.0 - fx) + at(x0 + 1, y0 + 1) * fx;
            top * (1.0 - fy) + bot * fy
        }
    }

    /// A white disc of radius `disc_r` at the centre of an `n` x `n` black field.
    fn disc_field(n: u32, disc_r: f32) -> Vec<f32> {
        let c = disc_centre(n);
        let n = n as usize;
        (0..n * n)
            .map(|i| {
                let (x, y) = ((i % n) as f32, (i / n) as f32);
                if ((x - c).powi(2) + (y - c).powi(2)).sqrt() <= disc_r {
                    255.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Convolve `src` with the OLD 37-tap ring rosette at `radius`, exactly as
    /// `video_shader_blur.wgsl` used to: 3 rings of 12 taps at `radius·k/3`, each
    /// weighted `exp(-r_ring²/2σ²)` with `σ = radius/2.5`, plus a centre tap of
    /// weight 1. Rings 2 and 3 are rotated by the golden angle, as they were.
    ///
    /// This is the deleted kernel, preserved in the one test that needs it: the
    /// test that proves the metric can see it. Without this, `high_frost_does_not_band`
    /// would be a test with no demonstrated power to fail.
    fn convolve_old_rosette(src: &[f32], n: u32, radius: f32) -> Vec<f32> {
        let sigma = radius / 2.5;
        let mut taps: Vec<(f32, f32, f32)> = vec![(0.0, 0.0, 1.0)];
        for ring in 1..=3 {
            let r = radius * ring as f32 / 3.0;
            let w = (-(r * r) / (2.0 * sigma * sigma)).exp();
            let ring_offset = (ring - 1) as f32 * 2.399_963_2;
            for i in 0..12 {
                let a = std::f32::consts::TAU * i as f32 / 12.0 + ring_offset;
                taps.push((r * a.cos(), r * a.sin(), w));
            }
        }
        let total: f32 = taps.iter().map(|t| t.2).sum();
        let sample = bilinear_field(src, n);
        let n = n as usize;
        (0..n * n)
            .map(|i| {
                let (x, y) = ((i % n) as f32, (i / n) as f32);
                taps.iter()
                    .map(|&(dx, dy, w)| w * sample(x + dx, y + dy))
                    .sum::<f32>()
                    / total
            })
            .collect()
    }

    /// Convolve `src` with a true isotropic Gaussian of `sigma` — the control.
    fn convolve_gaussian(src: &[f32], n: u32, sigma: f32) -> Vec<f32> {
        let radius = (sigma * 3.0).ceil() as i32;
        let kernel: Vec<f32> = (-radius..=radius)
            .map(|i| (-(i as f32).powi(2) / (2.0 * sigma * sigma)).exp())
            .collect();
        let norm: f32 = kernel.iter().sum();
        let n = n as usize;
        let at = |buf: &[f32], x: i32, y: i32| -> f32 {
            buf[(y.clamp(0, n as i32 - 1) as usize) * n + (x.clamp(0, n as i32 - 1) as usize)]
        };
        // Separable: horizontal then vertical.
        let mut tmp = vec![0.0f32; n * n];
        for y in 0..n as i32 {
            for x in 0..n as i32 {
                tmp[y as usize * n + x as usize] = kernel
                    .iter()
                    .enumerate()
                    .map(|(k, w)| w * at(src, x + k as i32 - radius, y))
                    .sum::<f32>()
                    / norm;
            }
        }
        let mut out = vec![0.0f32; n * n];
        for y in 0..n as i32 {
            for x in 0..n as i32 {
                out[y as usize * n + x as usize] = kernel
                    .iter()
                    .enumerate()
                    .map(|(k, w)| w * at(&tmp, x, y + k as i32 - radius))
                    .sum::<f32>()
                    / norm;
            }
        }
        out
    }

    /// Worst angular variation across the radii where a blurred disc's edge lives.
    fn worst_variation(sample: &dyn Fn(f32, f32) -> f32, c: f32, radii: &[f32]) -> f32 {
        radii
            .iter()
            .map(|&r| angular_variation(sample, c, c, r))
            .fold(0.0f32, f32::max)
    }

    /// Ceiling on `angular_variation` for a blur we are willing to ship.
    ///
    /// Set between the two kernels `banding_metric_catches_the_old_rosette`
    /// measures, and nearer the clean end: on that fixture an isotropic Gaussian
    /// scores ~0.018 (grain included) and the rosette ~0.195, so the line sits
    /// roughly a factor of 3 clear of both. The dual-Kawase is not perfectly
    /// isotropic — it is a cascade of box-ish 5- and 8-tap kernels, so it has mild
    /// 4-fold structure — which is why this is not pinned at the Gaussian's score.
    const BANDING_THRESHOLD: f32 = 0.05;

    /// The banding metric must actually catch the kernel it was written for.
    ///
    /// A test that cannot fail against the bug it targets is worthless, and this
    /// one CANNOT be demonstrated against the shipping code — the rosette is gone.
    /// So it is demonstrated against a faithful CPU reimplementation of it
    /// (`convolve_old_rosette`), with a true Gaussian of matched sigma as the
    /// control. Same disc, same sigma, same radii, same metric: the only variable
    /// is the kernel's shape.
    ///
    /// The radius is the one the OLD chain actually used at high frost on device —
    /// `55` source texels, the number in the bug report — and the Gaussian is
    /// matched to the `0.32813 · 55 = 18.05` sigma the rosette was solved to have.
    #[test]
    fn banding_metric_catches_the_old_rosette() {
        const RADIUS: f32 = 55.0;
        /// The rosette's own sigma per unit radius. This was a shipping constant
        /// (`RING_KERNEL_SIGMA_PER_RADIUS`) that the frost solve inverted; it
        /// survives only here, to size the control Gaussian fairly.
        const SIGMA_PER_RADIUS: f32 = 0.328_13;
        let sigma = SIGMA_PER_RADIUS * RADIUS;
        let (n, disc_r, radii) = banding_fixture(sigma);

        let disc = disc_field(n, disc_r);
        let c = disc_centre(n);
        let rosette = convolve_old_rosette(&disc, n, RADIUS);
        let gaussian = convolve_gaussian(&disc, n, sigma);

        let rosette_var = worst_variation(&bilinear_field(&rosette, n), c, &radii);
        let gaussian_var = worst_variation(&bilinear_field(&gaussian, n), c, &radii);

        assert!(
            gaussian_var < BANDING_THRESHOLD,
            "an isotropic Gaussian must pass the metric, got {gaussian_var:.4} \
             against a threshold of {BANDING_THRESHOLD}"
        );
        assert!(
            rosette_var > BANDING_THRESHOLD,
            "the metric fails to catch the 37-tap rosette that caused the bug: \
             got {rosette_var:.4}, threshold {BANDING_THRESHOLD} — the threshold \
             is too loose, or the metric is measuring the wrong thing"
        );
        // Not a squeaker: the rosette must be far worse, not marginally worse.
        assert!(
            rosette_var > gaussian_var * 5.0,
            "rosette {rosette_var:.4} vs gaussian {gaussian_var:.4} — the metric \
             barely separates them, so it will not separate them on real data"
        );
    }

    /// Render the banding fixture through the frosted path at `params` and return
    /// the worst angular variation around the blurred disc's edge.
    ///
    /// 1:1 geometry — the frame is the target's size — so the transform pass does
    /// no downscaling and the Kawase gets the sharpest input it will ever see.
    /// That is deliberate: a downscale would prefilter the frame and hide exactly
    /// the failure mode under test.
    fn measure_frosted_banding(
        params: CompositorBlurParams,
        format: wgpu::TextureFormat,
    ) -> Option<f32> {
        let sigma = kawase_sigma_model(params) as f32;
        let (n, disc_r, radii) = banding_fixture(sigma);
        let (_gpu_test, device, queue) = headless_device()?;
        let mut pipeline = VideoPipeline::new(&device, format);

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("banding test target"),
            size: wgpu::Extent3d {
                width: n,
                height: n,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        let disc = disc_field(n, disc_r);
        let mut data = vec![0u8; (n * n * 4) as usize];
        for (i, v) in disc.iter().enumerate() {
            let v = *v as u8;
            data[i * 4..i * 4 + 4].copy_from_slice(&[v, v, v, 255]);
        }

        let mut primitive = VideoPrimitive::new(VIDEO_ID_FROSTED);
        primitive.blur_params = params;
        primitive.update_frame(VideoFrame {
            id: VIDEO_ID_FROSTED,
            width: n,
            height: n,
            data: crate::backends::camera::types::FrameData::Copied(data.into()),
            format: PixelFormat::RGBA,
            stride: n * 4,
            yuv_planes: None,
        });
        primitive.update_viewport(
            n as f32,
            n as f32,
            1.0,
            crate::app::preview_geometry::BarInsets::default(),
        );

        let bounds = Rectangle {
            x: 0.0,
            y: 0.0,
            width: n as f32,
            height: n as f32,
        };
        let viewport = Viewport::with_physical_size(cosmic::iced::Size::new(n, n), 1.0);
        primitive.prepare(&mut pipeline, &device, &queue, &bounds, &viewport);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        primitive.render(
            &pipeline,
            &mut encoder,
            &view,
            &Rectangle {
                x: 0,
                y: 0,
                width: n,
                height: n,
            },
        );

        // `copy_texture_to_buffer` needs 256-byte-aligned rows; the fixture's size
        // follows sigma, so pad rather than assume.
        let bytes_per_row = (n * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (bytes_per_row * n) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(n),
                },
            },
            wgpu::Extent3d {
                width: n,
                height: n,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let padded = slice.get_mapped_range().to_vec();
        let pixels: Vec<u8> = (0..n)
            .flat_map(|y| {
                let start = (y * bytes_per_row) as usize;
                padded[start..start + (n * 4) as usize].to_vec()
            })
            .collect();

        // In the shader's own space, so one threshold covers both formats.
        let luma: Vec<f32> = (0..(n * n) as usize)
            .map(|i| {
                (0..3)
                    .map(|k| decode_channel(pixels[i * 4 + k], format))
                    .sum::<f32>()
                    / 3.0
            })
            .collect();
        Some(worst_variation(
            &bilinear_field(&luma, n),
            disc_centre(n),
            &radii,
        ))
    }

    /// THE test this rewrite exists for: frost must not band, at any level.
    ///
    /// `banding_metric_catches_the_old_rosette` establishes that the metric and
    /// the threshold catch the kernel that caused the bug. This runs the SHIPPING
    /// chain — frost level -> `BLUR_PARAMS` entry -> transform pass -> real Kawase
    /// kernels -> composite — over the same fixture (scaled to each level's sigma)
    /// and asserts it comes out clean.
    ///
    /// It passes for a structural reason, not a tuned one: every Kawase kernel is
    /// dense relative to its own level — taps are `0.5 · offset` apart in a texture
    /// whose content is `2^i` downscaled — and only ever samples data the previous
    /// pass band-limited. There is no sparse lattice anywhere in the chain to
    /// ghost, which is why the top of the range is no worse than the bottom.
    /// # Run over BOTH target formats, because only one of them ships
    ///
    /// `SURFACE_FORMAT` is what the blur targets really have on device, and it is
    /// the harder case: every one of the chain's `2 * passes` hops requantises
    /// through the sRGB curve, whose 8-bit grid is coarsest exactly in the
    /// highlights — which is where this fixture's disc lives. Banding is a
    /// quantisation artefact, so testing only the uniform-grid format measured
    /// the friendlier of the two and called it shipping. The linear format is
    /// kept alongside it to separate a kernel regression (both fail) from a
    /// precision one (only sRGB fails).
    ///
    /// Measured on HEAD, angular variation against the 0.05 threshold — levels
    /// 0/6/9/13, linear: 0.0285 / 0.0364 / 0.0299 / 0.0268; sRGB: 0.0276 /
    /// 0.0293 / 0.0278 / 0.0310. The sRGB path is NOT the worse one, which is
    /// itself the result worth having: the chain's own dithering (`FROSTED_NOISE`
    /// at the composite) covers the coarser highlight grid, so the extra
    /// round-trips cost nothing measurable. Note the headroom is only ~1.4x at
    /// the tightest point, so a change that doubles the variation fails here
    /// rather than merely looking worse on device.
    #[test]
    fn high_frost_does_not_band() {
        // The whole range, because the claim is structural. 9 and 13 are the point
        // — the top band (4 passes), where the old chain's solved radius was
        // largest and its rings furthest apart — but 0 and 6 cost nothing and pin
        // that nothing regressed at the levels that already looked fine.
        for format in BLUR_TARGET_FORMATS {
            for level in [0u8, 6, 9, 13] {
                let params = compositor_blur_params(level);
                let Some(var) = measure_frosted_banding(params, format) else {
                    skip_no_gpu("high_frost_does_not_band");
                    return;
                };
                assert!(
                    var < BANDING_THRESHOLD,
                    "frost level {level} ({} passes, offset {:.3}) bands into a \
                     {format:?} target: angular variation {var:.4} exceeds \
                     {BANDING_THRESHOLD}",
                    params.passes,
                    params.offset
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Kernel transcription
    //
    // `kawase_sigma_matches_the_variance_model` checks the SHAPE of the chain
    // against an analytic model, which is a weak instrument: dropping one of the
    // downsample's four diagonal taps moves the sigma by ~3%, far inside its
    // tolerance. So the kernels are also checked directly, tap for tap, against a
    // CPU transcription of upstream's two frag shaders — the strong statement that
    // our port IS their kernel.
    // -----------------------------------------------------------------------

    /// Bilinear texture fetch with ClampToEdge, matching what the GPU sampler does
    /// for `textureSample` at normalized coords `v` over an `n` x `n` RGBA image.
    fn cpu_texture_sample(src: &[f32], n: u32, v: (f32, f32)) -> [f32; 4] {
        let (x, y) = (v.0 * n as f32 - 0.5, v.1 * n as f32 - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let at = |ix: i32, iy: i32| -> [f32; 4] {
            let ix = ix.clamp(0, n as i32 - 1) as usize;
            let iy = iy.clamp(0, n as i32 - 1) as usize;
            let i = (iy * n as usize + ix) * 4;
            [src[i], src[i + 1], src[i + 2], src[i + 3]]
        };
        let (x0, y0) = (x0 as i32, y0 as i32);
        let (a, b, c, d) = (
            at(x0, y0),
            at(x0 + 1, y0),
            at(x0, y0 + 1),
            at(x0 + 1, y0 + 1),
        );
        std::array::from_fn(|k| {
            let top = a[k] * (1.0 - fx) + b[k] * fx;
            let bot = c[k] * (1.0 - fx) + d[k] * fx;
            top * (1.0 - fy) + bot * fy
        })
    }

    /// CPU transcription of `blur_downsample.frag` / `blur_upsample.frag`, from
    /// cosmic-comp — written from THEIR source, not from ours, so it is an
    /// independent statement of the kernel rather than a copy of the thing under
    /// test.
    fn cpu_kawase(
        src: &[f32],
        n: u32,
        dst: u32,
        uv_scale: f32,
        sub: u32,
        offset: f32,
        down: bool,
    ) -> Vec<[f32; 4]> {
        let hp = 0.5 / sub as f32;
        let add = |sum: &mut [f32; 4], s: [f32; 4], w: f32| {
            for k in 0..4 {
                sum[k] += s[k] * w;
            }
        };
        (0..dst * dst)
            .map(|i| {
                let (dx, dy) = ((i % dst) as f32 + 0.5, (i / dst) as f32 + 0.5);
                let v = (dx / dst as f32 * uv_scale, dy / dst as f32 * uv_scale);
                let tap = |ox: f32, oy: f32| {
                    cpu_texture_sample(src, n, (v.0 + ox * offset, v.1 + oy * offset))
                };
                let mut sum = [0.0f32; 4];
                if down {
                    add(&mut sum, tap(0.0, 0.0), 4.0);
                    add(&mut sum, tap(-hp, -hp), 1.0);
                    add(&mut sum, tap(hp, hp), 1.0);
                    add(&mut sum, tap(hp, -hp), 1.0);
                    add(&mut sum, tap(-hp, hp), 1.0);
                } else {
                    add(&mut sum, tap(-hp * 2.0, 0.0), 1.0);
                    add(&mut sum, tap(-hp, hp), 2.0);
                    add(&mut sum, tap(0.0, hp * 2.0), 1.0);
                    add(&mut sum, tap(hp, hp), 2.0);
                    add(&mut sum, tap(hp * 2.0, 0.0), 1.0);
                    add(&mut sum, tap(hp, -hp), 2.0);
                    add(&mut sum, tap(0.0, -hp * 2.0), 1.0);
                    add(&mut sum, tap(-hp, -hp), 2.0);
                }
                if sum[3] == 0.0 {
                    [0.0; 4]
                } else {
                    std::array::from_fn(|k| sum[k] / sum[3])
                }
            })
            .collect()
    }

    /// Run ONE Kawase step on the GPU with hand-built uniforms, bypassing
    /// `prepare()`/`render()`: it is the kernel itself under test, not the chain
    /// that drives it.
    #[allow(clippy::too_many_arguments)]
    fn gpu_kawase_step(
        src_rgba: &[u8],
        n: u32,
        dst: u32,
        uv_scale: f32,
        sub: u32,
        offset: f32,
        down: bool,
        format: wgpu::TextureFormat,
    ) -> Option<Vec<f32>> {
        let (_gpu_test, device, queue) = headless_device()?;
        let mut pipeline = VideoPipeline::new(&device, format);

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("kawase kernel test target"),
            size: wgpu::Extent3d {
                width: n,
                height: n,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        pipeline.upload(
            &device,
            &queue,
            VideoFrame {
                id: VIDEO_ID_BLUR,
                width: n,
                height: n,
                data: crate::backends::camera::types::FrameData::Copied(src_rgba.to_vec().into()),
                format: PixelFormat::RGBA,
                stride: n * 4,
                yuv_planes: None,
            },
        );
        pipeline.get_or_create_binding(&device, VIDEO_ID_BLUR, 0)?;
        let binding = pipeline.bindings.get(&(VIDEO_ID_BLUR, 0))?;
        queue.write_buffer(
            &binding.viewport_buffer,
            0,
            bytemuck::cast_slice(&[ViewportUniform {
                viewport_size: [sub as f32, sub as f32],
                uv_scale: [uv_scale, uv_scale],
                kawase_offset: offset,
                ..Default::default()
            }]),
        );

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("kawase kernel test pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            render_pass.set_viewport(0.0, 0.0, dst as f32, dst as f32, 0.0, 1.0);
            render_pass.set_scissor_rect(0, 0, dst, dst);
            render_pass.set_pipeline(if down {
                &pipeline.pipeline_kawase_down
            } else {
                &pipeline.pipeline_kawase_up
            });
            render_pass.set_bind_group(0, Some(&binding.bind_group), &[]);
            render_pass.draw(0..3, 0..1);
        }

        let bytes_per_row = (n * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (bytes_per_row * n) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(n),
                },
            },
            wgpu::Extent3d {
                width: n,
                height: n,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let padded = slice.get_mapped_range().to_vec();
        // Returned in the SHADER's space, on a 0..1 scale: on an sRGB target the
        // hardware encoded what the kernel wrote, and it is the kernel that is
        // under test, not the transfer function.
        let mut out = Vec::with_capacity((dst * dst * 3) as usize);
        for y in 0..dst {
            let row = (y * bytes_per_row) as usize;
            for x in 0..dst {
                let i = row + (x * 4) as usize;
                for k in 0..3 {
                    out.push(decode_channel(padded[i + k], format) / 255.0);
                }
            }
        }
        Some(out)
    }

    /// Our Kawase kernels must be cosmic-comp's Kawase kernels, tap for tap.
    ///
    /// Both entry points are run on the GPU over a deterministic pseudo-random
    /// image — chosen so EVERY tap position carries independent information, which
    /// a smooth image would not — and compared per pixel against a CPU
    /// transcription of upstream's `blur_downsample.frag` / `blur_upsample.frag`.
    ///
    /// This is what catches a dropped tap, a weight of 2 where 1 belongs, a sign
    /// flip on a diagonal, or a `half_pixel` computed off the wrong size. The
    /// sigma test cannot: dropping one of the downsample's four diagonals moves
    /// the sigma by ~3%, well inside its tolerance.
    ///
    /// The geometry is the real one — a source sub-rect smaller than the texture,
    /// `uv_scale` renormalizing to full-texture coords — because that mapping is
    /// itself the part of the port most likely to be wrong.
    #[test]
    fn kawase_kernels_match_cosmic_comp() {
        const N: u32 = 64;
        // Deterministic hash noise: every texel independent, so no tap can be
        // dropped without moving the result.
        let src_rgba: Vec<u8> = (0..N * N)
            .flat_map(|i| {
                let h = |k: u32| {
                    ((i.wrapping_mul(2_654_435_761).wrapping_add(k * 40_503)) >> 13 & 0xff) as u8
                };
                [h(1), h(2), h(3), 255]
            })
            .collect();
        let src_f: Vec<f32> = src_rgba.iter().map(|&v| f32::from(v) / 255.0).collect();

        // (down, dst, uv_scale, sub, offset). The first is a level-0 downsample
        // (source sub-rect = whole texture); the second is an upsample reading a
        // half-size sub-rect, i.e. `uv_scale = 0.5`, which is the case where a
        // mistaken normalization would hide.
        for format in BLUR_TARGET_FORMATS {
            for (down, dst, uv_scale, sub, offset) in [
                (true, 32u32, 1.0f32, 64u32, 3.0f32),
                (false, 64, 0.5, 32, 2.5),
            ] {
                let Some(gpu) =
                    gpu_kawase_step(&src_rgba, N, dst, uv_scale, sub, offset, down, format)
                else {
                    skip_no_gpu("kawase_kernels_match_cosmic_comp");
                    return;
                };
                let cpu = cpu_kawase(&src_f, N, dst, uv_scale, sub, offset, down);

                let worst = cpu
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        (0..3)
                            .map(|k| (gpu[i * 3 + k] - c[k]).abs())
                            .fold(0.0f32, f32::max)
                    })
                    .fold(0.0f32, f32::max);
                // The GPU quantises to 8 bits and interpolates at reduced
                // precision; any real kernel error on this input is orders larger.
                //
                // The sRGB target gets a wider allowance because its 8-bit grid is
                // NON-uniform: decoding a code back to linear multiplies the step
                // by the curve's slope there, so the same one-code error reads
                // larger. Measured worst error on HEAD, in levels — linear:
                // 0.50 (down) / 0.62 (up); sRGB: 0.97 / 1.03. So the format that
                // actually ships carries ~1.7x the requantisation error the suite
                // used to measure, which is the gap this parameterisation closes.
                // Both tolerances keep ~3-4x headroom over the measurement, which
                // a dropped tap or a mis-weighted one clears by orders.
                let tol = if format.is_srgb() { 3.0 } else { 2.5 } / 255.0;
                assert!(
                    worst < tol,
                    "{} kernel disagrees with cosmic-comp's by up to {:.4} ({:.1} \
                     levels) at dst {dst}, uv_scale {uv_scale}, offset {offset}, \
                     target {format:?}",
                    if down { "downsample" } else { "upsample" },
                    worst,
                    worst * 255.0
                );
            }
        }
    }

    /// Render a flat mid-grey frame through `video_id`'s blur chain and return
    /// `(mean, stddev)` of the centre region, in 8-bit levels.
    ///
    /// Flat and mid-grey on purpose: a blur of a constant is that constant, so
    /// anything the result varies by is the composite's own doing, and mid-grey
    /// leaves headroom for grain to swing both ways instead of clamping.
    fn measure_composite_flat(video_id: u64) -> Option<(f32, f32)> {
        const N: u32 = 64;
        const GREY: u8 = 128;
        let (_gpu_test, device, queue) = headless_device()?;
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut pipeline = VideoPipeline::new(&device, format);

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("composite flat test target"),
            size: wgpu::Extent3d {
                width: N,
                height: N,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        let primitive = VideoPrimitive::new(video_id);
        primitive.update_frame(VideoFrame {
            id: video_id,
            width: N,
            height: N,
            data: crate::backends::camera::types::FrameData::Copied(
                std::iter::repeat_n([GREY, GREY, GREY, 255], (N * N) as usize)
                    .flatten()
                    .collect::<Vec<u8>>()
                    .into(),
            ),
            format: PixelFormat::RGBA,
            stride: N * 4,
            yuv_planes: None,
        });
        primitive.update_viewport(
            N as f32,
            N as f32,
            1.0,
            crate::app::preview_geometry::BarInsets::default(),
        );

        let bounds = Rectangle {
            x: 0.0,
            y: 0.0,
            width: N as f32,
            height: N as f32,
        };
        let viewport = Viewport::with_physical_size(cosmic::iced::Size::new(N, N), 1.0);
        primitive.prepare(&mut pipeline, &device, &queue, &bounds, &viewport);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        primitive.render(
            &pipeline,
            &mut encoder,
            &view,
            &Rectangle {
                x: 0,
                y: 0,
                width: N,
                height: N,
            },
        );
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (N * N * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(N * 4),
                    rows_per_image: Some(N),
                },
            },
            wgpu::Extent3d {
                width: N,
                height: N,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let pixels = slice.get_mapped_range().to_vec();

        // The centre quarter only: the Kawase's region normalization has its own
        // (correct) say near the border, and that is not what is being measured.
        let vals: Vec<f32> = (N / 4..N * 3 / 4)
            .flat_map(|y| (N / 4..N * 3 / 4).map(move |x| ((y * N + x) * 4) as usize))
            .map(|i| f32::from(pixels[i]))
            .collect();
        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
        Some((mean, var.sqrt()))
    }

    /// The composite must dim the transition blur and grain the frosted chrome —
    /// and must not confuse the two.
    ///
    /// Both are claims about the material that a sigma measurement is blind to by
    /// construction (it normalizes the profile), so they are asserted here on a
    /// flat field, where a blur contributes nothing and the composite is the only
    /// thing that can move a pixel.
    ///
    /// This also pins where the dim LIVES. It used to be applied once per pass and
    /// compound over exactly three; the Kawase's pass count is level-dependent, so
    /// the same per-pass factor would now make the transition blur's darkness a
    /// function of the theme's frost setting — which is absurd, and which this
    /// catches: [`TRANSITION_BLUR_DIM`] is a total, applied once, at the composite.
    #[test]
    fn composite_dims_the_transition_and_grains_the_frost() {
        let Some((frost_mean, frost_std)) = measure_composite_flat(VIDEO_ID_FROSTED) else {
            skip_no_gpu("composite_dims_the_transition_and_grains_the_frost");
            return;
        };
        let Some((blur_mean, blur_std)) = measure_composite_flat(VIDEO_ID_BLUR) else {
            return;
        };

        // Frosted: undimmed. A blur of flat 128 is 128.
        assert!(
            (frost_mean - 128.0).abs() < 3.0,
            "the frosted backdrop must not dim: flat 128 came out at {frost_mean:.1}"
        );
        // Frosted: grained. `FROSTED_NOISE` is a uniform ±0.015 swing, i.e. ±3.8
        // levels, whose stddev is `2*3.8/sqrt(12) = 2.2`.
        let want_std = FROSTED_NOISE * 255.0 / 12.0f32.sqrt();
        assert!(
            (frost_std - want_std).abs() < 0.6,
            "the frosted backdrop's grain should have stddev ~{want_std:.2} levels, \
             got {frost_std:.2} — cosmic-comp's glass has grain and ours must too"
        );

        // Transition: dimmed by the TOTAL, once.
        let want_mean = 128.0 * TRANSITION_BLUR_DIM;
        assert!(
            (blur_mean - want_mean).abs() < 3.0,
            "the transition blur must dim flat 128 to {want_mean:.1}, got {blur_mean:.1}"
        );
        // Transition: not grained. It is a veil, not glass.
        assert!(
            blur_std < 0.6,
            "the transition blur must not be grained, got stddev {blur_std:.2}"
        );
    }

    /// Drive `FrostedScrim`'s ACTUAL draw pattern through the real
    /// `prepare()`/`render()` path, and report what lands inside the bars.
    ///
    /// This is the scrim's geometry, not a model of it: ONE primitive cloned
    /// once per bar — each clone taking its OWN `FrameViewportData` (see
    /// `VideoPrimitive`'s `Clone`) — while each clone is `prepare`d with, and
    /// `render`ed scissored to, its own bar rect. That asymmetry (viewport =
    /// whole preview, scissor = one bar) is the whole reason the scrim can fail
    /// where a panel does not, so it has to be exercised, not asserted about.
    ///
    /// The geometry is the phone's: a tall portrait window, a landscape sensor
    /// mounted at 90°, and a 1:1 aspect crop — the configuration the crop bars
    /// actually appear in.
    ///
    /// The frame is a GREYSCALE ramp and `letterbox_color` is pure MAGENTA, so
    /// the green channel separates the two by construction: green = 0 means the
    /// bar sampled letterbox, green with real variance means it sampled blurred
    /// preview. Returns `(mean, stddev)` of green inside the top and bottom bars.
    ///
    /// `filter` goes to the scrim AND the chip, because that is the only
    /// configuration the app can produce: every `VIDEO_ID_FROSTED` primitive is
    /// minted by one `make_primitive` call from one `VideoWidgetConfig`. They
    /// must agree — the chain's output is cached per `video_id`, so two
    /// consumers with different filters would race for it exactly as ones with
    /// different fits do (see `BlurTargets`).
    fn scrim_bar_green(blend: f32, filter: FilterType) -> Option<((f32, f32), (f32, f32))> {
        // Portrait window, landscape sensor at 90° — a quarter-scale OnePlus 6T.
        const WW: u32 = 540;
        const WH: u32 = 1170;
        const SW: u32 = 648;
        const SH: u32 = 485;
        const BAR_TOP: f32 = 40.0;
        const BAR_BOTTOM: f32 = 60.0;

        let (_gpu_test, device, queue) = headless_device()?;
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut pipeline = VideoPipeline::new(&device, format);

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frosted scrim bar target"),
            size: wgpu::Extent3d {
                width: WW,
                height: WH,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        // A diagonal greyscale ramp: it survives any amount of blur with real
        // variance left, so "flat" can only mean "not the preview".
        let mut data = vec![0u8; (SW * SH * 4) as usize];
        for y in 0..SH {
            for x in 0..SW {
                let i = ((y * SW + x) * 4) as usize;
                let v = (((x * 255 / SW) + (y * 255 / SH)) / 2) as u8;
                data[i..i + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }

        let mut primitive = VideoPrimitive::new(VIDEO_ID_FROSTED);
        primitive.rotation = 1;
        primitive.filter_type = filter;
        // A 1:1 crop of the landscape sensor — what `photo_aspect_ratio` yields
        // for the ratio the crop bars exist to frame.
        primitive.crop_uv = Some((0.125, 0.0, 0.875, 1.0));
        // Nothing in the theme is this colour; the preview ramp is grey.
        primitive.letterbox_color = [1.0, 0.0, 1.0, 1.0];
        primitive.update_frame(VideoFrame {
            id: VIDEO_ID_FROSTED,
            width: SW,
            height: SH,
            data: crate::backends::camera::types::FrameData::Copied(data.into()),
            format: PixelFormat::RGBA,
            stride: SW * 4,
            yuv_planes: None,
        });
        // Exactly `FrostedScrim::draw`: the FULL window is the viewport size;
        // the bars are only ever the scissor.
        primitive.update_viewport(
            WW as f32,
            WH as f32,
            blend,
            crate::app::preview_geometry::BarInsets {
                top: BAR_TOP,
                bottom: BAR_BOTTOM,
                ..Default::default()
            },
        );

        let viewport = Viewport::with_physical_size(cosmic::iced::Size::new(WW, WH), 1.0);

        let bars = [
            Rectangle {
                x: 0.0,
                y: 0.0,
                width: WW as f32,
                height: BAR_TOP,
            },
            Rectangle {
                x: 0.0,
                y: WH as f32 - BAR_BOTTOM,
                width: WW as f32,
                height: BAR_BOTTOM,
            },
        ];

        // One cloned primitive per bar, each with its own `FrameViewportData` —
        // the scrim's real structure, so each bar's `prepare()` keeps its own
        // rect instead of the last one deciding for all four.
        let clones: Vec<VideoPrimitive> = bars.iter().map(|_| primitive.clone()).collect();
        for (clone, bar) in clones.iter().zip(bars.iter()) {
            clone.prepare(&mut pipeline, &device, &queue, bar, &viewport);
        }

        // The scrim is NEVER alone: the zoom/fit chips are `FrostedContainer`s,
        // they take `VIDEO_ID_FROSTED` too, and they sit in a layer above the
        // bars, so they prepare after them. They also report the full LAYER rect
        // as their preview geometry where the scrim reports its own layout
        // bounds — and that disagreement, left to reach `ensure_blur_targets`,
        // is what blanked the bars. Modelling the scrim on its own is what let
        // this test pass on a build the device rendered black; see
        // `frosted_consumers_that_disagree_do_not_blank_the_scrim`.
        let mut chip = VideoPrimitive::new(VIDEO_ID_FROSTED);
        chip.rotation = 1;
        chip.filter_type = filter;
        chip.crop_uv = Some((0.125, 0.0, 0.875, 1.0));
        chip.letterbox_color = [1.0, 0.0, 1.0, 1.0];
        // Its layer rect differs from the scrim's bounds by a couple of px, as
        // the phone's do. The exact sign does not matter — only that the two
        // consumers of one chain disagree, and that the disagreement survives
        // clamping to the render target and so reaches `ensure_blur_targets`.
        chip.update_viewport(
            WW as f32 - 2.0,
            WH as f32 - 2.0,
            blend,
            crate::app::preview_geometry::BarInsets {
                top: BAR_TOP,
                bottom: BAR_BOTTOM,
                ..Default::default()
            },
        );
        chip.prepare(
            &mut pipeline,
            &device,
            &queue,
            &Rectangle {
                x: 200.0,
                y: 500.0,
                width: 100.0,
                height: 40.0,
            },
            &viewport,
        );

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // Clear to RED (green = 0 too): a bar that draws nothing at all must not
        // be able to pass as a bar that drew letterbox.
        {
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frosted scrim bar clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        for (clone, bar) in clones.iter().zip(bars.iter()) {
            let clip = Rectangle {
                x: bar.x as u32,
                y: bar.y as u32,
                width: bar.width as u32,
                height: bar.height as u32,
            };
            clone.render(&pipeline, &mut encoder, &view, &clip);
        }

        let row = (WW * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frosted scrim bar readback"),
            size: (row * WH) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(WH),
                },
            },
            wgpu::Extent3d {
                width: WW,
                height: WH,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let pixels = readback.slice(..).get_mapped_range().to_vec();

        let stats = |y0: u32, y1: u32| -> (f32, f32) {
            let mut v = Vec::new();
            for y in y0..y1 {
                for x in 0..WW {
                    v.push(pixels[(y * row + x * 4 + 1) as usize] as f32);
                }
            }
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            let var = v.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / v.len() as f32;
            (mean, var.sqrt())
        };
        Some((stats(0, BAR_TOP as u32), stats(WH - BAR_BOTTOM as u32, WH)))
    }

    /// Every bar gets its OWN rounded silhouette — not the last one's.
    ///
    /// # The guarantee this pins, and why it is currently invisible
    ///
    /// `VideoPrimitive`'s `Clone` copies the `FrameViewportData` rather than
    /// sharing the `Arc`, so each of `FrostedScrim`'s four `draw_primitive` clones
    /// carries its own `panel_rect` and `corner_radius` — written by ITS `prepare()`
    /// from ITS bar rect. Sharing the `Arc` would let iced_wgpu's
    /// prepare-all-then-render-all ordering hand every bar the LAST bar's
    /// silhouette.
    ///
    /// Nothing observes that today: `FrostedScrim` passes `corner_radius: 0.0`, and
    /// at radius 0 the SDF degenerates to the full rect, so all four bars agree by
    /// accident and the deep `Clone` is inert. A NON-ZERO radius is what makes the
    /// guarantee observable — so this test supplies one. It is the regression test
    /// for the day the scrim (or any other multi-draw frosted widget) is given
    /// rounded bars, which is the moment the shared-`Arc` bug would have shipped.
    ///
    /// # How it separates the two
    ///
    /// Two bars, far apart, each `prepare`d with its own rect and a radius large
    /// enough to bite. The target is cleared RED and the frame is white, so
    /// `redness` reads coverage directly. If each bar keeps its own `panel_rect`,
    /// both bars' centres are covered and both bars' corners are cut. If they
    /// shared one, the TOP bar would be composited against the BOTTOM bar's rect —
    /// a rect it lies entirely outside — so its SDF would be positive everywhere,
    /// alpha 0, and the red clear would survive across the whole top bar.
    #[test]
    fn each_frosted_bar_keeps_its_own_silhouette() {
        const WW: u32 = 256;
        const WH: u32 = 256;
        const BAR_H: u32 = 64;
        const RADIUS: f32 = 24.0;

        let Some((_gpu_test, device, queue)) = headless_device() else {
            skip_no_gpu("each_frosted_bar_keeps_its_own_silhouette");
            return;
        };
        let format = SURFACE_FORMAT;
        let mut pipeline = VideoPipeline::new(&device, format);

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("per-bar silhouette target"),
            size: wgpu::Extent3d {
                width: WW,
                height: WH,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        // All-white: a blur of white is white, so anything not white is the SDF.
        let mut primitive = VideoPrimitive::new(VIDEO_ID_FROSTED);
        primitive.corner_radius = RADIUS;
        primitive.update_frame(VideoFrame {
            id: VIDEO_ID_FROSTED,
            width: WW,
            height: WH,
            data: crate::backends::camera::types::FrameData::Copied(
                vec![255u8; (WW * WH * 4) as usize].into(),
            ),
            format: PixelFormat::RGBA,
            stride: WW * 4,
            yuv_planes: None,
        });
        primitive.update_viewport(
            WW as f32,
            WH as f32,
            1.0,
            crate::app::preview_geometry::BarInsets::default(),
        );

        // Top and bottom bars, as `scrim_bars` emits them.
        let bars = [
            Rectangle {
                x: 0.0,
                y: 0.0,
                width: WW as f32,
                height: BAR_H as f32,
            },
            Rectangle {
                x: 0.0,
                y: (WH - BAR_H) as f32,
                width: WW as f32,
                height: BAR_H as f32,
            },
        ];

        let viewport = Viewport::with_physical_size(cosmic::iced::Size::new(WW, WH), 1.0);
        let clones: Vec<VideoPrimitive> = bars.iter().map(|_| primitive.clone()).collect();
        for (clone, bar) in clones.iter().zip(bars.iter()) {
            clone.prepare(&mut pipeline, &device, &queue, bar, &viewport);
        }

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("per-bar silhouette clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        for (clone, bar) in clones.iter().zip(bars.iter()) {
            clone.render(
                &pipeline,
                &mut encoder,
                &view,
                &Rectangle {
                    x: bar.x as u32,
                    y: bar.y as u32,
                    width: bar.width as u32,
                    height: bar.height as u32,
                },
            );
        }

        let row = (WW * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (row * WH) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(WH),
                },
            },
            wgpu::Extent3d {
                width: WW,
                height: WH,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let pixels = readback.slice(..).get_mapped_range().to_vec();
        let at = |x: u32, y: u32| -> [u8; 4] {
            let i = (y * row + x * 4) as usize;
            [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
        };

        for (name, bar) in [("top", bars[0]), ("bottom", bars[1])] {
            // The bar's own centre is covered by its own silhouette.
            let (cx, cy) = (
                (bar.x + bar.width / 2.0) as u32,
                (bar.y + bar.height / 2.0) as u32,
            );
            let centre = at(cx, cy);
            assert!(
                redness(centre) < 40,
                "the {name} bar's centre ({cx},{cy}) must be covered by the {name} \
                 bar's OWN rounded rect, got {centre:?} — a red centre means this \
                 bar was composited against another bar's panel_rect, i.e. the \
                 clones are sharing one FrameViewportData again"
            );

            // And its own corner is cut by the radius, at its own rect.
            let corner = at(bar.x as u32, bar.y as u32);
            assert!(
                redness(corner) > 80,
                "the {name} bar's corner must be rounded away by its own \
                 corner_radius, got {corner:?} — the per-bar radius is not being \
                 applied"
            );
        }
    }

    /// In FILL mode the scrim's bars must show BLURRED PREVIEW, never a flat
    /// fill.
    ///
    /// In Fill the preview covers the whole window, so every bar — the top/bottom
    /// chrome bars and the aspect crop bars alike — sits over real preview
    /// content. A bar that comes out uniform is the bug this pins: it means the
    /// bar sampled `letterbox_color` (or nothing) where live preview was.
    ///
    /// # This test has teeth
    ///
    /// It FAILS on the dual-Kawase rewrite as shipped (94c5281..70d74a0), which
    /// the device rendered with black bars: green mean 0.9, stddev 1.26 — an
    /// orphaned, never-rendered-into blur texture, plus film grain. It only has
    /// those teeth because `scrim_bar_green` puts a second `VIDEO_ID_FROSTED`
    /// consumer on screen alongside the scrim, as the app always does. Without
    /// one it passed happily on that build, which is how the regression shipped;
    /// see `frosted_consumers_that_disagree_do_not_blank_the_scrim`.
    ///
    /// It ALSO fails on c2dcf0c, the commit before the dual-Kawase rewrite: at
    /// rotation 1 with an aspect crop — i.e. the phone's actual configuration —
    /// the old chain returned a flat bar (green mean 1.8, stddev 0.8) because its
    /// pass 1 applied the crop with `content_fit_mode` HARDCODED to 0 while the
    /// screen fit ran later at the real blend, so the crop got letterboxed into
    /// the intermediate and the bars sampled that letterbox. Measured on HEAD:
    /// mean 186.0, stddev 22.2. So the rewrite FIXED this case, and the assert
    /// below separates the two by two orders of magnitude on the stddev.
    #[test]
    fn frosted_scrim_bars_are_not_flat_in_fill() {
        let Some((top, bottom)) = scrim_bar_green(1.0, FilterType::Standard) else {
            skip_no_gpu("frosted_scrim_bars_are_not_flat_in_fill");
            return;
        };

        for (name, (mean, std)) in [("top", top), ("bottom", bottom)] {
            // Green survived: this is the grey preview ramp, not the magenta
            // letterbox and not the red clear.
            assert!(
                mean > 20.0,
                "the {name} scrim bar must show blurred preview in Fill mode, but its \
                 green mean is {mean:.1} — it sampled letterbox_color or nothing, \
                 not the preview"
            );
            // And it is IMAGE, not a flat fill: a ramp blurs to a ramp.
            assert!(
                std > 5.0,
                "the {name} scrim bar must carry real blurred image content in Fill \
                 mode, but it is uniform (green stddev {std:.2}) — a flat bar means \
                 the blur chain painted a constant where live preview should be"
            );
        }
    }

    /// The frosted backdrop blurs the FILTERED image — the one the sharp preview
    /// shows.
    ///
    /// The bug this pins reached a device: with Sketch selected the preview was a
    /// black-and-white pencil drawing while the bottom scrim bar's blur was the
    /// raw colour scene, because `make_primitive` left `filter_type` at Standard
    /// and pass 0 of the blur chain gated the filter to modes <= 12. Either half
    /// alone is enough to reproduce it, so this drives the whole path — the real
    /// `prepare`/`render`, the real chain, a readback — and checks the blurred
    /// output actually MOVES when a filter is on.
    ///
    /// # This test has teeth
    ///
    /// Measured (green mean, stddev) inside the top bar in Fill, over the
    /// greyscale ramp:
    ///
    /// * Standard: 186.0, 22.2
    /// * Negative: 68.1, 22.1 — the ramp inverted. Blur is linear, so
    ///   `blur(255-x) = 255-blur(x)`: the two means sum to 254.1 of a possible
    ///   255, which no unfiltered backdrop can fake.
    /// * Pencil: 243.9, 2.3 — the ramp is smooth, so Sobel finds no edges and the
    ///   sketch is blank paper. The stddev collapses to 2.3, which is the frosted
    ///   grain's own 2.2 and nothing else.
    ///
    /// This drives the primitive directly, so it pins the SHADER half; the
    /// `make_primitive` half is `make_primitive_copies_every_preview_transform`,
    /// and either one alone reproduces the device bug. Restoring the `<= 12` gate
    /// puts Pencil back at 186.0 / 22.2 — the unfiltered ramp, measured — and
    /// fails both Pencil asserts. Negative does NOT fail there: mode 10 passed
    /// that gate. It stays because it is the cheap check that modes 0-12 still
    /// reach pass 0 at all, where Pencil is the only case covering 13 and 14 —
    /// the two that re-sample, and so the two a prelude change can break.
    #[test]
    fn frosted_scrim_bars_blur_the_filtered_image() {
        let Some(((plain, plain_std), _)) = scrim_bar_green(1.0, FilterType::Standard) else {
            skip_no_gpu("frosted_scrim_bars_blur_the_filtered_image");
            return;
        };
        let ((neg, _), _) = scrim_bar_green(1.0, FilterType::Negative).expect("adapter vanished");
        let ((pencil, pencil_std), _) =
            scrim_bar_green(1.0, FilterType::Pencil).expect("adapter vanished");

        // Negative inverts the ramp, and the blur that follows is linear.
        let sum = plain + neg;
        assert!(
            (sum - 255.0).abs() < 6.0,
            "the backdrop must blur the NEGATIVE of the ramp: blur is linear, so a \
             plain bar ({plain:.1}) and an inverted one ({neg:.1}) must sum to 255, \
             got {sum:.1} — a backdrop still blurring the unfiltered frame reads \
             {plain:.1} for both and sums to {:.1}",
            plain * 2.0
        );

        // Pencil finds no edges in a smooth ramp: blank paper, grain only.
        assert!(
            pencil > 230.0,
            "the backdrop must blur the PENCIL SKETCH of the ramp — a smooth ramp \
             has no Sobel edges, so it sketches to near-white paper — got green mean \
             {pencil:.1} against the unfiltered {plain:.1}"
        );
        assert!(
            pencil_std < 5.0,
            "the pencil backdrop must carry the frosted grain and nothing else \
             (stddev ~2.2), got {pencil_std:.2} against the unfiltered ramp's \
             {plain_std:.2} — a bar that still varies like the ramp is a bar that \
             never ran the filter"
        );
    }

    /// In FIT mode the scrim's bars are letterbox, and that is CORRECT — do not
    /// "fix" it.
    ///
    /// This pins the geometry so the intent survives. In Fit the preview is
    /// contained into the window MINUS the bars (`content_height =
    /// viewport_size.y - bar_top - bar_bottom`, see `video_shader_blur.wgsl`), so
    /// the bar regions are *by construction* exactly where the preview is not.
    /// There is no preview there to blur, and the window background is the honest
    /// answer — which is what the sharp preview underneath leaves for it too.
    ///
    /// What the bars are FILLED with is a separate question, and its answer has
    /// changed: the chain still fills the letterbox with `letterbox_color` (the
    /// Kawase kernels need alpha as a region mask), but the composite now cuts it
    /// back out so the one window-background layer beneath — translucent under
    /// COSMIC's window frosting — shows through. See
    /// `the_composite_cuts_the_letterbox_where_the_fit_math_says`, which measures
    /// that on the blue channel. This test speaks only to the GEOMETRY, and the
    /// green channel it reads is 0 either way.
    ///
    /// So the panels and the scrim need the SAME fit, not different ones. Giving
    /// the scrim its own `video_id` and a forced Cover fit would fill the bars
    /// with a cover-cropped image that appears NOWHERE else on screen, breaking
    /// the module's whole premise — that the blurred slice lines up with the
    /// sharp preview beside it — and costing a third chain for the privilege.
    ///
    /// This has been the behaviour since the first frosted commit (0b5d872): the
    /// backdrop has always been fed the preview's live `cover_blend` and bars.
    /// It is NOT a regression of the dual-Kawase rewrite, which leaves it
    /// unchanged (c2dcf0c: green mean 0.0; HEAD: 0.9, the difference being the
    /// grain and a little real bleed the new chain picks up at the frame edge).
    ///
    /// Note this direction is one-sided by nature and cannot police the blur: a
    /// blanked bar reads green ~0 too, so "letterbox" and "the chain handed us an
    /// orphaned texture" are indistinguishable here. That is what
    /// `frosted_scrim_bars_are_not_flat_in_fill` is for; do not lean on this one
    /// to catch a broken chain.
    #[test]
    fn frosted_scrim_bars_are_letterbox_in_fit() {
        let Some((top, bottom)) = scrim_bar_green(0.0, FilterType::Standard) else {
            skip_no_gpu("frosted_scrim_bars_are_letterbox_in_fit");
            return;
        };

        for (name, (mean, _std)) in [("top", top), ("bottom", bottom)] {
            assert!(
                mean < 10.0,
                "the {name} scrim bar should be letterbox in Fit mode (green mean \
                 {mean:.1}, expected ~0): in Fit the preview is contained BETWEEN \
                 the bars, so there is no preview under them to blur. If this now \
                 shows image content, the blur has stopped tracking the preview's \
                 fit and no longer lines up with it."
            );
        }
    }

    /// Render a full-window frosted backdrop at `blend` over a RED clear and
    /// return the target's rows.
    ///
    /// The separation is by CHANNEL, and it is what makes the assertions below
    /// unambiguous:
    ///
    /// * clear = pure RED — nothing painted here at all;
    /// * `letterbox_color` = pure MAGENTA — the chain's letterbox fill reached the
    ///   screen (blue is the tell, and only the letterbox has any);
    /// * the frame = a mid-grey RAMP — real blurred preview (green is the tell,
    ///   and only the preview has any).
    ///
    /// The window is square and the sensor 2:1, so at Fit the image contains to
    /// the middle half of the window with a quarter of letterbox above and below —
    /// large, flat regions that a boundary off by a pixel cannot fake.
    fn frosted_letterbox_rows(blend: f32) -> Option<(Vec<[f32; 3]>, u32, [f32; 4])> {
        const N: u32 = 256;
        const SW: u32 = 256;
        const SH: u32 = 128;

        let (_gpu_test, device, queue) = headless_device()?;
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut pipeline = VideoPipeline::new(&device, format);

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frosted letterbox target"),
            size: wgpu::Extent3d {
                width: N,
                height: N,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        // A grey ramp well clear of 0 on every channel, so "the preview painted
        // here" is unmistakable even after the blur has averaged it.
        let mut data = vec![0u8; (SW * SH * 4) as usize];
        for y in 0..SH {
            for x in 0..SW {
                let i = ((y * SW + x) * 4) as usize;
                let v = 80 + ((x * 120 / SW) + (y * 120 / SH)) as u8 / 2;
                data[i..i + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }

        let mut primitive = VideoPrimitive::new(VIDEO_ID_FROSTED);
        primitive.letterbox_color = [1.0, 0.0, 1.0, 1.0];
        primitive.update_frame(VideoFrame {
            id: VIDEO_ID_FROSTED,
            width: SW,
            height: SH,
            data: crate::backends::camera::types::FrameData::Copied(data.into()),
            format: PixelFormat::RGBA,
            stride: SW * 4,
            yuv_planes: None,
        });
        primitive.update_viewport(
            N as f32,
            N as f32,
            blend,
            crate::app::preview_geometry::BarInsets::default(),
        );

        let viewport = Viewport::with_physical_size(cosmic::iced::Size::new(N, N), 1.0);
        let full = Rectangle {
            x: 0.0,
            y: 0.0,
            width: N as f32,
            height: N as f32,
        };
        primitive.prepare(&mut pipeline, &device, &queue, &full, &viewport);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("frosted letterbox clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        primitive.render(
            &pipeline,
            &mut encoder,
            &view,
            &Rectangle {
                x: 0,
                y: 0,
                width: N,
                height: N,
            },
        );

        let row_bytes = (N * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frosted letterbox readback"),
            size: (row_bytes * N) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: Some(N),
                },
            },
            wgpu::Extent3d {
                width: N,
                height: N,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let pixels = readback.slice(..).get_mapped_range().to_vec();

        // Row means, so a single stray texel cannot decide anything.
        let rows: Vec<[f32; 3]> = (0..N)
            .map(|y| {
                let mut acc = [0f32; 3];
                for x in 0..N {
                    let i = (y * row_bytes + x * 4) as usize;
                    for (c, a) in acc.iter_mut().enumerate() {
                        *a += pixels[i + c] as f32;
                    }
                }
                acc.map(|a| a / N as f32)
            })
            .collect();

        let rect = content_rect_uv(
            [N as f32, N as f32],
            (SW, SH),
            0,
            [0.0, 0.0],
            [1.0, 1.0],
            blend,
            0.0,
            0.0,
        );
        Some((rows, N, rect))
    }

    /// THE test for the letterbox cut: in Fit the composite must paint NOTHING
    /// outside the image, and `content_rect_uv` must say where that is.
    ///
    /// # What it separates
    ///
    /// The chain's own letterbox fill is magenta and nothing else on screen has
    /// any blue, so blue in the letterbox rows means the fill was composited —
    /// which is precisely the bug: over a translucent window background that fill
    /// stacks a second opaque copy of the background colour and the compositor's
    /// frosted glass disappears exactly where there is no camera image to justify
    /// it (issue #569). Red survives instead, untouched, which is what lets the
    /// one window-background layer beneath show through.
    ///
    /// # Why it also pins `content_rect_uv` itself
    ///
    /// That function duplicates the fit math of `video_shader_blur.wgsl` in Rust,
    /// and duplicated geometry drifts. So this does not merely check "something
    /// was cut" — it checks the cut lands where the SHADER's own letterbox
    /// early-out puts it, by asserting the painted band is exactly the rows
    /// `content_rect_uv` claims and the rows either side are untouched. A drift in
    /// either copy moves one and not the other.
    #[test]
    fn the_composite_cuts_the_letterbox_where_the_fit_math_says() {
        let Some((rows, n, rect)) = frosted_letterbox_rows(0.0) else {
            skip_no_gpu("the_composite_cuts_the_letterbox_where_the_fit_math_says");
            return;
        };

        // A 2:1 sensor contained into a square window: the middle half.
        assert!(
            (rect[1] - 0.25).abs() < 0.01 && (rect[3] - 0.75).abs() < 0.01,
            "a 2:1 image contained in a square window occupies its middle half; \
             content_rect_uv said {rect:?}"
        );
        let y0 = (rect[1] * n as f32) as u32;
        let y1 = (rect[3] * n as f32) as u32;

        // Outside the image: the red clear, untouched. One row of slack either
        // side for the antialiased edge.
        for y in (0..y0.saturating_sub(1)).chain(y1 + 1..n) {
            let [r, g, b] = rows[y as usize];
            assert!(
                b < 4.0 && g < 4.0 && r > 250.0,
                "row {y} is letterbox and must be left untouched (the red clear), \
                 got rgb ({r:.1}, {g:.1}, {b:.1}). Blue means the chain's magenta \
                 letterbox fill was composited — the very fill that hides the \
                 window's frosted glass."
            );
        }

        // Inside it: real blurred preview. Stay clear of the edge, where the
        // blur legitimately mixes the letterbox fill inward.
        for y in y0 + 12..y1 - 12 {
            let [_, g, _] = rows[y as usize];
            assert!(
                g > 60.0,
                "row {y} is inside the image and must carry blurred preview, got \
                 green {g:.1} — the cut has eaten live preview"
            );
        }
    }

    /// ...and in Fill the cut must do nothing at all: Cover covers the window, so
    /// every row is image and every row must still be painted.
    ///
    /// The failure this guards against is the cut going the other way — a
    /// `content_rect` that falls short of the window would carve blurred preview
    /// out of the scrim bars and chips in the mode the app spends most of its
    /// time in.
    #[test]
    fn the_composite_cuts_nothing_in_fill() {
        let Some((rows, n, rect)) = frosted_letterbox_rows(1.0) else {
            skip_no_gpu("the_composite_cuts_nothing_in_fill");
            return;
        };
        assert!(
            rect[1] <= 0.0 && rect[3] >= 1.0,
            "Cover fills the window, so there is no letterbox to cut: {rect:?}"
        );
        for y in 0..n {
            let [_, g, _] = rows[y as usize];
            assert!(
                g > 60.0,
                "row {y} must carry blurred preview in Fill, got green {g:.1}"
            );
        }
    }

    /// With radius 0 (the crop-bar scrim) the backdrop must stay a full square:
    /// no SDF, no accidental rounding, corners fully covered.
    #[test]
    fn frosted_zero_radius_stays_square() {
        let Some(px) = render_frosted_corner(0.0) else {
            skip_no_gpu("frosted_zero_radius_stays_square");
            return;
        };
        for (x, y) in [(0usize, 0usize), (63, 0), (0, 63), (63, 63), (32, 32)] {
            let p = px[y * 64 + x];
            assert!(
                redness(p) < 40,
                "({x},{y}) should be fully covered when radius is 0, got {p:?}"
            );
        }
    }

    /// Render the phone's ACTUAL scrim geometry — the numbers captured from the
    /// live bug on device — and report the green mean/stddev inside the top and
    /// bottom bars.
    ///
    /// Every number here is measured, not modelled: a 948x586 logical window at
    /// scale_factor 1.5 (render target 1422x879), a 1280x960 frame at rotation
    /// 0, Fill (`cover_blend` = 1.0), `photo_aspect_ratio` Native so there is NO
    /// crop, bars of 47 and 174 logical px, and the two bar rects the scrim
    /// actually emitted — (1,1,946,47) and (1,411,946,174), scissored to
    /// (2,2,1419,70) and (2,617,1419,261).
    ///
    /// `with_container` decides whether a `FrostedContainer` (the zoom/fit
    /// chips, which are on screen whenever the bars are) shares the frame. It
    /// takes `VIDEO_ID_FROSTED` too, and — this is the whole point — reports the
    /// full LAYER rect (0,0,948,586) as its preview geometry where the scrim
    /// reports its own layout bounds (1,1,946,584), which its parent inset by a
    /// pixel. It prepares AFTER the scrim, because the chips sit in a layer
    /// above the bars.
    ///
    /// The frame is a greyscale ramp and `letterbox_color` is magenta, so green
    /// separates preview from letterbox; the target is cleared RED so "drew
    /// nothing" cannot pass as "drew letterbox".
    fn device_scrim_bar_green(level: u8, with_container: bool) -> Option<((f32, f32), (f32, f32))> {
        const RT_W: u32 = 1422;
        const RT_H: u32 = 879;
        const SW: u32 = 1280;
        const SH: u32 = 960;

        let (_gpu_test, device, queue) = headless_device()?;
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut pipeline = VideoPipeline::new(&device, format);
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("device scrim bar target"),
            size: wgpu::Extent3d {
                width: RT_W,
                height: RT_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        // A diagonal ramp: whatever the blur does to it, it keeps real variance,
        // so "flat" can only mean "not the preview".
        let mut data = vec![0u8; (SW * SH * 4) as usize];
        for y in 0..SH {
            for x in 0..SW {
                let i = ((y * SW + x) * 4) as usize;
                let v = (((x * 255 / SW) + (y * 255 / SH)) / 2) as u8;
                data[i..i + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }

        let mut primitive = VideoPrimitive::new(VIDEO_ID_FROSTED);
        primitive.rotation = 0;
        primitive.crop_uv = None;
        primitive.mirror_horizontal = true;
        primitive.blur_params = compositor_blur_params(level);
        primitive.letterbox_color = [1.0, 0.0, 1.0, 1.0];
        primitive.update_frame(VideoFrame {
            id: VIDEO_ID_FROSTED,
            width: SW,
            height: SH,
            data: crate::backends::camera::types::FrameData::Copied(data.into()),
            format: PixelFormat::RGBA,
            stride: SW * 4,
            yuv_planes: None,
        });
        // Exactly `FrostedScrim::draw`: its layout bounds are both the reported
        // preview geometry and the fit's viewport size.
        primitive.update_viewport(
            946.0,
            584.0,
            1.0,
            crate::app::preview_geometry::BarInsets {
                top: 47.0,
                bottom: 174.0,
                ..Default::default()
            },
        );

        let viewport = Viewport::with_physical_size(cosmic::iced::Size::new(RT_W, RT_H), 1.5);
        let bars = [
            Rectangle {
                x: 1.0,
                y: 1.0,
                width: 946.0,
                height: 47.0,
            },
            Rectangle {
                x: 1.0,
                y: 411.0,
                width: 946.0,
                height: 174.0,
            },
        ];

        // One cloned primitive per bar, each with its own `FrameViewportData` and
        // its own copy of the frame — the scrim's real structure.
        let clones: Vec<VideoPrimitive> = bars.iter().map(|_| primitive.clone()).collect();
        for (clone, bar) in clones.iter().zip(bars.iter()) {
            clone.prepare(&mut pipeline, &device, &queue, bar, &viewport);
        }

        // `FrostedContainer::draw`: the full layer rect, prepared after the bars.
        let chip = if with_container {
            let mut chip = VideoPrimitive::new(VIDEO_ID_FROSTED);
            chip.blur_params = compositor_blur_params(level);
            chip.letterbox_color = [1.0, 0.0, 1.0, 1.0];
            chip.update_viewport(
                948.0,
                586.0,
                1.0,
                crate::app::preview_geometry::BarInsets {
                    top: 47.0,
                    bottom: 174.0,
                    ..Default::default()
                },
            );
            let rect = Rectangle {
                x: 400.0,
                y: 300.0,
                width: 100.0,
                height: 40.0,
            };
            chip.prepare(&mut pipeline, &device, &queue, &rect, &viewport);
            Some((chip, rect))
        } else {
            None
        };

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("device scrim bar clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        let clips = [
            Rectangle {
                x: 2u32,
                y: 2,
                width: 1419,
                height: 70,
            },
            Rectangle {
                x: 2u32,
                y: 617,
                width: 1419,
                height: 261,
            },
        ];
        for (clone, clip) in clones.iter().zip(clips.iter()) {
            clone.render(&pipeline, &mut encoder, &view, clip);
        }
        if let Some((chip, rect)) = &chip {
            let clip = Rectangle {
                x: (rect.x * 1.5) as u32,
                y: (rect.y * 1.5) as u32,
                width: (rect.width * 1.5) as u32,
                height: (rect.height * 1.5) as u32,
            };
            chip.render(&pipeline, &mut encoder, &view, &clip);
        }

        let row = (RT_W * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("device scrim bar readback"),
            size: (row * RT_H) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(RT_H),
                },
            },
            wgpu::Extent3d {
                width: RT_W,
                height: RT_H,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let pixels = readback.slice(..).get_mapped_range().to_vec();

        let stats = |y0: u32, y1: u32| -> (f32, f32) {
            let mut v = Vec::new();
            for y in y0..y1 {
                for x in 0..RT_W {
                    v.push(pixels[(y * row + x * 4 + 1) as usize] as f32);
                }
            }
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            let var = v.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / v.len() as f32;
            (mean, var.sqrt())
        };
        Some((stats(2, 72), stats(617, 878)))
    }

    /// A second `VIDEO_ID_FROSTED` consumer must not blank the scrim's bars.
    ///
    /// # The bug this pins
    ///
    /// `FrostedContainer` and `FrostedScrim` share ONE blur chain, keyed by
    /// `VIDEO_ID_FROSTED`, and `BlurTargets` documents the invariant that makes
    /// that sound: "each reports the FULL preview viewport rather than its own
    /// panel rect — so pass 0 and the Kawase are identical for all of them".
    ///
    /// They did not. The container reports its layer rect (the whole window,
    /// 948x586); the scrim reports its own layout bounds (946x584), inset a
    /// pixel by its parent. Those round to different ping-pong texture sizes
    /// (1422x879 vs 1419x876), so whichever consumer prepared LAST re-sized the
    /// targets — and every consumer that had already prepared was left holding a
    /// composite bind group over an orphaned texture the chain never rendered
    /// into. A fresh wgpu texture reads as zeroes, and the composite takes only
    /// `.rgb` and writes alpha = 1, so the bar came out OPAQUE BLACK plus film
    /// grain. On the phone the chips prepare after the bars, so the chips blurred
    /// and the bars went black — exactly the asymmetry the bug report describes.
    ///
    /// Before the dual-Kawase rewrite (94c5281) the intermediates were a fixed
    /// quarter-res size, so the two consumers' disagreement about the preview
    /// rect never reached `ensure_blur_targets` and was harmless. The rewrite
    /// made the targets follow the reported rect at screen resolution, which is
    /// what turned a latent disagreement into a black scrim. That is why the
    /// device bisect lands on it.
    ///
    /// # Why this asserts equality rather than a threshold
    ///
    /// The real invariant is not "the bars are bright", it is "what the scrim
    /// shows does not depend on who else is on screen". So it renders the bars
    /// with and without a coexisting container and demands the same image. That
    /// catches the blanking (78 -> 0.9) and would also catch a subtler
    /// draw-order-dependent fit, which a brightness threshold would sail past.
    #[test]
    fn frosted_consumers_that_disagree_do_not_blank_the_scrim() {
        let Some((alone_top, alone_bot)) = device_scrim_bar_green(6, false) else {
            skip_no_gpu("frosted_consumers_that_disagree_do_not_blank_the_scrim");
            return;
        };
        let (with_top, with_bot) = device_scrim_bar_green(6, true).expect("adapter vanished");

        // The bars carry live preview at all: the ramp's blurred green, not the
        // magenta letterbox (green 0), not the red clear (green 0), and not the
        // black-plus-grain an orphaned texture composites to (mean ~0.9, stddev
        // ~1.3 — the film grain alone).
        for (name, (mean, std)) in [("top", alone_top), ("bottom", alone_bot)] {
            assert!(
                mean > 20.0 && std > 5.0,
                "the {name} bar must show blurred preview even with no other \
                 frosted consumer on screen (green mean {mean:.1}, stddev {std:.2})"
            );
        }

        // And a coexisting FrostedContainer changes NOTHING about them.
        for (name, (alone, with)) in [
            ("top", (alone_top, with_top)),
            ("bottom", (alone_bot, with_bot)),
        ] {
            assert!(
                (alone.0 - with.0).abs() < 1.0 && (alone.1 - with.1).abs() < 1.0,
                "a FrostedContainer sharing VIDEO_ID_FROSTED must not change what \
                 the {name} scrim bar shows, but the bar went from green \
                 mean {:.1}/stddev {:.2} alone to mean {:.1}/stddev {:.2} \
                 alongside it. The two consumers report different full-preview \
                 rects, so one of them re-sized the shared blur targets and \
                 orphaned the other's composite binding — see this test's docs.",
                alone.0,
                alone.1,
                with.0,
                with.1
            );
        }
    }
}
