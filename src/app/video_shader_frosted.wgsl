// SPDX-License-Identifier: GPL-3.0-only
// Portions Copyright (C) System76, Inc. — grain derived from cosmic-comp
// (GPL-3.0-only), which took it from niri.
//
// Final per-panel composite for the blur chain.
//
// The Kawase ping-pong (see `video_shader_kawase.wgsl`) leaves a fully blurred
// copy of the preview, at PHYSICAL SCREEN resolution and in screen space, in one
// texture. This pass blits a slice of it onto the target for ONE panel / scrim
// bar: a single sample, plus the rounded-corner SDF, the dim and the grain.
//
// It is deliberately the only part of the chain that runs per panel and at
// screen resolution, so it is deliberately the cheapest: one tap. Everything
// expensive happens once per frame per `video_id`, upstream of here.
//
// # Why the mapping is identity
//
// Under the old architecture this pass had to reproduce the whole cover/contain
// fit, because the intermediates lived in frame space. They no longer do: pass 0
// (`video_shader_blur.wgsl`) renders the transformed preview straight into a
// screen-resolution target, and the Kawase passes never leave that space. So the
// blur texture already IS the screen, and `tex_coords` — 0..1 over the render
// pass viewport, which is set to the same preview rect the texture was sized
// from — maps onto it 1:1. No fit, no letterbox, no crop, no zoom here.

@group(0) @binding(0)
var texture_blur: texture_2d<f32>;

@group(0) @binding(1)
var sampler_blur: sampler;

struct ViewportUniform {
    viewport_size: vec2<f32>,
    content_fit_mode: f32,
    filter_mode: u32,
    // Panel corner radius in PHYSICAL px (0 = square).
    corner_radius: f32,
    mirror_horizontal: u32,
    uv_offset: vec2<f32>,
    uv_scale: vec2<f32>,
    crop_uv_min: vec2<f32>,
    crop_uv_max: vec2<f32>,
    zoom_level: f32,
    rotation: u32,
    bar_top_height: f32,
    bar_bottom_height: f32,
    kawase_offset: f32,
    // RGB multiplier. The transition blur darkens; the frosted chrome does not.
    dim_factor: f32,
    letterbox_color: vec4<f32>,
    // The panel rect (x, y, w, h) in PHYSICAL px of the render target, i.e. the
    // same space as `@builtin(position)`.
    panel_rect: vec4<f32>,
    // Film-grain amplitude. cosmic-comp uses 0.03 on its frosted surfaces; the
    // transition blur passes 0 (see the note in `fs_main`).
    noise: f32,
    // The rect the preview IMAGE occupies — (min_x, min_y, max_x, max_y),
    // normalized over this pass's own `tex_coords` space. Its complement is the
    // letterbox. (0,0,1,1) means "no letterbox", and is skipped.
    //
    // A vec4 aligns to 16, so WGSL lands this at offset 128 on its own — exactly
    // where the Rust struct's `_pad: [f32; 3]` leaves it. No explicit padding
    // member here, because a `vec3<f32>` pad would ALSO align to 16 and push
    // itself to 128, taking `content_rect` to 144.
    content_rect: vec4<f32>,
    bar_left_width: f32,
    bar_right_width: f32,
}

@group(0) @binding(2)
var<uniform> viewport: ViewportUniform;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    let x = f32((vertex_index & 1u) << 2u) - 1.0;
    let y = f32((vertex_index & 2u) << 1u) - 1.0;

    out.position = vec4<f32>(x, -y, 0.0, 1.0);
    out.tex_coords = vec2<f32>((x + 1.0) * 0.5, (y + 1.0) * 0.5);

    return out;
}

// Verbatim from cosmic-comp `shaders/clipped_surface.frag:55-59` (which took it
// from niri), name included, so our grain is theirs and not a lookalike.
//
// Keeping upstream's name costs this module the filters prelude, which defines an
// unrelated `hash()`: concatenating both redefines the symbol and the pipeline
// fails to compile. This module takes GEOMETRY_FUNCTIONS only.
fn hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 727.727);
    p3 += vec3<f32>(dot(p3, p3 + 33.33));
    return fract((p3.x + p3.y) * p3.z);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var rgb_val = textureSample(texture_blur, sampler_blur, in.tex_coords).rgb;

    // Dim the result. The transition blur passes < 1.0 here as a subtle "the
    // camera is switching" cue; the frosted backdrop passes 1.0, since its
    // translucency comes from the panel tint on top rather than from dimming.
    //
    // Under the old 3-pass ring blur this was applied ONCE PER PASS and
    // compounded to 0.85³. There is no longer a fixed pass count to compound
    // over (the Kawase runs 1..4 passes, twice each, depending on the frost
    // level), so the dim now lives here and here only — see
    // `TRANSITION_BLUR_DIM`, which carries the same total.
    rgb_val = rgb_val * viewport.dim_factor;

    // Film grain, ported from cosmic-comp `clipped_surface.frag:87-92`. It is
    // part of how their glass looks — not an afterthought — and it dithers the
    // very quantisation steps that made our old kernel band. Applied at the
    // final composite, which is where they apply it (their frosted surface
    // shader, downstream of their blur).
    if (viewport.noise > 0.0) {
        let noise_hash = hash(in.tex_coords);
        let noise_amount = fract(noise_hash) - 0.5;
        rgb_val += vec3<f32>(noise_amount * viewport.noise);
    }

    // Round the frosted backdrop's corners here rather than by scissoring the
    // blur to a rounded-rect strip list: a scissor is integer-pixel binary
    // coverage, so it can only ever produce a staircase. The SDF gives a true
    // antialiased edge that matches the panel tint drawn on top (iced rounds its
    // quads the same way), and it costs one draw instead of ~2*radius blits.
    //
    // `panel_rect` and `corner_radius` are in physical px, the same space as
    // `@builtin(position)` — which is independent of the viewport override the
    // frosted backdrop uses to span the full preview.
    var alpha = 1.0;
    if (viewport.corner_radius > 0.0 && viewport.panel_rect.z > 0.0) {
        let half_size = viewport.panel_rect.zw * 0.5;
        let center = viewport.panel_rect.xy + half_size;
        let dist = rounded_box_sdf(in.position.xy - center, half_size, viewport.corner_radius);
        alpha = 1.0 - smoothstep(-1.0, 1.0, dist);
    }

    // Cut the LETTERBOX back out.
    //
    // Pass 0 fills it with an opaque `letterbox_color`, and has to: the Kawase
    // kernels normalize by `sum.a` and read alpha as a region mask, so a
    // transparent letterbox would bleed the image outward past its own edge
    // instead of blending it toward the background. That fill belongs to the
    // chain, not to the screen — composited here it would stack a second opaque
    // copy of the window background over the one translucent layer that is meant
    // to show the compositor's frosted glass through. So the chain keeps it and
    // this pass drops it, leaving blurred preview where there IS preview and the
    // window background where there is not.
    //
    // `content_rect` is normalized over this pass's `tex_coords`, which span the
    // full preview rect — the same rect pass 0 rendered into, so the boundary
    // lands exactly where the sharp preview's own letterbox early-out puts it.
    // (0,0,1,1) is "the image covers everything" (Fill, and the default every
    // other pass carries), and skips the work entirely.
    if (viewport.content_rect.x > 0.0 || viewport.content_rect.y > 0.0
        || viewport.content_rect.z < 1.0 || viewport.content_rect.w < 1.0) {
        // In px, so the antialiased edge is half a pixel wide rather than half
        // the screen. `viewport_size` is the preview rect this pass spans.
        let half_size = (viewport.content_rect.zw - viewport.content_rect.xy)
            * 0.5 * viewport.viewport_size;
        let center = (viewport.content_rect.xy + viewport.content_rect.zw)
            * 0.5 * viewport.viewport_size;
        let dist = rounded_box_sdf(
            in.tex_coords * viewport.viewport_size - center,
            half_size,
            0.0,
        );
        alpha = alpha * (1.0 - smoothstep(-0.5, 0.5, dist));
    }

    return vec4<f32>(
        clamp(rgb_val.r, 0.0, 1.0),
        clamp(rgb_val.g, 0.0, 1.0),
        clamp(rgb_val.b, 0.0, 1.0),
        alpha
    );
}
