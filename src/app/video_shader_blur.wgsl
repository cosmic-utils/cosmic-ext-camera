// SPDX-License-Identifier: GPL-3.0-only
// PASS 0 of the blur chain: the TRANSFORM pass.
//
// It applies every preview transform — scroll clip, mirror, sensor rotation,
// cover/contain fit, letterbox fill, crop remap, digital zoom, colour filter —
// and renders the result into a PHYSICAL SCREEN RESOLUTION target. It runs NO
// blur kernel at all.
//
// # Why the kernel is gone
//
// This pass used to double as a blur pass: it sampled the sharp, full-resolution
// sensor frame through a 37-tap ring rosette. At high frost thickness the ring
// spacing reached ~18 sensor texels between taps across un-prefiltered data, and
// the 12-fold ring structure ghosted into the output as visible BANDING (found
// on device). A sparse lattice sampling sharp data is a structurally bad idea and
// no radius tuning fixes it.
//
// The chain now separates the two jobs. This pass only RESAMPLES (one bilinear
// tap), and `video_shader_kawase.wgsl` — a verbatim port of cosmic-comp's own
// dual-Kawase — does all the blurring, over progressively halved copies of THIS
// pass's output. Every Kawase kernel is dense relative to its own level and only
// ever samples data the previous pass band-limited, so it cannot band.
//
// This pass is also what makes the Kawase offsets literally cosmic-comp's: they
// are authored in physical screen px, and by rendering into a screen-resolution
// screen-space target here, the rest of the chain works in exactly that unit. No
// texel-to-screen conversion exists any more, because there is nothing to
// convert.

@group(0) @binding(0)
var texture_blur: texture_2d<f32>;

@group(0) @binding(1)
var sampler_blur: sampler;

struct ViewportUniform {
    viewport_size: vec2<f32>,   // Full widget size
    content_fit_mode: f32,      // 0.0 = Contain, 1.0 = Cover
    filter_mode: u32,           // Filter index (this pass owns the filter; 0 = none)
    corner_radius: f32,         // Unused here — the final composite rounds the corners
    mirror_horizontal: u32,     // 0 = normal, 1 = mirrored horizontally
    uv_offset: vec2<f32>,       // UV offset for scroll clipping (0-1)
    uv_scale: vec2<f32>,        // UV scale for scroll clipping (0-1)
    crop_uv_min: vec2<f32>,     // Crop UV min (u_min, v_min) - normalized 0-1
    crop_uv_max: vec2<f32>,     // Crop UV max (u_max, v_max) - normalized 0-1
    zoom_level: f32,            // Digital zoom (1.0 = none); applied HERE, and only here
    rotation: u32,              // Sensor rotation: 0=None, 1=90CW, 2=180, 3=270CW
    bar_top_height: f32,        // Top bar height in pixels
    bar_bottom_height: f32,     // Bottom bar height in pixels
    kawase_offset: f32,         // Unused here — read by the Kawase passes
    dim_factor: f32,            // Unused here — applied by the final composite
    letterbox_color: vec4<f32>, // RGBA fill for letterbox (alpha unused)
    panel_rect: vec4<f32>,      // Unused here — read by the final composite
    noise: f32,
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

// Vertex shader - creates a fullscreen quad
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Generate fullscreen triangle vertices
    let x = f32((vertex_index & 1u) << 2u) - 1.0;
    let y = f32((vertex_index & 2u) << 1u) - 1.0;

    out.position = vec4<f32>(x, -y, 0.0, 1.0);
    out.tex_coords = vec2<f32>((x + 1.0) * 0.5, (y + 1.0) * 0.5);

    return out;
}

// Fragment shader — transform only, one bilinear tap.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Apply scroll clipping UV transformation
    var tex_coords = viewport.uv_offset + in.tex_coords * viewport.uv_scale;

    // Apply horizontal mirror if enabled (selfie mode)
    // This happens BEFORE rotation so the mirror is in screen space
    if (viewport.mirror_horizontal == 1u) {
        tex_coords.x = 1.0 - tex_coords.x;
    }

    // Apply rotation correction for sensor orientation
    if (viewport.rotation == 1u) {
        // 90 CW sensor -> sample rotated 90 CW: (u,v) -> (1-v, u)
        tex_coords = vec2<f32>(1.0 - tex_coords.y, tex_coords.x);
    } else if (viewport.rotation == 2u) {
        // 180 sensor -> rotate 180: (u,v) -> (1-u, 1-v)
        tex_coords = vec2<f32>(1.0 - tex_coords.x, 1.0 - tex_coords.y);
    } else if (viewport.rotation == 3u) {
        // 270 CW sensor -> sample rotated 270 CW: (u,v) -> (v, 1-u)
        tex_coords = vec2<f32>(tex_coords.y, 1.0 - tex_coords.x);
    }

    // Cover/Contain blend with bar-aware centering and a blended crop region.
    // See video_shader.wgsl for the rationale — same model is used here so the
    // blur texture tracks the main preview during the fit/fill animation.
    let blend = viewport.content_fit_mode;
    let effective_crop_min = mix(viewport.crop_uv_min, vec2<f32>(0.0, 0.0), blend);
    let effective_crop_max = mix(viewport.crop_uv_max, vec2<f32>(1.0, 1.0), blend);
    {
        let raw_tex_size = vec2<f32>(textureDimensions(texture_blur));
        var tex_size_dim = raw_tex_size;
        if (viewport.rotation == 1u || viewport.rotation == 3u) {
            tex_size_dim = vec2<f32>(raw_tex_size.y, raw_tex_size.x);
        }
        // crop_range is in texture-orientation; swap to display before multiplying.
        // See video_shader.wgsl for the rationale.
        var crop_range = effective_crop_max - effective_crop_min;
        if (viewport.rotation == 1u || viewport.rotation == 3u) {
            crop_range = vec2<f32>(crop_range.y, crop_range.x);
        }
        let effective_tex = tex_size_dim * crop_range;

        let content_width = viewport.viewport_size.x - viewport.bar_left_width - viewport.bar_right_width;
        let content_height = viewport.viewport_size.y - viewport.bar_top_height - viewport.bar_bottom_height;
        let content_center_x = (viewport.bar_left_width + content_width * 0.5) / viewport.viewport_size.x;
        let content_center_y = (viewport.bar_top_height + content_height * 0.5) / viewport.viewport_size.y;
        let contain_zoom = min(content_width / effective_tex.x, content_height / effective_tex.y);
        let cover_zoom = max(viewport.viewport_size.x / effective_tex.x, viewport.viewport_size.y / effective_tex.y);
        let zoom = mix(contain_zoom, cover_zoom, blend);
        let center_x = mix(content_center_x, 0.5, blend);
        let center_y = mix(content_center_y, 0.5, blend);
        var scale = vec2<f32>(
            viewport.viewport_size.x / (effective_tex.x * zoom),
            viewport.viewport_size.y / (effective_tex.y * zoom),
        );

        if (viewport.rotation == 1u || viewport.rotation == 3u) {
            scale = vec2<f32>(scale.y, scale.x);
        }

        // Rotate the screen-space pivot into texture-UV space — see video_shader.wgsl.
        let pivot_x = select(center_x, 1.0 - center_x, viewport.mirror_horizontal == 1u);
        var pivot = vec2<f32>(pivot_x, center_y);
        if (viewport.rotation == 1u) {
            pivot = vec2<f32>(1.0 - pivot.y, pivot.x);
        } else if (viewport.rotation == 2u) {
            pivot = vec2<f32>(1.0 - pivot.x, 1.0 - pivot.y);
        } else if (viewport.rotation == 3u) {
            pivot = vec2<f32>(pivot.y, 1.0 - pivot.x);
        }
        tex_coords = (tex_coords - pivot) * scale + vec2<f32>(0.5, 0.5);
    }

    // For the blur backdrop, paint letterbox with the theme's background
    // color (opaque) instead of discarding to transparent — the previous
    // discard let the COSMIC window background show through in Contain /
    // Fit mode. Return *before* the crop remap so we can short-circuit;
    // the regular post-remap letterbox check is unnecessary here because
    // anything out of [0,1] is letterbox by definition.
    if (tex_coords.x < 0.0 || tex_coords.x > 1.0 || tex_coords.y < 0.0 || tex_coords.y > 1.0) {
        return vec4<f32>(viewport.letterbox_color.rgb, 1.0);
    }

    // Apply the blended crop remap
    tex_coords = mix(effective_crop_min, effective_crop_max, tex_coords);

    // Apply digital zoom (center crop), AFTER the crop remap and AFTER the
    // letterbox early-out — the same place video_shader.wgsl applies it, so the
    // frosted backdrop samples exactly the region the sharp preview shows. The
    // order matters and is not cosmetic:
    //
    // * after the crop remap, so the pivot is the centre of the FULL texture in
    //   texture UV space, as it is in video_shader.wgsl. Zooming before the
    //   remap would pivot on the centre of the *crop* instead, which only
    //   coincides while the crop is centred.
    // * after the letterbox early-out, so the letterbox extent stays a property
    //   of the fit, not of the zoom — again matching video_shader.wgsl, whose
    //   early-out also precedes its zoom.
    //
    // THIS IS THE ONLY PASS THAT ZOOMS, because it is the only pass that ever
    // samples the source frame. Everything downstream — the Kawase ping-pong and
    // the final composite — works on this pass's screen-space output, in which
    // the zoom is already baked, and leaves `zoom_level` at its 1.0 default.
    if (viewport.zoom_level > 1.0) {
        let inv_zoom = 1.0 / viewport.zoom_level;
        tex_coords = (tex_coords - vec2<f32>(0.5, 0.5)) * inv_zoom + vec2<f32>(0.5, 0.5);
    }

    // ONE bilinear tap. This pass resamples; it does not blur. See the header
    // for why the 37-tap ring rosette that used to live here was the cause of
    // the banding, and `video_shader_kawase.wgsl` for what replaced it.
    var rgb_val = textureSample(texture_blur, sampler_blur, tex_coords).rgb;

    // Apply the filter here — the one pass that sees the source frame, so the
    // filter is visible in the blur exactly as it is in the sharp preview. The
    // whole range, including the two filters that re-sample: the backdrop that
    // stops at 12 is the backdrop that blurs a colour scene behind a pencil
    // sketch (found on device).
    //
    // Pencil arrives here WITHOUT the preblur the sharp preview gets — `render`
    // routes VIDEO_ID_FROSTED down the Kawase branch, which has no preblur pass.
    // That is the right trade and not an omission: the preblur exists to keep
    // Sobel off sensor noise, and every edge this pass emits is about to be
    // dual-Kawase'd across the screen anyway, so the extra noise is destroyed by
    // the next pass rather than shown. Buying it back would cost a full-res
    // pass on top of a chain already at ~2.6x screen area per frame.
    rgb_val = apply_texture_filter(
        rgb_val,
        viewport.filter_mode,
        tex_coords,
        texture_blur,
        sampler_blur,
    );

    // Opaque, always. The Kawase passes normalize by `sum.a` and treat a = 0 as
    // "outside the region" (see `video_shader_kawase.wgsl`), so this pass MUST
    // emit alpha = 1 across the whole target — letterbox included, which is why
    // the early-out above fills rather than discards. The rounded silhouette and
    // the dim belong to the final composite, not here.
    return vec4<f32>(rgb_val, 1.0);
}
