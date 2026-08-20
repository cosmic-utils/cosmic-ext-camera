// SPDX-License-Identifier: GPL-3.0-only
// GPU shader for direct RGBA texture rendering with object-fit: cover support
// Filter functions are prepended by the Rust code from shaders/filters.wgsl

@group(0) @binding(0)
var texture_rgba: texture_2d<f32>;

@group(0) @binding(1)
var sampler_video: sampler;

struct ViewportUniform {
    viewport_size: vec2<f32>,   // Full widget size
    content_fit_mode: f32,      // 0.0 = Contain, 1.0 = Cover (interpolated during animation)
    filter_mode: u32,           // Filter index (0-15)
    corner_radius: f32,         // Corner radius in pixels (0 = no rounding)
    mirror_horizontal: u32,     // 0 = normal, 1 = mirrored horizontally
    uv_offset: vec2<f32>,       // UV offset for scroll clipping (0-1)
    uv_scale: vec2<f32>,        // UV scale for scroll clipping (0-1)
    crop_uv_min: vec2<f32>,     // Crop UV min (u_min, v_min) - normalized 0-1
    crop_uv_max: vec2<f32>,     // Crop UV max (u_max, v_max) - normalized 0-1
    zoom_level: f32,            // Zoom level (1.0 = no zoom, 2.0 = 2x zoom)
    rotation: u32,              // Sensor rotation: 0=None, 1=90CW, 2=180, 3=270CW
    bar_top_height: f32,        // Top bar height in pixels (for contain centering)
    bar_bottom_height: f32,     // Bottom bar height in pixels
    kawase_offset: f32,        // Unused here — read by the Kawase passes
    dim_factor: f32,           // Unused here — applied by the frosted composite
    letterbox_color: vec4<f32>, // RGBA — only used by the blur pass; declared here so the struct matches
    // The rect (x, y, w, h) the corners are cut from, in PHYSICAL px of the
    // render target, i.e. the same space as `@builtin(position)`.
    panel_rect: vec4<f32>,
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

// `apply_texture_filter` comes from the shared texture-filter prelude
// (src/shaders/texture_filters.wgsl) and `rounded_box_sdf` from the shared
// geometry prelude (src/shaders/geometry.wgsl), both concatenated ahead of this
// file in `VideoPipeline::new`.

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

// Fragment shader - RGBA passthrough with Cover mode support
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Apply scroll clipping UV transformation
    // This maps the visible portion's UV (0-1) to the correct portion of the full widget
    var tex_coords = viewport.uv_offset + in.tex_coords * viewport.uv_scale;

    // Apply horizontal mirror if enabled (selfie mode)
    // This happens BEFORE rotation so the mirror is in screen space
    if (viewport.mirror_horizontal == 1u) {
        tex_coords.x = 1.0 - tex_coords.x;
    }

    // Apply rotation correction for sensor orientation
    // Transforms UV coordinates to correct for physical sensor rotation
    // For a sensor mounted N degrees CW, we rotate the UV coords N degrees CW
    // to sample from the correct position in the rotated texture
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

    // Apply Cover/Contain blend (0.0 = Contain, 1.0 = Cover, intermediate = animating).
    //
    // The effective crop region is itself blended: at 1.0 it degenerates to the
    // full texture (no crop), at 0.0 it is the full aspect-ratio crop.  This keeps
    // the animated transition continuous — no discrete snap at the endpoints even
    // when Cover and Contain would nominally show different regions.
    let blend = viewport.content_fit_mode;
    let effective_crop_min = mix(viewport.crop_uv_min, vec2<f32>(0.0, 0.0), blend);
    let effective_crop_max = mix(viewport.crop_uv_max, vec2<f32>(1.0, 1.0), blend);
    {
        // Get texture dimensions, accounting for rotation
        let raw_tex_size = vec2<f32>(textureDimensions(texture_rgba));
        var tex_size = raw_tex_size;
        if (viewport.rotation == 1u || viewport.rotation == 3u) {
            tex_size = vec2<f32>(raw_tex_size.y, raw_tex_size.x);
        }
        // Effective dimensions after the (blended) crop.
        // `crop_range` is in texture-orientation (crop UVs are sensor-space, see
        // PhotoAspectRatio::crop_uv); `tex_size` is in display-orientation (swapped
        // above). Swap `crop_range` to display-orientation before multiplying so
        // `effective_tex` has the right aspect on rotated sensors. Without this,
        // an aspect-ratio crop in Contain mode produces a wrong-shape letterbox
        // and distorts the sampled image on the phone (rotation 1 / 3).
        var crop_range = effective_crop_max - effective_crop_min;
        if (viewport.rotation == 1u || viewport.rotation == 3u) {
            crop_range = vec2<f32>(crop_range.y, crop_range.x);
        }
        let effective_tex = tex_size * crop_range;

        // Content area between the effective UI edges (for contain centering)
        let content_width = viewport.viewport_size.x - viewport.bar_left_width - viewport.bar_right_width;
        let content_height = viewport.viewport_size.y - viewport.bar_top_height - viewport.bar_bottom_height;
        let content_center_x = (viewport.bar_left_width + content_width * 0.5) / viewport.viewport_size.x;
        let content_center_y = (viewport.bar_top_height + content_height * 0.5) / viewport.viewport_size.y;

        // Zoom levels using the blended effective texture dimensions
        let contain_zoom = min(content_width / effective_tex.x, content_height / effective_tex.y);
        let cover_zoom = max(viewport.viewport_size.x / effective_tex.x, viewport.viewport_size.y / effective_tex.y);

        let zoom = mix(contain_zoom, cover_zoom, blend);
        let center_x = mix(content_center_x, 0.5, blend);
        let center_y = mix(content_center_y, 0.5, blend);

        var scale = vec2<f32>(
            viewport.viewport_size.x / (effective_tex.x * zoom),
            viewport.viewport_size.y / (effective_tex.y * zoom),
        );
        // For 90/270 rotations, swap scale factors since we're in rotated UV space
        if (viewport.rotation == 1u || viewport.rotation == 3u) {
            scale = vec2<f32>(scale.y, scale.x);
        }

        // The pivot below is in screen-space; rotate it into texture-UV space so it
        // matches `tex_coords` (already rotated above). Without this, the asymmetric
        // `center_y` lands on the wrong axis for 90/270 rotations and is inverted for
        // 180 — Contain centering and aspect-ratio crops drift off-axis on the phone.
        // `tex_coords` was mirrored around the window above. Mirror the pivot
        // too, so the fitted image stays on its physical side of an asymmetric
        // left/right content area while only its pixels are reflected.
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

    // Discard letterbox *before* the crop remap.  The intermediate range [0,1]
    // is the rendered image region; outside that range the crop remap can still
    // produce an in-range texture UV (when `crop_min > 0`), so a post-remap check
    // would silently stretch the crop across the letterbox.
    if (tex_coords.x < 0.0 || tex_coords.x > 1.0 || tex_coords.y < 0.0 || tex_coords.y > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Apply the blended crop remap.  Uses the same `effective_crop_*` as the zoom
    // above so the [0,1] intermediate maps onto the same region the zoom sized for.
    tex_coords = mix(effective_crop_min, effective_crop_max, tex_coords);

    // Apply digital zoom (center crop)
    // At zoom_level 2.0, show only center 50% of the image
    if (viewport.zoom_level > 1.0) {
        let inv_zoom = 1.0 / viewport.zoom_level;
        tex_coords = (tex_coords - vec2<f32>(0.5, 0.5)) * inv_zoom + vec2<f32>(0.5, 0.5);
    }

    // Sample RGBA texture
    var pixel = textureSample(texture_rgba, sampler_video, tex_coords);
    var color = pixel.rgb;

    // Apply the filter (0-14) using the shared preludes.
    color = apply_texture_filter(
        color,
        viewport.filter_mode,
        tex_coords,
        texture_rgba,
        sampler_video,
    );

    // Round the corners off the widget's own rect, exactly as the frosted
    // composite does: `panel_rect` and `corner_radius` in physical px, against
    // `@builtin(position)`. NOT off `viewport_size` — that is the box the fit
    // math works in, and passes that sample an intermediate match it to the
    // intermediate rather than the widget (see the pre-blur's second pass), so a
    // silhouette cut from it lands somewhere other than the widget's edge.
    var alpha = pixel.a;
    if (viewport.corner_radius > 0.0 && viewport.panel_rect.z > 0.0) {
        let half_size = viewport.panel_rect.zw * 0.5;
        let center = viewport.panel_rect.xy + half_size;
        let dist = rounded_box_sdf(in.position.xy - center, half_size, viewport.corner_radius);
        alpha = pixel.a * (1.0 - smoothstep(-1.0, 1.0, dist));
    }

    return vec4<f32>(color, alpha);
}
