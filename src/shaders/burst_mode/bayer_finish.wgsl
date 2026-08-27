// SPDX-License-Identifier: GPL-3.0-only
//
// HDR+ finishing pipeline: demosaic merged Bayer planes → full-res RGBA
// with white balance and colour correction matrix.
//
// Input:  Merged Bayer planes packed as RGBA f32 at half resolution
//         R = red channel, G = green_r (Gr), B = blue channel, A = green_b (Gb)
//
// Output: Full-resolution RGBA f32 image
//
// Per HDR+ paper Section 6: the finishing pipeline applies
// black-level subtraction (already done during plane extraction),
// white balance → demosaic → colour correction.
// Gamma/tonemapping is handled separately by the tonemap shader.

// Merged Bayer planes at half resolution (read-only)
@group(0) @binding(0) var<storage, read> bayer_planes: array<vec4<f32>>;
// Output full-resolution RGBA (read-write)
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
// Parameters
@group(0) @binding(2) var<uniform> params: FinishParams;

struct FinishParams {
    // Half-resolution dimensions (Bayer plane dimensions)
    half_width: u32,
    half_height: u32,
    // Full-resolution dimensions
    full_width: u32,
    full_height: u32,
    // White balance gains
    gain_r: f32,
    gain_b: f32,
    // Colour correction matrix (row-major, 3×3 padded to 3×vec4)
    ccm_row0: vec4<f32>,  // xyz used, w=pad
    ccm_row1: vec4<f32>,
    ccm_row2: vec4<f32>,
    // Use ISP colour processing (1=apply WB+CCM, 0=skip)
    use_colour: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Sample a Bayer plane pixel with bounds clamping
fn sample_plane(x: i32, y: i32) -> vec4<f32> {
    let cx = clamp(x, 0, i32(params.half_width) - 1);
    let cy = clamp(y, 0, i32(params.half_height) - 1);
    return bayer_planes[u32(cy) * params.half_width + u32(cx)];
}

// Bilinear interpolation helper for a single channel
fn bilinear_channel_with_peak(fx: f32, fy: f32, ix: i32, iy: i32, ch: u32) -> vec2<f32> {
    let s00 = sample_plane(ix, iy)[ch];
    let s10 = sample_plane(ix + 1, iy)[ch];
    let s01 = sample_plane(ix, iy + 1)[ch];
    let s11 = sample_plane(ix + 1, iy + 1)[ch];

    let wx = fx - f32(ix);
    let wy = fy - f32(iy);

    let value = mix(mix(s00, s10, wx), mix(s01, s11, wx), wy);
    let peak = max(max(s00, s10), max(s01, s11));
    return vec2<f32>(value, peak);
}

@compute @workgroup_size(16, 16)
fn demosaic_and_finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;

    if ox >= params.full_width || oy >= params.full_height {
        return;
    }

    // Map full-res pixel to Bayer plane coordinates
    // Each Bayer plane pixel corresponds to a 2×2 region in the full image.
    // The plane pixel centers are at (0.5, 0.5), (1.5, 0.5), etc. in plane coords,
    // which map to (1, 1), (3, 1), etc. in full-res coords.
    //
    // For full-res pixel (ox, oy), the corresponding plane coordinate is:
    //   plane_x = (ox - 0.5) / 2.0  (shifted by 0.5 because plane pixels are centered)
    //   plane_y = (oy - 0.5) / 2.0
    //
    // But channel offsets within the 2×2 Bayer pattern matter:
    // - R is at (0,0) in each quad → centered at plane pixel position
    // - Gr is at (1,0) → shifted +0.5 in x relative to R
    // - Gb is at (0,1) → shifted +0.5 in y relative to R
    // - B is at (1,1) → shifted +0.5 in both x and y

    // Base plane coordinate for this full-res pixel
    let px = (f32(ox) - 0.5) * 0.5;
    let py = (f32(oy) - 0.5) * 0.5;

    // Integer part for bilinear sampling
    let ix = i32(floor(px));
    let iy = i32(floor(py));

    // Interpolate each channel from its plane using bilinear filtering
    // Channel 0 = R, 1 = Gr, 2 = B, 3 = Gb
    let r_sample = bilinear_channel_with_peak(px, py, ix, iy, 0u);
    let gr_sample = bilinear_channel_with_peak(px, py, ix, iy, 1u);
    let b_sample = bilinear_channel_with_peak(px, py, ix, iy, 2u);
    let gb_sample = bilinear_channel_with_peak(px, py, ix, iy, 3u);
    let r = r_sample.x;
    let gr = gr_sample.x;
    let b = b_sample.x;
    let gb = gb_sample.x;

    // Average the two green channels (Gr and Gb) for the green component
    let g = (gr + gb) * 0.5;
    let sensor_peak = clamp(
        max(max(r_sample.y, b_sample.y), max(gr_sample.y, gb_sample.y)),
        0.0,
        1.0,
    );

    // Apply white balance and colour correction if enabled
    var rgb = vec3<f32>(r, g, b);

    if params.use_colour == 1u {
        rgb = apply_raw_colour(
            rgb,
            sensor_peak,
            params.gain_r,
            params.gain_b,
            params.ccm_row0.xyz,
            params.ccm_row1.xyz,
            params.ccm_row2.xyz,
        );
    }

    // Clamp to valid range (CCM can produce values outside 0..1)
    rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));

    // Write output (alpha = 1.0)
    let out_idx = oy * params.full_width + ox;
    output[out_idx] = vec4<f32>(rgb, 1.0);
}
