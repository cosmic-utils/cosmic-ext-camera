// SPDX-License-Identifier: GPL-3.0-only
// GPU compute shader for Bayer to RGBA conversion (demosaicing/debayering)
//
// Supports all four Bayer patterns:
// - RGGB (0): Row 0: R G R G..., Row 1: G B G B...
// - BGGR (1): Row 0: B G B G..., Row 1: G R G R...
// - GRBG (2): Row 0: G R G R..., Row 1: B G B G...
// - GBRG (3): Row 0: G B G B..., Row 1: R G R G...
//
// Algorithm: Bilinear interpolation
// - Each pixel directly measures one color (R, G, or B based on position)
// - The other two colors are interpolated from neighboring pixels
//
// White balance gains are read from a storage buffer (binding 3).
// When ISP provides gains, the CPU writes them directly to the buffer.
// When ISP gains are absent, the GPU AWB shader computes and writes them.

struct DebayerParams {
    width: u32,
    height: u32,
    pattern: u32,           // 0=RGGB, 1=BGGR, 2=GRBG, 3=GBRG
    use_isp_colour: u32,    // 1 = apply gains+CCM, 0 = raw output
    black_level: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    ccm_row0: vec4<f32>,
    ccm_row1: vec4<f32>,
    ccm_row2: vec4<f32>,
}

struct AwbGains {
    gain_r: f32,
    gain_b: f32,
}

@group(0) @binding(0) var tex_bayer: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: DebayerParams;
@group(0) @binding(3) var<storage, read> awb_gains: AwbGains;

// Get pixel value with bounds checking (clamp to edge)
fn sample_bayer(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0i, i32(params.width) - 1i);
    let cy = clamp(y, 0i, i32(params.height) - 1i);
    return textureLoad(tex_bayer, vec2(u32(cx), u32(cy)), 0).r;
}

// Determine what color channel this pixel position measures based on Bayer pattern
// Returns: 0=R, 1=G (on R row), 2=G (on B row), 3=B
fn get_pixel_type(x: u32, y: u32, pattern: u32) -> u32 {
    let even_x = (x & 1u) == 0u;
    let even_y = (y & 1u) == 0u;

    // Pattern layout (2x2 superpixel):
    // RGGB: [R,G][G,B] -> (0,0)=R, (1,0)=G, (0,1)=G, (1,1)=B
    // BGGR: [B,G][G,R] -> (0,0)=B, (1,0)=G, (0,1)=G, (1,1)=R
    // GRBG: [G,R][B,G] -> (0,0)=G, (1,0)=R, (0,1)=B, (1,1)=G
    // GBRG: [G,B][R,G] -> (0,0)=G, (1,0)=B, (0,1)=R, (1,1)=G

    if (pattern == 0u) { // RGGB
        if (even_y) {
            return select(1u, 0u, even_x); // even_x=R, odd_x=G
        } else {
            return select(3u, 2u, even_x); // even_x=G, odd_x=B
        }
    } else if (pattern == 1u) { // BGGR
        if (even_y) {
            return select(1u, 3u, even_x); // even_x=B, odd_x=G
        } else {
            return select(0u, 2u, even_x); // even_x=G, odd_x=R
        }
    } else if (pattern == 2u) { // GRBG
        if (even_y) {
            return select(0u, 1u, even_x); // even_x=G, odd_x=R
        } else {
            return select(2u, 3u, even_x); // even_x=B, odd_x=G
        }
    } else { // GBRG (pattern == 3)
        if (even_y) {
            return select(3u, 1u, even_x); // even_x=G, odd_x=B
        } else {
            return select(2u, 0u, even_x); // even_x=R, odd_x=G
        }
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let ix = i32(x);
    let iy = i32(y);

    let pixel_type = get_pixel_type(x, y, params.pattern);

    var r: f32;
    var g: f32;
    var b: f32;

    let center = sample_bayer(ix, iy);
    var sensor_peak_raw = center;

    // Bilinear interpolation based on pixel position in Bayer pattern
    if (pixel_type == 0u) {
        // Red pixel - interpolate G and B
        let top_left = sample_bayer(ix - 1, iy - 1);
        let top = sample_bayer(ix, iy - 1);
        let top_right = sample_bayer(ix + 1, iy - 1);
        let left = sample_bayer(ix - 1, iy);
        let right = sample_bayer(ix + 1, iy);
        let bottom_left = sample_bayer(ix - 1, iy + 1);
        let bottom = sample_bayer(ix, iy + 1);
        let bottom_right = sample_bayer(ix + 1, iy + 1);
        r = center;
        g = (left + right + top + bottom) * 0.25;
        b = (top_left + top_right + bottom_left + bottom_right) * 0.25;
        sensor_peak_raw = max(
            max(max(top_left, top), max(top_right, left)),
            max(max(center, right), max(bottom_left, max(bottom, bottom_right))),
        );
    } else if (pixel_type == 1u) {
        // Green pixel on R row - interpolate R (horizontal neighbors) and B (vertical neighbors)
        let top = sample_bayer(ix, iy - 1);
        let left = sample_bayer(ix - 1, iy);
        let right = sample_bayer(ix + 1, iy);
        let bottom = sample_bayer(ix, iy + 1);
        r = (left + right) * 0.5;
        g = center;
        b = (top + bottom) * 0.5;
        sensor_peak_raw = max(max(center, top), max(bottom, max(left, right)));
    } else if (pixel_type == 2u) {
        // Green pixel on B row - interpolate R (vertical neighbors) and B (horizontal neighbors)
        let top = sample_bayer(ix, iy - 1);
        let left = sample_bayer(ix - 1, iy);
        let right = sample_bayer(ix + 1, iy);
        let bottom = sample_bayer(ix, iy + 1);
        r = (top + bottom) * 0.5;
        g = center;
        b = (left + right) * 0.5;
        sensor_peak_raw = max(max(center, top), max(bottom, max(left, right)));
    } else {
        // Blue pixel - interpolate R and G
        let top_left = sample_bayer(ix - 1, iy - 1);
        let top = sample_bayer(ix, iy - 1);
        let top_right = sample_bayer(ix + 1, iy - 1);
        let left = sample_bayer(ix - 1, iy);
        let right = sample_bayer(ix + 1, iy);
        let bottom_left = sample_bayer(ix - 1, iy + 1);
        let bottom = sample_bayer(ix, iy + 1);
        let bottom_right = sample_bayer(ix + 1, iy + 1);
        r = (top_left + top_right + bottom_left + bottom_right) * 0.25;
        g = (left + right + top + bottom) * 0.25;
        b = center;
        sensor_peak_raw = max(
            max(max(top_left, top), max(top_right, left)),
            max(max(center, right), max(bottom_left, max(bottom, bottom_right))),
        );
    }

    // Apply white balance and colour correction (in linear space, before gamma)
    if (params.use_isp_colour == 1u) {
        // Subtract sensor black level so gains don't amplify the DC offset per-channel
        let bl = params.black_level;
        let scale = 1.0 / (1.0 - bl);
        r = max(r - bl, 0.0) * scale;
        g = max(g - bl, 0.0) * scale;
        b = max(b - bl, 0.0) * scale;
        let sensor_peak = clamp(max(sensor_peak_raw - bl, 0.0) * scale, 0.0, 1.0);

        // White balance, clipped-highlight reconstruction, and sensor RGB -> sRGB CCM.
        let corrected = apply_raw_colour(
            vec3(r, g, b),
            sensor_peak,
            awb_gains.gain_r,
            awb_gains.gain_b,
            params.ccm_row0.xyz,
            params.ccm_row1.xyz,
            params.ccm_row2.xyz,
        );
        r = corrected.r;
        g = corrected.g;
        b = corrected.b;
    }

    // Clamp to [0,1] range
    let rgb = clamp(vec3(r, g, b), vec3(0.0), vec3(1.0));

    // Apply sRGB gamma only when ISP colour processing is active (data is linear).
    // Without ISP colour, the data is in sensor-native space where gamma is meaningless.
    if (params.use_isp_colour == 1u) {
        let srgb = vec3(linear_to_srgb(rgb.r), linear_to_srgb(rgb.g), linear_to_srgb(rgb.b));
        textureStore(output, vec2(x, y), vec4(srgb, 1.0));
    } else {
        textureStore(output, vec2(x, y), vec4(rgb, 1.0));
    }
}

// Convert linear light to sRGB gamma-corrected value
fn linear_to_srgb(c: f32) -> f32 {
    if (c <= 0.0031308) {
        return c * 12.92;
    }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}
