// SPDX-License-Identifier: GPL-3.0-only
// Shared RAW camera-colour transform with neutral clipped highlights.

fn finite_or_zero(value: f32) -> f32 {
    return select(0.0, value, (value == value) && (abs(value) <= 3.0e38));
}

fn repair_highlight(
    sensor_peak: f32,
    original: vec3<f32>,
    clipped: vec3<f32>,
) -> vec3<f32> {
    let original_colour = vec3<f32>(
        finite_or_zero(original.r),
        finite_or_zero(original.g),
        finite_or_zero(original.b),
    );
    let clipped_colour = vec3<f32>(
        finite_or_zero(clipped.r),
        finite_or_zero(clipped.g),
        finite_or_zero(clipped.b),
    );
    let hard = clamp(original_colour, vec3<f32>(0.0), vec3<f32>(1.0));
    let recovered = clamp(clipped_colour, vec3<f32>(0.0), vec3<f32>(1.0));

    let transition = clamp(
        (clamp(finite_or_zero(sensor_peak), 0.0, 1.0) - 0.980) / (0.995 - 0.980),
        0.0,
        1.0,
    );
    let blend = transition * transition * (3.0 - 2.0 * transition);
    return mix(hard, recovered, blend);
}

fn apply_raw_colour(
    sensor_rgb: vec3<f32>,
    sensor_peak: f32,
    gain_r: f32,
    gain_b: f32,
    ccm_row0: vec3<f32>,
    ccm_row1: vec3<f32>,
    ccm_row2: vec3<f32>,
) -> vec3<f32> {
    let gains = max(vec3<f32>(gain_r, 1.0, gain_b), vec3<f32>(0.0));
    let balanced = sensor_rgb * gains;
    let corrected = vec3<f32>(
        dot(balanced, ccm_row0),
        dot(balanced, ccm_row1),
        dot(balanced, ccm_row2),
    );

    // Saturate white-balanced camera channels before the CCM. Keep the original
    // still path below sensor clipping, then blend to the neutral reconstruction
    // only as a contributing sample clips.
    let clipped_balanced = clamp(balanced, vec3<f32>(0.0), vec3<f32>(1.0));
    let clipped_corrected = vec3<f32>(
        dot(clipped_balanced, ccm_row0),
        dot(clipped_balanced, ccm_row1),
        dot(clipped_balanced, ccm_row2),
    );
    return repair_highlight(sensor_peak, corrected, clipped_corrected);
}
