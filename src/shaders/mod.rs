// SPDX-License-Identifier: GPL-3.0-only
//! Shared shader definitions and GPU pipelines
//!
//! This module provides the single source of truth for shader implementations.
//! All components (preview, photo capture, virtual camera) use these shared shaders.
//!
//! ## Pipelines
//!
//! - **YUV Convert**: Converts YUV frames (NV12, I420, YUYV) to RGBA on GPU
//! - **GPU Filter**: Applies visual filters (sepia, mono, etc.) to RGBA frames
//! - **Histogram**: Analyzes brightness distribution for exposure metering
//!
//! All pipelines operate on RGBA textures for uniform downstream processing.

mod gpu_convert;
mod gpu_filter;
mod histogram_pipeline;

pub use gpu_convert::{GpuConvertPipeline, GpuFrameInput, get_gpu_convert_pipeline};
pub use gpu_filter::{GpuFilterPipeline, apply_filter_gpu_rgba, get_gpu_filter_pipeline};
pub use histogram_pipeline::{BrightnessMetrics, analyze_brightness_gpu};

/// Precompile all GPU shader pipelines so the first capture doesn't pay compilation cost.
///
/// Triggers device creation and pipeline compilation for both the convert and filter
/// pipelines. Safe to call from an async task at startup.
pub async fn warmup_gpu_pipelines() -> Result<(), String> {
    // 1. Warm up the convert pipeline (debayer, yuv, awb, unpack, filter)
    {
        let mut guard = get_gpu_convert_pipeline().await?;
        if let Some(pipeline) = guard.as_mut() {
            pipeline.warmup_pipelines();
        }
    } // Drop convert lock before acquiring filter lock

    // 2. Warm up the standalone filter pipeline (triggers device + pipeline creation)
    {
        let _guard = get_gpu_filter_pipeline().await?;
    }

    Ok(())
}

/// Shared filter functions (WGSL)
/// Contains: luminance(), hash(), apply_filter()
/// Used by: preview shaders, photo capture, virtual camera
pub const FILTER_FUNCTIONS: &str = include_str!("filters.wgsl");

/// Shared RAW white-balance, highlight reconstruction, and colour transform.
/// Used by both single-frame and burst Bayer finishing shaders.
pub const RAW_COLOUR_FUNCTIONS: &str = include_str!("raw_colour.wgsl");

/// Shared texture-sampling filter functions (WGSL)
/// Contains: `apply_texture_filter()` — the filters that must re-sample the
/// source texture (Chromatic Aberration, Pencil), which is why they are not in
/// [`FILTER_FUNCTIONS`]: that prelude also feeds compute modules, and
/// `textureSample` is fragment-only.
/// Requires [`FILTER_FUNCTIONS`] ahead of it.
/// Used by: the sharp preview shader, pass 0 of the frosted blur chain
pub const TEXTURE_FILTER_FUNCTIONS: &str = include_str!("texture_filters.wgsl");

/// Shared UI-geometry functions (WGSL)
/// Contains: rounded_box_sdf()
/// Used by: the video shader, the frosted composite, the gallery thumbnail shader
pub const GEOMETRY_FUNCTIONS: &str = include_str!("geometry.wgsl");

#[cfg(test)]
fn apply_raw_colour_reference(
    sensor_rgb: [f32; 3],
    sensor_peak: f32,
    gains: [f32; 2],
    ccm: [[f32; 3]; 3],
) -> [f32; 3] {
    let balanced = [
        sensor_rgb[0] * gains[0].max(0.0),
        sensor_rgb[1],
        sensor_rgb[2] * gains[1].max(0.0),
    ];
    let corrected: [f32; 3] = std::array::from_fn(|row| {
        (0..3)
            .map(|column| balanced[column] * ccm[row][column])
            .sum::<f32>()
    });
    let clipped_balanced = balanced.map(|channel| channel.clamp(0.0, 1.0));
    let clipped_corrected: [f32; 3] = std::array::from_fn(|row| {
        (0..3)
            .map(|column| clipped_balanced[column] * ccm[row][column])
            .sum::<f32>()
    });
    let peak = if sensor_peak.is_finite() {
        sensor_peak
    } else {
        0.0
    }
    .clamp(0.0, 1.0);
    let transition = ((peak - 0.980) / (0.995 - 0.980)).clamp(0.0, 1.0);
    let blend = transition * transition * (3.0 - 2.0 * transition);

    std::array::from_fn(|channel| {
        let finite_or_zero = |value: f32| if value.is_finite() { value } else { 0.0 };
        let original = finite_or_zero(corrected[channel]).clamp(0.0, 1.0);
        let clipped = finite_or_zero(clipped_corrected[channel]).clamp(0.0, 1.0);
        original + blend * (clipped - original)
    })
}

#[cfg(test)]
mod tests {
    use super::{RAW_COLOUR_FUNCTIONS, apply_raw_colour_reference};

    const IDENTITY_CCM: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    fn assert_near(actual: [f32; 3], expected: [f32; 3]) {
        for channel in 0..3 {
            assert!(
                (actual[channel] - expected[channel]).abs() < 1.0e-6,
                "channel {channel}: expected {}, got {}",
                expected[channel],
                actual[channel]
            );
        }
    }

    fn validate_shader(source: &str) {
        let module = naga::front::wgsl::parse_str(source).expect("WGSL should parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("WGSL should validate");
    }

    #[test]
    fn clipped_pixel_front_camera_highlight_is_neutral() {
        let sensor_rgb = [0.7296, 0.9954, 0.9771];
        let gains = [2.1832347, 1.6135153];
        let ccm = [
            [2.100_02, -0.7964949, -0.30352515],
            [-0.469_05, 1.9604946, -0.49144363],
            [-0.1991288, -0.585788, 1.7849169],
        ];

        let output = apply_raw_colour_reference(sensor_rgb, 0.99509333, gains, ccm);

        assert!(output.iter().all(|channel| channel.is_finite()));
        assert!(
            output.iter().all(|channel| *channel >= 0.80),
            "clipped white became non-white: {output:?}"
        );
        let spread = output.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - output.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            spread < 0.18,
            "clipped white retained a colour cast: {output:?}"
        );
    }

    #[test]
    fn clipped_pixel_matches_preview_colour() {
        let sensor_rgb = [0.7296, 0.9954, 0.9771];
        let gains = [2.1832347, 1.6135153];
        let ccm = [
            [2.100_02, -0.7964949, -0.30352515],
            [-0.469_05, 1.9604946, -0.49144363],
            [-0.1991288, -0.585788, 1.7849169],
        ];

        let output = apply_raw_colour_reference(sensor_rgb, 0.99509333, gains, ccm);

        assert_near(output, [1.0, 0.9909827, 1.0]);
    }

    #[test]
    fn in_gamut_raw_colour_is_unchanged() {
        let output = apply_raw_colour_reference([0.2, 0.3, 0.1], 0.3, [2.0, 1.5], IDENTITY_CCM);

        assert_near(output, [0.4, 0.3, 0.15]);
    }

    #[test]
    fn unclipped_saturated_colour_matches_original_pipeline() {
        let output = apply_raw_colour_reference([0.9, 0.1, 0.1], 0.9, [2.0, 1.5], IDENTITY_CCM);

        assert_near(output, [1.0, 0.1, 0.15]);
    }

    #[test]
    fn unclipped_out_of_gamut_colour_matches_original_pipeline() {
        let ccm = [
            [2.100_02, -0.7964949, -0.30352515],
            [-0.469_05, 1.9604946, -0.49144363],
            [-0.1991288, -0.585788, 1.7849169],
        ];
        let output = apply_raw_colour_reference([0.6, 0.6, 0.55], 0.6, [2.1832347, 1.6135153], ccm);

        assert_near(output, [1.0, 0.12574552, 0.97167516]);
    }

    #[test]
    fn highlight_knee_has_exact_original_and_recovered_endpoints() {
        let ccm = [
            [2.100_02, -0.7964949, -0.30352515],
            [-0.469_05, 1.9604946, -0.49144363],
            [-0.1991288, -0.585788, 1.7849169],
        ];
        let original =
            apply_raw_colour_reference([0.6, 0.6, 0.55], 0.980, [2.1832347, 1.6135153], ccm);
        let recovered =
            apply_raw_colour_reference([0.6, 0.6, 0.55], 0.995, [2.1832347, 1.6135153], ccm);

        assert_near(original, [1.0, 0.12574552, 0.97167516]);
        assert_near(recovered, [1.0, 0.27112326, 1.0]);
    }

    #[test]
    fn clipped_in_gamut_colour_is_unchanged() {
        let output = apply_raw_colour_reference([0.9, 0.1, 0.1], 1.0, [1.0, 1.0], IDENTITY_CCM);

        assert_near(output, [0.9, 0.1, 0.1]);
    }

    #[test]
    fn non_finite_highlight_input_is_contained() {
        let output = apply_raw_colour_reference(
            [f32::NAN, f32::INFINITY, -0.2],
            1.0,
            [1.0, 1.0],
            IDENTITY_CCM,
        );

        assert!(output.iter().all(|channel| channel.is_finite()));
        assert_near(output, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn raw_colour_blends_recovered_colour_only_after_both_ccms() {
        let shader = include_str!("raw_colour.wgsl");
        let original_ccm = shader.find("dot(balanced").expect("missing original CCM");
        let clipped_clamp = shader
            .find("let clipped_balanced = clamp(balanced")
            .expect("missing clipped-channel clamp");
        let clipped_ccm = shader
            .find("dot(clipped_balanced")
            .expect("missing clipped-colour CCM");
        let blend = shader
            .find("return repair_highlight(sensor_peak")
            .expect("missing sensor-gated highlight blend");

        assert!(original_ccm < clipped_clamp);
        assert!(clipped_clamp < clipped_ccm);
        assert!(clipped_ccm < blend);
    }

    #[test]
    fn debayer_uses_highlight_safe_colour_transform() {
        let shader = include_str!("debayer.wgsl");

        assert!(
            shader.contains("apply_raw_colour("),
            "debayer shader bypasses RAW highlight reconstruction"
        );
    }

    #[test]
    fn burst_finishing_uses_highlight_safe_colour_transform() {
        let shader = include_str!("burst_mode/bayer_finish.wgsl");

        assert!(
            shader.contains("apply_raw_colour("),
            "burst finishing bypasses RAW highlight reconstruction"
        );
    }

    #[test]
    fn burst_peak_uses_bilinear_contributors() {
        let shader = include_str!("burst_mode/bayer_finish.wgsl");

        assert!(
            shader.contains("bilinear_channel_with_peak"),
            "burst clipping detection can hide a clipped contributing sample"
        );
    }

    #[test]
    fn composed_raw_shaders_validate() {
        for body in [
            include_str!("debayer.wgsl"),
            include_str!("burst_mode/bayer_finish.wgsl"),
        ] {
            validate_shader(&format!("{RAW_COLOUR_FUNCTIONS}\n{body}"));
        }
    }
}
