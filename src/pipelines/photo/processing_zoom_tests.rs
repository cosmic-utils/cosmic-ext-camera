// SPDX-License-Identifier: GPL-3.0-only
//! Per-frame completed ScalerCrop evidence through the actual still processor.
use super::*;
use crate::backends::camera::types::{FrameData, FrameMetadata};
use crate::backends::camera::zoom::{FrameScalerZoom, ScalerCropBounds, ZoomRect};

pub(crate) fn applied(zoom: f32) -> FrameScalerZoom {
    let base = ZoomRect {
        x: 0,
        y: 0,
        width: 120,
        height: 80,
    };
    let bounds = ScalerCropBounds::new(
        ZoomRect {
            width: 1,
            height: 1,
            ..base
        },
        base,
        base,
        base,
        false,
    )
    .unwrap();
    bounds.applied(Some(bounds.crop(zoom))).unwrap()
}

fn frame(applied_zoom: Option<f32>) -> Arc<CameraFrame> {
    Arc::new(CameraFrame {
        width: 12,
        height: 8,
        stride: 48,
        format: PixelFormat::RGBA,
        data: FrameData::from_owned_vec(
            (0..8)
                .flat_map(|y| (0..12).flat_map(move |x| [x, y, 37, 255]))
                .collect(),
        ),
        yuv_planes: None,
        captured_at: std::time::Instant::now(),
        sensor_timestamp_ns: None,
        libcamera_metadata: applied_zoom.map(|zoom| FrameMetadata {
            scaler_zoom: Some(applied(zoom)),
            ..Default::default()
        }),
    })
}

#[tokio::test]
async fn completed_scaler_crop_keeps_aspect_region_before_residual_zoom() {
    let processor = PostProcessor::new(PostProcessingConfig {
        crop_rect: Some((2, 1, 8, 6)),
        zoom_level: 4.0,
        ..Default::default()
    });
    for (evidence, dimensions, origin) in [
        (None, (2, 2), (5, 3)),
        (Some(2.0), (4, 3), (4, 2)),
        (Some(4.0), (8, 6), (2, 1)),
    ] {
        let result = processor.process(frame(evidence)).await.unwrap();
        assert_eq!(result.image.dimensions(), dimensions);
        assert_eq!(result.image.get_pixel(0, 0).0, [origin.0, origin.1, 37]);
    }
}

#[test]
fn raw_still_ignores_scaler_evidence_and_yuv_uses_it() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu("raw_still_ignores_scaler_evidence_and_yuv_uses_it");
        return;
    };
    pollster::block_on(async {
        for format in [PixelFormat::BayerRGGB, PixelFormat::YUYV] {
            let mut source = (*frame(Some(2.0))).clone();
            source.format = format;
            source.stride = if format.is_bayer() { 12 } else { 24 };
            source.data =
                FrameData::from_owned_vec(vec![64; (source.stride * source.height) as usize]);
            let processor = PostProcessor::new(PostProcessingConfig {
                zoom_level: 4.0,
                ..Default::default()
            });
            let with_evidence = processor.process(Arc::new(source.clone())).await.unwrap();
            let gpu_rgba = processor.gpu_rgba_for_test(&source).await.unwrap();
            assert_eq!(
                gpu_rgba.len(),
                (with_evidence.width * with_evidence.height * 4) as usize
            );
            source.libcamera_metadata = None;
            let without = processor.process(Arc::new(source)).await.unwrap();
            assert_eq!(without.image.dimensions(), (3, 2));
            if format.is_bayer() {
                assert_eq!(
                    with_evidence.image, without.image,
                    "RAW sensor pixels are never ISP-cropped"
                );
            } else {
                assert_eq!(with_evidence.image.dimensions(), (6, 4));
            }
        }
    });
}

#[tokio::test]
async fn processed_still_uses_completed_crop_per_frame_not_requested_total() {
    let processor = PostProcessor::new(PostProcessingConfig {
        zoom_level: 4.0,
        ..Default::default()
    });
    for (evidence, dimensions, origin) in [
        (None, (3, 2), (4, 3)),
        (Some(1.0), (3, 2), (4, 3)),
        (Some(2.0), (6, 4), (3, 2)),
        (Some(4.0), (12, 8), (0, 0)),
        (None, (3, 2), (4, 3)),
    ] {
        let result = processor.process(frame(evidence)).await.unwrap();
        assert_eq!(
            (result.width, result.height),
            dimensions,
            "evidence={evidence:?}"
        );
        assert_eq!(result.image.get_pixel(0, 0).0, [origin.0, origin.1, 37]);
    }
}

#[tokio::test]
async fn asymmetric_cover_crop_tracks_full_frame_zoom_pivot() {
    let processor = PostProcessor::new(PostProcessingConfig {
        crop_rect: Some((0, 0, 12, 4)),
        zoom_level: 2.0,
        ..Default::default()
    });
    let software = processor.process(frame(None)).await.unwrap();
    let mut native = (*frame(Some(2.0))).clone();
    native.data = FrameData::from_owned_vec(
        (0..8)
            .flat_map(|y| (0..12).flat_map(move |x| [3 + x / 2, 2 + y / 2, 37, 255]))
            .collect(),
    );
    let native = processor.process(Arc::new(native)).await.unwrap();
    assert_eq!(software.image.get_pixel(0, 0).0, [3, 2, 37]);
    assert_eq!(software.image.get_pixel(0, 0), native.image.get_pixel(0, 0));
}
