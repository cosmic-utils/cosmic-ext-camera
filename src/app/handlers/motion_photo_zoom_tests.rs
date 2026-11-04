// SPDX-License-Identifier: GPL-3.0-only
use super::*;
use crate::backends::camera::types::{FrameData, FrameMetadata, PixelFormat};
use crate::backends::camera::zoom::{ScalerCropBounds, ZoomRect};

#[test]
fn shutter_snapshot_uses_animated_zoom_not_the_reset_destination() {
    let model = AppModel {
        zoom_level: 1.0,
        zoom_animation: Some(crate::app::state::ZoomAnimation {
            from: 4.0,
            start: std::time::Instant::now(),
        }),
        ..Default::default()
    };
    assert!(model.motion_photo_snapshot().zoom > 3.9);
}

fn frame(native: bool) -> Arc<CameraFrame> {
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
    // Native 2x replicates the center source pixels at the same stream size.
    let data = (0..8)
        .flat_map(|y| {
            (0..12).flat_map(move |x| {
                let (sx, sy) = if native {
                    (3 + x / 2, 2 + y / 2)
                } else {
                    (x, y)
                };
                [sx as u8 * 20, sy as u8 * 20, 37, 255]
            })
        })
        .collect();
    Arc::new(CameraFrame {
        width: 12,
        height: 8,
        stride: 48,
        format: PixelFormat::RGBA,
        data: FrameData::from_owned_vec(data),
        yuv_planes: None,
        captured_at: Instant::now(),
        sensor_timestamp_ns: None,
        libcamera_metadata: native.then(|| FrameMetadata {
            scaler_zoom: bounds.applied(Some(bounds.crop(2.0))),
            ..Default::default()
        }),
    })
}

#[test]
fn burst_frame_crop_uses_shutter_total_with_per_frame_isp_evidence() {
    let mut model = AppModel {
        zoom_level: 2.0,
        preview_fit_to_view: true,
        ..Default::default()
    };
    let snapshot = model.motion_photo_snapshot();
    model.motion_photo_capture = Some(MotionPhotoCapture {
        snapshot,
        result: None,
        cancel: Arc::new(AtomicBool::new(false)),
        temporary_path: PathBuf::new(),
    });
    model.zoom_level = 4.0;
    assert_eq!(
        model.motion_photo_burst_frame_crop(&frame(false)),
        Some((3, 2, 6, 4))
    );
    assert_eq!(model.motion_photo_burst_frame_crop(&frame(true)), None);
    let mut raw = (*frame(true)).clone();
    raw.format = PixelFormat::BayerRGGB;
    assert_eq!(
        model.motion_photo_burst_frame_crop(&raw),
        Some((3, 2, 6, 4))
    );
    assert_eq!(
        model.motion_photo_capture.as_ref().unwrap().snapshot.zoom,
        2.0
    );
}

#[tokio::test]
async fn clip_native_first_then_software_keeps_encoder_caps_and_noop_bytes() {
    let processor = continuous_processor(PostProcessingConfig {
        zoom_level: 2.0,
        ..Default::default()
    });
    let mut dimensions = None;
    let native = process_clip_frame(
        &processor,
        frame(true),
        &mut dimensions,
        SensorRotation::None,
        false,
    )
    .await
    .unwrap();
    let repeated = process_clip_frame(
        &processor,
        frame(true),
        &mut dimensions,
        SensorRotation::None,
        false,
    )
    .await
    .unwrap();
    let lag = process_clip_frame(
        &processor,
        frame(false),
        &mut dimensions,
        SensorRotation::None,
        false,
    )
    .await
    .unwrap();
    assert_eq!(native.dimensions(), (12, 8));
    assert_eq!(native, repeated, "unchanged geometry must not resample");
    assert_eq!(native.dimensions(), lag.dimensions());
    for (left, right) in native.pixels().zip(lag.pixels()) {
        for c in 0..3 {
            assert!(left[c].abs_diff(right[c]) <= 5);
        }
    }
}

#[tokio::test]
async fn clip_scaler_settling_feeds_real_memory_encoder_without_caps_changes() {
    use crate::pipelines::motion_photo_ring::test_fixture;
    let processor = continuous_processor(PostProcessingConfig {
        zoom_level: 2.0,
        ..Default::default()
    });
    let mut dimensions = None;
    let first = process_clip_frame(
        &processor,
        frame(false),
        &mut dimensions,
        SensorRotation::None,
        false,
    )
    .await
    .unwrap();
    let mut encoder = MemoryEncoder::new_monitored(
        &test_fixture::software_encoder(),
        first.width(),
        first.height(),
        10,
        800,
        None,
        Instant::now(),
        RecordingMonitor::new("x264enc".into()),
    )
    .unwrap();
    test_fixture::push_frame(&mut encoder, first, 0);
    for (i, native) in [true, true, false, true].into_iter().enumerate() {
        let image = process_clip_frame(
            &processor,
            frame(native),
            &mut dimensions,
            SensorRotation::None,
            false,
        )
        .await
        .unwrap();
        test_fixture::push_frame(&mut encoder, image, (i as u64 + 1) * 100_000_000);
    }
    encoder.check_error().unwrap();
}

#[tokio::test]
async fn clip_scaler_settling_keeps_dimensions_and_source_geometry() {
    let processor = continuous_processor(PostProcessingConfig {
        zoom_level: 2.0,
        ..Default::default()
    });
    let mut dimensions = None;
    let lag = process_clip_frame(
        &processor,
        frame(false),
        &mut dimensions,
        SensorRotation::Rotate90,
        true,
    )
    .await
    .unwrap();
    let native = process_clip_frame(
        &processor,
        frame(true),
        &mut dimensions,
        SensorRotation::Rotate90,
        true,
    )
    .await
    .unwrap();
    assert_eq!(lag.dimensions(), (4, 6));
    assert_eq!(
        native.dimensions(),
        lag.dimensions(),
        "encoder caps must never change with ScalerCrop completion"
    );
    for (left, right) in lag.pixels().zip(native.pixels()) {
        for c in 0..3 {
            assert!(
                left[c].abs_diff(right[c]) <= 5,
                "source geometry moved: {left:?} != {right:?}"
            );
        }
    }
}
