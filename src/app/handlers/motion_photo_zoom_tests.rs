// SPDX-License-Identifier: GPL-3.0-only
use super::*;
use crate::backends::camera::types::{FrameData, FrameMetadata, PixelFormat};
use crate::backends::camera::zoom::{ScalerCropBounds, ZoomRect};

#[tokio::test]
async fn live_zoom_preserves_recorder_and_accepted_postroll() {
    let mut model = AppModel {
        current_frame: Some(frame(false)),
        preview_fit_to_view: true,
        zoom_level: 1.0,
        ..Default::default()
    };
    model.config.motion_photo = true;
    model.config.record_audio = false;
    model.available_video_encoders =
        vec![crate::pipelines::motion_photo_ring::test_fixture::software_encoder()];
    model.sync_motion_photo_recorder();
    let original = model.motion_photo_recorder.as_ref().unwrap().cancel.clone();
    for zoom in [2.0, 1.0] {
        let _ = model.update(crate::app::Message::PinchZoom(zoom));
        assert!(
            model
                .motion_photo_recorder
                .as_ref()
                .is_some_and(|recorder| { Arc::ptr_eq(&original, &recorder.cancel) }),
            "pre-roll must survive zoom before the shutter"
        );
    }
    model.start_motion_photo_capture().unwrap();

    let mut retained = true;
    for message in [
        crate::app::Message::PinchZoom(2.0),
        crate::app::Message::ResetZoom,
        crate::app::Message::ZoomAnimationTick,
    ] {
        let _ = model.update(message);
        retained &= model
            .motion_photo_recorder
            .as_ref()
            .is_some_and(|recorder| {
                Arc::ptr_eq(&original, &recorder.cancel) && !original.load(Ordering::Acquire)
            });
    }
    let shutter_zoom = model.motion_photo_capture.as_ref().unwrap().snapshot.zoom;
    if let Some(recorder) = model.motion_photo_recorder.take() {
        recorder.stop_and_wait().await.unwrap();
    }
    if let Some(worker) = model.motion_photo_retiring.take() {
        tokio::task::spawn_blocking(move || worker.join().unwrap())
            .await
            .unwrap();
    }
    assert!(
        retained,
        "zoom must not retire the prebuffer or its accepted post-roll"
    );
    assert_eq!(shutter_zoom, 1.0, "the still retains its shutter-time zoom");
}

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

#[tokio::test]
async fn encoded_motion_clip_follows_live_zoom_through_pre_and_postroll() {
    let source = || {
        let mut source = (*frame(false)).clone();
        source.width = 64;
        source.height = 48;
        source.stride = 256;
        source.data = FrameData::from_owned_vec(
            (0..48)
                .flat_map(|y| (0..64).flat_map(move |x| [x * 3, y * 4, 37, 255]))
                .collect(),
        );
        Arc::new(source)
    };
    let mut model = AppModel {
        current_frame: Some(source()),
        preview_fit_to_view: true,
        photo_aspect_ratio: crate::app::PhotoAspectRatio::Native,
        zoom_level: 1.0,
        ..Default::default()
    };
    model.config.motion_photo = true;
    model.config.record_audio = false;
    model.available_video_encoders =
        vec![crate::pipelines::motion_photo_ring::test_fixture::software_encoder()];
    model.sync_motion_photo_recorder();
    let original = model.motion_photo_recorder.as_ref().unwrap().cancel.clone();
    let deadline = Instant::now() + Duration::from_secs(10);
    model
        .warm_motion_photo_recorder(Duration::from_millis(50))
        .await;
    model.start_motion_photo_capture().unwrap();
    let _ = model.update(crate::app::Message::PinchZoom(2.0));
    while !model
        .motion_photo_capture
        .as_ref()
        .unwrap()
        .result
        .as_ref()
        .unwrap()
        .is_finished()
    {
        assert!(Instant::now() < deadline, "accepted clip did not finish");
        model.forward_motion_photo_frame(&source());
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(Arc::ptr_eq(
        &original,
        &model.motion_photo_recorder.as_ref().unwrap().cancel,
    ));
    let capture = model.motion_photo_capture.as_mut().unwrap();
    assert_eq!(capture.snapshot.zoom, 1.0);
    let clip = capture.result.take().unwrap().await.unwrap().unwrap();
    model
        .motion_photo_recorder
        .take()
        .unwrap()
        .stop_and_wait()
        .await
        .unwrap();

    let decoded = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&clip.0)
        .args([
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "1", "pipe:1",
        ])
        .output()
        .unwrap();
    assert!(
        decoded.status.success(),
        "{}: {}",
        clip.0.display(),
        String::from_utf8_lossy(&decoded.stderr)
    );
    let (frames, remainder) = decoded.stdout.as_chunks::<{ 64 * 48 * 3 }>();
    assert!(remainder.is_empty(), "decoder returned an incomplete frame");
    assert!(
        frames.len() >= 2,
        "clip must contain both pre- and post-roll"
    );
    let before = frames.first().unwrap();
    let after = frames.last().unwrap();
    assert!(
        after[0] > before[0] + 25 && after[1] > before[1] + 15,
        "post-roll must record the new centered zoom, not the original recipe: before={:?}, after={:?}",
        &before[..3],
        &after[..3]
    );
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
        2.0,
        &mut dimensions,
        SensorRotation::None,
        false,
    )
    .await
    .unwrap();
    let repeated = process_clip_frame(
        &processor,
        frame(true),
        2.0,
        &mut dimensions,
        SensorRotation::None,
        false,
    )
    .await
    .unwrap();
    let lag = process_clip_frame(
        &processor,
        frame(false),
        2.0,
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
async fn live_zoom_keeps_unzoomed_resolution_even_when_first_frame_is_zoomed() {
    let processor = continuous_processor(PostProcessingConfig::default());
    let mut dimensions = None;
    let mut images = Vec::new();
    for (desired, native) in [(2.0, false), (2.0, true), (1.0, true), (1.0, false)] {
        images.push(
            process_clip_frame(
                &processor,
                frame(native),
                desired,
                &mut dimensions,
                SensorRotation::None,
                false,
            )
            .await
            .unwrap(),
        );
    }
    for image in &images {
        assert_eq!(
            image.dimensions(),
            (12, 8),
            "zoom must not select encoder resolution"
        );
    }
    let expected = processor.process(frame(false)).await.unwrap().image;
    assert_eq!(
        images.last().unwrap(),
        &expected,
        "reset restores the full frame"
    );
}

#[tokio::test]
async fn live_clip_zoom_preserves_aspect_orientation_and_still_configuration() {
    let processor = continuous_processor(PostProcessingConfig {
        crop_rect: Some((1, 1, 10, 6)),
        zoom_level: 2.0,
        ..Default::default()
    });
    let still = processor.process(frame(false)).await.unwrap().image;
    let mut dimensions = Some((10, 6));
    let mut images = Vec::new();
    for desired in [1.0, 2.0, 1.0] {
        images.push(
            process_clip_frame(
                &processor,
                frame(false),
                desired,
                &mut dimensions,
                SensorRotation::Rotate90,
                true,
            )
            .await
            .unwrap(),
        );
    }
    for image in &images {
        assert_eq!(image.dimensions(), (6, 10));
    }
    assert_ne!(images[0], images[1]);
    assert_eq!(images[0], images[2]);
    assert_eq!(processor.process(frame(false)).await.unwrap().image, still);
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
        2.0,
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
            2.0,
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
        2.0,
        &mut dimensions,
        SensorRotation::Rotate90,
        true,
    )
    .await
    .unwrap();
    let native = process_clip_frame(
        &processor,
        frame(true),
        2.0,
        &mut dimensions,
        SensorRotation::Rotate90,
        true,
    )
    .await
    .unwrap();
    assert_eq!(lag.dimensions(), (8, 12));
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
