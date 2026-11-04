// SPDX-License-Identifier: GPL-3.0-only

use crate::backends::camera::types::CameraFrame;

type CropRect = (u32, u32, u32, u32);

/// Final HDR export region, after merging full-resolution source frames.
pub(crate) fn burst_zoom_crop(
    frame: &CameraFrame,
    crop: Option<CropRect>,
    desired_zoom: f32,
) -> Result<Option<CropRect>, String> {
    use crate::pipelines::photo::{PostProcessingConfig, PostProcessor};
    let config = PostProcessingConfig {
        crop_rect: crop,
        zoom_level: crate::backends::camera::zoom::residual_zoom(desired_zoom, frame),
        ..Default::default()
    };
    let region = PostProcessor::final_region(frame.width, frame.height, &config)?;
    Ok((region != (0, 0, frame.width, frame.height)).then_some(region))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::state::{AppModel, BurstModeState};
    use crate::backends::camera::types::{FrameData, FrameMetadata, PixelFormat};
    use crate::backends::camera::zoom::{ScalerCropBounds, ZoomRect};
    use std::sync::Arc;

    fn frame(applied_zoom: Option<f32>) -> Arc<CameraFrame> {
        let base = ZoomRect {
            x: 0,
            y: 0,
            width: 4000,
            height: 3000,
        };
        let min = ZoomRect {
            x: 0,
            y: 0,
            width: 100,
            height: 100,
        };
        let bounds = ScalerCropBounds::new(min, base, base, base, false).unwrap();
        Arc::new(CameraFrame {
            width: 400,
            height: 300,
            data: FrameData::from_owned_vec(vec![0; 400 * 300 * 4]),
            format: PixelFormat::RGBA,
            stride: 400 * 4,
            yuv_planes: None,
            captured_at: std::time::Instant::now(),
            sensor_timestamp_ns: None,
            libcamera_metadata: Some(FrameMetadata {
                scaler_zoom: applied_zoom.and_then(|zoom| bounds.applied(Some(bounds.crop(zoom)))),
                ..Default::default()
            }),
        })
    }

    #[test]
    fn hdr_crop_uses_residual_zoom_about_the_full_frame_pivot() {
        let source = frame(None);
        assert_eq!(
            burst_zoom_crop(&source, Some((40, 20, 320, 240)), 2.0).unwrap(),
            Some((120, 85, 160, 120))
        );
        let native = frame(Some(2.0));
        assert_eq!(
            burst_zoom_crop(&native, Some((40, 20, 320, 240)), 2.0).unwrap(),
            Some((40, 20, 320, 240))
        );
        assert_eq!(burst_zoom_crop(&native, None, 2.0).unwrap(), None);
        assert_eq!(
            burst_zoom_crop(&native, None, 4.0).unwrap(),
            Some((100, 75, 200, 150))
        );
    }

    #[test]
    fn hdr_collection_never_merges_different_applied_crops() {
        let mut burst = BurstModeState::default();
        burst.capture_zoom = 2.0;
        burst.start_capture(3);
        assert!(!burst.add_frame(frame(Some(1.0))));
        assert!(!burst.add_frame(frame(Some(2.0))));
        assert_eq!(
            burst.frames_captured(),
            1,
            "old crop must not enter this merge"
        );
        assert!(!burst.add_frame(frame(Some(2.0))));
        assert!(burst.add_frame(frame(Some(2.0))));
    }

    #[test]
    fn new_applied_crop_invalidates_queued_qr_results() {
        let mut model = AppModel {
            current_frame: Some(frame(Some(1.0))),
            exposure_settings: Some(Default::default()),
            ..Default::default()
        };
        model.config.motion_photo = false;
        let generation = model
            .qr_detection_scheduler
            .try_start(true, std::time::Instant::now())
            .unwrap();
        let _ = model.handle_camera_frame(frame(Some(2.0)));
        assert!(
            !model
                .qr_detection_scheduler
                .finish(generation, std::time::Instant::now())
        );
    }

    #[test]
    fn hdr_acquisition_keeps_zoom_fixed_while_video_remains_interactive() {
        let mut model = AppModel {
            zoom_level: 2.0,
            ..Default::default()
        };
        model.burst_mode.start_capture(3);
        let _ = model.handle_pinch_zoom(4.0);
        assert_eq!(model.zoom_level, 2.0);
        model.burst_mode.reset();
        model.mode = crate::app::CameraMode::Video;
        let _ = model.handle_pinch_zoom(4.0);
        assert_eq!(model.zoom_level, 4.0);
        assert_eq!(model.zoom_control.zoom(), 4.0);
    }

    #[test]
    fn queued_old_camera_frames_never_replace_the_frozen_transition_frame() {
        let frozen = frame(Some(2.0));
        let mut model = AppModel {
            current_frame: Some(Arc::clone(&frozen)),
            exposure_settings: Some(Default::default()),
            ..Default::default()
        };
        model.config.motion_photo = false;
        model.transition_state.start();
        let mut queued = (*frame(Some(1.0))).clone();
        queued.captured_at = model.transition_state.transition_start_time.unwrap()
            - std::time::Duration::from_millis(1);
        let _ = model.handle_camera_frame(Arc::new(queued));
        assert!(Arc::ptr_eq(model.current_frame.as_ref().unwrap(), &frozen));
    }

    #[test]
    fn processed_shutter_waits_for_zoom_out_to_supply_missing_pixels() {
        let mut model = AppModel {
            current_frame: Some(frame(Some(4.0))),
            zoom_level: 1.0,
            ..Default::default()
        };
        assert!(
            !model.can_accept_photo(),
            "old 4x cannot represent a new 1x shutter"
        );
        model.current_frame = Some(frame(Some(1.0)));
        assert!(model.can_accept_photo());
        model.zoom_level = 4.0;
        model.current_frame = Some(frame(Some(2.0)));
        assert!(
            model.can_accept_photo(),
            "software can supply the remaining zoom-in"
        );
    }

    #[test]
    fn hdr_rejects_identical_old_crops_that_exceed_shutter_zoom() {
        let mut burst = BurstModeState::default();
        burst.start_capture(3);
        for _ in 0..3 {
            assert!(!burst.add_frame(frame(Some(4.0))));
        }
        assert_eq!(burst.frames_captured(), 0);
        assert!(!burst.add_frame(frame(Some(1.0))));
        assert!(!burst.add_frame(frame(Some(1.0))));
        assert!(burst.add_frame(frame(Some(1.0))));
    }
}
