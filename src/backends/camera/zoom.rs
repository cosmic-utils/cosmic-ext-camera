// SPDX-License-Identifier: GPL-3.0-only

use std::sync::atomic::{AtomicU32, Ordering};

/// Desired zoom only. Capabilities and applied state belong to each capture generation.
#[derive(Debug)]
pub struct ZoomControl(AtomicU32);

impl Default for ZoomControl {
    fn default() -> Self {
        Self(AtomicU32::new(1.0_f32.to_bits()))
    }
}

fn valid_zoom(zoom: f32) -> f32 {
    if zoom.is_finite() { zoom.max(1.0) } else { 1.0 }
}

impl ZoomControl {
    pub fn set_zoom(&self, zoom: f32) {
        self.0.store(valid_zoom(zoom).to_bits(), Ordering::Relaxed);
    }

    pub fn zoom(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }
}

/// Remaining software crop. Only completed processed-frame evidence removes zoom.
pub(crate) fn residual_zoom(
    desired: f32,
    frame: &crate::backends::camera::types::CameraFrame,
) -> f32 {
    let desired = valid_zoom(desired);
    if frame.format.is_bayer() {
        return desired;
    }
    frame
        .libcamera_metadata
        .as_ref()
        .and_then(|m| m.scaler_zoom)
        .map_or(desired, |applied| applied.residual_zoom(desired))
}

/// Software can add magnification but cannot restore pixels discarded by an
/// earlier native crop while zoom-out requests are still in flight.
pub(crate) fn frame_can_represent_zoom(
    desired: f32,
    frame: &crate::backends::camera::types::CameraFrame,
) -> bool {
    if frame.format.is_bayer() {
        return true;
    }
    let desired = f64::from(valid_zoom(desired));
    frame
        .libcamera_metadata
        .as_ref()
        .and_then(|m| m.scaler_zoom)
        .is_none_or(|applied| {
            f64::from(applied.base.width) <= f64::from(applied.crop.width) * desired
                && f64::from(applied.base.height) <= f64::from(applied.crop.height) * desired
        })
}

/// Sensor-coordinate rectangle; independent of stream/output pixel dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZoomRect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl ZoomRect {
    fn valid(self) -> bool {
        self.width > 0
            && self.height > 0
            && i64::from(self.x) + i64::from(self.width) <= i64::from(i32::MAX)
            && i64::from(self.y) + i64::from(self.height) <= i64::from(i32::MAX)
    }

    fn contains(self, other: Self) -> bool {
        self.valid()
            && other.valid()
            && other.x >= self.x
            && other.y >= self.y
            && i64::from(other.x) + i64::from(other.width)
                <= i64::from(self.x) + i64::from(self.width)
            && i64::from(other.y) + i64::from(other.height)
                <= i64::from(self.y) + i64::from(self.height)
    }
}

/// Immutable, post-configure capability snapshot. Never shared across generations.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScalerCropBounds {
    base: ZoomRect,
    min_width: u32,
    min_height: u32,
}

/// Evidence copied only from a completed processed request's ScalerCrop metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameScalerZoom {
    base: ZoomRect,
    crop: ZoomRect,
}

impl FrameScalerZoom {
    /// Configured 1x image region in native sensor coordinates.
    pub fn base(self) -> ZoomRect {
        self.base
    }
    /// Applied image region reported by the completed request.
    pub fn crop_rect(self) -> ZoomRect {
        self.crop
    }

    pub fn applied_zoom(self) -> f32 {
        (self.base.width as f32 / self.crop.width as f32)
            .min(self.base.height as f32 / self.crop.height as f32)
            .max(1.0)
    }

    pub(crate) fn residual_zoom(self, desired: f32) -> f32 {
        (valid_zoom(desired) / self.applied_zoom()).max(1.0)
    }
}

impl ScalerCropBounds {
    pub(crate) fn with_output_aspect(mut self, width: u32, height: u32) -> Option<Self> {
        if width == 0 || height == 0 {
            return None;
        }
        let (w, h) = if u64::from(self.base.width) * u64::from(height)
            > u64::from(self.base.height) * u64::from(width)
        {
            (
                (u64::from(self.base.height) * u64::from(width) / u64::from(height)) as u32,
                self.base.height,
            )
        } else {
            (
                self.base.width,
                (u64::from(self.base.width) * u64::from(height) / u64::from(width)) as u32,
            )
        };
        if w < self.min_width || h < self.min_height {
            return None;
        }
        self.base.x = (i64::from(self.base.x) + i64::from((self.base.width - w) / 2)) as i32;
        self.base.y = (i64::from(self.base.y) + i64::from((self.base.height - h) / 2)) as i32;
        self.base.width = w;
        self.base.height = h;
        Some(self)
    }

    pub(crate) fn applied(self, completed_crop: Option<ZoomRect>) -> Option<FrameScalerZoom> {
        let crop = completed_crop?;
        // Scalar residual zoom is only valid for centered, uniform crops. Permit
        // the single-pixel rounding inherent in integer-centered rectangles.
        let center_x = (2 * (i64::from(crop.x) - i64::from(self.base.x)) + i64::from(crop.width)
            - i64::from(self.base.width))
        .abs();
        let center_y = (2 * (i64::from(crop.y) - i64::from(self.base.y)) + i64::from(crop.height)
            - i64::from(self.base.height))
        .abs();
        let aspect_error = (i128::from(crop.width) * i128::from(self.base.height)
            - i128::from(crop.height) * i128::from(self.base.width))
        .abs();
        (self.base.contains(crop)
            && crop.width >= self.min_width
            && crop.height >= self.min_height
            && center_x <= 1
            && center_y <= 1
            && aspect_error <= i128::from(self.base.width.max(self.base.height)))
        .then_some(FrameScalerZoom {
            base: self.base,
            crop,
        })
    }

    pub(crate) fn new(
        min: ZoomRect,
        max: ZoomRect,
        base: ZoomRect,
        maximum: ZoomRect,
        primary_is_raw: bool,
    ) -> Option<Self> {
        if primary_is_raw
            || !max.contains(base)
            || !maximum.contains(base)
            || min.width == 0
            || min.height == 0
            || min.width > base.width
            || min.height > base.height
        {
            return None;
        }
        Some(Self {
            base,
            min_width: min.width,
            min_height: min.height,
        })
    }

    pub(crate) fn crop(self, desired: f32) -> ZoomRect {
        let limit = (self.base.width as f64 / self.min_width as f64)
            .min(self.base.height as f64 / self.min_height as f64);
        let zoom = f64::from(valid_zoom(desired)).min(limit);
        let width = (self.base.width as f64 / zoom).ceil() as u32;
        let height = (self.base.height as f64 / zoom).ceil() as u32;
        ZoomRect {
            x: (i64::from(self.base.x) + i64::from((self.base.width - width) / 2)) as i32,
            y: (i64::from(self.base.y) + i64::from((self.base.height - height) / 2)) as i32,
            width,
            height,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: i32, y: i32, width: u32, height: u32) -> ZoomRect {
        ZoomRect {
            x,
            y,
            width,
            height,
        }
    }

    #[test]
    fn centered_crop_honors_default_fov_origin_and_minimum_size() {
        let base = rect(200, 400, 3200, 1800);
        let bounds = ScalerCropBounds::new(
            rect(100, 200, 640, 480),
            rect(100, 200, 4000, 3000),
            base,
            rect(0, 0, 5000, 4000),
            false,
        )
        .unwrap();
        assert_eq!(bounds.crop(1.0), base);
        assert_eq!(bounds.crop(2.0), rect(1000, 850, 1600, 900));
        let limited = bounds.crop(100.0);
        assert!(limited.width >= 640 && limited.height >= 480);
        assert!((limited.width as f32 / limited.height as f32 - 3200.0 / 1800.0).abs() < 0.004);
        assert_eq!(bounds.crop(f32::NAN), base);
        assert!(ScalerCropBounds::new(rect(0, 0, 1, 1), base, base, base, true).is_none());
        assert!(
            ScalerCropBounds::new(rect(0, 0, 1, 1), base, rect(0, 0, 0, 0), base, false).is_none()
        );
    }

    #[test]
    fn baseline_matches_output_aspect_without_changing_visible_fov() {
        let base = rect(100, 200, 4000, 3000);
        let bounds = ScalerCropBounds::new(rect(0, 0, 100, 100), base, base, base, false).unwrap();
        let wide = bounds.with_output_aspect(1920, 1080).unwrap();
        assert_eq!(wide.crop(1.0), rect(100, 575, 4000, 2250));
        assert_eq!(wide.crop(2.0), rect(1100, 1137, 2000, 1125));
        assert!(bounds.with_output_aspect(0, 1080).is_none());
        let tiny = ScalerCropBounds::new(rect(0, 0, 3900, 2900), base, base, base, false).unwrap();
        assert!(tiny.with_output_aspect(1920, 1080).is_none());
    }

    #[test]
    fn completed_crop_evidence_accounts_for_lag_without_double_zoom() {
        let base = rect(100, 200, 4000, 3000);
        let bounds = ScalerCropBounds::new(rect(0, 0, 100, 100), base, base, base, false).unwrap();
        assert!(
            bounds.applied(None).is_none(),
            "requested storage is not evidence"
        );
        let applied = bounds.applied(Some(rect(1100, 950, 2000, 1500))).unwrap();
        assert_eq!(applied.applied_zoom(), 2.0);
        assert_eq!(applied.residual_zoom(4.0), 2.0);
        assert_eq!(applied.residual_zoom(2.0), 1.0);
        assert_eq!(applied.residual_zoom(1.0), 1.0);
        assert!(bounds.applied(Some(rect(0, 0, 0, 0))).is_none());
        assert!(bounds.applied(Some(rect(5000, 200, 1000, 1000))).is_none());
    }

    #[test]
    fn malformed_or_nonuniform_completed_crop_cannot_acknowledge_scalar_zoom() {
        let base = rect(100, 200, 4000, 3000);
        let bounds = ScalerCropBounds::new(rect(0, 0, 100, 100), base, base, base, false).unwrap();
        for crop in [
            rect(2050, 1675, 99, 74),
            rect(1100, 1200, 2000, 1000),
            rect(1101, 950, 2000, 1500),
        ] {
            assert!(
                bounds.applied(Some(crop)).is_none(),
                "invalid evidence: {crop:?}"
            );
        }
    }

    #[test]
    fn residual_frame_zoom_requires_evidence_and_ignores_bayer() {
        use crate::backends::camera::types::{CameraFrame, FrameData, FrameMetadata, PixelFormat};
        let mut frame = CameraFrame {
            width: 2,
            height: 2,
            format: PixelFormat::RGBA,
            stride: 8,
            data: FrameData::from_owned_vec(vec![0; 16]),
            yuv_planes: None,
            captured_at: std::time::Instant::now(),
            sensor_timestamp_ns: None,
            libcamera_metadata: None,
        };
        assert_eq!(residual_zoom(4.0, &frame), 4.0);
        let base = rect(0, 0, 4000, 3000);
        let bounds = ScalerCropBounds::new(rect(0, 0, 100, 100), base, base, base, false).unwrap();
        frame.libcamera_metadata = Some(FrameMetadata {
            scaler_zoom: bounds.applied(Some(rect(1000, 750, 2000, 1500))),
            ..Default::default()
        });
        assert_eq!(residual_zoom(4.0, &frame), 2.0);
        assert_eq!(residual_zoom(2.0, &frame), 1.0);
        frame.format = PixelFormat::BayerRGGB;
        assert_eq!(residual_zoom(4.0, &frame), 4.0);
    }

    #[test]
    fn desired_zoom_defaults_resets_and_rejects_invalid_values() {
        let zoom = ZoomControl::default();
        assert_eq!(zoom.zoom(), 1.0);
        zoom.set_zoom(3.0);
        assert_eq!(zoom.zoom(), 3.0);
        for invalid in [0.0, -2.0, f32::NAN, f32::INFINITY] {
            zoom.set_zoom(invalid);
            assert_eq!(zoom.zoom(), 1.0);
        }
        zoom.set_zoom(2.0);
        zoom.set_zoom(1.0);
        assert_eq!(zoom.zoom(), 1.0);
    }
}
