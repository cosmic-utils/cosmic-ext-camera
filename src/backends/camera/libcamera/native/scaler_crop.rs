// SPDX-License-Identifier: GPL-3.0-only

use crate::backends::camera::zoom::{FrameScalerZoom, ScalerCropBounds, ZoomControl, ZoomRect};
use libcamera::{control::ControlList, controls, geometry::Rectangle};

#[derive(Default)]
pub(super) struct NativeScalerCrop {
    bounds: Option<ScalerCropBounds>,
    baseline_verified: bool,
    zoom_submitted: bool,
    warned_unverified: bool,
}

fn zoom_rect(rect: Rectangle) -> ZoomRect {
    ZoomRect {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
    }
}

impl NativeScalerCrop {
    fn new(bounds: Option<ScalerCropBounds>) -> Self {
        Self {
            bounds,
            ..Default::default()
        }
    }

    /// Read only after configure: capability rectangles and properties may change per mode.
    pub(super) fn discover(
        cam: &libcamera::camera::Camera<'_>,
        primary_is_raw: bool,
        width: u32,
        height: u32,
    ) -> Self {
        use libcamera::{control::ControlEntry, properties};
        if primary_is_raw {
            return Self::default();
        }
        let info = cam.controls().find(controls::ScalerCrop::ID).ok();
        let mut scaler = Self::from_values(
            info.map(|i| i.min()),
            info.map(|i| i.max()),
            info.map(|i| i.def()),
            cam.properties()
                .get::<properties::ScalerCropMaximum>()
                .ok()
                .map(|p| p.0.into()),
            primary_is_raw,
        );
        scaler.bounds = scaler
            .bounds
            .and_then(|b| b.with_output_aspect(width, height));
        scaler
    }

    fn from_values(
        min: Option<libcamera::control_value::ControlValue>,
        max: Option<libcamera::control_value::ControlValue>,
        default: Option<libcamera::control_value::ControlValue>,
        maximum: Option<libcamera::control_value::ControlValue>,
        primary_is_raw: bool,
    ) -> Self {
        let parse = |value| Rectangle::try_from(value).ok().map(zoom_rect);
        Self::new((|| {
            ScalerCropBounds::new(
                parse(min?)?,
                parse(max?)?,
                parse(default?)?,
                parse(maximum?)?,
                primary_is_raw,
            )
        })())
    }

    /// Call for every submission, after reuse has cleared request controls.
    pub(super) fn apply(&mut self, controls: &mut ControlList, desired: &ZoomControl) {
        if let Some(bounds) = self.bounds {
            let crop = bounds.crop(if self.baseline_verified {
                desired.zoom()
            } else {
                1.0
            });
            self.zoom_submitted |= crop != bounds.crop(1.0);
            let _ = controls.set(controls::ScalerCrop(Rectangle {
                x: crop.x,
                y: crop.y,
                width: crop.width,
                height: crop.height,
            }));
        }
    }

    pub(super) fn may_dispatch_processed(&mut self, evidence: Option<FrameScalerZoom>) -> bool {
        let verified = !self.zoom_submitted || evidence.is_some();
        if !verified && !self.warned_unverified {
            tracing::warn!(
                "Dropping processed frames without valid ScalerCrop metadata after native zoom activation"
            );
            self.warned_unverified = true;
        }
        verified
    }

    /// The caller supplies completed-request metadata, never requested controls.
    pub(super) fn completed(&mut self, metadata: &ControlList) -> Option<FrameScalerZoom> {
        let bounds = self.bounds?;
        let crop = metadata
            .get::<controls::ScalerCrop>()
            .ok()
            .map(|c| zoom_rect(c.0));
        let evidence = bounds.applied(crop);
        if crop == Some(bounds.crop(1.0)) {
            self.baseline_verified = true;
        }
        evidence
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::camera::zoom::{ScalerCropBounds, ZoomControl, ZoomRect};
    use libcamera::{control::ControlList, controls, geometry::Rectangle};

    #[test]
    fn queue_controls_are_refreshed_after_clear_without_claiming_applied_zoom() {
        let base = ZoomRect {
            x: 100,
            y: 200,
            width: 4000,
            height: 3000,
        };
        let min = ZoomRect {
            width: 100,
            height: 100,
            ..base
        };
        let mut scaler = NativeScalerCrop::new(ScalerCropBounds::new(min, base, base, base, false));
        let zoom = ZoomControl::default();
        let mut requested = ControlList::new();
        let mut metadata = ControlList::new();
        metadata
            .set(controls::ScalerCrop(Rectangle {
                x: base.x,
                y: base.y,
                width: base.width,
                height: base.height,
            }))
            .unwrap();
        assert!(scaler.completed(&metadata).is_some());
        metadata.clear();
        zoom.set_zoom(2.0);
        scaler.apply(&mut requested, &zoom);
        let crop = requested.get::<controls::ScalerCrop>().unwrap().0;
        assert_eq!(
            (crop.x, crop.y, crop.width, crop.height),
            (1100, 950, 2000, 1500)
        );
        assert!(scaler.completed(&metadata).is_none());
        metadata.set(controls::ScalerCrop(crop)).unwrap();
        assert_eq!(scaler.completed(&metadata).unwrap().applied_zoom(), 2.0);
        requested.clear(); // Request::reuse clears its control list.
        zoom.set_zoom(1.0);
        scaler.apply(&mut requested, &zoom);
        assert_eq!(
            requested.get::<controls::ScalerCrop>().unwrap().0.width,
            4000
        );
        metadata.clear();
        metadata
            .set(controls::ScalerCrop(Rectangle { width: 0, ..crop }))
            .unwrap();
        assert!(scaler.completed(&metadata).is_none());
    }

    #[test]
    fn missing_metadata_after_zoom_submission_blocks_only_unverified_processed_frames() {
        let base = ZoomRect {
            x: 0,
            y: 0,
            width: 4000,
            height: 3000,
        };
        let min = ZoomRect {
            width: 100,
            height: 100,
            ..base
        };
        let mut scaler = NativeScalerCrop::new(ScalerCropBounds::new(min, base, base, base, false));
        assert!(scaler.may_dispatch_processed(None));
        let zoom = ZoomControl::default();
        let mut metadata = ControlList::new();
        let mut requested = ControlList::new();
        metadata
            .set(controls::ScalerCrop(Rectangle {
                x: 0,
                y: 0,
                width: 4000,
                height: 3000,
            }))
            .unwrap();
        let baseline = scaler.completed(&metadata);
        assert!(scaler.may_dispatch_processed(baseline));
        zoom.set_zoom(2.0);
        scaler.apply(&mut requested, &zoom);
        assert!(!scaler.may_dispatch_processed(None));
        metadata
            .set(requested.get::<controls::ScalerCrop>().unwrap())
            .unwrap();
        let evidence = scaler.completed(&metadata);
        assert!(scaler.may_dispatch_processed(evidence));
        zoom.set_zoom(1.0);
        scaler.apply(&mut requested, &zoom);
        assert!(
            !scaler.may_dispatch_processed(None),
            "reset submission is not applied acknowledgement"
        );
        let mut next_generation =
            NativeScalerCrop::new(ScalerCropBounds::new(min, base, base, base, false));
        assert!(next_generation.may_dispatch_processed(None));
        next_generation.apply(&mut requested, &zoom);
        assert_eq!(
            requested.get::<controls::ScalerCrop>().unwrap().0.width,
            4000
        );
    }

    #[test]
    fn advertised_without_baseline_metadata_never_requests_zoom() {
        let base = ZoomRect {
            x: 100,
            y: 200,
            width: 4000,
            height: 3000,
        };
        let min = ZoomRect {
            width: 100,
            height: 100,
            ..base
        };
        let mut scaler = NativeScalerCrop::new(ScalerCropBounds::new(min, base, base, base, false));
        let zoom = ZoomControl::default();
        zoom.set_zoom(3.0);
        let mut requested = ControlList::new();
        let empty_metadata = ControlList::new();
        for _ in 0..5 {
            assert!(scaler.completed(&empty_metadata).is_none());
            requested.clear();
            scaler.apply(&mut requested, &zoom);
            assert_eq!(
                requested.get::<controls::ScalerCrop>().unwrap().0.width,
                4000
            );
        }
    }

    #[test]
    fn configured_capability_values_require_all_rectangles_and_processed_primary() {
        use libcamera::control_value::ControlValue;
        let base = Rectangle {
            x: 200,
            y: 400,
            width: 3200,
            height: 1800,
        };
        let min = Rectangle {
            x: 0,
            y: 0,
            width: 100,
            height: 100,
        };
        let build = |raw, minimum| {
            NativeScalerCrop::from_values(
                minimum,
                Some(base.into()),
                Some(base.into()),
                Some(base.into()),
                raw,
            )
        };
        let mut list = ControlList::new();
        let zoom = ZoomControl::default();
        zoom.set_zoom(2.0);
        let mut supported = build(false, Some(min.into()));
        list.set(controls::ScalerCrop(base)).unwrap();
        supported.completed(&list);
        list.clear();
        supported.apply(&mut list, &zoom);
        assert_eq!(list.get::<controls::ScalerCrop>().unwrap().0.width, 1600);
        for mut scaler in [
            build(true, Some(min.into())),
            build(false, None),
            build(false, Some(ControlValue::from(true))),
        ] {
            list.clear();
            scaler.apply(&mut list, &zoom);
            assert!(list.is_empty());
        }
    }

    #[test]
    fn absent_capabilities_do_not_submit_or_acknowledge_scaler_crop() {
        let mut scaler = NativeScalerCrop::new(None);
        let zoom = ZoomControl::default();
        zoom.set_zoom(4.0);
        let mut requested = ControlList::new();
        scaler.apply(&mut requested, &zoom);
        assert!(requested.get::<controls::ScalerCrop>().is_err());
        requested
            .set(controls::ScalerCrop(Rectangle {
                x: 0,
                y: 0,
                width: 100,
                height: 100,
            }))
            .unwrap();
        assert!(scaler.completed(&requested).is_none());
    }
}
