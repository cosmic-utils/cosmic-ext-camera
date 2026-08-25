// SPDX-License-Identifier: GPL-3.0-only

//! Camera preview widget implementation

use crate::app::state::{AppModel, Message};
use crate::app::video_widget::{self, VideoContentFit};
use crate::backends::camera::types::SensorRotation;
use cosmic::Element;
use cosmic::iced::Length;
use cosmic::widget;
use tracing::{debug, info};

/// The preview transforms captured at the start of a blur transition.
///
/// While `VIDEO_ID_BLUR` is showing, the displayed frame is *frozen* — it
/// belongs to the old camera (or is a burst still), so live state must not
/// re-transform it. These are snapshotted at transition start and passed to
/// [`AppModel::preview_video_config`] as an override.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrozenPreviewTransforms {
    /// Sensor rotation of the camera that produced the frozen frame.
    pub rotation: SensorRotation,
    /// Whether the frozen frame was mirrored.
    pub mirror: bool,
    /// Digital zoom the frozen frame was last rendered at.
    pub zoom: f32,
}

impl AppModel {
    pub(crate) fn is_waiting_for_preview_frame(&self) -> bool {
        self.current_frame.is_none()
    }

    fn preview_loading_indicator(&self) -> Element<'_, Message> {
        widget::container(widget::progress_bar::indeterminate_circular())
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(cosmic::iced::alignment::Horizontal::Center)
            .align_y(cosmic::iced::alignment::Vertical::Center)
            .style(crate::app::overlay_style::window_bg_style)
            .into()
    }

    /// Whether the preview should be mirrored (front cameras only, not file sources)
    pub(crate) fn should_mirror_preview(&self) -> bool {
        let is_back = self
            .available_cameras
            .get(self.current_camera_index)
            .and_then(|c| c.camera_location.as_deref())
            == Some("back");
        self.config.mirror_preview && !self.current_frame_is_file_source && !is_back
    }

    /// Whether captured media (photo / video / timelapse) should be mirrored
    /// to match the preview. Only applies when the preview itself is mirrored
    /// AND the user has opted in via the `mirror_captures` setting.
    pub(crate) fn should_mirror_captures(&self) -> bool {
        self.config.mirror_captures && self.should_mirror_preview()
    }

    /// Whether the preview is currently painting the *frozen* blur frame —
    /// either a camera transition or HDR+ burst processing.
    pub(crate) fn preview_is_blurred(&self) -> bool {
        self.transition_state.should_blur()
            || self.burst_mode.stage == crate::app::state::BurstModeStage::Processing
    }

    /// The transforms every surface derived from `current_frame` must render
    /// with: the snapshot taken at blur-transition start while the frozen blur
    /// frame is showing, or `None` (live state) otherwise.
    ///
    /// The frosted backdrop blurs the *same* frame the preview does, so it must
    /// freeze with it — on BOTH paths. HDR+/burst keeps `current_frame`
    /// throughout. A camera switch clears it, but only until the new camera's
    /// first frame refills it, and `should_blur()` stays true for a further
    /// `blur_duration_ms` after that — so the backdrop renders, frozen, for the
    /// bulk of the switch too (against the snapshot the first frame re-took for
    /// the new camera, which is why the two surfaces still agree). Letting the
    /// backdrop track live zoom instead drifted the blurred slice against the
    /// preview for the whole transition.
    pub(crate) fn preview_transforms(&self) -> Option<FrozenPreviewTransforms> {
        self.preview_is_blurred()
            .then_some(FrozenPreviewTransforms {
                rotation: self.blur_frame_rotation,
                mirror: self.blur_frame_mirror,
                zoom: self.blur_frame_zoom,
            })
    }

    /// Build the `VideoWidgetConfig` describing the LIVE preview's transforms
    /// (cover/contain blend, crop, zoom, mirror, rotation, bar heights,
    /// letterbox colour) for `video_id`.
    ///
    /// Returns `None` when no frame is available. This is the single source of
    /// truth for the preview transforms so the frosted backdrop can reuse the
    /// exact same config and its blur lines up with the sharp preview.
    ///
    /// `frozen` overrides the sensor rotation / mirror / zoom with the values
    /// snapshotted when a blur transition started; pass `None` to use the
    /// current camera's live state. Callers should pass
    /// [`AppModel::preview_transforms`] rather than deciding for themselves, so
    /// the preview and the frosted backdrop never disagree about which frame
    /// they are transforming.
    pub fn preview_video_config(
        &self,
        video_id: u64,
        frozen: Option<FrozenPreviewTransforms>,
    ) -> Option<video_widget::VideoWidgetConfig> {
        let frame = self.current_frame.as_ref()?;

        let cover_blend = self.cover_blend();
        let filter_mode = self.selected_filter;

        let live = || FrozenPreviewTransforms {
            rotation: self.current_frame_rotation,
            mirror: self.should_mirror_preview(),
            zoom: self.current_zoom_level(),
        };
        let transforms = frozen.unwrap_or_else(live);
        // Compose the sensor mounting rotation with the compositor's display
        // transform so the preview reorients when the phone rotates between
        // portrait and landscape. The sensor's `Rotate*` is the rotation needed
        // to make the sensor upright when the display is at 0°; display
        // orientation is the rotation the display itself has from its natural
        // up. Summing both keeps the image upright relative to the user.
        //
        // Composing here rather than at the call site keeps the frosted backdrop
        // rotating in lockstep with the sharp preview.
        let rotation = self
            .preview_adjusted_rotation(transforms.rotation, transforms.mirror)
            .gpu_rotation_code();

        let crop_uv = match self.mode {
            crate::app::state::CameraMode::Photo if !self.current_frame_is_file_source => {
                self.photo_aspect_ratio.crop_uv(frame.width, frame.height)
            }
            _ => None,
        };

        let zoom_level = transforms.zoom;
        let scroll_zoom_enabled = self.mode.supports_fit_and_zoom();

        // The colour the blur chain's transform pass fills the letterbox with.
        //
        // RGB only: it is the colour the Kawase kernels blend *toward* at the
        // image's edge, so it has to match the window background the letterbox
        // actually shows — but it is never composited to screen as a fill any
        // more. The composite cuts the letterbox out entirely (see
        // `content_rect_uv` in `video_primitive`) and lets the one window
        // background layer paint it, so that the frosted backdrop cannot stack a
        // second opaque copy of this colour over a translucent window.
        //
        // Hence `bg_color()` rather than [`window_bg_color`]: this is the opaque
        // RGB of the background, with no alpha to carry, and pass 0 requires an
        // alpha of 1 regardless (the Kawase kernels read alpha as a region mask —
        // see `video_shader_kawase.wgsl`).
        let bg = cosmic::theme::active().cosmic().bg_color();
        let letterbox_color = [bg.red, bg.green, bg.blue, 1.0];

        Some(video_widget::VideoWidgetConfig {
            video_id,
            content_fit: VideoContentFit::Cover,
            filter_type: filter_mode,
            corner_radius: 0.0,
            mirror_horizontal: transforms.mirror,
            rotation,
            crop_uv,
            zoom_level,
            scroll_zoom_enabled,
            cover_blend: Some(cover_blend),
            insets: self.bar_insets(),
            letterbox_color,
        })
    }

    /// Build the camera preview widget
    ///
    /// Uses custom video widget with handle caching for optimized rendering.
    /// Shows a centered loading spinner until the first frame is available.
    /// Shows a blurred last frame during camera transitions.
    pub fn build_camera_preview(&self) -> Element<'_, Message> {
        if self.is_waiting_for_preview_frame() {
            return self.preview_loading_indicator();
        }

        // Build the main video preview (either current frame or placeholder)
        if let Some(frame) = &self.current_frame {
            static VIEW_FRAME_COUNT: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let count = VIEW_FRAME_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count.is_multiple_of(30) {
                debug!(
                    frame = count,
                    width = frame.width,
                    height = frame.height,
                    bytes = frame.data.len(),
                    "Rendering frame with video widget"
                );
            }

            // Use custom video widget with GPU primitive rendering
            // During transitions or HDR+ processing, use blur shader (video_id=1)
            let is_processing_hdr =
                self.burst_mode.stage == crate::app::state::BurstModeStage::Processing;
            let should_blur = self.preview_is_blurred();
            if should_blur && count.is_multiple_of(10) {
                let reason = if is_processing_hdr {
                    "HDR+ processing"
                } else {
                    "transition"
                };
                info!("Applying blur to frame during {}", reason);
            }
            let video_id = if should_blur {
                crate::app::video_primitive::VIDEO_ID_BLUR
            } else {
                crate::app::video_primitive::VIDEO_ID_NORMAL
            };

            // Every transform (frozen-vs-live state, cover blend, crop, bar
            // heights, letterbox colour) comes from the shared builders so the
            // frosted backdrop can never drift out of alignment with this.
            // It only returns `None` when there is no frame, already ruled out.
            if let Some(config) = self.preview_video_config(video_id, self.preview_transforms()) {
                let video_elem = video_widget::video_widget(frame.clone(), config);

                return widget::container(video_elem)
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .align_x(cosmic::iced::alignment::Horizontal::Center)
                    .align_y(cosmic::iced::alignment::Vertical::Center)
                    .into();
            }
        }

        static NO_FRAME_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let count = NO_FRAME_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count.is_multiple_of(30) {
            info!(render_count = count, "No frame available in view");
        }

        self.preview_loading_indicator()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::state::{BurstModeStage, CameraMode};
    use crate::app::video_primitive::{VIDEO_ID_BLUR, VIDEO_ID_FROSTED, VIDEO_ID_NORMAL};
    use crate::app::video_widget::VideoWidgetConfig;
    use crate::backends::display_orientation::DisplayOrientation;

    #[test]
    fn preview_spinner_is_shown_until_the_first_frame() {
        assert!(AppModel::default().is_waiting_for_preview_frame());
        assert!(!model().is_waiting_for_preview_frame());
    }

    #[test]
    fn preview_adjusted_rotation_keeps_derived_previews_upright() {
        let m = AppModel {
            display_orientation: DisplayOrientation::Rotate270,
            ..Default::default()
        };

        assert_eq!(
            m.preview_adjusted_rotation(SensorRotation::Rotate90, true),
            SensorRotation::None
        );
    }

    #[test]
    fn manual_controls_position_does_not_rotate_preview() {
        let bottom = AppModel::default();
        let side = AppModel {
            config: crate::config::Config {
                controls_position: crate::config::ControlsPosition::Left,
                ..Default::default()
            },
            ..Default::default()
        };

        assert_eq!(
            side.preview_adjusted_rotation(SensorRotation::Rotate90, false),
            bottom.preview_adjusted_rotation(SensorRotation::Rotate90, false)
        );
    }
    use crate::backends::camera::types::{CameraFrame, FrameData, PixelFormat};
    use std::sync::Arc;

    /// An `AppModel` with a frame, distinguishable live and frozen transforms,
    /// and nothing blurring — the baseline every test below perturbs.
    ///
    /// Live and frozen are deliberately DIFFERENT on all three axes (rotation,
    /// mirror, zoom), so a test that claims "frozen" cannot pass by accidentally
    /// reading live state, or the reverse.
    fn model() -> AppModel {
        let mut m = AppModel {
            current_frame: Some(Arc::new(CameraFrame {
                width: 1280,
                height: 960,
                data: FrameData::Copied(vec![0u8; 1280 * 960 * 4].into()),
                format: PixelFormat::RGBA,
                stride: 1280 * 4,
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: None,
            })),
            mode: CameraMode::Photo,
            current_frame_is_file_source: false,
            // Live transforms.
            current_frame_rotation: SensorRotation::Rotate90,
            zoom_level: 2.0,
            // Frozen transforms — different from live on every axis.
            blur_frame_rotation: SensorRotation::Rotate180,
            blur_frame_mirror: false,
            blur_frame_zoom: 3.5,
            ..Default::default()
        };
        m.config.mirror_preview = true;
        m
    }

    /// Nothing blurring: the preview renders LIVE state, so there is no frozen
    /// snapshot to impose.
    #[test]
    fn preview_transforms_are_live_when_not_blurred() {
        let m = model();
        assert!(!m.preview_is_blurred());
        assert_eq!(m.preview_transforms(), None);

        // And `None` really does resolve to live state, rather than to some
        // default: the config must carry the LIVE zoom/rotation, not the frozen
        // ones the model also holds.
        let config = m.preview_video_config(VIDEO_ID_NORMAL, None).unwrap();
        assert_eq!(config.zoom_level, 2.0);
        assert_eq!(
            config.rotation,
            SensorRotation::Rotate90.gpu_rotation_code()
        );
    }

    /// A camera transition freezes the transforms.
    ///
    /// The blurred frame belongs to the OLD camera, so re-transforming it with
    /// the new camera's live rotation/mirror/zoom would twist a still image that
    /// is not the new camera's at all.
    #[test]
    fn preview_transforms_freeze_during_a_transition() {
        let mut m = model();
        m.transition_state.in_transition = true;
        m.transition_state.first_frame_time = None;

        assert!(m.transition_state.should_blur());
        assert!(m.preview_is_blurred());
        let frozen = m.preview_transforms().expect("a transition must freeze");
        assert_eq!(frozen.rotation, SensorRotation::Rotate180);
        assert!(!frozen.mirror);
        assert_eq!(frozen.zoom, 3.5);

        // And the frozen snapshot reaches the config, overriding live state.
        let config = m
            .preview_video_config(VIDEO_ID_BLUR, m.preview_transforms())
            .unwrap();
        assert_eq!(
            config.zoom_level, 3.5,
            "the frozen frame must keep the zoom it was last displayed at; \
             tracking live zoom eases the STILL image instead of the preview"
        );
        assert_eq!(
            config.rotation,
            SensorRotation::Rotate180.gpu_rotation_code()
        );
        assert!(!config.mirror_horizontal);
    }

    /// HDR+/burst processing freezes them too — the other path onto the blur.
    ///
    /// This one is easy to lose: `should_blur()` is false throughout, so a
    /// `preview_transforms` that only consulted `transition_state` would sail
    /// through the transition test above and still drift on every HDR+ capture.
    #[test]
    fn preview_transforms_freeze_during_burst_processing() {
        let mut m = model();
        m.burst_mode.stage = BurstModeStage::Processing;

        assert!(
            !m.transition_state.should_blur(),
            "this test is only meaningful while the transition path is quiet"
        );
        assert!(m.preview_is_blurred());
        let frozen = m
            .preview_transforms()
            .expect("HDR+ processing must freeze the transforms");
        assert_eq!(frozen.zoom, 3.5);
        assert_eq!(frozen.rotation, SensorRotation::Rotate180);
    }

    /// The other burst stages do NOT blur — `Capturing` in particular, where the
    /// preview is still live.
    #[test]
    fn preview_is_not_blurred_in_other_burst_stages() {
        for stage in [
            BurstModeStage::Idle,
            BurstModeStage::Capturing,
            BurstModeStage::Complete,
            BurstModeStage::Error,
        ] {
            let mut m = model();
            m.burst_mode.stage = stage;
            assert!(!m.preview_is_blurred(), "{stage:?} must not blur");
            assert_eq!(m.preview_transforms(), None, "{stage:?}");
        }
    }

    /// THE invariant this branch exists to hold: the preview and the frosted
    /// backdrop resolve to the SAME transforms, in every blur state.
    ///
    /// # Why this is the test that matters
    ///
    /// The two surfaces are stacked on each other — the backdrop blurs the very
    /// frame the preview draws sharp — so any disagreement about rotation,
    /// mirror, zoom, crop, fit or bar heights shows up directly as a blurred
    /// slice sliding against the sharp preview under it. That is not
    /// hypothetical: it is the zoom-drift bug this branch fixed (c2dcf0c), where
    /// the backdrop tracked live zoom while the preview held the frozen one, and
    /// the two drifted apart for the whole transition.
    ///
    /// Both callers reach `preview_video_config(id, preview_transforms())` —
    /// `build_camera_preview` here, `frosted_panel`/`frosted_bars` in
    /// `overlay_style.rs`. The agreement IS the fix, so it is pinned by
    /// comparing the WHOLE config rather than the fields that happen to be
    /// interesting today: a new transform field that only one surface honoured
    /// would fail this without anyone having to remember to extend it.
    #[test]
    fn the_preview_and_the_backdrop_always_agree() {
        let live = || (model(), VIDEO_ID_NORMAL);
        let transition = || {
            let mut m = model();
            m.transition_state.in_transition = true;
            (m, VIDEO_ID_BLUR)
        };
        let burst = || {
            let mut m = model();
            m.burst_mode.stage = BurstModeStage::Processing;
            (m, VIDEO_ID_BLUR)
        };

        for (name, (m, preview_id)) in [
            ("live", live()),
            ("transition", transition()),
            ("burst processing", burst()),
        ] {
            // Exactly what `build_camera_preview` builds...
            let preview = m
                .preview_video_config(preview_id, m.preview_transforms())
                .unwrap();
            // ...and exactly what `frosted_panel` / `frosted_bars` build.
            let backdrop = m
                .preview_video_config(VIDEO_ID_FROSTED, m.preview_transforms())
                .unwrap();

            // The video_id is the ONE field they are supposed to differ on: it is
            // what routes the backdrop down the blur chain.
            assert_eq!(backdrop.video_id, VIDEO_ID_FROSTED);
            assert_eq!(preview.video_id, preview_id);

            let normalized = VideoWidgetConfig {
                video_id: preview_id,
                ..backdrop
            };
            assert_eq!(
                normalized, preview,
                "in the {name} state the frosted backdrop and the sharp preview \
                 resolved to DIFFERENT transforms. They are stacked on each other \
                 over the same frame, so the blurred slice will slide against the \
                 preview — this is the black-bars / zoom-drift class of bug."
            );
        }
    }

    /// The frozen-vs-live choice is MATERIAL — which is what makes
    /// `preview_transforms()` load-bearing rather than decoration.
    ///
    /// `the_preview_and_the_backdrop_always_agree` can only police
    /// `preview_video_config` itself; the historical bug lived one level up, in
    /// the CALLERS — the backdrop decided its own transforms and chose live ones
    /// while the preview held the frozen snapshot. So this pins the other half:
    /// during a transition, `None` and `preview_transforms()` produce genuinely
    /// different configs. A caller that passes the wrong one is therefore
    /// choosing a visibly different image, and `preview_transforms()` is the only
    /// reason both surfaces pick the same one.
    ///
    /// If this ever passes vacuously — i.e. the two configs come out equal — the
    /// agreement test above has quietly lost its meaning too.
    #[test]
    fn the_frozen_and_live_transforms_actually_differ() {
        let mut m = model();
        m.transition_state.in_transition = true;

        let frozen = m
            .preview_video_config(VIDEO_ID_FROSTED, m.preview_transforms())
            .unwrap();
        let live = m.preview_video_config(VIDEO_ID_FROSTED, None).unwrap();

        assert_ne!(
            frozen, live,
            "the frozen snapshot and live state must differ during a transition, \
             otherwise every test asserting the two surfaces agree is vacuous"
        );
        assert_eq!(frozen.zoom_level, 3.5);
        assert_eq!(live.zoom_level, 2.0);
    }

    /// No frame, no config — for either surface. The backdrop's callers rely on
    /// this to fall back to a plain translucent panel rather than blurring
    /// nothing.
    #[test]
    fn preview_video_config_needs_a_frame() {
        let mut m = model();
        m.current_frame = None;
        assert!(m.preview_video_config(VIDEO_ID_NORMAL, None).is_none());
        assert!(m.preview_video_config(VIDEO_ID_FROSTED, None).is_none());
    }
}
