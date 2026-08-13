// SPDX-License-Identifier: GPL-3.0-only

use crate::constants::BitratePreset;
use cosmic::cosmic_config::{self, CosmicConfigEntry, cosmic_config_derive::CosmicConfigEntry};
use cosmic::{Theme, theme};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Preferred location of the capture-controls panel.
///
/// `Bottom` keeps the existing orientation-aware behavior: the panel follows
/// the physical bottom edge reported by the compositor. `Left` and `Right`
/// are explicit window-edge overrides for desktops and devices without useful
/// orientation data.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum ControlsPosition {
    #[default]
    Bottom,
    Left,
    Right,
}

impl ControlsPosition {
    pub const ALL: [Self; 3] = [Self::Bottom, Self::Left, Self::Right];
}

/// Photo output format preference
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum PhotoOutputFormat {
    /// JPEG format (lossy, smaller files)
    #[default]
    Jpeg,
    /// PNG format (lossless, larger files)
    Png,
    /// DNG format (raw image data)
    Dng,
}

impl PhotoOutputFormat {
    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            PhotoOutputFormat::Jpeg => "jpg",
            PhotoOutputFormat::Png => "png",
            PhotoOutputFormat::Dng => "dng",
        }
    }

    /// Get display name for this format
    pub fn display_name(&self) -> &'static str {
        match self {
            PhotoOutputFormat::Jpeg => "JPEG",
            PhotoOutputFormat::Png => "PNG",
            PhotoOutputFormat::Dng => "DNG (Raw)",
        }
    }

    /// Get all available formats
    pub const ALL: [PhotoOutputFormat; 3] = [
        PhotoOutputFormat::Jpeg,
        PhotoOutputFormat::Png,
        PhotoOutputFormat::Dng,
    ];
}

/// Burst mode setting
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum BurstModeSetting {
    /// Burst mode disabled (default - experimental feature)
    #[default]
    Off,
    /// Auto-detect frame count based on scene brightness
    Auto,
    /// Fixed 4 frames
    Frames4,
    /// Fixed 6 frames
    Frames6,
    /// Fixed 8 frames
    Frames8,
    /// Fixed 50 frames
    Frames50,
}

impl BurstModeSetting {
    /// Check if burst mode is enabled (not Off)
    pub fn is_enabled(&self) -> bool {
        !matches!(self, BurstModeSetting::Off)
    }

    /// Get the fixed frame count, if any
    pub fn frame_count(&self) -> Option<usize> {
        match self {
            BurstModeSetting::Off => None,
            BurstModeSetting::Auto => None,
            BurstModeSetting::Frames4 => Some(4),
            BurstModeSetting::Frames6 => Some(6),
            BurstModeSetting::Frames8 => Some(8),
            BurstModeSetting::Frames50 => Some(50),
        }
    }

    /// Get all available settings
    pub const ALL: [BurstModeSetting; 6] = [
        BurstModeSetting::Off,
        BurstModeSetting::Auto,
        BurstModeSetting::Frames4,
        BurstModeSetting::Frames6,
        BurstModeSetting::Frames8,
        BurstModeSetting::Frames50,
    ];
}

/// Audio encoder preference
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum AudioEncoder {
    /// Opus codec (preferred - best quality)
    #[default]
    Opus,
    /// AAC codec (fallback - good compatibility)
    AAC,
}

impl AudioEncoder {
    /// Get display name for this encoder
    pub fn display_name(&self) -> &'static str {
        match self {
            AudioEncoder::Opus => "Opus",
            AudioEncoder::AAC => "AAC",
        }
    }

    /// Get all available encoders
    pub const ALL: [AudioEncoder; 2] = [AudioEncoder::Opus, AudioEncoder::AAC];
}

/// Timelapse interval setting
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum TimelapseInterval {
    /// 2 frames per second (500ms between captures, default)
    #[default]
    Fps2,
    /// 1 frame per second
    Sec1,
    /// 2 seconds between captures
    Sec2,
    /// 5 seconds between captures
    Sec5,
    /// 10 seconds between captures
    Sec10,
    /// 30 seconds between captures
    Sec30,
    /// 1 minute between captures
    Min1,
    /// 5 minutes between captures
    Min5,
}

impl TimelapseInterval {
    /// Get interval duration in milliseconds
    pub fn millis(&self) -> u64 {
        match self {
            TimelapseInterval::Fps2 => 500,
            TimelapseInterval::Sec1 => 1_000,
            TimelapseInterval::Sec2 => 2_000,
            TimelapseInterval::Sec5 => 5_000,
            TimelapseInterval::Sec10 => 10_000,
            TimelapseInterval::Sec30 => 30_000,
            TimelapseInterval::Min1 => 60_000,
            TimelapseInterval::Min5 => 300_000,
        }
    }

    /// Get display name for this interval
    pub fn display_name(&self) -> &'static str {
        match self {
            TimelapseInterval::Fps2 => "2 fps",
            TimelapseInterval::Sec1 => "1 second",
            TimelapseInterval::Sec2 => "2 seconds",
            TimelapseInterval::Sec5 => "5 seconds",
            TimelapseInterval::Sec10 => "10 seconds",
            TimelapseInterval::Sec30 => "30 seconds",
            TimelapseInterval::Min1 => "1 minute",
            TimelapseInterval::Min5 => "5 minutes",
        }
    }

    /// Get all available intervals
    pub const ALL: [TimelapseInterval; 8] = [
        TimelapseInterval::Fps2,
        TimelapseInterval::Sec1,
        TimelapseInterval::Sec2,
        TimelapseInterval::Sec5,
        TimelapseInterval::Sec10,
        TimelapseInterval::Sec30,
        TimelapseInterval::Min1,
        TimelapseInterval::Min5,
    ];
}

/// Composition guide overlay for camera preview
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum CompositionGuide {
    /// No guide overlay
    #[default]
    None,
    /// Rule of Thirds (2H + 2V lines at 1/3 and 2/3)
    RuleOfThirds,
    /// Phi Grid (2H + 2V lines at 0.382 and 0.618)
    PhiGrid,
    /// Fibonacci Spiral — focus top-left
    SpiralTopLeft,
    /// Fibonacci Spiral — focus top-right
    SpiralTopRight,
    /// Fibonacci Spiral — focus bottom-left
    SpiralBottomLeft,
    /// Fibonacci Spiral — focus bottom-right
    SpiralBottomRight,
    /// Diagonal lines from corners
    Diagonals,
    /// Crosshair (1H + 1V line through center)
    Crosshair,
}

impl CompositionGuide {
    /// Get all available guides
    pub const ALL: [CompositionGuide; 9] = [
        CompositionGuide::None,
        CompositionGuide::RuleOfThirds,
        CompositionGuide::PhiGrid,
        CompositionGuide::SpiralTopLeft,
        CompositionGuide::SpiralTopRight,
        CompositionGuide::SpiralBottomLeft,
        CompositionGuide::SpiralBottomRight,
        CompositionGuide::Diagonals,
        CompositionGuide::Crosshair,
    ];
}

/// Application theme preference
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum AppTheme {
    /// Follow system theme (dark or light based on system setting)
    #[default]
    System,
    /// Always use dark theme
    Dark,
    /// Always use light theme
    Light,
}

impl AppTheme {
    /// Get the COSMIC theme for this app theme preference.
    ///
    /// On non-COSMIC desktops, `system_dark()`/`system_light()`/`system_preference()`
    /// read broken defaults from cosmic_config, so we use built-in themes instead.
    /// For `System` mode, the initial theme defaults to dark; the portal subscription
    /// in `mod.rs` sends the correct value asynchronously once connected.
    pub fn theme(&self) -> Theme {
        if is_cosmic_desktop() {
            match self {
                Self::Dark => {
                    let mut t = theme::system_dark();
                    t.theme_type.prefer_dark(Some(true));
                    t
                }
                Self::Light => {
                    let mut t = theme::system_light();
                    t.theme_type.prefer_dark(Some(false));
                    t
                }
                Self::System => theme::system_preference(),
            }
        } else {
            match self {
                Self::Dark => Theme::dark(),
                Self::Light | Self::System => Theme::light(),
            }
        }
    }
}

/// User override for how overlay chrome is painted over the live preview.
///
/// [`Self::System`] defers to the desktop (see
/// `crate::app::overlay_style::overlay_surface`); the other three pin one
/// surface regardless of what the desktop asks for. Offered because the frosted
/// backdrop costs a blur chain per frame, which is not a trade every machine —
/// or every user — wants to make.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum OverlayEffect {
    /// Follow COSMIC's window-frosting setting. Only meaningful on COSMIC — see
    /// [`Self::available_for`].
    System,
    /// Always paint the live-blurred frosted backdrop.
    Frosted,
    /// Flat translucent tint over the sharp preview; never blurs.
    Translucent,
    /// Opaque panels; never blurs.
    Off,
}

/// Defaults to `System` on COSMIC — follow the desktop's window-frosting toggle
/// — and to [`Self::Translucent`] elsewhere. Translucent is the safe baseline
/// off-COSMIC: no compositor blur to lean on, no per-frame blur chain to pay
/// for. `Frosted` and `Off` are only ever reached by the user picking them.
///
/// Environment-dependent, like [`AppTheme::theme`] — and stable for the process
/// lifetime, since `is_cosmic_desktop()` caches.
impl Default for OverlayEffect {
    fn default() -> Self {
        if is_cosmic_desktop() {
            Self::System
        } else {
            Self::Translucent
        }
    }
}

impl OverlayEffect {
    /// Every variant, in dropdown order. **Not** the list to offer the user —
    /// that is [`Self::available`]. This is the stable, environment-independent
    /// order used to encode the value (config, the draw-time global) and to
    /// drive exhaustive tests.
    pub const ALL: [OverlayEffect; 4] = [
        OverlayEffect::System,
        OverlayEffect::Frosted,
        OverlayEffect::Translucent,
        OverlayEffect::Off,
    ];

    /// [`Self::ALL`] minus [`Self::System`], for desktops that have no frosting
    /// setting to follow.
    const WITHOUT_SYSTEM: [OverlayEffect; 3] = [
        OverlayEffect::Frosted,
        OverlayEffect::Translucent,
        OverlayEffect::Off,
    ];

    /// The effects to offer the user, in dropdown order.
    ///
    /// **This slice, not [`Self::ALL`], defines the dropdown's index space** —
    /// it is shorter off-COSMIC, so indices are NOT variant discriminants. Both
    /// the value→index read and the index→value write must go through here or
    /// they will disagree off-COSMIC.
    pub fn available() -> &'static [Self] {
        Self::available_for(is_cosmic_desktop())
    }

    /// [`Self::available`] as a pure function of the environment.
    ///
    /// Off-COSMIC there is no desktop flag to follow, so `System` would resolve
    /// to exactly [`Self::Translucent`] — a choice the user could make that
    /// changes nothing. Hiding it removes a distinction without a difference.
    fn available_for(is_cosmic: bool) -> &'static [Self] {
        if is_cosmic {
            &Self::ALL
        } else {
            &Self::WITHOUT_SYSTEM
        }
    }

    /// This effect as the user's desktop can actually honour it.
    ///
    /// A config written on COSMIC can be read on a desktop that does not offer
    /// `System` (shared home directory, changed `XDG_CURRENT_DESKTOP`). Resolved
    /// at READ time rather than normalised on load, so the stored `System`
    /// survives a trip through another desktop and still means "follow COSMIC"
    /// when the user comes back.
    pub fn effective(self) -> Self {
        self.effective_for(is_cosmic_desktop())
    }

    /// [`Self::effective`] as a pure function of the environment.
    pub(crate) fn effective_for(self, is_cosmic: bool) -> Self {
        if self == Self::System && !is_cosmic {
            Self::Translucent
        } else {
            self
        }
    }

    /// This effect's position in the dropdown, via [`Self::effective`] so a
    /// stored `System` read off-COSMIC selects the entry it fell back to rather
    /// than leaving the dropdown blank.
    pub fn dropdown_index(self) -> usize {
        self.dropdown_index_for(is_cosmic_desktop())
    }

    /// The effect a dropdown selection means, or `None` if the index is not one
    /// this desktop offers.
    pub fn from_dropdown_index(index: usize) -> Option<Self> {
        Self::from_dropdown_index_for(index, is_cosmic_desktop())
    }

    /// [`Self::dropdown_index`] as a pure function of the environment.
    fn dropdown_index_for(self, is_cosmic: bool) -> usize {
        let effective = self.effective_for(is_cosmic);
        Self::available_for(is_cosmic)
            .iter()
            .position(|e| *e == effective)
            .unwrap_or(0)
    }

    /// [`Self::from_dropdown_index`] as a pure function of the environment.
    fn from_dropdown_index_for(index: usize, is_cosmic: bool) -> Option<Self> {
        Self::available_for(is_cosmic).get(index).copied()
    }
}

/// Whether we're running on the COSMIC desktop (cached for process lifetime).
pub fn is_cosmic_desktop() -> bool {
    static IS_COSMIC: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("XDG_CURRENT_DESKTOP")
            .map(|d| d.to_ascii_uppercase().contains("COSMIC"))
            .unwrap_or(false)
    });
    *IS_COSMIC
}

/// Camera format settings for a specific camera (used for both photo and video modes)
#[derive(Debug, Clone, CosmicConfigEntry, Eq, PartialEq, Default, Serialize, Deserialize)]
pub struct FormatSettings {
    /// Resolution width
    pub width: u32,
    /// Resolution height
    pub height: u32,
    /// Framerate
    pub framerate: Option<u32>,
    /// Pixel format (e.g., "YUYV", "MJPG", "H264")
    pub pixel_format: String,
}

/// Backwards compatibility alias
pub type VideoSettings = FormatSettings;

#[derive(Debug, Clone, CosmicConfigEntry, Eq, PartialEq, Serialize, Deserialize)]
#[version = 21]
pub struct Config {
    /// Application theme preference (System, Dark, Light)
    pub app_theme: AppTheme,
    /// How overlay chrome is painted over the preview (frosted / translucent /
    /// opaque), or System to follow COSMIC's frosting setting
    pub overlay_effect: OverlayEffect,
    /// Capture-controls panel location. Bottom follows compositor orientation;
    /// Left and Right explicitly override the UI layout edge.
    pub controls_position: ControlsPosition,
    /// Default camera mode on launch
    pub default_mode: crate::app::CameraMode,
    /// Folder name for saving captures (photos go to XDG Pictures, videos go to XDG Videos)
    pub save_folder_name: String,
    /// Last camera path that successfully delivered a frame. Used as the
    /// preferred selection on next launch.
    pub last_camera_path: Option<String>,
    /// Camera path currently being initialized. Set immediately on switch /
    /// startup, cleared once the first frame from that camera arrives. If
    /// this is still set at the next launch, the previous session crashed
    /// before the camera produced a frame — the path is added to
    /// `failed_camera_paths` and skipped during selection (issue #410).
    pub pending_camera_path: Option<String>,
    /// Camera paths that crashed during the current recovery cycle. Skipped
    /// when picking the startup camera; cleared once any camera successfully
    /// produces a frame.
    pub failed_camera_paths: Vec<String>,
    /// Video mode settings per camera (key = camera device path)
    pub video_settings: HashMap<String, FormatSettings>,
    /// Photo mode settings per camera (key = camera device path)
    pub photo_settings: HashMap<String, FormatSettings>,
    /// Last selected video encoder index
    pub last_video_encoder_index: Option<usize>,
    /// Bug report submission URL (GitHub issues URL)
    pub bug_report_url: String,
    /// Mirror camera preview horizontally (selfie mode)
    pub mirror_preview: bool,
    /// Apply the same horizontal mirroring to captured photos / videos /
    /// timelapse output. Only effective when `mirror_preview` is on and a
    /// front-facing camera is selected. Default off — captured media is
    /// stored as the sensor delivered it (matches Android / iOS defaults).
    pub mirror_captures: bool,
    /// Video encoder bitrate preset (Low, Medium, High)
    pub bitrate_preset: BitratePreset,
    /// Virtual camera feature enabled (disabled by default)
    pub virtual_camera_enabled: bool,
    /// Photo output format (JPEG, PNG, or DNG)
    pub photo_output_format: PhotoOutputFormat,
    /// Save raw burst frames as DNG files (for debugging burst mode pipeline)
    pub save_burst_raw: bool,
    /// Burst mode setting (Off, Auto, or fixed frame count)
    pub burst_mode_setting: BurstModeSetting,
    /// Record audio with video
    pub record_audio: bool,
    /// Audio encoder preference (Opus or AAC)
    pub audio_encoder: AudioEncoder,
    /// Composition guide overlay for camera preview
    pub composition_guide: CompositionGuide,
    /// Timelapse capture interval
    pub timelapse_interval: TimelapseInterval,
    /// Haptic feedback on capture, mode switch, etc.
    pub haptic_feedback: bool,
    /// Photo aspect ratio preference
    pub photo_aspect_ratio: crate::app::PhotoAspectRatio,
    /// Show entire frame (Contain) instead of filling the window (Cover)
    pub preview_fit_to_view: bool,
    /// User-rebound keyboard shortcuts. Only contains user overrides;
    /// the full default set is computed at runtime.
    /// An empty SerializedKeyBind means the action is intentionally unbound.
    pub key_bindings: std::collections::HashMap<
        crate::app::keybind::Action,
        crate::app::keybind::SerializedKeyBind,
    >,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            app_theme: AppTheme::default(),           // Default to System theme
            overlay_effect: OverlayEffect::default(), // System on COSMIC, Translucent elsewhere
            controls_position: ControlsPosition::default(),
            default_mode: crate::app::CameraMode::default(), // Default to Photo
            save_folder_name: crate::constants::DEFAULT_SAVE_FOLDER.to_string(),
            last_camera_path: None,
            pending_camera_path: None,
            failed_camera_paths: Vec::new(),
            video_settings: HashMap::new(),
            photo_settings: HashMap::new(),
            last_video_encoder_index: None,
            bug_report_url:
                "https://github.com/cosmic-utils/cosmic-ext-camera/issues/new?template=bug_report_from_app.yml"
                    .to_string(),
            mirror_preview: true,   // Default to mirrored (selfie mode)
            mirror_captures: false, // Captured media unmirrored by default
            bitrate_preset: BitratePreset::default(), // Default to Medium
            virtual_camera_enabled: false, // Disabled by default
            photo_output_format: PhotoOutputFormat::default(), // Default to JPEG
            save_burst_raw: false,  // Disabled by default (debugging feature)
            burst_mode_setting: BurstModeSetting::default(), // Default to Auto
            record_audio: true,     // Enable audio recording by default
            audio_encoder: AudioEncoder::default(), // Default to Opus
            composition_guide: CompositionGuide::default(), // Default to None
            timelapse_interval: TimelapseInterval::default(), // Default to 2 fps
            haptic_feedback: true,  // Enable haptic feedback by default
            photo_aspect_ratio: crate::app::PhotoAspectRatio::default(),
            preview_fit_to_view: false,
            key_bindings: std::collections::HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controls_position_defaults_to_bottom() {
        assert_eq!(
            Config::default().controls_position,
            ControlsPosition::Bottom
        );
        assert_eq!(ControlsPosition::ALL[0], ControlsPosition::Bottom);
    }

    /// Off-COSMIC there is no frosting flag to follow, so `System` would mean
    /// exactly `Translucent`. It must not be offered.
    #[test]
    fn system_is_offered_only_on_cosmic() {
        assert!(OverlayEffect::available_for(true).contains(&OverlayEffect::System));
        assert!(!OverlayEffect::available_for(false).contains(&OverlayEffect::System));

        assert_eq!(OverlayEffect::available_for(true).len(), 4);
        assert_eq!(OverlayEffect::available_for(false).len(), 3);

        // The three real surfaces are always reachable.
        for effect in [
            OverlayEffect::Frosted,
            OverlayEffect::Translucent,
            OverlayEffect::Off,
        ] {
            for is_cosmic in [true, false] {
                assert!(
                    OverlayEffect::available_for(is_cosmic).contains(&effect),
                    "{effect:?} unreachable on is_cosmic={is_cosmic}"
                );
            }
        }
    }

    /// value -> index -> value must be identity in BOTH environments.
    ///
    /// The dropdown's index space is `available()`, which is one shorter
    /// off-COSMIC. Hardcoding `0 => System, 1 => Frosted, ...` passes on COSMIC
    /// and silently shifts every choice by one off it — index 0 would decode to
    /// `System`, which is not even in the list. This is the test that catches it.
    #[test]
    fn dropdown_index_round_trips_in_both_environments() {
        for is_cosmic in [true, false] {
            let available = OverlayEffect::available_for(is_cosmic);

            // Every offered variant survives value -> index -> value.
            for &effect in available {
                let index = effect.dropdown_index_for(is_cosmic);
                assert_eq!(
                    OverlayEffect::from_dropdown_index_for(index, is_cosmic),
                    Some(effect),
                    "{effect:?} did not round-trip at index {index} \
                     (is_cosmic={is_cosmic})"
                );
            }

            // ...and every offered index survives index -> value -> index.
            for index in 0..available.len() {
                let effect = OverlayEffect::from_dropdown_index_for(index, is_cosmic)
                    .expect("in-range index must decode");
                assert_eq!(
                    effect.dropdown_index_for(is_cosmic),
                    index,
                    "index {index} decoded to {effect:?}, which re-encodes elsewhere \
                     (is_cosmic={is_cosmic})"
                );
            }

            // Out of range must be rejected, not wrapped or defaulted.
            assert_eq!(
                OverlayEffect::from_dropdown_index_for(available.len(), is_cosmic),
                None
            );
        }
    }

    /// The mapping must agree with the labels the dropdown was actually built
    /// from: `mod.rs` maps `available()` to strings positionally, so index N
    /// must mean `available()[N]` on both desktops.
    #[test]
    fn dropdown_index_matches_the_options_it_labels() {
        // On COSMIC the list is the full ALL order...
        assert_eq!(
            OverlayEffect::from_dropdown_index_for(0, true),
            Some(OverlayEffect::System)
        );
        assert_eq!(
            OverlayEffect::from_dropdown_index_for(3, true),
            Some(OverlayEffect::Off)
        );

        // ...and off-COSMIC every index means one entry EARLIER in ALL. A
        // handler that hardcoded the ALL order would decode each of these wrong.
        assert_eq!(
            OverlayEffect::from_dropdown_index_for(0, false),
            Some(OverlayEffect::Frosted)
        );
        assert_eq!(
            OverlayEffect::from_dropdown_index_for(1, false),
            Some(OverlayEffect::Translucent)
        );
        assert_eq!(
            OverlayEffect::from_dropdown_index_for(2, false),
            Some(OverlayEffect::Off)
        );
        assert_eq!(OverlayEffect::from_dropdown_index_for(3, false), None);
    }

    /// A stored `System` read off-COSMIC must select the entry it fell back to,
    /// not index 0 by accident and not a blank dropdown.
    #[test]
    fn stored_system_selects_its_fallback_entry_off_cosmic() {
        let index = OverlayEffect::System.dropdown_index_for(false);
        assert_eq!(
            OverlayEffect::from_dropdown_index_for(index, false),
            Some(OverlayEffect::Translucent),
            "System off-COSMIC must display as Translucent"
        );
        assert!(index < OverlayEffect::available_for(false).len());
    }

    /// Pins the actual off-COSMIC index layout, so a hardcoded ALL-based
    /// mapping is caught rather than merely being self-consistent.
    #[test]
    fn off_cosmic_indices_are_shifted_relative_to_all() {
        let available = OverlayEffect::available_for(false);
        assert_eq!(available[0], OverlayEffect::Frosted);
        assert_eq!(available[1], OverlayEffect::Translucent);
        assert_eq!(available[2], OverlayEffect::Off);
        // ...whereas ALL[0] is System. The two index spaces genuinely differ.
        assert_eq!(OverlayEffect::ALL[0], OverlayEffect::System);
        assert_ne!(available[0], OverlayEffect::ALL[0]);
    }

    /// A `System` written on COSMIC and read elsewhere falls back to the
    /// off-COSMIC default rather than blanking the dropdown.
    #[test]
    fn stored_system_falls_back_off_cosmic() {
        assert_eq!(
            OverlayEffect::System.effective_for(false),
            OverlayEffect::Translucent
        );
        // ...and is preserved on COSMIC, which is why we resolve at read time
        // instead of normalising on load.
        assert_eq!(
            OverlayEffect::System.effective_for(true),
            OverlayEffect::System
        );

        // The fallback must be something the desktop actually offers.
        for is_cosmic in [true, false] {
            for effect in OverlayEffect::ALL {
                assert!(
                    OverlayEffect::available_for(is_cosmic)
                        .contains(&effect.effective_for(is_cosmic)),
                    "{effect:?}.effective_for({is_cosmic}) is not in the dropdown"
                );
            }
        }
    }

    /// The default is environment-dependent: `System` on COSMIC (follow the
    /// desktop), `Translucent` elsewhere (the flat off-COSMIC baseline).
    #[test]
    fn default_effect_is_system_on_cosmic_translucent_elsewhere() {
        // The pure form, so both branches are covered wherever this runs.
        let default_for = |is_cosmic| {
            if is_cosmic {
                OverlayEffect::System
            } else {
                OverlayEffect::Translucent
            }
        };
        assert_eq!(default_for(true), OverlayEffect::System);
        assert_eq!(default_for(false), OverlayEffect::Translucent);

        // The real `Default` agrees with the pure form for this environment...
        assert_eq!(
            OverlayEffect::default(),
            default_for(is_cosmic_desktop()),
            "Default disagrees with the expected environment default"
        );

        // ...and the default is always offered by the dropdown it defaults in.
        for is_cosmic in [true, false] {
            assert!(
                OverlayEffect::available_for(is_cosmic).contains(&default_for(is_cosmic)),
                "default is not offered on is_cosmic={is_cosmic}"
            );
        }
    }
}
