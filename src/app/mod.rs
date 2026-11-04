// SPDX-License-Identifier: GPL-3.0-only

//! Main application module for Camera
//!
//! This module contains the application state, message handling, UI rendering,
//! and business logic for the camera application.
//!
//! # Architecture
//!
//! - `state`: Application state types (AppModel, Message, CameraMode, etc.)
//! - `camera_preview`: Camera preview display widget
//! - `controls`: Capture button and recording UI
//! - `bottom_bar`: Gallery, mode switcher, camera switcher
//! - `settings`: Settings drawer UI
//! - `format_picker`: Format/resolution picker UI and logic
//! - `dropdowns`: Dropdown management
//! - `camera_ops`: Camera operations (switching cameras, changing formats)
//! - `ui`: Shared UI helpers
//! - `view`: Main view rendering
//! - `update`: Message handling
//!
//! # Main Types
//!
//! - `AppModel`: Main application state with camera management
//! - `Message`: All possible user interactions and system events
//! - `CameraMode`: Photo or Video capture modes

pub mod bar_layout;
mod bottom_bar;
mod camera_ops;
mod camera_preview;
mod composition_overlay;
mod config_persistence;
mod controls;
mod dropdowns;
pub mod exposure_picker;
mod filter_picker;
mod format_picker;
pub mod frame_processor;
mod frosted_backdrop;
mod gallery_primitive;
mod gallery_widget;
mod handlers;
pub mod insights;
pub mod keybind;
mod motor_picker;
mod overlay_style;
mod preview_geometry;
pub mod qr_overlay;
mod recording_error;
pub mod settings;
mod state;
mod toggle_feedback;
mod ui;
mod update;
mod video_primitive;
#[cfg(test)]
pub(crate) use video_primitive::tests::review_decoded_preview;
mod video_widget;
mod view;
mod window_controls;
mod zoom;

// Re-export public API
use crate::backends::camera::types::{CameraDevice, CameraFormat};
use crate::config::Config;
use crate::fl;
use cosmic::cosmic_config::{self, CosmicConfigEntry};
use cosmic::iced::Subscription;
use cosmic::widget::{self, about::About};
use cosmic::{ApplicationExt, Element, Task};
use iced_futures::subscription;
pub use recording_error::EncoderOperation;
pub use state::{
    AppFlags, AppModel, BurstModeStage, BurstModeState, CameraMode, ContextPage, FileSource,
    FilterType, Message, PhotoAspectRatio, PhotoTimerSetting, PrewarmResults, RecordingState,
    TimelapseState, VirtualCameraState,
};
use std::sync::{Arc, Mutex};
use tracing::{debug, error, info, warn};

/// Helper to create a subscription with an ID and a stream, replacing the removed `run_with_id`.
///
/// The `id` is hashed for subscription identity. The `stream` is the actual event stream.
fn subscription_with_id<I, S, T>(id: I, stream: S) -> Subscription<T>
where
    I: std::hash::Hash + 'static,
    S: futures::Stream<Item = T> + Send + 'static,
    T: 'static,
{
    use std::hash::Hash;
    struct IdStream<I, S> {
        id: I,
        stream: S,
    }
    impl<I, S> subscription::Recipe for IdStream<I, S>
    where
        I: Hash + 'static,
        S: futures::Stream + Send + 'static,
    {
        type Output = S::Item;
        fn hash(&self, state: &mut subscription::Hasher) {
            std::any::TypeId::of::<I>().hash(state);
            self.id.hash(state);
        }
        fn stream(
            self: Box<Self>,
            _input: subscription::EventStream,
        ) -> iced_futures::BoxStream<Self::Output> {
            Box::pin(self.stream)
        }
    }
    subscription::from_recipe(IdStream { id, stream })
}

/// Get the photo save directory
///
/// Uses XDG Pictures directory for proper flatpak compatibility.
/// Falls back to $HOME/Pictures if XDG directory is unavailable.
pub fn get_photo_directory(folder_name: &str) -> std::path::PathBuf {
    let (base_dir, source) = if let Some(xdg_dir) = dirs::picture_dir() {
        (xdg_dir, "XDG Pictures")
    } else {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        (
            std::path::Path::new(&home).join("Pictures").to_path_buf(),
            "$HOME/Pictures fallback",
        )
    };
    let photo_dir = base_dir.join(folder_name);
    debug!(
        path = %photo_dir.display(),
        source = source,
        "Resolved photo directory"
    );
    photo_dir
}

/// Get the video save directory
///
/// Uses XDG Videos directory for proper flatpak compatibility.
/// Falls back to $HOME/Videos if XDG directory is unavailable.
pub fn get_video_directory(folder_name: &str) -> std::path::PathBuf {
    let (base_dir, source) = if let Some(xdg_dir) = dirs::video_dir() {
        (xdg_dir, "XDG Videos")
    } else {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        (
            std::path::Path::new(&home).join("Videos").to_path_buf(),
            "$HOME/Videos fallback",
        )
    };
    let video_dir = base_dir.join(folder_name);
    debug!(
        path = %video_dir.display(),
        source = source,
        "Resolved video directory"
    );
    video_dir
}

/// Ensure the photo directory exists, creating it if necessary
fn ensure_photo_directory(folder_name: &str) -> Result<std::path::PathBuf, std::io::Error> {
    let photo_dir = get_photo_directory(folder_name);
    std::fs::create_dir_all(&photo_dir)?;
    info!(path = %photo_dir.display(), "Photo directory ready");
    Ok(photo_dir)
}

/// Ensure the video directory exists, creating it if necessary
fn ensure_video_directory(folder_name: &str) -> Result<std::path::PathBuf, std::io::Error> {
    let video_dir = get_video_directory(folder_name);
    std::fs::create_dir_all(&video_dir)?;
    info!(path = %video_dir.display(), "Video directory ready");
    Ok(video_dir)
}

/// Pick which camera to start with, skipping any path known to have crashed
/// the previous launch. Preference order:
///   1. `last_path` if present in `cameras` and not in `failed_paths`
///   2. First camera whose path is not in `failed_paths`
///   3. 0 (every available camera has failed — fall back to index 0 anyway
///      so the app still tries to show *something*; the next launch will
///      re-add it to `failed_paths` if it crashes again).
fn pick_startup_camera_index(
    cameras: &[crate::backends::camera::types::CameraDevice],
    last_path: Option<&str>,
    failed_paths: &[String],
) -> usize {
    if let Some(last_path) = last_path
        && !failed_paths.iter().any(|p| p == last_path)
        && let Some((idx, _)) = cameras
            .iter()
            .enumerate()
            .find(|(_, c)| c.path == last_path)
    {
        info!(index = idx, "Starting with saved camera");
        return idx;
    }
    if let Some((idx, cam)) = cameras
        .iter()
        .enumerate()
        .find(|(_, c)| !c.path.is_empty() && !failed_paths.iter().any(|p| p == &c.path))
    {
        if !failed_paths.is_empty() {
            warn!(
                index = idx,
                path = %cam.path,
                failed_count = failed_paths.len(),
                "Skipping previously-crashing camera(s), starting with next available"
            );
        }
        return idx;
    }
    if !failed_paths.is_empty() {
        warn!(
            failed_count = failed_paths.len(),
            "Every available camera previously crashed — falling back to index 0"
        );
    }
    0
}

/// Resolve camera identity and formats without re-querying an explicitly cached
/// camera-zero result (including an empty result). Callers retain their warm or
/// asynchronous scheduling and choose when to construct the backend.
fn resolve_startup_camera(
    cameras: Vec<CameraDevice>,
    cached_default_formats: Option<Vec<CameraFormat>>,
    last_path: Option<&str>,
    failed_paths: &[String],
    query_formats: impl FnOnce(&CameraDevice) -> Vec<CameraFormat>,
) -> (Vec<CameraDevice>, usize, Vec<CameraFormat>) {
    let index = pick_startup_camera_index(&cameras, last_path, failed_paths);
    let formats = if let Some(formats) = cached_default_formats.filter(|_| index == 0) {
        formats
    } else {
        cameras
            .get(index)
            .filter(|camera| !camera.path.is_empty())
            .map(query_formats)
            .unwrap_or_default()
    };
    (cameras, index, formats)
}

const REPOSITORY: &str = "https://github.com/cosmic-utils/cosmic-ext-camera";

/// App icon for the about page (pre-rendered PNG avoids SVG parsing lag on first open)
const APP_ICON: &[u8] =
    include_bytes!("../../resources/icons/hicolor/256x256/apps/io.github.cosmic_utils.camera.png");

fn app_title() -> String {
    fl!("camera")
}

impl AppModel {
    fn update_title(&mut self) -> Task<cosmic::Action<Message>> {
        if let Some(window_id) = self.core.main_window_id() {
            self.set_window_title(app_title(), window_id)
        } else {
            Task::none()
        }
    }
}

impl cosmic::Application for AppModel {
    /// The async executor that will be used to run your application's commands.
    type Executor = cosmic::executor::Default;

    /// Data that your application receives to its init method.
    type Flags = AppFlags;

    /// Messages which the application and its widgets will emit.
    type Message = Message;

    /// Unique identifier in RDNN (reverse domain name notation) format.
    const APP_ID: &'static str = "io.github.cosmic_utils.camera";

    fn core(&self) -> &cosmic::Core {
        &self.core
    }

    fn core_mut(&mut self) -> &mut cosmic::Core {
        &mut self.core
    }

    /// Initializes the application with any given flags and startup commands.
    fn init(
        mut core: cosmic::Core,
        flags: Self::Flags,
    ) -> (Self, Task<cosmic::Action<Self::Message>>) {
        let init_start = std::time::Instant::now();
        crate::startup::milestone("app_init_start");

        // Disable libcosmic's built-in keyboard navigation (Tab/Shift+Tab/Esc/F11/Ctrl+F).
        // Tab focus traversal is removed by user request; Esc handling is restored by our
        // own keybind subscription, which emits Message::Escape on Esc presses.
        core.set_keyboard_nav(false);

        // Create the about widget
        let about = About::default()
            .name(fl!("camera"))
            .icon(widget::icon::from_raster_bytes(APP_ICON))
            .version(env!("GIT_VERSION"))
            .author("Frederic Laing")
            .license("GPL-3.0-only")
            .license_url("https://www.gnu.org/licenses/gpl-3.0.html")
            .developers([("Frederic Laing", "dev@fredfunk.tech")])
            .links([
                (fl!("repository"), REPOSITORY),
                (
                    fl!("about-support"),
                    "https://github.com/cosmic-utils/cosmic-ext-camera/issues",
                ),
            ])
            .comments(
                "Burst mode algorithm based on Google HDR+ (SIGGRAPH 2016) \
                with implementation guidance from hdr-plus-swift by Martin Marek (GPL-3.0).",
            );

        // Load configuration
        let (config_handler, mut config) =
            match cosmic_config::Config::new(Self::APP_ID, Config::VERSION) {
                Ok(handler) => {
                    let config = match Config::get_entry(&handler) {
                        Ok(config) => config,
                        Err((errors, config)) => {
                            error!(?errors, "Errors loading config");
                            config
                        }
                    };
                    (Some(handler), config)
                }
                Err(err) => {
                    error!(%err, "Failed to create config handler");
                    (None, Config::default())
                }
            };
        crate::startup::milestone("config_loaded");

        // Publish the overlay effect before the first draw: the colour roots in
        // `overlay_style` read a global, not `self.config`.
        crate::app::overlay_style::init_overlay_effect(config.overlay_effect);

        // Recover from a previous crash: if a camera switch was in flight and
        // the app died before the new camera produced a frame, that path is
        // still in `pending_camera_path`. Add it to the failed list so we
        // skip it during selection below (issue #410).
        if let Some(crashed) = config.pending_camera_path.take() {
            warn!(
                path = %crashed,
                "Previous session crashed before camera delivered a frame — marking path as failed"
            );
            if !config.failed_camera_paths.contains(&crashed) {
                config.failed_camera_paths.push(crashed);
            }
            if let Some(handler) = config_handler.as_ref()
                && let Err(err) = config.write_entry(handler)
            {
                error!(?err, "Failed to persist crash-recovery state");
            }
        }

        // Ensure photo and video directories exist
        if let Err(e) = ensure_photo_directory(&config.save_folder_name) {
            error!(error = %e, "Failed to create photo directory");
        }
        if let Err(e) = ensure_video_directory(&config.save_folder_name) {
            error!(error = %e, "Failed to create video directory");
        }

        // Collect pre-warmed results from background thread (started before event loop).
        // GStreamer init, audio/camera/video enumeration ran in parallel with
        // Wayland/wgpu setup, so this join should be near-instant.
        let prewarm = flags.prewarm.and_then(|h| h.join().ok());
        crate::startup::milestone("prewarm_joined");

        let (available_audio_devices, available_video_encoders, mut camera_enum_handle) =
            if let Some(pw) = prewarm {
                // GStreamer was already initialized by the prewarm thread, but call again
                // to ensure the main thread's state is consistent (gst::init is idempotent)
                let _ = gstreamer::init();
                (pw.audio_devices, pw.video_encoders, pw.camera_enum)
            } else {
                // Fallback: no prewarm available (e.g. CLI paths), do sync init
                if let Err(e) = gstreamer::init() {
                    error!(error = %e, "Failed to initialize GStreamer");
                }
                let audio = crate::backends::audio::enumerate_audio_devices();
                let video = crate::media::encoders::video::enumerate_video_encoders();
                (audio, video, None)
            };

        // Start with empty camera list - will be populated by async task
        let available_cameras = Vec::new();
        let current_camera_index = 0;
        let available_formats = Vec::new();
        let initial_format = None;
        let camera_dropdown_options = Vec::new();

        let current_audio_device_index = 0; // Default device is sorted first
        let audio_dropdown_options: Vec<String> = available_audio_devices
            .iter()
            .map(|dev| {
                if dev.is_default {
                    format!("{} (Default)", dev.name)
                } else {
                    dev.name.clone()
                }
            })
            .collect();

        // Use saved encoder index, or default to 0 (best encoder is sorted first)
        let current_video_encoder_index = config
            .last_video_encoder_index
            .filter(|&idx| idx < available_video_encoders.len())
            .unwrap_or(0);
        let video_encoder_dropdown_options: Vec<String> = available_video_encoders
            .iter()
            .map(|enc| {
                enc.display_name
                    .replace(" (HW)", " (hardware accelerated)")
                    .replace(" (SW)", " (software)")
            })
            .collect();

        // Recording state survives capture-pipeline restarts.
        let recording_bridge = crate::backends::camera::RecordingBridge::default();

        // Read the Copy flags before `flags.preview_source` is moved below.
        let preview_spoof_recording = flags.preview_spoof_recording;
        let preview_fake_camera = flags.preview_fake_camera;

        // Convert preview source path to FileSource if provided
        let preview_file_source = flags.preview_source.and_then(|path| {
            if !path.exists() {
                error!(path = %path.display(), "Preview source file not found");
                return None;
            }
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase());
            match ext.as_deref() {
                Some("png" | "jpg" | "jpeg" | "webp" | "bmp" | "gif") => {
                    info!(path = %path.display(), "Using image as preview source");
                    Some(FileSource::Image(path))
                }
                Some("mp4" | "webm" | "mkv" | "avi" | "mov") => {
                    info!(path = %path.display(), "Using video as preview source");
                    Some(FileSource::Video(path))
                }
                _ => {
                    warn!(path = %path.display(), "Unknown file extension for preview source, treating as image");
                    Some(FileSource::Image(path))
                }
            }
        });
        let has_preview_source = preview_file_source.is_some();
        // Construct the app model with the runtime's core.
        let initial_mode = config.default_mode;
        let initial_aspect_ratio = config.photo_aspect_ratio;
        let initial_preview_fit = config.preview_fit_to_view;
        let virtual_camera_enabled = config.virtual_camera_enabled;
        let mut config_persistence = config_persistence::ConfigPersistence::default();
        config_persistence.prepare(&config);
        let mut app = AppModel {
            core,
            context_page: ContextPage::default(),
            settings_page: crate::app::state::SettingsPage::default(),
            bindings: crate::app::keybind::Bindings::with_overrides(&config.key_bindings),
            recording_keybind: None,
            about,
            config,
            config_handler,
            config_persistence,
            mode: initial_mode,
            carousel_button_slide: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            quick_record: crate::app::state::QuickRecordState::Idle,
            capture_scale_from: 1.0,
            capture_scale_to: 1.0,
            capture_anim_start: None,
            photo_btn_scale_from: 1.0,
            photo_btn_scale_to: 1.0,
            photo_btn_anim_start: None,
            recording: RecordingState::default(),
            preview_spoof_recording,
            recording_session_counter: 0,
            virtual_camera: VirtualCameraState::default(),
            virtual_camera_file_source: preview_file_source,
            current_frame_is_file_source: has_preview_source,
            current_frame_rotation: crate::backends::camera::types::SensorRotation::None,
            display_orientation: crate::backends::display_orientation::DisplayOrientation::default(
            ),
            window_controls: window_controls::WindowControls::default(),
            blur_frame_rotation: crate::backends::camera::types::SensorRotation::None,
            blur_frame_mirror: false,
            blur_frame_zoom: 1.0,
            video_file_progress: None,
            video_preview_seek_position: 0.0,
            video_file_paused: false,
            video_playback_control_tx: None,
            video_preview_control_tx: None,
            video_preview_stop_tx: None,
            file_source_preview_receiver: None,
            photo_jobs: std::collections::BTreeMap::new(),
            photo_processing_gate: state::PhotoProcessingGate::default(),
            next_photo_id: 1,
            hardware_photo_id: None,
            last_published_photo_id: 0,
            shutdown_waiting_for_recorder: false,
            recording_jobs: std::collections::BTreeSet::new(),
            encoder_operations: std::collections::BTreeMap::new(),
            recording_failure: None,
            is_capturing: false,
            motion_photo_recorder: None,
            motion_photo_retiring: None,
            motion_photo_capture: None,
            motion_photo_error: None,
            toggle_feedback: None,
            format_picker_visible: false,
            exposure_picker_visible: false,
            color_picker_visible: false,
            tools_menu_visible: false,
            motor_picker_visible: false,
            exposure_settings: None,
            color_settings: None,
            available_exposure_controls:
                crate::app::exposure_picker::AvailableExposureControls::default(),
            exposure_mode_model: {
                let mut model = cosmic::widget::segmented_button::SingleSelectModel::builder()
                    .insert(|b| b.text(fl!("exposure-auto-mode")))
                    .insert(|b| b.text(fl!("exposure-manual-mode")))
                    .build();
                model.activate_position(0); // Start with Auto
                model
            },
            burst_mode: BurstModeState::default(),
            auto_detected_frame_count: 1, // Start with 1 (no HDR+) until first brightness evaluation
            hdr_override_disabled: false,
            selected_filter: FilterType::default(),
            recording_filter_code: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            flash: state::FlashState {
                enabled: false,
                active: false,
                hardware: {
                    let hw = crate::flash::FlashHardware::detect();
                    if hw.has_devices() {
                        info!(
                            count = hw.devices.len(),
                            "Flash hardware detected and writable"
                        );
                    } else if hw.has_error() {
                        warn!("Flash hardware detected but not writable — permission error");
                    } else {
                        info!("No flash hardware detected");
                    }
                    hw
                },
                error_popup: None,
            },
            photo_timer_setting: PhotoTimerSetting::default(),
            photo_timer_countdown: None,
            photo_timer_tick_start: None,
            photo_aspect_ratio: initial_aspect_ratio,
            zoom_level: 1.0,
            zoom_control: Default::default(),
            zoom_animation: None,
            preview_fit_to_view: initial_preview_fit,
            fit_animation: None,
            ui_hidden: false,
            last_bug_report_path: None,
            last_media_path: None,
            pending_close: false,
            gallery_thumbnail: None,
            picker_selected_resolution: None,
            pending_hotplug_switch: None,
            recording_bridge: Some(recording_bridge),
            camera_cancel_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            camera_stream_restart_counter: 0,
            still_capture: Default::default(),
            current_frame: None,
            available_cameras,
            camera_discovery_complete: false,
            current_camera_index,
            pending_persist_camera: None,
            available_formats: available_formats.clone(),
            active_format: initial_format,
            available_audio_devices,
            current_audio_device_index,
            available_video_encoders,
            current_video_encoder_index,
            camera_dropdown_options,
            audio_dropdown_options,
            video_encoder_dropdown_options,
            bitrate_preset_dropdown_options: crate::constants::BitratePreset::ALL
                .iter()
                .map(|p| p.display_name().to_string())
                .collect(),
            theme_dropdown_options: vec![fl!("match-desktop"), fl!("dark"), fl!("light")],
            // Built from `available()`, so `System` is absent off-COSMIC.
            overlay_effect_dropdown_options: crate::config::OverlayEffect::available()
                .iter()
                .map(|effect| match effect {
                    crate::config::OverlayEffect::System => fl!("overlay-effect-system"),
                    crate::config::OverlayEffect::Frosted => fl!("overlay-effect-frosted"),
                    crate::config::OverlayEffect::Translucent => fl!("overlay-effect-translucent"),
                    crate::config::OverlayEffect::Off => fl!("overlay-effect-off"),
                })
                .collect(),
            controls_position_dropdown_options: vec![
                fl!("controls-position-bottom"),
                fl!("controls-position-left"),
                fl!("controls-position-right"),
            ],

            burst_mode_frame_count_dropdown_options: vec![
                fl!("hdr-plus-off"),
                fl!("hdr-plus-auto"),
                fl!("hdr-plus-frames-4"),
                fl!("hdr-plus-frames-6"),
                fl!("hdr-plus-frames-8"),
                fl!("hdr-plus-frames-50"),
            ],
            photo_output_format_dropdown_options: crate::config::PhotoOutputFormat::ALL
                .iter()
                .map(|f| f.display_name().to_string())
                .collect(),
            audio_encoder_dropdown_options: crate::config::AudioEncoder::ALL
                .iter()
                .map(|e| e.display_name().to_string())
                .collect(),
            composition_guide_dropdown_options: vec![
                fl!("guide-none"),
                fl!("guide-rule-of-thirds"),
                fl!("guide-phi-grid"),
                fl!("guide-spiral-top-left"),
                fl!("guide-spiral-top-right"),
                fl!("guide-spiral-bottom-left"),
                fl!("guide-spiral-bottom-right"),
                fl!("guide-diagonal"),
                fl!("guide-crosshair"),
            ],
            default_mode_dropdown_options: dropdowns::default_mode_options(virtual_camera_enabled),
            timelapse: TimelapseState::default(),
            timelapse_interval_dropdown_options: crate::config::TimelapseInterval::ALL
                .iter()
                .map(|i| i.display_name().to_string())
                .collect(),
            device_info_visible: false,
            audio_probe: None,
            audio_levels_snapshot: None,
            screen_width: 0.0,
            screen_height: 0.0,
            transition_state: crate::app::state::TransitionState::default(),
            // QR detection enabled by default
            qr_detection_enabled: true,
            qr_detections: Vec::new(),
            qr_detections_shown_at: None,
            qr_detection_scheduler: Default::default(),
            pending_qr_frame: None,
            // Privacy cover detection
            privacy_cover_closed: false,
            idle_inhibit: None,
            // Insights drawer
            insights: Default::default(),
        };

        // Make context drawer overlay the content instead of reserving space
        app.core.window.context_is_overlay = true;
        // Disable content container to prevent layout gaps
        app.core.window.content_container = false;
        // Always hide headerbar — custom window controls are in the top bar overlay
        app.core.window.show_headerbar = false;
        // libcosmic's template paints a rounded border but only clips the surface
        // through COSMIC's corner-radius protocol. On other compositors our
        // full-bleed preview remains rectangular and protrudes below that border.
        app.core.window.use_template = crate::config::is_cosmic_desktop();

        // Warm launches often finish camera enumeration before AppModel construction.
        // Install those results immediately so capture can start on the first loop.
        let last_camera_path = app.config.last_camera_path.clone();
        let failed_camera_paths = app.config.failed_camera_paths.clone();
        let (cameras_ready_at_init, early_camera_followup_task) = if !preview_fake_camera
            && camera_enum_handle
                .as_ref()
                .is_some_and(std::thread::JoinHandle::is_finished)
        {
            let handle = camera_enum_handle
                .take()
                .expect("camera enumeration handle");
            let (cameras, default_formats) =
                handle.join().unwrap_or_else(|_| (Vec::new(), Vec::new()));
            let (cameras, camera_index, formats) = resolve_startup_camera(
                cameras,
                Some(default_formats),
                last_camera_path.as_deref(),
                &failed_camera_paths,
                |camera| crate::backends::camera::create_backend().get_formats(camera, false),
            );
            let task = app.handle_cameras_initialized(cameras, camera_index, formats);
            (true, task)
        } else {
            (false, Task::none())
        };

        // Preview harness only (`--preview-spoof-recording`): enter Video mode
        // with a spoofed active recording so the "recording in progress" shot can
        // be captured without running the encoder. `recording` is set to
        // `Recording` so every is_recording()-gated widget renders correctly, but
        // no GStreamer pipeline is started and no frames are diverted from the
        // preview (we never call `set_recording_sender`); the indicator shows a
        // fixed placeholder duration. See preview/capture-previews.sh.
        if preview_spoof_recording {
            app.mode = CameraMode::Video;
            app.recording_session_counter += 1;
            let (stop_tx, _stop_rx) = tokio::sync::oneshot::channel();
            app.recording = RecordingState::start(
                app.recording_session_counter,
                String::new(),
                stop_tx,
                Some(Default::default()),
            );
        }

        // Initialize cameras — camera enum thread was started before the event loop
        // so it overlaps with framework Vulkan init (~280ms). We wrap the join in an
        // async Task so it doesn't block init() or the first render.
        let init_task = if preview_fake_camera {
            // Preview harness (`--preview-fake-camera`): skip real enumeration and
            // present synthetic devices so the full camera UI renders. Frames come
            // from `--preview-source`; with a file source active the camera
            // subscription never runs, so these devices are never opened. The
            // prewarm enumeration handle, if any, is simply dropped.
            let (cameras, formats) =
                crate::backends::camera::synthetic::synthetic_preview_cameras();
            Task::done(cosmic::Action::App(Message::CamerasInitialized(
                cameras, 0, formats,
            )))
        } else if cameras_ready_at_init {
            Task::none()
        } else if let Some(handle) = camera_enum_handle {
            let failed_paths = failed_camera_paths.clone();
            Task::perform(
                async move {
                    // Join the prewarm camera thread — it started ~170ms ago,
                    // so it may already be done or nearly done
                    let (cameras, default_formats) =
                        handle.join().unwrap_or_else(|_| (Vec::new(), Vec::new()));
                    info!(count = cameras.len(), "Prewarmed camera enumeration ready");

                    resolve_startup_camera(
                        cameras,
                        Some(default_formats),
                        last_camera_path.as_deref(),
                        &failed_paths,
                        |camera| {
                            crate::backends::camera::create_backend().get_formats(camera, false)
                        },
                    )
                },
                |(cameras, index, formats)| {
                    cosmic::Action::App(Message::CamerasInitialized(cameras, index, formats))
                },
            )
        } else {
            // No prewarm — fall back to full async enumeration
            let failed_paths = failed_camera_paths.clone();
            Task::perform(
                async move {
                    info!("Enumerating cameras asynchronously (no prewarm)");
                    let backend = crate::backends::camera::create_backend();
                    let cameras = backend.enumerate_cameras();
                    info!(count = cameras.len(), "Found camera(s)");

                    resolve_startup_camera(
                        cameras,
                        None,
                        last_camera_path.as_deref(),
                        &failed_paths,
                        |camera| backend.get_formats(camera, false),
                    )
                },
                |(cameras, index, formats)| {
                    cosmic::Action::App(Message::CamerasInitialized(cameras, index, formats))
                },
            )
        };

        // Load initial gallery thumbnail
        let folder_name = app.config.save_folder_name.clone();
        let folder_name2 = folder_name.clone();
        let load_thumbnail_task = Task::perform(
            async move {
                crate::storage::load_latest_thumbnail(
                    get_photo_directory(&folder_name),
                    get_video_directory(&folder_name2),
                )
                .await
            },
            |handle| cosmic::Action::App(Message::GalleryThumbnailLoaded(handle)),
        );

        // If a preview source was provided via CLI, trigger loading it
        let preview_source_task = if let Some(ref source) = app.virtual_camera_file_source {
            let source_clone = source.clone();
            Task::done(cosmic::Action::App(Message::VirtualCameraFileSelected(
                Some(source_clone),
            )))
        } else {
            Task::none()
        };

        // Log available encoders in parallel (non-critical, doesn't block camera init)
        let encoder_log_task = Task::perform(
            async { crate::media::encoders::log_available_encoders() },
            |_| cosmic::Action::App(Message::Noop),
        );

        // Precompile conversion/photo/filter shaders on a background blocking worker.
        //
        // Wait for the iced renderer to seed the shared GPU device first so warmup
        // binds its pipelines to the renderer's device (no second wgpu::Instance).
        // The 2s timeout protects against headless/test paths where no renderer
        // ever calls `try_seed_shared_gpu_from_renderer` — there, warmup falls
        // through and creates its own compute-only device.
        let gpu_warmup_task = Task::perform(
            async {
                crate::gpu::wait_for_renderer_seed_or_timeout(std::time::Duration::from_secs(2))
                    .await;
                crate::shaders::warmup_gpu_pipelines().await
            },
            |result| cosmic::Action::App(Message::GpuPipelinesWarmed(result)),
        );

        // Apply the theme from config on startup.
        // On non-COSMIC desktops with System theme, query the XDG portal for the
        // actual color scheme so we don't briefly flash the wrong theme.
        let theme_task = if !crate::config::is_cosmic_desktop()
            && app.config.app_theme == crate::config::AppTheme::System
        {
            Task::perform(
                async {
                    use ashpd::desktop::settings::{ColorScheme, Settings};
                    match Settings::new().await {
                        Ok(settings) => settings
                            .color_scheme()
                            .await
                            .map(|scheme| matches!(scheme, ColorScheme::PreferDark))
                            .unwrap_or(false),
                        Err(_) => false,
                    }
                },
                |is_dark| cosmic::Action::App(Message::PortalColorSchemeChanged(is_dark)),
            )
        } else {
            cosmic::command::set_theme(app.config.app_theme.theme())
        };

        info!(
            elapsed_ms = init_start.elapsed().as_millis(),
            "Application init complete"
        );
        crate::startup::milestone("app_init_complete");

        let title_task = app.update_title();

        (
            app,
            Task::batch([
                title_task,
                init_task,
                early_camera_followup_task,
                encoder_log_task,
                load_thumbnail_task,
                preview_source_task,
                gpu_warmup_task,
                theme_task,
            ]),
        )
    }

    /// Override the application style so window-control icons in our custom
    /// title bar keep their foreground color when maximized. libcosmic's
    /// default `style()` swaps `icon_color` to `bg_color` for maximized
    /// windows, which `widget::button::icon` inherits as its ambient color.
    ///
    /// The ambient icon color is the accent, so every symbolic icon that doesn't
    /// name a color of its own — the gallery button, the picker glyphs, the
    /// indicators — matches the title bar and the camera switcher instead of
    /// being the odd plain-white one out (issue #565). Content surfaces are
    /// unaffected: the context drawer's own container publishes `primary.on`,
    /// which wins over this for everything nested inside it.
    fn style(&self) -> Option<cosmic::iced::theme::Style> {
        let theme = cosmic::theme::active();
        let cosmic = theme.cosmic();
        Some(cosmic::iced::theme::Style {
            background_color: cosmic::iced::Color::TRANSPARENT,
            text_color: cosmic.on_bg_color().into(),
            icon_color: crate::app::overlay_style::chrome_accent_color(&theme),
        })
    }

    /// Track the latest window dimensions so capture-time crop logic can
    /// compute the visible content aspect ratio without reading from a
    /// canvas-side global on the render thread.
    fn on_window_resize(&mut self, _id: cosmic::iced::window::Id, width: f32, height: f32) {
        self.screen_width = width;
        self.screen_height = height;
    }

    // NOTE: `context_drawer` is deliberately not implemented. libcosmic renders
    // that drawer as an iced overlay pinned to the top of the main view, which
    // for this app — whose title bar is drawn *inside* the view — hid the window
    // controls whenever the drawer opened (issue #565). `AppModel::view` builds
    // the drawer itself instead, offset below the title bar.

    /// Handle escape key - close any open drawers or pickers
    fn on_escape(&mut self) -> Task<cosmic::Action<Self::Message>> {
        if self.pending_close {
            return Task::none();
        }
        // Defensive: the subscription switches to `capture_subscription` while
        // recording, which routes Esc to `CancelKeyBindRecording` directly, so
        // this branch is normally unreachable. Kept so any other code path
        // calling `on_escape` still clears the dialog cleanly.
        if self.recording_keybind.is_some() {
            self.recording_keybind = None;
            return Task::none();
        }

        // Abort photo timer countdown if active
        if self.photo_timer_countdown.is_some() {
            return self.handle_abort_photo_timer();
        }

        // Close color picker and return to tools menu
        if self.color_picker_visible {
            self.color_picker_visible = false;
            self.tools_menu_visible = true;
            return Task::none();
        }

        // Close exposure picker and return to tools menu
        if self.exposure_picker_visible {
            self.exposure_picker_visible = false;
            self.tools_menu_visible = true;
            return Task::none();
        }

        // Close tools menu if open
        if self.tools_menu_visible {
            self.tools_menu_visible = false;
            return Task::none();
        }

        // Close format picker if open
        if self.format_picker_visible {
            self.format_picker_visible = false;
            return Task::none();
        }

        // Close context drawer if open (about, settings, filters)
        if self.core.window.show_context {
            self.core.window.show_context = false;
            self.sync_audio_probe();
            return Task::none();
        }

        // Last: bring hidden chrome back. With everything hidden there is no
        // on-screen affordance left, so Esc has to be a way out for anyone who
        // doesn't remember the toggle shortcut.
        if self.ui_hidden {
            return self.handle_toggle_ui_chrome();
        }

        Task::none()
    }

    /// Describes the interface based on the current state of the application model.
    fn view(&self) -> Element<'_, Self::Message> {
        self.with_recording_failure_dialog(AppModel::view(self))
    }

    fn dialog(&self) -> Option<Element<'_, Self::Message>> {
        self.recording_failure_dialog()
    }

    /// Register subscriptions for this application.
    fn subscription(&self) -> Subscription<Self::Message> {
        use cosmic::iced::futures::{SinkExt, StreamExt};

        let config_sub = self
            .core()
            .watch_config::<Config>(Self::APP_ID)
            .map(|update| Message::UpdateConfig(update.config));

        // Get current camera device path and format
        let current_camera = self
            .available_cameras
            .get(self.current_camera_index)
            .cloned();
        let camera_index = self.current_camera_index;
        let current_format = self.active_format.clone();
        let cancel_flag = Arc::clone(&self.camera_cancel_flag);
        let still_capture = Arc::clone(&self.still_capture);
        let zoom_control = Arc::clone(&self.zoom_control);
        // Create a unique ID based on format properties to trigger restart when format changes
        let format_id = current_format
            .as_ref()
            .map(|f| (f.width, f.height, f.framerate, f.pixel_format.clone()));

        // Include whether cameras are initialized in the subscription ID
        // This ensures the subscription restarts when cameras become available
        let cameras_initialized = !self.available_cameras.is_empty();

        // Restart counter forces subscription to restart (e.g., after HDR+ processing)
        let restart_counter = self.camera_stream_restart_counter;

        // Get the shared recording sender Arc so the capture thread can forward
        // frames directly to the appsrc recording pipeline (libcamera only).
        let recording_sender = self.recording_bridge.as_ref().map(|s| s.recording_sender());

        // Check if file source is active - if so, don't run camera subscription
        // This applies in Virtual mode OR when --preview-source was used (any mode)
        let file_source_active = self.virtual_camera_file_source.is_some();

        let camera_sub = if file_source_active {
            // No camera subscription when file source is active (file source handles preview)
            Subscription::none()
        } else {
            subscription_with_id(
                (
                    "camera",
                    camera_index,
                    format_id,
                    // NOTE: is_recording is NOT included here!
                    // This allows preview to continue during recording
                    cameras_initialized,
                    restart_counter, // Forces restart (HDR+ processing, mode switch with role change)
                ),
                cosmic::iced::stream::channel(100, async move |mut output| {
                    info!(camera_index, "Camera subscription started");
                    crate::startup::milestone("camera_subscription_started");

                    // No artificial delay needed - PipelineManager serializes all operations
                    // and ensures proper cleanup before creating new pipelines

                    let mut frame_count = 0u64;
                    let mut last_forward = std::time::Instant::now();
                    let mut is_first_pipeline = true;
                    loop {
                        // Check cancel flag at the start of each loop iteration
                        // This prevents creating new pipelines after mode switch
                        if cancel_flag.load(std::sync::atomic::Ordering::Acquire) {
                            info!("Cancel flag set - subscription loop exiting");
                            break;
                        }

                        // If no camera available yet (cameras not initialized), just exit the subscription
                        // The subscription will restart when cameras become available (cameras_initialized flag changes)
                        let Some(camera) = &current_camera else {
                            info!(
                                "No camera available - subscription will restart when cameras are initialized"
                            );
                            break;
                        };

                        // Extract format parameters
                        let (width, height, framerate, pixel_format) =
                            if let Some(fmt) = &current_format {
                                (
                                    Some(fmt.width),
                                    Some(fmt.height),
                                    fmt.framerate,
                                    Some(fmt.pixel_format.as_str()),
                                )
                            } else {
                                (None, None, None, None)
                            };

                        info!(name = %camera.name, path = %camera.path, "Creating camera");

                        if let Some(fmt) = &current_format {
                            info!(format = %fmt, "Using format");
                        }

                        {
                            // Check cancel flag before creating pipeline
                            if cancel_flag.load(std::sync::atomic::Ordering::Acquire) {
                                info!("Cancel flag set before pipeline creation - skipping");
                                break;
                            }

                            // Note: Preview continues during recording since VideoRecorder has its own pipeline

                            // Give previous pipeline time to clean up (skip on first startup)
                            if !is_first_pipeline {
                                tokio::time::sleep(tokio::time::Duration::from_millis(
                                    crate::constants::latency::PIPELINE_CLEANUP_DELAY_MS,
                                ))
                                .await;

                                // Check cancel flag again after brief wait
                                if cancel_flag.load(std::sync::atomic::Ordering::Acquire) {
                                    info!("Cancel flag set after cleanup wait - skipping");
                                    break;
                                }
                            }

                            // Create camera pipeline based on backend type
                            use crate::backends::camera::pipeline::CapturePipeline;
                            use crate::backends::camera::types::CameraFormat;

                            let (sender, mut receiver) =
                                cosmic::iced::futures::channel::mpsc::channel(
                                    crate::constants::latency::FRAME_CHANNEL_CAPACITY,
                                );

                            let format = CameraFormat {
                                width: width.unwrap_or(640),
                                height: height.unwrap_or(480),
                                framerate,
                                hardware_accelerated: true, // Assume HW acceleration available
                                pixel_format: pixel_format.unwrap_or("MJPEG").to_string(),
                            };

                            let pipeline_opt = {
                                info!(
                                    backend = if camera.path.starts_with("pipewire:") {
                                        "pipewire-v4l2"
                                    } else {
                                        "libcamera"
                                    },
                                    "Creating camera pipeline"
                                );

                                // Small delay to allow previous pipeline to fully release camera hardware
                                // This is necessary because libcamera's "simple" pipeline handler
                                // needs time to release V4L2 resources before a new pipeline can start
                                // Skip on first startup since there is no previous pipeline
                                if !is_first_pipeline {
                                    tokio::time::sleep(tokio::time::Duration::from_millis(300))
                                        .await;
                                }

                                let camera_name = &camera.path;

                                // Pass capture resolution intact. The native negotiation
                                // caps preview only if a separate raw stream succeeds.

                                // Use the shared recording sender from the manager,
                                // or a dummy Arc if no manager is available.
                                let rec_sender = recording_sender
                                    .clone()
                                    .unwrap_or_else(|| Arc::new(Mutex::new(None)));

                                match CapturePipeline::new(
                                    camera_name,
                                    &format,
                                    true,
                                    crate::backends::camera::libcamera::PipelineSharedState {
                                        frame_sender: sender,
                                        still_capture: Arc::clone(&still_capture),
                                        recording_sender: rec_sender,
                                        zoom: Arc::clone(&zoom_control),

                                        cancel_flag: Arc::clone(&cancel_flag),
                                    },
                                ) {
                                    Ok(pipeline) => {
                                        info!("Camera capture pipeline started");
                                        crate::startup::milestone("camera_pipeline_ready");
                                        Some(pipeline)
                                    }
                                    Err(e) => {
                                        error!(error = %e, "Failed to create camera pipeline");
                                        None
                                    }
                                }
                            };

                            is_first_pipeline = false;

                            if let Some(pipeline) = pipeline_opt {
                                let preview_metrics = pipeline.preview_metrics();
                                info!("Waiting for frames from pipeline...");
                                // Keep pipeline alive and forward frames
                                loop {
                                    // Check cancel flag first (set when switching cameras/modes)
                                    if cancel_flag.load(std::sync::atomic::Ordering::Acquire) {
                                        info!(
                                            "Cancel flag set - Camera subscription being cancelled"
                                        );
                                        break;
                                    }

                                    // Check if subscription is still active before processing next frame
                                    if output.is_closed() {
                                        info!(
                                            "Output channel closed - Camera subscription being cancelled"
                                        );
                                        break;
                                    }

                                    // Wait for next frame with a timeout to periodically check cancellation
                                    // Timeout only affects cancel flag checking - frames arrive immediately when ready
                                    match tokio::time::timeout(
                                        tokio::time::Duration::from_millis(
                                            crate::constants::latency::CANCEL_CHECK_INTERVAL_MS,
                                        ),
                                        receiver.next(),
                                    )
                                    .await
                                    {
                                        Ok(Some(frame)) => {
                                            // Drain any queued frames to get the most recent one (reduces latency)
                                            let mut latest_frame = frame;
                                            let mut drained_count = 0u32;
                                            while let Ok(newer_frame) = receiver.try_recv() {
                                                latest_frame = newer_frame;
                                                drained_count += 1;
                                            }

                                            // Frame pacing: smooth out bursty ISP delivery.
                                            // If we forwarded a frame recently, wait until the
                                            // next target interval before sending another.
                                            // This turns ISP bursts (3-4 frames at once) into
                                            // evenly-spaced UI updates.
                                            const MIN_FRAME_INTERVAL: tokio::time::Duration =
                                                tokio::time::Duration::from_millis(30);
                                            let since_last = last_forward.elapsed();
                                            if since_last < MIN_FRAME_INTERVAL && drained_count > 0
                                            {
                                                let wait =
                                                    MIN_FRAME_INTERVAL.saturating_sub(since_last);
                                                tokio::time::sleep(wait).await;
                                                // After sleeping, drain again to get the absolute
                                                // latest frame
                                                while let Ok(newer_frame) = receiver.try_recv() {
                                                    latest_frame = newer_frame;
                                                    drained_count += 1;
                                                }
                                            }

                                            preview_metrics.record_drops(drained_count as u64);
                                            frame_count += 1;
                                            if frame_count == 1 {
                                                crate::startup::milestone(
                                                    "first_frame_received_by_subscription",
                                                );
                                            }
                                            last_forward = std::time::Instant::now();
                                            // Calculate frame latency (time from capture to subscription delivery)
                                            let latency_us =
                                                latest_frame.captured_at.elapsed().as_micros();

                                            if frame_count.is_multiple_of(30) {
                                                debug!(
                                                    frame = frame_count,
                                                    width = latest_frame.width,
                                                    height = latest_frame.height,
                                                    latency_ms = latency_us as f64 / 1000.0,
                                                    drained = drained_count,
                                                    "Received frame from pipeline"
                                                );
                                            }

                                            // Use try_send to avoid blocking the subscription when UI is busy
                                            // Dropping frames is fine for live preview - we want the latest frame
                                            match output.try_send(Message::CameraFrame(Arc::new(
                                                latest_frame,
                                            ))) {
                                                Ok(_) => {
                                                    if frame_count.is_multiple_of(30) {
                                                        debug!(
                                                            frame = frame_count,
                                                            "Frame forwarded to UI"
                                                        );
                                                    }
                                                }
                                                Err(e) => {
                                                    preview_metrics.record_drops(1);
                                                    // Always log dropped frames for diagnostics
                                                    tracing::warn!(
                                                        frame = frame_count,
                                                        error = ?e,
                                                        "Frame dropped (UI channel full) - stuttering likely"
                                                    );
                                                    // Check if channel is closed (subscription cancelled)
                                                    if e.is_disconnected() {
                                                        info!(
                                                            "Output channel disconnected - Camera subscription being cancelled"
                                                        );
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                        Ok(None) => {
                                            info!("Camera pipeline frame stream ended");
                                            break;
                                        }
                                        Err(_) => {
                                            // Timeout - continue loop to check if channel is closed
                                            continue;
                                        }
                                    }
                                }
                                info!("Cleaning up camera pipeline");
                                // Pipeline will be dropped here, stopping the camera
                                drop(pipeline);
                            } else {
                                error!("Failed to initialize pipeline");
                                info!("Waiting 5 seconds before retry...");
                                // Wait a bit before retrying
                                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                            }
                        }
                    }
                }),
            )
        }; // End of camera_sub if/else

        // Reconcile discovery snapshots, including PipeWire node creation/removal
        // without /dev changes (session-manager restart, monitor availability).
        // Native discovery is cached until video nodes change; PipeWire is always fresh.
        let initial_camera_paths: std::collections::BTreeSet<String> = self
            .available_cameras
            .iter()
            .map(|d| d.path.clone())
            .collect();
        let hotplug_sub = subscription_with_id(
            "camera_hotplug",
            cosmic::iced::stream::channel(10, async move |mut output| {
                let mut last_paths = initial_camera_paths;
                let mut last_video_nodes =
                    crate::backends::camera::v4l2_utils::scan_video_device_nodes();
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    if output.is_closed() {
                        break;
                    }
                    let current_nodes =
                        crate::backends::camera::v4l2_utils::scan_video_device_nodes();
                    let added: std::collections::BTreeSet<_> = current_nodes
                        .difference(&last_video_nodes)
                        .cloned()
                        .collect();
                    let removed: Vec<String> = last_video_nodes
                        .difference(&current_nodes)
                        .filter_map(|n| n.to_str().map(String::from))
                        .collect();
                    last_video_nodes = current_nodes;
                    if !removed.is_empty()
                        && output
                            .send(Message::HotplugDeviceRemoved(removed))
                            .await
                            .is_err()
                    {
                        break;
                    }
                    let mut cameras = match tokio::task::spawn_blocking(|| {
                        crate::backends::camera::libcamera::LibcameraBackend::new()
                            .enumerate_cameras_for_hotplug()
                    })
                    .await
                    {
                        Ok(cameras) => cameras,
                        Err(e) => {
                            warn!(error = %e, "Camera discovery task failed");
                            continue;
                        }
                    };
                    cameras.retain(|d| {
                        d.v4l2_path()
                            .is_none_or(|p| std::path::Path::new(p).exists())
                    });
                    let paths: std::collections::BTreeSet<_> =
                        cameras.iter().map(|d| d.path.clone()).collect();
                    if paths != last_paths {
                        last_paths = paths;
                        if output
                            .send(Message::CameraListChanged(cameras))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    if !added.is_empty()
                        && crate::backends::camera::libcamera::native::is_capture_active()
                    {
                        // Native libcamera cannot re-enumerate while its manager is
                        // capturing. Retain the existing native hotplug placeholder
                        // mechanism; selecting it stops capture and does full discovery.
                        // PipeWire fallback additions above never use placeholders.
                        let new_devices =
                            crate::backends::camera::v4l2_utils::discover_v4l2_capture_devices(
                                &added,
                            );
                        if new_devices.is_empty() {
                            // Preserve native hotplug's readiness retry: a device
                            // node can exist before VIDIOC_QUERYCAP succeeds.
                            for node in &added {
                                last_video_nodes.remove(node);
                            }
                            continue;
                        }
                        if output
                            .send(Message::HotplugDeviceAdded(new_devices))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            }),
        );

        // Audio device hotplug monitoring subscription
        let current_audio_devices = self.available_audio_devices.clone();
        let audio_hotplug_sub = subscription_with_id(
            "audio_hotplug",
            cosmic::iced::stream::channel(10, async move |mut output| {
                info!("Audio device hotplug monitoring started");

                let mut last_devices = current_audio_devices;

                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

                    // Enumeration is blocking: a Unix-socket handshake with
                    // PulseAudio, and on the fallback path a `pw-dump` fork plus
                    // a parse of the whole PipeWire graph. Off the async worker
                    // threads it goes, so a slow audio server can never stall
                    // whatever else the runtime is driving.
                    let Ok(new_devices) = tokio::task::spawn_blocking(
                        crate::backends::audio::enumerate_audio_devices,
                    )
                    .await
                    else {
                        warn!("Audio enumeration task failed - stopping hotplug monitoring");
                        break;
                    };

                    let devices_changed = last_devices.len() != new_devices.len()
                        || !last_devices.iter().all(|d| {
                            new_devices
                                .iter()
                                .any(|nd| nd.serial == d.serial && nd.name == d.name)
                        });

                    if devices_changed {
                        info!(
                            old_count = last_devices.len(),
                            new_count = new_devices.len(),
                            "Audio device list changed - hotplug event detected"
                        );

                        last_devices = new_devices.clone();

                        if output
                            .send(Message::AudioListChanged(new_devices))
                            .await
                            .is_err()
                        {
                            warn!("Failed to send audio list changed message - channel closed");
                            break;
                        }
                    }
                }

                info!("Audio device hotplug monitoring stopped");
            }),
        );

        // File source preview subscription - receives frames from file streaming thread
        let file_source_preview_sub = if let Some(ref receiver) = self.file_source_preview_receiver
        {
            let receiver = receiver.clone();
            subscription_with_id(
                "file_source_preview",
                cosmic::iced::stream::channel(10, async move |mut output| {
                    loop {
                        // Hold the lock for one full recv+drain cycle so we
                        // don't repeatedly take/release it on the inner loop,
                        // and so a hypothetical second consumer can't race
                        // partway through draining.
                        let latest = {
                            let mut guard = receiver.lock().await;
                            let Some(first) = guard.recv().await else {
                                break;
                            };
                            let mut latest = first;
                            while let Ok(newer) = guard.try_recv() {
                                latest = newer;
                            }
                            latest
                        };

                        if output.send(Message::CameraFrame(latest)).await.is_err() {
                            break;
                        }
                    }
                }),
            )
        } else {
            Subscription::none()
        };

        // Timer countdown animation subscription (30fps for smooth fade)
        let timer_animation_sub = if self.photo_timer_countdown.is_some() {
            cosmic::iced::time::every(std::time::Duration::from_millis(33))
                .map(|_| Message::PhotoTimerAnimationFrame)
        } else {
            Subscription::none()
        };

        // Privacy cover status polling subscription (every 3 seconds)
        // Only runs if the camera has privacy control support
        let privacy_polling_sub = if self.available_exposure_controls.has_privacy {
            let device_path = self.get_v4l2_device_path();
            if let Some(path) = device_path {
                let path = path.clone();
                subscription_with_id(
                    ("privacy_polling", path.clone()),
                    cosmic::iced::stream::channel(1, async move |mut output| {
                        use crate::backends::camera::v4l2_controls;
                        loop {
                            // Poll privacy control status
                            let is_closed =
                                v4l2_controls::get_control(&path, v4l2_controls::V4L2_CID_PRIVACY)
                                    .map(|v| v != 0)
                                    .unwrap_or(false);

                            if output
                                .send(Message::PrivacyCoverStatusChanged(is_closed))
                                .await
                                .is_err()
                            {
                                break;
                            }

                            // Wait 3 seconds before next poll
                            tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
                        }
                    }),
                )
            } else {
                Subscription::none()
            }
        } else {
            Subscription::none()
        };

        // Brightness evaluation subscription (every 1 second when in Auto mode)
        // Updates auto_detected_frame_count based on scene brightness
        // Initial delay of 2 seconds at startup to allow camera to stabilize
        let brightness_eval_sub = if self.mode == CameraMode::Photo
            && matches!(
                self.config.burst_mode_setting,
                crate::config::BurstModeSetting::Auto
            )
            && !self.hdr_override_disabled
            && self.current_frame.is_some()
        {
            // Evaluate brightness every 1 second
            let interval = std::time::Duration::from_secs(1);
            cosmic::iced::time::every(interval).map(|_| Message::BrightnessEvaluationTick)
        } else {
            Subscription::none()
        };

        // Update insights metrics every 500ms when the Insights drawer is open
        let insights_update_sub =
            if self.context_page == ContextPage::Insights && self.core.window.show_context {
                let interval = std::time::Duration::from_millis(500);
                cosmic::iced::time::every(interval).map(|_| Message::UpdateInsightsMetrics)
            } else {
                Subscription::none()
            };

        // 100 ms audio level snapshot — only while a level source is active.
        let audio_level_sub = if self.audio_probe.is_some() || self.recording.is_recording() {
            let interval = std::time::Duration::from_millis(100);
            cosmic::iced::time::every(interval).map(|_| Message::AudioLevelTick)
        } else {
            Subscription::none()
        };

        // On COSMIC desktops, subscribe to the cosmic_config theme configs so
        // accent / palette / mode edits made in COSMIC Settings propagate to
        // the running app. Without this, an accent edit in the *current*
        // light/dark mode wouldn't take effect until the user toggled the
        // other mode and back (issue #290). The handler just re-evaluates
        // `AppTheme::theme()`, which re-reads cosmic_config on each call.
        let cosmic_theme_sub = if crate::config::is_cosmic_desktop() {
            let dark = self
                .core()
                .watch_config::<cosmic::cosmic_theme::Theme>(cosmic::cosmic_theme::DARK_THEME_ID)
                .map(|_| Message::CosmicThemeChanged);
            let light = self
                .core()
                .watch_config::<cosmic::cosmic_theme::Theme>(cosmic::cosmic_theme::LIGHT_THEME_ID)
                .map(|_| Message::CosmicThemeChanged);
            let mode = self
                .core()
                .watch_config::<cosmic::cosmic_theme::ThemeMode>(
                    cosmic::cosmic_theme::THEME_MODE_ID,
                )
                .map(|_| Message::CosmicThemeChanged);
            Subscription::batch([dark, light, mode])
        } else {
            Subscription::none()
        };

        // On non-COSMIC desktops, subscribe to XDG portal color-scheme changes
        // so theme updates when user changes their desktop appearance
        let portal_theme_sub = if !crate::config::is_cosmic_desktop()
            && self.config.app_theme == crate::config::AppTheme::System
        {
            subscription_with_id(
                "portal-color-scheme",
                cosmic::iced::stream::channel(10, async move |mut output| {
                    use ashpd::desktop::settings::{ColorScheme, Settings};

                    let Ok(settings) = Settings::new().await else {
                        tracing::warn!("Failed to create XDG Settings portal proxy");
                        std::future::pending::<()>().await;
                        return;
                    };

                    let send_scheme =
                        |output: &mut cosmic::iced::futures::channel::mpsc::Sender<Message>,
                         scheme: ColorScheme| {
                            let is_dark = matches!(scheme, ColorScheme::PreferDark);
                            output
                                .try_send(Message::PortalColorSchemeChanged(is_dark))
                                .ok();
                        };

                    // Send initial color scheme
                    if let Ok(scheme) = settings.color_scheme().await {
                        send_scheme(&mut output, scheme);
                    }

                    // Subscribe to live changes via ashpd's D-Bus signal stream
                    if let Ok(mut stream) = settings.receive_color_scheme_changed().await {
                        while let Some(scheme) = StreamExt::next(&mut stream).await {
                            send_scheme(&mut output, scheme);
                        }
                    }

                    tracing::warn!("Portal color-scheme stream ended");
                    std::future::pending::<()>().await;
                }),
            )
        } else {
            Subscription::none()
        };

        let keybind_sub = if self.pending_close {
            Subscription::none()
        } else if self.recording_keybind.is_some() {
            keybind::capture_subscription()
        } else {
            keybind::subscription(
                self.bindings.clone(),
                self.mode,
                self.virtual_camera_file_source.is_some(),
                self.recording.is_recording(),
            )
        };

        // Volume-key shutter on phones: the compositor swallows the keys
        // for system audio so they never reach iced as keyboard events.
        // The `volume_keys` backend reads `/dev/input/event*` directly,
        // grabs each device exclusively while the window is focused
        // (so the compositor doesn't *also* change the volume) and
        // forwards presses as `Message::Capture` — same as the spacebar.
        let volume_keys_sub = subscription_with_id(
            "volume_keys",
            cosmic::iced::stream::channel(8, async move |mut output| {
                let mut rx = crate::backends::volume_keys::start();
                while let Some(_key) = rx.recv().await {
                    if output.send(Message::Capture).await.is_err() {
                        break;
                    }
                }
            }),
        );

        // Display-orientation subscription: opens a private Wayland
        // connection on a background thread, reads `wl_output::transform`
        // and forwards changes as `DisplayOrientationChanged` so the
        // preview rotation can include the screen's rotation alongside
        // the sensor mounting.
        let display_orientation_sub = subscription_with_id(
            "display_orientation",
            cosmic::iced::stream::channel(8, async move |mut output| {
                let mut rx = crate::backends::display_orientation::start();
                while let Some(orient) = rx.recv().await {
                    if output
                        .send(Message::DisplayOrientationChanged(orient))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            }),
        );

        // Translate Wayland keyboard-focus transitions for the camera
        // window into `WindowFocusChanged` messages; the update handler
        // forwards them into the volume_keys backend so we only grab the
        // hardware shutter buttons while the camera is in focus.
        let window_focus_sub = subscription::filter_map("window_focus", |event| {
            let subscription::Event::Interaction { event, .. } = event else {
                return None;
            };
            match event {
                cosmic::iced::Event::Window(cosmic::iced::window::Event::CloseRequested) => {
                    Some(Message::WindowClose)
                }
                cosmic::iced::Event::Window(cosmic::iced::window::Event::Focused) => {
                    Some(Message::WindowFocusChanged(true))
                }
                cosmic::iced::Event::Window(cosmic::iced::window::Event::Unfocused) => {
                    Some(Message::WindowFocusChanged(false))
                }
                _ => None,
            }
        });

        // Lifecycle reconciliation keeps running even when a source has stopped
        // producing frames; retired encoder workers never overlap replacements.
        let motion_photo_lifecycle = if self.pending_close
            || self.config.motion_photo
            || self.motion_photo_retiring.is_some()
        {
            cosmic::iced::time::every(std::time::Duration::from_millis(100)).map(|_| Message::Noop)
        } else {
            Subscription::none()
        };
        Subscription::batch([
            self.encoder_failure_subscriptions(),
            window_controls::subscription(),
            motion_photo_lifecycle,
            config_sub,
            camera_sub,
            hotplug_sub,
            audio_hotplug_sub,
            file_source_preview_sub,
            timer_animation_sub,
            privacy_polling_sub,
            brightness_eval_sub,
            insights_update_sub,
            audio_level_sub,
            portal_theme_sub,
            cosmic_theme_sub,
            keybind_sub,
            volume_keys_sub,
            window_focus_sub,
            display_orientation_sub,
        ])
    }

    fn on_app_exit(&mut self) -> Option<Self::Message> {
        Some(Message::WindowClose)
    }

    /// Handles messages emitted by the application and its widgets.
    fn update(&mut self, message: Self::Message) -> Task<cosmic::Action<Self::Message>> {
        self.update(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framework_exit_is_routed_through_media_drain() {
        let mut model = AppModel::default();
        assert!(matches!(
            cosmic::Application::on_app_exit(&mut model),
            Some(Message::WindowClose)
        ));
    }

    #[test]
    fn window_title_uses_localized_app_name() {
        assert_eq!(app_title(), fl!("camera"));
    }
}

#[cfg(test)]
mod startup_camera_tests {
    use super::*;

    fn cameras(paths: &[&str]) -> Vec<CameraDevice> {
        paths
            .iter()
            .map(|path| CameraDevice {
                name: (*path).into(),
                path: (*path).into(),
                ..Default::default()
            })
            .collect()
    }

    #[test]
    fn startup_resolver_preserves_selection_cache_and_query_boundaries() {
        // Prewarm's empty formats are authoritative, including failed/empty camera-zero fallback.
        for paths in [&["a", "b"][..], &[""][..], &[][..]] {
            let (_, index, formats) =
                resolve_startup_camera(cameras(paths), Some(Vec::new()), None, &[], |_| {
                    panic!("cached camera-zero formats must not be queried again")
                });
            assert_eq!(index, 0);
            assert!(formats.is_empty());
        }
        for (paths, last, failed, expected_index, expected_query) in [
            (vec!["a", "b"], None, vec![], 0, Some("a")),
            (vec!["a", "b"], Some("b"), vec![], 1, Some("b")),
            (vec!["a", "b"], Some("a"), vec!["a"], 1, Some("b")),
            (vec!["a", "b"], Some("b"), vec!["a", "b"], 0, Some("a")),
            (vec!["", "b"], None, vec![], 1, Some("b")),
            (vec![""], None, vec![], 0, None),
            (vec![], None, vec![], 0, None),
        ] {
            let mut queried = None;
            let failed: Vec<_> = failed.into_iter().map(str::to_string).collect();
            let (resolved, index, formats) =
                resolve_startup_camera(cameras(&paths), None, last, &failed, |camera| {
                    queried = Some(camera.path.clone());
                    Vec::new()
                });
            assert_eq!(
                resolved.iter().map(|c| c.path.as_str()).collect::<Vec<_>>(),
                paths
            );
            assert_eq!(index, expected_index);
            assert_eq!(queried.as_deref(), expected_query);
            assert!(formats.is_empty());
        }
        let cached = vec![CameraFormat {
            width: 640,
            height: 480,
            framerate: None,
            hardware_accelerated: false,
            pixel_format: "YUYV".into(),
        }];
        let (_, _, formats) =
            resolve_startup_camera(cameras(&["a"]), Some(cached.clone()), None, &[], |_| {
                panic!("populated cache is authoritative too")
            });
        assert_eq!(formats, cached);
        let (_, index, _) = resolve_startup_camera(
            cameras(&["a", "b"]),
            Some(cached),
            Some("b"),
            &[],
            |camera| {
                assert_eq!(camera.path, "b");
                Vec::new()
            },
        );
        assert_eq!(index, 1);
    }
}
