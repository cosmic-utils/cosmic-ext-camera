// SPDX-License-Identifier: GPL-3.0-only

//! Application state management

use crate::app::exposure_picker::{
    AvailableExposureControls, ColorSettings, ExposureMode, ExposureSettings, MeteringMode,
};
use crate::app::frame_processor::QrDetection;
use crate::backends::audio::AudioDevice;
use crate::backends::camera::CameraBackendManager;
use crate::backends::camera::types::{CameraDevice, CameraFormat, CameraFrame};
use crate::config::Config;
use crate::media::encoders::video::EncoderInfo;
use crate::pipelines::video::SharedAudioLevels;
use cosmic::cosmic_config;
use cosmic::widget::about::About;
use std::sync::Arc;
use std::time::Instant;

/// Recording state machine
///
/// Simple two-state design: either recording or not.
#[derive(Default)]
pub enum RecordingState {
    /// Not recording
    #[default]
    Idle,
    /// Actively recording
    Recording {
        /// Monotonic ID assigned by `AppModel::recording_session_counter` at
        /// start time. Used by `handle_recording_stopped` to detect a stale
        /// stop event arriving after a new session has already taken over.
        session: u64,
        /// When recording started
        start_time: Instant,
        /// Output file path
        file_path: String,
        /// Channel to signal stop
        stop_sender: Option<tokio::sync::oneshot::Sender<()>>,
        /// Shared live audio levels from the GStreamer recording pipeline
        audio_levels: Option<SharedAudioLevels>,
    },
}

impl std::fmt::Debug for RecordingState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecordingState::Idle => write!(f, "Idle"),
            RecordingState::Recording {
                session,
                start_time,
                file_path,
                ..
            } => f
                .debug_struct("Recording")
                .field("session", session)
                .field("elapsed", &start_time.elapsed())
                .field("file_path", file_path)
                .finish(),
        }
    }
}

impl RecordingState {
    /// Check if currently recording
    pub fn is_recording(&self) -> bool {
        matches!(self, RecordingState::Recording { .. })
    }

    /// Get the elapsed recording duration
    pub fn elapsed_duration(&self) -> u64 {
        match self {
            RecordingState::Idle => 0,
            RecordingState::Recording { start_time, .. } => start_time.elapsed().as_secs(),
        }
    }

    /// Take the stop sender (consumes it)
    pub fn take_stop_sender(&mut self) -> Option<tokio::sync::oneshot::Sender<()>> {
        match self {
            RecordingState::Idle => None,
            RecordingState::Recording { stop_sender, .. } => stop_sender.take(),
        }
    }

    /// Start recording
    pub fn start(
        session: u64,
        file_path: String,
        stop_sender: tokio::sync::oneshot::Sender<()>,
        audio_levels: Option<SharedAudioLevels>,
    ) -> Self {
        RecordingState::Recording {
            session,
            start_time: Instant::now(),
            file_path,
            stop_sender: Some(stop_sender),
            audio_levels,
        }
    }

    /// Get the shared audio levels handle (if recording with audio)
    pub fn audio_levels(&self) -> Option<&SharedAudioLevels> {
        match self {
            RecordingState::Recording { audio_levels, .. } => audio_levels.as_ref(),
            _ => None,
        }
    }

    /// Session ID of the current recording, or `None` if idle.
    pub fn session(&self) -> Option<u64> {
        match self {
            RecordingState::Idle => None,
            RecordingState::Recording { session, .. } => Some(*session),
        }
    }
}

/// State machine for long-press-to-record in Photo mode.
/// Tap (<300ms) captures a photo, long press (≥300ms) starts video recording.
#[derive(Default)]
pub enum QuickRecordState {
    #[default]
    Idle,
    /// Finger/mouse is down. Frame captured for potential photo.
    Pressed {
        press_time: std::time::Instant,
        captured_frame: Option<std::sync::Arc<crate::backends::camera::types::CameraFrame>>,
    },
    /// Recording is active (threshold exceeded).
    Recording,
}

impl QuickRecordState {
    pub fn is_pressed(&self) -> bool {
        matches!(self, Self::Pressed { .. })
    }

    pub fn is_recording(&self) -> bool {
        matches!(self, Self::Recording)
    }
}

/// Virtual camera streaming state machine
#[derive(Default)]
pub enum VirtualCameraState {
    /// Not streaming
    #[default]
    Idle,
    /// Actively streaming to virtual camera
    Streaming {
        /// When streaming started
        start_time: Instant,
        /// Channel to signal stop
        stop_sender: Option<tokio::sync::oneshot::Sender<()>>,
        /// Channel to send frames to the virtual camera pipeline
        frame_sender: tokio::sync::mpsc::UnboundedSender<Arc<CameraFrame>>,
        /// Channel to send filter updates to the virtual camera pipeline
        filter_sender: tokio::sync::watch::Sender<FilterType>,
        /// Whether streaming from a file source (image/video)
        is_file_source: bool,
    },
}

impl std::fmt::Debug for VirtualCameraState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VirtualCameraState::Idle => write!(f, "Idle"),
            VirtualCameraState::Streaming { start_time, .. } => {
                write!(f, "Streaming {{ elapsed: {:?} }}", start_time.elapsed())
            }
        }
    }
}

impl VirtualCameraState {
    /// Check if currently streaming
    pub fn is_streaming(&self) -> bool {
        matches!(self, VirtualCameraState::Streaming { .. })
    }

    /// Get the elapsed streaming duration
    pub fn elapsed_duration(&self) -> u64 {
        match self {
            VirtualCameraState::Idle => 0,
            VirtualCameraState::Streaming { start_time, .. } => start_time.elapsed().as_secs(),
        }
    }

    /// Take the stop sender (consumes it)
    pub fn take_stop_sender(&mut self) -> Option<tokio::sync::oneshot::Sender<()>> {
        match self {
            VirtualCameraState::Idle => None,
            VirtualCameraState::Streaming { stop_sender, .. } => stop_sender.take(),
        }
    }

    /// Send a frame to the virtual camera pipeline
    pub fn send_frame(&self, frame: Arc<CameraFrame>) -> bool {
        match self {
            VirtualCameraState::Idle => false,
            VirtualCameraState::Streaming { frame_sender, .. } => frame_sender.send(frame).is_ok(),
        }
    }

    /// Start streaming
    ///
    /// When `is_file_source` is true, the stream originates from a file (image/video)
    /// rather than a live camera.
    pub fn start(
        stop_sender: tokio::sync::oneshot::Sender<()>,
        frame_sender: tokio::sync::mpsc::UnboundedSender<Arc<CameraFrame>>,
        filter_sender: tokio::sync::watch::Sender<FilterType>,
        is_file_source: bool,
    ) -> Self {
        VirtualCameraState::Streaming {
            start_time: Instant::now(),
            stop_sender: Some(stop_sender),
            frame_sender,
            filter_sender,
            is_file_source,
        }
    }

    /// Check if streaming from a file source
    pub fn is_file_source(&self) -> bool {
        match self {
            VirtualCameraState::Idle => false,
            VirtualCameraState::Streaming { is_file_source, .. } => *is_file_source,
        }
    }

    /// Update the filter for virtual camera streaming
    pub fn set_filter(&self, filter: FilterType) -> bool {
        match self {
            VirtualCameraState::Idle => false,
            VirtualCameraState::Streaming { filter_sender, .. } => {
                filter_sender.send(filter).is_ok()
            }
        }
    }
}

/// Timelapse capture state machine
///
/// Frames are sent directly to a video encoder via a channel — no photos
/// are written to disk.
#[derive(Default)]
pub enum TimelapseState {
    /// Not running
    #[default]
    Idle,
    /// Actively capturing timelapse frames
    Running {
        /// When timelapse started
        start_time: Instant,
        /// Number of frames sent so far
        shots_taken: u32,
        /// Interval in milliseconds between captures
        interval_ms: u64,
        /// Channel to send frames to the encoder task
        frame_sender: tokio::sync::mpsc::UnboundedSender<Arc<CameraFrame>>,
    },
    /// Encoder is finalising the video file
    Finalising {
        /// When timelapse started (for display)
        start_time: Instant,
        /// Total frames sent
        shots_taken: u32,
    },
}

impl std::fmt::Debug for TimelapseState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimelapseState::Idle => write!(f, "Idle"),
            TimelapseState::Running {
                shots_taken,
                start_time,
                ..
            } => f
                .debug_struct("Running")
                .field("shots_taken", shots_taken)
                .field("elapsed", &start_time.elapsed())
                .finish(),
            TimelapseState::Finalising { shots_taken, .. } => f
                .debug_struct("Finalising")
                .field("shots_taken", shots_taken)
                .finish(),
        }
    }
}

impl TimelapseState {
    /// Check if timelapse is currently running (capturing)
    pub fn is_running(&self) -> bool {
        matches!(self, TimelapseState::Running { .. })
    }

    /// Check if the encoder is finalising the video
    pub fn is_finalising(&self) -> bool {
        matches!(self, TimelapseState::Finalising { .. })
    }

    /// Check if active (running or finalising)
    pub fn is_active(&self) -> bool {
        !matches!(self, TimelapseState::Idle)
    }

    /// Get elapsed duration in seconds
    pub fn elapsed_duration(&self) -> u64 {
        match self {
            TimelapseState::Idle => 0,
            TimelapseState::Running { start_time, .. }
            | TimelapseState::Finalising { start_time, .. } => start_time.elapsed().as_secs(),
        }
    }

    /// Get number of shots taken
    pub fn shots_taken(&self) -> u32 {
        match self {
            TimelapseState::Idle => 0,
            TimelapseState::Running { shots_taken, .. }
            | TimelapseState::Finalising { shots_taken, .. } => *shots_taken,
        }
    }

    /// Increment shot count, returns new count
    pub fn increment_shots(&mut self) -> u32 {
        match self {
            TimelapseState::Running { shots_taken, .. } => {
                *shots_taken += 1;
                *shots_taken
            }
            _ => 0,
        }
    }

    /// Send a frame to the encoder. Returns false if the channel is closed.
    pub fn send_frame(&self, frame: Arc<CameraFrame>) -> bool {
        match self {
            TimelapseState::Running { frame_sender, .. } => frame_sender.send(frame).is_ok(),
            _ => false,
        }
    }
}

/// Holds a D-Bus connection + request handle for org.freedesktop.portal.Inhibit.
///
/// The inhibition lasts until Close() is called on the request handle (done on
/// Drop) or the connection is dropped, so the connection must stay alive. The
/// Inhibit portal is used because it needs no finish-args: portals are always
/// reachable in the Flatpak sandbox. Desktops that ship no Inhibit backend
/// (COSMIC, tracked upstream in pop-os/xdg-desktop-portal-cosmic#342) simply
/// don't inhibit; we intentionally keep no direct-D-Bus fallback for them.
pub struct InhibitGuard {
    pub connection: zbus::blocking::Connection,
    pub handle: zbus::zvariant::OwnedObjectPath,
}

impl Drop for InhibitGuard {
    fn drop(&mut self) {
        let _ = self.connection.call_method(
            Some("org.freedesktop.portal.Desktop"),
            &self.handle,
            Some("org.freedesktop.portal.Request"),
            "Close",
            &(),
        );
    }
}

impl std::fmt::Debug for InhibitGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InhibitGuard")
            .field("handle", &self.handle.as_str())
            .finish()
    }
}

/// Burst mode state for multi-frame burst capture
///
/// Tracks the state of burst mode photo capture and processing.
/// Encapsulates all burst mode related state including the async processing
/// communication primitives.
#[derive(Debug)]
pub struct BurstModeState {
    /// Current processing stage
    pub stage: BurstModeStage,
    /// Progress during Processing stage (0.0 - 1.0)
    /// During Capturing, progress is derived from frame_buffer.len()
    pub processing_progress: f32,
    /// Frame buffer for collecting burst frames (private - use add_frame/take_frames)
    frame_buffer: Vec<Arc<CameraFrame>>,
    /// Target frame count for current capture (set from config at capture start)
    pub target_frame_count: usize,
    /// Shared atomic for processing progress updates (progress * 1000 for 0.1% precision)
    /// Only present during Processing stage
    progress_atomic: Option<Arc<std::sync::atomic::AtomicU32>>,
    /// Channel receiver for processing result
    /// Only present during Processing stage
    result_rx: Option<std::sync::mpsc::Receiver<Result<String, String>>>,
}

/// Burst mode processing stages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BurstModeStage {
    /// Waiting to start
    #[default]
    Idle,
    /// Capturing burst frames
    Capturing,
    /// Processing frames (aligning, merging, tone mapping)
    Processing,
    /// Processing complete
    Complete,
    /// Error occurred
    Error,
}

impl BurstModeState {
    /// Add a frame to the capture buffer
    ///
    /// Returns `true` if the target frame count has been reached.
    pub fn add_frame(&mut self, frame: Arc<CameraFrame>) -> bool {
        self.frame_buffer.push(frame);
        self.frame_buffer.len() >= self.target_frame_count
    }

    /// Take all frames from the buffer, leaving it empty
    pub fn take_frames(&mut self) -> Vec<Arc<CameraFrame>> {
        std::mem::take(&mut self.frame_buffer)
    }

    /// Check if capture/processing is in progress
    pub fn is_active(&self) -> bool {
        matches!(
            self.stage,
            BurstModeStage::Capturing | BurstModeStage::Processing
        )
    }

    /// Check if we're actively collecting frames (derived from stage)
    pub fn is_collecting_frames(&self) -> bool {
        self.stage == BurstModeStage::Capturing
    }

    /// Number of frames captured so far (derived from buffer)
    pub fn frames_captured(&self) -> usize {
        self.frame_buffer.len()
    }

    /// Get current progress (0.0 - 1.0)
    ///
    /// During Capturing: derived from frame_buffer.len() / target_frame_count
    /// During Processing: from processing_progress field
    /// Complete: 1.0
    /// Other stages: 0.0
    pub fn progress(&self) -> f32 {
        match self.stage {
            BurstModeStage::Capturing => {
                if self.target_frame_count == 0 {
                    0.0
                } else {
                    self.frame_buffer.len() as f32 / self.target_frame_count as f32
                }
            }
            BurstModeStage::Processing => self.processing_progress,
            BurstModeStage::Complete => 1.0,
            _ => 0.0,
        }
    }

    /// Start capture - clears buffer and sets state to Capturing
    pub fn start_capture(&mut self, target_frame_count: usize) {
        self.frame_buffer.clear();
        self.stage = BurstModeStage::Capturing;
        self.processing_progress = 0.0;
        self.target_frame_count = target_frame_count;
    }

    /// Set frames directly (used when frames are captured via capture_photo() in multistream mode)
    pub fn set_frames(&mut self, frames: Vec<Arc<CameraFrame>>) {
        self.target_frame_count = frames.len();
        self.frame_buffer = frames;
    }

    /// Mark complete
    pub fn complete(&mut self) {
        self.stage = BurstModeStage::Complete;
    }

    /// Mark error
    pub fn error(&mut self) {
        self.stage = BurstModeStage::Error;
    }

    /// Reset to idle
    pub fn reset(&mut self) {
        self.stage = BurstModeStage::Idle;
        self.processing_progress = 0.0;
        self.frame_buffer.clear();
        self.progress_atomic = None;
        self.result_rx = None;
    }

    /// Start processing and set up communication channels.
    /// Returns the atomic counter that the processing task should update.
    pub fn start_processing_task(
        &mut self,
    ) -> (
        Arc<std::sync::atomic::AtomicU32>,
        std::sync::mpsc::Sender<Result<String, String>>,
    ) {
        self.stage = BurstModeStage::Processing;
        self.processing_progress = 0.0;

        // Create shared atomic for progress updates (progress * 1000 for 0.1% precision)
        let progress_atomic = Arc::new(std::sync::atomic::AtomicU32::new(0));
        self.progress_atomic = Some(Arc::clone(&progress_atomic));

        // Create channel for result
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        self.result_rx = Some(result_rx);

        (progress_atomic, result_tx)
    }

    /// Poll progress from the processing task.
    /// Updates internal progress and returns true if still processing.
    pub fn poll_progress(&mut self) -> bool {
        if self.stage != BurstModeStage::Processing {
            return false;
        }

        // Update progress from atomic
        if let Some(atomic) = &self.progress_atomic {
            let progress_raw = atomic.load(std::sync::atomic::Ordering::Relaxed);
            self.processing_progress = progress_raw as f32 / 1000.0;
        }

        true
    }

    /// Try to get the processing result.
    /// Returns Some(result) if complete, None if still processing or not in processing state.
    pub fn try_get_result(&mut self) -> Option<Result<String, String>> {
        if let Some(rx) = &self.result_rx {
            match rx.try_recv() {
                Ok(result) => {
                    // Clear processing state
                    self.progress_atomic = None;
                    self.result_rx = None;
                    Some(result)
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => None,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    // Channel closed unexpectedly
                    self.progress_atomic = None;
                    self.result_rx = None;
                    Some(Err("Processing task terminated unexpectedly".to_string()))
                }
            }
        } else {
            None
        }
    }

    /// Clear processing state (called when not in Processing stage)
    pub fn clear_processing_state(&mut self) {
        self.progress_atomic = None;
        self.result_rx = None;
    }
}

impl Default for BurstModeState {
    fn default() -> Self {
        Self {
            stage: BurstModeStage::default(),
            processing_progress: 0.0,
            frame_buffer: Vec::new(),
            target_frame_count: 8, // Will be overwritten when capture starts
            progress_atomic: None,
            result_rx: None,
        }
    }
}

/// Grouped flash-related state.
///
/// Keeping these four fields together makes it obvious which fields belong to
/// the flash subsystem and reduces the risk of forgetting one during a reset
/// (e.g. switching cameras must clear both `enabled` and the error popup).
#[derive(Default)]
pub struct FlashState {
    /// Flash enabled for photo capture (user toggle).
    pub enabled: bool,
    /// Flash overlay is currently being drawn (screen-flash path).
    pub active: bool,
    /// Detected flash hardware (LED nodes + permission status).
    pub hardware: crate::flash::FlashHardware,
    /// Permission error popup message — shown when hardware was detected but
    /// is not writable from the current sandbox / user.
    pub error_popup: Option<String>,
}

/// The application model stores app-specific state used to describe its interface and
/// drive its logic.
#[cfg_attr(test, derive(Default))]
pub struct AppModel {
    /// Application state which is managed by the COSMIC runtime.
    pub core: cosmic::Core,
    /// Display a context drawer with the designated page if defined.
    pub context_page: ContextPage,
    /// Which settings sub-page is shown while the Settings drawer is open.
    pub settings_page: SettingsPage,
    /// Active key bindings (defaults merged with user overrides).
    pub bindings: crate::app::keybind::Bindings,
    /// Set while the user is recording a new keybind via the Settings page.
    pub recording_keybind: Option<RecordingKeyBindState>,
    /// The about page for this app.
    pub about: About,
    /// Configuration data that persists between application runs.
    pub config: Config,
    /// Configuration handler for saving settings
    pub config_handler: Option<cosmic_config::Config>,
    /// Current camera mode (Photo or Video)
    pub mode: CameraMode,
    /// Shared button inward slide (written by carousel update, read by SlideH draw).
    /// Updated every frame so buttons track the carousel even when view() isn't called.
    pub carousel_button_slide: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// Quick-record state machine (long-press-to-record in Photo mode)
    pub quick_record: QuickRecordState,
    /// Capture button scale animation state
    pub capture_scale_from: f32,
    pub capture_scale_to: f32,
    pub capture_anim_start: Option<std::time::Instant>,
    /// Animation state for the photo-during-recording button (separate from main button)
    pub photo_btn_scale_from: f32,
    pub photo_btn_scale_to: f32,
    pub photo_btn_anim_start: Option<std::time::Instant>,
    /// Recording state (idle, recording, or paused)
    pub recording: RecordingState,
    /// Preview harness only: when set, the recording indicator shows a fixed
    /// placeholder duration instead of the live elapsed time, so the "recording
    /// in progress" screenshot is deterministic. Enabled by
    /// `--preview-spoof-recording`; see `AppFlags::preview_spoof_recording`.
    pub preview_spoof_recording: bool,
    /// Monotonic counter that mints a unique ID for each recording session.
    /// `handle_start_recording` increments this before assigning the new
    /// session; `handle_recording_stopped` uses it to ignore late stop events
    /// from a previous session whose async finalizer outlived its UI lifetime.
    pub recording_session_counter: u64,
    /// Virtual camera state (idle or streaming)
    pub virtual_camera: VirtualCameraState,
    /// File source for virtual camera (image or video to stream instead of camera)
    pub virtual_camera_file_source: Option<FileSource>,
    /// Whether the current frame is from a file source (vs camera)
    pub current_frame_is_file_source: bool,
    /// Rotation of the camera that produced the current frame
    /// (used during blur transitions to maintain correct rotation)
    pub current_frame_rotation: crate::backends::camera::types::SensorRotation,
    /// Display orientation reported by the compositor via
    /// `wl_output::geometry.transform`. Combined with the sensor's
    /// mounting rotation at render time so the preview stays upright
    /// when the phone is rotated between portrait and landscape.
    pub display_orientation: crate::backends::display_orientation::DisplayOrientation,
    /// Rotation of the camera that produced the blur frame
    /// (captured at start of blur transition to maintain correct rotation during transition)
    pub blur_frame_rotation: crate::backends::camera::types::SensorRotation,
    /// Whether the blur frame was mirrored (captured at start of blur transition)
    pub blur_frame_mirror: bool,
    /// Digital zoom the blur frame was rendered at (captured at start of blur
    /// transition). The blurred frame is frozen, so it must keep the zoom it was
    /// last displayed with — otherwise an in-flight zoom animation (mode switch,
    /// zoom reset) eases the *still* image instead of the live preview.
    pub blur_frame_zoom: f32,
    /// Video file playback progress (position_secs, duration_secs, progress 0.0-1.0)
    pub video_file_progress: Option<(f64, f64, f64)>,
    /// Video preview seek position (used when not streaming to store desired start position)
    pub video_preview_seek_position: f64,
    /// Whether video file playback is paused
    pub video_file_paused: bool,
    /// Channel to send playback control commands to the streaming thread
    pub video_playback_control_tx: Option<tokio::sync::mpsc::UnboundedSender<VideoPlaybackCommand>>,
    /// Channel to send playback control commands to the preview thread (when not streaming)
    pub video_preview_control_tx: Option<tokio::sync::mpsc::UnboundedSender<VideoPlaybackCommand>>,
    /// Stop sender for preview playback thread
    pub video_preview_stop_tx: Option<tokio::sync::oneshot::Sender<()>>,
    /// Receiver for preview frames from file source streaming
    pub file_source_preview_receiver: Option<
        std::sync::Arc<tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<Arc<CameraFrame>>>>,
    >,
    /// Whether a photo capture is in progress
    pub is_capturing: bool,
    /// Whether the format picker is visible (iOS-style popup)
    pub format_picker_visible: bool,
    /// Whether the exposure picker is visible (iOS-style popup)
    pub exposure_picker_visible: bool,
    /// Whether the color picker is visible (iOS-style popup)
    pub color_picker_visible: bool,
    /// Whether the tools menu is visible (iOS-style popup)
    pub tools_menu_visible: bool,

    // ===== Motor/PTZ Controls =====
    /// Whether motor controls picker is visible
    pub motor_picker_visible: bool,

    /// Current exposure settings for active camera
    pub exposure_settings: Option<ExposureSettings>,
    /// Current color/image adjustment settings for active camera
    pub color_settings: Option<ColorSettings>,
    /// Available exposure controls for current camera (queried from V4L2)
    pub available_exposure_controls: AvailableExposureControls,
    /// Segmented button model for exposure mode (Auto/Manual)
    pub exposure_mode_model: cosmic::widget::segmented_button::SingleSelectModel,
    /// Base exposure time (in 100µs units) captured when entering manual mode
    /// Used to calculate EV-based adjustments in non-advanced mode
    pub base_exposure_time: Option<i32>,
    /// Burst mode state (enabled, capture/processing progress)
    pub burst_mode: BurstModeState,
    /// Auto-detected frame count based on current scene brightness (1-8)
    /// Updated every 1 second when in Auto mode via BrightnessEvaluationTick
    pub auto_detected_frame_count: usize,
    /// User override to disable HDR+ even when auto-detection suggests using it
    /// Reset when switching burst mode settings or on app restart
    pub hdr_override_disabled: bool,
    /// Currently selected filter
    pub selected_filter: FilterType,
    /// Live filter code shared with the recording pusher (AtomicU32).
    /// Updated on every filter change so the recorder sees it in real-time.
    pub recording_filter_code: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// All flash-related state, grouped to keep reset/configuration
    /// transitions in one place. See [`FlashState`].
    pub flash: FlashState,
    /// Photo timer setting (off, 3s, 5s, 10s)
    pub photo_timer_setting: PhotoTimerSetting,
    /// Photo timer countdown (remaining seconds, None when not counting)
    pub photo_timer_countdown: Option<u8>,
    /// When the current countdown second started (for fade animation)
    pub photo_timer_tick_start: Option<Instant>,
    /// Photo aspect ratio (native, 4:3, 16:9, 1:1)
    pub photo_aspect_ratio: PhotoAspectRatio,
    /// Current (settled) zoom level (1.0 = no zoom, 2.0 = 2x zoom, etc.)
    pub zoom_level: f32,
    /// In-flight zoom transition, or `None` when settled on `zoom_level`.
    /// Set by the zoom-reset button; pinch / step zoom set `zoom_level`
    /// directly and clear this so they remain real-time.
    pub zoom_animation: Option<ZoomAnimation>,
    /// Show entire frame (Contain mode) instead of filling the window (Cover mode)
    pub preview_fit_to_view: bool,
    /// In-flight fit/fill transition, or `None` when settled on `preview_fit_to_view`.
    pub fit_animation: Option<FitAnimation>,
    /// Hide every piece of overlay chrome (top bar, carousel, capture button,
    /// fit/fill and zoom chips) leaving just the live preview. Session-only and
    /// deliberately not persisted: restoring it at launch would open the app
    /// with no visible controls. Esc and the shortcut both bring the UI back.
    pub ui_hidden: bool,
    /// Path to last generated bug report
    pub last_bug_report_path: Option<String>,
    /// Path to the most recently saved photo or video (for gallery pre-selection)
    pub last_media_path: Option<String>,
    /// Set when the user pressed the close button while a recording or
    /// timelapse was active. The handlers that complete those operations
    /// (`handle_recording_stopped` / `handle_timelapse_assembly_complete`)
    /// check this flag and exit the process once the file is finalized.
    pub pending_close: bool,
    /// Latest gallery thumbnail (cached)
    pub gallery_thumbnail: Option<cosmic::widget::image::Handle>,
    /// Gallery thumbnail RGBA data for custom rendering (Arc for cheap cloning)
    pub gallery_thumbnail_rgba: Option<(Arc<Vec<u8>>, u32, u32)>,
    /// Currently selected resolution in the picker (width for grouping)
    pub picker_selected_resolution: Option<u32>,
    /// V4L2 device path the user is trying to switch to (set when switching
    /// to a hotplugged camera that needs full re-enumeration).
    pub pending_hotplug_switch: Option<String>,
    /// Camera backend manager
    pub backend_manager: Option<CameraBackendManager>,
    /// Flag to cancel camera subscription (used when switching backends/cameras)
    pub camera_cancel_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Counter to force camera stream restart (incremented to change subscription ID)
    pub camera_stream_restart_counter: u32,
    /// Shared flag to request a still capture from the raw stream (multistream mode)
    pub still_capture_requested: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Shared storage for the latest still frame from the raw stream (multistream mode)
    pub latest_still_frame: std::sync::Arc<std::sync::Mutex<Option<CameraFrame>>>,
    /// Notifier fired by the capture thread when a new still frame is stored.
    /// Lets `wait_for_still_frame` await on a notification instead of polling.
    pub still_frame_notify: std::sync::Arc<tokio::sync::Notify>,
    /// Current camera frame
    pub current_frame: Option<Arc<CameraFrame>>,
    /// Available camera devices
    pub available_cameras: Vec<CameraDevice>,
    /// Current camera index
    pub current_camera_index: usize,
    /// Camera selection awaiting first-frame confirmation before being
    /// promoted to `config.last_camera_path`. Stored as (path, since): the
    /// `since` instant is used to filter out stale frames captured from the
    /// previous camera before the switch (issue #410).
    pub pending_persist_camera: Option<(String, Instant)>,
    /// Available formats for current camera
    pub available_formats: Vec<CameraFormat>,
    /// Currently active format being used by camera
    pub active_format: Option<CameraFormat>,
    /// Available audio input devices
    pub available_audio_devices: Vec<AudioDevice>,
    /// Current audio device index
    pub current_audio_device_index: usize,
    /// Available video encoders
    pub available_video_encoders: Vec<EncoderInfo>,
    /// Current video encoder index
    pub current_video_encoder_index: usize,
    /// Cached mode information (for consolidated dropdown)
    pub mode_list: Vec<crate::backends::camera::types::CameraFormat>,
    /// Dropdown options (cached for UI)
    pub camera_dropdown_options: Vec<String>,
    pub audio_dropdown_options: Vec<String>,
    pub video_encoder_dropdown_options: Vec<String>,
    pub mode_dropdown_options: Vec<String>,
    pub pixel_format_dropdown_options: Vec<String>,
    pub resolution_dropdown_options: Vec<String>,
    pub framerate_dropdown_options: Vec<String>,
    pub codec_dropdown_options: Vec<String>,
    /// Bitrate preset dropdown options
    pub bitrate_preset_dropdown_options: Vec<String>,
    /// Theme dropdown options (Match Desktop, Dark, Light)
    pub theme_dropdown_options: Vec<String>,
    /// Overlay effect dropdown options (System, Frosted Glass, Translucent,
    /// Off). Built from `OverlayEffect::available()`, so System is absent
    /// off-COSMIC — index with that same slice, never with `ALL`.
    pub overlay_effect_dropdown_options: Vec<String>,
    /// Capture-controls position dropdown options (Bottom, Left, Right).
    pub controls_position_dropdown_options: Vec<String>,
    /// Burst mode merge mode dropdown options (Quality FFT, Fast Spatial)
    pub burst_mode_merge_dropdown_options: Vec<String>,
    /// Burst mode frame count dropdown options (Auto, 4, 6, 8 frames)
    pub burst_mode_frame_count_dropdown_options: Vec<String>,
    /// Photo output format dropdown options (JPEG, PNG, DNG)
    pub photo_output_format_dropdown_options: Vec<String>,
    /// Audio encoder dropdown options (Opus, AAC)
    pub audio_encoder_dropdown_options: Vec<String>,
    /// Composition guide dropdown options
    pub composition_guide_dropdown_options: Vec<String>,
    /// Default mode dropdown options (Photo, Video, Timelapse, Virtual)
    pub default_mode_dropdown_options: Vec<String>,
    /// Whether the device info panel is visible
    pub device_info_visible: bool,

    /// Live pre-recording audio level probe (Some only while the settings
    /// drawer is open with audio recording enabled and no real recording).
    pub audio_probe: Option<crate::pipelines::audio_probe::AudioLevelProbe>,
    /// Shared levels owned by `audio_probe`. Mirrors `audio_probe.levels()`
    /// when the probe is alive; `None` otherwise.
    pub probe_audio_levels: Option<crate::pipelines::audio_level::SharedAudioLevels>,
    /// 100 ms snapshot of whichever level source is currently active
    /// (recorder during recording, probe otherwise). The render path reads
    /// this without locking any mutex.
    pub audio_levels_snapshot: Option<crate::pipelines::audio_level::AudioLevels>,

    /// Latest window/canvas dimensions (logical pixels), set by
    /// `Application::on_window_resize`. Used by capture-time crop logic
    /// to compute the visible content aspect ratio. Both default to 0.0
    /// before the first resize event; consumers fall back to 16:9.
    pub screen_width: f32,
    pub screen_height: f32,

    /// Transition state for camera/settings changes
    pub transition_state: TransitionState,

    // ===== QR Code Detection =====
    /// Whether QR code detection is enabled
    pub qr_detection_enabled: bool,
    /// Current QR code detections (updated at 1 FPS)
    pub qr_detections: Vec<QrDetection>,
    /// Last time QR detection was processed
    pub last_qr_detection_time: Option<Instant>,

    // ===== Privacy Cover Detection =====
    /// Whether the camera privacy cover is closed (blocking the camera)
    pub privacy_cover_closed: bool,

    // ===== Idle Inhibit =====
    /// Active idle + suspend inhibit via the XDG Inhibit portal (held while the
    /// camera is in use; released on Drop). Absent on desktops that ship no
    /// Inhibit portal backend (e.g. COSMIC).
    pub idle_inhibit: Option<InhibitGuard>,

    // ===== Timelapse =====
    /// Timelapse capture state
    pub timelapse: TimelapseState,
    /// Timelapse interval dropdown options (cached for UI)
    pub timelapse_interval_dropdown_options: Vec<String>,

    // ===== Insights Drawer =====
    /// Insights drawer diagnostic state
    pub insights: super::insights::InsightsState,
}

/// In-flight animation between Cover and Contain preview modes (and the
/// associated bottom-bar scrim height, which differs between Photo and the
/// other modes).
///
/// The targets are always `AppModel::settled_blend()` and
/// `AppModel::settled_bottom_ui_height()`. By construction, callers mutate
/// `self.mode` / `self.preview_fit_to_view` before installing a new
/// animation, and neither field changes mid-animation (a re-trigger replaces
/// the whole struct), so both settled values are stable throughout playback.
/// Snapshot of all values that animate through a fit/fill transition,
/// captured by callers before they mutate `self.mode` or
/// `self.preview_fit_to_view`. The animation interpolates linearly from
/// every field toward its corresponding settled value via shared eased
/// progress (`AppModel::fit_animation_eased`), so adding a fifth animated
/// channel later is a one-place change here plus the matching
/// `AppModel::capture_fit_state` reader.
#[derive(Debug, Clone, Copy)]
pub struct FitFrom {
    /// Blend value at the moment the animation started (for mid-animation reversals).
    pub blend: f32,
    /// Top-bar scrim / shader bar at the moment the animation started.
    /// 0 in View mode, `TOP_BAR_HEIGHT` otherwise. Drives the canvas scrim,
    /// the shader's `bar_top_px`, and the composition guide's content rect.
    pub top_ui_height: f32,
    /// Bottom-bar scrim / shader bar at the moment the animation started.
    /// 0 in View mode (the preview takes the full window in fit/fill);
    /// the bottom carousel renders on top of the live preview without a scrim.
    pub bottom_ui_height: f32,
    /// Height of the empty placeholder above the bottom bar at the moment
    /// the animation started. 0 in View (capture button hidden, fit/zoom
    /// row sits flush above the carousel); the capture-area height
    /// otherwise. Drives the column layout above the bottom bar.
    pub capture_area_height: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct FitAnimation {
    /// When the animation began.
    pub start: Instant,
    /// Snapshot captured at animation start. Each channel interpolates
    /// from this value toward the current settled value.
    pub from: FitFrom,
    /// Set when the source or destination of this animation is the
    /// `View` mode (i.e. when the capture-area / scrim heights cross 0).
    /// The view-mode rendering paths (capture-button slot, etc.) consult
    /// this rather than comparing animated heights against settled values
    /// so the contract is explicit.
    pub is_view_boundary: bool,
}

/// In-flight zoom transition driven by the zoom-reset button. The target
/// is always `AppModel::zoom_level` (the settled value); pinch and step
/// zoom set the target directly without an animation, clearing this so
/// they stay real-time.
#[derive(Debug, Clone, Copy)]
pub struct ZoomAnimation {
    /// When the animation began.
    pub start: Instant,
    /// Zoom level at the moment the animation started.
    pub from: f32,
}

/// State for smooth blur transitions when changing camera settings
#[derive(Debug, Clone)]
pub struct TransitionState {
    /// Whether we're currently in a transition (blur is active)
    pub in_transition: bool,
    /// Timestamp when transition started (for detecting camera restart)
    pub transition_start_time: Option<Instant>,
    /// Timestamp when first new frame arrived (for blur countdown)
    pub first_frame_time: Option<Instant>,
    /// Whether UI should be disabled during transition (to prevent user interaction)
    pub ui_disabled: bool,
    /// Duration in ms to keep blur active after first frame arrives (default: 1000ms)
    pub blur_duration_ms: u64,
}

impl Default for TransitionState {
    fn default() -> Self {
        Self {
            in_transition: false,
            transition_start_time: None,
            first_frame_time: None,
            ui_disabled: false,
            blur_duration_ms: crate::constants::latency::CAMERA_SWITCH_BLUR_HOLD_MS,
        }
    }
}

/// Camera modes
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum CameraMode {
    #[default]
    Photo,
    Video,
    /// Virtual camera mode - streams filtered video to a virtual camera
    Virtual,
    /// Timelapse mode - captures photos at a configurable interval
    Timelapse,
    /// View mode — minimal-UI live preview. No capture controls; only the
    /// mode carousel, fit/fill toggle, and zoom button are shown, and the
    /// top/bottom UI scrim is fully transparent.
    View,
}

impl CameraMode {
    /// All available camera modes
    pub const ALL: [CameraMode; 5] = [
        CameraMode::Photo,
        CameraMode::Video,
        CameraMode::Timelapse,
        CameraMode::Virtual,
        CameraMode::View,
    ];

    /// Whether this mode renders a minimal-chrome live preview (no capture
    /// or recording controls, transparent scrim). Currently just `View`.
    pub fn is_view_only(self) -> bool {
        matches!(self, CameraMode::View)
    }

    /// Whether this mode supports the fit-to-view (Contain) preview toggle
    /// and the manual zoom controls. Photo lets you frame a photo; View
    /// just lets you inspect the feed.
    pub fn supports_fit_and_zoom(self) -> bool {
        matches!(self, CameraMode::Photo | CameraMode::View)
    }
}

/// File source for virtual camera streaming
///
/// When set, the virtual camera streams from this file instead of the camera.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileSource {
    /// Stream from an image file (static frame)
    Image(std::path::PathBuf),
    /// Stream from a video file (loops automatically, no audio)
    Video(std::path::PathBuf),
}

/// Application initialization flags
///
/// Results from pre-warming work done before the iced event loop starts.
/// By starting this work before `cosmic::app::run()`, it overlaps with
/// Wayland/wgpu initialization and is ready by the time `init()` runs.
pub struct PrewarmResults {
    pub audio_devices: Vec<crate::backends::audio::AudioDevice>,
    pub video_encoders: Vec<crate::media::encoders::video::EncoderInfo>,
    /// Camera enumeration thread handle. Started before the event loop so it
    /// overlaps with framework Vulkan init. Collected via async Task (not joined
    /// in init()) to avoid blocking the first render.
    pub camera_enum: Option<
        std::thread::JoinHandle<(
            Vec<crate::backends::camera::types::CameraDevice>,
            Vec<crate::backends::camera::types::CameraFormat>,
        )>,
    >,
}

/// These are passed from the command line to configure the app on startup.
#[derive(Default)]
pub struct AppFlags {
    /// Optional file to use as the camera preview source instead of a real camera.
    /// Can be an image (PNG, JPG, JPEG, WEBP) or video (MP4, WEBM, MKV).
    pub preview_source: Option<std::path::PathBuf>,
    /// Preview harness only: start in Video mode with a spoofed active recording
    /// — the recording indicator and a fixed elapsed timer are shown, but no
    /// encoder pipeline is started. Lets the screenshot harness capture the
    /// "recording in progress" state without spending CI resources on a real
    /// encode. See `preview/capture-previews.sh`.
    pub preview_spoof_recording: bool,
    /// Preview harness only: present synthetic camera devices instead of real
    /// enumeration, so the full camera UI renders while frames come from the
    /// file source. Lets the harness run with no camera and no dma-buf provider.
    /// See `crate::backends::camera::synthetic`.
    pub preview_fake_camera: bool,
    /// Pre-warmed results from background thread started before the event loop.
    /// If present, init() skips the synchronous enumeration.
    pub prewarm: Option<std::thread::JoinHandle<PrewarmResults>>,
}

/// Commands for controlling video file playback
#[derive(Debug, Clone)]
pub enum VideoPlaybackCommand {
    /// Seek to a specific position in seconds
    Seek(f64),
    /// Toggle play/pause
    TogglePause,
    /// Set paused state explicitly
    SetPaused(bool),
}

/// Photo timer settings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhotoTimerSetting {
    /// No timer (immediate capture)
    #[default]
    Off,
    /// 3 second countdown
    Sec3,
    /// 5 second countdown
    Sec5,
    /// 10 second countdown
    Sec10,
}

impl PhotoTimerSetting {
    /// Get the number of seconds for this setting
    pub fn seconds(&self) -> u8 {
        match self {
            PhotoTimerSetting::Off => 0,
            PhotoTimerSetting::Sec3 => 3,
            PhotoTimerSetting::Sec5 => 5,
            PhotoTimerSetting::Sec10 => 10,
        }
    }

    /// Cycle to next setting: Off -> 3s -> 5s -> 10s -> Off
    pub fn next(&self) -> Self {
        match self {
            PhotoTimerSetting::Off => PhotoTimerSetting::Sec3,
            PhotoTimerSetting::Sec3 => PhotoTimerSetting::Sec5,
            PhotoTimerSetting::Sec5 => PhotoTimerSetting::Sec10,
            PhotoTimerSetting::Sec10 => PhotoTimerSetting::Off,
        }
    }
}

/// Photo aspect ratio settings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PhotoAspectRatio {
    /// Native camera aspect ratio (no cropping)
    #[default]
    Native,
    /// 4:3 aspect ratio
    Ratio4x3,
    /// 16:9 aspect ratio
    Ratio16x9,
    /// 2:1 aspect ratio (18:9)
    Ratio2x1,
    /// 1:1 square aspect ratio
    Ratio1x1,
}

impl PhotoAspectRatio {
    /// Tolerance for aspect ratio matching (allows for minor pixel rounding differences)
    const RATIO_TOLERANCE: f32 = 0.02;

    /// Detect which defined aspect ratio matches the given frame dimensions
    /// Returns None if the native ratio doesn't match any defined ratio
    pub fn from_frame_dimensions(width: u32, height: u32) -> Option<Self> {
        if height == 0 {
            return None;
        }
        let frame_ratio = width as f32 / height as f32;

        // Check each defined ratio
        if (frame_ratio - 4.0 / 3.0).abs() < Self::RATIO_TOLERANCE {
            Some(PhotoAspectRatio::Ratio4x3)
        } else if (frame_ratio - 16.0 / 9.0).abs() < Self::RATIO_TOLERANCE {
            Some(PhotoAspectRatio::Ratio16x9)
        } else if (frame_ratio - 2.0).abs() < Self::RATIO_TOLERANCE {
            Some(PhotoAspectRatio::Ratio2x1)
        } else if (frame_ratio - 1.0).abs() < Self::RATIO_TOLERANCE {
            Some(PhotoAspectRatio::Ratio1x1)
        } else {
            None
        }
    }

    /// Get the default aspect ratio for given frame dimensions
    /// If native matches a defined ratio, use that; otherwise use Native
    pub fn default_for_frame(width: u32, height: u32) -> Self {
        Self::from_frame_dimensions(width, height).unwrap_or(PhotoAspectRatio::Native)
    }

    /// Cycle to next aspect ratio: Native → 4:3 → 16:9 → 2:1 → 1:1 → Native
    pub fn next_for_frame(&self, _frame_width: u32, _frame_height: u32) -> Self {
        match self {
            PhotoAspectRatio::Native => PhotoAspectRatio::Ratio4x3,
            PhotoAspectRatio::Ratio4x3 => PhotoAspectRatio::Ratio16x9,
            PhotoAspectRatio::Ratio16x9 => PhotoAspectRatio::Ratio2x1,
            PhotoAspectRatio::Ratio2x1 => PhotoAspectRatio::Ratio1x1,
            PhotoAspectRatio::Ratio1x1 => PhotoAspectRatio::Native,
        }
    }

    /// Get the aspect ratio as a float (width / height) in *sensor* /
    /// landscape orientation, or None for Native. Used by the GPU shader
    /// (which works in texture space and rotates separately) and as the
    /// canonical, orientation-independent value for capture math.
    pub fn ratio(&self) -> Option<f32> {
        match self {
            PhotoAspectRatio::Native => None,
            PhotoAspectRatio::Ratio4x3 => Some(4.0 / 3.0),
            PhotoAspectRatio::Ratio16x9 => Some(16.0 / 9.0),
            PhotoAspectRatio::Ratio2x1 => Some(2.0),
            PhotoAspectRatio::Ratio1x1 => Some(1.0),
        }
    }

    /// Aspect ratio in *display* orientation: when the screen is portrait,
    /// the user sees the rotated frame, so a "2:1" preference reads as 1:2
    /// on screen. Display-space callers (canvas overlay, capture-time
    /// crop_for_cover_mode, composition guide, icon picker) must use this
    /// to stay consistent with what the rotated preview shows.
    pub fn display_ratio(&self, is_portrait: bool) -> Option<f32> {
        let r = self.ratio()?;
        Some(if is_portrait { 1.0 / r } else { r })
    }

    /// Calculate crop rectangle for a given frame size.
    /// Returns (x, y, width, height) for the crop region.
    ///
    /// `frame_width` / `frame_height` are in *display* orientation
    /// (rotation-swapped) and `is_portrait` flips the target ratio so a
    /// "2:1" pref crops the screen-portrait frame to a 1:2 region. Callers
    /// in the capture path swap the result back to sensor orientation
    /// before applying it to the raw sensor frame.
    pub fn crop_rect(
        &self,
        frame_width: u32,
        frame_height: u32,
        is_portrait: bool,
    ) -> (u32, u32, u32, u32) {
        let Some(target_ratio) = self.display_ratio(is_portrait) else {
            return (0, 0, frame_width, frame_height);
        };

        let frame_ratio = frame_width as f32 / frame_height as f32;

        if frame_ratio > target_ratio {
            // Frame is wider than target - crop sides
            let new_width = (frame_height as f32 * target_ratio) as u32;
            let x = (frame_width - new_width) / 2;
            (x, 0, new_width, frame_height)
        } else {
            // Frame is taller than target - crop top/bottom
            let new_height = (frame_width as f32 / target_ratio) as u32;
            let y = (frame_height - new_height) / 2;
            (0, y, frame_width, new_height)
        }
    }

    /// Calculate crop UV coordinates for shader use.
    /// Returns (u_min, v_min, u_max, v_max) in 0-1 range, in *texture* space.
    ///
    /// The shader's screen→texture rotation runs before this `mix(crop_min,
    /// crop_max, tex)`, so crop UVs are rotation-independent and computed on
    /// the original sensor dimensions and ratio.
    pub fn crop_uv(&self, frame_width: u32, frame_height: u32) -> Option<(f32, f32, f32, f32)> {
        let Some(target_ratio) = self.ratio() else {
            return None; // Native - no cropping
        };

        let frame_ratio = frame_width as f32 / frame_height as f32;

        if frame_ratio > target_ratio {
            // Frame is wider than target - crop sides
            let scale = target_ratio / frame_ratio;
            let offset = (1.0 - scale) / 2.0;
            Some((offset, 0.0, 1.0 - offset, 1.0))
        } else {
            // Frame is taller than target - crop top/bottom
            let scale = frame_ratio / target_ratio;
            let offset = (1.0 - scale) / 2.0;
            Some((0.0, offset, 1.0, 1.0 - offset))
        }
    }

    /// Get the default aspect ratio accounting for sensor rotation
    /// For 90°/270° rotations, the frame dimensions are swapped after GStreamer applies rotation
    pub fn default_for_frame_with_rotation(
        width: u32,
        height: u32,
        rotation: crate::backends::camera::types::SensorRotation,
    ) -> Self {
        // For 90°/270° rotations, swap dimensions to match the rotated frame
        let (effective_width, effective_height) = if rotation.swaps_dimensions() {
            (height, width)
        } else {
            (width, height)
        };
        Self::default_for_frame(effective_width, effective_height)
    }
}

/// Filter types for camera preview
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterType {
    /// No filter applied (displays as "ORIGINAL")
    #[default]
    Standard,
    /// Black & white / monochrome filter
    Mono,
    /// Sepia tone filter (warm brownish tint)
    Sepia,
    /// Noir filter (high contrast black & white)
    Noir,
    /// Vivid - boosted saturation and contrast
    Vivid,
    /// Cool - blue color temperature shift
    Cool,
    /// Warm - orange/amber color temperature
    Warm,
    /// Fade - lifted blacks with muted colors
    Fade,
    /// Duotone - two-color gradient mapping
    Duotone,
    /// Vignette - darkened edges
    Vignette,
    /// Negative - inverted colors
    Negative,
    /// Posterize - reduced color levels (pop-art)
    Posterize,
    /// Solarize - partially inverted tones
    Solarize,
    /// Chromatic Aberration - RGB channel split
    ChromaticAberration,
    /// Pencil - pencil sketch drawing
    Pencil,
}

impl FilterType {
    /// Get the GPU shader filter code for this filter type.
    ///
    /// Used by GPU shaders to select the appropriate filter function.
    /// These codes must match the filter_mode values in the WGSL shaders.
    #[inline]
    pub fn gpu_filter_code(&self) -> u32 {
        match self {
            FilterType::Standard => 0,
            FilterType::Mono => 1,
            FilterType::Sepia => 2,
            FilterType::Noir => 3,
            FilterType::Vivid => 4,
            FilterType::Cool => 5,
            FilterType::Warm => 6,
            FilterType::Fade => 7,
            FilterType::Duotone => 8,
            FilterType::Vignette => 9,
            FilterType::Negative => 10,
            FilterType::Posterize => 11,
            FilterType::Solarize => 12,
            FilterType::ChromaticAberration => 13,
            FilterType::Pencil => 14,
        }
    }

    /// Whether this filter needs a pre-blur pass for better quality.
    ///
    /// Multi-pass filters get a Gaussian blur applied to the source before
    /// the main filter runs. This reduces sensor noise and produces cleaner
    /// results for filters that rely on spatial operations like edge detection.
    #[inline]
    pub fn needs_preblur(&self) -> bool {
        matches!(self, FilterType::Pencil)
    }

    /// Reconstruct a FilterType from a GPU shader filter code.
    ///
    /// Returns `Standard` for unknown codes.
    #[inline]
    pub fn from_gpu_filter_code(code: u32) -> Self {
        match code {
            1 => FilterType::Mono,
            2 => FilterType::Sepia,
            3 => FilterType::Noir,
            4 => FilterType::Vivid,
            5 => FilterType::Cool,
            6 => FilterType::Warm,
            7 => FilterType::Fade,
            8 => FilterType::Duotone,
            9 => FilterType::Vignette,
            10 => FilterType::Negative,
            11 => FilterType::Posterize,
            12 => FilterType::Solarize,
            13 => FilterType::ChromaticAberration,
            14 => FilterType::Pencil,
            _ => FilterType::Standard,
        }
    }
}

/// The context page to display in the context drawer.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum ContextPage {
    #[default]
    Settings,
    Filters,
    Insights,
    /// Keyboard shortcuts rebinding page (opened from the Settings drawer).
    KeyBindings,
}

/// Which sub-page is shown inside the Settings context drawer.
///
/// The drawer is a drill-down: `Root` shows the theme picker, a short category
/// menu and the reset button, and each other variant is a focused sub-page
/// reached from it with a back button.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum SettingsPage {
    /// Top-level category menu.
    #[default]
    Root,
    Camera,
    Photo,
    Video,
    Timelapse,
    Overlay,
    VirtualCamera,
    BugReports,
    About,
}

/// Transient state while the user is recording a new keybind for an Action.
#[derive(Clone, Debug)]
pub struct RecordingKeyBindState {
    pub action: crate::app::keybind::Action,
    pub captured: Option<cosmic::widget::menu::key_bind::KeyBind>,
    pub conflict_with: Option<crate::app::keybind::Action>,
}

/// Messages emitted by the application and its widgets.
///
/// Messages are organized into logical groups:
/// - **UI Navigation**: Toggle context pages, pickers, and UI states
/// - **Camera Control**: Camera selection, frames, transitions
/// - **Format Selection**: Resolution, framerate, codec, format picker
/// - **Capture Operations**: Photo capture, video recording
/// - **Gallery**: Thumbnail loading, gallery opening
/// - **Filters**: Filter selection and picker
/// - **Settings**: Configuration, audio/video encoder selection
/// - **System**: Bug reports, recovery, external URLs
#[derive(Debug, Clone)]
pub enum Message {
    // ===== UI Navigation =====
    /// Open external URL (repository, etc.)
    LaunchUrl(String),
    /// Toggle context drawer page (Settings, Insights, ...)
    ToggleContextPage(ContextPage),
    /// Navigate to a settings sub-page within the Settings drawer.
    OpenSettingsPage(SettingsPage),
    /// Toggle format picker visibility
    ToggleFormatPicker,
    /// Close format picker
    CloseFormatPicker,
    /// Toggle device info panel visibility
    ToggleDeviceInfo,

    // ===== Tools Menu =====
    /// Toggle tools menu visibility
    ToggleToolsMenu,
    /// Close tools menu (click outside)
    CloseToolsMenu,

    // ===== Exposure Controls =====
    /// Toggle exposure picker visibility
    ToggleExposurePicker,
    /// Close exposure picker (click outside)
    CloseExposurePicker,
    /// Set exposure mode (Auto, Manual, Shutter Priority, Aperture Priority)
    SetExposureMode(ExposureMode),
    /// Set exposure compensation (EV bias) - value in 0.001 EV units
    SetExposureCompensation(i32),
    /// Reset exposure compensation to 0 and return to aperture priority mode
    ResetExposureCompensation,
    /// Set exposure time (100us units, only in manual mode)
    SetExposureTime(i32),
    /// Set gain value
    SetGain(i32),
    /// Set ISO sensitivity
    SetIsoSensitivity(i32),
    /// Set metering mode
    SetMeteringMode(MeteringMode),
    /// Toggle auto exposure priority (allow frame rate variation)
    ToggleAutoExposurePriority,
    /// Exposure controls queried from camera (boxed to reduce enum size)
    ExposureControlsQueried(
        Box<AvailableExposureControls>,
        ExposureSettings,
        ColorSettings,
    ),
    /// Exposure control change applied successfully
    ExposureControlApplied,
    /// White balance toggled, with optional temperature value when switching to manual
    WhiteBalanceToggled(Option<i32>),
    /// Exposure control change failed
    ExposureControlFailed(String),
    /// Base exposure time captured (for non-advanced EV slider)
    ExposureBaseTimeCaptured(i32),
    /// Set backlight compensation value
    SetBacklightCompensation(i32),
    /// Set manual focus position (absolute)
    SetFocusAbsolute(i32),
    /// Toggle auto focus on/off
    ToggleFocusAuto,
    /// Reset all exposure settings to defaults
    ResetExposureSettings,
    /// Exposure mode selected via segmented button
    ExposureModeSelected(cosmic::widget::segmented_button::Entity),

    // ===== Color Controls =====
    /// Toggle color picker visibility
    ToggleColorPicker,
    /// Close color picker (click outside)
    CloseColorPicker,
    /// Set contrast value
    SetContrast(i32),
    /// Set saturation value
    SetSaturation(i32),
    /// Set sharpness value
    SetSharpness(i32),
    /// Set hue value
    SetHue(i32),
    /// Toggle auto white balance
    ToggleAutoWhiteBalance,
    /// Set white balance temperature (Kelvin)
    SetWhiteBalanceTemperature(i32),
    /// Reset all color settings to defaults
    ResetColorSettings,

    // ===== Camera Control =====
    /// Switch to next camera
    SwitchCamera,
    /// Select specific camera by index
    SelectCamera(usize),
    /// New camera frame received from pipeline
    CameraFrame(Arc<CameraFrame>),
    /// Cameras initialized asynchronously during startup
    CamerasInitialized(
        Vec<crate::backends::camera::types::CameraDevice>,
        usize,
        Vec<crate::backends::camera::types::CameraFormat>,
    ),
    /// Broken video encoders detected by background probe
    BrokenEncodersDetected(Vec<String>),
    /// Camera list changed (hotplug event)
    CameraListChanged(Vec<crate::backends::camera::types::CameraDevice>),
    /// Device nodes in /dev/video* were removed — a camera was unplugged.
    /// Contains the removed node names (e.g. `["video8", "video9"]`).
    /// Only stops capture if the current camera was affected.
    HotplugDeviceRemoved(Vec<String>),
    /// New /dev/video* device nodes appeared — a camera was plugged in.
    /// Contains lightweight V4L2 device info (path, card name) for the new nodes.
    /// Does NOT stop the current capture stream.
    HotplugDeviceAdded(Vec<(String, String)>),
    /// Audio device list changed (hotplug event)
    AudioListChanged(Vec<crate::backends::audio::AudioDevice>),
    /// Start camera transition (capture last frame and show blur)
    StartCameraTransition,
    /// Clear blur transition after delay
    ClearTransitionBlur,
    /// Toggle mirror preview (horizontal flip)
    ToggleMirrorPreview,
    /// Toggle whether the same mirroring also applies to captured media
    /// (photos / videos / timelapse output).
    ToggleMirrorCaptures,
    /// Toggle haptic feedback
    ToggleHapticFeedback,

    // ===== Motor/PTZ Controls =====
    /// Toggle motor controls picker visibility
    ToggleMotorPicker,
    /// Close motor controls picker
    CloseMotorPicker,
    /// Set V4L2 pan absolute position
    SetPanAbsolute(i32),
    /// Set V4L2 tilt absolute position (V4L2 cameras)
    SetTiltAbsolute(i32),
    /// Set V4L2 zoom absolute position
    SetZoomAbsolute(i32),
    /// Reset pan/tilt to center position
    ResetPanTilt,

    // ===== Format Selection =====
    /// Switch between Photo/Video mode
    SetMode(CameraMode),
    /// Select mode from dropdown by index
    SelectMode(usize),
    /// Select pixel format from dropdown
    SelectPixelFormat(String),
    /// Select resolution from dropdown
    SelectResolution(String),
    /// Select framerate from dropdown
    SelectFramerate(String),
    /// Select codec from dropdown
    SelectCodec(String),
    /// Select resolution in picker (grouped view)
    PickerSelectResolution(u32),
    /// Select specific format in picker
    PickerSelectFormat(usize),
    /// Select bitrate preset
    SelectBitratePreset(usize),

    // ===== Capture Operations =====
    /// Capture photo
    Capture,
    /// Toggle flash for photo capture
    ToggleFlash,
    /// Dismiss flash permission error popup
    DismissFlashError,
    /// Toggle burst mode for photo capture (multi-frame HDR+ burst)
    ToggleBurstMode,
    /// Set burst mode frame count (0 = Auto, 1 = 4 frames, 2 = 6 frames, 3 = 8 frames)
    SetBurstModeFrameCount(usize),
    /// Burst mode capture progress update (overall_progress 0.0-1.0)
    BurstModeProgress(f32),
    /// Burst mode frames collected, ready to process
    BurstModeFramesCollected,
    /// Burst mode raw frames captured via capture_photo() (multistream mode)
    BurstModeRawFramesCaptured(Result<Vec<std::sync::Arc<CameraFrame>>, String>),
    /// Burst mode capture complete (path or error)
    BurstModeComplete(Result<String, String>),
    /// Poll burst mode processing progress (timer-based)
    PollBurstModeProgress,
    /// Reset burst mode state after completion/error
    ResetBurstModeState,
    /// Periodic brightness evaluation tick (every 1 second in Auto mode)
    /// Updates auto_detected_frame_count based on scene brightness
    BrightnessEvaluationTick,
    /// Cycle photo aspect ratio (native -> 4:3 -> 16:9 -> 1:1 -> native)
    CyclePhotoAspectRatio,
    /// Flash duration complete, now capture the photo
    FlashComplete,
    /// Cycle photo timer setting (off -> 3s -> 5s -> 10s -> off)
    CyclePhotoTimer,
    /// Photo timer tick (countdown)
    PhotoTimerTick,
    /// Photo timer animation frame (for fade effect)
    PhotoTimerAnimationFrame,
    /// Abort photo timer countdown
    AbortPhotoTimer,
    /// Zoom in (increase zoom level)
    ZoomIn,
    /// Zoom out (decrease zoom level)
    ZoomOut,
    /// Reset zoom to 1.0
    ResetZoom,
    /// Toggle between Cover (fill) and Contain (fit) preview mode
    TogglePreviewFit,
    /// Show/hide all overlay chrome, leaving just the live preview
    ToggleUiChrome,
    /// Animation tick for fit/fill transition
    FitAnimationTick,
    /// Animation tick for the zoom-reset transition
    ZoomAnimationTick,
    /// Wayland keyboard-focus changed for the app window. Toggles the
    /// volume-key `EVIOCGRAB` and dispatch gate so we only consume the
    /// hardware shutter buttons while the camera is in focus.
    WindowFocusChanged(bool),
    /// Compositor reported a new display transform (phone rotated, or
    /// initial output geometry at startup). Stored on `AppModel` and
    /// composed with the sensor mounting rotation at preview time so
    /// the preview reorients with the screen.
    DisplayOrientationChanged(crate::backends::display_orientation::DisplayOrientation),
    /// Window control: close
    WindowClose,
    /// Window control: minimize
    WindowMinimize,
    /// Window control: toggle maximize
    WindowToggleMaximize,
    /// Window control: start drag
    WindowDrag,
    /// Pinch-to-zoom: set absolute zoom level from touch gesture
    PinchZoom(f32),
    /// Photo was saved successfully with the given file path
    PhotoSaved(Result<String, String>),
    /// Clear capture animation after brief delay
    ClearCaptureAnimation,
    /// Toggle video recording
    ToggleRecording,
    /// Video recording started successfully
    RecordingStarted(String),
    /// Video recording stopped successfully. `session` matches the
    /// `RecordingState::Recording.session` assigned at start; the handler
    /// uses it to drop stale events from an already-superseded session.
    RecordingStopped {
        session: u64,
        result: Result<String, String>,
    },
    /// Update recording duration (every second)
    UpdateRecordingDuration,
    /// Start recording after camera is released
    StartRecordingAfterDelay,
    /// Capture button pressed down (for quick-record state machine)
    CaptureButtonPressed,
    /// Capture button released (for quick-record state machine)
    CaptureButtonReleased,
    /// Long-press threshold reached — start quick recording
    QuickRecordThreshold,

    // ===== Virtual Camera =====
    /// Toggle virtual camera streaming (start/stop)
    ToggleVirtualCamera,
    /// Virtual camera streaming started successfully
    VirtualCameraStarted,
    /// Virtual camera streaming stopped
    VirtualCameraStopped(Result<(), String>),
    /// Update virtual camera streaming duration (every second)
    UpdateVirtualCameraDuration,
    /// Open file picker to select image/video for virtual camera
    OpenVirtualCameraFile,
    /// File selected for virtual camera streaming
    VirtualCameraFileSelected(Option<FileSource>),
    /// Clear the virtual camera file source (use camera instead)
    ClearVirtualCameraFile,
    /// File source preview frame loaded (for displaying before streaming starts)
    /// For videos, includes optional duration in seconds
    FileSourcePreviewLoaded(Option<Arc<CameraFrame>>, Option<f64>),
    /// Video file source playback progress update (position_secs, duration_secs, progress 0.0-1.0)
    VideoFileProgress(f64, f64, f64),
    /// Seek video file to a specific position in seconds
    VideoFileSeek(f64),
    /// Preview frame loaded after seeking while not streaming
    VideoSeekPreviewLoaded(Option<Arc<CameraFrame>>),
    /// Preview playback frame update (frame and progress info)
    VideoPreviewPlaybackUpdate(Arc<CameraFrame>, f64, f64, f64),
    /// Preview playback stopped (thread finished)
    VideoPreviewPlaybackStopped,
    /// Toggle video file play/pause
    ToggleVideoPlayPause,
    /// Start video preview playback (triggered after streaming stops)
    StartVideoPreviewPlayback,

    // ===== Gallery =====
    /// Open gallery in file manager
    OpenGallery,
    /// Refresh the gallery thumbnail
    RefreshGalleryThumbnail,
    /// Gallery thumbnail loaded
    GalleryThumbnailLoaded(Option<crate::storage::GalleryThumbnailData>),

    // ===== Filters =====
    /// Select a filter
    SelectFilter(FilterType),

    // ===== Settings & Device Selection =====
    /// Configuration updated
    UpdateConfig(Config),
    /// Set application theme (System, Dark, Light)
    SetAppTheme(usize),
    /// Set the overlay effect. Index into `OverlayEffect::available()`.
    SetOverlayEffect(usize),
    /// Set the capture-controls panel position (Bottom, Left, Right).
    SetControlsPosition(usize),
    /// Set default camera mode on launch
    SelectDefaultMode(usize),
    /// XDG portal color scheme changed (true = dark, false = light)
    PortalColorSchemeChanged(bool),
    /// COSMIC theme config (light/dark palette or active mode) changed —
    /// re-apply so accent / palette edits propagate without a manual toggle.
    CosmicThemeChanged,
    /// Select audio input device
    SelectAudioDevice(usize),
    /// Select video encoder
    SelectVideoEncoder(usize),
    /// Select photo output format (JPEG, PNG, DNG)
    SelectPhotoOutputFormat(usize),
    /// Toggle recording audio with video
    ToggleRecordAudio,
    /// Fired by the 100 ms subscription whenever a level source is active.
    AudioLevelTick,
    /// Select audio encoder (Opus, AAC)
    SelectAudioEncoder(usize),
    /// Toggle saving raw burst frames as DNG (debugging feature)
    ToggleSaveBurstRaw,
    /// Select composition guide overlay by dropdown index
    SelectCompositionGuide(usize),
    /// Reset all settings to defaults
    ResetAllSettings,
    /// Toggle virtual camera feature enabled
    ToggleVirtualCameraEnabled,

    // ===== Timelapse =====
    /// Start/stop timelapse capture
    ToggleTimelapse,
    /// Timelapse interval timer tick — capture next frame
    TimelapseTick,
    /// Cycle to the next camera mode (swipe left)
    NextMode,
    /// Cycle to the previous camera mode (swipe right)
    PrevMode,
    /// Set timelapse interval from dropdown
    SetTimelapseInterval(usize),
    /// Timelapse video assembly completed (path or error)
    TimelapseAssemblyComplete(Result<String, String>),

    // ===== System & Recovery =====
    /// Camera backend recovery started
    CameraRecoveryStarted { attempt: u32, max_attempts: u32 },
    /// Camera backend recovery succeeded
    CameraRecoverySucceeded,
    /// Camera backend recovery failed
    CameraRecoveryFailed(String),
    /// Audio backend recovery started
    AudioRecoveryStarted { attempt: u32, max_attempts: u32 },
    /// Audio backend recovery succeeded
    AudioRecoverySucceeded,
    /// Audio backend recovery failed
    AudioRecoveryFailed(String),
    /// Generate bug report
    GenerateBugReport,
    /// Bug report generated successfully with path
    BugReportGenerated(Result<String, String>),
    /// Show bug report in file manager
    ShowBugReport,

    // ===== QR Code Detection =====
    /// Toggle QR code detection on/off
    ToggleQrDetection,
    /// QR detection results updated
    QrDetectionsUpdated(Vec<QrDetection>),
    /// Open URL from QR code
    QrOpenUrl(String),
    /// Connect to WiFi network from QR code
    QrConnectWifi {
        ssid: String,
        password: Option<String>,
        security: String,
        hidden: bool,
    },
    /// Copy text from QR code to clipboard
    QrCopyText(String),

    // ===== Privacy Cover Detection =====
    /// Privacy cover status changed (true = cover closed/camera blocked)
    PrivacyCoverStatusChanged(bool),

    // ===== Insights Drawer =====
    /// Update insights metrics from pipeline
    UpdateInsightsMetrics,
    /// Copy pipeline string to clipboard
    CopyPipelineString,
    /// Capture single frame from all running streams (raw .bin + metadata JSON)
    InsightsCaptureFrames,
    /// Capture burst (6 frames) from all running streams (raw .bin + metadata JSON)
    InsightsCaptureBurst,
    /// Insights capture complete (list of saved file paths, or error)
    InsightsCaptureComplete(Result<Vec<String>, String>),

    /// GPU shader pipelines precompiled at startup
    GpuPipelinesWarmed(Result<(), String>),

    // ===== Keyboard shortcuts =====
    /// Open the keyboard-shortcuts rebinding page (a context drawer).
    OpenKeyBindingsPage,
    /// Triggered by the Esc key when libcosmic's keyboard_nav is disabled.
    Escape,
    /// Begin capturing a new keybind for the given action.
    StartRecordingKeyBind(crate::app::keybind::Action),
    /// A key was captured during recording.
    KeyBindRecordingCaptured(cosmic::widget::menu::key_bind::KeyBind),
    /// User cancelled the record-keybind dialog.
    CancelKeyBindRecording,
    /// User confirmed the recorded combo — save (optionally unbinding the conflict).
    KeyBindSave,
    /// Reset a single action's binding to its default.
    ResetKeyBindToDefault(crate::app::keybind::Action),
    /// Reset every action's binding to its default.
    ResetAllKeyBindings,

    /// No-op message for async tasks that don't need a response
    Noop,
}

impl TransitionState {
    /// Start a transition - enable blur, disable UI, and wait for first frame.
    ///
    /// The hold runs *after* the new camera's first frame arrives, and the UI
    /// stays disabled for all of it, so it is added to every camera switch in
    /// full. It only needs to cover auto-exposure settling, not the pipeline
    /// restart (the blur is already up for that).
    pub fn start(&mut self) -> cosmic::Task<Message> {
        self.start_with_duration(crate::constants::latency::CAMERA_SWITCH_BLUR_HOLD_MS, true)
    }

    /// Start a transition with custom blur duration
    /// Used for HDR+ completion (shorter blur) vs camera switches (longer blur)
    pub fn start_with_duration(
        &mut self,
        duration_ms: u64,
        disable_ui: bool,
    ) -> cosmic::Task<Message> {
        self.in_transition = true;
        self.ui_disabled = disable_ui;
        self.blur_duration_ms = duration_ms;
        self.transition_start_time = Some(Instant::now());
        self.first_frame_time = None; // Reset - waiting for first new frame

        cosmic::Task::none()
    }

    /// Called when a new frame arrives during transition.
    /// Only starts the blur countdown when the frame is from the NEW camera
    /// (captured after the transition started), so stale queued frames from the
    /// old camera don't prematurely trigger the countdown.
    pub fn on_frame_received(&mut self, captured_at: Instant) -> Option<cosmic::Task<Message>> {
        if !self.in_transition {
            return None;
        }

        // Only start the blur countdown for frames from the new camera
        let is_new_frame = self.transition_start_time.is_none_or(|t| captured_at >= t);

        if self.first_frame_time.is_none() && is_new_frame {
            self.first_frame_time = Some(Instant::now());

            // Schedule blur removal after configured duration from NOW
            let duration_ms = self.blur_duration_ms;
            return Some(cosmic::Task::perform(
                async move {
                    tokio::time::sleep(std::time::Duration::from_millis(duration_ms)).await;
                },
                |_| Message::ClearTransitionBlur,
            ));
        }

        None
    }

    /// Check if blur should still be active
    pub fn should_blur(&self) -> bool {
        if !self.in_transition {
            return false;
        }

        // If we haven't received a frame yet, keep blurring the old frame
        // (or show black if no old frame exists)
        let Some(first_frame_time) = self.first_frame_time else {
            return true;
        };

        // Once first frame arrives, blur for configured duration
        first_frame_time.elapsed() < std::time::Duration::from_millis(self.blur_duration_ms)
    }

    /// Clear the blur and end transition.
    /// Preserves `transition_start_time` so post-transition stale frames can still be filtered.
    pub fn clear(&mut self) {
        self.in_transition = false;
        self.ui_disabled = false; // Re-enable UI
        // Keep transition_start_time — used by handle_camera_frame to drop stale frames
        self.first_frame_time = None;
    }
}

impl AppModel {
    /// The most recent audio levels snapshot, or `None` if neither a
    /// recording nor a probe is producing levels right now.
    pub fn current_audio_levels(&self) -> Option<&crate::pipelines::audio_level::AudioLevels> {
        self.audio_levels_snapshot.as_ref()
    }

    /// Start, stop, or restart the audio-level probe to match the current
    /// app state. Idempotent — safe to call from any handler whose work
    /// might change `record_audio`, the selected device, the recording
    /// state, or the settings-drawer visibility.
    pub fn sync_audio_probe(&mut self) {
        let drawer_open =
            self.context_page == ContextPage::Settings && self.core.window.show_context;
        let want = drawer_open && self.config.record_audio && !self.recording.is_recording();

        let selected_device = self
            .available_audio_devices
            .get(self.current_audio_device_index);
        let desired_device: Option<String> = selected_device.map(|d| d.node_name.clone());
        let desired_rate_hz: u32 = selected_device.map(|d| d.sample_rate).unwrap_or(0);

        // Tear down if running and no longer wanted, or if the device changed.
        if let Some(probe) = self.audio_probe.take() {
            let device_changed = probe.device().map(str::to_string) != desired_device;
            if !want || device_changed {
                probe.stop();
                self.probe_audio_levels = None;
            } else {
                // Still valid — put it back.
                self.audio_probe = Some(probe);
            }
        }

        // Start if wanted and not running.
        if want && self.audio_probe.is_none() {
            match crate::pipelines::audio_probe::AudioLevelProbe::start(
                desired_device.as_deref(),
                desired_rate_hz,
            ) {
                Ok(probe) => {
                    self.probe_audio_levels = Some(probe.levels());
                    self.audio_probe = Some(probe);
                }
                Err(e) => {
                    tracing::info!(error = %e, "Audio level probe failed to start");
                    self.probe_audio_levels = None;
                }
            }
        }
    }
}
