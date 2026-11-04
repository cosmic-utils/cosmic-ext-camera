// SPDX-License-Identifier: GPL-3.0-only

//! Native libcamera pipeline using libcamera-rs bindings
//!
//! Replaces the GStreamer-based multi-stream pipeline with direct libcamera access,
//! providing:
//! - Direct buffer lifecycle control (no GStreamer overhead)
//! - Per-frame metadata (exposure, gain, colour temperature)
//! - Direct camera property access (model, location, rotation)
//!
//! # Architecture
//!
//! All libcamera objects (CameraManager, ActiveCamera, FrameBuffers) live on a
//! dedicated capture thread. This avoids Send issues with libcamera's raw pointers
//! and keeps the entire pipeline lifecycle on one thread.
//!
//! ```text
//! ┌──────────────────────┐         ┌───────────────────────┐
//! │  NativeLibcamera     │         │   Capture Thread      │
//! │  Pipeline (main)     │         │                       │
//! │                      │  init   │  CameraManager        │
//! │  stop_flag ──────────┼────────►│  ActiveCamera         │
//! │  still_capture ─────┼────────►│  FrameBuffers         │
//! │                      │         │  Request loop         │
//! │  still_capture ◄────┼─────────│                       │
//! │  frame_sender   ◄────┼─────────│  (all libcamera ops)  │
//! └──────────────────────┘         └───────────────────────┘
//! ```

mod capture_thread;
pub mod diagnostics;
pub(crate) mod pixel_formats;
mod scaler_crop;

pub use diagnostics::is_capture_active;

use crate::backends::camera::preview_metrics::PreviewMetrics;
use crate::backends::camera::still_capture::StillCapture;
use crate::backends::camera::types::*;
use capture_thread::{CaptureThreadInitResult, CaptureThreadParams, capture_thread_main};
use diagnostics::clear_global_diagnostics;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use tracing::{error, info};

/// Shared communication handles passed into the native pipeline
pub(crate) struct PipelineSharedState {
    pub(crate) zoom: Arc<crate::backends::camera::zoom::ZoomControl>,
    pub(crate) frame_sender: FrameSender,
    pub(crate) still_capture: Arc<StillCapture>,
    pub(crate) recording_sender: Arc<Mutex<Option<tokio::sync::mpsc::Sender<RecordingFrame>>>>,
    /// Cancel flag from the subscription — allows the capture thread to abort
    /// before creating a CameraManager if a newer mode switch superseded this one.
    pub(crate) cancel_flag: Arc<AtomicBool>,
}

/// Native libcamera pipeline using direct libcamera-rs bindings
///
/// All libcamera objects live on a dedicated capture thread.
/// The main thread communicates via atomic flags and mutexed shared state.
pub(crate) struct NativeLibcameraPipeline {
    pub(crate) preview_metrics: Arc<PreviewMetrics>,
    /// Capture processing thread (owns all libcamera objects)
    capture_thread: Option<JoinHandle<()>>,

    /// Stop flag for capture thread
    stop_flag: Arc<AtomicBool>,
    /// Preview frame counter
    preview_frame_count: Arc<AtomicU64>,
    /// Still frame counter
    still_frame_count: Arc<AtomicU64>,
    /// Dynamically-settable sender for direct recording path (bypasses UI thread).
    /// Set to Some(tx) when recording starts, None when recording stops.
    /// Retain channel ownership for the pipeline's lifetime, including after an
    /// unexpected capture-thread exit.
    _recording_sender: Arc<Mutex<Option<tokio::sync::mpsc::Sender<RecordingFrame>>>>,
}

impl NativeLibcameraPipeline {
    /// Create and start a new native libcamera pipeline
    ///
    /// # Arguments
    /// * `camera_id` - libcamera camera ID (from enumeration)
    /// * `preview_format` - Requested capture resolution, retained on single-stream fallback
    /// * `limit_preview` - Cap a dual-stream viewfinder at 1080p (GUI only)
    /// * `shared` - Shared communication handles (frame sender, still capture, recording)
    pub(crate) fn new(
        camera_id: &str,
        preview_format: &CameraFormat,
        limit_preview: bool,
        shared: PipelineSharedState,
    ) -> BackendResult<Self> {
        info!(
            camera = camera_id,
            preview = %preview_format,
            limit_preview,
            "Creating native libcamera pipeline"
        );

        let preview_metrics = PreviewMetrics::start(camera_id);

        // Shared state
        let stop_flag = Arc::new(AtomicBool::new(false));
        let preview_frame_count = Arc::new(AtomicU64::new(0));
        let still_frame_count = Arc::new(AtomicU64::new(0));

        // Channel for thread to report initialization result
        let (init_tx, init_rx) =
            std::sync::mpsc::sync_channel::<Result<CaptureThreadInitResult, BackendError>>(1);

        let params = CaptureThreadParams {
            zoom: Arc::clone(&shared.zoom),
            camera_id: camera_id.to_string(),
            preview_metrics: Arc::clone(&preview_metrics),
            preview_width: preview_format.width,
            preview_height: preview_format.height,
            limit_preview,
            stop_flag: Arc::clone(&stop_flag),
            still_capture: Arc::clone(&shared.still_capture),
            preview_frame_count: Arc::clone(&preview_frame_count),
            still_frame_count: Arc::clone(&still_frame_count),
            frame_sender: shared.frame_sender,
            recording_sender: Arc::clone(&shared.recording_sender),
            cancel_flag: Arc::clone(&shared.cancel_flag),
        };

        // Spawn capture thread - it owns all libcamera objects
        let capture_thread = std::thread::Builder::new()
            .name("libcamera-capture".to_string())
            .spawn(move || {
                capture_thread_main(params, init_tx);
            })
            .map_err(|e| {
                BackendError::InitializationFailed(format!("Spawn capture thread: {}", e))
            })?;

        // Wait for initialization result from capture thread
        let init_result = init_rx.recv().map_err(|_| {
            BackendError::InitializationFailed(
                "Capture thread died during initialization".to_string(),
            )
        })??;

        info!(
            multistream = init_result.is_multistream,
            "Native libcamera pipeline started"
        );

        super::record_stream_capability(camera_id, init_result.is_multistream);

        Ok(Self {
            capture_thread: Some(capture_thread),

            preview_metrics,
            stop_flag,
            preview_frame_count,
            still_frame_count,
            _recording_sender: shared.recording_sender,
        })
    }

    /// Get frame counts for debugging
    fn frame_counts(&self) -> (u64, u64) {
        (
            self.preview_frame_count.load(Ordering::Relaxed),
            self.still_frame_count.load(Ordering::Relaxed),
        )
    }
}

impl Drop for NativeLibcameraPipeline {
    fn drop(&mut self) {
        self.preview_metrics.stop();
        clear_global_diagnostics();
        self.stop_flag.store(true, Ordering::Release);

        // Wait for capture thread to finish (it will stop camera and drop all libcamera objects).
        // Hardware release delay is handled by the *new* capture thread (if any) to avoid
        // blocking the UI thread here.
        if let Some(thread) = self.capture_thread.take()
            && let Err(e) = thread.join()
        {
            error!("Capture thread panicked: {:?}", e);
        }

        let (preview_count, still_count) = self.frame_counts();
        info!(
            preview_frames = preview_count,
            still_frames = still_count,
            "Native libcamera pipeline dropped"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pipeline_without_hardware(metrics: Arc<PreviewMetrics>) -> NativeLibcameraPipeline {
        NativeLibcameraPipeline {
            preview_metrics: metrics,

            capture_thread: None,
            stop_flag: Arc::new(AtomicBool::new(false)),
            preview_frame_count: Arc::new(AtomicU64::new(0)),
            still_frame_count: Arc::new(AtomicU64::new(0)),
            _recording_sender: Arc::new(Mutex::new(None)),
        }
    }

    #[test]
    fn native_pipeline_retains_recording_sender_after_worker_exit() {
        let metrics = PreviewMetrics::start("native-recording-owner-test");
        let pipeline = pipeline_without_hardware(metrics);
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        *pipeline._recording_sender.lock().unwrap() = Some(tx);
        // The fixture has no running worker: pipeline ownership alone keeps
        // the recording channel alive, including after an unexpected exit.
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        drop(pipeline);
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
        ));
    }

    #[test]
    fn native_pipeline_drop_clears_preview_session_with_retained_tracker() {
        let metrics = PreviewMetrics::start("native-pipeline-drop-test");
        let pipeline = pipeline_without_hardware(Arc::clone(&metrics));
        let stop_flag = Arc::clone(&pipeline.stop_flag);
        drop(pipeline);
        assert!(stop_flag.load(Ordering::Acquire));
        assert!(PreviewMetrics::active_for("native-pipeline-drop-test").is_none());
    }
}
