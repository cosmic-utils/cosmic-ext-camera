// SPDX-License-Identifier: GPL-3.0-only
// Camera backend using native libcamera-rs bindings

//! Camera backend abstraction
//!
//! Native libcamera is preferred. Processed V4L2 cameras it does not enumerate
//! are discovered and streamed through the user's PipeWire session.
//!
//! Capture pipelines own camera lifecycle and acquisition. [`RecordingBridge`]
//! keeps the recording channel shared across pipeline restarts.

pub mod libcamera;
pub(crate) mod pipeline;
pub(crate) mod pipewire;
pub(crate) mod preview_metrics;
pub mod recording;
pub(crate) mod still_capture;
pub mod synthetic;
pub mod types;
mod usb_name;
pub mod v4l2_controls;
pub mod v4l2_utils;
pub mod zoom;

pub use recording::RecordingBridge;
pub use types::*;

/// Camera discovery and format queries. Capture lifecycle belongs to pipelines.
pub trait CameraBackend: Send + Sync {
    /// Enumerate available cameras on this backend.
    fn enumerate_cameras(&self) -> Vec<CameraDevice>;

    /// Query supported formats, optionally restricted to video recording.
    fn get_formats(&self, device: &CameraDevice, video_mode: bool) -> Vec<CameraFormat>;
}

/// Create a new backend instance
pub fn create_backend() -> Box<dyn CameraBackend> {
    Box::new(libcamera::LibcameraBackend::new())
}
