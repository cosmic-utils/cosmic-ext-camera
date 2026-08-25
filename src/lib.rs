// SPDX-License-Identifier: GPL-3.0-only

//! Camera - Modern camera app for Linux desktops and phones
//!
//! This library provides the core functionality for the Camera application,
//! including camera capture, video recording, and photo processing.
//!
//! # Architecture
//!
//! The crate is organized into several modules:
//!
//! - [`app`]: Main application logic and UI
//! - [`backends`]: Camera and audio backend abstraction
//! - [`media`]: Media encoding, decoding, and color conversion
//! - [`pipelines`]: Photo and video capture pipelines
//! - [`config`]: User configuration handling
//! - [`storage`]: File storage and thumbnail management
//!
//! # Example
//!
//! ```ignore
//! // This is a GUI application, typically run via:
//! // camera
//! ```

pub mod app;
pub mod backends;
pub mod bug_report;
pub mod config;
pub mod constants;
pub mod errors;
pub mod flash;
pub mod gpu;
pub mod i18n;
pub mod media;
pub mod network_manager;
pub mod pipelines;
pub mod shaders;
pub mod startup;
pub mod storage;
pub mod terminal;
#[cfg(test)]
pub(crate) mod test_gpu;

// Re-export commonly used types
pub use app::frame_processor::{QrAction, QrDetection};
pub use app::{AppModel, CameraMode, FilterType, Message};
pub use config::Config;
pub use constants::BitratePreset;
