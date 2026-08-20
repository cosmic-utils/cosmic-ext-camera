// SPDX-License-Identifier: GPL-3.0-only

//! Backend abstraction layer for camera and audio capture
//!
//! This module provides platform-specific backend implementations for:
//! - Camera capture via libcamera
//! - Audio device enumeration via PipeWire
//! - Virtual camera output via PipeWire
//!
//! # Architecture
//!
//! The backend layer abstracts hardware access, providing a consistent API
//! regardless of the underlying capture method:
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │                  App Layer                   │
//! └────────────────────┬────────────────────────┘
//!                      │
//! ┌────────────────────┴────────────────────────┐
//! │              Backend Layer                   │
//! │  ┌─────────────┐    ┌──────────────────┐   │
//! │  │    Audio    │    │     Camera       │   │
//! │  │  (PipeWire) │    │   (libcamera)    │   │
//! │  └─────────────┘    └──────────────────┘   │
//! │                     ┌──────────────────┐   │
//! │                     │ Virtual Camera   │   │
//! │                     │   (PipeWire)     │   │
//! │                     └──────────────────┘   │
//! └─────────────────────────────────────────────┘
//! ```
//!
//! # Modules
//!
//! - [`audio`]: Audio device enumeration and selection
//! - [`camera`]: Camera backend with device enumeration and frame capture
//! - [`virtual_camera`]: Virtual camera sink for streaming filtered video

pub mod audio;
pub mod camera;
pub mod display_orientation;
pub mod haptic;
pub mod virtual_camera;
pub mod volume_keys;
