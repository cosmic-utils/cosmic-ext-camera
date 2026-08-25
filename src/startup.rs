// SPDX-License-Identifier: GPL-3.0-only

//! Process-wide startup timing markers.
//!
//! All milestones use the same monotonic clock so startup benchmarks can parse
//! one structured log stream without relying on wall-clock timestamps.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

static PROCESS_START: OnceLock<Instant> = OnceLock::new();

/// Start the process-wide startup clock as early as possible.
pub fn start() {
    let _ = PROCESS_START.set(Instant::now());
}

/// Milliseconds elapsed since [`start`].
pub fn elapsed_ms() -> u128 {
    PROCESS_START
        .get_or_init(Instant::now)
        .elapsed()
        .as_millis()
}

/// Emit a structured startup milestone on the shared monotonic timeline.
pub fn milestone(name: &'static str) {
    tracing::info!(
        target: "camera::startup",
        milestone = name,
        elapsed_ms = elapsed_ms(),
        "startup milestone"
    );
}

/// Emit a milestone only on its first occurrence in this process.
pub fn milestone_once(name: &'static str, emitted: &AtomicBool) {
    if !emitted.swap(true, Ordering::Relaxed) {
        milestone(name);
    }
}
