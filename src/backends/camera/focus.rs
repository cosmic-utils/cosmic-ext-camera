// SPDX-License-Identifier: GPL-3.0-only
//! Autofocus trigger shared between the UI and the capture thread.

use std::sync::atomic::{AtomicBool, Ordering};

/// One-shot autofocus scan request. The UI requests a scan (for example
/// when the shutter is pressed); the capture thread puts `AfTrigger::Start`
/// on the next request it queues.
#[derive(Debug, Default)]
pub struct FocusControl {
    trigger: AtomicBool,
}

impl FocusControl {
    /// Ask for an autofocus scan on the next queued request.
    pub fn request_scan(&self) {
        self.trigger.store(true, Ordering::Release);
    }

    /// Take a pending scan request, clearing it.
    pub fn take_scan_request(&self) -> bool {
        self.trigger.swap(false, Ordering::AcqRel)
    }
}

#[cfg(test)]
mod tests {
    use super::FocusControl;

    #[test]
    fn scan_request_is_taken_once() {
        let focus = FocusControl::default();
        assert!(!focus.take_scan_request());
        focus.request_scan();
        assert!(focus.take_scan_request());
        assert!(!focus.take_scan_request());
    }
}
