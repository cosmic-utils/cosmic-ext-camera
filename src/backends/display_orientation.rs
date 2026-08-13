// SPDX-License-Identifier: GPL-3.0-only

//! Display orientation backend.
//!
//! This backend uses a private Wayland connection to read
//! `wl_output.geometry.transform`. Wayland object identities are local to a
//! connection, so the iced/libcosmic `wl_surface` cannot be correlated with
//! outputs bound here. Consequently, automatic orientation is enabled only
//! while exactly one transform-capable output exists. With multiple outputs
//! the backend fails safe to [`DisplayOrientation::Rotate0`] rather than
//! choosing an arbitrary output.
//!
//! Reflected output transforms cannot be represented by the application's
//! rotation-only preview/capture pipeline. They are explicitly rejected and
//! fall back to `Rotate0`; they are never silently collapsed to a rotation.

use std::collections::HashMap;
use std::os::fd::{AsFd, AsRawFd};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tracing::{debug, info, warn};
use wayland_client::protocol::{wl_callback, wl_output, wl_registry};
use wayland_client::{Connection, Dispatch, QueueHandle};

const POLL_INTERVAL: Duration = Duration::from_millis(100);
const RECONNECT_DELAY: Duration = Duration::from_millis(500);

/// Display orientation reported by the compositor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DisplayOrientation {
    #[default]
    Rotate0,
    Rotate90,
    Rotate180,
    Rotate270,
    // Kept for compatibility with layout code, but the Wayland backend never
    // emits these until the full rendering/capture pipeline supports reflection.
    Flipped0,
    Flipped90,
    Flipped180,
    Flipped270,
}

impl DisplayOrientation {
    pub fn degrees(self) -> u32 {
        match self {
            Self::Rotate0 | Self::Flipped0 => 0,
            Self::Rotate90 | Self::Flipped90 => 90,
            Self::Rotate180 | Self::Flipped180 => 180,
            Self::Rotate270 | Self::Flipped270 => 270,
        }
    }

    fn from_transform(t: wl_output::Transform) -> Option<Self> {
        match t {
            wl_output::Transform::Normal => Some(Self::Rotate0),
            wl_output::Transform::_90 => Some(Self::Rotate90),
            wl_output::Transform::_180 => Some(Self::Rotate180),
            wl_output::Transform::_270 => Some(Self::Rotate270),
            wl_output::Transform::Flipped
            | wl_output::Transform::Flipped90
            | wl_output::Transform::Flipped180
            | wl_output::Transform::Flipped270 => None,
            _ => None,
        }
    }
}

#[derive(Default)]
struct OutputData {
    pending: Option<DisplayOrientation>,
    current: Option<DisplayOrientation>,
}

struct DispatchState {
    tx: UnboundedSender<DisplayOrientation>,
    outputs: HashMap<u32, OutputData>,
    last_emitted: Option<DisplayOrientation>,
    initial_enumeration_complete: bool,
}

impl DispatchState {
    fn new(tx: UnboundedSender<DisplayOrientation>) -> Self {
        Self {
            tx,
            outputs: HashMap::new(),
            last_emitted: None,
            initial_enumeration_complete: false,
        }
    }

    fn emit(&mut self, orientation: DisplayOrientation) {
        if self.last_emitted != Some(orientation) {
            self.last_emitted = Some(orientation);
            let _ = self.tx.send(orientation);
        }
    }

    fn add_output(&mut self, name: u32) {
        self.outputs.insert(name, OutputData::default());
        if self.initial_enumeration_complete && self.outputs.len() == 2 {
            warn!("multiple wl_outputs detected; automatic orientation disabled");
            self.emit(DisplayOrientation::Rotate0);
        }
    }

    fn remove_output(&mut self, name: u32) {
        if self.outputs.remove(&name).is_none() {
            return;
        }
        if self.initial_enumeration_complete && self.outputs.len() == 1 {
            let orientation = self.outputs.values().next().and_then(|o| o.current);
            if let Some(orientation) = orientation {
                info!(
                    ?orientation,
                    "single wl_output restored; automatic orientation enabled"
                );
                self.emit(orientation);
            }
        }
    }

    fn commit_transform(&mut self, name: u32, orientation: DisplayOrientation) {
        if let Some(output) = self.outputs.get_mut(&name) {
            output.current = Some(orientation);
        }
        if self.initial_enumeration_complete
            && self.outputs.len() == 1
            && self.outputs.contains_key(&name)
        {
            info!(?orientation, "display orientation updated");
            self.emit(orientation);
        }
    }

    fn finish_initial_enumeration(&mut self) {
        self.initial_enumeration_complete = true;
        match self.outputs.len() {
            1 => {
                let orientation = self.outputs.values().next().and_then(|o| o.current);
                if let Some(orientation) = orientation {
                    self.emit(orientation);
                }
            }
            2.. => self.emit(DisplayOrientation::Rotate0),
            _ => {}
        }
    }
}

impl Dispatch<wl_registry::WlRegistry, ()> for DispatchState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        match event {
            wl_registry::Event::Global {
                name,
                interface,
                version,
            } if interface == "wl_output" => {
                if version < 2 {
                    debug!(name, version, "wl_output below v2; transform unavailable");
                    return;
                }
                let _output =
                    registry.bind::<wl_output::WlOutput, _, _>(name, version.min(4), qh, name);
                state.add_output(name);
            }
            wl_registry::Event::GlobalRemove { name } => state.remove_output(name),
            _ => {}
        }
    }
}

impl Dispatch<wl_callback::WlCallback, ()> for DispatchState {
    fn event(
        state: &mut Self,
        _: &wl_callback::WlCallback,
        event: wl_callback::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if matches!(event, wl_callback::Event::Done { .. }) {
            state.finish_initial_enumeration();
        }
    }
}

impl Dispatch<wl_output::WlOutput, u32> for DispatchState {
    fn event(
        state: &mut Self,
        _: &wl_output::WlOutput,
        event: wl_output::Event,
        name: &u32,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wl_output::Event::Geometry { transform, .. } => {
                let orientation = transform
                    .into_result()
                    .ok()
                    .and_then(DisplayOrientation::from_transform);
                let fallback = orientation.unwrap_or_else(|| {
                    warn!(
                        name,
                        "unsupported reflected/unknown output transform; using Rotate0"
                    );
                    DisplayOrientation::Rotate0
                });
                if let Some(output) = state.outputs.get_mut(name) {
                    output.pending = Some(fallback);
                }
            }
            wl_output::Event::Done => {
                let orientation = state.outputs.get_mut(name).and_then(|o| o.pending.take());
                if let Some(orientation) = orientation {
                    state.commit_transform(*name, orientation);
                }
            }
            _ => {}
        }
    }
}

/// A cancellable orientation listener. Dropping it signals and joins its sole
/// worker thread, so subscription recreation cannot accumulate listeners.
pub struct Listener {
    rx: UnboundedReceiver<DisplayOrientation>,
    cancelled: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl Listener {
    pub async fn recv(&mut self) -> Option<DisplayOrientation> {
        self.rx.recv().await
    }

    #[cfg(test)]
    fn spawn_for_test(worker: impl FnOnce(Arc<AtomicBool>) + Send + 'static) -> Self {
        let (_tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker_cancelled = cancelled.clone();
        let worker = std::thread::spawn(move || worker(worker_cancelled));
        Self {
            rx,
            cancelled,
            worker: Some(worker),
        }
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Start one reconnecting, cancellable listener worker.
pub fn start() -> Listener {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancelled = cancelled.clone();
    let worker = std::thread::Builder::new()
        .name("display-orientation".into())
        .spawn(move || supervise(tx, &worker_cancelled))
        .map_err(|error| warn!(%error, "failed to spawn display-orientation thread"))
        .ok();
    Listener {
        rx,
        cancelled,
        worker,
    }
}

fn supervise(tx: UnboundedSender<DisplayOrientation>, cancelled: &AtomicBool) {
    while !cancelled.load(Ordering::Acquire) && !tx.is_closed() {
        // Deduplicate within this connection-loss episode. A new connection
        // may emit a non-zero orientation, so the next loss must be allowed to
        // send the safe fallback again.
        let mut last_emitted = None;
        if let Err(error) = run_connection(tx.clone(), cancelled) {
            warn!(%error, "display-orientation connection lost; reconnecting");
            emit_fallback_on_disconnect(&tx, &mut last_emitted);
        }
        let slices = RECONNECT_DELAY.as_millis() / POLL_INTERVAL.as_millis();
        for _ in 0..slices {
            if cancelled.load(Ordering::Acquire) || tx.is_closed() {
                return;
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }
}

fn emit_fallback_on_disconnect(
    tx: &UnboundedSender<DisplayOrientation>,
    last_emitted: &mut Option<DisplayOrientation>,
) {
    if *last_emitted != Some(DisplayOrientation::Rotate0) {
        *last_emitted = Some(DisplayOrientation::Rotate0);
        let _ = tx.send(DisplayOrientation::Rotate0);
    }
}

fn run_connection(
    tx: UnboundedSender<DisplayOrientation>,
    cancelled: &AtomicBool,
) -> Result<(), Box<dyn std::error::Error>> {
    let conn = Connection::connect_to_env()?;
    let display = conn.display();
    let mut queue = conn.new_event_queue();
    let qh = queue.handle();
    let _registry = display.get_registry(&qh, ());
    let _initial_registry_sync = display.sync(&qh, ());
    let mut state = DispatchState::new(tx.clone());
    info!("display-orientation listener connected");

    while !cancelled.load(Ordering::Acquire) && !tx.is_closed() {
        queue.dispatch_pending(&mut state)?;
        queue.flush()?;
        let Some(guard) = queue.prepare_read() else {
            continue;
        };
        let mut pollfd = libc::pollfd {
            fd: guard.connection_fd().as_fd().as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: `pollfd` points to one initialized entry for the duration of the call.
        let ready = unsafe { libc::poll(&mut pollfd, 1, POLL_INTERVAL.as_millis() as i32) };
        if ready < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        if ready > 0 {
            guard.read()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_transform_maps_rotations() {
        assert_eq!(
            DisplayOrientation::from_transform(wl_output::Transform::Normal),
            Some(DisplayOrientation::Rotate0)
        );
        assert_eq!(
            DisplayOrientation::from_transform(wl_output::Transform::_90),
            Some(DisplayOrientation::Rotate90)
        );
        assert_eq!(
            DisplayOrientation::from_transform(wl_output::Transform::_180),
            Some(DisplayOrientation::Rotate180)
        );
        assert_eq!(
            DisplayOrientation::from_transform(wl_output::Transform::_270),
            Some(DisplayOrientation::Rotate270)
        );
    }

    #[test]
    fn flipped_transforms_are_rejected_instead_of_collapsed_to_rotation() {
        assert_eq!(
            DisplayOrientation::from_transform(wl_output::Transform::Flipped90),
            None
        );
    }

    #[test]
    fn multiple_outputs_disable_automatic_orientation() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut state = DispatchState::new(tx);
        state.add_output(7);
        state.finish_initial_enumeration();
        state.commit_transform(7, DisplayOrientation::Rotate90);
        assert_eq!(rx.try_recv(), Ok(DisplayOrientation::Rotate90));
        state.add_output(9);
        assert_eq!(rx.try_recv(), Ok(DisplayOrientation::Rotate0));
        state.commit_transform(9, DisplayOrientation::Rotate270);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn returning_to_one_output_reenables_its_known_orientation() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut state = DispatchState::new(tx);
        state.add_output(7);
        state.add_output(9);
        state.finish_initial_enumeration();
        assert_eq!(rx.try_recv(), Ok(DisplayOrientation::Rotate0));
        state.commit_transform(9, DisplayOrientation::Rotate270);
        state.remove_output(7);
        assert_eq!(rx.try_recv(), Ok(DisplayOrientation::Rotate270));
    }

    #[test]
    fn initial_enumeration_never_emits_an_arbitrary_output() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut state = DispatchState::new(tx);
        state.add_output(7);
        state.commit_transform(7, DisplayOrientation::Rotate90);
        state.add_output(9);
        state.commit_transform(9, DisplayOrientation::Rotate270);
        assert!(rx.try_recv().is_err());

        state.finish_initial_enumeration();
        assert_eq!(rx.try_recv(), Ok(DisplayOrientation::Rotate0));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn connection_loss_falls_back_to_rotate0_once() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut last_emitted = Some(DisplayOrientation::Rotate90);
        emit_fallback_on_disconnect(&tx, &mut last_emitted);
        assert_eq!(rx.try_recv(), Ok(DisplayOrientation::Rotate0));

        emit_fallback_on_disconnect(&tx, &mut last_emitted);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn listener_drop_cancels_and_joins_worker() {
        let stopped = Arc::new(AtomicBool::new(false));
        let worker_stopped = stopped.clone();
        let listener = Listener::spawn_for_test(move |cancelled| {
            while !cancelled.load(Ordering::Acquire) {
                std::thread::sleep(Duration::from_millis(1));
            }
            worker_stopped.store(true, Ordering::Release);
        });
        drop(listener);
        assert!(stopped.load(Ordering::Acquire));
    }
}
