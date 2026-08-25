// SPDX-License-Identifier: GPL-3.0-only

//! Shared GPU test harness.
//!
//! Exists so that every GPU test in the crate goes through ONE `wgpu::Instance`.
//! Building an instance per test is what made `cargo test --lib` SIGSEGV inside
//! lavapipe at default parallelism, so the device lives here rather than in any
//! one test module — and any new GPU test should reach for `headless_device()`
//! instead of standing up its own.

use iced_wgpu::wgpu;
use std::sync::{LazyLock, Mutex, MutexGuard, PoisonError};

/// The ONE headless wgpu device the GPU tests share, or `None` when the
/// machine has no usable adapter.
///
/// Shared, and it has to be. Each `wgpu::Instance` loads and initialises the
/// Vulkan ICD; building ten of them concurrently — which is exactly what
/// `cargo test`'s default thread pool did, one per GPU test — SIGSEGV'd
/// inside lavapipe about half the time. The device is only ever read from
/// here, and wgpu's `Device`/`Queue` are internally synchronised Arc handles,
/// so one instance for the whole binary is both correct and enough.
static HEADLESS_DEVICE: LazyLock<Option<(wgpu::Device, wgpu::Queue)>> = LazyLock::new(|| {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("frosted corner test device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        memory_hints: wgpu::MemoryHints::default(),
        experimental_features: wgpu::ExperimentalFeatures::default(),
        trace: wgpu::Trace::Off,
    }))
    .ok()
});

/// Serializes use of [`HEADLESS_DEVICE`].
///
/// wgpu's GLES backend protects one adapter context with a non-reentrant lock.
/// Concurrent commands from cloned `Device` handles can hold that context long
/// enough for another test to hit wgpu-hal's deadlock timeout. Vulkan permits
/// this concurrency, but Alpine's riscv64 runner only exposes GLES.
static HEADLESS_DEVICE_LOCK: Mutex<()> = Mutex::new(());

/// Exclusive access to the shared headless device, or `None` when the machine
/// has no usable adapter (CI) — in which case the caller skips via
/// [`skip_no_gpu`] rather than fails.
///
/// Keep the returned guard alive until every GPU resource created by the test
/// has been dropped. The tuple shape makes the guard explicit at each call site.
pub(crate) fn headless_device() -> Option<(MutexGuard<'static, ()>, wgpu::Device, wgpu::Queue)> {
    let guard = HEADLESS_DEVICE_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    HEADLESS_DEVICE
        .clone()
        .map(|(device, queue)| (guard, device, queue))
}

/// Called wherever a GPU test degrades to a skip because there is no adapter.
///
/// A `println!` and a `return` is a PASS, so without this every GPU test in the
/// crate reports green on an adapter-less machine while asserting nothing — and
/// these tests are the only thing between this branch and the black-bars
/// regression they were written for. Set `CI_REQUIRE_GPU=1` to turn the skip
/// into a hard failure, so CI can demand real coverage while local dev on a
/// headless box still skips gracefully.
pub(crate) fn skip_no_gpu(what: &str) {
    assert!(
        std::env::var("CI_REQUIRE_GPU").as_deref() != Ok("1"),
        "{what} needs a GPU adapter and CI_REQUIRE_GPU=1 forbids skipping, but no \
         usable wgpu adapter was found"
    );
    println!("Skipping {what} (no GPU adapter)");
}

#[cfg(test)]
mod tests {
    use super::headless_device;
    use std::sync::{Arc, Barrier, mpsc};
    use std::time::Duration;

    #[test]
    fn headless_device_allows_only_one_gpu_test_at_a_time() {
        let Some(first) = headless_device() else {
            return;
        };
        let started = Arc::new(Barrier::new(2));
        let child_started = Arc::clone(&started);
        let (acquired_tx, acquired_rx) = mpsc::channel();

        let child = std::thread::spawn(move || {
            child_started.wait();
            let second = headless_device();
            acquired_tx.send(second.is_some()).unwrap();
        });

        started.wait();
        assert!(
            acquired_rx
                .recv_timeout(Duration::from_millis(100))
                .is_err(),
            "a second GPU test acquired the shared GLES context concurrently"
        );

        drop(first);
        assert_eq!(
            acquired_rx.recv_timeout(Duration::from_secs(2)),
            Ok(true),
            "the next GPU test did not acquire the context after it was released"
        );
        child.join().unwrap();
    }
}
