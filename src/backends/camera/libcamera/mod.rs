// SPDX-License-Identifier: GPL-3.0-only

//! libcamera backend using native libcamera-rs bindings
//!
//! This backend uses direct libcamera-rs bindings for camera access,
//! providing multi-stream capture capabilities (preview + raw still capture)
//! with per-frame metadata (exposure, gain, colour temperature).
//!
//! Discovery and format queries use [`LibcameraBackend`]. GUI subscriptions and
//! CLI/terminal handles own capture pipelines independently; the discovery
//! backend retains no active camera or preview frame.
//!
//! # Multi-Stream Capture
//!
//! libcamera supports multiple streams per camera:
//! - ViewFinder stream: ISP-processed preview (e.g., ABGR8888)
//! - Raw stream: Bypasses ISP for full-resolution Bayer capture
//!
//! This allows simultaneous 1080p preview and full-resolution photo capture.

mod discovery_cache;
pub(crate) mod native;
mod stream_negotiation;

pub(crate) use native::{NativeLibcameraPipeline, PipelineSharedState};

use super::CameraBackend;
use super::pipeline::CapturePipeline;
use super::types::*;
use super::v4l2_utils;
use libcamera::camera_manager::CameraManager;
use libcamera::stream::StreamRole;
use native::pixel_formats::{is_bayer_format, pixel_format_name};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use tracing::{debug, info, warn};

/// Global lock for CameraManager creation.
///
/// libcamera only allows one CameraManager to exist at a time.
/// This mutex ensures that enumeration, format queries, and pipeline creation
/// don't overlap and trigger the "Multiple CameraManager objects" abort.
static CAMERA_MANAGER_LOCK: Mutex<()> = Mutex::new(());

/// Detect the libcamera pipeline handler from a camera ID.
///
/// The "simple" pipeline handler (Software ISP, used by phone platforms) uses
/// device-tree paths that start with "/base/" (older libcamera) or "platform/"
/// (libcamera 0.7+). UVC IDs can also start with a device-tree path, but end
/// with a USB vendor/product suffix; leave those and other unknown IDs
/// unidentified. This heuristic is only for hardware-release behavior and
/// diagnostics; stream support is probed independently through libcamera.
///
/// This is the single source of truth for the check: both enumeration and the
/// capture thread's hardware-release delay depend on it.
pub(crate) fn detect_pipeline_handler(camera_id: &str) -> Option<&'static str> {
    if let Some((_, usb_id)) = camera_id.rsplit_once('-')
        && let Some((vendor, product)) = usb_id.split_once(':')
        && vendor.len() == 4
        && product.len() == 4
        && vendor
            .bytes()
            .chain(product.bytes())
            .all(|b| b.is_ascii_hexdigit())
    {
        return None;
    }
    if camera_id.starts_with("/base/") || camera_id.starts_with("platform/") {
        Some("simple")
    } else {
        None
    }
}

/// Native discovery snapshot, reused during capture and unchanged hotplug polls.
static CACHED_DEVICES: std::sync::LazyLock<
    std::sync::RwLock<discovery_cache::NativeDiscoveryCache>,
> = std::sync::LazyLock::new(|| std::sync::RwLock::new(Default::default()));

/// Publish the accepted configuration back to discovery/capture consumers.
fn record_stream_capability(camera_id: &str, multistream: bool) {
    if let Ok(mut devices) = CACHED_DEVICES.write()
        && let Some(device) = devices.devices.iter_mut().find(|d| d.path == camera_id)
    {
        device.supports_multistream = multistream;
    }
    if !multistream
        && let Ok(mut cache) = CACHED_FORMATS.write()
        && let Some(formats) = cache.get_mut(camera_id)
    {
        formats.retain(|format| !format.pixel_format.starts_with("Bayer"));
    }
}

pub(crate) fn cached_stream_capability(camera_id: &str) -> Option<bool> {
    CACHED_DEVICES
        .read()
        .ok()?
        .devices
        .iter()
        .find(|d| d.path == camera_id)
        .map(|d| d.supports_multistream)
}

/// Cached format results per device path, returned when capture is active.
/// Key: device path, Value: formats
static CACHED_FORMATS: std::sync::LazyLock<
    std::sync::RwLock<std::collections::HashMap<String, Vec<CameraFormat>>>,
> = std::sync::LazyLock::new(|| std::sync::RwLock::new(std::collections::HashMap::new()));

/// Discovery and format queries for native and PipeWire fallback cameras.
pub struct LibcameraBackend;

impl Default for LibcameraBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl LibcameraBackend {
    pub fn new() -> Self {
        Self
    }
}

/// Build a sysfs device-tree path for a camera property.
fn dt_sysfs_path(camera_path: &str, property: &str) -> String {
    if camera_path.starts_with('/') {
        format!("/sys/firmware/devicetree{}/{}", camera_path, property)
    } else {
        format!("/sys/firmware/devicetree/base/{}/{}", camera_path, property)
    }
}

/// Device tree properties read for a camera.
struct DeviceTreeProperties {
    sensor_model: Option<String>,
    location: Option<String>,
    rotation: SensorRotation,
}

/// Read sensor model, location, and rotation from the device tree in one pass.
///
/// - `compatible`: null-terminated strings; first entry is the sensor model.
/// - `orientation`: big-endian u32 — 0=front, 1=back, 2=external.
/// - `rotation`: big-endian u32 — sensor mounting angle in degrees.
fn read_device_tree_properties(camera_path: &str) -> DeviceTreeProperties {
    // Sensor model from `compatible`
    let sensor_model = {
        let dt_path = dt_sysfs_path(camera_path, "compatible");
        match std::fs::read(&dt_path) {
            Ok(bytes) if !bytes.is_empty() => {
                let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                let model = String::from_utf8_lossy(&bytes[..end]).trim().to_string();
                if model.is_empty() {
                    None
                } else {
                    debug!(dt_path = %dt_path, model = %model, "Read sensor model from device tree");
                    Some(model)
                }
            }
            Ok(_) => None,
            Err(e) => {
                debug!(dt_path = %dt_path, error = %e, "Could not read sensor model from device tree");
                None
            }
        }
    };

    // Location from `orientation`
    let location = {
        let dt_path = dt_sysfs_path(camera_path, "orientation");
        match std::fs::read(&dt_path) {
            Ok(bytes) if bytes.len() >= 4 => {
                let value = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                let loc = match value {
                    0 => "front",
                    1 => "back",
                    _ => "external",
                };
                debug!(dt_path = %dt_path, value, location = loc, "Read orientation from device tree");
                Some(loc.to_string())
            }
            _ => None,
        }
    };

    // Rotation from `rotation`
    let rotation = {
        let dt_path = dt_sysfs_path(camera_path, "rotation");
        match std::fs::read(&dt_path) {
            Ok(bytes) if bytes.len() >= 4 => {
                let degrees = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                debug!(dt_path = %dt_path, degrees, "Read rotation from device tree");
                SensorRotation::from_degrees_int(degrees as i32)
            }
            _ => SensorRotation::default(),
        }
    };

    DeviceTreeProperties {
        sensor_model,
        location,
        rotation,
    }
}

/// Generate a pretty display name from location (and optionally sensor model)
fn generate_pretty_name(sensor_model: Option<&str>, location: Option<&str>) -> Option<String> {
    match location {
        Some("front") => Some("Front Camera".to_string()),
        Some("back") => Some("Back Camera".to_string()),
        Some("external") => sensor_model.map(|m| m.to_string()),
        _ if sensor_model.is_some() => Some("Camera".to_string()),
        _ => None,
    }
}

/// Prefix non-Bayer-prefixed raw format names with "Bayer".
fn bayer_display_name(format_name: String, is_bayer: bool) -> String {
    if is_bayer && !format_name.starts_with("Bayer") {
        format!("Bayer{}", format_name.to_uppercase())
    } else {
        format_name
    }
}

/// Collect formats from a single stream configuration.
///
/// When `synthesize_intermediate` is true, adds common resolution tiers (1080p, 720p)
/// by scaling the native sensor size. This is only appropriate for ISP-processed
/// (non-Bayer) streams where the hardware can produce any output size.
fn collect_stream_formats(
    config: &libcamera::camera::CameraConfiguration,
    index: usize,
    framerate: Option<Framerate>,
    synthesize_intermediate: bool,
) -> Vec<CameraFormat> {
    let Some(cfg) = config.get(index) else {
        return Vec::new();
    };
    let stream_formats = cfg.formats();
    let pixel_formats = stream_formats.pixel_formats();
    let mut formats = Vec::new();

    for pf in &*pixel_formats {
        let format_name = pixel_format_name(pf);
        let is_bayer = is_bayer_format(pf);
        let display_name = bayer_display_name(format_name, is_bayer);

        let mut max_w = 0u32;
        let mut max_h = 0u32;

        for size in stream_formats.sizes(pf) {
            if size.width * size.height > max_w * max_h {
                max_w = size.width;
                max_h = size.height;
            }
            formats.push(CameraFormat {
                width: size.width,
                height: size.height,
                framerate,
                hardware_accelerated: true,
                pixel_format: display_name.clone(),
            });
        }

        // Synthesize common intermediate resolutions for ISP-processed formats.
        // libcamera's StreamFormats only lists discrete native sizes (e.g., 4656x3496
        // and 640x480), but the ISP can produce any size via downscaling. We add
        // standard resolution tiers by scaling the native size while preserving the
        // sensor's aspect ratio (rounded to even for codec compatibility).
        if synthesize_intermediate && !is_bayer && max_w > 0 && max_h > 0 {
            for &target_h in &[1080u32, 720] {
                if max_h > target_h {
                    let scale = target_h as f64 / max_h as f64;
                    let w = ((max_w as f64 * scale) as u32) & !1;
                    if w > 0 {
                        formats.push(CameraFormat {
                            width: w,
                            height: target_h,
                            framerate,
                            hardware_accelerated: true,
                            pixel_format: display_name.clone(),
                        });
                    }
                }
            }
        }
    }

    formats
}

/// Query formats for a camera using an existing CameraManager reference.
///
/// This avoids creating a new CameraManager and is used both during enumeration
/// (to pre-populate the cache for all cameras) and during direct format queries.
fn query_camera_formats(
    cam: &libcamera::camera::Camera,
    supports_multistream: bool,
    video_mode: bool,
) -> Vec<CameraFormat> {
    let config = match cam.generate_configuration(&[StreamRole::ViewFinder]) {
        Some(config) => config,
        None => return Vec::new(),
    };

    let framerate = if video_mode {
        Some(Framerate::from_int(30))
    } else {
        None
    };

    let mut formats = collect_stream_formats(&config, 0, framerate, true);

    // Also probe raw stream formats if multistream is supported
    if supports_multistream && let Ok(raw_config) = stream_negotiation::prepare(cam, true, None) {
        formats.extend(collect_stream_formats(&raw_config, 1, framerate, false));
    }

    // Sort by resolution (highest first), then by pixel format (Bayer formats first)
    formats.sort_by(|a, b| {
        (b.width * b.height)
            .cmp(&(a.width * a.height))
            .then_with(|| {
                b.pixel_format
                    .starts_with("Bayer")
                    .cmp(&a.pixel_format.starts_with("Bayer"))
            })
    });
    formats.dedup_by(|a, b| {
        a.width == b.width && a.height == b.height && a.pixel_format == b.pixel_format
    });

    formats
}

impl LibcameraBackend {
    /// Refresh PipeWire on every poll, but only reprobe native devices after
    /// the kernel video-node topology changes. Explicit enumeration stays fresh.
    pub(crate) fn enumerate_cameras_for_hotplug(&self) -> Vec<CameraDevice> {
        self.discover_cameras(false)
    }

    fn discover_cameras(&self, force_native: bool) -> Vec<CameraDevice> {
        debug!("Enumerating cameras via libcamera-rs");

        let _lock = CAMERA_MANAGER_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        // Check under the lock to avoid TOCTOU race with capture thread
        if native::is_capture_active() {
            debug!("Capture active, returning cached camera list");
            return super::pipewire::append_fallbacks(
                CACHED_DEVICES
                    .read()
                    .map(|g| g.devices.clone())
                    .unwrap_or_default(),
            );
        }

        let video_nodes = discovery_cache::scan_video_topology().ok();
        if !force_native {
            let cached = CACHED_DEVICES.read().ok().and_then(|cache| {
                video_nodes
                    .as_ref()
                    .filter(|nodes| cache.matches(nodes))
                    .map(|_| cache.devices.clone())
            });
            // Release the cache guard before external PipeWire discovery so
            // capture can publish updated stream capabilities independently.
            if let Some(devices) = cached {
                return super::pipewire::append_fallbacks(devices);
            }
        }

        let mgr = match CameraManager::new() {
            Ok(mgr) => mgr,
            Err(e) => {
                warn!(error = %e, "Failed to create CameraManager");
                return super::pipewire::append_fallbacks(Vec::new());
            }
        };

        let libcamera_version = Some(mgr.version().to_string());
        let cameras = mgr.cameras();
        let mut devices = Vec::new();

        for cam in cameras.iter() {
            let camera_id = cam.id().to_string();
            let props = cam.properties();

            // Read device tree properties as fallback for libcamera properties
            let dt = read_device_tree_properties(&camera_id);

            // Get model from libcamera properties, fall back to device tree
            let sensor_model = props
                .get::<libcamera::properties::Model>()
                .ok()
                .map(|m| m.0.clone())
                .or(dt.sensor_model);

            // Get location from libcamera properties, fall back to device tree
            let camera_location = props
                .get::<libcamera::properties::Location>()
                .ok()
                .map(|l| match l {
                    libcamera::properties::Location::CameraFront => "front".to_string(),
                    libcamera::properties::Location::CameraBack => "back".to_string(),
                    libcamera::properties::Location::CameraExternal => "external".to_string(),
                })
                .or(dt.location);

            // Get rotation from libcamera properties, fall back to device tree
            let rotation = props
                .get::<libcamera::properties::Rotation>()
                .ok()
                .map(|r| SensorRotation::from_degrees_int(r.0))
                .unwrap_or(dt.rotation);

            // Generate pretty name
            let pretty_name =
                generate_pretty_name(sensor_model.as_deref(), camera_location.as_deref())
                    .unwrap_or_else(|| sensor_model.as_deref().unwrap_or(&camera_id).to_string());

            let pipeline_handler = detect_pipeline_handler(&camera_id).map(str::to_string);

            let supports_multistream = stream_negotiation::prepare(&cam, true, None).is_ok();

            debug!(
                camera_id = %camera_id,
                pretty_name = %pretty_name,
                sensor_model = ?sensor_model,
                camera_location = ?camera_location,
                rotation = %rotation,
                pipeline_handler = ?pipeline_handler,
                supports_multistream,
                "Found libcamera device"
            );

            // Find the underlying V4L2 device for UVC cameras so that
            // exposure/motor/color controls work via V4L2 ioctls.
            let device_info =
                v4l2_utils::find_v4l2_device_for_libcamera(&camera_id).map(|v4l2_path| {
                    v4l2_utils::build_device_info(&v4l2_path, sensor_model.as_deref())
                });

            // USB cameras may expose a generic UVC or positional label.
            // Enrich the display name from the USB parent's udev
            // database, keeping the native ID and original sensor metadata.
            let pretty_name = device_info.as_ref().map_or(pretty_name.clone(), |info| {
                super::usb_name::display_name(&pretty_name, &info.path)
            });

            devices.push(CameraDevice {
                name: pretty_name,
                path: camera_id,
                device_info,
                rotation,
                pipeline_handler,
                supports_multistream,
                sensor_model,
                camera_location,
                libcamera_version: libcamera_version.clone(),
                lens_actuator_path: None,
            });
        }

        // Discover lens actuators and associate with back camera
        let actuators = v4l2_utils::discover_lens_actuators();
        if let Some(back_cam) = devices
            .iter_mut()
            .find(|d| d.camera_location.as_deref() == Some("back"))
            && let Some((actuator_path, actuator_name)) = actuators.first()
        {
            info!(
                actuator = %actuator_path,
                actuator_name = %actuator_name,
                camera = %back_cam.name,
                "Associated lens actuator with back camera"
            );
            back_cam.lens_actuator_path = Some(actuator_path.clone());
        }

        // Sort by path for consistent ordering
        devices.sort_by(|a, b| a.path.cmp(&b.path));

        // Record only successful native discovery. In particular, a failed
        // CameraManager creation must leave the topology invalidated for retry.
        if let Some(video_nodes) = video_nodes
            && let Ok(mut cache) = CACHED_DEVICES.write()
        {
            cache.update(video_nodes, devices.clone());
        }

        // Pre-populate format cache for all cameras while we have the CameraManager open.
        // This ensures get_formats() can return cached results when capture is active
        // (e.g., during camera switching when the old pipeline hasn't stopped yet).
        if let Ok(mut cache) = CACHED_FORMATS.write() {
            for dev in &devices {
                if let Some(cam) = mgr.get(&dev.path) {
                    let formats = query_camera_formats(&cam, dev.supports_multistream, false);
                    debug!(
                        camera = %dev.path,
                        format_count = formats.len(),
                        "Pre-cached formats"
                    );
                    cache.insert(dev.path.clone(), formats);
                }
            }
        }

        debug!(count = devices.len(), "Enumerated libcamera cameras");
        super::pipewire::append_fallbacks(devices)
    }
}

impl CameraBackend for LibcameraBackend {
    fn enumerate_cameras(&self) -> Vec<CameraDevice> {
        self.discover_cameras(true)
    }

    fn get_formats(&self, device: &CameraDevice, video_mode: bool) -> Vec<CameraFormat> {
        if device.path.starts_with("pipewire:") {
            return super::pipewire::formats(device);
        }
        debug!(device_path = %device.path, video_mode, "Getting formats via libcamera-rs");

        // Return cached formats if available (populated by enumerate_cameras).
        // This avoids creating a redundant CameraManager (~130ms on ARM).
        if !video_mode
            && let Some(cached) = CACHED_FORMATS
                .read()
                .ok()
                .and_then(|cache| cache.get(&device.path).cloned())
            && !cached.is_empty()
        {
            debug!(count = cached.len(), "Returning cached formats");
            return cached;
        }

        // Try to acquire the lock without blocking. If the capture thread holds it
        // (e.g. during pipeline teardown), return cached formats so the UI stays responsive.
        let _lock = match CAMERA_MANAGER_LOCK.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::WouldBlock) => {
                let cached = CACHED_FORMATS
                    .read()
                    .ok()
                    .and_then(|cache| cache.get(&device.path).cloned())
                    .unwrap_or_default();
                debug!(
                    count = cached.len(),
                    "Lock contended (pipeline restarting), returning cached formats"
                );
                return cached;
            }
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
        };

        // Check under the lock to avoid TOCTOU race with capture thread
        if native::is_capture_active() {
            let cached = CACHED_FORMATS
                .read()
                .ok()
                .and_then(|cache| cache.get(&device.path).cloned())
                .unwrap_or_default();
            debug!(
                count = cached.len(),
                "Capture active, returning cached formats"
            );
            return cached;
        }

        let mgr = match CameraManager::new() {
            Ok(mgr) => mgr,
            Err(e) => {
                warn!(error = %e, "Failed to create CameraManager for format query");
                return Vec::new();
            }
        };

        let cam = match mgr.get(&device.path) {
            Some(cam) => cam,
            None => {
                warn!(path = %device.path, "Camera not found for format query");
                return Vec::new();
            }
        };

        let supports_multistream = stream_negotiation::prepare(&cam, true, None).is_ok();
        let formats = query_camera_formats(&cam, supports_multistream, video_mode);

        debug!(count = formats.len(), "Enumerated formats via libcamera-rs");

        // Cache formats for this device (used when capture is active)
        if let Ok(mut cache) = CACHED_FORMATS.write() {
            cache.insert(device.path.clone(), formats.clone());
        }

        formats
    }
}

// =========================================================================
// Public convenience API for CLI/terminal use
// =========================================================================

/// Opaque handle that keeps a camera pipeline alive. Drop to stop.
pub struct CameraPipelineHandle {
    _pipeline: CapturePipeline,
    recording_sender: Arc<Mutex<Option<tokio::sync::mpsc::Sender<RecordingFrame>>>>,
}

impl CameraPipelineHandle {
    /// Set or clear the recording frame sender for video capture.
    pub fn set_recording_sender(&self, sender: Option<tokio::sync::mpsc::Sender<RecordingFrame>>) {
        if let Ok(mut guard) = self.recording_sender.lock() {
            *guard = sender;
        }
    }
}

/// Create a camera preview pipeline.
///
/// Returns an opaque handle (drop to stop) and a frame receiver.
/// Used by CLI and terminal tools that don't use the full app framework.
pub fn create_pipeline(
    device: &CameraDevice,
    format: &CameraFormat,
) -> Result<(CameraPipelineHandle, FrameReceiver), String> {
    let (sender, receiver) = futures::channel::mpsc::channel(10);
    let recording_sender: Arc<Mutex<Option<tokio::sync::mpsc::Sender<RecordingFrame>>>> =
        Arc::new(Mutex::new(None));
    let pipeline = CapturePipeline::new(
        &device.path,
        format,
        false,
        PipelineSharedState {
            zoom: Default::default(),
            frame_sender: sender,
            still_capture: Default::default(),
            recording_sender: Arc::clone(&recording_sender),
            cancel_flag: Arc::new(AtomicBool::new(false)),
        },
    )
    .map_err(|e| format!("{}", e))?;
    Ok((
        CameraPipelineHandle {
            _pipeline: pipeline,
            recording_sender,
        },
        receiver,
    ))
}

#[cfg(test)]
mod tests {
    use super::detect_pipeline_handler;

    #[test]
    #[ignore = "requires an idle PipeWire fallback camera"]
    fn live_pipewire_hotplug_polls_reuse_native_discovery() {
        use super::super::CameraBackend;
        use futures::StreamExt;

        let backend = super::LibcameraBackend::new();
        let devices = backend.enumerate_cameras();
        let device = devices
            .iter()
            .find(|d| d.path.starts_with("pipewire:"))
            .expect("PipeWire fallback camera");
        let formats = backend.get_formats(device, true);
        let format = formats
            .iter()
            .find(|f| f.width == 640 && f.height == 480)
            .unwrap_or(&formats[0]);
        let (_handle, mut receiver) =
            super::create_pipeline(device, format).expect("fallback pipeline");
        assert!(!super::native::is_capture_active());

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        // Settle the bounded native readiness retries before measuring the
        // steady-state polls. PipeWire continues streaming throughout.
        for _ in 0..2 {
            backend.enumerate_cameras_for_hotplug();
        }
        println!("HOTPLUG_CACHE_BEGIN");
        for _ in 0..3 {
            let frame = runtime
                .block_on(async {
                    tokio::time::timeout(std::time::Duration::from_secs(8), receiver.next()).await
                })
                .expect("preview frame within eight seconds")
                .expect("live preview");
            assert!(!frame.data.is_empty());
            let cameras = super::LibcameraBackend::new().enumerate_cameras_for_hotplug();
            assert!(cameras.iter().any(|d| d.path == device.path));
        }
        println!("HOTPLUG_CACHE_END");
        // An explicit refresh must still create a fresh native manager even
        // while the PipeWire fallback is streaming.
        let cameras = backend.enumerate_cameras();
        assert!(cameras.iter().any(|d| d.path == device.path));
        println!("EXPLICIT_REFRESH_END");
    }

    #[test]
    #[ignore = "requires an idle Logitech C270 webcam"]
    fn live_uvc_negotiates_single_stream_despite_stale_dual_hint() {
        use super::super::CameraBackend;
        use futures::StreamExt;
        let backend = super::LibcameraBackend::new();
        let mut device = backend
            .enumerate_cameras()
            .into_iter()
            .find(|d| d.path.ends_with("-046d:0825"))
            .expect("Logitech C270");
        assert!(
            !device.supports_multistream,
            "discovery must probe actual stream support"
        );
        let formats = backend.get_formats(&device, false);
        let format = formats.first().expect("webcam format");
        device.supports_multistream = true;
        let (sender, mut receiver) = futures::channel::mpsc::channel(1);
        let pipeline = super::CapturePipeline::new(
            &device.path,
            format,
            device.supports_multistream,
            super::PipelineSharedState {
                zoom: Default::default(),
                frame_sender: sender,
                still_capture: Default::default(),
                recording_sender: Default::default(),
                cancel_flag: Default::default(),
            },
        )
        .expect("single-stream fallback");
        assert_eq!(super::cached_stream_capability(&device.path), Some(false));
        let frame = futures::executor::block_on(receiver.next()).expect("live webcam frame");
        assert_eq!((frame.width, frame.height), (format.width, format.height));
        assert!(!frame.data.is_empty());
        println!(
            "Negotiated single-stream app frame: {}x{} {:?}",
            frame.width, frame.height, frame.format
        );
        drop(pipeline);
    }

    #[test]
    #[ignore = "requires a live PipeWire V4L2 camera"]
    fn pipewire_fallback_captures_live_frame() {
        use super::super::CameraBackend;
        let backend = super::LibcameraBackend::new();
        let devices = backend.enumerate_cameras();
        let device = devices
            .iter()
            .find(|d| d.path.starts_with("pipewire:"))
            .expect("PipeWire camera");
        let formats = backend.get_formats(device, true);
        let format = formats
            .iter()
            .find(|f| f.width == 640 && f.height == 480)
            .unwrap_or(&formats[0]);
        let (_handle, mut receiver) =
            super::create_pipeline(device, format).expect("fallback pipeline");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(8);
        loop {
            if let Ok(frame) = receiver.try_recv() {
                assert_eq!((frame.width, frame.height), (format.width, format.height));
                assert!(!frame.data.is_empty());
                assert!(
                    frame.sensor_timestamp_ns.is_some(),
                    "PipeWire capture timestamp must reach recorder"
                );
                println!(
                    "PipeWire app frame: {}x{} {:?}, stride={}, bytes={}",
                    frame.width,
                    frame.height,
                    frame.format,
                    frame.stride,
                    frame.data.len()
                );
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "no app frame within eight seconds"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    #[test]
    #[ignore = "requires a PipeWire V4L2 camera and no native libcamera support"]
    fn pipewire_fallback_discovers_live_camera() {
        use super::super::CameraBackend;
        let devices = super::LibcameraBackend::new().enumerate_cameras();

        assert!(
            devices.iter().any(|d| d.path.starts_with("pipewire:")),
            "unsupported native camera must be discovered through PipeWire: {devices:?}"
        );
    }

    /// The hardware-release delay before a new CameraManager is created is
    /// gated on this returning Some("simple"). A false negative would skip a
    /// delay that real hardware needs, so pin both phone path formats.
    #[test]
    fn simple_handler_detected_for_device_tree_paths() {
        // libcamera 0.7+ (postmarketOS / qcom platforms)
        assert_eq!(
            detect_pipeline_handler("platform/soc@0/ac5a000.camss"),
            Some("simple")
        );
        // Older libcamera
        assert_eq!(
            detect_pipeline_handler("/base/soc/ac5a000.camss"),
            Some("simple")
        );
    }

    #[test]
    fn device_tree_usb_webcams_are_not_simple_pipeline() {
        for id in [
            "/base/soc/usb@702280000-1.2.3:1.0-046d:0825",
            "/base/soc/usb@b02280000-1.2.3:1.0-046d:0825",
            "platform/soc/usb@702280000-1.2.3:1.0-046d:0825",
        ] {
            let handler = detect_pipeline_handler(id);
            assert_eq!(
                handler, None,
                "USB camera must not use the simple handler: {id}"
            );
        }
    }

    #[test]
    fn usb_webcams_have_no_known_handler() {
        // UVC IDs also use ACPI/USB topology paths on non-device-tree hosts.
        assert_eq!(
            detect_pipeline_handler("\\_SB_.PCI0.GPP7.UP00.DP60.XH00.RHUB.POT9-4:1.0-046d:0825"),
            None
        );
        assert_eq!(detect_pipeline_handler(""), None);
    }
}
