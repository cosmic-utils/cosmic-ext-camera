// SPDX-License-Identifier: GPL-3.0-only

//! Capture thread implementation — owns all libcamera objects.
//!
//! All libcamera types (CameraManager, ActiveCamera, FrameBuffers) are created and
//! dropped on this single thread to avoid Send/Sync issues with libcamera's raw
//! pointers.

use super::diagnostics::{
    CAPTURE_ACTIVE, CAPTURE_RELEASED, DIAGNOSTICS, DiagnosticParams, MJPEG_DECODE_TIME_US,
    PREVIEW_FRAME_COUNT, STILL_FRAME_COUNT, StreamDiag, publish_diagnostics,
};
use super::pixel_formats::{map_pixel_format, pixel_format_name};
use crate::backends::camera::types::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{debug, error, info, warn};

/// Parameters passed to the capture thread for initialization
pub(crate) struct CaptureThreadParams {
    pub(crate) camera_id: String,
    pub(crate) preview_width: u32,
    pub(crate) preview_height: u32,
    pub(crate) supports_multistream: bool,
    pub(crate) stop_flag: Arc<AtomicBool>,
    pub(crate) latest_preview: Arc<Mutex<Option<CameraFrame>>>,
    pub(crate) latest_still: Arc<Mutex<Option<CameraFrame>>>,
    pub(crate) still_requested: Arc<AtomicBool>,
    /// Notified when a new still frame has been stored. Lets consumers await
    /// the next still instead of polling the mutex.
    pub(crate) still_frame_notify: Arc<tokio::sync::Notify>,
    pub(crate) preview_frame_count: Arc<AtomicU64>,
    pub(crate) still_frame_count: Arc<AtomicU64>,
    pub(crate) frame_sender: FrameSender,
    pub(crate) recording_sender: Arc<Mutex<Option<tokio::sync::mpsc::Sender<RecordingFrame>>>>,
    pub(crate) jpeg_recording_mode: Arc<AtomicBool>,
    /// Cancel flag — checked before creating CameraManager to abort if a newer
    /// mode switch has superseded this one.
    pub(crate) cancel_flag: Arc<AtomicBool>,
}

/// Result of capture thread initialization (sent back to main thread)
pub(crate) struct CaptureThreadInitResult {
    pub(crate) is_multistream: bool,
}

/// Extract the number of bytes actually written to each plane of a frame buffer.
/// Returns a Vec with one entry per plane. Falls back to `fallback_lens` if
/// metadata is unavailable.
fn planes_bytes_used(
    buf: &impl libcamera::framebuffer::AsFrameBuffer,
    fallback_lens: &[usize],
) -> Vec<usize> {
    if let Some(meta) = buf.metadata() {
        let meta_planes: Vec<usize> = meta
            .planes()
            .into_iter()
            .map(|p| p.bytes_used as usize)
            .collect();
        if !meta_planes.is_empty() {
            return meta_planes;
        }
    }
    fallback_lens.to_vec()
}

/// Memory-map a vec of frame buffers, returning an error with the given label on failure.
fn mmap_buffers(
    buffers: Vec<libcamera::framebuffer_allocator::FrameBuffer>,
    label: &str,
) -> Result<
    Vec<
        libcamera::framebuffer_map::MemoryMappedFrameBuffer<
            libcamera::framebuffer_allocator::FrameBuffer,
        >,
    >,
    BackendError,
> {
    buffers
        .into_iter()
        .map(|fb| {
            libcamera::framebuffer_map::MemoryMappedFrameBuffer::new(fb)
                .map_err(|e| BackendError::InitializationFailed(format!("mmap {}: {:?}", label, e)))
        })
        .collect()
}

/// Allocate buffers for an optional stream. Returns None when stream is None.
fn alloc_optional_buffers(
    alloc: &mut libcamera::framebuffer_allocator::FrameBufferAllocator,
    stream: &Option<libcamera::stream::Stream>,
    label: &str,
) -> Result<Option<Vec<libcamera::framebuffer_allocator::FrameBuffer>>, BackendError> {
    match stream {
        Some(s) => {
            let bufs = alloc.alloc(s).map_err(|e| {
                BackendError::InitializationFailed(format!("Alloc {} buffers: {}", label, e))
            })?;
            Ok(Some(bufs))
        }
        None => Ok(None),
    }
}

/// Requeue a completed request, warning on failure unless stop is requested.
fn requeue_request(
    cam: &mut libcamera::camera::ActiveCamera,
    req: libcamera::request::Request,
    stop_flag: &AtomicBool,
) {
    if let Err((_, e)) = cam.queue_request(req)
        && !stop_flag.load(Ordering::Acquire)
    {
        warn!(error = %e, "Failed to re-queue request");
    }
}

/// Read back format info from a configured stream at the given index.
/// Returns (format_name, size, mapped_pixel_format, stride).
///
/// `config_repr` is the string from `config.to_string_repr()` which contains
/// the actual negotiated format names (e.g. "SRGGB10_CSI2P") that the Rust
/// binding's `get_pixel_format()` fourcc may not preserve (it returns the
/// unpacked fourcc "RG10" for CSI2P packed Bayer formats).
fn read_stream_config(
    config: &libcamera::camera::CameraConfiguration,
    index: usize,
    config_repr: &str,
) -> (String, libcamera::geometry::Size, Option<PixelFormat>, u32) {
    let pf = config.get(index).map(|c| c.get_pixel_format());
    let mut name = pf.map(pixel_format_name).unwrap_or_default();
    let size = config
        .get(index)
        .map(|c| c.get_size())
        .unwrap_or(libcamera::geometry::Size::new(0, 0));
    let stride = config.get(index).map(|c| c.get_stride()).unwrap_or(0);
    let mapped = pf.and_then(map_pixel_format);

    // Fix format name from config string when the Rust binding loses CSI2P info.
    // Config repr format: "WxH-FORMAT/CS WxH-FORMAT/CS ..."
    if let Some(actual_name) = parse_format_from_config_repr(config_repr, index)
        && actual_name != name
    {
        name = actual_name;
    }

    (name, size, mapped, stride)
}

/// Extract the format name for stream `index` from the config string representation.
/// Format: "1436x1080-ABGR8888/sRGB 3280x2464-SRGGB10_CSI2P/RAW"
fn parse_format_from_config_repr(config_repr: &str, index: usize) -> Option<String> {
    let stream_desc = config_repr.split_whitespace().nth(index)?;
    // "WxH-FORMAT/COLORSPACE" → extract FORMAT
    let after_dash = stream_desc.split_once('-')?.1;
    let format_name = after_dash.split_once('/')?.0;
    Some(format_name.to_string())
}

/// Spatial layout of a camera frame buffer.
struct FrameLayout {
    width: u32,
    height: u32,
    format: PixelFormat,
    stride: u32,
}

/// Build a CameraFrame from copied buffer data and metadata.
///
/// For YUV formats (NV12, NV21, I420), computes `yuv_planes` from the
/// stride and dimensions so that consumers (terminal preview, QR detector,
/// burst mode) don't need format-specific fallback calculations.
fn build_camera_frame(
    data: Arc<[u8]>,
    layout: FrameLayout,
    captured_at: Instant,
    sensor_timestamp_ns: Option<u64>,
    metadata: FrameMetadata,
) -> CameraFrame {
    let yuv_planes = compute_yuv_planes(layout.format, layout.width, layout.height, layout.stride);

    CameraFrame {
        width: layout.width,
        height: layout.height,
        data: FrameData::Copied(data),
        format: layout.format,
        stride: layout.stride,
        yuv_planes,
        captured_at,
        sensor_timestamp_ns,
        libcamera_metadata: Some(metadata),
    }
}

/// Compute YUV plane offsets for multi-plane YUV formats.
///
/// Returns `None` for non-planar formats (RGBA, YUYV, Bayer, etc.)
/// where the data is either single-plane or packed.
fn compute_yuv_planes(
    format: PixelFormat,
    width: u32,
    height: u32,
    stride: u32,
) -> Option<YuvPlanes> {
    match format {
        PixelFormat::NV12 | PixelFormat::NV21 => {
            let y_size = (stride * height) as usize;
            let uv_stride = stride; // interleaved UV, same stride as Y
            let uv_height = height.div_ceil(2);
            let uv_size = (uv_stride * uv_height) as usize;
            Some(YuvPlanes {
                y_offset: 0,
                y_size,
                uv_offset: y_size,
                uv_size,
                uv_stride,
                v_offset: 0,
                v_size: 0,
                v_stride: 0,
                uv_width: width.div_ceil(2),
                uv_height,
            })
        }
        PixelFormat::I420 => {
            let y_size = (stride * height) as usize;
            let uv_stride = stride.div_ceil(2);
            let uv_height = height.div_ceil(2);
            let uv_size = (uv_stride * uv_height) as usize;
            Some(YuvPlanes {
                y_offset: 0,
                y_size,
                uv_offset: y_size,
                uv_size,
                uv_stride,
                v_offset: y_size + uv_size,
                v_size: uv_size,
                v_stride: uv_stride,
                uv_width: width.div_ceil(2),
                uv_height,
            })
        }
        _ => None,
    }
}

/// Extract buffer data from a completed request, copy it, and build a CameraFrame.
///
/// Common pattern used by video and raw buffer processing. Returns None if the
/// buffer is missing, empty, or has no plane data.
fn process_buffer(
    req: &libcamera::request::Request,
    stream: &libcamera::stream::Stream,
    size: libcamera::geometry::Size,
    config_stride: u32,
    pixel_format: PixelFormat,
    captured_at: Instant,
    metadata: &FrameMetadata,
) -> Option<CameraFrame> {
    use libcamera::framebuffer::AsFrameBuffer;
    use libcamera::framebuffer_map::MemoryMappedFrameBuffer;
    type MmapFB = MemoryMappedFrameBuffer<libcamera::framebuffer_allocator::FrameBuffer>;

    let buf = req.buffer::<MmapFB>(stream)?;
    let planes = buf.data();
    if planes.is_empty() {
        return None;
    }

    let fallback_lens: Vec<usize> = planes.iter().map(|p| p.len()).collect();
    let used = planes_bytes_used(buf, &fallback_lens);

    // Concatenate all mmap planes into a single contiguous buffer.
    // libcamera may return multi-plane formats (e.g. NV12) as separate
    // mmap regions depending on the driver/platform.
    let total_bytes: usize = used.iter().sum();
    if total_bytes == 0 {
        return None;
    }

    let data: Arc<[u8]> = if planes.len() == 1 {
        let bytes = used[0].min(planes[0].len());
        Arc::from(&planes[0][..bytes])
    } else {
        let mut combined = Vec::with_capacity(total_bytes);
        for (plane, &bytes) in planes.iter().zip(used.iter()) {
            let bytes = bytes.min(plane.len());
            combined.extend_from_slice(&plane[..bytes]);
        }
        Arc::from(combined)
    };

    let stride = if config_stride > 0 {
        config_stride
    } else {
        compute_stride(pixel_format, size.width, data.len(), size.height)
    };

    let sensor_timestamp_ns = buf.metadata().map(|m| m.timestamp());
    Some(build_camera_frame(
        data,
        FrameLayout {
            width: size.width,
            height: size.height,
            format: pixel_format,
            stride,
        },
        captured_at,
        sensor_timestamp_ns,
        metadata.clone(),
    ))
}

/// Main entry point for the capture thread
///
/// Creates all libcamera objects, runs the capture loop, and cleans up on exit.
/// Reports initialization success/failure via `init_tx`.
pub(crate) fn capture_thread_main(
    params: CaptureThreadParams,
    init_tx: std::sync::mpsc::SyncSender<Result<CaptureThreadInitResult, BackendError>>,
) {
    match capture_thread_setup_and_run(params, &init_tx) {
        Ok(()) => info!("Capture thread exiting normally"),
        Err(e) => {
            error!(error = %e, "Capture thread failed");
            // Clear CAPTURE_ACTIVE so the camera can be retried after failure.
            // The success path clears this at the end of capture_thread_setup_and_run,
            // but error paths via `?` skip that cleanup.
            {
                let _lock = super::super::CAMERA_MANAGER_LOCK
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                CAPTURE_ACTIVE.store(false, Ordering::Release);
                CAPTURE_RELEASED.1.notify_all();
            }
            // If init_tx hasn't been used yet, report the error
            let _ = init_tx.try_send(Err(e));
        }
    }
}

/// How often to emit per-frame diagnostic log messages (every Nth frame).
const LOG_EVERY_N_FRAMES: u64 = 30;

/// Maximum number of progressive scans accepted in a single MJPEG frame.
/// Baseline MJPEG has one scan; the limit only exists to bound decode time on
/// malformed or hostile input.
const MJPEG_SCAN_LIMIT: u32 = 500;

/// Stream format metadata read back after configure.
struct StreamFormats {
    vf_size: libcamera::geometry::Size,
    vf_stride: u32,
    vf_pixel_format: PixelFormat,
    vf_format_name: String,
    vf_is_mjpeg: bool,
    raw_size: libcamera::geometry::Size,
    raw_stride: u32,
    raw_format_name: String,
    raw_pixel_format: Option<PixelFormat>,
}

/// Set up libcamera and run the capture loop.
///
/// All libcamera objects (CameraManager, Camera, ActiveCamera, buffers) must
/// live in this function's scope because they form a borrow chain that prevents
/// splitting ownership across returned structs. The three logical phases are
/// separated into clearly marked sections.
fn capture_thread_setup_and_run(
    mut params: CaptureThreadParams,
    init_tx: &std::sync::mpsc::SyncSender<Result<CaptureThreadInitResult, BackendError>>,
) -> Result<(), BackendError> {
    use libcamera::camera_manager::CameraManager;

    crate::startup::milestone("capture_thread_setup_start");

    // Wait for any previous CameraManager to be fully dropped.
    // libcamera enforces a singleton — creating a second instance is fatal.
    // The old capture thread signals CAPTURE_RELEASED after dropping its manager.
    //
    // We must hold CAMERA_MANAGER_LOCK when checking/setting CAPTURE_ACTIVE and
    // when creating CameraManager to serialize all threads through this section.
    let mgr = {
        let mut mgr_lock = super::super::CAMERA_MANAGER_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if CAPTURE_ACTIVE.load(Ordering::Acquire) {
            // Release CAMERA_MANAGER_LOCK while waiting so the old thread can
            // acquire it during its cleanup path.
            drop(mgr_lock);
            info!("Waiting for previous CameraManager to be released...");
            let (lock, cvar) = &CAPTURE_RELEASED;
            let guard = lock.lock().unwrap_or_else(|e| e.into_inner());
            let cancel = &params.cancel_flag;
            let _guard = cvar
                .wait_timeout_while(guard, std::time::Duration::from_secs(5), |_| {
                    CAPTURE_ACTIVE.load(Ordering::Acquire) && !cancel.load(Ordering::Acquire)
                })
                .unwrap_or_else(|e| e.into_inner());

            // Re-acquire CAMERA_MANAGER_LOCK before proceeding — this serializes
            // multiple waiting threads so only one creates a CameraManager.
            mgr_lock = super::super::CAMERA_MANAGER_LOCK
                .lock()
                .unwrap_or_else(|e| e.into_inner());

            // If the old CameraManager still hasn't been released, bail out
            // rather than creating a fatal duplicate.
            if CAPTURE_ACTIVE.load(Ordering::Acquire) && !params.cancel_flag.load(Ordering::Acquire)
            {
                drop(mgr_lock);
                return Err(BackendError::InitializationFailed(
                    "Timed out waiting for previous CameraManager to release".to_string(),
                ));
            }
        }

        // Early out if superseded by a newer mode switch
        if params.cancel_flag.load(Ordering::Acquire) {
            info!("Cancel flag set — aborting capture thread startup");
            return Ok(());
        }

        // Mark ourselves as active while still holding the lock — no other thread
        // can slip in between this store and CameraManager::new().
        CAPTURE_ACTIVE.store(true, Ordering::Release);

        // No hardware-release delay: the condvar wait above already guarantees
        // the previous CameraManager has been dropped, which is the only
        // ordering libcamera actually documents. The old flat 500 ms sleep on
        // top of that was a guess, and it dominated every camera switch.
        //
        // If a device turns out to need settling time after the drop, the fix
        // belongs here — but as a retry around the acquire/start below, keyed on
        // the real failure, not an unconditional wait every device pays.
        CameraManager::new()
            .map_err(|e| BackendError::InitializationFailed(format!("CameraManager::new: {}", e)))?
    };
    crate::startup::milestone("capture_camera_manager_ready");

    info!(version = mgr.version(), "libcamera version");

    // Get camera by ID
    let cam = mgr.get(&params.camera_id).ok_or_else(|| {
        BackendError::DeviceNotFound(format!("Camera '{}' not found", params.camera_id))
    })?;

    // Acquire exclusive access
    let mut active_cam = cam
        .acquire()
        .map_err(|e| BackendError::InitializationFailed(format!("Camera acquire: {}", e)))?;

    let (config, is_multistream) = configure_streams(&cam, &mut active_cam, &params)?;

    let formats = read_stream_formats(&config, is_multistream);

    info!(
        vf_format = %formats.vf_format_name,
        vf_size = format!("{}x{}", formats.vf_size.width, formats.vf_size.height),
        vf_stride = formats.vf_stride,
        vf_pixel_format = ?formats.vf_pixel_format,
        raw_format = %formats.raw_format_name,
        raw_size = format!("{}x{}", formats.raw_size.width, formats.raw_size.height),
        raw_pixel_format = ?formats.raw_pixel_format,
        multistream = is_multistream,
        "Actual configured formats (post-configure)"
    );

    // Get stream references (must be done after configure)
    let stream_vf = config
        .get(0)
        .and_then(|c| c.stream())
        .ok_or_else(|| BackendError::InitializationFailed("No viewfinder stream".to_string()))?;

    let stream_raw = if is_multistream {
        config.get(1).and_then(|c| c.stream())
    } else {
        None
    };

    let (alloc, requests) = allocate_and_create_requests(
        &cam,
        &mut active_cam,
        &stream_vf,
        &stream_raw,
        is_multistream,
    )?;

    // Subscribe to request completion via channel
    let rx = active_cam.subscribe_request_completed();

    // Start camera
    active_cam
        .start(None)
        .map_err(|e| BackendError::InitializationFailed(format!("Camera start: {}", e)))?;
    crate::startup::milestone("camera_hardware_started");

    // Queue all requests
    for req in requests {
        if let Err((_, e)) = active_cam.queue_request(req) {
            error!(error = %e, "Failed to queue initial request");
        }
    }

    info!("Camera started, all requests queued");

    // Publish diagnostics
    publish_diagnostics(DiagnosticParams {
        is_multistream,
        vf: StreamDiag {
            size: formats.vf_size,
            format_name: formats.vf_format_name.clone(),
            stride: formats.vf_stride,
        },
        raw: StreamDiag {
            size: formats.raw_size,
            format_name: formats.raw_format_name.clone(),
            stride: formats.raw_stride,
        },
    });

    // Create MJPEG decompressor if needed — BEFORE reporting success so that
    // init failures are propagated to the caller via init_tx.
    let mut jpeg_decompressor = if formats.vf_is_mjpeg {
        let mut decompressor = turbojpeg::Decompressor::new()
            .map_err(|e| BackendError::Other(format!("turbojpeg init: {e}")))?;
        // Bound the work a single frame can cost us. A camera emitting a
        // progressive JPEG with a pathological scan count would otherwise stall
        // the capture thread frame after frame; MJPEG in practice is baseline,
        // so a generous limit never fires for well-behaved hardware.
        if let Err(e) = decompressor.set_scan_limit(MJPEG_SCAN_LIMIT) {
            debug!(error = %e, "Could not set MJPEG scan limit");
        }
        if let Ok(mut d) = DIAGNOSTICS.write() {
            d.mjpeg_decoder_name = Some("turbojpeg (libjpeg-turbo)".to_string());
        }
        Some(decompressor)
    } else {
        None
    };

    // Report successful initialization (after all fallible init steps)
    init_tx
        .send(Ok(CaptureThreadInitResult { is_multistream }))
        .map_err(|_| {
            BackendError::InitializationFailed("Main thread dropped init receiver".to_string())
        })?;

    run_capture_loop(
        &mut active_cam,
        &rx,
        &stream_vf,
        &stream_raw,
        &formats,
        is_multistream,
        &mut jpeg_decompressor,
        &mut params,
    );

    // Stop camera (ActiveCamera::drop also does this, but explicit is cleaner)
    info!("Capture loop ending, stopping camera");
    let _ = active_cam.stop();

    // Drop all libcamera objects in correct dependency order.
    // Previously, dropping mgr before cam/alloc caused use-after-free (SIGSEGV)
    // because those objects still referenced the CameraManager's internal state.
    drop(rx);
    drop(alloc);
    drop(config);
    drop(active_cam);
    drop(cam);
    drop(mgr);

    // Now clear the flag under the lock so that any enumerate_cameras()/get_formats()
    // call that was waiting will see CAPTURE_ACTIVE=false and proceed normally.
    // Clear the flag and wake any waiting capture thread immediately.
    {
        let _lock = super::super::CAMERA_MANAGER_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        CAPTURE_ACTIVE.store(false, Ordering::Release);
        CAPTURE_RELEASED.1.notify_all();
    }
    info!("CameraManager released");

    Ok(())
}

/// Generate camera configuration with stream roles and apply to hardware.
fn configure_streams(
    cam: &libcamera::camera::Camera<'_>,
    active_cam: &mut libcamera::camera::ActiveCamera<'_>,
    params: &CaptureThreadParams,
) -> Result<(libcamera::camera::CameraConfiguration, bool), BackendError> {
    use libcamera::stream::StreamRole;

    // All modes use the same stream layout to avoid pipeline restarts on mode switch.
    // ViewFinder+Raw on multistream cameras: ViewFinder for preview and video encoding
    // (via appsrc), Raw for full-res Bayer photo/timelapse capture.
    // ViewFinder only on single-stream cameras.
    let roles = if params.supports_multistream {
        vec![StreamRole::ViewFinder, StreamRole::Raw]
    } else {
        vec![StreamRole::ViewFinder]
    };

    info!(
        second_stream = params.supports_multistream,
        "Requesting stream roles"
    );

    let mut config = cam.generate_configuration(&roles).ok_or_else(|| {
        BackendError::InitializationFailed("Failed to generate camera configuration".to_string())
    })?;

    let is_multistream = config.len() >= 2 && params.supports_multistream;

    // Configure stream sizes
    if let Some(mut vf_cfg) = config.get_mut(0) {
        let size = libcamera::geometry::Size::new(params.preview_width, params.preview_height);
        vf_cfg.set_size(size);
        debug!(
            width = params.preview_width,
            height = params.preview_height,
            "Set viewfinder size"
        );
    }

    let status = config.validate();
    info!(
        status = ?status,
        config = config.to_string_repr(),
        "Configuration validated"
    );

    active_cam
        .configure(&mut config)
        .map_err(|e| BackendError::InitializationFailed(format!("Camera configure: {}", e)))?;

    Ok((config, is_multistream))
}

/// Read back actual stream formats after hardware configure.
/// (configure may adjust stride/size beyond what validate reported)
fn read_stream_formats(
    config: &libcamera::camera::CameraConfiguration,
    is_multistream: bool,
) -> StreamFormats {
    let config_repr = config.to_string_repr();
    let (vf_format_name, vf_size, vf_mapped, vf_stride) =
        read_stream_config(config, 0, &config_repr);
    let vf_is_mjpeg = config
        .get(0)
        .map(|c| c.get_pixel_format().fourcc() == u32::from_le_bytes(*b"MJPG"))
        .unwrap_or(false);
    let vf_pixel_format = if vf_is_mjpeg {
        info!("Viewfinder using MJPEG — will decode to native YUV via turbojpeg");
        PixelFormat::I420
    } else {
        vf_mapped.unwrap_or_else(|| {
            // Falling back to ABGR for an unmapped format mislabels the
            // per-pixel stride (e.g. 1-byte R8 frames get treated as 4-byte
            // ABGR), which downstream crashes the QR detector with an
            // index-out-of-bounds and wgpu with "bytes per row is less than
            // a complete row" (issue #406). Log loudly so the format that
            // slipped through `map_pixel_format` is easy to identify.
            error!(
                format = %vf_format_name,
                "Unknown viewfinder pixel format — falling back to ABGR; \
                 downstream pipeline will likely fail. Add a mapping in \
                 `src/backends/camera/libcamera/native/pixel_formats.rs`."
            );
            PixelFormat::ABGR
        })
    };

    let (raw_format_name, raw_size, raw_pixel_format, raw_stride) = if is_multistream {
        let (name, size, mapped, stride) = read_stream_config(config, 1, &config_repr);
        (name, size, mapped, stride)
    } else {
        (
            String::new(),
            libcamera::geometry::Size::new(0, 0),
            None,
            0u32,
        )
    };

    StreamFormats {
        vf_size,
        vf_stride,
        vf_pixel_format,
        vf_format_name,
        vf_is_mjpeg,
        raw_size,
        raw_stride,
        raw_format_name,
        raw_pixel_format,
    }
}

/// Allocate frame buffers, memory-map them, and create capture requests.
///
/// Returns the allocator (which must outlive the requests) and the request vector.
fn allocate_and_create_requests(
    cam: &libcamera::camera::Camera<'_>,
    active_cam: &mut libcamera::camera::ActiveCamera<'_>,
    stream_vf: &libcamera::stream::Stream,
    stream_raw: &Option<libcamera::stream::Stream>,
    is_multistream: bool,
) -> Result<
    (
        libcamera::framebuffer_allocator::FrameBufferAllocator,
        Vec<libcamera::request::Request>,
    ),
    BackendError,
> {
    use libcamera::framebuffer_allocator::FrameBufferAllocator;
    use libcamera::request::Request;

    let mut alloc = FrameBufferAllocator::new(cam);

    let vf_buffers = alloc.alloc(stream_vf).map_err(|e| {
        BackendError::InitializationFailed(format!("Alloc viewfinder buffers: {}", e))
    })?;

    let raw_buffers = if is_multistream {
        alloc_optional_buffers(&mut alloc, stream_raw, "raw")?
    } else {
        None
    };

    info!(
        vf_buffers = vf_buffers.len(),
        raw_buffers = raw_buffers.as_ref().map(|b| b.len()).unwrap_or(0),
        "Allocated buffers"
    );

    // Wrap buffers in memory-mapped wrappers for data access
    let vf_mapped = mmap_buffers(vf_buffers, "vf")?;
    let raw_mapped = raw_buffers.map(|b| mmap_buffers(b, "raw")).transpose()?;

    // Create requests with buffers
    let buf_count = vf_mapped.len();
    let mut requests: Vec<Request> = Vec::with_capacity(buf_count);
    let mut vf_iter = vf_mapped.into_iter();
    let mut raw_iter = raw_mapped.map(|v| v.into_iter());

    for i in 0..buf_count {
        let mut req = active_cam.create_request(Some(i as u64)).ok_or_else(|| {
            BackendError::InitializationFailed("Failed to create request".to_string())
        })?;

        let vf_buf = vf_iter.next().ok_or_else(|| {
            BackendError::InitializationFailed("Not enough vf buffers".to_string())
        })?;
        req.add_buffer(stream_vf, vf_buf)
            .map_err(|e| BackendError::InitializationFailed(format!("Add vf buffer: {}", e)))?;

        if let Some(ref mut raw_it) = raw_iter
            && let Some(raw_buf) = raw_it.next()
            && let Some(sr) = stream_raw
        {
            req.add_buffer(sr, raw_buf).map_err(|e| {
                BackendError::InitializationFailed(format!("Add raw buffer: {}", e))
            })?;
        }

        requests.push(req);
    }

    info!(requests = requests.len(), "Created capture requests");
    Ok((alloc, requests))
}

/// Run the capture loop — receive completed requests, process viewfinder/raw
/// buffers, send frames to consumers. Returns when `stop_flag` is set or the
/// request channel disconnects.
#[allow(clippy::too_many_arguments)]
fn run_capture_loop(
    active_cam: &mut libcamera::camera::ActiveCamera<'_>,
    rx: &std::sync::mpsc::Receiver<libcamera::request::Request>,
    stream_vf: &libcamera::stream::Stream,
    stream_raw: &Option<libcamera::stream::Stream>,
    formats: &StreamFormats,
    is_multistream: bool,
    jpeg_decompressor: &mut Option<turbojpeg::Decompressor>,
    params: &mut CaptureThreadParams,
) {
    use libcamera::framebuffer::AsFrameBuffer;
    use libcamera::framebuffer_map::MemoryMappedFrameBuffer;
    use libcamera::request::{RequestStatus, ReuseFlag};

    info!("Entering capture loop");

    // Reusable buffer for MJPEG→YUV decompression.
    // Avoids allocating + zeroing several MB per frame.
    let mut jpeg_yuv_buf: Vec<u8> = Vec::new();

    type MmapFB = MemoryMappedFrameBuffer<libcamera::framebuffer_allocator::FrameBuffer>;

    while !params.stop_flag.load(Ordering::Acquire) {
        // Wait for completed request with timeout
        let mut req = match rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(req) => req,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                info!("Request channel disconnected, stopping capture loop");
                break;
            }
        };

        if req.status() != RequestStatus::Complete {
            debug!(status = ?req.status(), "Skipping non-complete request");
            req.reuse(ReuseFlag::REUSE_BUFFERS);
            requeue_request(active_cam, req, &params.stop_flag);
            continue;
        }

        let captured_at = Instant::now();
        let frame_num = params.preview_frame_count.fetch_add(1, Ordering::Relaxed);

        // Extract per-frame metadata
        let metadata = extract_metadata(&req);

        // Process viewfinder buffer
        if let Some(vf_buf) = req.buffer::<MmapFB>(stream_vf) {
            // Extract kernel buffer timestamp (CLOCK_BOOTTIME ns) for video recording PTS
            let sensor_timestamp_ns = vf_buf.metadata().map(|m| m.timestamp());
            let planes = vf_buf.data();
            if !planes.is_empty() {
                let fallback_lens: Vec<usize> = planes.iter().map(|p| p.len()).collect();
                let used = planes_bytes_used(vf_buf, &fallback_lens);
                let total_bytes_used: usize = used.iter().sum();

                let expected_size = formats.vf_stride as usize * formats.vf_size.height as usize;
                if frame_num == 0 {
                    info!(
                        bytes_used = total_bytes_used,
                        num_planes = planes.len(),
                        plane_lens = ?fallback_lens,
                        expected_size,
                        vf_stride = formats.vf_stride,
                        width = formats.vf_size.width,
                        height = formats.vf_size.height,
                        "First frame buffer diagnostics"
                    );
                }

                // Skip frames with no data (e.g. first frame from some UVC cameras)
                if total_bytes_used == 0 {
                    req.reuse(ReuseFlag::REUSE_BUFFERS);
                    requeue_request(active_cam, req, &params.stop_flag);
                    continue;
                }

                // Concatenate all mmap planes into a single contiguous buffer.
                // libcamera may return multi-plane formats (e.g. NV12) as separate
                // mmap regions depending on the driver/platform.
                let combined_data: Vec<u8> = if planes.len() == 1 {
                    let bytes = used[0].min(planes[0].len());
                    planes[0][..bytes].to_vec()
                } else {
                    let mut combined = Vec::with_capacity(total_bytes_used);
                    for (plane, &bytes) in planes.iter().zip(used.iter()) {
                        let bytes = bytes.min(plane.len());
                        combined.extend_from_slice(&plane[..bytes]);
                    }
                    combined
                };
                let data_slice = combined_data.as_slice();

                // If JPEG recording mode is active and this is an MJPEG stream,
                // send raw JPEG bytes to the recorder BEFORE the CPU decode.
                // The recorder's VA-API pipeline will decode on GPU.
                let jpeg_sent_to_recorder = if formats.vf_is_mjpeg
                    && params.jpeg_recording_mode.load(Ordering::Relaxed)
                {
                    if let Ok(guard) = params.recording_sender.lock()
                        && let Some(ref tx) = *guard
                    {
                        let seq = metadata.sequence;
                        let send_result = tx.try_send(RecordingFrame::Jpeg {
                            data: Arc::from(data_slice),
                            width: formats.vf_size.width,
                            height: formats.vf_size.height,
                            sensor_timestamp_ns,
                            sequence: seq,
                        });
                        if send_result.is_err() {
                            crate::pipelines::video::stats::rec_stats_capture_dropped();
                            if frame_num.is_multiple_of(LOG_EVERY_N_FRAMES) {
                                warn!(frame = frame_num, seq = ?seq, "JPEG recording frame dropped (channel full)");
                            }
                        } else {
                            crate::pipelines::video::stats::rec_stats_capture_sent();
                        }
                        if frame_num.is_multiple_of(LOG_EVERY_N_FRAMES) {
                            debug!(
                                frame = frame_num,
                                seq = ?seq,
                                sensor_ts_ms = ?sensor_timestamp_ns.map(|t| t / 1_000_000),
                                "Sent JPEG to recorder"
                            );
                        }
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };

                // Decode MJPEG or copy raw data from mmap BEFORE req.reuse()
                let frame = if let Some(decompressor) = jpeg_decompressor {
                    let decode_start = Instant::now();
                    let decode_result = decode_mjpeg_frame(
                        decompressor,
                        data_slice,
                        captured_at,
                        sensor_timestamp_ns,
                        &metadata,
                        frame_num,
                        &mut jpeg_yuv_buf,
                    );
                    MJPEG_DECODE_TIME_US
                        .store(decode_start.elapsed().as_micros() as u64, Ordering::Relaxed);
                    match decode_result {
                        Some(f) => f,
                        None => {
                            // Decode failed, skip this frame
                            req.reuse(ReuseFlag::REUSE_BUFFERS);
                            requeue_request(active_cam, req, &params.stop_flag);
                            continue;
                        }
                    }
                } else {
                    let data: Arc<[u8]> = Arc::from(data_slice);

                    let stride = if formats.vf_stride > 0 {
                        formats.vf_stride
                    } else {
                        compute_stride(
                            formats.vf_pixel_format,
                            formats.vf_size.width,
                            data.len(),
                            formats.vf_size.height,
                        )
                    };

                    build_camera_frame(
                        data,
                        FrameLayout {
                            width: formats.vf_size.width,
                            height: formats.vf_size.height,
                            format: formats.vf_pixel_format,
                            stride,
                        },
                        captured_at,
                        sensor_timestamp_ns,
                        metadata.clone(),
                    )
                };

                dispatch_viewfinder_frame(
                    frame,
                    frame_num,
                    params,
                    is_multistream,
                    jpeg_sent_to_recorder,
                );
            }
        } else if frame_num.is_multiple_of(LOG_EVERY_N_FRAMES) {
            warn!("No viewfinder buffer in completed request");
        }

        // Process raw buffer only when still capture is requested (multistream only).
        // Acquire pairs with the Release store in `request_still_capture`.
        if is_multistream
            && params.still_requested.load(Ordering::Acquire)
            && let Some(sr) = stream_raw
        {
            let raw_pf = formats.raw_pixel_format.unwrap_or(PixelFormat::BayerRGGB);
            if let Some(still_frame) = process_buffer(
                &req,
                sr,
                formats.raw_size,
                formats.raw_stride,
                raw_pf,
                captured_at,
                &metadata,
            ) {
                let still_count = params.still_frame_count.fetch_add(1, Ordering::Relaxed);
                info!(
                    frame = still_count,
                    width = still_frame.width,
                    height = still_frame.height,
                    format = ?still_frame.format,
                    "Raw still frame captured"
                );

                let stored = if let Ok(mut still) = params.latest_still.lock() {
                    *still = Some(still_frame);
                    params.still_requested.store(false, Ordering::Release);
                    true
                } else {
                    false
                };
                if stored {
                    params.still_frame_notify.notify_waiters();
                }
            }
        }

        // Update global frame counts for insights (lock-free)
        PREVIEW_FRAME_COUNT.store(
            params.preview_frame_count.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        STILL_FRAME_COUNT.store(
            params.still_frame_count.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );

        // Reuse request and re-queue
        req.reuse(ReuseFlag::REUSE_BUFFERS);
        requeue_request(active_cam, req, &params.stop_flag);
    }
}

/// Dispatch a completed viewfinder frame to all consumers:
/// 1. Store as latest preview (for still capture fallback)
/// 2. Handle single-stream still capture
/// 3. Send to recording (VF→encoder via appsrc)
/// 4. Send to UI preview channel
fn dispatch_viewfinder_frame(
    frame: CameraFrame,
    frame_num: u64,
    params: &mut CaptureThreadParams,
    is_multistream: bool,
    skip_recording: bool,
) {
    static FIRST_FRAME: AtomicBool = AtomicBool::new(false);
    crate::startup::milestone_once("first_frame_captured", &FIRST_FRAME);

    // Store latest preview frame
    if let Ok(mut latest) = params.latest_preview.lock() {
        *latest = Some(frame.clone());
    }

    // In single-stream mode, also handle still capture from preview.
    // Acquire pairs with the Release store in `request_still_capture`.
    if !is_multistream && params.still_requested.load(Ordering::Acquire) {
        info!(
            frame = frame_num,
            width = frame.width,
            height = frame.height,
            "Still frame captured from preview stream"
        );
        let stored = if let Ok(mut still) = params.latest_still.lock() {
            *still = Some(frame.clone());
            true
        } else {
            false
        };
        params.still_requested.store(false, Ordering::Release);
        if stored {
            params.still_frame_notify.notify_waiters();
        }
    }

    // Send ViewFinder frames to recording (via appsrc encoder pipeline).
    // Skip if raw JPEG was already sent to the recorder in JPEG mode.
    if !skip_recording
        && let Ok(guard) = params.recording_sender.lock()
        && let Some(ref tx) = *guard
    {
        let seq = frame.libcamera_metadata.as_ref().and_then(|m| m.sequence);
        let send_result = tx.try_send(RecordingFrame::Decoded(Arc::new(frame.clone())));
        if send_result.is_err() {
            crate::pipelines::video::stats::rec_stats_capture_dropped();
        } else {
            crate::pipelines::video::stats::rec_stats_capture_sent();
        }
        if frame_num.is_multiple_of(LOG_EVERY_N_FRAMES) {
            debug!(
                frame = frame_num,
                seq = ?seq,
                sensor_ts_ms = ?frame.sensor_timestamp_ns.map(|t| t / 1_000_000),
                dropped = send_result.is_err(),
                "Sent decoded frame to recorder"
            );
        }
    }

    // Send to UI (preview)
    if let Err(e) = params.frame_sender.try_send(frame)
        && frame_num.is_multiple_of(LOG_EVERY_N_FRAMES)
    {
        debug!(error = ?e, "Preview frame dropped (channel full)");
    }
}

/// Extract per-frame metadata from a completed request
fn extract_metadata(req: &libcamera::request::Request) -> FrameMetadata {
    use libcamera::controls;
    let meta = req.metadata();

    FrameMetadata {
        exposure_time: meta
            .get::<controls::ExposureTime>()
            .ok()
            .map(|v| v.0 as u64),
        analogue_gain: meta.get::<controls::AnalogueGain>().ok().map(|v| v.0),
        digital_gain: meta.get::<controls::DigitalGain>().ok().map(|v| v.0),
        colour_temperature: meta
            .get::<controls::ColourTemperature>()
            .ok()
            .map(|v| v.0 as u32),
        sequence: Some(req.sequence()),
        lens_position: meta.get::<controls::LensPosition>().ok().map(|v| v.0),
        sensor_timestamp: meta
            .get::<controls::SensorTimestamp>()
            .ok()
            .map(|v| v.0 as u64),
        af_state: None,
        ae_state: None,
        awb_state: None,
        colour_gains: meta.get::<controls::ColourGains>().ok().map(|v| v.0),
        colour_correction_matrix: meta
            .get::<controls::ColourCorrectionMatrix>()
            .ok()
            .map(|v| v.0),
        black_level: meta.get::<controls::SensorBlackLevels>().ok().map(|v| {
            // SensorBlackLevels are 16-bit [R, Gr, Gb, B]; average and normalize
            let levels = v.0;
            let avg =
                (levels[0] as f32 + levels[1] as f32 + levels[2] as f32 + levels[3] as f32) / 4.0;
            avg / 65535.0
        }),
        lux: meta.get::<controls::Lux>().ok().map(|v| v.0),
        focus_fom: meta.get::<controls::FocusFoM>().ok().map(|v| v.0),
    }
}

/// Decode a MJPEG frame to native planar YUV via turbojpeg
///
/// turbojpeg's `decompress_to_yuv` produces the native subsampling:
/// - 4:2:0 JPEG → I420 (Sub2x2)
/// - 4:2:2 JPEG → I422 (Sub2x1)
/// - 4:4:4 JPEG → I444 (None)
/// - etc.
///
/// The GPU shader handles all subsampling types via `textureDimensions()`.
///
/// `yuv_buf` is a reusable buffer to avoid allocating on every frame.
/// Returns `None` if decoding fails (caller should skip the frame).
fn decode_mjpeg_frame(
    decompressor: &mut turbojpeg::Decompressor,
    jpeg_data: &[u8],
    captured_at: Instant,
    sensor_timestamp_ns: Option<u64>,
    metadata: &FrameMetadata,
    frame_num: u64,
    yuv_buf: &mut Vec<u8>,
) -> Option<CameraFrame> {
    // Read JPEG header to get dimensions and subsampling
    let header = match decompressor.read_header(jpeg_data) {
        Ok(h) => h,
        Err(e) => {
            if frame_num.is_multiple_of(LOG_EVERY_N_FRAMES) {
                warn!(error = %e, "MJPEG header read failed, skipping frame");
            }
            return None;
        }
    };

    let width = header.width;
    let height = header.height;

    // Compute YUV buffer size for this subsampling
    let align = 1; // no row padding
    let buf_len = match turbojpeg::yuv_pixels_len(width, align, height, header.subsamp) {
        Ok(len) => len,
        Err(e) => {
            warn!(error = %e, subsamp = ?header.subsamp, "Failed to compute YUV buffer size");
            return None;
        }
    };

    // Resize reusable buffer (free if same size, which it will be after first frame)
    yuv_buf.resize(buf_len, 0);

    let yuv_image = turbojpeg::YuvImage {
        pixels: &mut yuv_buf[..],
        width,
        align,
        height,
        subsamp: header.subsamp,
    };

    if let Err(e) = decompressor.decompress_to_yuv(jpeg_data, yuv_image) {
        // libjpeg-turbo classifies some MJPEG quirks as *warnings* — the YUV
        // output is still written and visually fine. The Rust `turbojpeg`
        // wrapper surfaces these as `Err`, which would otherwise make us drop
        // every frame from cameras that emit them.
        //
        // Known offenders: Logitech C270 (046d:0825) and several other UVC
        // webcams emit MJPEG with restart-marker stuffing bytes between
        // segments, triggering "extraneous bytes before marker 0xdN".
        let msg = e.to_string();
        let recoverable = msg.contains("extraneous bytes") || msg.contains("Premature end");
        if !recoverable {
            if frame_num.is_multiple_of(LOG_EVERY_N_FRAMES) {
                warn!(error = %e, subsamp = ?header.subsamp, "MJPEG decompress to YUV failed, skipping frame");
            }
            return None;
        }
        if frame_num == 0 {
            debug!(
                error = %e,
                subsamp = ?header.subsamp,
                "Recoverable MJPEG warning — accepting decoded YUV anyway"
            );
        }
    }

    // Compute UV plane dimensions based on subsampling
    let (uv_w, uv_h) = match header.subsamp {
        turbojpeg::Subsamp::None => (width, height), // 4:4:4
        turbojpeg::Subsamp::Sub2x1 => (width.div_ceil(2), height), // 4:2:2
        turbojpeg::Subsamp::Sub1x2 => (width, height.div_ceil(2)), // 4:4:0
        turbojpeg::Subsamp::Gray => (0, 0),          // Grayscale
        _ => (width.div_ceil(2), height.div_ceil(2)), // 4:2:0
    };

    // Determine standard YUV format name from subsampling
    let format_name = match header.subsamp {
        turbojpeg::Subsamp::Sub2x2 => "I420",
        turbojpeg::Subsamp::Sub2x1 => "I422",
        turbojpeg::Subsamp::None => "I444",
        turbojpeg::Subsamp::Sub1x2 => "I440",
        turbojpeg::Subsamp::Gray => "Y800",
        _ => "YUV",
    };

    // Update decoded format and stream info on first frame
    if frame_num == 0 {
        // Baseline Huffman is what UVC cameras emit; anything else costs
        // noticeably more CPU per frame, so record it for the insights panel.
        let entropy_note = match (header.is_progressive, header.is_arithmetic) {
            (true, true) => Some("progressive, arithmetic"),
            (true, false) => Some("progressive"),
            (false, true) => Some("arithmetic"),
            (false, false) => None,
        };

        info!(
            subsamp = ?header.subsamp,
            format_name,
            jpeg_size = jpeg_data.len(),
            yuv_size = buf_len,
            width,
            height,
            uv_w,
            uv_h,
            entropy = entropy_note.unwrap_or("baseline"),
            "First MJPEG frame decoded to {format_name} via turbojpeg"
        );

        if let Some(note) = entropy_note {
            warn!(
                entropy = note,
                "Camera emits non-baseline MJPEG — decoding will be slower than usual"
            );
        }

        if let Ok(mut d) = DIAGNOSTICS.write() {
            d.mjpeg_decoded_format = Some(format_name.to_string());
            d.mjpeg_entropy_note = entropy_note.map(str::to_string);
            // Update stream info pixel format to show actual decoded format
            if let Some(ref mut info) = d.preview_stream_info {
                info.1 = format!("{} (MJPEG)", format_name);
            }
        }
    }

    let y_size = width * height;
    let uv_size = uv_w * uv_h;

    let yuv_planes = YuvPlanes {
        y_offset: 0,
        y_size,
        uv_offset: y_size,
        uv_size,
        uv_stride: uv_w as u32,
        v_offset: y_size + uv_size,
        v_size: uv_size,
        v_stride: uv_w as u32,
        uv_width: uv_w as u32,
        uv_height: uv_h as u32,
    };

    // Copy into Arc<[u8]> for shared ownership across preview/still/channel.
    // The Vec buffer itself is reused across frames (no alloc after first frame).
    let data: Arc<[u8]> = Arc::from(&yuv_buf[..]);

    Some(CameraFrame {
        width: width as u32,
        height: height as u32,
        data: FrameData::Copied(data),
        format: match header.subsamp {
            turbojpeg::Subsamp::Sub2x2 => PixelFormat::I420,
            // No PixelFormat::I422 variant yet; treat 4:2:2 as I420 for now
            // (plane layout is still correct via yuv_planes, only the enum tag is imprecise)
            _ => PixelFormat::I420,
        },
        stride: width as u32,
        yuv_planes: Some(yuv_planes),
        captured_at,
        sensor_timestamp_ns,
        libcamera_metadata: Some(metadata.clone()),
    })
}

/// Compute stride from pixel format, width, and buffer size
fn compute_stride(pf: PixelFormat, width: u32, buffer_size: usize, height: u32) -> u32 {
    if height == 0 {
        return width;
    }

    // For Bayer formats, derive stride from buffer size (may include padding)
    if pf.is_bayer() {
        return (buffer_size as u32) / height;
    }

    // For standard formats, compute from format's bytes per pixel
    let bpp = pf.bytes_per_pixel();
    let computed = (width as f32 * bpp) as u32;

    // Cross-check with buffer: if buffer is larger, use actual stride
    let buffer_stride = (buffer_size as u32) / height;

    // Use the larger of computed and buffer-derived stride
    // (buffer may have alignment padding)
    computed.max(buffer_stride)
}
