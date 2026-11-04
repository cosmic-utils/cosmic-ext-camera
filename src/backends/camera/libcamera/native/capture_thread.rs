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
use super::scaler_crop::NativeScalerCrop;
use crate::backends::camera::preview_metrics::PreviewMetrics;
use crate::backends::camera::still_capture::{StillCapture, StillTicket};
use crate::backends::camera::types::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{debug, error, info, warn};

const MJPEG_DECODE_WORKERS: usize = 2;
const MJPEG_DECODE_QUEUE_CAPACITY: usize = 2;
const CAPTURE_COPY_POOL_CACHE_CAPACITY: usize =
    crate::constants::latency::FRAME_CHANNEL_CAPACITY + 2;

/// Parameters passed to the capture thread for initialization
pub(crate) struct CaptureThreadParams {
    pub(crate) zoom: Arc<crate::backends::camera::zoom::ZoomControl>,
    pub(crate) camera_id: String,
    pub(crate) preview_metrics: Arc<PreviewMetrics>,
    pub(crate) preview_width: u32,
    pub(crate) preview_height: u32,
    pub(crate) limit_preview: bool,
    pub(crate) stop_flag: Arc<AtomicBool>,
    pub(crate) still_capture: Arc<StillCapture>,
    pub(crate) preview_frame_count: Arc<AtomicU64>,
    pub(crate) still_frame_count: Arc<AtomicU64>,
    pub(crate) frame_sender: FrameSender,
    pub(crate) recording_sender: Arc<Mutex<Option<tokio::sync::mpsc::Sender<RecordingFrame>>>>,
    /// Cancel flag — checked before creating CameraManager to abort if a newer
    /// mode switch has superseded this one.
    pub(crate) cancel_flag: Arc<AtomicBool>,
}

/// Result of capture thread initialization (sent back to main thread)
pub(crate) struct CaptureThreadInitResult {
    pub(crate) is_multistream: bool,
}

trait QueueWorker<Job, Output>: Send + 'static {
    fn process(&mut self, job: Job) -> Output;
}

struct BoundedWorkerQueue<Job> {
    job_tx: Option<std::sync::mpsc::SyncSender<Job>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl<Job> BoundedWorkerQueue<Job>
where
    Job: Send + 'static,
{
    fn new_with_result_sender<Worker, Output>(
        workers: Vec<Worker>,
        queue_capacity: usize,
        result_tx: std::sync::mpsc::Sender<Output>,
    ) -> Self
    where
        Worker: QueueWorker<Job, Output>,
        Output: Send + 'static,
    {
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel(queue_capacity);
        let job_rx = Arc::new(Mutex::new(job_rx));
        let workers = workers
            .into_iter()
            .map(|mut worker| {
                let job_rx = Arc::clone(&job_rx);
                let result_tx = result_tx.clone();
                std::thread::spawn(move || {
                    loop {
                        let job = {
                            let receiver = job_rx.lock().unwrap_or_else(|e| e.into_inner());
                            receiver.recv()
                        };
                        let Ok(job) = job else {
                            break;
                        };
                        if result_tx.send(worker.process(job)).is_err() {
                            break;
                        }
                    }
                })
            })
            .collect();

        Self {
            job_tx: Some(job_tx),
            workers,
        }
    }

    fn try_submit_preview(
        &self,
        job: Job,
        metrics: &PreviewMetrics,
    ) -> Result<(), std::sync::mpsc::TrySendError<Job>> {
        let result = self.try_submit(job);
        if result.is_err() {
            metrics.record_drops(1);
        }
        result
    }

    fn try_submit(&self, job: Job) -> Result<(), std::sync::mpsc::TrySendError<Job>> {
        self.job_tx
            .as_ref()
            .expect("worker queue is active")
            .try_send(job)
    }
}

impl<Job> Drop for BoundedWorkerQueue<Job> {
    fn drop(&mut self) {
        self.job_tx.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

struct MjpegDecodeJob {
    data: FrameData,
    captured_at: Instant,
    sensor_timestamp_ns: Option<u64>,
    metadata: FrameMetadata,
    frame_num: u64,
    dispatch_context: FrameDispatchContext<RecordingFrame>,
}

struct MjpegDecodeOutput {
    frame_num: u64,
    frame: Option<CameraFrame>,
    dispatch_context: FrameDispatchContext<RecordingFrame>,
    decode_time_us: u64,
}

// Keep decoded frames inline: boxing would add an allocation and indirection to
// every preview frame solely to make the uncommon request variant smaller.
#[allow(clippy::large_enum_variant)]
enum CaptureEvent {
    Request(libcamera::request::Request),
    MjpegDecoded(MjpegDecodeOutput),
}

struct MjpegDecodeWorker {
    decompressor: turbojpeg::Decompressor,
    yuv_pool: FrameDataPool,
}

impl QueueWorker<MjpegDecodeJob, CaptureEvent> for MjpegDecodeWorker {
    fn process(&mut self, job: MjpegDecodeJob) -> CaptureEvent {
        let decode_start = Instant::now();
        let frame = decode_mjpeg_frame(
            &mut self.decompressor,
            job.data.as_ref(),
            job.captured_at,
            job.sensor_timestamp_ns,
            &job.metadata,
            job.frame_num,
            &self.yuv_pool,
        );
        CaptureEvent::MjpegDecoded(MjpegDecodeOutput {
            frame_num: job.frame_num,
            frame,
            dispatch_context: job.dispatch_context,
            decode_time_us: decode_start.elapsed().as_micros() as u64,
        })
    }
}

type MjpegDecodeQueue = BoundedWorkerQueue<MjpegDecodeJob>;

struct FrameDispatchContext<T> {
    capture_still: Option<StillTicket>,
    recording_sender: Option<tokio::sync::mpsc::Sender<T>>,
}

fn snapshot_frame_dispatch_context<T>(
    is_multistream: bool,
    still_capture: &StillCapture,
    recording_sender: &Mutex<Option<tokio::sync::mpsc::Sender<T>>>,
) -> FrameDispatchContext<T> {
    FrameDispatchContext {
        capture_still: if is_multistream {
            None
        } else {
            still_capture.pending()
        },
        recording_sender: recording_sender
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone(),
    }
}

fn advance_mjpeg_dispatch_watermark(
    last_dispatched_frame: &mut Option<u64>,
    frame_num: u64,
    decode_succeeded: bool,
) -> bool {
    if !decode_succeeded || last_dispatched_frame.is_some_and(|last| frame_num <= last) {
        return false;
    }
    *last_dispatched_frame = Some(frame_num);
    true
}

fn create_mjpeg_decode_queue(
    result_tx: std::sync::mpsc::Sender<CaptureEvent>,
) -> Result<MjpegDecodeQueue, BackendError> {
    let yuv_pool = FrameDataPool::new(MJPEG_DECODE_WORKERS + MJPEG_DECODE_QUEUE_CAPACITY);
    let workers = (0..MJPEG_DECODE_WORKERS)
        .map(|_| {
            let mut decompressor = turbojpeg::Decompressor::new()
                .map_err(|e| BackendError::Other(format!("turbojpeg init: {e}")))?;
            if let Err(e) = decompressor.set_scan_limit(MJPEG_SCAN_LIMIT) {
                debug!(error = %e, "Could not set MJPEG scan limit");
            }
            Ok(MjpegDecodeWorker {
                decompressor,
                yuv_pool: yuv_pool.clone(),
            })
        })
        .collect::<Result<Vec<_>, BackendError>>()?;

    Ok(BoundedWorkerQueue::new_with_result_sender(
        workers,
        MJPEG_DECODE_QUEUE_CAPACITY,
        result_tx,
    ))
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

/// Copy completed capture planes into refcounted, recyclable application storage.
///
/// The copy is required because the libcamera request is requeued immediately
/// after dispatch while preview, QR, recording, and still consumers may retain
/// the frame. Pool storage is returned only after the last `FrameData` clone is
/// dropped, so a later capture can never overwrite a retained frame.
fn pooled_copy_frame_planes(planes: &[&[u8]], used: &[usize], pool: &FrameDataPool) -> FrameData {
    assert_eq!(
        planes.len(),
        used.len(),
        "expected one bytes-used entry per capture plane"
    );
    let copied_len = planes
        .iter()
        .zip(used)
        .map(|(plane, &bytes)| bytes.min(plane.len()))
        .sum();
    let mut lease = pool.acquire(copied_len);

    for (plane, &bytes) in planes.iter().zip(used) {
        let bytes = bytes.min(plane.len());
        lease.extend_from_slice(&plane[..bytes]);
    }

    lease.freeze()
}

/// Empty or missing completed viewfinder data loses a preview before copying.
fn preview_data_available(total_bytes_used: usize, metrics: &PreviewMetrics) -> bool {
    if total_bytes_used == 0 {
        metrics.record_drops(1);
        false
    } else {
        true
    }
}

/// Measure only the real mmap-to-owned copy, excluding MJPEG decode and dispatch.
fn measured_copy_frame_planes(
    planes: &[&[u8]],
    used: &[usize],
    pool: &FrameDataPool,
    metrics: &PreviewMetrics,
) -> FrameData {
    let copy_start = Instant::now();
    let data = pooled_copy_frame_planes(planes, used, pool);
    metrics.record_copy(copy_start.elapsed());
    data
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
    mut req: libcamera::request::Request,
    stop_flag: &AtomicBool,
    zoom: &crate::backends::camera::zoom::ZoomControl,
    scaler_crop: &mut NativeScalerCrop,
) {
    // Request::reuse clears controls. Reapply even when resetting to 1x.
    scaler_crop.apply(req.controls_mut(), zoom);
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
    data: FrameData,
    layout: FrameLayout,
    captured_at: Instant,
    sensor_timestamp_ns: Option<u64>,
    metadata: FrameMetadata,
) -> CameraFrame {
    let yuv_planes = compute_yuv_planes(layout.format, layout.width, layout.height, layout.stride);

    CameraFrame {
        width: layout.width,
        height: layout.height,
        data,
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
                color: crate::backends::camera::types::YuvColor::Legacy,
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
                color: crate::backends::camera::types::YuvColor::Legacy,
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

    let data = if planes.len() == 1 {
        let bytes = used[0].min(planes[0].len());
        planes[0][..bytes].to_vec()
    } else {
        let mut combined = Vec::with_capacity(total_bytes);
        for (plane, &bytes) in planes.iter().zip(used.iter()) {
            let bytes = bytes.min(plane.len());
            combined.extend_from_slice(&plane[..bytes]);
        }
        combined
    };

    let stride = if config_stride > 0 {
        config_stride
    } else {
        compute_stride(pixel_format, size.width, data.len(), size.height)
    };

    let sensor_timestamp_ns = buf.metadata().map(|m| m.timestamp());
    Some(build_camera_frame(
        FrameData::from_owned_vec(data),
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

/// Clear only this session on every thread exit, including initialization errors
/// and unwinding. The shared tracker protects any newer session's registration.
struct PreviewSessionGuard(Arc<PreviewMetrics>);

impl Drop for PreviewSessionGuard {
    fn drop(&mut self) {
        self.0.stop();
    }
}

/// Main entry point for the capture thread
///
/// Creates all libcamera objects, runs the capture loop, and cleans up on exit.
/// Reports initialization success/failure via `init_tx`.
pub(crate) fn capture_thread_main(
    params: CaptureThreadParams,
    init_tx: std::sync::mpsc::SyncSender<Result<CaptureThreadInitResult, BackendError>>,
) {
    let _preview_session = PreviewSessionGuard(Arc::clone(&params.preview_metrics));
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
    let mut scaler_crop = NativeScalerCrop::discover(
        &cam,
        formats.vf_pixel_format.is_bayer(),
        formats.vf_size.width,
        formats.vf_size.height,
    );

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

    // Route camera requests and worker completions through one channel so a
    // decoded frame wakes the capture loop immediately instead of being polled.
    let (event_tx, event_rx) = std::sync::mpsc::channel();
    let request_event_tx = event_tx.clone();
    active_cam.on_request_completed(move |request| {
        let _ = request_event_tx.send(CaptureEvent::Request(request));
    });

    // Start camera
    active_cam
        .start(None)
        .map_err(|e| BackendError::InitializationFailed(format!("Camera start: {}", e)))?;
    crate::startup::milestone("camera_hardware_started");

    // Queue all requests
    for mut req in requests {
        scaler_crop.apply(req.controls_mut(), &params.zoom);
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

    // Create MJPEG workers before reporting success so decoder initialization
    // failures are propagated to the caller via init_tx.
    let jpeg_decode_queue = if formats.vf_is_mjpeg {
        if let Ok(mut d) = DIAGNOSTICS.write() {
            d.mjpeg_decoder_name = Some(format!(
                "turbojpeg (libjpeg-turbo, {MJPEG_DECODE_WORKERS} workers)"
            ));
        }
        Some(create_mjpeg_decode_queue(event_tx.clone())?)
    } else {
        None
    };

    // Reserve one RAW photo workspace in the background using negotiated, not
    // requested, dimensions. The guard retires it on every capture-thread exit.
    let _photo_session = if is_multistream {
        formats
            .raw_pixel_format
            .zip(normal_raw_layout(&formats.raw_format_name))
            .and_then(|(format, layout)| {
                crate::shaders::PhotoSession::start(
                    formats.raw_size.width,
                    formats.raw_size.height,
                    format,
                    layout,
                    formats.raw_stride,
                )
            })
    } else if formats.vf_pixel_format.is_bayer() {
        normal_raw_layout(&formats.vf_format_name).and_then(|layout| {
            crate::shaders::PhotoSession::start(
                formats.vf_size.width,
                formats.vf_size.height,
                formats.vf_pixel_format,
                layout,
                formats.vf_stride,
            )
        })
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
        &event_rx,
        &stream_vf,
        &stream_raw,
        &formats,
        is_multistream,
        jpeg_decode_queue,
        &mut scaler_crop,
        &mut params,
    );

    // Stop camera (ActiveCamera::drop also does this, but explicit is cleaner)
    info!("Capture loop ending, stopping camera");
    let _ = active_cam.stop();

    // Drop all libcamera objects in correct dependency order.
    // Previously, dropping mgr before cam/alloc caused use-after-free (SIGSEGV)
    // because those objects still referenced the CameraManager's internal state.
    drop(event_tx);
    drop(event_rx);
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
    super::super::stream_negotiation::negotiate(|dual| {
        let size = super::super::stream_negotiation::preview_size(
            params.preview_width,
            params.preview_height,
            dual && params.limit_preview,
        );
        let mut config = super::super::stream_negotiation::prepare(cam, dual, Some(size))?;
        active_cam
            .configure(&mut config)
            .map_err(|e| format!("Camera configure: {e}"))?;
        info!(
            second_stream = dual,
            config = config.to_string_repr(),
            "Configured stream layout"
        );
        Ok(config)
    })
    .map_err(BackendError::InitializationFailed)
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

/// Dispatch one completed MJPEG decode to preview and recording consumers.
fn dispatch_mjpeg_frame(
    output: MjpegDecodeOutput,
    params: &mut CaptureThreadParams,
    is_multistream: bool,
    last_dispatched_frame: &mut Option<u64>,
) {
    if !advance_mjpeg_dispatch_watermark(
        last_dispatched_frame,
        output.frame_num,
        output.frame.is_some(),
    ) {
        // A failed decode or stale output loses one preview, never both.
        params.preview_metrics.record_drops(1);
        return;
    }

    MJPEG_DECODE_TIME_US.store(output.decode_time_us, Ordering::Relaxed);
    if let Some(frame) = output.frame {
        dispatch_viewfinder_frame(
            frame,
            output.frame_num,
            params,
            is_multistream,
            output.dispatch_context,
        );
    }
}

fn drain_pending_events<T>(receiver: &std::sync::mpsc::Receiver<T>) {
    while receiver.try_recv().is_ok() {}
}

/// Run the capture loop until `stop_flag` is set or the event channel closes.
#[allow(clippy::too_many_arguments)]
fn run_capture_loop(
    active_cam: &mut libcamera::camera::ActiveCamera<'_>,
    event_rx: &std::sync::mpsc::Receiver<CaptureEvent>,
    stream_vf: &libcamera::stream::Stream,
    stream_raw: &Option<libcamera::stream::Stream>,
    formats: &StreamFormats,
    is_multistream: bool,
    jpeg_decode_queue: Option<MjpegDecodeQueue>,
    scaler_crop: &mut NativeScalerCrop,
    params: &mut CaptureThreadParams,
) {
    use libcamera::framebuffer::AsFrameBuffer;
    use libcamera::framebuffer_map::MemoryMappedFrameBuffer;
    use libcamera::request::{RequestStatus, ReuseFlag};

    info!("Entering capture loop");
    let mut last_dispatched_mjpeg_frame = None;
    let capture_copy_pool = FrameDataPool::new(CAPTURE_COPY_POOL_CACHE_CAPACITY);

    type MmapFB = MemoryMappedFrameBuffer<libcamera::framebuffer_allocator::FrameBuffer>;

    while !params.stop_flag.load(Ordering::Acquire) {
        // Worker completions share this channel with libcamera requests, so
        // completed decodes wake the loop immediately. The timeout exists only
        // to observe the atomic stop flag if camera delivery itself stalls.
        let event = match event_rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(event) => event,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                info!("Capture event channel disconnected, stopping capture loop");
                break;
            }
        };
        let mut req = match event {
            CaptureEvent::MjpegDecoded(output) => {
                dispatch_mjpeg_frame(
                    output,
                    params,
                    is_multistream,
                    &mut last_dispatched_mjpeg_frame,
                );
                continue;
            }
            CaptureEvent::Request(req) => req,
        };

        if req.status() != RequestStatus::Complete {
            debug!(status = ?req.status(), "Skipping non-complete request");
            req.reuse(ReuseFlag::REUSE_BUFFERS);
            requeue_request(
                active_cam,
                req,
                &params.stop_flag,
                &params.zoom,
                scaler_crop,
            );
            continue;
        }

        let captured_at = Instant::now();
        let frame_num = params.preview_frame_count.fetch_add(1, Ordering::Relaxed);

        // Extract per-frame metadata
        let mut metadata = extract_metadata(&req);
        metadata.scaler_zoom = scaler_crop.completed(req.metadata());
        metadata.raw_layout = normal_raw_layout(&formats.vf_format_name);

        // Process viewfinder buffer
        if !scaler_crop.may_dispatch_processed(metadata.scaler_zoom) {
            // Do not software-crop pixels whose ISP crop cannot be verified.
            // RAW still handling below remains independent and can still run.
            params.preview_metrics.record_drops(1);
        } else if let Some(vf_buf) = req.buffer::<MmapFB>(stream_vf) {
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
                if !preview_data_available(total_bytes_used, &params.preview_metrics) {
                    req.reuse(ReuseFlag::REUSE_BUFFERS);
                    requeue_request(
                        active_cam,
                        req,
                        &params.stop_flag,
                        &params.zoom,
                        scaler_crop,
                    );
                    continue;
                }

                // Bind the still generation before copying or scheduling decode;
                // delayed viewfinder work must not claim a later shutter request.
                let dispatch_context = snapshot_frame_dispatch_context(
                    is_multistream,
                    &params.still_capture,
                    &params.recording_sender,
                );
                // Copy mmap data once into recyclable owned storage before the
                // request is requeued. Multi-plane formats are concatenated in
                // plane order for the existing FrameData/YuvPlanes contract.
                let captured_data = measured_copy_frame_planes(
                    &planes,
                    &used,
                    &capture_copy_pool,
                    &params.preview_metrics,
                );
                if let Some(queue) = jpeg_decode_queue.as_ref() {
                    let jpeg_data = captured_data;
                    if queue
                        .try_submit_preview(
                            MjpegDecodeJob {
                                data: jpeg_data,
                                captured_at,
                                sensor_timestamp_ns,
                                metadata: metadata.clone(),
                                frame_num,
                                dispatch_context,
                            },
                            &params.preview_metrics,
                        )
                        .is_err()
                        && frame_num.is_multiple_of(LOG_EVERY_N_FRAMES)
                    {
                        debug!(frame = frame_num, "MJPEG decode queue full, dropping frame");
                    }
                } else {
                    let data = captured_data;

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

                    let frame = build_camera_frame(
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
                    );
                    dispatch_viewfinder_frame(
                        frame,
                        frame_num,
                        params,
                        is_multistream,
                        dispatch_context,
                    );
                }
            } else {
                preview_data_available(0, &params.preview_metrics);
            }
        } else {
            preview_data_available(0, &params.preview_metrics);
            if frame_num.is_multiple_of(LOG_EVERY_N_FRAMES) {
                warn!("No viewfinder buffer in completed request");
            }
        }

        // Process raw buffer only when still capture is requested (multistream only).
        if is_multistream
            && let Some(ticket) = params.still_capture.pending()
            && let Some(sr) = stream_raw
        {
            let raw_pf = formats.raw_pixel_format.unwrap_or(PixelFormat::BayerRGGB);
            let raw_metadata =
                raw_frame_metadata(&metadata, normal_raw_layout(&formats.raw_format_name));
            if let Some(still_frame) = process_buffer(
                &req,
                sr,
                formats.raw_size,
                formats.raw_stride,
                raw_pf,
                captured_at,
                &raw_metadata,
            ) {
                let still_count = params.still_frame_count.fetch_add(1, Ordering::Relaxed);
                info!(
                    frame = still_count,
                    width = still_frame.width,
                    height = still_frame.height,
                    format = ?still_frame.format,
                    "Raw still frame captured"
                );

                params.still_capture.publish(ticket, still_frame);
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
        requeue_request(
            active_cam,
            req,
            &params.stop_flag,
            &params.zoom,
            scaler_crop,
        );
    }

    // Stop and join decode workers, then release completed outputs that retain
    // capture-time recording sender clones.
    drop(jpeg_decode_queue);
    drain_pending_events(event_rx);
}

/// Dispatch a completed viewfinder frame to all consumers:
/// 1. Handle ticketed single-stream still capture
/// 2. Send to recording (VF→encoder via appsrc)
/// 3. Send to UI preview channel
fn dispatch_viewfinder_frame(
    frame: CameraFrame,
    frame_num: u64,
    params: &mut CaptureThreadParams,
    is_multistream: bool,
    dispatch_context: FrameDispatchContext<RecordingFrame>,
) {
    static FIRST_FRAME: AtomicBool = AtomicBool::new(false);
    crate::startup::milestone_once("first_frame_captured", &FIRST_FRAME);

    // In single-stream mode, also handle still capture from preview.
    if !is_multistream && let Some(ticket) = dispatch_context.capture_still {
        info!(
            frame = frame_num,
            width = frame.width,
            height = frame.height,
            "Still frame captured from preview stream"
        );
        params.still_capture.publish(ticket, frame.clone());
    }

    // Send ViewFinder frames to recording (via appsrc encoder pipeline).
    if let Some(ref tx) = dispatch_context.recording_sender {
        let seq = frame.libcamera_metadata.as_ref().and_then(|m| m.sequence);
        let send_result = tx.try_send(Arc::new(frame.clone()));
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

    // Record actual produced output once, even when UI delivery is backpressured.
    params.preview_metrics.record_frame(&frame);
    // Send to UI (preview)
    if let Err(e) = params.frame_sender.try_send(frame) {
        params.preview_metrics.record_drops(1);
        if frame_num.is_multiple_of(LOG_EVERY_N_FRAMES) {
            debug!(error = ?e, "Preview frame dropped (channel full)");
        }
    }
}

// Negotiated libcamera name, not stride or native sensor calibration.
fn normal_raw_layout(name: &str) -> Option<crate::backends::camera::types::RawLayout> {
    use crate::backends::camera::types::RawLayout;
    let suffix = ["SRGGB", "SBGGR", "SGRBG", "SGBRG"]
        .iter()
        .find_map(|p| name.strip_prefix(p))?;
    if suffix == "8" {
        return Some(RawLayout::U8);
    }
    let packed = suffix.strip_suffix("_CSI2P");
    let bits = packed.unwrap_or(suffix).parse::<u32>().ok()?;
    if packed.is_some() {
        [10, 12, 14]
            .contains(&bits)
            .then_some(RawLayout::Csi2 { bits })
    } else {
        [10, 12, 14, 16]
            .contains(&bits)
            .then_some(RawLayout::Unpacked16 { bits })
    }
}

// Same reviewed conversion, named so producer units/order can be tested directly.
fn normal_sensor_black_levels(levels: libcamera::controls::SensorBlackLevels) -> [f32; 4] {
    levels.0.map(|n| n as f32 / 65535.0)
}

/// RAW pixels never inherit the processed stream's ISP crop evidence.
fn raw_frame_metadata(metadata: &FrameMetadata, layout: Option<RawLayout>) -> FrameMetadata {
    let mut raw = metadata.clone();
    raw.raw_layout = layout;
    raw.scaler_zoom = None;
    raw
}

/// Extract per-frame metadata from a completed request.
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
        scaler_zoom: None,
        raw_layout: None,
        // The standard libcamera 0.7 API provides no documented WhiteLevel
        // control. Do not guess vendor IDs or substitute a calibration value.
        sensor_white_levels: None,
        sensor_black_levels: meta
            .get::<controls::SensorBlackLevels>()
            .ok()
            .map(normal_sensor_black_levels),
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
/// Decoded storage comes from a refcount-aware pool and is returned only after
/// every consumer has dropped the frame.
/// Returns `None` if decoding fails (caller should skip the frame).
fn decode_mjpeg_frame(
    decompressor: &mut turbojpeg::Decompressor,
    jpeg_data: &[u8],
    captured_at: Instant,
    sensor_timestamp_ns: Option<u64>,
    metadata: &FrameMetadata,
    frame_num: u64,
    yuv_pool: &FrameDataPool,
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

    let mut yuv_buf = yuv_pool.acquire(buf_len);
    yuv_buf.resize(buf_len, 0);

    let yuv_image = turbojpeg::YuvImage {
        pixels: yuv_buf.as_mut_slice(),
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

    // Decoder geometry owns both pitches and sampling-group padding.
    let geometry = turbojpeg::YuvImage {
        pixels: (),
        width,
        height,
        align,
        subsamp: header.subsamp,
    };
    // Gray has only Y backing; TurboJPEG's UV helpers return hypothetical
    // geometry even for Gray. Empty spans retain full-range luma metadata,
    // while consumers bind their existing bounded 1x1 dummy UV textures.
    let (uv_w, uv_h) = if header.subsamp == turbojpeg::Subsamp::Gray {
        (0, 0)
    } else {
        (geometry.uv_width(), geometry.uv_height())
    };

    // Determine standard YUV format name from subsampling
    let format_name = match header.subsamp {
        turbojpeg::Subsamp::Sub2x2 => "I420",
        turbojpeg::Subsamp::Sub2x1 => "I422",
        turbojpeg::Subsamp::None => "I444",
        turbojpeg::Subsamp::Sub1x2 => "I440",
        turbojpeg::Subsamp::Sub4x1 => "I411",
        turbojpeg::Subsamp::Sub1x4 => "I441",
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

    // TurboJPEG pads the Y plane to complete subsampling groups, even at align=1.
    let y_stride = geometry.y_width();
    let y_size = y_stride * geometry.y_height();
    let uv_size = uv_w * uv_h;

    let yuv_planes = YuvPlanes {
        color: crate::backends::camera::types::YuvColor::Full601,
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

    let data = yuv_buf.freeze();

    Some(CameraFrame {
        width: width as u32,
        height: height as u32,
        data,
        format: match header.subsamp {
            turbojpeg::Subsamp::Gray => PixelFormat::Gray8,
            turbojpeg::Subsamp::Sub2x2 => PixelFormat::I420,
            // No PixelFormat::I422 variant yet; treat 4:2:2 as I420 for now
            // (plane layout is still correct via yuv_planes, only the enum tag is imprecise)
            _ => PixelFormat::I420,
        },
        stride: y_stride as u32,
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

#[cfg(test)]
mod tests {
    #[test]
    fn raw_frame_metadata_never_inherits_processed_scaler_crop() {
        use crate::backends::camera::zoom::{ScalerCropBounds, ZoomRect};
        let base = ZoomRect {
            x: 0,
            y: 0,
            width: 4000,
            height: 3000,
        };
        let bounds = ScalerCropBounds::new(
            ZoomRect {
                width: 100,
                height: 100,
                ..base
            },
            base,
            base,
            base,
            false,
        )
        .unwrap();
        let original = super::FrameMetadata {
            scaler_zoom: bounds.applied(Some(ZoomRect {
                x: 1000,
                y: 750,
                width: 2000,
                height: 1500,
            })),
            exposure_time: Some(1000),
            ..Default::default()
        };
        let raw =
            super::raw_frame_metadata(&original, Some(super::RawLayout::Unpacked16 { bits: 10 }));
        assert!(raw.scaler_zoom.is_none());
        assert_eq!(raw.exposure_time, Some(1000));
        assert_eq!(
            raw.raw_layout,
            Some(super::RawLayout::Unpacked16 { bits: 10 })
        );
        assert!(original.scaler_zoom.is_some());
    }

    use super::{
        BoundedWorkerQueue, QueueWorker, advance_mjpeg_dispatch_watermark, drain_pending_events,
        pooled_copy_frame_planes, snapshot_frame_dispatch_context,
    };
    use crate::backends::camera::still_capture::StillCapture;
    use crate::backends::camera::types::FrameDataPool;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::{Duration, Instant};

    #[test]
    fn review_r2_real_mjpeg_411_441_all_consumers() {
        let Some((_guard, device, queue)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu("review_r2_real_mjpeg_411_441_all_consumers");
            return;
        };
        pollster::block_on(async {
            for (subsamp, w, h) in [
                (turbojpeg::Subsamp::Sub4x1, 9, 4),
                (turbojpeg::Subsamp::Sub1x4, 9, 4),
                (turbojpeg::Subsamp::Sub1x4, 9, 5),
            ] {
                let rgb = [210u8, 50, 80];
                let pixels = rgb.repeat(w * h);
                let mut compressor = turbojpeg::Compressor::new().unwrap();
                compressor.set_quality(100).unwrap();
                compressor.set_subsamp(subsamp).unwrap();
                let jpeg = compressor
                    .compress_to_vec(turbojpeg::Image {
                        pixels: &pixels[..],
                        width: w,
                        height: h,
                        pitch: w * 3,
                        format: turbojpeg::PixelFormat::RGB,
                    })
                    .unwrap();
                let frame = super::decode_mjpeg_frame(
                    &mut turbojpeg::Decompressor::new().unwrap(),
                    &jpeg,
                    Instant::now(),
                    None,
                    &Default::default(),
                    1,
                    &FrameDataPool::new(1),
                )
                .unwrap();
                let p = frame.yuv_planes.unwrap();
                let expected = if subsamp == turbojpeg::Subsamp::Sub4x1 {
                    (3, 4)
                } else {
                    (9, (h as u32).div_ceil(4))
                };
                assert_eq!((p.uv_width, p.uv_height), expected, "{subsamp:?}");
                let preview = crate::app::review_decoded_preview(&device, &queue, &frame);
                let recording = crate::pipelines::video::recorder::convert_frame_to_rgba(&frame)
                    .await
                    .unwrap();
                let photo =
                    crate::pipelines::photo::processing::PostProcessor::new(Default::default())
                        .process(Arc::new(frame))
                        .await
                        .unwrap();
                for out in [&preview, &recording] {
                    for pixel in out.as_chunks::<4>().0 {
                        for c in 0..3 {
                            assert!(pixel[c].abs_diff(rgb[c]) <= 2, "{subsamp:?}: {pixel:?}");
                        }
                    }
                }
                for pixel in photo.image.pixels() {
                    for c in 0..3 {
                        assert!(pixel[c].abs_diff(rgb[c]) <= 2, "{subsamp:?}: {pixel:?}");
                    }
                }
            }
        });
    }

    fn review_n1_gray_frame(
        w: usize,
        h: usize,
        luma: u8,
    ) -> crate::backends::camera::types::CameraFrame {
        let pixels = [luma; 3].repeat(w * h);
        let mut compressor = turbojpeg::Compressor::new().unwrap();
        compressor.set_quality(100).unwrap();
        compressor.set_subsamp(turbojpeg::Subsamp::Gray).unwrap();
        let jpeg = compressor
            .compress_to_vec(turbojpeg::Image {
                pixels: &pixels[..],
                width: w,
                height: h,
                pitch: w * 3,
                format: turbojpeg::PixelFormat::RGB,
            })
            .unwrap();
        super::decode_mjpeg_frame(
            &mut turbojpeg::Decompressor::new().unwrap(),
            &jpeg,
            Instant::now(),
            None,
            &Default::default(),
            1,
            &FrameDataPool::new(1),
        )
        .unwrap()
    }

    #[test]
    fn review_n1_real_gray_jpeg_declares_only_backed_luminance() {
        for (w, h) in [(9, 4), (7, 13)] {
            let frame = review_n1_gray_frame(w, h, 83);
            assert_eq!(
                frame.format,
                crate::backends::camera::types::PixelFormat::Gray8
            );
            let p = frame.yuv_planes.unwrap();
            assert_eq!(p.color, crate::backends::camera::types::YuvColor::Full601);
            assert_eq!((p.uv_width, p.uv_height, p.uv_size, p.v_size), (0, 0, 0, 0));
            assert_eq!((p.uv_stride, p.v_stride), (0, 0));
            assert_eq!(p.y_offset + p.y_size, frame.data.as_ref().len());
            assert_eq!(p.uv_offset + p.uv_size, frame.data.as_ref().len());
            assert_eq!(p.v_offset + p.v_size, frame.data.as_ref().len());
            assert_eq!(frame.data.as_ref(), vec![83; w * h]);
            let input = crate::shaders::GpuFrameInput::from_camera_frame(&frame).unwrap();
            assert!(input.uv_data.is_none() && input.v_data.is_none());
        }
    }

    #[test]
    fn review_n1_real_gray_jpeg_bounds_still_dummy_chroma() {
        let Some((_guard, device, queue)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu("review_n1_real_gray_jpeg_bounds_still_dummy_chroma");
            return;
        };
        pollster::block_on(async {
            let mut converter = crate::shaders::GpuConvertPipeline::new().await.unwrap();
            for (w, h, luma) in [(9, 4, 83), (7, 13, 83), (9, 4, 128)] {
                let frame = review_n1_gray_frame(w, h, luma);
                assert_eq!(
                    frame.format,
                    crate::backends::camera::types::PixelFormat::Gray8
                );
                converter
                    .convert(&crate::shaders::GpuFrameInput::from_camera_frame(&frame).unwrap())
                    .unwrap();
                assert_eq!(converter.review_chroma_texture_sizes(), [(1, 1); 2]);
                let converted = converter
                    .read_rgba_to_cpu(w as u32, h as u32)
                    .await
                    .unwrap();
                let preview = crate::app::review_decoded_preview(&device, &queue, &frame);
                let recording = crate::pipelines::video::recorder::convert_frame_to_rgba(&frame)
                    .await
                    .unwrap();
                let photo =
                    crate::pipelines::photo::processing::PostProcessor::new(Default::default())
                        .process(Arc::new(frame))
                        .await
                        .unwrap();
                for out in [converted, preview, recording] {
                    assert_eq!(out, [luma, luma, luma, 255].repeat(w * h));
                }
                assert_eq!(photo.image.as_raw(), &[luma; 3].repeat(w * h));
            }
        });
    }

    #[test]
    fn review_n1_real_gray_jpeg_bounds_preview_dummy_chroma() {
        let Some((_guard, device, queue)) = crate::test_gpu::headless_device() else {
            crate::test_gpu::skip_no_gpu("review_n1_real_gray_jpeg_bounds_preview_dummy_chroma");
            return;
        };
        for (w, h) in [(9, 4), (7, 13)] {
            let frame = review_n1_gray_frame(w, h, 83);
            assert_eq!(
                crate::app::review_decoded_preview(&device, &queue, &frame),
                [83, 83, 83, 255].repeat(w * h)
            );
        }
    }

    #[test]
    fn normal_mjpeg_photo_uses_decoder_full_range() {
        let Some((_guard, _device, _queue)) = crate::test_gpu::headless_device_with_compute()
        else {
            crate::test_gpu::skip_no_gpu("normal_mjpeg_photo_uses_decoder_full_range");
            return;
        };
        pollster::block_on(async {
            let mut converter = crate::shaders::GpuConvertPipeline::new().await.unwrap();
            for subsamp in [
                turbojpeg::Subsamp::None,
                turbojpeg::Subsamp::Sub2x1,
                turbojpeg::Subsamp::Sub2x2,
            ] {
                for rgb in [[32, 32, 32], [128, 128, 128], [210, 50, 80]] {
                    let pixels = rgb.repeat(9);
                    let mut compressor = turbojpeg::Compressor::new().unwrap();
                    compressor.set_quality(100).unwrap();
                    compressor.set_subsamp(subsamp).unwrap();
                    let jpeg = compressor
                        .compress_to_vec(turbojpeg::Image {
                            pixels: &pixels[..],
                            width: 3,
                            height: 3,
                            pitch: 9,
                            format: turbojpeg::PixelFormat::RGB,
                        })
                        .unwrap();
                    let mut decoder = turbojpeg::Decompressor::new().unwrap();
                    let frame = super::decode_mjpeg_frame(
                        &mut decoder,
                        &jpeg,
                        Instant::now(),
                        None,
                        &crate::backends::camera::types::FrameMetadata::default(),
                        1,
                        &FrameDataPool::new(1),
                    )
                    .unwrap();
                    converter
                        .convert(&crate::shaders::GpuFrameInput::from_camera_frame(&frame).unwrap())
                        .unwrap();
                    let out = converter.read_rgba_to_cpu(3, 3).await.unwrap();
                    for pixel in out.as_chunks::<4>().0 {
                        for c in 0..3 {
                            assert!(
                                pixel[c].abs_diff(rgb[c]) <= 2,
                                "{subsamp:?}: {rgb:?} -> {pixel:?}"
                            );
                        }
                    }
                }
            }
        });
    }

    #[test]
    fn normal_negotiated_raw_layout_is_not_inferred_from_stride() {
        use super::normal_raw_layout;
        use crate::backends::camera::types::RawLayout;
        assert_eq!(
            normal_raw_layout("SRGGB10_CSI2P"),
            Some(RawLayout::Csi2 { bits: 10 })
        );
        assert_eq!(
            normal_raw_layout("SBGGR12"),
            Some(RawLayout::Unpacked16 { bits: 12 })
        );
        assert_eq!(normal_raw_layout("SGRBG8"), Some(RawLayout::U8));
        assert_eq!(normal_raw_layout("SRGGB10_IPU3"), None);
        assert_eq!(normal_raw_layout("NV12"), None);
        for cfa in ["SRGGB", "SBGGR", "SGRBG", "SGBRG"] {
            for bits in [10, 12, 14, 16] {
                assert_eq!(
                    normal_raw_layout(&format!("{cfa}{bits}")),
                    Some(RawLayout::Unpacked16 { bits })
                );
            }
            for bits in [10, 12, 14] {
                assert_eq!(
                    normal_raw_layout(&format!("{cfa}{bits}_CSI2P")),
                    Some(RawLayout::Csi2 { bits })
                );
            }
        }
        let levels = super::normal_sensor_black_levels(libcamera::controls::SensorBlackLevels([
            0, 16384, 32768, 65535,
        ]));
        assert_eq!(levels, [0.0, 16384.0 / 65535.0, 32768.0 / 65535.0, 1.0]);
        assert!(
            super::FrameMetadata::default()
                .sensor_black_levels
                .is_none()
        );
        assert!(super::FrameMetadata::default().raw_layout.is_none());
    }

    fn dispatch_test_params(
        frame_sender: super::FrameSender,
        preview_metrics: Arc<super::PreviewMetrics>,
    ) -> super::CaptureThreadParams {
        super::CaptureThreadParams {
            zoom: Default::default(),
            camera_id: "native-dispatch-metrics-test".into(),
            preview_metrics,
            preview_width: 2,
            preview_height: 1,
            limit_preview: false,
            stop_flag: Arc::new(AtomicBool::new(false)),
            still_capture: Default::default(),
            preview_frame_count: Arc::new(super::AtomicU64::new(0)),
            still_frame_count: Arc::new(super::AtomicU64::new(0)),
            frame_sender,
            recording_sender: Arc::new(Mutex::new(None)),
            cancel_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    fn dispatch_test_frame() -> super::CameraFrame {
        super::build_camera_frame(
            super::FrameData::from_owned_vec(vec![1; 8]),
            super::FrameLayout {
                width: 2,
                height: 1,
                format: super::PixelFormat::ABGR,
                stride: 8,
            },
            Instant::now(),
            None,
            super::FrameMetadata::default(),
        )
    }

    #[test]
    fn native_dispatch_delivers_owned_still_recording_and_preview_without_cache() {
        let metrics = super::PreviewMetrics::start("native-dispatch-consumers-test");
        let (sender, mut preview) = cosmic::iced::futures::channel::mpsc::channel(1);
        let mut params = dispatch_test_params(sender, metrics);
        let (tx, mut recording) = tokio::sync::mpsc::channel(1);
        *params.recording_sender.lock().unwrap() = Some(tx);
        let request = params.still_capture.request();
        let frame = dispatch_test_frame();
        let identity = frame.data.as_ptr();
        let captured_at = frame.captured_at;
        let context =
            snapshot_frame_dispatch_context(false, &params.still_capture, &params.recording_sender);
        super::dispatch_viewfinder_frame(frame, 1, &mut params, false, context);
        let still = request.take().expect("ticketed still");
        let preview = preview.try_recv().expect("preview channel");
        let recording = recording.try_recv().unwrap();
        drop(params);
        for frame in [&still, &preview, recording.as_ref()] {
            assert_eq!(frame.data.as_ref(), &[1; 8]);
            assert_eq!(frame.data.as_ptr(), identity);
            assert_eq!(frame.captured_at, captured_at);
        }
    }

    #[test]
    fn mjpeg_decode_dispatch_keeps_capture_time_recording_sender() {
        let metrics = super::PreviewMetrics::start("mjpeg-decoded-recording-test");
        let (sender, mut preview) = cosmic::iced::futures::channel::mpsc::channel(1);
        let mut params = dispatch_test_params(sender, metrics);
        let (old_tx, mut old_rx) = tokio::sync::mpsc::channel(1);
        let (new_tx, mut new_rx) = tokio::sync::mpsc::channel(1);
        *params.recording_sender.lock().unwrap() = Some(old_tx);
        let request = params.still_capture.request();
        let context =
            snapshot_frame_dispatch_context(false, &params.still_capture, &params.recording_sender);
        *params.recording_sender.lock().unwrap() = Some(new_tx);
        // Real JPEG decode feeds the same completed-event path as the worker.
        let frame = review_n1_gray_frame(9, 4, 83);
        let expected = frame.clone();
        let mut watermark = None;
        super::dispatch_mjpeg_frame(
            super::MjpegDecodeOutput {
                frame_num: 1,
                frame: Some(frame),
                dispatch_context: context,
                decode_time_us: 1,
            },
            &mut params,
            false,
            &mut watermark,
        );
        let recorded = old_rx
            .try_recv()
            .expect("recording destination captured before decode");
        assert_eq!(recorded.format, super::PixelFormat::Gray8);
        assert_eq!(recorded.data.as_ref(), expected.data.as_ref());
        assert_eq!(recorded.data.as_ptr(), expected.data.as_ptr());
        assert_eq!(
            request.take().unwrap().data.as_ref(),
            expected.data.as_ref()
        );
        assert_eq!(
            preview.try_recv().unwrap().data.as_ref(),
            expected.data.as_ref()
        );
        assert_eq!(watermark, Some(1));
        assert!(matches!(
            old_rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
        ));
        assert!(matches!(
            new_rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
    }

    #[test]
    fn delayed_viewfinder_publication_cannot_satisfy_new_still_request() {
        let metrics = super::PreviewMetrics::start("delayed-still-ticket-test");
        let (sender, _receiver) = cosmic::iced::futures::channel::mpsc::channel(4);
        let mut params = dispatch_test_params(sender, metrics);
        let a = params.still_capture.request();
        let context_a =
            snapshot_frame_dispatch_context(false, &params.still_capture, &params.recording_sender);
        drop(a);
        let b = params.still_capture.request();
        let ticket_b = params.still_capture.pending().unwrap();
        super::dispatch_viewfinder_frame(dispatch_test_frame(), 1, &mut params, false, context_a);
        assert!(b.take().is_none());
        assert_eq!(params.still_capture.pending(), Some(ticket_b));
        let context_b =
            snapshot_frame_dispatch_context(false, &params.still_capture, &params.recording_sender);
        super::dispatch_viewfinder_frame(dispatch_test_frame(), 2, &mut params, false, context_b);
        assert!(b.take().is_some());
    }

    #[test]
    fn native_dispatch_counts_every_full_channel_drop_and_produced_frame() {
        let metrics = crate::backends::camera::preview_metrics::PreviewMetrics::start(
            "native-dispatch-metrics-test",
        );
        let (mut sender, _receiver) = cosmic::iced::futures::channel::mpsc::channel(1);
        while sender.try_send(dispatch_test_frame()).is_ok() {}
        let mut params = dispatch_test_params(sender, Arc::clone(&metrics));
        for frame_num in [1, 2] {
            super::dispatch_viewfinder_frame(
                dispatch_test_frame(),
                frame_num,
                &mut params,
                false,
                super::FrameDispatchContext {
                    capture_still: None,
                    recording_sender: None,
                },
            );
        }
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.dropped_frames, 2);
        assert_eq!(snapshot.frame_count, 2);
        assert_eq!(
            (snapshot.width, snapshot.height, snapshot.frame_size_bytes),
            (2, 1, 8)
        );
        assert_eq!(snapshot.output_format, Some(super::PixelFormat::ABGR));
    }

    #[test]
    fn native_mjpeg_rejected_outputs_count_once_without_advancing_watermark() {
        let metrics = super::PreviewMetrics::start("native-mjpeg-rejected-test");
        let (sender, _receiver) = cosmic::iced::futures::channel::mpsc::channel(4);
        let mut params = dispatch_test_params(sender, Arc::clone(&metrics));
        let mut watermark = Some(2);
        for (frame_num, frame) in [(3, None), (1, Some(dispatch_test_frame())), (0, None)] {
            super::dispatch_mjpeg_frame(
                super::MjpegDecodeOutput {
                    frame_num,
                    frame,
                    dispatch_context: super::FrameDispatchContext {
                        capture_still: None,
                        recording_sender: None,
                    },
                    decode_time_us: 100,
                },
                &mut params,
                false,
                &mut watermark,
            );
        }
        assert_eq!(metrics.snapshot().dropped_frames, 3);
        assert_eq!(metrics.snapshot().frame_count, 0);
        assert_eq!(watermark, Some(2));
    }

    #[test]
    fn native_decode_queue_rejections_count_even_between_log_frames() {
        let metrics = super::PreviewMetrics::start("native-queue-metrics-test");
        let (job_tx, _job_rx) = std::sync::mpsc::sync_channel(1);
        let queue = BoundedWorkerQueue {
            job_tx: Some(job_tx),
            workers: Vec::new(),
        };
        queue.try_submit_preview(1_u32, &metrics).unwrap();
        assert!(queue.try_submit_preview(2_u32, &metrics).is_err());
        assert!(queue.try_submit_preview(3_u32, &metrics).is_err());
        assert_eq!(metrics.snapshot().dropped_frames, 2);
        assert_eq!(metrics.snapshot().frame_count, 0);
    }

    #[test]
    fn native_cancelled_capture_stops_its_preview_session_without_hardware() {
        let metrics = super::PreviewMetrics::start("native-cancelled-session-test");
        let (sender, _receiver) = cosmic::iced::futures::channel::mpsc::channel(1);
        let params = dispatch_test_params(sender, Arc::clone(&metrics));
        params.cancel_flag.store(true, Ordering::Release);
        let (init_tx, _init_rx) = std::sync::mpsc::sync_channel(1);
        super::capture_thread_main(params, init_tx);
        assert!(super::PreviewMetrics::active_for("native-cancelled-session-test").is_none());
    }

    #[test]
    fn native_empty_preview_buffers_count_one_loss_each() {
        let metrics = super::PreviewMetrics::start("native-empty-buffer-test");
        for total_bytes in [0, 8, 0] {
            let has_data = super::preview_data_available(total_bytes, &metrics);
            assert_eq!(has_data, total_bytes > 0);
        }
        assert_eq!(metrics.snapshot().dropped_frames, 2);
        assert_eq!(metrics.snapshot().copy_time_us, None);
    }

    struct BlockingWorker {
        started: Arc<AtomicUsize>,
        release: Arc<(Mutex<bool>, Condvar)>,
    }

    impl QueueWorker<u32, u32> for BlockingWorker {
        fn process(&mut self, job: u32) -> u32 {
            self.started.fetch_add(1, Ordering::Release);
            let (lock, condvar) = &*self.release;
            let mut released = lock.lock().unwrap();
            while !*released {
                released = condvar.wait(released).unwrap();
            }
            job
        }
    }

    struct IdentityWorker;

    impl QueueWorker<u32, u32> for IdentityWorker {
        fn process(&mut self, job: u32) -> u32 {
            job
        }
    }

    #[test]
    fn native_capture_copy_records_real_copy_without_counting_a_frame() {
        let metrics = super::PreviewMetrics::start("native-copy-metrics-test");
        let pool = FrameDataPool::new(1);
        assert_eq!(metrics.snapshot().copy_time_us, None);
        let copied =
            super::measured_copy_frame_planes(&[&[1, 2, 99], &[3, 4]], &[2, 2], &pool, &metrics);
        assert_eq!(copied.as_ref(), &[1, 2, 3, 4]);
        assert!(metrics.snapshot().copy_time_us.is_some());
        assert_eq!(metrics.snapshot().frame_count, 0);
        assert_eq!(metrics.snapshot().dropped_frames, 0);
    }

    #[test]
    fn pooled_capture_copy_reuses_storage_for_clamped_multi_plane_bytes() {
        let pool = FrameDataPool::new(1);
        let first = pooled_copy_frame_planes(&[&[1, 2, 3, 99], &[4, 5, 88, 77]], &[3, 2], &pool);
        let original_ptr = first.as_ptr();

        assert_eq!(first.as_ref(), &[1, 2, 3, 4, 5]);
        drop(first);

        let reused = pooled_copy_frame_planes(&[&[6, 7, 8, 66], &[9, 55]], &[3, 8], &pool);

        assert_eq!(reused.as_ref(), &[6, 7, 8, 9, 55]);
        assert_eq!(reused.as_ptr(), original_ptr);
    }

    #[test]
    fn pooled_capture_copy_never_overwrites_a_retained_frame() {
        let pool = FrameDataPool::new(1);
        let first = pooled_copy_frame_planes(&[&[1, 2, 3, 4]], &[4], &pool);
        let retained = first.clone();

        let second = pooled_copy_frame_planes(&[&[9, 8, 7, 6]], &[4], &pool);

        assert_eq!(retained.as_ref(), &[1, 2, 3, 4]);
        assert_eq!(second.as_ref(), &[9, 8, 7, 6]);
        assert_ne!(retained.as_ptr(), second.as_ptr());
    }

    #[test]
    #[should_panic(expected = "one bytes-used entry per capture plane")]
    fn pooled_capture_copy_rejects_mismatched_plane_metadata() {
        let pool = FrameDataPool::new(1);

        let _ = pooled_copy_frame_planes(&[&[1, 2], &[3, 4]], &[2], &pool);
    }

    #[test]
    fn bounded_worker_queue_notifies_an_external_result_channel() {
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let queue = BoundedWorkerQueue::new_with_result_sender(vec![IdentityWorker], 1, result_tx);

        queue.try_submit(42).unwrap();

        assert_eq!(result_rx.recv_timeout(Duration::from_secs(1)), Ok(42));
    }

    #[test]
    fn draining_pending_events_releases_held_senders() {
        let (sink_tx, sink_rx) = tokio::sync::mpsc::channel::<u8>(1);
        let (event_tx, event_rx) = std::sync::mpsc::channel();
        event_tx.send(sink_tx.clone()).unwrap();
        drop(event_tx);
        drop(sink_tx);
        assert!(!sink_rx.is_closed());

        drain_pending_events(&event_rx);

        assert!(sink_rx.is_closed());
    }

    #[test]
    fn bounded_worker_queue_runs_two_workers_and_rejects_excess_jobs() {
        let started = Arc::new(AtomicUsize::new(0));
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let workers = (0..2)
            .map(|_| BlockingWorker {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
            })
            .collect();
        let (result_tx, _result_rx) = std::sync::mpsc::channel();
        let queue = BoundedWorkerQueue::new_with_result_sender(workers, 1, result_tx);

        queue.try_submit(1).unwrap();
        let first_deadline = Instant::now() + Duration::from_secs(1);
        while started.load(Ordering::Acquire) != 1 && Instant::now() < first_deadline {
            std::thread::yield_now();
        }
        queue.try_submit(2).unwrap();
        let second_deadline = Instant::now() + Duration::from_secs(1);
        while started.load(Ordering::Acquire) != 2 && Instant::now() < second_deadline {
            std::thread::yield_now();
        }

        queue.try_submit(3).unwrap();
        let excess_was_rejected = queue.try_submit(4).is_err();

        assert_eq!(started.load(Ordering::Acquire), 2);
        assert!(excess_was_rejected);

        let (lock, condvar) = &*release;
        *lock.lock().unwrap() = true;
        condvar.notify_all();
    }

    #[test]
    fn failed_newer_decode_does_not_suppress_an_older_success() {
        let mut last_dispatched = None;

        assert!(!advance_mjpeg_dispatch_watermark(
            &mut last_dispatched,
            2,
            false
        ));
        assert!(advance_mjpeg_dispatch_watermark(
            &mut last_dispatched,
            1,
            true
        ));
        assert_eq!(last_dispatched, Some(1));
    }

    #[test]
    fn older_decode_cannot_replace_a_newer_success() {
        let mut last_dispatched = None;

        assert!(advance_mjpeg_dispatch_watermark(
            &mut last_dispatched,
            2,
            true
        ));
        assert!(!advance_mjpeg_dispatch_watermark(
            &mut last_dispatched,
            1,
            true
        ));
        assert_eq!(last_dispatched, Some(2));
    }

    #[test]
    fn async_dispatch_uses_capture_time_still_and_recording_state() {
        let still_capture = Arc::new(StillCapture::default());
        let (old_tx, mut old_rx) = tokio::sync::mpsc::channel(1);
        let (new_tx, mut new_rx) = tokio::sync::mpsc::channel(1);
        let recording_sender = Mutex::new(Some(old_tx));

        let context = snapshot_frame_dispatch_context(false, &still_capture, &recording_sender);
        let _request = still_capture.request();
        *recording_sender.lock().unwrap() = Some(new_tx);

        assert!(context.capture_still.is_none());
        context.recording_sender.unwrap().try_send(7_u8).unwrap();
        assert_eq!(old_rx.try_recv(), Ok(7));
        assert!(new_rx.try_recv().is_err());
    }
}
