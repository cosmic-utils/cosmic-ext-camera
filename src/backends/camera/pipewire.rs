// SPDX-License-Identifier: GPL-3.0-only
//! Processed V4L2 cameras exposed by the user's PipeWire session.
//! `pw-dump` is a runtime prerequisite; streaming is exclusively `pipewiresrc`.
use super::libcamera::PipelineSharedState;
use super::preview_metrics::{PreviewMetrics, StreamDiagnostics};
use super::types::*;

use gstreamer as gst;
use gstreamer_video::prelude::*;
use serde_json::Value;
use std::process::{Command, Stdio};
#[cfg(test)]
use std::sync::Mutex;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

#[path = "pipewire_metadata.rs"]
mod metadata;

pub(crate) struct PipeWirePipeline {
    pub(crate) preview_metrics: Arc<PreviewMetrics>,
    stop_flag: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

fn build_stream(
    source: gst::Element,
    format: &CameraFormat,
) -> Result<(gst::Pipeline, gstreamer_app::AppSink), String> {
    let pipeline = gst::Pipeline::new();
    let is_jpeg = format.pixel_format == "MJPEG";
    let mut caps = gst::Caps::builder(if is_jpeg { "image/jpeg" } else { "video/x-raw" })
        .field("width", format.width as i32)
        .field("height", format.height as i32);
    if !is_jpeg {
        caps = caps.field("format", &format.pixel_format);
    }
    if let Some(rate) = format.framerate {
        caps = caps.field(
            "framerate",
            gst::Fraction::new(rate.num as i32, rate.denom as i32),
        );
    }
    let filter = gst::ElementFactory::make("capsfilter")
        .name("camera-input")
        .property("caps", caps.build())
        .build()
        .map_err(|e| e.to_string())?;
    let convert = gst::ElementFactory::make("videoconvert")
        .build()
        .map_err(|e| e.to_string())?;
    let sink = gstreamer_app::AppSink::builder()
        .caps(
            &gst::Caps::builder("video/x-raw")
                .field("format", "RGBA")
                .build(),
        )
        .sync(false)
        .max_buffers(2)
        .drop(true)
        .build();
    let decoder = if is_jpeg {
        Some(
            gst::ElementFactory::make("jpegdec")
                .build()
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    let mut elements = vec![&source, &filter];
    if let Some(decoder) = &decoder {
        elements.push(decoder);
    }
    elements.extend([&convert, sink.upcast_ref()]);
    pipeline
        .add_many(elements.iter().copied())
        .map_err(|e| e.to_string())?;
    gst::Element::link_many(elements).map_err(|e| e.to_string())?;
    metadata::install(&source)?;
    Ok((pipeline, sink))
}

fn sample_to_frame(
    sample: &gst::Sample,
    metrics: Option<&PreviewMetrics>,
) -> Result<CameraFrame, String> {
    let caps = sample.caps().ok_or("missing sample caps")?;
    let info = gstreamer_video::VideoInfo::from_caps(caps).map_err(|e| e.to_string())?;
    if info.format() != gstreamer_video::VideoFormat::Rgba {
        return Err("expected RGBA from videoconvert".into());
    }
    let buffer = sample.buffer().ok_or("missing sample buffer")?;
    let mapped = gstreamer_video::VideoFrameRef::from_buffer_ref_readable(buffer, &info)
        .map_err(|e| e.to_string())?;
    let stride = mapped.plane_stride()[0];
    let row_bytes = info.width() as usize * 4;
    if stride < 0 || (stride as usize) < row_bytes {
        return Err("invalid RGBA stride".into());
    }
    let plane = mapped.plane_data(0).map_err(|e| e.to_string())?;
    let copy_started = Instant::now();
    let mut data = Vec::with_capacity(row_bytes * info.height() as usize);
    for row in 0..info.height() as usize {
        let start = row * stride as usize;
        data.extend_from_slice(
            plane
                .get(start..start + row_bytes)
                .ok_or("truncated RGBA plane")?,
        );
    }
    if let Some(metrics) = metrics {
        metrics.record_copy(copy_started.elapsed());
    }
    Ok(CameraFrame {
        width: info.width(),
        height: info.height(),
        data: FrameData::from_owned_vec(data),
        format: PixelFormat::RGBA,
        stride: info.width() * 4,
        yuv_planes: None,
        captured_at: Instant::now(),
        sensor_timestamp_ns: None,
        libcamera_metadata: None,
    })
}

fn stream_diagnostics(serial: &str, caps: &gst::CapsRef) -> StreamDiagnostics {
    let structure = caps.structure(0);
    let jpeg = structure.is_some_and(|s| s.name() == "image/jpeg");
    let source_format = if jpeg {
        "MJPEG".into()
    } else {
        structure
            .and_then(|s| s.get::<String>("format").ok())
            .unwrap_or_else(|| "unknown".into())
    };
    let framerate = structure
        .and_then(|s| s.get::<gst::Fraction>("framerate").ok())
        .and_then(|f| {
            (f.numer() > 0 && f.denom() > 0)
                .then(|| Framerate::new(f.numer() as u32, f.denom() as u32))
        });
    let decode = if jpeg { " ! jpegdec" } else { "" };
    StreamDiagnostics {
        pipeline_string: format!(
            "pipewiresrc target-object={serial} use-bufferpool=false do-timestamp=true ! {caps}{decode} ! videoconvert ! video/x-raw,format=RGBA ! appsink"
        ),
        cpu_processing: Some(if jpeg {
            "MJPEG → decoded video (jpegdec) → RGBA (videoconvert); buffer copy".into()
        } else {
            format!("{source_format} → RGBA (videoconvert); buffer copy")
        }),
        source_format,
        framerate,
        decoder: jpeg.then_some("jpegdec"),
    }
}

impl PipeWirePipeline {
    pub(crate) fn new(
        serial: &str,
        format: &CameraFormat,
        mut shared: PipelineSharedState,
    ) -> BackendResult<Self> {
        if serial.is_empty() || !serial.bytes().all(|b| b.is_ascii_digit()) {
            return Err(BackendError::DeviceNotFound(
                "invalid PipeWire serial".into(),
            ));
        }
        if shared.cancel_flag.load(Ordering::Acquire) {
            return Err(BackendError::InitializationFailed(
                "capture cancelled".into(),
            ));
        }
        gst::init().map_err(|e| BackendError::InitializationFailed(e.to_string()))?;
        // Shared GUI storage outlives pipelines; never carry a still from the
        // previous camera/format across a restart.
        shared.still_capture.reset();
        let device = CameraDevice {
            path: format!("pipewire:{serial}"),
            ..Default::default()
        };
        let available = formats(&device);
        if !available.iter().any(|f| {
            f.width == format.width
                && f.height == format.height
                && f.pixel_format == format.pixel_format
                && (format.framerate.is_none() || f.framerate == format.framerate)
        }) {
            return Err(BackendError::FormatNotSupported(format!(
                "PipeWire node {serial} does not advertise {format} {}",
                format.pixel_format
            )));
        }
        let source = gst::ElementFactory::make("pipewiresrc")
            .property("target-object", serial)
            .property("use-bufferpool", false)
            .property("do-timestamp", true)
            .build()
            .map_err(|e| BackendError::InitializationFailed(e.to_string()))?;
        let (pipeline, sink) =
            build_stream(source, format).map_err(BackendError::InitializationFailed)?;
        let preview_metrics = PreviewMetrics::start(&device.path);
        let stop_flag = Arc::new(AtomicBool::new(false));
        let mut result = Self {
            preview_metrics: Arc::clone(&preview_metrics),
            stop_flag: Arc::clone(&stop_flag),
            thread: None,
        };
        let serial = serial.to_owned();
        let (init_tx, init_rx) = std::sync::mpsc::sync_channel(1);
        result.thread = Some(
            std::thread::Builder::new()
                .name("pipewire-camera".into())
                .spawn(move || {
                    let mut init_tx = Some(init_tx);
                    let mut published_stream = false;
                    let mut appsink_dropped = 0u64;
                    let has_drop_counter = sink.find_property("dropped").is_some();
                    let start = Instant::now();
                    let bus = pipeline.bus().unwrap();
                    let run = || -> Result<(), String> {
                        pipeline
                            .set_state(gst::State::Playing)
                            .map_err(|e| e.to_string())?;
                        Ok(())
                    };
                    let mut failure = run().err();
                    while failure.is_none()
                        && !stop_flag.load(Ordering::Acquire)
                        && !shared.cancel_flag.load(Ordering::Acquire)
                    {
                        if let Some(message) =
                            bus.pop_filtered(&[gst::MessageType::Error, gst::MessageType::Eos])
                        {
                            failure = Some(match message.view() {
                                gst::MessageView::Error(e) => {
                                    format!("{} ({:?})", e.error(), e.debug())
                                }
                                _ => "PipeWire camera disconnected (EOS)".into(),
                            });
                            break;
                        }
                        if let Some(sample) =
                            sink.try_pull_sample(gst::ClockTime::from_mseconds(100))
                        {
                            if !published_stream
                                && let Some(caps) = pipeline
                                    .by_name("camera-input")
                                    .and_then(|element| element.static_pad("src"))
                                    .and_then(|pad| pad.current_caps())
                            {
                                preview_metrics.set_stream(stream_diagnostics(&serial, &caps));
                                published_stream = true;
                            }
                            // GStreamer >= 1.28 exposes losses inside the bounded appsink.
                            if has_drop_counter {
                                let dropped = sink.property::<u64>("dropped");
                                preview_metrics
                                    .record_drops(dropped.saturating_sub(appsink_dropped));
                                appsink_dropped = dropped;
                            }
                            let still_ticket = shared.still_capture.pending();
                            match sample_to_frame(&sample, Some(&preview_metrics)) {
                                Ok(mut frame) => {
                                    // Translate stream PTS to CLOCK_BOOTTIME, the recorder's
                                    // timestamp domain. This is a PipeWire presentation timestamp,
                                    // not fabricated libcamera sensor metadata.
                                    if let (Some(buffer), Some(running)) =
                                        (sample.buffer(), pipeline.current_running_time())
                                        && let Some(pts) = buffer.pts()
                                        && let Some(segment) = sample
                                            .segment()
                                            .and_then(|s| s.downcast_ref::<gst::ClockTime>())
                                        && let Some(pts) = segment.to_running_time(pts)
                                    {
                                        let age = running.nseconds().saturating_sub(pts.nseconds());
                                        let mut ts =
                                            std::mem::MaybeUninit::<libc::timespec>::uninit();
                                        // SAFETY: valid writable timespec; read only on success.
                                        if unsafe {
                                            libc::clock_gettime(
                                                libc::CLOCK_BOOTTIME,
                                                ts.as_mut_ptr(),
                                            )
                                        } == 0
                                        {
                                            let ts = unsafe { ts.assume_init() };
                                            let now_boot = ts.tv_sec as u64 * 1_000_000_000
                                                + ts.tv_nsec as u64;
                                            frame.sensor_timestamp_ns = now_boot.checked_sub(age);
                                            frame.captured_at = Instant::now()
                                                .checked_sub(Duration::from_nanos(age))
                                                .unwrap_or(frame.captured_at);
                                        }
                                    }
                                    if let Some(tx) = init_tx.take() {
                                        let _ = tx.send(Ok(()));
                                    }
                                    dispatch_frame(
                                        frame,
                                        still_ticket,
                                        &mut shared,
                                        &preview_metrics,
                                    );
                                }
                                Err(e) => {
                                    failure = Some(e);
                                    break;
                                }
                            }
                        } else if init_tx.is_some() && start.elapsed() > Duration::from_secs(5) {
                            failure = Some(
                                "PipeWire camera produced no frames within five seconds".into(),
                            );
                        }
                    }
                    preview_metrics.stop();
                    let _ = pipeline.set_state(gst::State::Null);
                    if let Some(tx) = init_tx.take() {
                        let _ = tx.send(Err(failure
                            .clone()
                            .unwrap_or_else(|| "capture cancelled".into())));
                    }
                    if let Some(error) = failure {
                        tracing::warn!(%error, "PipeWire camera stream stopped");
                    }
                })
                .map_err(|e| BackendError::InitializationFailed(e.to_string()))?,
        );
        init_rx
            .recv_timeout(Duration::from_secs(7))
            .map_err(|e| BackendError::InitializationFailed(e.to_string()))?
            .map_err(BackendError::InitializationFailed)?;
        Ok(result)
    }

    pub(crate) fn stop(&self) -> BackendResult<()> {
        self.preview_metrics.stop();
        self.stop_flag.store(true, Ordering::Release);
        Ok(())
    }
}
impl Drop for PipeWirePipeline {
    fn drop(&mut self) {
        let _ = self.stop();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Publish a copied frame without retaining a second latest-preview slot.
/// The ticket was observed before sample copying; never resnapshot it here.
fn dispatch_frame(
    frame: CameraFrame,
    still_ticket: Option<super::still_capture::StillTicket>,
    shared: &mut PipelineSharedState,
    preview_metrics: &PreviewMetrics,
) {
    preview_metrics.record_frame(&frame);
    if let Some(ticket) = still_ticket {
        shared.still_capture.publish(ticket, frame.clone());
    }
    if let Some(tx) = shared.recording_sender.lock().unwrap().as_ref() {
        let _ = tx.try_send(Arc::new(frame.clone()));
    }
    if shared.frame_sender.try_send(frame).is_err() {
        preview_metrics.record_drops(1);
    }
}

#[derive(Clone, Debug)]
struct Node {
    device: CameraDevice,
    formats: Vec<CameraFormat>,
    identities: Vec<String>,
}

fn positive_int(value: &Value) -> Option<u32> {
    value
        .as_u64()
        .filter(|v| *v > 0 && *v <= i32::MAX as u64)
        .map(|v| v as u32)
}

fn choice_values(value: &Value) -> Vec<&Value> {
    if let Some(map) = value.as_object()
        && map.contains_key("default")
    {
        // For ranges/steps advertise the supplied default, not invented modes.
        return map
            .iter()
            .filter(|(k, _)| k.as_str() == "default" || k.starts_with("alt"))
            .map(|(_, v)| v)
            .take(32)
            .collect();
    }
    vec![value]
}

fn nodes_from_dump(dump: &Value) -> Vec<Node> {
    if gstreamer::init().is_err() {
        return Vec::new();
    }
    let Some(objects) = dump.as_array() else {
        return Vec::new();
    };
    // Device metadata bridges libcamera's camera ID to kernel dev_t values.
    // A dev_t shared by multiple libcamera cameras is ambiguous: retain every
    // native camera, but suppress a possible duplicate fallback on that controller.
    let mut native_ids: std::collections::HashMap<u64, std::collections::HashSet<String>> =
        std::collections::HashMap::new();
    for object in objects {
        let props = &object["info"]["props"];
        if props["device.api"] == "libcamera"
            && let Some(id) = props["api.libcamera.path"].as_str()
        {
            for number in device_numbers(&props["device.devids"]) {
                native_ids.entry(number).or_default().insert(id.to_string());
            }
        }
    }
    let mut nodes = Vec::new();
    for object in objects {
        if object["type"] != "PipeWire:Interface:Node" {
            continue;
        }
        let props = &object["info"]["props"];
        if props["media.class"] != "Video/Source" || props["device.api"] != "v4l2" {
            continue;
        }
        let Some(path) = props["api.v4l2.path"].as_str() else {
            continue;
        };
        if !std::path::Path::new(path).is_absolute() {
            continue;
        }
        let Some(serial) = props["object.serial"].as_u64() else {
            continue;
        };
        let mut formats = Vec::new();
        if let Some(params) = object["info"]["params"]["EnumFormat"].as_array() {
            for p in params {
                if p["mediaType"] != "video"
                    || (p["mediaSubtype"] != "raw" && p["mediaSubtype"] != "mjpg")
                {
                    continue;
                }
                let jpeg = Value::String("MJPEG".into());
                let pixels = if p["mediaSubtype"] == "mjpg" {
                    vec![&jpeg]
                } else {
                    choice_values(&p["format"])
                };
                for pixel in pixels {
                    let Some(pixel) = pixel.as_str() else {
                        continue;
                    };
                    if pixel != "MJPEG" && pixel.parse::<gstreamer_video::VideoFormat>().is_err() {
                        continue;
                    }
                    for size in choice_values(&p["size"]) {
                        let (Some(width), Some(height)) =
                            (positive_int(&size["width"]), positive_int(&size["height"]))
                        else {
                            continue;
                        };
                        for rate in choice_values(&p["framerate"]) {
                            let framerate =
                                match (positive_int(&rate["num"]), positive_int(&rate["denom"])) {
                                    (Some(n), Some(d)) => Some(Framerate::new(n, d)),
                                    _ => None,
                                };
                            formats.push(CameraFormat {
                                width,
                                height,
                                framerate,
                                hardware_accelerated: false,
                                pixel_format: pixel.into(),
                            });
                        }
                    }
                }
            }
        }
        formats.sort_by_key(|f| {
            (
                std::cmp::Reverse(u64::from(f.width) * u64::from(f.height)),
                f.pixel_format.clone(),
                f.framerate.map(|r| (r.num, r.denom)),
            )
        });
        formats.dedup();
        if formats.is_empty() {
            continue;
        }
        let name = props["node.description"]
            .as_str()
            .or(props["node.nick"].as_str())
            .unwrap_or(path)
            .to_string();
        nodes.push(Node {
            identities: {
                let mut ids = path_identities(path);
                let device_props = objects
                    .iter()
                    .find(|o| {
                        o["type"] == "PipeWire:Interface:Device" && o["id"] == props["device.id"]
                    })
                    .map(|o| &o["info"]["props"]);
                for metadata in std::iter::once(props).chain(device_props) {
                    if let Some(identity) = metadata["device.sysfs.path"]
                        .as_str()
                        .and_then(usb_interface_identity)
                    {
                        ids.push(identity);
                    }
                }
                let numbers = device_props
                    .and_then(|p| single_device_number(&p["device.devids"]))
                    .or_else(|| single_device_number(&props["device.devids"]));
                if let Some(devid) = numbers {
                    ids.push(format!("rdev:{devid}"));
                    if let Some(cameras) = native_ids.get(&devid) {
                        ids.extend(cameras.iter().map(|id| format!("libcamera:{id}")));
                    }
                }
                ids
            },
            device: CameraDevice {
                name,
                path: format!("pipewire:{serial}"),
                device_info: Some(DeviceInfo {
                    card: props["api.v4l2.cap.card"].as_str().unwrap_or("").into(),
                    driver: props["api.v4l2.cap.driver"].as_str().unwrap_or("").into(),
                    path: path.into(),
                    real_path: std::fs::canonicalize(path)
                        .map(|p| p.to_string_lossy().into_owned())
                        .unwrap_or_else(|_| path.into()),
                }),
                ..Default::default()
            },
            formats,
        });
    }
    nodes
}

fn discover() -> Vec<Node> {
    let mut child = match Command::new("pw-dump")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            tracing::warn!(error = %e, "PipeWire camera discovery needs pw-dump");
            return Vec::new();
        }
    };
    // Drain stdout concurrently: a registry dump can exceed a pipe's capacity.
    let stdout = child.stdout.take().unwrap();
    let reader = std::thread::spawn(move || {
        use std::io::Read;
        let mut bytes = Vec::new();
        stdout
            .take(8 * 1024 * 1024)
            .read_to_end(&mut bytes)
            .map(|_| bytes)
    });
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => break,
            Ok(Some(_)) | Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader.join();
                return Vec::new();
            }
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader.join();
                tracing::warn!("PipeWire discovery timed out");
                return Vec::new();
            }
        }
    }
    reader
        .join()
        .ok()
        .and_then(Result::ok)
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .map(|dump| nodes_from_dump(&dump))
        .unwrap_or_default()
}

pub(crate) fn append_fallbacks(native: Vec<CameraDevice>) -> Vec<CameraDevice> {
    merge_nodes(native, discover())
}

pub(crate) fn same_physical_camera(a: &CameraDevice, b: &CameraDevice) -> bool {
    let (Some(a), Some(b)) = (&a.device_info, &b.device_info) else {
        return false;
    };
    let a_ids: std::collections::HashSet<_> = path_identities(&a.path)
        .into_iter()
        .chain(path_identities(&a.real_path))
        .collect();
    path_identities(&b.path)
        .into_iter()
        .chain(path_identities(&b.real_path))
        .any(|id| a_ids.contains(&id))
}

fn merge_nodes(mut native: Vec<CameraDevice>, nodes: Vec<Node>) -> Vec<CameraDevice> {
    let mut seen: std::collections::HashSet<String> = native
        .iter()
        .flat_map(|d| {
            d.device_info
                .as_ref()
                .map(|info| {
                    path_identities(&info.real_path)
                        .into_iter()
                        .chain(path_identities(&info.path))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        })
        .collect();
    seen.extend(native.iter().map(|d| format!("libcamera:{}", d.path)));
    for node in nodes {
        let path = node.device.device_info.as_ref().unwrap().real_path.as_str();
        if node.identities.iter().any(|id| seen.contains(id))
            || native.iter().any(|d| {
                d.path == node.device.path
                    || d.device_info
                        .as_ref()
                        .is_some_and(|info| info.real_path == path || info.path == path)
            })
        {
            continue;
        }
        // Registry camera IDs bridge to actual native descriptors only. They
        // cannot prove that two distinct fallback endpoints are one sensor.
        seen.extend(
            node.identities
                .into_iter()
                .filter(|id| !id.starts_with("libcamera:")),
        );
        native.push(node.device);
    }
    native
}

pub(crate) fn formats(device: &CameraDevice) -> Vec<CameraFormat> {
    discover()
        .into_iter()
        .find(|n| n.device.path == device.path)
        .map(|n| n.formats)
        .unwrap_or_default()
}

// Missing/aggregate identities are not evidence that two cameras are identical.
fn single_device_number(value: &Value) -> Option<u64> {
    let numbers = device_numbers(value);
    if numbers.len() != 1 {
        return None;
    }
    Some(numbers[0])
}

fn device_numbers(value: &Value) -> Vec<u64> {
    if let Some(array) = value.as_array() {
        return array
            .iter()
            .filter_map(Value::as_u64)
            .filter(|v| *v != 0)
            .collect();
    }
    // SPA JSON allows a trailing comma, unlike serde_json's strict JSON parser.
    let Some(text) = value
        .as_str()
        .and_then(|s| s.trim().strip_prefix('['))
        .and_then(|s| s.strip_suffix(']'))
    else {
        return Vec::new();
    };
    let text = text.trim();
    let parts = text.strip_suffix(',').unwrap_or(text).split(',');
    let parsed: Result<Vec<u64>, _> = parts.map(|p| p.trim().parse::<u64>()).collect();
    parsed
        .unwrap_or_default()
        .into_iter()
        .filter(|v| *v != 0)
        .collect()
}

fn path_identities(path: &str) -> Vec<String> {
    use std::os::unix::fs::{FileTypeExt, MetadataExt};
    if path.is_empty() {
        return Vec::new();
    }
    let canonical = std::fs::canonicalize(path)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| path.into());
    let mut ids = vec![format!("path:{canonical}")];
    if let Ok(meta) = std::fs::metadata(path)
        && meta.file_type().is_char_device()
    {
        ids.push(format!("rdev:{}", meta.rdev()));
    }
    if let Some(name) = std::path::Path::new(&canonical)
        .file_name()
        .and_then(|n| n.to_str())
        && name.starts_with("video")
        && let Ok(device) = std::fs::canonicalize(format!("/sys/class/video4linux/{name}/device"))
        && let Some(identity) = usb_interface_identity(&device.to_string_lossy())
    {
        ids.push(identity);
    }
    ids
}

fn usb_interface_identity(sysfs: &str) -> Option<String> {
    // USB interfaces identify UVC camera functions; an ISP/controller parent
    // does NOT identify one sensor, so platform paths are deliberately excluded.
    let path = sysfs.strip_prefix("/sys").unwrap_or(sysfs);
    let parent = path.split("/video4linux/").next()?;
    if !parent.contains("/usb") || !parent.rsplit('/').next()?.contains(':') {
        return None;
    }
    Some(format!("usb-interface:{parent}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn node(serial: u64, path: &str, devid: u64) -> Value {
        json!({"id": serial, "type": "PipeWire:Interface:Node", "info": {
            "props": {"media.class": "Video/Source", "device.api": "v4l2", "api.v4l2.path": path,
                "object.serial": serial, "node.description": "Identical webcam", "device.devids": format!("[ {devid} ]")},
            "params": {"EnumFormat": [{"mediaType": "video", "mediaSubtype": "raw", "format": "NV12",
                "size": {"width":640, "height":480}, "framerate":{"num":30,"denom":1}}]}}})
    }
    #[test]
    fn dedup_pipewire_nodes_by_device_number_not_name_or_serial() {
        let dump = json!([
            node(101, "/dev/fake-video-a", 8123),
            node(102, "/dev/fake-alias-a", 8123),
            node(103, "/dev/fake-video-b", 8124)
        ]);
        let devices = merge_nodes(vec![], nodes_from_dump(&dump));
        assert_eq!(
            devices.len(),
            2,
            "aliases must merge, same-model distinct cameras must survive"
        );
        assert_eq!(devices[0].path, "pipewire:101");
        assert_eq!(devices[1].path, "pipewire:103");
    }

    #[test]
    fn dedup_usb_video_endpoints_by_sysfs_interface_preserves_distinct_webcams() {
        let mut a = node(101, "/dev/fake-a", 8123);
        let mut alias = node(102, "/dev/fake-alias", 8124);
        let mut b = node(103, "/dev/fake-b", 8125);
        a["info"]["props"]["device.sysfs.path"] =
            json!("/devices/pci/usb1/1-2/1-2:1.0/video4linux/video0");
        alias["info"]["props"]["device.sysfs.path"] =
            json!("/devices/pci/usb1/1-2/1-2:1.0/video4linux/video2");
        b["info"]["props"]["device.sysfs.path"] =
            json!("/devices/pci/usb1/1-3/1-3:1.0/video4linux/video4");
        let devices = merge_nodes(vec![], nodes_from_dump(&json!([a, alias, b])));
        assert_eq!(
            devices.len(),
            2,
            "same USB camera endpoint aliases must merge"
        );
        assert_eq!(devices[1].path, "pipewire:103");
    }

    #[test]
    fn native_preferred_via_pipewire_device_metadata_without_v4l2_path() {
        let mut fallback = node(101, "/dev/fake-video-a", 8123);
        fallback["info"]["props"]
            .as_object_mut()
            .unwrap()
            .remove("device.devids");
        fallback["info"]["props"]["device.id"] = json!(88);
        let dump = json!([
            {"id":88,"type":"PipeWire:Interface:Device","info":{"props":{"device.devids":"[ 8123 ]"}}},
            {"id":89,"type":"PipeWire:Interface:Device","info":{"props":{"device.api":"libcamera", "api.libcamera.path":"platform/sensor-1", "device.devids":"[ 8123 ]"}}},
            fallback
        ]);
        let native = CameraDevice {
            path: "platform/sensor-1".into(),
            name: "Native".into(),
            supports_multistream: true,
            ..Default::default()
        };
        let devices = merge_nodes(vec![native], nodes_from_dump(&dump));
        assert_eq!(
            devices.len(),
            1,
            "same physical camera must retain native representation"
        );
        assert_eq!(devices[0].path, "platform/sensor-1");
        assert!(devices[0].supports_multistream);
    }

    #[test]
    fn processed_choice_formats_and_mjpeg_are_negotiable() {
        let mut raw = node(101, "/dev/fake-a", 8123);
        raw["info"]["params"]["EnumFormat"][0]["format"] = json!({"default":"NV12","alt1":"YUY2"});
        raw["info"]["params"]["EnumFormat"][0]["size"] = json!({"default":{"width":640,"height":480},"min":{"width":16,"height":16},"max":{"width":1920,"height":1080}});
        raw["info"]["params"]["EnumFormat"][0]["framerate"] =
            json!({"default":{"num":30,"denom":1},"alt1":{"num":60000,"denom":1001}});
        let mut jpeg = node(102, "/dev/fake-b", 8124);
        jpeg["info"]["params"]["EnumFormat"][0]["mediaSubtype"] = json!("mjpg");
        jpeg["info"]["params"]["EnumFormat"][0]
            .as_object_mut()
            .unwrap()
            .remove("format");
        let nodes = nodes_from_dump(&json!([raw, jpeg]));
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].formats.len(), 4);
        assert!(
            nodes[0]
                .formats
                .iter()
                .all(|f| f.width == 640 && f.height == 480),
            "do not invent intermediate range modes"
        );
        assert_eq!(nodes[1].formats[0].pixel_format, "MJPEG");
    }

    #[test]
    fn padded_nv12_video_meta_is_respected_by_conversion() {
        gst::init().unwrap();
        let caps = gst::Caps::builder("video/x-raw")
            .field("format", "NV12")
            .field("width", 8i32)
            .field("height", 4i32)
            .field("framerate", gst::Fraction::new(30, 1))
            .build();
        let source = gstreamer_app::AppSrc::builder().caps(&caps).build();
        let format = CameraFormat {
            width: 8,
            height: 4,
            framerate: Some(Framerate::from_int(30)),
            pixel_format: "NV12".into(),
            hardware_accelerated: false,
        };
        let (pipeline, sink) = build_stream(source.clone().upcast(), &format).unwrap();
        // Correct upstream metadata must reach videoconvert without application repair.
        let mut bytes = vec![0u8; 16 * 6];
        for row in 0..4 {
            bytes[row * 16..row * 16 + 8].fill(100);
        }
        for row in 4..6 {
            bytes[row * 16..row * 16 + 8].fill(128);
        }
        let mut buffer = gst::Buffer::from_mut_slice(bytes);
        gstreamer_video::VideoMeta::add_full(
            buffer.get_mut().unwrap(),
            gstreamer_video::VideoFrameFlags::empty(),
            gstreamer_video::VideoFormat::Nv12,
            8,
            4,
            &[0, 16 * 4],
            &[16, 16],
        )
        .unwrap();
        pipeline.set_state(gst::State::Playing).unwrap();
        source.push_buffer(buffer).unwrap();
        source.end_of_stream().unwrap();
        let sample = sink.try_pull_sample(gst::ClockTime::from_seconds(2));
        pipeline.set_state(gst::State::Null).unwrap();
        let frame = sample_to_frame(&sample.expect("converted padded NV12 frame"), None).unwrap();
        for pixel in frame.data.as_chunks::<4>().0 {
            let min = *pixel[..3].iter().min().unwrap();
            let max = *pixel[..3].iter().max().unwrap();
            assert!(
                max - min <= 2,
                "neutral chroma must remain gray, got {pixel:?}"
            );
        }
    }

    #[test]
    fn mjpeg_stream_decodes_to_owned_rgba() {
        gst::init().unwrap();
        let source = gst::parse::bin_from_description(
            "videotestsrc num-buffers=1 ! video/x-raw,width=16,height=16,framerate=30/1 ! jpegenc",
            true,
        )
        .unwrap()
        .upcast();
        let format = CameraFormat {
            width: 16,
            height: 16,
            framerate: Some(Framerate::from_int(30)),
            pixel_format: "MJPEG".into(),
            hardware_accelerated: false,
        };
        let (pipeline, sink) = build_stream(source, &format).expect("MJPEG capture graph");
        pipeline.set_state(gst::State::Playing).unwrap();
        let sample = sink.try_pull_sample(gst::ClockTime::from_seconds(2));
        pipeline.set_state(gst::State::Null).unwrap();
        let frame = sample_to_frame(&sample.expect("decoded MJPEG sample"), None).unwrap();
        assert_eq!(frame.format, PixelFormat::RGBA);
        assert_eq!(frame.data.len(), 16 * 16 * 4);
    }

    #[test]
    fn missing_device_path_is_not_a_fallback_camera() {
        assert!(nodes_from_dump(&json!([node(101, "", 8123)])).is_empty());
    }

    #[test]
    fn native_path_is_canonicalized_even_without_cached_real_path() {
        use std::os::unix::fs::MetadataExt;
        let devid = std::fs::metadata("/dev/null").unwrap().rdev();
        let native = CameraDevice {
            path: "native-id".into(),
            device_info: Some(DeviceInfo {
                path: "/dev/../dev/null".into(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let devices = merge_nodes(
            vec![native],
            nodes_from_dump(&json!([node(101, "/dev/null", devid)])),
        );
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].path, "native-id");
    }

    fn shared_state() -> (PipelineSharedState, FrameReceiver) {
        let (frame_sender, receiver) = futures::channel::mpsc::channel(1);
        (
            PipelineSharedState {
                zoom: Default::default(),
                frame_sender,
                still_capture: Default::default(),
                recording_sender: Arc::new(Mutex::new(None)),
                cancel_flag: Arc::new(AtomicBool::new(false)),
            },
            receiver,
        )
    }

    #[test]
    fn absent_or_aggregate_identity_never_merges_same_model_cameras() {
        let mut a = node(101, "/dev/fake-a", 8123);
        let mut b = node(102, "/dev/fake-b", 8124);
        a["info"]["props"]
            .as_object_mut()
            .unwrap()
            .remove("device.devids");
        b["info"]["props"]["device.devids"] = json!("[ 8123, 8124, ]");
        let native = CameraDevice {
            name: "Identical webcam".into(),
            path: "unidentified-native-camera".into(),
            ..Default::default()
        };
        assert_eq!(
            merge_nodes(vec![native], nodes_from_dump(&json!([a, b]))).len(),
            3
        );
        assert_eq!(single_device_number(&json!("[ 8123, 8124, ]")), None);
        assert_eq!(single_device_number(&json!("[ 8123, ]")), Some(8123));
    }

    #[test]
    fn malformed_device_identity_is_not_repaired_to_merge_distinct_cameras() {
        let mut b = node(102, "/dev/fake-b", 9000);
        b["info"]["props"]["device.devids"] = json!("[ 8123,, ]");
        let dump = json!([node(101, "/dev/fake-a", 8123), b]);
        assert_eq!(merge_nodes(vec![], nodes_from_dump(&dump)).len(), 2);
        assert_eq!(single_device_number(&json!("[ 8123, ]")), Some(8123));
    }

    #[test]
    fn ambiguous_native_controller_preserves_sensors_but_suppresses_uncertain_fallback() {
        let dump = json!([
            {"id":88,"info":{"props":{"device.api":"libcamera","api.libcamera.path":"sensor-a","device.devids":"[ 8123, ]"}}},
            {"id":89,"info":{"props":{"device.api":"libcamera","api.libcamera.path":"sensor-b","device.devids":"[ 8123, ]"}}},
            node(101,"/dev/fake-a",8123)
        ]);
        let native = vec![
            CameraDevice {
                path: "sensor-a".into(),
                ..Default::default()
            },
            CameraDevice {
                path: "sensor-b".into(),
                ..Default::default()
            },
        ];
        let devices = merge_nodes(native, nodes_from_dump(&dump));
        assert_eq!(
            devices.len(),
            2,
            "do not expose a possible duplicate on a known native controller"
        );
        assert_eq!(devices[0].path, "sensor-a");
        assert_eq!(devices[1].path, "sensor-b");
        assert_eq!(
            usb_interface_identity("/devices/platform/isp/video4linux/video0"),
            None
        );
    }

    #[test]
    fn ambiguous_registry_bridge_does_not_merge_two_unrepresented_fallback_cameras() {
        let dump = json!([
            {"id":88,"info":{"props":{"device.api":"libcamera","api.libcamera.path":"sensor-a","device.devids":"[ 8123, 8124, ]"}}},
            {"id":89,"info":{"props":{"device.api":"libcamera","api.libcamera.path":"sensor-b","device.devids":"[ 8123, 8124, ]"}}},
            node(101,"/dev/fake-a",8123),node(102,"/dev/fake-b",8124)
        ]);
        assert_eq!(
            merge_nodes(vec![], nodes_from_dump(&dump)).len(),
            2,
            "only an actual native descriptor can suppress via a registry bridge"
        );
    }

    #[test]
    fn only_processed_v4l2_sources_are_fallbacks() {
        let mut libcamera = node(101, "/dev/fake-a", 8123);
        libcamera["info"]["props"]["device.api"] = json!("libcamera");
        let mut bayer = node(102, "/dev/fake-b", 8124);
        bayer["info"]["params"]["EnumFormat"][0]["format"] = json!("rggb");
        let mut metadata = node(103, "/dev/fake-c", 8125);
        metadata["info"]["params"]["EnumFormat"][0]["mediaSubtype"] = json!("metadata");
        let mut sink = node(104, "/dev/fake-d", 8126);
        sink["info"]["props"]["media.class"] = json!("Video/Sink");
        let mut invalid = node(105, "/dev/fake-e", 8127);
        invalid["info"]["params"]["EnumFormat"][0]["size"]["width"] = json!(0);
        assert!(nodes_from_dump(&json!([libcamera, bayer, metadata, sink, invalid])).is_empty());
    }

    fn rgba_sample(bytes: Vec<u8>, offset: usize, stride: i32) -> gst::Sample {
        gst::init().unwrap();
        let mut buffer = gst::Buffer::from_mut_slice(bytes);
        gstreamer_video::VideoMeta::add_full(
            buffer.get_mut().unwrap(),
            gstreamer_video::VideoFrameFlags::empty(),
            gstreamer_video::VideoFormat::Rgba,
            2,
            2,
            &[offset],
            &[stride],
        )
        .unwrap();
        let caps = gst::Caps::builder("video/x-raw")
            .field("format", "RGBA")
            .field("width", 2i32)
            .field("height", 2i32)
            .field("framerate", gst::Fraction::new(30, 1))
            .build();
        gst::Sample::builder().buffer(&buffer).caps(&caps).build()
    }

    #[test]
    fn copied_sample_reaches_still_recording_and_preview_under_backpressure() {
        let metrics = PreviewMetrics::start("pipewire-dispatch-consumers-test");
        let sample = rgba_sample((1..=16).collect(), 0, 8);
        let frame = sample_to_frame(&sample, None).unwrap();
        let (mut shared, mut preview) = shared_state();
        let (tx, mut recording) = tokio::sync::mpsc::channel(1);
        *shared.recording_sender.lock().unwrap() = Some(tx);
        let request = shared.still_capture.request();
        let ticket = shared.still_capture.pending();
        dispatch_frame(frame.clone(), ticket, &mut shared, &metrics);
        let still = request.take().unwrap();
        let delivered = preview.try_recv().unwrap();
        let recorded = recording.try_recv().unwrap();
        for output in [&still, &delivered, recorded.as_ref()] {
            assert_eq!(output.data.as_ref(), &(1..=16).collect::<Vec<_>>());
            assert_eq!(output.data.as_ptr(), frame.data.as_ptr());
            assert_eq!(output.captured_at, frame.captured_at);
        }
        // Fill both bounded queues. Still publication must remain independent.
        while shared.frame_sender.try_send(frame.clone()).is_ok() {}
        let tx = shared.recording_sender.lock().unwrap().clone().unwrap();
        tx.try_send(Arc::new(frame.clone())).unwrap();
        let stale = shared.still_capture.request();
        let stale_ticket = shared.still_capture.pending();
        drop(stale);
        let fresh = shared.still_capture.request();
        let fresh_ticket = shared.still_capture.pending();
        dispatch_frame(frame.clone(), stale_ticket, &mut shared, &metrics);
        assert!(fresh.take().is_none());
        assert_eq!(shared.still_capture.pending(), fresh_ticket);
        dispatch_frame(frame, fresh_ticket, &mut shared, &metrics);
        let retained = fresh.take().unwrap();
        assert_eq!(metrics.snapshot().dropped_frames, 2);
        drop(shared);
        drop(sample);
        assert_eq!(retained.data.as_ref(), &(1..=16).collect::<Vec<_>>());
    }

    #[test]
    fn rgba_copy_respects_video_meta_offset_and_row_padding() {
        let sample = rgba_sample(
            vec![
                99, 99, 99, 99, 1, 2, 3, 4, 5, 6, 7, 8, 99, 99, 99, 99, 9, 10, 11, 12, 13, 14, 15,
                16, 99, 99, 99, 99,
            ],
            4,
            12,
        );
        let frame = sample_to_frame(&sample, None).unwrap();
        drop(sample);
        assert_eq!(
            &*frame.data,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
        assert_eq!(frame.stride, 8);
        assert!(frame.yuv_planes.is_none());
        assert!(frame.libcamera_metadata.is_none());
    }

    #[test]
    fn actual_rgba_buffer_copy_publishes_a_measurement() {
        let metrics = PreviewMetrics::start("pipewire:copy-test");
        let sample = rgba_sample(vec![0; 16], 0, 8);
        assert!(metrics.snapshot().copy_time_us.is_none());
        let frame = sample_to_frame(&sample, Some(&metrics)).unwrap();
        metrics.record_frame(&frame);
        let snapshot = metrics.snapshot();
        assert!(snapshot.copy_time_us.is_some());
        assert_eq!(snapshot.frame_size_bytes, frame.data.len());
        assert_eq!(snapshot.output_format, Some(PixelFormat::RGBA));
    }

    #[test]
    fn negotiated_caps_describe_the_actual_pipewire_cpu_pipeline() {
        gst::init().unwrap();
        let caps = gst::Caps::builder("video/x-raw")
            .field("format", "NV12")
            .field("width", 1328i32)
            .field("height", 1760i32)
            .field("framerate", gst::Fraction::new(30, 1))
            .build();
        let raw = stream_diagnostics("123", &caps);
        assert_eq!(raw.source_format, "NV12");
        assert_eq!(raw.framerate, Some(Framerate::new(30, 1)));
        assert!(
            raw.pipeline_string
                .contains("pipewiresrc target-object=123")
        );
        assert!(raw.cpu_processing.unwrap().contains("videoconvert"));
        assert_eq!(raw.decoder, None);
        let jpeg = stream_diagnostics("123", &gst::Caps::builder("image/jpeg").build());
        assert_eq!(jpeg.source_format, "MJPEG");
        assert_eq!(jpeg.decoder, Some("jpegdec"));
        assert!(jpeg.pipeline_string.contains("! jpegdec ! videoconvert"));
    }

    #[test]
    fn negative_rgba_stride_is_rejected_without_panicking() {
        let sample = rgba_sample(vec![0; 24], 12, -12);
        assert!(sample_to_frame(&sample, None).is_err());
    }

    #[test]
    fn cancelled_pipeline_does_not_start_capture() {
        let (shared, _receiver) = shared_state();
        shared.cancel_flag.store(true, Ordering::Release);
        let format = CameraFormat {
            width: 640,
            height: 480,
            framerate: Some(Framerate::from_int(30)),
            pixel_format: "NV12".into(),
            hardware_accelerated: false,
        };
        let start = Instant::now();
        assert!(matches!(
            PipeWirePipeline::new("101", &format, shared),
            Err(BackendError::InitializationFailed(_))
        ));
        assert!(start.elapsed() < Duration::from_secs(1));
    }

    #[test]
    #[ignore = "requires a live PipeWire V4L2 camera"]
    fn live_pipeline_photo_format_change_restart_and_shutdown() {
        use super::super::CameraBackend;
        let backend = super::super::libcamera::LibcameraBackend::new();
        let devices = backend.enumerate_cameras();
        let device = devices
            .iter()
            .find(|d| d.path.starts_with("pipewire:"))
            .expect("PipeWire camera");
        let formats = backend.get_formats(device, true);
        let small = formats
            .iter()
            .find(|f| f.width == 640 && f.height == 480)
            .unwrap_or(&formats[0]);
        let start_pipeline = |format: &CameraFormat| {
            let (shared, receiver) = shared_state();
            let still = shared.still_capture.clone();
            let pipeline =
                super::super::pipeline::CapturePipeline::new(&device.path, format, false, shared)
                    .unwrap();
            (pipeline, still, receiver)
        };
        let capture = |still: &Arc<super::super::still_capture::StillCapture>| {
            let request = still.request();
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                if let Some(frame) = request.take() {
                    break frame;
                }
                assert!(Instant::now() < deadline, "ticketed still timed out");
                std::thread::sleep(Duration::from_millis(10));
            }
        };
        let (pipeline, still, mut receiver) = start_pipeline(small);
        let photo = capture(&still);
        assert!(receiver.try_recv().is_ok());
        assert_eq!((photo.width, photo.height), (small.width, small.height));
        let first = PreviewMetrics::active_for(&device.path).unwrap().snapshot();
        let mut insights = crate::app::insights::InsightsState::default();
        insights.refresh_preview(&device.path, Some(&first), Instant::now());
        assert_eq!(insights.backend_type, "PipeWire (V4L2)");
        assert_eq!(insights.frame_size_decoded, photo.data.len());
        assert_eq!(
            insights.preview_stream.as_ref().unwrap().pixel_format,
            "RGBA"
        );
        assert!(insights.copy_time_us.is_some());
        assert!(
            insights
                .full_pipeline_string
                .as_ref()
                .unwrap()
                .contains("pipewiresrc")
        );
        assert!(insights.frame_metadata.is_none());
        println!(
            "Live Insights: backend={}, frame_count={}, source_format={}, bytes={}, copy_us={:?}",
            insights.backend_type,
            first.frame_count,
            insights.format_chain.native_format,
            insights.frame_size_decoded,
            insights.copy_time_us
        );
        let larger = formats
            .iter()
            .find(|f| f.width == 1280 && f.height == 720)
            .unwrap_or(&formats[0]);
        drop(pipeline);
        let (pipeline, still, _receiver) = start_pipeline(larger);
        let photo = capture(&still);
        assert_eq!((photo.width, photo.height), (larger.width, larger.height));
        let second = PreviewMetrics::active_for(&device.path).unwrap().snapshot();
        assert_ne!(first.generation, second.generation);
        insights.refresh_preview(&device.path, Some(&second), Instant::now());
        assert_eq!(insights.frame_size_decoded, photo.data.len());
        drop(pipeline);
        let (pipeline, still, _receiver) = start_pipeline(small);
        assert!(!capture(&still).data.is_empty());
        let stop = Instant::now();
        let super::super::pipeline::CapturePipeline::PipeWire(fallback) = &pipeline else {
            panic!("PipeWire discovery must select the fallback pipeline");
        };
        fallback.stop().unwrap();
        fallback.stop().unwrap();
        drop(pipeline);
        assert!(stop.elapsed() < Duration::from_secs(2));
        assert!(PreviewMetrics::active_for(&device.path).is_none());
        insights.refresh_preview(&device.path, None, Instant::now());
        assert!(insights.full_pipeline_string.is_none());
        assert!(insights.copy_time_us.is_none());
        assert!(insights.dropped_frames.is_none());
        println!(
            "PipeWire backend: photo {}x{}, format change {}x{}, switch/restart + idempotent shutdown verified",
            small.width, small.height, larger.width, larger.height
        );
    }

    #[test]
    #[ignore = "requires a live PipeWire V4L2 camera"]
    fn live_shared_still_recording_backpressure_cancel_and_restart() {
        use futures::{FutureExt, StreamExt};
        let node = discover().into_iter().next().expect("PipeWire camera");
        let format = node
            .formats
            .iter()
            .find(|f| f.width == 640 && f.height == 480)
            .unwrap_or(&node.formats[0]);
        let (shared, mut receiver) = shared_state();
        let old_request = shared.still_capture.request();
        assert!(shared.still_capture.publish(
            shared.still_capture.pending().unwrap(),
            CameraFrame {
                width: 1,
                height: 1,
                data: FrameData::from_owned_vec(vec![0; 4]),
                format: PixelFormat::RGBA,
                stride: 4,
                yuv_planes: None,
                captured_at: Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: None,
            }
        ));
        let still_capture = Arc::clone(&shared.still_capture);
        let cancel = Arc::clone(&shared.cancel_flag);
        let (tx, mut recording) = tokio::sync::mpsc::channel(1);
        *shared.recording_sender.lock().unwrap() = Some(tx);
        let pipeline =
            super::super::pipeline::CapturePipeline::new(&node.device.path, format, false, shared)
                .unwrap();
        assert!(
            old_request.take().is_none(),
            "pipeline restart must not expose previous camera's still frame"
        );
        let request = still_capture.request();
        let deadline = Instant::now() + Duration::from_secs(2);
        let still = loop {
            if let Some(frame) = request.take() {
                break frame;
            }
            assert!(
                Instant::now() < deadline,
                "still capture blocked by full preview/recording queues"
            );
            std::thread::sleep(Duration::from_millis(10));
        };
        assert_eq!((still.width, still.height), (format.width, format.height));
        assert!(old_request.take().is_none());
        assert!(!still_capture.has_request());
        assert!(!recording.try_recv().unwrap().data.is_empty());
        assert!(
            receiver.try_recv().is_ok(),
            "preview arrives through its channel"
        );
        // Watch real delivery backpressure instead of inferring zero losses.
        let metrics = pipeline.preview_metrics();
        let deadline = Instant::now() + Duration::from_secs(2);
        while metrics.snapshot().dropped_frames == 0 {
            assert!(
                Instant::now() < deadline,
                "full preview queue must count drops"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        println!(
            "Live Insights: measured preview drops={}",
            metrics.snapshot().dropped_frames
        );
        cancel.store(true, Ordering::Release);
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if matches!(receiver.next().now_or_never(), Some(None)) {
                break;
            }
            assert!(Instant::now() < deadline, "cancel must close stream");
            std::thread::sleep(Duration::from_millis(10));
        }
        let stop = Instant::now();
        drop(pipeline);
        assert!(stop.elapsed() < Duration::from_secs(2));
        assert_eq!(
            still.data.len(),
            format.width as usize * format.height as usize * 4,
            "copied still survives source teardown"
        );
        let (shared, _receiver) = shared_state();
        let pipeline =
            super::super::pipeline::CapturePipeline::new(&node.device.path, format, false, shared)
                .expect("camera released for restart");
        drop(pipeline);
        println!(
            "PipeWire GUI shared path: still + recording under backpressure, cancellation + restart verified"
        );
    }
}
