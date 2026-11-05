// SPDX-License-Identifier: GPL-3.0-only
//! Encoded Motion Photo memory storage, independent of disk and UI state.
use super::video::{RecordingMonitor, RecordingPhase};
use gstreamer as gst;
use std::collections::VecDeque;

#[derive(Clone)]
struct Packet {
    buffer: gst::Buffer,
    caps: gst::Caps,
    pts: u64,
    audio: bool,
    key: bool,
}

struct EncodedRing {
    packets: VecDeque<Packet>,
    bytes: usize,
    max_bytes: usize,
    max_time: u64,
    newest: u64,
}
impl EncodedRing {
    fn new(max_bytes: usize, max_time: u64) -> Self {
        Self {
            packets: VecDeque::new(),
            bytes: 0,
            max_bytes,
            max_time,
            newest: 0,
        }
    }
    fn push(&mut self, packet: Packet) {
        self.bytes += packet.buffer.size();
        self.newest = self.newest.max(packet.pts);
        self.packets.push_back(packet);
        let oldest = self.newest.saturating_sub(self.max_time);
        self.packets.retain(|p| p.pts >= oldest);
        self.bytes = self.packets.iter().map(|p| p.buffer.size()).sum();
        while self.bytes > self.max_bytes {
            if let Some(old) = self.packets.pop_front() {
                self.bytes -= old.buffer.size();
            } else {
                break;
            }
        }
    }
}

use crate::media::encoders::video::EncoderInfo;
use gst::prelude::*;
use gstreamer_app as gst_app;
use std::sync::{Arc, Mutex};

pub(crate) const PRE_ROLL_NS: u64 = 1_250_000_000;
pub(crate) const POST_ROLL_NS: u64 = 1_250_000_000;
const MEMORY_LIMIT: usize = 32 * 1024 * 1024;
// Retain the whole shutter window until post-roll completes, plus bounded
// processing grace. No raw-frame history and no continuously written files.
const RETENTION_NS: u64 = 4_000_000_000;

pub(crate) struct MemoryEncoder {
    pipeline: gst::Pipeline,
    source: gst_app::AppSrc,
    ring: Arc<Mutex<EncodedRing>>,
    fps: u32,
    audio: bool,
    monitor: RecordingMonitor,
    _finish: super::video::recording_monitor::FinishRecording,
}
fn make(name: &str) -> Result<gst::Element, String> {
    gst::ElementFactory::make(name)
        .build()
        .map_err(|e| e.to_string())
}
fn memory_sink(
    ring: Arc<Mutex<EncodedRing>>,
    audio: bool,
    monitor: Option<RecordingMonitor>,
) -> gst_app::AppSink {
    // Only Gst 1.20 APIs: max-buffers/drop (not 1.24 max-bytes).
    let sink = gst_app::AppSink::builder()
        .sync(false)
        .max_buffers(2)
        .drop(true)
        .build();
    sink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .new_sample(move |sink| {
                let sample = sink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                let buffer = sample.buffer().ok_or(gst::FlowError::Error)?;
                let segment = sample
                    .segment()
                    .and_then(|s| s.downcast_ref::<gst::ClockTime>())
                    .ok_or(gst::FlowError::Error)?;
                let pts = segment
                    .to_running_time(buffer.pts())
                    .ok_or(gst::FlowError::Error)?
                    .nseconds();
                let caps = sample.caps().ok_or(gst::FlowError::Error)?.to_owned();
                let mut normalized = buffer.copy();
                let b = normalized.make_mut();
                b.set_pts(gst::ClockTime::from_nseconds(pts));
                b.set_dts(segment.to_running_time(buffer.dts()));
                let key = !buffer.flags().contains(gst::BufferFlags::DELTA_UNIT);
                ring.lock()
                    .map_err(|_| gst::FlowError::Error)?
                    .push(Packet {
                        buffer: normalized,
                        caps,
                        pts,
                        audio,
                        key,
                    });
                if let Some(monitor) = &monitor {
                    monitor.note_output();
                }
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );
    sink
}
// A sync observer must leave messages queued for the existing bus consumers.
fn observe_errors(pipeline: &gst::Pipeline, monitor: &RecordingMonitor) {
    let monitor = monitor.clone();
    if let Some(bus) = pipeline.bus() {
        bus.set_sync_handler(move |_, message| {
            if let gst::MessageView::Error(error) = message.view() {
                monitor.report_error(format!(
                    "Motion Photo pipeline failed: {} ({:?})",
                    error.error(),
                    error.debug()
                ));
            }
            gst::BusSyncReply::Pass
        });
    }
}

impl MemoryEncoder {
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        info: &EncoderInfo,
        width: u32,
        height: u32,
        fps: u32,
        bitrate: u32,
        audio: Option<&str>,
        origin: std::time::Instant,
    ) -> Result<Self, String> {
        Self::new_monitored(
            info,
            width,
            height,
            fps,
            bitrate,
            audio,
            origin,
            RecordingMonitor::new(info.element_name.clone()),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_monitored(
        info: &EncoderInfo,
        width: u32,
        height: u32,
        fps: u32,
        bitrate: u32,
        audio: Option<&str>,
        origin: std::time::Instant,
        monitor: RecordingMonitor,
    ) -> Result<Self, String> {
        monitor.set_encoder(info.element_name.clone());
        monitor.set_phase(RecordingPhase::Starting);
        monitor.enable_activity_watchdog();
        if let Err(error) = super::motion_photo_encoder::validate_encoder(info)
            .and_then(|()| gst::init().map_err(|e| e.to_string()))
        {
            monitor.report_error(error.clone());
            monitor.finish();
            return Err(error);
        }
        let pipeline = gst::Pipeline::new();
        observe_errors(&pipeline, &monitor);
        let ring = Arc::new(Mutex::new(EncodedRing::new(MEMORY_LIMIT, RETENTION_NS)));
        let source = gst_app::AppSrc::builder()
            .caps(
                &gst::Caps::builder("video/x-raw")
                    .field("format", "RGB")
                    .field("width", width as i32)
                    .field("height", height as i32)
                    .field("framerate", gst::Fraction::new(fps as i32, 1))
                    .build(),
            )
            .format(gst::Format::Time)
            .is_live(true)
            .block(false)
            .max_buffers(2)
            .build();
        source.set_property_from_str("leaky-type", "downstream");
        let this = Self {
            pipeline,
            source,
            ring,
            fps,
            audio: audio.is_some(),
            _finish: super::video::recording_monitor::FinishRecording(monitor.clone()),
            monitor,
        };
        let result: Result<(), String> = (|| {
            let encoder = make(&info.element_name)?;
            super::motion_photo_encoder::configure_clip_video_encoder(
                &encoder, info, width, height, bitrate,
            )?;
            if matches!(info.element_name.as_str(), "x264enc" | "x265enc") {
                encoder.set_property_from_str("tune", "zerolatency");
            }
            // Short GOPs bound keyframe rounding. Intra-only where supported avoids
            // B-frame dependencies and makes the 1.25s pre-roll accurate to one frame.
            for prop in [
                "key-int-max",
                "keyframe-max-dist",
                "gop-size",
                "keyframe-period",
            ] {
                if encoder.find_property(prop).is_some() {
                    encoder.set_property_from_str(prop, "1");
                }
            }
            for prop in ["bframes", "b-frames", "lag-in-frames"] {
                if encoder.find_property(prop).is_some() {
                    encoder.set_property_from_str(prop, "0");
                }
            }
            let input = super::video::encoder_selection::prepare_video_encoder_input(
                &encoder,
                &info.element_name,
                width & !1,
                height & !1,
                fps,
            )?;
            let convert = make("videoconvert")?;
            let scale = make("videoscale")?;
            let caps = gst::ElementFactory::make("capsfilter")
                .property("caps", input.caps(fps))
                .build()
                .map_err(|e| e.to_string())?;
            let parser = make(
                info.codec
                    .parser_name()
                    .ok_or("No MP4 parser for selected encoder")?,
            )?;
            let sink = memory_sink(this.ring.clone(), false, Some(this.monitor.clone()));
            let encoded_caps = match info.codec {
                crate::media::encoders::video::VideoCodec::H264 => {
                    gst::Caps::builder("video/x-h264")
                        .field("stream-format", "avc")
                        .field("alignment", "au")
                        .build()
                }
                crate::media::encoders::video::VideoCodec::HEVC => {
                    gst::Caps::builder("video/x-h265")
                        .field("stream-format", "hvc1")
                        .field("alignment", "au")
                        .build()
                }
                crate::media::encoders::video::VideoCodec::AV1 => gst::Caps::builder("video/x-av1")
                    .field("stream-format", "obu-stream")
                    .field("alignment", "tu")
                    .build(),
                crate::media::encoders::video::VideoCodec::VP8 => {
                    return Err("Motion Photo does not support VP8".into());
                }
            };
            sink.set_caps(Some(&encoded_caps));
            let chain = [
                this.source.upcast_ref(),
                &convert,
                &scale,
                &caps,
                &encoder,
                &parser,
                sink.upcast_ref(),
            ];
            this.pipeline.add_many(chain).map_err(|e| e.to_string())?;
            gst::Element::link_many(chain).map_err(|e| e.to_string())?;
            if let Some(device) = audio {
                let sink = memory_sink(this.ring.clone(), true, None);
                this.pipeline.add(&sink).map_err(|e| e.to_string())?;
                let audio = gst::ElementFactory::make("pulsesrc")
                    .property("device", device)
                    .build()
                    .map_err(|e| e.to_string())?;
                super::motion_photo_encoder::add_aac_branch(
                    &this.pipeline,
                    sink.upcast_ref(),
                    &audio,
                )?;
            }
            let clock = gst::SystemClock::obtain();
            this.pipeline.use_clock(Some(&clock));
            this.pipeline.set_start_time(gst::ClockTime::NONE);
            this.pipeline.set_base_time(clock.time().saturating_sub(
                gst::ClockTime::from_nseconds(origin.elapsed().as_nanos() as u64),
            ));
            this.pipeline
                .set_state(gst::State::Playing)
                .map_err(|e| e.to_string())?;
            Ok(())
        })();
        if let Err(error) = result {
            // Publish before `this` is dropped: STREAMOFF can block forever.
            this.monitor.report_error(error.clone());
            return Err(error);
        }
        this.monitor.set_phase(RecordingPhase::Recording);
        Ok(this)
    }
    pub(crate) fn push(&mut self, image: image::RgbImage, pts: u64) -> Result<(), String> {
        self.monitor.note_input();
        self.check_error()?;
        let row = image.width() as usize * 3;
        let stride = (row + 3) & !3;
        let mut bytes = vec![0; stride * image.height() as usize];
        for (src, dst) in image
            .as_raw()
            .chunks_exact(row)
            .zip(bytes.chunks_exact_mut(stride))
        {
            dst[..row].copy_from_slice(src);
        }
        let mut buffer = gst::Buffer::from_mut_slice(bytes);
        let b = buffer.make_mut();
        b.set_pts(gst::ClockTime::from_nseconds(pts));
        b.set_duration(gst::ClockTime::from_nseconds(
            1_000_000_000 / self.fps as u64,
        ));
        self.source
            .push_buffer(buffer)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
    pub(crate) fn check_error(&self) -> Result<(), String> {
        if let Some(message) = self
            .pipeline
            .bus()
            .and_then(|bus| bus.pop_filtered(&[gst::MessageType::Error]))
            && let gst::MessageView::Error(e) = message.view()
        {
            return Err(format!(
                "Motion Photo encoding failed: {} ({:?})",
                e.error(),
                e.debug()
            ));
        }
        Ok(())
    }
    pub(crate) fn snapshot_window(&self, shutter: u64) -> Result<EncodedWindow, String> {
        self.check_error()?;
        let packets: Vec<_> = self
            .ring
            .lock()
            .map_err(|e| e.to_string())?
            .packets
            .iter()
            .cloned()
            .collect();
        let lower = shutter.saturating_sub(PRE_ROLL_NS);
        let upper = shutter.saturating_add(POST_ROLL_NS);
        let start = packets.iter().filter(|p| !p.audio && p.key && p.pts >= lower && p.pts <= shutter)
            .map(|p| p.pts).min().ok_or("Motion Photo pre-roll has no decodable frame before shutter (warming or slow source)")?;
        let mut packets: Vec<_> = packets
            .into_iter()
            .filter(|p| p.pts >= start && p.pts < upper)
            .collect();
        packets.sort_by_key(|p| p.pts);
        if !packets.iter().any(|p| !p.audio && p.pts > shutter) {
            return Err("Motion Photo source provided no post-roll frame; still preserved".into());
        }
        if packets.iter().filter(|p| !p.audio).count() < 2 {
            return Err("Motion Photo received fewer than two frames".into());
        }
        if self.audio && !packets.iter().any(|p| p.audio) {
            return Err("Motion Photo audio enabled but no AAC packets received".into());
        }
        Ok(EncodedWindow {
            packets,
            start,
            shutter,
        })
    }
    #[cfg(test)]
    pub(crate) fn pre_roll_ready(&self, shutter: u64) -> bool {
        let packets = self.ring.lock().unwrap();
        let lower = shutter.saturating_sub(PRE_ROLL_NS);
        // Allow two 100ms fixture-frame intervals for asynchronous encoding.
        // Require both an early keyframe and recent video, not just raw input.
        shutter >= PRE_ROLL_NS
            && packets
                .packets
                .iter()
                .any(|p| !p.audio && p.key && p.pts >= lower && p.pts <= lower + 200_000_000)
            && packets
                .packets
                .iter()
                .any(|p| !p.audio && p.pts >= shutter.saturating_sub(200_000_000))
    }

    #[cfg(test)]
    fn write_window(&self, path: &std::path::Path, shutter: u64) -> Result<u64, String> {
        self.snapshot_window(shutter)?.write(path)
    }
}
impl Drop for MemoryEncoder {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.monitor
                .report_error("Motion Photo encoder worker panicked".into());
        }
        self.monitor.set_phase(RecordingPhase::Cleanup);
        if let Err(error) = self.pipeline.set_state(gst::State::Null) {
            self.monitor
                .report_error(format!("Motion Photo cleanup failed: {error}"));
        }
    }
}
pub(crate) struct EncodedWindow {
    packets: Vec<Packet>,
    start: u64,
    shutter: u64,
}
impl EncodedWindow {
    #[cfg(test)]
    pub(crate) fn write(self, path: &std::path::Path) -> Result<u64, String> {
        self.write_monitored(path, RecordingMonitor::new("Motion Photo clip mux".into()))
    }
    pub(crate) fn write_monitored(
        self,
        path: &std::path::Path,
        monitor: RecordingMonitor,
    ) -> Result<u64, String> {
        monitor.set_phase(RecordingPhase::Finalizing);
        let result = mux_packets(path, &self.packets, self.start, &monitor)
            .map(|()| self.shutter - self.start);
        if let Err(error) = &result {
            monitor.report_error(error.clone());
        }
        monitor.finish();
        result
    }
}
fn mux_packets(
    path: &std::path::Path,
    packets: &[Packet],
    start: u64,
    monitor: &RecordingMonitor,
) -> Result<(), String> {
    // This function is called only on a worker, and only after a shutter.
    let pipeline = gst::Pipeline::new();
    struct Stop(gst::Pipeline, RecordingMonitor);
    impl Drop for Stop {
        fn drop(&mut self) {
            if std::thread::panicking() {
                self.1
                    .report_error("Motion Photo mux worker panicked".into());
            }
            self.1.set_phase(RecordingPhase::Cleanup);
            if let Err(error) = self.0.set_state(gst::State::Null) {
                self.1
                    .report_error(format!("Motion Photo mux cleanup failed: {error}"));
            }
        }
    }
    let _stop = Stop(pipeline.clone(), monitor.clone());
    observe_errors(&pipeline, monitor);
    let result = (|| {
        let muxer = make("mp4mux")?;
        let sink = gst::ElementFactory::make("filesink")
            .property("location", path.to_str().ok_or("Invalid clip path")?)
            .build()
            .map_err(|e| e.to_string())?;
        pipeline
            .add_many([&muxer, &sink])
            .map_err(|e| e.to_string())?;
        muxer.link(&sink).map_err(|e| e.to_string())?;
        let mut sources = Vec::new();
        for audio in [false, true] {
            if let Some(first) = packets.iter().find(|p| p.audio == audio) {
                let source = gst_app::AppSrc::builder()
                    .caps(&first.caps)
                    .format(gst::Format::Time)
                    .block(false)
                    .build();
                pipeline.add(&source).map_err(|e| e.to_string())?;
                let pad = muxer
                    .request_pad_simple(if audio { "audio_%u" } else { "video_%u" })
                    .ok_or("No muxer track")?;
                source
                    .static_pad("src")
                    .ok_or("No appsrc pad")?
                    .link(&pad)
                    .map_err(|e| e.to_string())?;
                sources.push((audio, source));
            }
        }
        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| e.to_string())?;
        for packet in packets {
            let mut buffer = packet.buffer.clone();
            let b = buffer.make_mut();
            b.set_pts(gst::ClockTime::from_nseconds(packet.pts - start));
            b.set_dts(
                packet
                    .buffer
                    .dts()
                    .map(|d| gst::ClockTime::from_nseconds(d.nseconds().saturating_sub(start))),
            );
            sources
                .iter()
                .find(|(audio, _)| *audio == packet.audio)
                .unwrap()
                .1
                .push_buffer(buffer)
                .map_err(|e| e.to_string())?;
        }
        for (_, source) in sources {
            source.end_of_stream().map_err(|e| e.to_string())?;
        }
        match pipeline.bus().ok_or("No mux bus")?.timed_pop_filtered(
            gst::ClockTime::from_seconds(5),
            &[gst::MessageType::Eos, gst::MessageType::Error],
        ) {
            Some(m) => match m.view() {
                gst::MessageView::Eos(_) => Ok(()),
                gst::MessageView::Error(e) => Err(format!(
                    "Motion Photo mux failed: {} ({:?})",
                    e.error(),
                    e.debug()
                )),
                _ => unreachable!(),
            },
            None => Err("Motion Photo mux timed out".into()),
        }
    })();
    if let Err(error) = &result {
        monitor.report_error(error.clone());
    }
    result
}

#[cfg(test)]
pub(crate) mod test_fixture {
    use super::*;
    use std::{
        path::Path,
        time::{Duration, Instant},
    };

    // Observe encoded output, not a fixed sleep or an EOS flush. Keep the real
    // live queue's two-buffer, downstream-leaky policy in every fixture.
    fn wait_for_packet(encoder: &MemoryEncoder, ready: impl Fn(&Packet) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            encoder.check_error().unwrap();
            if encoder.ring.lock().unwrap().packets.iter().any(&ready) {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "live encoder did not deliver fixture packet: state={:?}, queued={}",
                encoder.pipeline.current_state(),
                encoder.source.current_level_buffers(),
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    pub(crate) fn push_frame(encoder: &mut MemoryEncoder, image: image::RgbImage, pts: u64) {
        encoder.push(image, pts).unwrap();
        wait_for_packet(encoder, |p| !p.audio && p.pts == pts);
    }

    pub(crate) fn add_synthetic_audio(encoder: &mut MemoryEncoder) {
        // Match production: assemble both branches before starting the graph.
        // Adding an async sink during video preroll can strand it in Paused.
        encoder.pipeline.set_state(gst::State::Null).unwrap();
        let sink = memory_sink(encoder.ring.clone(), true, None);
        encoder.pipeline.add(&sink).unwrap();
        let audio = gst::ElementFactory::make("audiotestsrc")
            .property("is-live", true)
            .build()
            .unwrap();
        super::super::motion_photo_encoder::add_aac_branch(
            &encoder.pipeline,
            sink.upcast_ref(),
            &audio,
        )
        .unwrap();
        sink.sync_state_with_parent().unwrap();
        encoder.audio = true;
        encoder.pipeline.set_state(gst::State::Playing).unwrap();
    }

    pub(crate) fn wait_for_audio(encoder: &MemoryEncoder, pts: u64) {
        wait_for_packet(encoder, |p| p.audio && p.pts >= pts);
    }

    pub(crate) fn software_encoder() -> EncoderInfo {
        EncoderInfo {
            element_name: "x264enc".into(),
            display_name: "H.264".into(),
            codec: crate::media::encoders::video::VideoCodec::H264,
            is_hardware: false,
            priority: 0,
        }
    }

    pub(crate) fn write_clip(
        path: &Path,
        info: &EncoderInfo,
        width: u32,
        height: u32,
        fps: u32,
        frames: u8,
    ) {
        assert!(frames >= 2);
        let mut encoder =
            MemoryEncoder::new(info, width, height, fps, 800, None, Instant::now()).unwrap();
        for i in 0..frames {
            push_frame(
                &mut encoder,
                image::RgbImage::from_pixel(width, height, image::Rgb([i * 30, 40, 90])),
                u64::from(i) * (1_000_000_000 / u64::from(fps)),
            );
            assert!(
                !path.exists(),
                "continuous recorder must not write before snapshot"
            );
        }
        // These short fixtures fit wholly inside the real snapshot's window.
        // Put the shutter between the first two independently supplied frames.
        let window = encoder
            .snapshot_window(500_000_000 / u64::from(fps))
            .unwrap();
        assert_eq!(window.start, 0);
        let expected_pts: Vec<u64> = (0..frames)
            .map(|i| u64::from(i) * (1_000_000_000 / u64::from(fps)))
            .collect();
        assert_eq!(
            window.packets.iter().map(|p| p.pts).collect::<Vec<_>>(),
            expected_pts
        );
        window.write(path).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_cleanup_remains_monitored_through_pipeline_destruction() {
        let monitor = RecordingMonitor::new("x264enc".into());
        let encoder = MemoryEncoder::new_monitored(
            &test_fixture::software_encoder(),
            64,
            48,
            10,
            800,
            None,
            std::time::Instant::now(),
            monitor.clone(),
        )
        .unwrap();
        let observed = Arc::new(Mutex::new(None));
        let at_destruction = observed.clone();
        let status = monitor.clone();
        let _notify = encoder.pipeline.add_weak_ref_notify(move || {
            *at_destruction.lock().unwrap() = Some(status.subscribe().borrow().phase);
        });
        drop(encoder);
        assert_eq!(*observed.lock().unwrap(), Some(RecordingPhase::Cleanup));
        assert_eq!(monitor.subscribe().borrow().phase, RecordingPhase::Finished);
    }

    #[tokio::test]
    async fn memory_activity_watchdog_tracks_real_input_and_encoded_video_progress() {
        use std::time::{Duration, Instant};
        let monitor = RecordingMonitor::new("x264enc".into());
        let mut encoder = MemoryEncoder::new_monitored(
            &test_fixture::software_encoder(),
            64,
            48,
            10,
            800,
            None,
            Instant::now(),
            monitor.clone(),
        )
        .unwrap();
        // A real encoded frame must acknowledge this earlier outstanding input.
        monitor.note_input();
        test_fixture::push_frame(&mut encoder, image::RgbImage::new(64, 48), 0);
        tokio::time::sleep(Duration::from_secs(3)).await;
        let pad = encoder.source.static_pad("src").unwrap();
        let probe = pad
            .add_probe(gst::PadProbeType::BUFFER, |_, _| gst::PadProbeReturn::Drop)
            .unwrap();
        let blocked = Instant::now();
        encoder
            .push(image::RgbImage::new(64, 48), 100_000_000)
            .unwrap();
        let result = tokio::time::timeout(
            Duration::from_secs(32),
            super::super::video::wait_for_recording_failure(monitor.subscribe()),
        )
        .await;
        pad.remove_probe(probe);
        drop(encoder);
        let failure = result
            .expect(
                "pending video must be independently observed without more input or watch updates",
            )
            .unwrap();
        assert!(
            failure.detail.contains("no encoded output"),
            "{}",
            failure.detail
        );
        assert!(
            blocked.elapsed() >= Duration::from_secs(29),
            "the earlier successful output must reset the progress deadline"
        );
    }

    // Characterization of the real state-transition boundary: no camera/driver
    // is needed to hold STREAMOFF-equivalent cleanup independently of the UI.
    #[test]
    fn memory_cleanup_stays_visible_until_pipeline_shutdown_returns() {
        let monitor = RecordingMonitor::new("x264enc".into());
        let mut encoder = MemoryEncoder::new_monitored(
            &test_fixture::software_encoder(),
            64,
            48,
            10,
            800,
            None,
            std::time::Instant::now(),
            monitor.clone(),
        )
        .unwrap();
        test_fixture::push_frame(&mut encoder, image::RgbImage::new(64, 48), 0);
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let release = Mutex::new(Some(release_rx));
        encoder
            .pipeline
            .bus()
            .unwrap()
            .set_sync_handler(move |_, message| {
                if let gst::MessageView::StateChanged(state) = message.view()
                    && state.current() == gst::State::Ready
                    && let Some(receiver) = release.lock().unwrap().take()
                {
                    let _ = entered_tx.send(());
                    let _ = receiver.recv();
                }
                gst::BusSyncReply::Pass
            });
        let worker = std::thread::spawn(move || drop(encoder));
        let entered = entered_rx.recv_timeout(std::time::Duration::from_secs(5));
        let during = monitor.subscribe().borrow().clone();
        let _ = release_tx.send(());
        worker.join().unwrap();
        assert!(
            entered.is_ok(),
            "cleanup must exercise a real blocking state transition"
        );
        assert_eq!(during.phase, RecordingPhase::Cleanup);
        assert_eq!(monitor.subscribe().borrow().phase, RecordingPhase::Finished);
    }

    #[test]
    fn memory_unwind_reports_failure_before_driver_cleanup() {
        let monitor = RecordingMonitor::new("x264enc".into());
        let observer = monitor.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _encoder = MemoryEncoder::new_monitored(
                &test_fixture::software_encoder(),
                64,
                48,
                10,
                800,
                None,
                std::time::Instant::now(),
                monitor,
            )
            .unwrap();
            panic!("synthetic worker panic");
        }));
        assert!(result.is_err());
        assert!(
            observer
                .subscribe()
                .borrow()
                .error
                .as_deref()
                .unwrap()
                .contains("panicked")
        );
        assert_eq!(
            observer.subscribe().borrow().phase,
            RecordingPhase::Finished
        );
    }

    #[test]
    fn memory_startup_failure_is_sticky_before_finished_is_published() {
        let mut info = test_fixture::software_encoder();
        info.element_name = "missing-camera-motion-encoder".into();
        let monitor = RecordingMonitor::new(info.element_name.clone());
        let result = MemoryEncoder::new_monitored(
            &info,
            64,
            48,
            10,
            800,
            None,
            std::time::Instant::now(),
            monitor.clone(),
        );
        assert!(result.is_err());
        let status = monitor.subscribe().borrow().clone();
        assert!(
            status.error.is_some(),
            "cleanup completion must not race ahead of startup failure"
        );
        assert_eq!(status.phase, RecordingPhase::Finished);
    }

    #[test]
    fn mux_monitor_reports_failure_and_finishes_after_cleanup() {
        gst::init().unwrap();
        let monitor = RecordingMonitor::new("x264enc".into());
        let window = EncodedWindow {
            packets: vec![],
            start: 0,
            shutter: 1,
        };
        let path = std::env::temp_dir()
            .join(uuid::Uuid::new_v4().to_string())
            .join("missing/clip.mp4");
        assert!(window.write_monitored(&path, monitor.clone()).is_err());
        let status = monitor.subscribe().borrow().clone();
        assert_eq!(status.encoder, "x264enc");
        assert!(status.error.is_some());
        assert_eq!(status.phase, RecordingPhase::Finished);
        assert!(!path.exists());
    }

    #[test]
    fn memory_monitor_observes_bus_error_without_consuming_it_and_finishes_after_drop() {
        let monitor = super::super::video::RecordingMonitor::new("x264enc".into());
        let encoder = MemoryEncoder::new_monitored(
            &test_fixture::software_encoder(),
            64,
            48,
            10,
            800,
            None,
            std::time::Instant::now(),
            monitor.clone(),
        )
        .unwrap();
        let message =
            gst::message::Error::builder(gst::CoreError::Failed, "synthetic encoder failure")
                .build();
        encoder.pipeline.bus().unwrap().post(message).unwrap();
        assert!(
            monitor
                .subscribe()
                .borrow()
                .error
                .as_deref()
                .unwrap()
                .contains("synthetic encoder failure")
        );
        assert!(
            encoder
                .check_error()
                .unwrap_err()
                .contains("synthetic encoder failure")
        );
        drop(encoder);
        assert_eq!(
            monitor.subscribe().borrow().phase,
            super::super::video::RecordingPhase::Finished
        );
    }

    #[test]
    fn synthetic_audio_fixture_starts_both_streams_without_more_video_input() {
        for _ in 0..8 {
            let mut encoder = MemoryEncoder::new(
                &test_fixture::software_encoder(),
                64,
                48,
                10,
                800,
                None,
                std::time::Instant::now(),
            )
            .unwrap();
            test_fixture::add_synthetic_audio(&mut encoder);
            test_fixture::push_frame(
                &mut encoder,
                image::RgbImage::from_pixel(64, 48, image::Rgb([80, 40, 90])),
                0,
            );
            test_fixture::wait_for_audio(&encoder, 0);
            assert_eq!(encoder.pipeline.current_state(), gst::State::Playing);
        }
    }

    #[test]
    fn live_memory_appsrc_drops_backlog_instead_of_retaining_unbounded_frames() {
        let encoder = MemoryEncoder::new(
            &test_fixture::software_encoder(),
            64,
            48,
            10,
            800,
            None,
            std::time::Instant::now(),
        )
        .unwrap();
        assert_eq!(
            encoder
                .source
                .property_value("leaky-type")
                .serialize()
                .unwrap()
                .as_str(),
            "downstream"
        );
        assert_eq!(encoder.source.property::<u64>("max-buffers"), 2);
        assert!(!encoder.source.property::<bool>("block"));
        assert!(encoder.source.property::<bool>("is-live"));
    }

    fn assert_continuous_codec_decodes_on_shutter_timeline(
        name: &str,
        codec: crate::media::encoders::video::VideoCodec,
        expected_codec: &str,
        expected_tag: &str,
    ) {
        let info = EncoderInfo {
            element_name: name.into(),
            display_name: name.into(),
            codec,
            is_hardware: false,
            priority: 0,
        };
        let path = std::env::temp_dir().join(format!(
            "memory-{name}-preroll-{}.mp4",
            uuid::Uuid::new_v4()
        ));
        let origin = std::time::Instant::now();
        let shutter = 4_000_000_000;
        let mut encoder = MemoryEncoder::new(&info, 64, 48, 10, 800, None, origin)
            .unwrap_or_else(|e| panic!("{name}: continuous encoder failed: {e}"));
        // Advance the input timeline beyond retention, without EOS or files.
        for i in 0..55 {
            let pts = i * 100_000_000;
            let color = if pts < shutter {
                [220, 10, 10]
            } else {
                [10, 10, 220]
            };
            test_fixture::push_frame(
                &mut encoder,
                image::RgbImage::from_pixel(64, 48, image::Rgb(color)),
                pts,
            );
            assert!(
                !path.exists(),
                "{name}: continuous ring must not write disk"
            );
        }
        assert_eq!(encoder.pipeline.current_state(), gst::State::Playing);
        assert!(
            encoder
                .ring
                .lock()
                .unwrap()
                .packets
                .iter()
                .all(|p| p.pts >= 1_400_000_000),
            "{name}: old packets must have been evicted"
        );
        let window = encoder
            .snapshot_window(shutter)
            .unwrap_or_else(|e| panic!("{name}: warmed snapshot failed: {e}"));
        // Independent frame grid: [4s - 1.25s, 4s + 1.25s) selects
        // input frames 28..=52. Never derive the oracle from snapshot output.
        let expected_pts: Vec<u64> = (28..=52).map(|i| i * 100_000_000).collect();
        let expected_times: Vec<f64> = (0..25).map(|i| f64::from(i) / 10.0).collect();
        assert!(window.packets.iter().all(|p| !p.audio));
        assert_eq!(window.start, 2_800_000_000);
        assert_eq!(
            window.packets.iter().map(|p| p.pts).collect::<Vec<_>>(),
            expected_pts
        );
        assert!(
            window.packets.first().unwrap().key,
            "{name}: window must start on a keyframe"
        );
        let midpoint = encoder
            .write_window(&path, shutter)
            .unwrap_or_else(|e| panic!("{name}: MP4 mux failed: {e}"))
            as f64
            / 1e9;
        assert!(
            (midpoint - 1.2).abs() < 0.000_001,
            "{name}: shutter must be centered: {midpoint}"
        );

        let probe = std::process::Command::new("timeout")
            .args([
                "15s",
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
            ])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            probe.status.success(),
            "{name}: ffprobe failed: {}",
            String::from_utf8_lossy(&probe.stderr)
        );
        let data: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap();
        let streams = data["streams"].as_array().unwrap();
        assert_eq!(streams.len(), 1);
        assert_eq!(streams[0]["codec_name"], expected_codec);
        assert_eq!(streams[0]["codec_tag_string"], expected_tag);
        assert_eq!(streams[0]["width"], 64);
        assert_eq!(streams[0]["height"], 48);
        let duration: f64 = data["format"]["duration"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap();
        assert!(
            (2.4..=2.6).contains(&duration),
            "{name}: duration={duration}"
        );
        let times: Vec<f64> = data["frames"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| {
                f["best_effort_timestamp_time"]
                    .as_str()
                    .unwrap()
                    .parse()
                    .unwrap()
            })
            .collect();
        assert_eq!(
            times.len(),
            expected_times.len(),
            "{name}: mux/decode must retain every snapshot frame"
        );
        for (actual, expected) in times.iter().zip(&expected_times) {
            assert!(
                (actual - expected).abs() < 0.001,
                "{name}: presentation timestamp {actual} differs from input grid {expected}"
            );
        }
        assert!(
            times.windows(2).all(|t| t[1] > t[0]),
            "{name}: non-monotonic presentation timestamps: {times:?}"
        );
        assert!(times[0] < 0.001);
        assert!(
            times.iter().any(|t| *t < midpoint - 1.0),
            "{name}: missing pre-roll presentation"
        );
        assert!(
            times.iter().any(|t| *t > midpoint + 1.0),
            "{name}: missing post-roll presentation"
        );

        let decode = std::process::Command::new("timeout")
            .args(["15s", "ffmpeg", "-v", "error", "-xerror", "-i"])
            .arg(&path)
            .args([
                "-map",
                "0:v:0",
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ])
            .output()
            .unwrap();
        assert!(
            decode.status.success(),
            "{name}: full decode failed: {}",
            String::from_utf8_lossy(&decode.stderr)
        );
        let (frames, remainder) = decode.stdout.as_chunks::<{ 64 * 48 * 3 }>();
        assert!(remainder.is_empty(), "{name}: truncated decoded frame");
        assert_eq!(
            frames.len(),
            times.len(),
            "{name}: decoded frames and presentation timestamps differ"
        );
        for (frame, time) in frames.iter().zip(&times) {
            let mut sums = [0u64; 3];
            for pixel in frame.as_chunks::<3>().0 {
                for channel in 0..3 {
                    sums[channel] += u64::from(pixel[channel]);
                }
            }
            let red = sums[0] as f64 / (64.0 * 48.0);
            let blue = sums[2] as f64 / (64.0 * 48.0);
            if *time < midpoint - 0.001 {
                assert!(
                    red > 180.0 && blue < 50.0,
                    "{name}: BEFORE content wrong at {time}s: red={red} blue={blue}"
                );
            } else {
                assert!(
                    blue > 180.0 && red < 50.0,
                    "{name}: AFTER content wrong at {time}s: red={red} blue={blue}"
                );
            }
        }
        eprintln!(
            "continuous {name}: {} decoded frames, {duration}s, shutter {midpoint}s, presentation {}..{}s, BEFORE/AFTER verified",
            frames.len(),
            times[0],
            times.last().unwrap()
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn hevc_memory_packets_omit_encoder_info_sei() {
        let info = EncoderInfo {
            element_name: "x265enc".into(),
            display_name: "HEVC".into(),
            codec: crate::media::encoders::video::VideoCodec::HEVC,
            is_hardware: false,
            priority: 0,
        };
        let mut encoder =
            MemoryEncoder::new(&info, 64, 48, 10, 800, None, std::time::Instant::now()).unwrap();
        for i in 0..3 {
            test_fixture::push_frame(
                &mut encoder,
                image::RgbImage::from_pixel(64, 48, image::Rgb([220, 10, 10])),
                i * 100_000_000,
            );
        }
        let ring = encoder.ring.lock().unwrap();
        assert_eq!(ring.packets.len(), 3);
        for packet in &ring.packets {
            let bytes = packet.buffer.map_readable().unwrap();
            assert!(
                !bytes.windows(4).any(|s| s == b"x265"),
                "HEVC packets must omit the optional encoder-info SEI that older x265 plugins corrupt"
            );
        }
    }

    #[test]
    fn continuous_hevc_ring_decodes_before_and_after_on_shutter_timeline() {
        assert_continuous_codec_decodes_on_shutter_timeline(
            "x265enc",
            crate::media::encoders::video::VideoCodec::HEVC,
            "hevc",
            "hvc1",
        );
    }
    #[test]
    fn continuous_av1_ring_decodes_before_and_after_on_shutter_timeline() {
        assert_continuous_codec_decodes_on_shutter_timeline(
            "av1enc",
            crate::media::encoders::video::VideoCodec::AV1,
            "av1",
            "av01",
        );
    }

    #[test]
    fn continuous_memory_clip_decodes_before_and_after_with_centered_shutter() {
        use crate::media::encoders::video::{EncoderInfo, VideoCodec};
        let info = EncoderInfo {
            element_name: "x264enc".into(),
            display_name: "H264".into(),
            codec: VideoCodec::H264,
            is_hardware: false,
            priority: 0,
        };
        let origin = std::time::Instant::now();
        let path =
            std::env::temp_dir().join(format!("memory-preroll-{}.mp4", uuid::Uuid::new_v4()));
        let mut encoder = MemoryEncoder::new(&info, 64, 48, 10, 800, None, origin).unwrap();
        for i in 0..40 {
            let color = if i < 25 { [220, 10, 10] } else { [10, 10, 220] };
            test_fixture::push_frame(
                &mut encoder,
                image::RgbImage::from_pixel(64, 48, image::Rgb(color)),
                i * 100_000_000,
            );
            assert!(!path.exists(), "continuous recording must never write disk");
        }
        let midpoint = encoder.write_window(&path, 2_500_000_000).unwrap();
        assert_eq!(
            midpoint, 1_200_000_000,
            "input grid selects 1.3s as first frame"
        );
        let decode = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-xerror", "-i"])
            .arg(&path)
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .output()
            .unwrap();
        assert!(
            decode.status.success(),
            "{}",
            String::from_utf8_lossy(&decode.stderr)
        );
        let frames = decode.stdout.as_chunks::<{ 64 * 48 * 3 }>().0;
        assert!(
            frames.len() >= 23,
            "nominal clip should span 2.5 seconds, got {} frames",
            frames.len()
        );
        assert!(
            frames.first().unwrap()[0] > 180,
            "BEFORE red content missing"
        );
        assert!(
            frames.last().unwrap()[2] > 180,
            "AFTER blue content missing"
        );
        let probe = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
            ])
            .arg(&path)
            .output()
            .unwrap();
        let data: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap();
        let duration: f64 = data["format"]["duration"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap();
        assert!((2.3..=2.6).contains(&duration), "duration={duration}");
        eprintln!(
            "memory clip: {} frames, {duration}s, shutter {}s, decoded BEFORE/AFTER",
            frames.len(),
            midpoint as f64 / 1e9
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn continuous_aac_has_preroll_and_postroll_on_video_timeline() {
        use crate::media::encoders::video::{EncoderInfo, VideoCodec};
        let info = EncoderInfo {
            element_name: "x264enc".into(),
            display_name: "H264".into(),
            codec: VideoCodec::H264,
            is_hardware: false,
            priority: 0,
        };
        let origin = std::time::Instant::now();
        let path = std::env::temp_dir().join(format!("memory-aac-{}.mp4", uuid::Uuid::new_v4()));
        let mut encoder = MemoryEncoder::new(&info, 64, 48, 10, 800, None, origin).unwrap();
        test_fixture::add_synthetic_audio(&mut encoder);
        let mut last_pts = 0;
        for _ in 0..40 {
            last_pts = origin.elapsed().as_nanos() as u64;
            test_fixture::push_frame(
                &mut encoder,
                image::RgbImage::from_pixel(64, 48, image::Rgb([80, 40, 90])),
                last_pts,
            );
            // Real capture cadence keeps live synthetic audio aligned.
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        test_fixture::wait_for_audio(&encoder, last_pts);
        let midpoint = encoder.write_window(&path, 2_500_000_000).unwrap() as f64 / 1e9;
        let probe = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_packets",
                "-show_streams",
                "-of",
                "json",
            ])
            .arg(&path)
            .output()
            .unwrap();
        assert!(probe.status.success());
        let data: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap();
        let streams = data["streams"].as_array().unwrap();
        let audio_stream = streams.iter().find(|s| s["codec_type"] == "audio").unwrap();
        assert_eq!(audio_stream["codec_name"], "aac");
        assert_eq!(audio_stream["sample_rate"], "48000");
        assert_eq!(audio_stream["channels"], 1);
        let index = audio_stream["index"].as_u64().unwrap();
        let times: Vec<f64> = data["packets"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p["stream_index"].as_u64() == Some(index))
            .map(|p| p["pts_time"].as_str().unwrap().parse().unwrap())
            .collect();
        assert!(
            times[0] < 0.05,
            "AAC must begin alongside pre-roll video: {times:?}"
        );
        assert!(times.iter().any(|t| *t < midpoint - 1.0));
        assert!(times.iter().any(|t| *t > midpoint + 1.0));
        let decode = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-xerror", "-i"])
            .arg(&path)
            .args(["-map", "0", "-f", "null", "-"])
            .output()
            .unwrap();
        assert!(
            decode.status.success(),
            "{}",
            String::from_utf8_lossy(&decode.stderr)
        );
        eprintln!(
            "continuous AAC: first {}s last {}s shutter {midpoint}s, all streams decoded",
            times[0],
            times.last().unwrap()
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn missing_postroll_is_reported_instead_of_publishing_a_preroll_only_clip() {
        use crate::media::encoders::video::{EncoderInfo, VideoCodec};
        let info = EncoderInfo {
            element_name: "x264enc".into(),
            display_name: "H264".into(),
            codec: VideoCodec::H264,
            is_hardware: false,
            priority: 0,
        };
        let encoder =
            MemoryEncoder::new(&info, 64, 48, 10, 800, None, std::time::Instant::now()).unwrap();
        for pts in [1_300_000_000, 1_400_000_000, 1_500_000_000] {
            encoder.ring.lock().unwrap().push(Packet {
                buffer: gst::Buffer::from_mut_slice(vec![0; 10]),
                caps: gst::Caps::new_empty(),
                pts,
                audio: false,
                key: true,
            });
        }
        assert!(
            encoder.snapshot_window(2_500_000_000).is_err(),
            "a stopped source must preserve the still and report missing AFTER content"
        );
    }

    #[test]
    fn pre_roll_readiness_requires_an_early_keyframe_and_recent_video() {
        use crate::media::encoders::video::VideoCodec;
        let info = EncoderInfo {
            element_name: "x264enc".into(),
            display_name: "H264".into(),
            codec: VideoCodec::H264,
            is_hardware: false,
            priority: 0,
        };
        let encoder =
            MemoryEncoder::new(&info, 64, 48, 10, 800, None, std::time::Instant::now()).unwrap();
        let shutter = 2_500_000_000;
        assert!(!encoder.pre_roll_ready(shutter));
        for (pts, audio, key, ready) in [
            (1_300_000_000, false, false, false),
            (2_400_000_000, false, false, false),
            (1_300_000_000, true, true, false),
            (1_300_000_000, false, true, true),
        ] {
            encoder.ring.lock().unwrap().push(Packet {
                buffer: gst::Buffer::from_mut_slice(vec![0; 10]),
                caps: gst::Caps::new_empty(),
                pts,
                audio,
                key,
            });
            assert_eq!(encoder.pre_roll_ready(shutter), ready);
        }
        assert!(!encoder.pre_roll_ready(1_000_000_000));
        assert!(
            !encoder.pre_roll_ready(3_000_000_000),
            "stale output must not satisfy the warm-up barrier"
        );
    }

    #[test]
    fn late_encoded_video_cannot_expand_time_bound_behind_newer_audio() {
        gst::init().unwrap();
        let mut ring = EncodedRing::new(1000, 100);
        for (pts, audio) in [(1000, true), (0, false)] {
            ring.push(Packet {
                buffer: gst::Buffer::from_mut_slice(vec![0; 10]),
                caps: gst::Caps::new_empty(),
                pts,
                audio,
                key: true,
            });
        }
        assert!(
            ring.packets.iter().all(|p| p.pts >= 900),
            "audio/video interleaving must share a monotonic eviction watermark"
        );
    }

    #[test]
    fn encoded_ring_enforces_byte_and_time_limits() {
        gst::init().unwrap();
        let mut ring = EncodedRing::new(30, 100);
        for pts in (0..1000).step_by(10) {
            ring.push(Packet {
                buffer: gst::Buffer::from_mut_slice(vec![0; 10]),
                caps: gst::Caps::new_empty(),
                pts,
                audio: false,
                key: true,
            });
            assert!(ring.bytes <= 30, "encoded RAM must be bounded");
            assert!(ring.packets.back().unwrap().pts - ring.packets.front().unwrap().pts <= 100);
        }
        let mut ring = EncodedRing::new(10000, 100);
        for pts in (0..1000).step_by(10) {
            ring.push(Packet {
                buffer: gst::Buffer::from_mut_slice(vec![0; 10]),
                caps: gst::Caps::new_empty(),
                pts,
                audio: false,
                key: true,
            });
        }
        assert!(ring.packets.back().unwrap().pts - ring.packets.front().unwrap().pts <= 100);
    }
}
