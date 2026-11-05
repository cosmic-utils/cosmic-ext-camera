// SPDX-License-Identifier: GPL-3.0-only
//! Selected-codec configuration shared by the continuous memory recorder.
use crate::media::encoders::video::{EncoderInfo, VideoQuality, configure_video_encoder};
use gst::prelude::*;
use gstreamer as gst;
#[cfg(test)]
use std::path::Path;

pub(crate) fn validate_encoder(info: &EncoderInfo) -> Result<(), String> {
    if info.codec == crate::media::encoders::video::VideoCodec::VP8
        || info.element_name.contains("vp8")
        || info.element_name.contains("vp9")
    {
        return Err(
            "Motion Photo requires AVC, HEVC or AV1 in MP4; select a compatible video encoder"
                .into(),
        );
    }
    Ok(())
}

pub(super) fn configure_clip_video_encoder(
    encoder: &gst::Element,
    info: &EncoderInfo,
    width: u32,
    height: u32,
    bitrate: u32,
) -> Result<(), String> {
    configure_video_encoder(
        encoder,
        &info.element_name,
        VideoQuality::Low,
        width,
        height,
        Some(bitrate),
    );
    if info.element_name == "x265enc" {
        // Older x265 plugins can corrupt the optional encoder-info SEI when
        // fetching headers after encoding the first frame. h265parse then
        // splits the AU and loses its PTS. Omit only this diagnostic payload;
        // keep the codec headers, image quality and shutter timeline intact.
        encoder.set_property("option-string", "info=0");
    }
    if info.element_name.starts_with("v4l2") {
        if encoder.find_property("extra-controls").is_none() {
            return Err(
                "Selected V4L2 encoder cannot apply the Motion Photo bitrate preset".into(),
            );
        }
        let controls = gst::Structure::builder("controls")
            .field(
                "video_bitrate",
                bitrate.saturating_mul(1000).min(i32::MAX as u32) as i32,
            )
            .build();
        encoder.set_property("extra-controls", controls);
    }
    Ok(())
}

pub(super) fn add_aac_branch(
    pipeline: &gst::Pipeline,
    muxer: &gst::Element,
    audio: &gst::Element,
) -> Result<(), String> {
    let make = |name: &str| {
        gst::ElementFactory::make(name)
            .build()
            .map_err(|e| e.to_string())
    };
    let convert = make("audioconvert")?;
    let resample = make("audioresample")?;
    // Motion Photo permits mono/stereo AAC at 44.1, 48 or 96 kHz.
    // Pin a compliant layout instead of inheriting arbitrary device caps.
    let caps = gst::ElementFactory::make("capsfilter")
        .property(
            "caps",
            gst::Caps::builder("audio/x-raw")
                .field("rate", 48_000i32)
                .field("channels", 1i32)
                .build(),
        )
        .build()
        .map_err(|e| e.to_string())?;
    let encoder = ["fdkaacenc", "voaacenc", "avenc_aac"]
        .into_iter()
        .find_map(|name| make(name).ok())
        .ok_or("Motion Photo audio requires an AAC encoder")?;
    let parse = make("aacparse")?;
    let queue = make("queue")?;
    let chain = [audio, &convert, &resample, &caps, &encoder, &parse, &queue];
    pipeline.add_many(chain).map_err(|e| e.to_string())?;
    gst::Element::link_many(chain).map_err(|e| e.to_string())?;
    queue.link(muxer).map_err(|e| e.to_string())?;
    for element in chain {
        element
            .sync_state_with_parent()
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::encoders::video::{EncoderInfo, VideoCodec};
    use crate::pipelines::motion_photo_ring::{
        MemoryEncoder,
        test_fixture::{
            add_synthetic_audio, push_frame, software_encoder, wait_for_audio, write_clip,
        },
    };

    #[test]
    fn vp8_is_rejected_for_motion_photos_by_codec() {
        let info = EncoderInfo {
            element_name: "custom-hardware-encoder".into(),
            display_name: "VP8".into(),
            codec: VideoCodec::VP8,
            is_hardware: true,
            priority: 46,
        };
        assert!(validate_encoder(&info).is_err());
    }

    /// Recover the appended MP4 using the published directory, not the input size.
    fn check_packaged_clip_decodes_in_both_still_formats(clip: &Path) {
        let root =
            std::env::temp_dir().join(format!("motion-photo-package-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let pixels = image::RgbImage::from_pixel(64, 48, image::Rgb([60, 40, 90]));
        for ext in ["jpg", "avif"] {
            let still = root.join(format!("still.{ext}"));
            pixels.save(&still).unwrap();
            let packaged =
                crate::pipelines::motion_photo::embed_motion_photo(&still, clip, -1).unwrap();
            let bytes = std::fs::read(&packaged).unwrap();
            let start = bytes
                .windows(b"<x:xmpmeta".len())
                .position(|w| w == b"<x:xmpmeta")
                .unwrap();
            let end = bytes[start..]
                .windows(b"</x:xmpmeta>".len())
                .position(|w| w == b"</x:xmpmeta>")
                .unwrap()
                + start
                + b"</x:xmpmeta>".len();
            let xmp = std::str::from_utf8(&bytes[start..end]).unwrap();
            // Parse only the MotionPhoto item's directory attributes, independently
            // of the source clip and of JPEG/AVIF container offsets.
            let motion = xmp
                .split("<Container:Item ")
                .map(|item| item.split("/>").next().unwrap())
                .find(|item| item.contains("Item:Semantic=\"MotionPhoto\""))
                .unwrap();
            let length: usize = motion
                .split("Item:Length=\"")
                .nth(1)
                .unwrap()
                .split('"')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            assert!(length > 0 && length < bytes.len());
            let extracted = root.join(format!("recovered-{ext}.mp4"));
            let recovered = &bytes[bytes.len() - length..];
            assert_eq!(&recovered[4..8], b"ftyp");
            std::fs::write(&extracted, recovered).unwrap();
            let decode = std::process::Command::new("timeout")
                .args(["15s", "ffmpeg", "-v", "error", "-xerror", "-i"])
                .arg(&extracted)
                .args(["-map", "0", "-f", "null", "-"])
                .output()
                .unwrap();
            assert!(
                decode.status.success(),
                "{} + {ext}: {}",
                clip.display(),
                String::from_utf8_lossy(&decode.stderr)
            );
            image::open(&packaged).unwrap();
            eprintln!(
                "packaged {} + {ext}: recovered {length} bytes via MotionPhoto Item:Length; ffmpeg decoded all streams",
                clip.display()
            );
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn delayed_start_keeps_audio_on_the_video_shutter_timeline() {
        check_delayed_start_timestamps(false);
    }

    #[test]
    fn startup_buffered_frame_keeps_its_lead_over_later_audio() {
        check_delayed_start_timestamps(true);
    }

    fn check_delayed_start_timestamps(buffered: bool) {
        // Model 600ms of processing/build latency without sleeping to guess
        // pipeline readiness. Frame timestamps remain on the capture epoch.
        let origin = std::time::Instant::now() - std::time::Duration::from_millis(600);
        let buffered_pts = 0;
        let path =
            std::env::temp_dir().join(format!("motion-photo-clock-{}.mp4", uuid::Uuid::new_v4()));
        let mut encoder =
            MemoryEncoder::new(&software_encoder(), 64, 48, 10, 800, None, origin).unwrap();
        let audio_started = origin.elapsed().as_secs_f64();
        add_synthetic_audio(&mut encoder);
        let mut captured = Vec::new();
        if buffered {
            captured.push(0.0);
            push_frame(
                &mut encoder,
                image::RgbImage::from_pixel(64, 48, image::Rgb([60, 40, 90])),
                buffered_pts,
            );
        }
        for _ in 0..6 {
            let pts = origin.elapsed().as_nanos() as u64;
            captured.push(pts as f64 / 1e9);
            push_frame(
                &mut encoder,
                image::RgbImage::from_pixel(64, 48, image::Rgb([60, 40, 90])),
                pts,
            );
            // Capture cadence, not an asynchronous encoder-readiness barrier.
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let last_pts = (captured.last().unwrap() * 1e9) as u64;
        wait_for_audio(&encoder, last_pts);
        encoder
            .snapshot_window(last_pts - 50_000_000)
            .unwrap()
            .write(&path)
            .unwrap();
        let probe = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_packets",
                "-show_entries",
                "packet=stream_index,pts_time",
                "-of",
                "json",
            ])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            probe.status.success(),
            "ffprobe failed for {} ({}): {}",
            path.display(),
            probe.status,
            String::from_utf8_lossy(&probe.stderr)
        );
        let data: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap();
        let packets = data["packets"].as_array().unwrap();
        let times = |stream| {
            packets
                .iter()
                .filter(|p| p["stream_index"] == stream)
                .map(|p| p["pts_time"].as_str().unwrap().parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        };
        let mut video = times(0);
        video.sort_by(f64::total_cmp);
        let audio = times(1);
        std::fs::remove_file(path).unwrap();
        eprintln!(
            "delayed-start packet PTS: video={video:?}, audio first={:?}",
            audio.first()
        );
        let expected_audio_lead = audio_started - captured[0];
        assert!(
            (audio[0] - video[0] - expected_audio_lead).abs() < 0.08,
            "audio and video must share the shutter epoch, preserving buffering offset: expected={expected_audio_lead}"
        );
        // mp4mux may normalize the common epoch to the earliest stream PTS.
        let first_capture = captured[0];
        assert_eq!(video.len(), captured.len());
        for (actual, expected) in video.iter().zip(captured) {
            assert!(
                (actual - (expected - first_capture)).abs() < 0.03,
                "actual capture timing must survive startup"
            );
        }
    }

    #[test]
    fn synthetic_audio_is_aac_in_mp4_without_opening_a_microphone() {
        let path = std::env::temp_dir().join(format!(
            "motion-photo-aac-test-{}.mp4",
            uuid::Uuid::new_v4()
        ));
        let origin = std::time::Instant::now();
        let mut encoder =
            MemoryEncoder::new(&software_encoder(), 64, 48, 10, 800, None, origin).unwrap();
        add_synthetic_audio(&mut encoder);
        for i in 0..6 {
            push_frame(
                &mut encoder,
                image::RgbImage::from_pixel(64, 48, image::Rgb([60, 40, 90])),
                i * 100_000_000,
            );
        }
        wait_for_audio(&encoder, 500_000_000);
        encoder
            .snapshot_window(50_000_000)
            .unwrap()
            .write(&path)
            .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.windows(4).any(|x| x == b"mp4a"));
        assert!(!bytes.windows(4).any(|x| x == b"Opus"));
        let probe = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels",
                "-of",
                "json",
            ])
            .arg(&path)
            .output()
            .unwrap();
        assert!(probe.status.success());
        let streams: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap();
        assert_eq!(streams["streams"][0]["sample_rate"], "48000");
        assert_eq!(streams["streams"][0]["channels"], 1);
        check_packaged_clip_decodes_in_both_still_formats(&path);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn selected_hevc_and_av1_remain_in_mp4_containers() {
        for (name, codec, sample) in [
            ("x265enc", VideoCodec::HEVC, b"hvc1"),
            ("av1enc", VideoCodec::AV1, b"av01"),
        ] {
            let path = std::env::temp_dir().join(format!(
                "motion-photo-codec-test-{}.mp4",
                uuid::Uuid::new_v4()
            ));
            let info = EncoderInfo {
                element_name: name.into(),
                codec,
                ..software_encoder()
            };
            write_clip(&path, &info, 64, 48, 10, 3);
            let bytes = std::fs::read(&path).unwrap();
            assert_eq!(&bytes[4..8], b"ftyp");
            assert!(bytes.windows(4).any(|x| x == sample));
            check_packaged_clip_decodes_in_both_still_formats(&path);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn v4l2_without_bitrate_controls_fails_instead_of_silently_ignoring_preset() {
        gst::init().unwrap();
        let element = gst::ElementFactory::make("identity").build().unwrap();
        let info = EncoderInfo {
            element_name: "v4l2h264enc".into(),
            ..software_encoder()
        };
        let result = configure_clip_video_encoder(&element, &info, 64, 48, 800);
        assert!(matches!(result, Err(ref e) if e.contains("bitrate")));
    }

    #[test]
    fn incompatible_explicit_encoder_is_rejected_without_substitution() {
        let mut info = software_encoder();
        info.element_name = "vp9enc".into();
        let result = MemoryEncoder::new(&info, 64, 48, 10, 800, None, std::time::Instant::now());
        assert!(matches!(result, Err(ref e) if e.contains("AVC, HEVC or AV1")));
    }

    #[test]
    fn synthetic_frames_produce_a_real_mp4_with_selected_encoder() {
        gstreamer::init().unwrap();
        let path =
            std::env::temp_dir().join(format!("motion-photo-test-{}.mp4", uuid::Uuid::new_v4()));
        write_clip(&path, &software_encoder(), 64, 48, 10, 6);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[4..8], b"ftyp");
        assert!(bytes.windows(4).any(|x| x == b"moov"));
        assert!(bytes.windows(4).any(|x| x == b"avc1"));
        std::fs::remove_file(path).unwrap();
    }
}
