// SPDX-License-Identifier: GPL-3.0-only

//! CLI commands for camera operations
//!
//! This module provides command-line functionality for:
//! - Listing available cameras
//! - Taking photos
//! - Recording videos

use camera::backends::camera::CameraBackend;
use camera::backends::camera::libcamera::{LibcameraBackend, create_pipeline};
use camera::backends::camera::types::{CameraFormat, CameraFrame, SensorRotation};
use camera::pipelines::capture_metadata::{CaptureMetadata, CaptureOrientation};
use camera::pipelines::photo::PhotoPipeline;
use camera::pipelines::video::{
    AppsrcRecorderConfig, EncoderConfig, RecorderConfig, VideoRecorder,
};
use chrono::Local;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn capture_metadata_for_rotation(rotation: SensorRotation) -> CaptureMetadata {
    CaptureMetadata {
        captured_at: Some(Local::now().fixed_offset()),
        orientation: CaptureOrientation::from_capture(rotation, false),
        ..Default::default()
    }
}

#[cfg(test)]
mod publication_tests {
    use super::*;

    #[test]
    fn requested_photo_destination_never_overwrites_a_concurrent_output() {
        let root = std::env::temp_dir().join(format!("cli-publication-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let source = root.join("IMG_20261008_140506_123.jpg");
        let destination = root.join("requested.jpg");
        std::fs::write(&source, b"complete new photo").unwrap();
        // Simulate an output appearing after take_photo's initial validation.
        std::fs::write(&destination, b"concurrent output").unwrap();
        assert_eq!(
            publish_requested_photo(&source, &destination)
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::AlreadyExists
        );
        assert_eq!(std::fs::read(&destination).unwrap(), b"concurrent output");
        assert_eq!(std::fs::read(&source).unwrap(), b"complete new photo");
        std::fs::remove_file(&destination).unwrap();
        publish_requested_photo(&source, &destination).unwrap();
        assert_eq!(std::fs::read(&destination).unwrap(), b"complete new photo");
        assert!(!source.exists());
        publish_requested_photo(&destination, &destination).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }
}

/// List all available cameras
pub fn list_cameras() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GStreamer
    gstreamer::init()?;

    let backend = LibcameraBackend::new();
    let cameras = backend.enumerate_cameras();

    if cameras.is_empty() {
        println!("No cameras found.");
        return Ok(());
    }

    println!("Available cameras:");
    println!();
    for (index, camera) in cameras.iter().enumerate() {
        println!("  [{}] {}", index, camera.name);

        // Get formats for this camera
        let formats = backend.get_formats(camera, false);
        if !formats.is_empty() {
            // Group formats by resolution and show best framerate (as integer for display)
            let mut resolutions: Vec<(u32, u32, u32)> = Vec::new();
            for format in &formats {
                let fps = format.framerate.map(|f| f.as_int()).unwrap_or(30);
                if let Some(existing) = resolutions
                    .iter_mut()
                    .find(|(w, h, _)| *w == format.width && *h == format.height)
                {
                    if fps > existing.2 {
                        existing.2 = fps;
                    }
                } else {
                    resolutions.push((format.width, format.height, fps));
                }
            }

            // Sort by resolution (highest first)
            resolutions.sort_by_key(|b| std::cmp::Reverse(b.0 * b.1));

            // Show top 3 resolutions
            let display_count = resolutions.len().min(3);
            let res_strs: Vec<String> = resolutions
                .iter()
                .take(display_count)
                .map(|(w, h, fps)| format!("{}x{}@{}fps", w, h, fps))
                .collect();

            println!("      Formats: {}", res_strs.join(", "));
        }
        println!();
    }

    Ok(())
}

/// Take a photo using the specified camera
pub fn take_photo(
    camera_index: usize,
    output: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Refuse to overwrite an existing file when the user has specified one.
    if let Some(path) = output.as_ref()
        && !path.is_dir()
        && path.exists()
    {
        return Err(format!(
            "Output file already exists: {} (refusing to overwrite)",
            path.display()
        )
        .into());
    }

    // Initialize GStreamer
    gstreamer::init()?;

    // Enumerate cameras
    let backend = LibcameraBackend::new();
    let cameras = backend.enumerate_cameras();
    if cameras.is_empty() {
        return Err("No cameras found".into());
    }

    if camera_index >= cameras.len() {
        return Err(format!(
            "Camera index {} out of range (0-{})",
            camera_index,
            cameras.len() - 1
        )
        .into());
    }

    let camera = &cameras[camera_index];
    println!("Using camera: {}", camera.name);

    // Get formats and select best one for photos (highest resolution)
    let formats = backend.get_formats(camera, false);
    if formats.is_empty() {
        return Err("No formats available for camera".into());
    }

    let format = select_photo_format(&formats);
    println!("Capture format: {}x{}", format.width, format.height);

    // Determine output path
    let output_dir = if let Some(path) = output.as_ref() {
        if path.is_dir() {
            path.clone()
        } else {
            path.parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(get_default_photo_dir)
        }
    } else {
        get_default_photo_dir()
    };

    // Ensure output directory exists
    std::fs::create_dir_all(&output_dir)?;

    // Start camera pipeline
    println!("Capturing...");
    let (_handle, mut receiver) =
        create_pipeline(camera, &format).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    // Wait for frames to stabilize (camera warm-up)
    let start = Instant::now();
    let timeout = Duration::from_secs(5);
    let warmup = Duration::from_millis(500);
    let mut frame: Option<CameraFrame> = None;

    while start.elapsed() < timeout {
        match receiver.try_recv() {
            Ok(f) => {
                frame = Some(f);
                // After warmup period, use the next good frame
                if start.elapsed() > warmup {
                    break;
                }
            }
            _ => {
                // No frame available yet, wait a bit
                std::thread::sleep(Duration::from_millis(16));
            }
        }
    }

    let frame = frame.ok_or("Failed to capture frame from camera")?;

    // Use photo pipeline to save the image
    let mut photo_pipeline = PhotoPipeline::new();
    photo_pipeline.set_camera_metadata(capture_metadata_for_rotation(camera.rotation));

    // Create async runtime for the pipeline
    let rt = tokio::runtime::Runtime::new()?;
    let output_path = rt.block_on(async {
        photo_pipeline
            .capture_and_save(Arc::new(frame), output_dir)
            .await
    })?;

    // If user specified a filename, atomically publish it without overwriting
    if let Some(user_path) = output
        && !user_path.is_dir()
    {
        publish_requested_photo(&output_path, &user_path)?;
        println!("Photo saved: {}", user_path.display());
        return Ok(());
    }

    println!("Photo saved: {}", output_path.display());
    Ok(())
}

fn publish_requested_photo(
    source: &std::path::Path,
    destination: &std::path::Path,
) -> std::io::Result<()> {
    if source == destination {
        return Ok(());
    }
    // Both paths are in the requested directory. Publish the already-synced
    // photo atomically without rename's destination-clobber race.
    std::fs::hard_link(source, destination)?;
    std::fs::remove_file(source)
}

/// Record a video using the specified camera
pub fn record_video(
    camera_index: usize,
    duration: u64,
    output: Option<PathBuf>,
    enable_audio: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Refuse to overwrite an existing file when the user has specified one.
    if let Some(path) = output.as_ref()
        && path.exists()
    {
        return Err(format!(
            "Output file already exists: {} (refusing to overwrite)",
            path.display()
        )
        .into());
    }

    // Initialize GStreamer
    gstreamer::init()?;

    // Enumerate cameras
    let backend = LibcameraBackend::new();
    let cameras = backend.enumerate_cameras();
    if cameras.is_empty() {
        return Err("No cameras found".into());
    }

    if camera_index >= cameras.len() {
        return Err(format!(
            "Camera index {} out of range (0-{})",
            camera_index,
            cameras.len() - 1
        )
        .into());
    }

    let camera = &cameras[camera_index];
    println!("Using camera: {}", camera.name);

    // Get formats and select best one for video
    let formats = backend.get_formats(camera, true);
    if formats.is_empty() {
        return Err("No formats available for camera".into());
    }

    let format = select_video_format(&formats);
    let framerate = format.framerate.map(|f| f.as_int()).unwrap_or(30);
    println!(
        "Recording format: {}x{} @ {}fps",
        format.width, format.height, framerate
    );

    // Determine output path
    let output_path = if let Some(path) = output {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        path
    } else {
        let dir = get_default_video_dir();
        std::fs::create_dir_all(&dir)?;
        let timestamp = Local::now().format("%Y%m%d_%H%M%S_%3f");
        dir.join(format!("video_{}.mp4", timestamp))
    };

    println!("Output: {}", output_path.display());
    println!("Duration: {} seconds", duration);
    if enable_audio {
        println!("Audio: enabled");
    }

    // Start camera pipeline
    let (handle, mut receiver) =
        create_pipeline(camera, &format).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    // Wait for first frame to determine pixel format
    let start = Instant::now();
    let timeout = Duration::from_secs(5);
    let first_frame = loop {
        if start.elapsed() > timeout {
            return Err("Timeout waiting for camera frame".into());
        }
        match receiver.try_recv() {
            Ok(f) => break f,
            _ => std::thread::sleep(Duration::from_millis(16)),
        }
    };

    let pixel_format = first_frame.format;
    let width = first_frame.width;
    let height = first_frame.height;

    // Create frame channel for appsrc recording
    let (frame_tx, frame_rx) = tokio::sync::mpsc::channel(15);

    // Set recording sender on the pipeline handle
    handle.set_recording_sender(Some(frame_tx));

    // Create encoder config and video recorder
    let encoder_config = EncoderConfig::default();
    let capture_metadata = capture_metadata_for_rotation(camera.rotation);

    let rt = tokio::runtime::Runtime::new()?;
    let recorder = rt.block_on(async {
        let rt_handle = tokio::runtime::Handle::current();
        tokio::task::spawn_blocking(move || {
            let _guard = rt_handle.enter();
            VideoRecorder::new_from_appsrc(
                AppsrcRecorderConfig {
                    base: RecorderConfig {
                        width,
                        height,
                        framerate,
                        output_path: output_path.clone(),
                        encoder_config,
                        enable_audio,
                        audio_device: None,
                        audio_source_rate_hz: 0,
                        encoder_info: None,
                        capture_metadata,
                        audio_levels: Default::default(),
                    },
                    pixel_format,
                    live_filter_code: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
                    live_zoom: Default::default(),
                },
                frame_rx,
            )
        })
        .await
        .unwrap_or_else(|e| Err(format!("Task join error: {}", e)))
    })?;

    // Start recording
    println!();
    println!("Recording... (press Ctrl+C to stop early)");
    recorder.start()?;

    // Set up Ctrl+C handler
    let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_flag_clone = stop_flag.clone();
    ctrlc::set_handler(move || {
        stop_flag_clone.store(true, std::sync::atomic::Ordering::SeqCst);
    })?;

    // Wait for duration or Ctrl+C
    let start = Instant::now();
    let target_duration = Duration::from_secs(duration);

    while start.elapsed() < target_duration {
        if stop_flag.load(std::sync::atomic::Ordering::SeqCst) {
            println!();
            println!("Stopping early...");
            break;
        }

        // Print progress
        let elapsed = start.elapsed().as_secs();
        print!("\rRecording: {:02}:{:02}", elapsed / 60, elapsed % 60);
        std::io::Write::flush(&mut std::io::stdout())?;

        std::thread::sleep(Duration::from_millis(100));
    }
    println!();

    // Stop recording: clear recording sender to trigger EOS
    handle.set_recording_sender(None);
    std::thread::sleep(Duration::from_millis(300)); // Let EOS propagate

    let final_path = recorder.stop()?;
    println!("Video saved: {}", final_path.display());

    Ok(())
}

/// Select the best format for photo capture (highest resolution)
fn select_photo_format(formats: &[CameraFormat]) -> CameraFormat {
    formats
        .iter()
        .max_by_key(|f| f.width * f.height)
        .cloned()
        .unwrap_or_else(|| formats[0].clone())
}

/// Select the best format for video recording (balanced resolution and framerate)
fn select_video_format(formats: &[CameraFormat]) -> CameraFormat {
    // Prefer 1080p at 30fps, otherwise highest resolution with reasonable framerate
    let target_height = 1080;
    let target_fps: u32 = 30;

    // First try to find exact match
    if let Some(format) = formats.iter().find(|f| {
        f.height == target_height
            && f.framerate
                .map(|fps| fps.as_int() >= target_fps)
                .unwrap_or(false)
    }) {
        return format.clone();
    }

    // Otherwise find closest to 1080p with at least 24fps
    formats
        .iter()
        .filter(|f| f.framerate.map(|fps| fps.as_int() >= 24).unwrap_or(false))
        .min_by_key(|f| {
            let height_diff = (f.height as i32 - target_height as i32).abs();
            let fps_int = f.framerate.map(|fps| fps.as_int()).unwrap_or(30);
            let fps_diff = (fps_int as i32 - target_fps as i32).abs();
            height_diff * 10 + fps_diff // Prioritize resolution over framerate
        })
        .cloned()
        .unwrap_or_else(|| formats[0].clone())
}

use camera::constants::DEFAULT_SAVE_FOLDER;

/// Get default photo directory
fn get_default_photo_dir() -> PathBuf {
    dirs::picture_dir()
        .unwrap_or_else(|| dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")))
        .join(DEFAULT_SAVE_FOLDER)
}

/// Get default video directory
fn get_default_video_dir() -> PathBuf {
    dirs::video_dir()
        .unwrap_or_else(|| dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")))
        .join(DEFAULT_SAVE_FOLDER)
}

/// Process images through the burst mode pipeline
pub fn process_burst_mode(
    input: Vec<PathBuf>,
    output: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    use camera::pipelines::photo::EncodingFormat;
    use camera::pipelines::photo::burst_mode::{
        BurstModeConfig, SaveOutputParams, process_burst_mode as run_burst_mode, save_output,
    };

    // Collect all image paths from input (can be files or directories)
    let image_paths = collect_image_paths(&input)?;

    if image_paths.is_empty() {
        return Err("No PNG or DNG images found in input".into());
    }

    println!("Burst Mode Processing");
    println!("=====================");
    println!("Found {} images to process", image_paths.len());

    // Determine output directory
    let output_dir = if let Some(dir) = output {
        dir
    } else if input.len() == 1 && input[0].is_dir() {
        input[0].join("output")
    } else if let Some(first) = image_paths.first() {
        first
            .parent()
            .map(|p| p.join("output"))
            .unwrap_or_else(get_default_photo_dir)
    } else {
        get_default_photo_dir()
    };

    println!("Output directory: {}", output_dir.display());
    println!();

    // Create output directory
    std::fs::create_dir_all(&output_dir)?;

    // Load all frames
    println!("Loading images...");
    let frames = load_burst_mode_frames(&image_paths)?;

    if frames.is_empty() {
        return Err("Failed to load any images".into());
    }

    println!("Loaded {} frames", frames.len());
    if let Some(first) = frames.first() {
        println!("Frame size: {}x{}", first.width, first.height);
    }
    println!();

    // Process through burst mode pipeline
    println!("Processing...");
    let config = BurstModeConfig::default();

    let rt = tokio::runtime::Runtime::new()?;
    let result = rt.block_on(async {
        let start = std::time::Instant::now();
        let result = run_burst_mode(frames, config.clone(), None).await?;
        let duration = start.elapsed();
        println!("Processing time: {:.2}s", duration.as_secs_f64());
        println!("Output size: {}x{}", result.width, result.height);
        Ok::<_, String>(result)
    })?;

    // Save output
    let camera_metadata = CaptureMetadata {
        camera_name: Some("Burst Mode CLI".to_string()),
        camera_driver: None,
        exposure_time: None,
        iso: None,
        gain: None,
        ..Default::default()
    };

    let output_path = rt.block_on(async {
        save_output(
            &result,
            SaveOutputParams {
                output_dir,
                crop_rect: None,
                encoding_format: EncodingFormat::Jpeg,
                camera_metadata,
                filter: None,
            },
        )
        .await
    })?;

    println!();
    println!("Saved to: {}", output_path.display());

    Ok(())
}

/// Collect all image paths from input (files or directories, recursive).
fn collect_image_paths(input: &[PathBuf]) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    fn visit_dir(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let file_path = entry.path();
            if file_path.is_dir() {
                visit_dir(&file_path, out)?;
            } else if is_supported_image(&file_path) {
                out.push(file_path);
            }
        }
        Ok(())
    }

    let mut paths = Vec::new();
    for path in input {
        if path.is_dir() {
            visit_dir(path, &mut paths)?;
        } else if is_supported_image(path) {
            paths.push(path.clone());
        }
    }

    // Sort by filename for consistent ordering
    paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    Ok(paths)
}

/// Check if a path is a supported image file
fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .map(|ext| {
            let ext_lower = ext.to_string_lossy().to_lowercase();
            ext_lower == "png" || ext_lower == "dng"
        })
        .unwrap_or(false)
}

/// Load burst mode frames from image paths
fn load_burst_mode_frames(
    paths: &[PathBuf],
) -> Result<Vec<Arc<CameraFrame>>, Box<dyn std::error::Error>> {
    use camera::backends::camera::types::{CameraFrame, FrameData, PixelFormat};
    use image::GenericImageView;

    let mut frames = Vec::new();

    for path in paths {
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();

        let img = if ext == "dng" {
            // DNG is TIFF-based but is not recognized by image::open's extension dispatch.
            let reader = std::io::BufReader::new(std::fs::File::open(path)?);
            image::DynamicImage::from_decoder(image::codecs::tiff::TiffDecoder::new(reader)?)?
        } else {
            image::open(path)?
        };
        let (width, height) = img.dimensions();
        let data = FrameData::from_copied(Arc::from(img.to_rgba8().into_raw().into_boxed_slice()));
        let frame = CameraFrame {
            width,
            height,
            data,
            format: PixelFormat::RGBA,
            stride: width * 4,
            yuv_planes: None,
            captured_at: Instant::now(),
            sensor_timestamp_ns: None,
            libcamera_metadata: None,
        };

        println!(
            "  Loaded: {} ({}x{})",
            path.file_name()
                .map(|n| n.to_string_lossy())
                .unwrap_or_default(),
            frame.width,
            frame.height
        );
        frames.push(Arc::new(frame));
    }

    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn burst_inputs_preserve_pixels_geometry_and_missing_metadata() {
        let root = std::env::temp_dir().join(format!("burst-inputs-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let pixels = [23, 42, 91, 255].repeat(6);
        let image = image::RgbaImage::from_raw(3, 2, pixels.clone()).unwrap();
        let paths = [root.join("input.PNG"), root.join("input.DNG")];
        for (path, format) in paths
            .iter()
            .zip([image::ImageFormat::Png, image::ImageFormat::Tiff])
        {
            image.save_with_format(path, format).unwrap();
        }
        let before = Instant::now();
        let frames = load_burst_mode_frames(&paths).unwrap();
        assert_eq!(frames.len(), 2);
        for frame in frames {
            assert_eq!((frame.width, frame.height, frame.stride), (3, 2, 12));
            assert_eq!(
                frame.format,
                camera::backends::camera::types::PixelFormat::RGBA
            );
            assert_eq!(&frame.data[..], pixels);
            assert!(frame.yuv_planes.is_none());
            assert!(frame.sensor_timestamp_ns.is_none());
            assert!(frame.libcamera_metadata.is_none());
            assert!(frame.captured_at >= before);
        }
        std::fs::write(&paths[0], b"not an image").unwrap();
        assert!(load_burst_mode_frames(&paths).is_err());
        assert!(load_burst_mode_frames(&[root.join("missing.dng")]).is_err());
        assert!(load_burst_mode_frames(&[]).unwrap().is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn recording_metadata_preserves_sensor_orientation() {
        let metadata = capture_metadata_for_rotation(SensorRotation::Rotate90);
        assert_eq!(metadata.orientation, CaptureOrientation::Rotate270);
    }
}
