// SPDX-License-Identifier: GPL-3.0-only

use camera::app::AppModel;
use camera::i18n;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod cli;

const WINDOW_MIN_WIDTH: f32 = 360.0;
const WINDOW_MIN_HEIGHT: f32 = 180.0;

#[derive(Parser)]
#[command(name = "camera")]
#[command(about = "Modern camera app for Linux desktops and phones")]
#[command(version)]
#[command(subcommand_required = false)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Use an image or video file as the camera preview source instead of a real camera.
    /// Useful for testing, demos, or taking screenshots with consistent content.
    /// Supported formats: PNG, JPG, JPEG, WEBP (images) or MP4, WEBM, MKV (videos)
    #[arg(long, value_name = "FILE")]
    preview_source: Option<PathBuf>,

    /// Override the preview-mode window size, formatted as `WIDTHxHEIGHT`
    /// (e.g. `400x880` for a modern Linux-phone aspect). Only takes effect
    /// alongside `--preview-source`. Defaults to 900x700.
    #[arg(long, value_name = "WIDTHxHEIGHT", value_parser = parse_window_size)]
    preview_window: Option<(f32, f32)>,

    /// Preview harness only: boot straight into Video mode showing an active
    /// recording indicator, without starting the encoder. Lets the screenshot
    /// harness capture the "recording in progress" state without spending CI
    /// resources on a real encode. The elapsed timer shows a fixed placeholder.
    #[arg(long)]
    preview_spoof_recording: bool,

    /// Preview harness only: present synthetic camera devices instead of
    /// enumerating real ones, so the full camera UI (preview, switcher, format
    /// pickers) renders while frames come from `--preview-source`. This lets the
    /// screenshot harness run with no camera and no dma-buf provider, so it works
    /// on any runner. Meaningful only alongside `--preview-source`.
    #[arg(long)]
    preview_fake_camera: bool,
}

fn parse_window_size(s: &str) -> Result<(f32, f32), String> {
    let (w, h) = s
        .split_once('x')
        .ok_or_else(|| "expected WIDTHxHEIGHT (e.g. 400x880)".to_string())?;
    let width: f32 = w
        .trim()
        .parse()
        .map_err(|e| format!("invalid width: {e}"))?;
    let height: f32 = h
        .trim()
        .parse()
        .map_err(|e| format!("invalid height: {e}"))?;
    if !width.is_finite() || !height.is_finite() {
        return Err("window dimensions must be finite".to_string());
    }
    if width < 1.0 || height < 1.0 {
        return Err("window dimensions must be positive".to_string());
    }
    Ok((width, height))
}

#[derive(Subcommand)]
enum Commands {
    /// Run in terminal mode (renders camera to terminal)
    Terminal,

    /// List available cameras
    List,

    /// Take a photo
    Photo {
        /// Camera index to use (from 'camera list')
        #[arg(short, long, default_value = "0")]
        camera: usize,

        /// Output file path (default: ~/Pictures/Camera/photo_TIMESTAMP.jpg)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Record a video
    Video {
        /// Camera index to use (from 'camera list')
        #[arg(short, long, default_value = "0")]
        camera: usize,

        /// Recording duration in seconds
        #[arg(short, long, default_value = "10")]
        duration: u64,

        /// Output file path (default: ~/Videos/Camera/video_TIMESTAMP.mp4)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Enable audio recording
        #[arg(short, long)]
        audio: bool,
    },

    /// Process images through computational photography pipelines
    Process {
        #[command(subcommand)]
        mode: ProcessMode,
    },
}

#[derive(Subcommand)]
enum ProcessMode {
    /// Burst mode: multi-frame denoising and HDR+ pipeline
    BurstMode {
        /// Input images or directory containing images (PNG, DNG supported)
        #[arg(required = true)]
        input: Vec<PathBuf>,

        /// Output directory for processed images (default: same as input or ~/Pictures/Camera)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let is_terminal_mode = matches!(cli.command, Some(Commands::Terminal));

    // Initialize logging
    // Set RUST_LOG environment variable to control log level
    // Examples: RUST_LOG=debug, RUST_LOG=camera=debug, RUST_LOG=info
    //
    // In terminal mode, suppress all log output — stderr writes would corrupt
    // the ratatui TUI since the alternate screen only covers stdout.
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));

    if is_terminal_mode {
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .with_writer(std::io::sink)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .with_target(true)
            .with_level(true)
            .init();
    }

    tracing::info!("camera app starting");

    match cli.command {
        Some(Commands::Terminal) => camera::terminal::run(),
        Some(Commands::List) => cli::list_cameras(),
        Some(Commands::Photo { camera, output }) => cli::take_photo(camera, output),
        Some(Commands::Video {
            camera,
            duration,
            output,
            audio,
        }) => cli::record_video(camera, duration, output, audio),
        Some(Commands::Process { mode }) => match mode {
            ProcessMode::BurstMode { input, output } => cli::process_burst_mode(input, output),
        },
        None => run_gui(
            cli.preview_source,
            cli.preview_window,
            cli.preview_spoof_recording,
            cli.preview_fake_camera,
        ),
    }
}

fn run_gui(
    preview_source: Option<PathBuf>,
    preview_window: Option<(f32, f32)>,
    preview_spoof_recording: bool,
    preview_fake_camera: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Start pre-warming on background threads BEFORE the iced event loop.
    // This overlaps GStreamer init, device enumeration, and camera discovery
    // with Wayland/wgpu setup (~280ms of framework time we'd otherwise waste).
    //
    // Three parallel threads:
    //   1. Audio: pw-dump for PipeWire sources (~10ms)
    //   2. Camera: libcamera enumeration + format query (~170ms)
    //   3. Main prewarm: GStreamer init + video encoders, then collects 1 & 2
    let audio_handle = std::thread::spawn(|| {
        tracing::info!("prewarm: enumerating audio devices");
        let devices = camera::backends::audio::enumerate_audio_devices();
        tracing::info!(count = devices.len(), "prewarm: audio devices ready");
        devices
    });
    let camera_handle = std::thread::spawn(|| {
        tracing::info!("prewarm: enumerating cameras");
        let backend = camera::backends::camera::create_backend();
        let cameras = backend.enumerate_cameras();
        tracing::info!(count = cameras.len(), "prewarm: cameras enumerated");
        let formats = if let Some(cam) = cameras.first() {
            if !cam.path.is_empty() {
                backend.get_formats(cam, false)
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        tracing::info!(
            count = formats.len(),
            "prewarm: formats for default camera ready"
        );
        (cameras, formats)
    });
    let prewarm_handle = std::thread::spawn(move || {
        tracing::info!("prewarm: initializing GStreamer + video encoders");
        if let Err(e) = gstreamer::init() {
            tracing::error!(error = %e, "prewarm: failed to initialize GStreamer");
        }
        let video_encoders = camera::media::encoders::video::enumerate_video_encoders();
        tracing::info!(
            count = video_encoders.len(),
            "prewarm: GStreamer + video encoders ready"
        );

        let audio_devices = audio_handle.join().unwrap_or_else(|e| {
            tracing::warn!(error = ?e, "prewarm: audio enumeration thread panicked");
            Vec::new()
        });

        // Don't join camera_handle here — it takes ~170ms and would block init().
        // Pass it through so init() can wrap it in an async Task instead.
        camera::app::PrewarmResults {
            audio_devices,
            video_encoders,
            camera_enum: Some(camera_handle),
        }
    });

    // Get the system's preferred languages.
    let requested_languages = i18n_embed::DesktopLanguageRequester::requested_languages();

    // Enable localizations to be applied.
    i18n::init(&requested_languages);

    // Settings for configuring the application window and iced runtime.
    let mut settings = cosmic::app::Settings::default().size_limits(
        cosmic::iced::Limits::NONE
            .min_width(WINDOW_MIN_WIDTH)
            .min_height(WINDOW_MIN_HEIGHT),
    );

    // When a preview source is provided, set a fixed window size — defaults
    // to 900x700 (Flathub's recommended standard-display size) but the user
    // can override via `--preview-window WxH` to capture phone-aspect shots.
    if preview_source.is_some() {
        let (w, h) = preview_window.unwrap_or((900.0, 700.0));
        settings = settings.size(cosmic::iced::Size::new(w, h));
    }

    // Create app flags with pre-warm handle
    let flags = camera::app::AppFlags {
        preview_source,
        preview_spoof_recording,
        preview_fake_camera,
        prewarm: Some(prewarm_handle),
    };

    // Starts the application's event loop with flags
    cosmic::app::run::<AppModel>(settings, flags)?;

    Ok(())
}
