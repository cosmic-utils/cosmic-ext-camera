// SPDX-License-Identifier: GPL-3.0-only
//! One capture entry point for GUI, CLI and terminal.
use super::{
    libcamera::{NativeLibcameraPipeline, PipelineSharedState},
    pipewire::PipeWirePipeline,
    types::*,
};

pub(crate) enum CapturePipeline {
    Native(NativeLibcameraPipeline),
    PipeWire(PipeWirePipeline),
}
impl CapturePipeline {
    pub(crate) fn new(
        path: &str,
        format: &CameraFormat,
        multistream: bool,
        shared: PipelineSharedState,
    ) -> BackendResult<Self> {
        // Shared GUI still storage survives subscription replacement. Clear it
        // for both routes so a PipeWire frame/request cannot cross into a native
        // camera (including failed initialization), or vice versa.
        shared.still_capture.reset();
        if let Some(serial) = path.strip_prefix("pipewire:") {
            tracing::info!(
                backend = "pipewire-v4l2",
                target = serial,
                "Starting camera capture"
            );
            PipeWirePipeline::new(serial, format, shared).map(Self::PipeWire)
        } else {
            tracing::info!(
                backend = "libcamera",
                target = path,
                "Starting camera capture"
            );
            NativeLibcameraPipeline::new(path, format, multistream, shared).map(Self::Native)
        }
    }
    pub(crate) fn preview_metrics(&self) -> std::sync::Arc<super::preview_metrics::PreviewMetrics> {
        match self {
            Self::Native(p) => p.preview_metrics.clone(),
            Self::PipeWire(p) => p.preview_metrics.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex, atomic::AtomicBool};
    #[test]
    fn switching_to_native_clears_fallback_still_even_if_initialization_fails() {
        let (tx, _rx) = futures::channel::mpsc::channel(1);
        let frame = CameraFrame {
            width: 1,
            height: 1,
            data: FrameData::from_owned_vec(vec![0; 4]),
            format: PixelFormat::RGBA,
            stride: 4,
            yuv_planes: None,
            captured_at: std::time::Instant::now(),
            sensor_timestamp_ns: None,
            libcamera_metadata: None,
        };
        let still = Arc::new(super::super::still_capture::StillCapture::default());
        let request = still.request();
        let ticket = still.pending().unwrap();
        assert!(still.publish(ticket, frame));
        let shared = PipelineSharedState {
            zoom: Default::default(),
            frame_sender: tx,
            still_capture: still.clone(),
            recording_sender: Arc::new(Mutex::new(None)),
            cancel_flag: Arc::new(AtomicBool::new(false)),
        };
        assert!(
            CapturePipeline::new(
                "nonexistent-native-camera-for-switch-regression",
                &CameraFormat {
                    width: 1,
                    height: 1,
                    framerate: None,
                    pixel_format: "RGBA".into(),
                    hardware_accelerated: false
                },
                false,
                shared
            )
            .is_err()
        );
        assert!(
            request.take().is_none(),
            "a fallback still must not cross into the native camera lifecycle"
        );
        assert!(!still.has_request());
    }
}
