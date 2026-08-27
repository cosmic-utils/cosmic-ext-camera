// SPDX-License-Identifier: GPL-3.0-only
//! Burst mode photo capture pipeline
//!
//! Implements multi-frame burst denoising based on the HDR+ algorithm.
//! All processing is GPU-accelerated using WGPU compute shaders.
//!
//! # Acknowledgments
//!
//! Implementation inspired by:
//! - hdr-plus-swift (GPL-3.0) by Martin Marek
//!   <https://github.com/martin-marek/hdr-plus-swift>
//! - Google HDR+ Paper (SIGGRAPH 2016)
//!   "Burst photography for high dynamic range and low-light imaging on mobile cameras"
//!   <https://www.hdrplusdata.org/hdrplus.pdf>
//! - Night Sight Paper (SIGGRAPH Asia 2019)
//!   "Handheld Mobile Photography in Very Low Light"
//!
//! # Pipeline Overview
//!
//! ```text
//! Burst Capture (4-8 frames)
//!        │
//!        ▼
//! Reference Selection (GPU sharpness)
//!        │
//!        ▼
//! Pyramid Alignment (GPU, 4-level, L1/L2 hybrid)
//!        │
//!        ▼
//! Frame Merging (GPU spatial or FFT)
//!        │
//!        ▼
//! Tone Mapping (GPU shadow recovery)
//!        │
//!        ▼
//! Output Image
//! ```

mod bayer_planes;
pub mod burst;
pub mod fft_gpu;
mod gpu_helpers;
pub mod params;

use crate::backends::camera::types::{CameraFrame, PixelFormat, SensorRotation};
use crate::gpu::{self, wgpu};
use crate::shaders::{GpuFrameInput, get_gpu_convert_pipeline};
use bayer_planes::{BayerPlanes, extract_bayer_planes};
use std::sync::{Arc, RwLock};
use tracing::{debug, info, warn};

/// Progress callback for burst mode processing
///
/// Called with progress value (0.0 - 1.0) during processing stages.
/// The callback should be cheap to call as it may be invoked frequently.
pub type ProgressCallback = Arc<dyn Fn(f32) + Send + Sync>;

/// Buffer usage patterns for GPU memory allocation
#[derive(Debug, Clone, Copy)]
enum BufferKind {
    /// Read/write storage buffer that can be copied to/from
    Storage,
    /// Read-only storage buffer (no CPU writes after creation)
    StorageReadonly,
    /// Uniform buffer for shader parameters
    Uniform,
    /// Staging buffer for GPU-to-CPU readback
    Staging,
}

// Shader sources
const PYRAMID_SHADER: &str = include_str!("../../../shaders/burst_mode/pyramid.wgsl");
const SHARPNESS_SHADER: &str = include_str!("../../../shaders/burst_mode/sharpness.wgsl");
const ALIGN_TILE_SHADER: &str = include_str!("../../../shaders/burst_mode/align_tile.wgsl");
const WARP_SHADER: &str = include_str!("../../../shaders/burst_mode/warp.wgsl");
const TONEMAP_SHADER: &str = include_str!("../../../shaders/burst_mode/tonemap.wgsl");
const NOISE_ESTIMATE_SHADER: &str = include_str!("../../../shaders/burst_mode/noise_estimate.wgsl");
pub(crate) const CA_ESTIMATE_SHADER: &str =
    include_str!("../../../shaders/burst_mode/ca_estimate.wgsl");
const BAYER_FINISH_SHADER: &str = include_str!("../../../shaders/burst_mode/bayer_finish.wgsl");

// Common utilities reference (see common.wgsl for documentation)
#[cfg(test)]
const COMMON_SHADER_REF: &str = include_str!("../../../shaders/burst_mode/common.wgsl");

// GPU parameter structs imported from params module
use params::{
    AlignParams, BayerFinishParams, CAEstimateParams, LocalLumParams, LuminanceParams, NoiseParams,
    PyramidParams, SharpnessParams, WarpParams,
};

// Pipeline configuration constants
/// Number of pyramid levels for hierarchical alignment (L0=full, L1=1/2, L2=1/4, L3=1/8)
const PYRAMID_LEVELS: usize = 4;
/// Tile size for sharpness computation
const SHARPNESS_TILE_SIZE: u32 = 16;
/// Final tile size for warp operation
const WARP_TILE_SIZE: u32 = 32;
/// Convert u8 pixel data to normalized f32 (0.0-1.0)
/// Used for GPU buffer upload - converts 8-bit [0,255] to float [0.0,1.0]
#[inline]
pub(crate) fn u8_to_f32_normalized(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&x| x as f32 / 255.0).collect()
}

// `BayerOffsets` / `BayerPlanes` / `extract_bayer_planes` / extract_planes_*
// helpers live in the `bayer_planes` sub-module; see that file for details.

/// Convert a camera frame to RGBA format using GPU compute shader
///
/// If the frame is already RGBA, returns a copy of the data.
/// For YUV and other formats, uses GPU compute shader for conversion.
async fn convert_frame_to_rgba(frame: &CameraFrame) -> Result<Vec<u8>, String> {
    // Fast path: already RGBA — strip stride padding if present
    if frame.format == PixelFormat::RGBA {
        let row_bytes = (frame.width * 4) as usize;
        let stride = frame.stride as usize;
        if stride <= row_bytes {
            return Ok(frame.data.to_vec());
        }
        let mut out = Vec::with_capacity(row_bytes * frame.height as usize);
        for y in 0..frame.height as usize {
            out.extend_from_slice(&frame.data[y * stride..y * stride + row_bytes]);
        }
        return Ok(out);
    }

    let input = GpuFrameInput::from_camera_frame(frame)?;

    // Use GPU compute shader pipeline for conversion
    let mut pipeline_guard = get_gpu_convert_pipeline()
        .await
        .map_err(|e| format!("Failed to get GPU convert pipeline: {}", e))?;

    let pipeline = pipeline_guard
        .as_mut()
        .ok_or("GPU convert pipeline not initialized")?;

    // Run GPU conversion (synchronous, just dispatches compute shader)
    pipeline
        .convert(&input)
        .map_err(|e| format!("GPU conversion failed: {}", e))?;

    // Read back RGBA data from GPU to CPU memory
    pipeline
        .read_rgba_to_cpu(frame.width, frame.height)
        .await
        .map_err(|e| format!("Failed to read RGBA from GPU: {}", e))
}

/// Hierarchical alignment configuration per pyramid level.
/// Each entry: (tile_size, search_distance, use_l2_metric)
/// Level 0 (full): coarse tiles, L1 metric, small search
/// Levels 1-3: progressively finer, L2 metric, sub-pixel refinement
const ALIGN_LEVEL_CONFIGS: [(u32, u32, bool); PYRAMID_LEVELS] = [
    (32, 2, false), // Level 0 (full): L1 metric, integer-pixel
    (32, 2, true),  // Level 1 (1/2): L2 metric, sub-pixel
    (16, 2, true),  // Level 2 (1/4): L2 metric, sub-pixel
    (8, 4, true),   // Level 3 (1/8): L2 metric, larger search
];

/// Burst mode processing configuration
#[derive(Debug, Clone)]
pub struct BurstModeConfig {
    /// Number of frames to capture in burst (4, 6, or 8)
    pub frame_count: usize,
    /// Frame interval in milliseconds (~33ms for 30fps camera)
    pub frame_interval_ms: u32,
    /// Robustness parameter for merge (higher = more aggressive denoising)
    pub robustness: f32,
    /// Shadow boost strength for tone mapping (0.0 - 1.0)
    pub shadow_boost: f32,
    /// Local contrast enhancement strength (0.0 - 1.0)
    pub local_contrast: f32,
    /// Optional crop rectangle for aspect ratio (x, y, width, height)
    /// Applied after processing to match the preview aspect ratio
    pub crop_rect: Option<(u32, u32, u32, u32)>,
    /// Export raw burst frames as PNG files for testing/debugging
    /// Frames are saved to the output directory as frame_000.png, frame_001.png, etc.
    pub export_raw_frames: bool,
    /// Save raw burst frames as DNG files for sharing/debugging
    /// Frames are saved alongside the output photo in a timestamped subfolder
    pub save_burst_raw_dng: bool,
    /// Output encoding format (JPEG, PNG, or DNG)
    pub encoding_format: super::EncodingFormat,
    /// Camera metadata for DNG encoding
    pub camera_metadata: super::CameraMetadata,
    /// Sensor rotation to correct the image orientation
    pub rotation: SensorRotation,
    /// Mirror the final HDR+ output horizontally (selfie / front-camera).
    pub mirror_horizontal: bool,
}

impl Default for BurstModeConfig {
    fn default() -> Self {
        Self {
            frame_count: 8,
            frame_interval_ms: 33,
            robustness: 1.0,
            shadow_boost: 0.2,                            // Subtle shadow lifting
            local_contrast: 0.15,                         // Subtle contrast enhancement
            crop_rect: None,           // No cropping by default (native aspect ratio)
            export_raw_frames: false,  // Don't export raw frames by default
            save_burst_raw_dng: false, // Don't save raw DNG frames by default
            encoding_format: super::EncodingFormat::Jpeg, // Default to JPEG
            camera_metadata: super::CameraMetadata::default(),
            rotation: SensorRotation::None, // No rotation by default
            mirror_horizontal: false,
        }
    }
}

/// Burst mode processing stages (internal to pipeline)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BurstModePipelineStage {
    #[default]
    Idle,
    Capturing,
    SelectingReference,
    Aligning,
    Merging,
    ToneMapping,
    Complete,
    Error,
}

/// GPU-resident aligned frame - stays on GPU to avoid CPU round-trips
/// This eliminates ~192MB per frame of CPU memory and avoids GPU-CPU-GPU transfers
pub struct GpuAlignedFrame {
    /// GPU buffer containing RGBA f32 data (normalized 0.0-1.0)
    pub buffer: wgpu::Buffer,
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
}

impl std::fmt::Debug for GpuAlignedFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuAlignedFrame")
            .field("width", &self.width)
            .field("height", &self.height)
            .field(
                "buffer_size_mb",
                &(self.width as u64 * self.height as u64 * 16 / (1024 * 1024)),
            )
            .finish()
    }
}

/// Merged frame result
#[derive(Debug)]
pub struct MergedFrame {
    /// RGBA pixel data (u8)
    pub data: Vec<u8>,
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
}

/// Reference frame pyramids for alignment
struct ReferencePyramids {
    /// Luminance pyramid (4 levels)
    lum: Vec<wgpu::Buffer>,
}

/// Pre-allocated buffers for frame alignment, reused across multiple frames.
/// This avoids creating/destroying GPU buffers for each frame in the burst.
struct AlignmentBuffers {
    /// Comparison frame RGBA (overwritten for each frame)
    comp_rgba: wgpu::Buffer,
    /// Luminance pyramid for comparison frame (4 levels)
    comp_lum: Vec<wgpu::Buffer>,
    /// Alignment offset buffers (4 levels)
    align: Vec<wgpu::Buffer>,
    /// Dummy buffer for first level (no previous alignment)
    dummy_prev_align: wgpu::Buffer,
    /// Uniform buffer for luminance params
    lum_params: wgpu::Buffer,
    /// Uniform buffers for pyramid params (3 levels, L1-L3)
    pyramid_params: Vec<wgpu::Buffer>,
    /// Uniform buffers for align params (4 levels)
    align_params: Vec<wgpu::Buffer>,
    /// Uniform buffers for prev_n_tiles_x (4 levels)
    prev_n_tiles_x: Vec<wgpu::Buffer>,
    /// Uniform buffer for warp params
    warp_params: wgpu::Buffer,
}

/// Pooled staging buffers for GPU-to-CPU readback.
///
/// Avoids creating/destroying staging buffers for each readback operation.
/// Uses take/return pattern to avoid holding borrows across await points.
struct StagingBufferPool {
    /// Small buffer for scalars (< 1KB)
    small: Option<wgpu::Buffer>,
    /// Large buffer for frame data (dynamically sized)
    large: Option<wgpu::Buffer>,
    /// Size of the large buffer (to know if reallocation is needed)
    large_size: u64,
}

impl StagingBufferPool {
    fn new() -> Self {
        Self {
            small: None,
            large: None,
            large_size: 0,
        }
    }

    /// Take a small staging buffer from the pool (creating if needed).
    /// Returns owned buffer that must be returned via `return_small()`.
    fn take_small(&mut self, device: &wgpu::Device) -> wgpu::Buffer {
        self.small.take().unwrap_or_else(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging_pool_small"),
                size: 1024,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
    }

    /// Return a small staging buffer to the pool.
    fn return_small(&mut self, buffer: wgpu::Buffer) {
        self.small = Some(buffer);
    }

    /// Take a large staging buffer from the pool (creating/resizing if needed).
    /// Returns owned buffer that must be returned via `return_large()`.
    fn take_large(&mut self, device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        // If we have a large enough buffer, take it
        if self.large.is_some() && self.large_size >= size {
            return self.large.take().unwrap();
        }

        // Otherwise create a new one (dropping the old one if any)
        self.large_size = size;
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_pool_large"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Return a large staging buffer to the pool.
    fn return_large(&mut self, buffer: wgpu::Buffer) {
        self.large = Some(buffer);
    }
}

/// GPU pipeline for burst mode processing
///
/// All operations are GPU-accelerated using WGPU compute shaders.
pub struct BurstModeGpuPipeline {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    /// FFT merge pipeline (eagerly initialized for fail-fast behavior)
    /// Shares device/queue with this pipeline to avoid slow re-initialization.
    fft_pipeline: fft_gpu::FftMergePipeline,

    // Pyramid pipelines
    pyramid_downsample_gray: wgpu::ComputePipeline,

    // Sharpness pipelines
    sharpness_tiles: wgpu::ComputePipeline,
    sharpness_reduce: wgpu::ComputePipeline,

    // Alignment pipelines
    align_tiles: wgpu::ComputePipeline,
    align_correct_upsampling: wgpu::ComputePipeline,
    rgb_to_luminance: wgpu::ComputePipeline,

    // Warp pipelines
    warp_frame: wgpu::ComputePipeline,

    // Tonemap pipelines
    tonemap_local_lum: wgpu::ComputePipeline,
    tonemap_apply: wgpu::ComputePipeline,

    // Noise estimation pipelines
    noise_build_histogram: wgpu::ComputePipeline,
    noise_find_median: wgpu::ComputePipeline,
    noise_compute_mad: wgpu::ComputePipeline,
    noise_finalize: wgpu::ComputePipeline,

    // CA estimation pipelines
    ca_init_bins: wgpu::ComputePipeline,
    ca_estimate_offsets: wgpu::ComputePipeline,
    ca_fit_model: wgpu::ComputePipeline,

    // Bayer finishing pipeline (HDR+ Section 6: demosaic merged planes + WB + CCM)
    bayer_finish: wgpu::ComputePipeline,

    // Bind group layouts
    pyramid_layout: wgpu::BindGroupLayout,
    sharpness_layout: wgpu::BindGroupLayout,
    align_layout: wgpu::BindGroupLayout,
    luminance_layout: wgpu::BindGroupLayout,
    warp_layout: wgpu::BindGroupLayout,
    local_lum_layout: wgpu::BindGroupLayout,
    tonemap_layout: wgpu::BindGroupLayout,
    noise_layout: wgpu::BindGroupLayout,
    ca_layout: wgpu::BindGroupLayout,
    bayer_finish_layout: wgpu::BindGroupLayout,

    /// Pooled staging buffers for GPU readback (reduces allocations)
    staging_pool: RwLock<StagingBufferPool>,
}

use gpu_helpers::BindingKind;

impl BurstModeGpuPipeline {
    /// Create a compute pipeline with common defaults
    fn create_pipeline(
        device: &wgpu::Device,
        label: &str,
        layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        entry_point: &str,
    ) -> wgpu::ComputePipeline {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Create a GPU buffer with the specified usage pattern
    fn create_buffer(&self, label: &str, size: u64, kind: BufferKind) -> wgpu::Buffer {
        let usage = match kind {
            BufferKind::Storage => {
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC
            }
            BufferKind::StorageReadonly => wgpu::BufferUsages::STORAGE,
            BufferKind::Uniform => wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            BufferKind::Staging => wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        };
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Create a storage buffer (read/write by compute shaders, can be copied to/from)
    fn create_storage_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.create_buffer(label, size, BufferKind::Storage)
    }

    /// Create a storage buffer that is read-only (no COPY_DST)
    fn create_storage_buffer_readonly(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.create_buffer(label, size, BufferKind::StorageReadonly)
    }

    /// Create a uniform buffer for shader parameters
    fn create_uniform_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.create_buffer(label, size, BufferKind::Uniform)
    }

    /// Create a staging buffer for GPU-to-CPU readback
    fn create_staging_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.create_buffer(label, size, BufferKind::Staging)
    }

    /// Dispatch a single compute pass with the given pipeline and bind group
    fn dispatch_compute(
        &self,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
    ) {
        self.dispatch_compute_batch(label, &[(pipeline, bind_group, workgroups)]);
    }

    /// Dispatch multiple compute passes in a single encoder submission
    ///
    /// More efficient than multiple dispatch_compute calls when passes don't depend
    /// on each other's output (can be batched together).
    fn dispatch_compute_batch(
        &self,
        label: &str,
        passes: &[(&wgpu::ComputePipeline, &wgpu::BindGroup, (u32, u32, u32))],
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        for (pipeline, bind_group, workgroups) in passes {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, Some(*bind_group), &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Yield to allow other GPU work (like desktop compositor) to run
    ///
    /// With low-priority queue (Family 1 + VK_EXT_global_priority LOW) and
    /// small chunked dispatches, the GPU should automatically preempt our
    /// work for higher-priority compositor rendering. We just need to poll
    /// to ensure work is submitted; no explicit sleep needed.
    async fn yield_to_compositor(&self) {
        // Poll to submit pending work - the low-priority queue handles preemption
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
    }

    /// Read data back from GPU buffer to CPU
    ///
    /// Creates a staging buffer, copies data, maps it, and returns the result.
    /// This is an async operation that waits for the GPU to finish.
    async fn read_buffer<T: bytemuck::Pod>(
        &self,
        src_buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<T>, String> {
        let size = (count * std::mem::size_of::<T>()) as u64;
        let staging = self.create_staging_buffer("readback_staging", size);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_encoder"),
            });
        encoder.copy_buffer_to_buffer(src_buffer, 0, &staging, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        receiver
            .await
            .map_err(|_| "Failed to receive map result")?
            .map_err(|e| format!("{:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Create a new GPU pipeline with all shaders loaded
    pub async fn new() -> Result<Self, String> {
        info!("Initializing burst mode GPU pipeline (all operations GPU-accelerated)");

        // Get shared GPU device to avoid creating multiple wgpu instances
        let gpu = gpu::get_shared_gpu().await?;
        let device = gpu.device;
        let queue = gpu.queue;

        info!(
            adapter = %gpu.info.adapter_name,
            backend = ?gpu.info.backend,
            low_priority = gpu.info.low_priority_enabled,
            "Using shared GPU device for burst mode"
        );

        let max_buffer_size = device.limits().max_storage_buffer_binding_size as u64;

        // Load all shader modules
        let pyramid_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pyramid_shader"),
            source: wgpu::ShaderSource::Wgsl(PYRAMID_SHADER.into()),
        });

        let sharpness_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sharpness_shader"),
            source: wgpu::ShaderSource::Wgsl(SHARPNESS_SHADER.into()),
        });

        let align_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("align_shader"),
            source: wgpu::ShaderSource::Wgsl(ALIGN_TILE_SHADER.into()),
        });

        let warp_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("warp_shader"),
            source: wgpu::ShaderSource::Wgsl(WARP_SHADER.into()),
        });

        let tonemap_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tonemap_shader"),
            source: wgpu::ShaderSource::Wgsl(TONEMAP_SHADER.into()),
        });

        let noise_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("noise_estimate_shader"),
            source: wgpu::ShaderSource::Wgsl(NOISE_ESTIMATE_SHADER.into()),
        });

        let ca_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ca_estimate_shader"),
            source: wgpu::ShaderSource::Wgsl(CA_ESTIMATE_SHADER.into()),
        });

        // Create bind group layouts
        let pyramid_layout = Self::create_pyramid_layout(&device);
        let sharpness_layout = Self::create_sharpness_layout(&device);
        let align_layout = Self::create_align_layout(&device);
        let luminance_layout = Self::create_luminance_layout(&device);
        let warp_layout = Self::create_warp_layout(&device);
        let local_lum_layout = Self::create_local_lum_layout(&device);
        let tonemap_layout = Self::create_tonemap_layout(&device);
        let noise_layout = Self::create_noise_layout(&device);
        let ca_layout = Self::create_ca_layout(&device);
        let bayer_finish_layout = Self::create_bayer_finish_layout(&device);

        // Create pipeline layouts
        let pyramid_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pyramid_pipeline_layout"),
                bind_group_layouts: &[&pyramid_layout],
                immediate_size: 0,
            });

        let sharpness_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sharpness_pipeline_layout"),
                bind_group_layouts: &[&sharpness_layout],
                immediate_size: 0,
            });

        let align_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("align_pipeline_layout"),
                bind_group_layouts: &[&align_layout],
                immediate_size: 0,
            });

        let luminance_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("luminance_pipeline_layout"),
                bind_group_layouts: &[&luminance_layout],
                immediate_size: 0,
            });

        let warp_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("warp_pipeline_layout"),
            bind_group_layouts: &[&warp_layout],
            immediate_size: 0,
        });

        let local_lum_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("local_lum_pipeline_layout"),
                bind_group_layouts: &[&local_lum_layout],
                immediate_size: 0,
            });

        let tonemap_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tonemap_pipeline_layout"),
                bind_group_layouts: &[&tonemap_layout],
                immediate_size: 0,
            });

        let noise_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("noise_pipeline_layout"),
                bind_group_layouts: &[&noise_layout],
                immediate_size: 0,
            });

        let ca_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ca_pipeline_layout"),
            bind_group_layouts: &[&ca_layout],
            immediate_size: 0,
        });

        // Create compute pipelines
        let pyramid_downsample_gray = Self::create_pipeline(
            &device,
            "pyramid_downsample_gray",
            &pyramid_pipeline_layout,
            &pyramid_module,
            "downsample_gray",
        );
        let sharpness_tiles = Self::create_pipeline(
            &device,
            "sharpness_tiles",
            &sharpness_pipeline_layout,
            &sharpness_module,
            "compute_sharpness_tiles",
        );
        let sharpness_reduce = Self::create_pipeline(
            &device,
            "sharpness_reduce",
            &sharpness_pipeline_layout,
            &sharpness_module,
            "reduce_sharpness",
        );
        let align_tiles = Self::create_pipeline(
            &device,
            "align_tiles",
            &align_pipeline_layout,
            &align_module,
            "align_tiles_parallel",
        );
        let align_correct_upsampling = Self::create_pipeline(
            &device,
            "align_correct_upsampling",
            &align_pipeline_layout,
            &align_module,
            "correct_upsampling_error",
        );
        let rgb_to_luminance = Self::create_pipeline(
            &device,
            "rgb_to_luminance",
            &luminance_pipeline_layout,
            &align_module,
            "rgb_to_luminance",
        );
        let warp_frame = Self::create_pipeline(
            &device,
            "warp_frame",
            &warp_pipeline_layout,
            &warp_module,
            "warp_frame",
        );

        // Tonemap pipelines
        let tonemap_local_lum = Self::create_pipeline(
            &device,
            "tonemap_local_lum",
            &local_lum_pipeline_layout,
            &tonemap_module,
            "compute_local_luminance",
        );
        let tonemap_apply = Self::create_pipeline(
            &device,
            "tonemap_apply",
            &tonemap_pipeline_layout,
            &tonemap_module,
            "tonemap",
        );

        // Noise estimation pipelines
        let noise_build_histogram = Self::create_pipeline(
            &device,
            "noise_build_histogram",
            &noise_pipeline_layout,
            &noise_module,
            "build_histogram",
        );
        let noise_find_median = Self::create_pipeline(
            &device,
            "noise_find_median",
            &noise_pipeline_layout,
            &noise_module,
            "find_median_from_histogram",
        );
        let noise_compute_mad = Self::create_pipeline(
            &device,
            "noise_compute_mad",
            &noise_pipeline_layout,
            &noise_module,
            "compute_mad_tiles",
        );
        let noise_finalize = Self::create_pipeline(
            &device,
            "noise_finalize",
            &noise_pipeline_layout,
            &noise_module,
            "finalize_noise_estimate",
        );

        // CA estimation pipelines
        let ca_init_bins = Self::create_pipeline(
            &device,
            "ca_init_bins",
            &ca_pipeline_layout,
            &ca_module,
            "init_bins",
        );
        let ca_estimate_offsets = Self::create_pipeline(
            &device,
            "ca_estimate_offsets",
            &ca_pipeline_layout,
            &ca_module,
            "estimate_ca_offsets",
        );
        let ca_fit_model = Self::create_pipeline(
            &device,
            "ca_fit_model",
            &ca_pipeline_layout,
            &ca_module,
            "fit_ca_model",
        );

        // Bayer finishing pipeline (HDR+ Section 6: demosaic + WB + CCM)
        let bayer_finish_source = format!(
            "{}\n{}",
            crate::shaders::RAW_COLOUR_FUNCTIONS,
            BAYER_FINISH_SHADER
        );
        let bayer_finish_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bayer_finish_shader"),
            source: wgpu::ShaderSource::Wgsl(bayer_finish_source.into()),
        });
        let bayer_finish_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bayer_finish_pipeline_layout"),
                bind_group_layouts: &[&bayer_finish_layout],
                immediate_size: 0,
            });
        let bayer_finish = Self::create_pipeline(
            &device,
            "bayer_finish",
            &bayer_finish_pipeline_layout,
            &bayer_finish_module,
            "demosaic_and_finish",
        );

        // Initialize FFT pipeline (eagerly, for fail-fast behavior)
        let fft_pipeline =
            fft_gpu::FftMergePipeline::new(device.clone(), queue.clone(), max_buffer_size)?;

        info!("Night mode GPU pipeline initialized successfully");

        Ok(Self {
            device,
            queue,
            pyramid_downsample_gray,
            sharpness_tiles,
            sharpness_reduce,
            align_tiles,
            align_correct_upsampling,
            rgb_to_luminance,
            warp_frame,
            tonemap_local_lum,
            tonemap_apply,
            noise_build_histogram,
            noise_find_median,
            noise_compute_mad,
            noise_finalize,
            ca_init_bins,
            ca_estimate_offsets,
            ca_fit_model,
            bayer_finish,
            pyramid_layout,
            sharpness_layout,
            align_layout,
            luminance_layout,
            warp_layout,
            local_lum_layout,
            tonemap_layout,
            noise_layout,
            ca_layout,
            bayer_finish_layout,
            fft_pipeline,
            staging_pool: RwLock::new(StagingBufferPool::new()),
        })
    }

    /// Create a bind group from a list of buffers with sequential binding indices (0, 1, 2, ...)
    ///
    /// This reduces boilerplate when all bindings are simple buffer bindings.
    fn bind_group(
        &self,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &entries,
        })
    }

    fn create_pyramid_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        use BindingKind::*;
        gpu_helpers::create_layout(
            device,
            "pyramid_layout",
            &[StorageRead, StorageReadWrite, Uniform],
        )
    }

    fn create_sharpness_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        use BindingKind::*;
        gpu_helpers::create_layout(
            device,
            "sharpness_layout",
            &[StorageRead, StorageReadWrite, StorageReadWrite, Uniform],
        )
    }

    fn create_align_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        use BindingKind::*;
        gpu_helpers::create_layout(
            device,
            "align_layout",
            &[
                StorageRead,
                StorageRead,
                StorageReadWrite,
                StorageRead,
                Uniform,
                Uniform,
            ],
        )
    }

    fn create_luminance_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        use BindingKind::*;
        gpu_helpers::create_layout(
            device,
            "luminance_layout",
            &[StorageRead, StorageReadWrite, Uniform],
        )
    }

    fn create_warp_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        use BindingKind::*;
        // Bindings: input, output, alignment, params
        gpu_helpers::create_layout(
            device,
            "warp_layout",
            &[
                StorageRead,      // 0: input_frame
                StorageReadWrite, // 1: output_frame
                StorageRead,      // 2: alignment
                Uniform,          // 3: params
            ],
        )
    }

    fn create_local_lum_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        use BindingKind::*;
        gpu_helpers::create_layout(
            device,
            "local_lum_layout",
            &[
                StorageRead,      // 0: input image
                StorageReadWrite, // 1: local luminance output
                Uniform,          // 2: params
                StorageReadWrite, // 3: global brightness accumulator (for adaptive tone mapping)
            ],
        )
    }

    fn create_tonemap_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        use BindingKind::*;
        gpu_helpers::create_layout(
            device,
            "tonemap_layout",
            &[
                StorageRead,
                StorageReadWrite,
                StorageRead,
                Uniform,
                Uniform,
                Uniform,
                Uniform,
            ],
        )
    }

    fn create_noise_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        use BindingKind::*;
        gpu_helpers::create_layout(
            device,
            "noise_layout",
            &[
                StorageRead,
                StorageReadWrite,
                StorageReadWrite,
                StorageReadWrite,
                Uniform,
            ],
        )
    }

    fn create_ca_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        use BindingKind::*;
        // CA estimation shader bindings:
        // 0: reference frame (RGBA f32) - read
        // 1: bin_data (per-radius-bin accumulators) - read/write
        // 2: params (CAEstimateParams) - uniform
        // 3: ca_coefficients (output) - read/write
        gpu_helpers::create_layout(
            device,
            "ca_layout",
            &[StorageRead, StorageReadWrite, Uniform, StorageReadWrite],
        )
    }

    fn create_bayer_finish_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        use BindingKind::*;
        // Bindings: bayer_planes(r), output(rw), params(u)
        gpu_helpers::create_layout(
            device,
            "bayer_finish_layout",
            &[StorageRead, StorageReadWrite, Uniform],
        )
    }

    /// Compute sharpness of a frame using GPU
    pub async fn compute_sharpness(&self, frame: &CameraFrame) -> Result<f32, String> {
        let rgba_data = convert_frame_to_rgba(frame).await?;
        let frame_f32 = u8_to_f32_normalized(&rgba_data);
        self.compute_sharpness_from_f32_rgba(&frame_f32, frame.width, frame.height)
            .await
    }

    /// Core sharpness computation on pre-prepared f32 RGBA data
    async fn compute_sharpness_from_f32_rgba(
        &self,
        data: &[f32],
        width: u32,
        height: u32,
    ) -> Result<f32, String> {
        let pixel_count = (width * height) as usize;

        let frame_buffer = self.create_storage_buffer(
            "sharpness_frame",
            (pixel_count * 4 * std::mem::size_of::<f32>()) as u64,
        );

        let n_tiles_x = width.div_ceil(SHARPNESS_TILE_SIZE);
        let n_tiles_y = height.div_ceil(SHARPNESS_TILE_SIZE);
        let n_tiles = (n_tiles_x * n_tiles_y) as usize;

        let partial_buffer = self.create_storage_buffer_readonly(
            "sharpness_partial",
            (n_tiles * 2 * std::mem::size_of::<f32>()) as u64,
        );

        let result_buffer =
            self.create_storage_buffer("sharpness_result", std::mem::size_of::<f32>() as u64);

        let params = SharpnessParams {
            width,
            height,
            tile_size: SHARPNESS_TILE_SIZE,
            n_tiles_x,
            n_tiles_y,
            _padding: [0; 3],
        };

        let params_buffer = self.create_uniform_buffer(
            "sharpness_params",
            std::mem::size_of::<SharpnessParams>() as u64,
        );

        self.queue
            .write_buffer(&frame_buffer, 0, bytemuck::cast_slice(data));
        self.queue
            .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&[params]));

        let bind_group = self.bind_group(
            "sharpness_bind_group",
            &self.sharpness_layout,
            &[
                &frame_buffer,
                &partial_buffer,
                &result_buffer,
                &params_buffer,
            ],
        );

        // Stage 1: Compute per-tile sharpness
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sharpness_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sharpness_tiles_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sharpness_tiles);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(n_tiles_x, n_tiles_y, 1);
        }

        // Stage 2: Reduce to single value
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sharpness_reduce_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sharpness_reduce);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Read back result using pooled staging buffer
        let staging_buffer = self.staging_pool.write().unwrap().take_small(&self.device);

        encoder.copy_buffer_to_buffer(
            &result_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of::<f32>() as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        receiver
            .await
            .map_err(|_| "Failed to receive map result")?
            .map_err(|e| format!("{:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        // Only read the first 4 bytes (f32) - pooled buffer may be larger
        let result: f32 = *bytemuck::from_bytes(&data[..std::mem::size_of::<f32>()]);
        drop(data);
        staging_buffer.unmap();

        // Return staging buffer to pool for reuse
        self.staging_pool
            .write()
            .unwrap()
            .return_small(staging_buffer);

        Ok(result)
    }

    /// Estimate noise standard deviation using GPU
    ///
    /// Uses histogram-based MAD (Median Absolute Deviation) method on Laplacian-filtered
    /// image data. This is more robust than variance-based methods for natural images.
    ///
    /// # Arguments
    /// * `data` - RGBA pixel data (u8)
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    /// Estimated noise standard deviation (in pixel value units, 0-255)
    pub async fn estimate_noise_gpu(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<f32, String> {
        let pixel_count = (width * height) as usize;

        // Pack RGBA u8 into u32 for GPU (the shader expects this format)
        let packed_data: Vec<u32> = data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| {
                u32::from(chunk[0])
                    | (u32::from(chunk[1]) << 8)
                    | (u32::from(chunk[2]) << 16)
                    | (u32::from(chunk[3]) << 24)
            })
            .collect();

        // Histogram parameters
        const NUM_BINS: u32 = 256;
        // Laplacian range is 0-1020 (4*255), map to 256 bins
        // bin = laplacian * (256/1020) ≈ laplacian * 0.251
        let bin_scale: f32 = NUM_BINS as f32 / 1020.0;

        let n_tiles_x = width.div_ceil(SHARPNESS_TILE_SIZE);
        let n_tiles_y = height.div_ceil(SHARPNESS_TILE_SIZE);
        let n_tiles = (n_tiles_x * n_tiles_y) as usize;

        // Create buffers
        let input_buffer = self.create_storage_buffer(
            "noise_input",
            (pixel_count * std::mem::size_of::<u32>()) as u64,
        );

        let histogram_buffer = self.create_storage_buffer(
            "noise_histogram",
            (NUM_BINS as usize * std::mem::size_of::<u32>()) as u64,
        );

        let partial_buffer = self.create_storage_buffer_readonly(
            "noise_partial",
            (n_tiles * 2 * std::mem::size_of::<f32>()) as u64,
        );

        let output_buffer = self.create_storage_buffer(
            "noise_output",
            (4 * std::mem::size_of::<f32>()) as u64, // [noise_sd, median, mad, count]
        );

        let params = NoiseParams {
            width,
            height,
            num_bins: NUM_BINS,
            bin_scale,
            median_value: 0.0, // Will be set after pass 1
            _padding0: 0,
            _padding1: 0,
            _padding2: 0,
        };

        let params_buffer =
            self.create_uniform_buffer("noise_params", std::mem::size_of::<NoiseParams>() as u64);

        // Upload data
        self.queue
            .write_buffer(&input_buffer, 0, bytemuck::cast_slice(&packed_data));
        self.queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Clear histogram to zero
        let zero_histogram = vec![0u32; NUM_BINS as usize];
        self.queue
            .write_buffer(&histogram_buffer, 0, bytemuck::cast_slice(&zero_histogram));

        // Create bind group
        let bind_group = self.bind_group(
            "noise_bind_group",
            &self.noise_layout,
            &[
                &input_buffer,
                &histogram_buffer,
                &partial_buffer,
                &output_buffer,
                &params_buffer,
            ],
        );

        // Pass 1 & 2: Build histogram and find median
        self.dispatch_compute_batch(
            "noise_histogram",
            &[
                (
                    &self.noise_build_histogram,
                    &bind_group,
                    (n_tiles_x, n_tiles_y, 1),
                ),
                (&self.noise_find_median, &bind_group, (1, 1, 1)),
            ],
        );

        // Yield to compositor after histogram passes
        self.yield_to_compositor().await;

        // Read median from output buffer
        let median_values = self.read_buffer::<f32>(&output_buffer, 4).await?;
        let median = median_values[1]; // output[1] = median

        // Update params with median for MAD pass
        let params_with_median = NoiseParams {
            width,
            height,
            num_bins: NUM_BINS,
            bin_scale,
            median_value: median,
            _padding0: 0,
            _padding1: 0,
            _padding2: 0,
        };
        self.queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params_with_median));

        // Pass 3 & 4: Compute MAD and finalize
        self.dispatch_compute_batch(
            "noise_mad",
            &[
                (
                    &self.noise_compute_mad,
                    &bind_group,
                    (n_tiles_x, n_tiles_y, 1),
                ),
                (&self.noise_finalize, &bind_group, (1, 1, 1)),
            ],
        );

        // Yield to compositor after MAD passes
        self.yield_to_compositor().await;

        // Read final result
        let result = self.read_buffer::<f32>(&output_buffer, 4).await?;
        let noise_sd = result[0];
        let median = result[1];
        let mad = result[2];
        debug!(noise_sd, median, mad, "GPU noise estimation complete");

        Ok(noise_sd)
    }

    /// Estimate chromatic aberration coefficients from reference frame
    ///
    /// Analyzes edge pixels to estimate radial CA model coefficients:
    /// - ca_r_coeff: Red channel radial scaling coefficient
    /// - ca_b_coeff: Blue channel radial scaling coefficient
    ///
    /// Uses the model: scale = 1 + coeff * (radius/max_radius)²
    pub async fn estimate_ca_coefficients(
        &self,
        ref_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) -> Result<(f32, f32), String> {
        const NUM_RADIUS_BINS: u32 = 16;

        // Create buffers for CA estimation
        // bin_data: 3 atomic i32s per bin (r_sum_scaled, b_sum_scaled, count)
        // Using i32 for atomic operations (floats scaled by 1,000,000 in shader)
        let bin_data_size = (NUM_RADIUS_BINS * 3) as u64 * std::mem::size_of::<i32>() as u64;
        let bin_data_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ca_bin_data"),
            size: bin_data_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Output coefficients: [ca_r, ca_b]
        let coeffs_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ca_coefficients"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Params buffer
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ca_params"),
            size: std::mem::size_of::<CAEstimateParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = CAEstimateParams {
            width,
            height,
            center_x: width as f32 / 2.0,
            center_y: height as f32 / 2.0,
            edge_threshold: 0.08,  // Gradient threshold for edge detection
            radial_alignment: 0.6, // Min dot product with radial direction
            num_radius_bins: NUM_RADIUS_BINS,
            search_radius: 4, // ±0.5 pixel search
        };
        self.queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Create bind group
        let bind_group = self.bind_group(
            "ca_bind_group",
            &self.ca_layout,
            &[ref_buffer, &bin_data_buffer, &params_buffer, &coeffs_buffer],
        );

        // Pass 1: Initialize bins to zero
        self.dispatch_compute(
            "ca_init_bins",
            &self.ca_init_bins,
            &bind_group,
            ((NUM_RADIUS_BINS * 3).div_ceil(64), 1, 1),
        );

        // Pass 2: Estimate CA offsets at edge pixels
        self.dispatch_compute(
            "ca_estimate_offsets",
            &self.ca_estimate_offsets,
            &bind_group,
            (width.div_ceil(16), height.div_ceil(16), 1),
        );

        // Pass 3: Fit quadratic model
        self.dispatch_compute("ca_fit_model", &self.ca_fit_model, &bind_group, (1, 1, 1));

        // Read back coefficients
        let result = self.read_buffer::<f32>(&coeffs_buffer, 2).await?;
        let ca_r_coeff = result[0];
        let ca_b_coeff = result[1];

        debug!(ca_r_coeff, ca_b_coeff, "CA estimation complete");

        Ok((ca_r_coeff, ca_b_coeff))
    }

    /// Select reference frame (sharpest) from burst
    ///
    /// HDR+ paper Section 4: "To minimize perceived shutter lag, we choose the
    /// reference frame from the first 3 frames in the burst." We limit the search
    /// to the first 3 candidates for latency, as later frames are further from
    /// the shutter press moment.
    pub async fn select_reference_frame(
        &self,
        frames: &[Arc<CameraFrame>],
    ) -> Result<usize, String> {
        if frames.is_empty() {
            return Err("No frames provided".to_string());
        }

        if frames.len() == 1 {
            return Ok(0);
        }

        // HDR+ paper: search only the first 3 frames to minimize perceived shutter lag
        let search_count = frames.len().min(3);
        info!(
            frame_count = frames.len(),
            search_count, "Selecting reference frame from first {} (GPU)", search_count
        );

        let mut max_sharpness = f32::MIN;
        let mut ref_idx = 0;

        for (idx, frame) in frames[..search_count].iter().enumerate() {
            let sharpness = self.compute_sharpness(frame).await?;
            debug!(frame = idx, sharpness, "Frame sharpness");

            if sharpness > max_sharpness {
                max_sharpness = sharpness;
                ref_idx = idx;
            }
        }

        debug!(
            reference = ref_idx,
            sharpness = max_sharpness,
            "Reference frame selected"
        );
        Ok(ref_idx)
    }

    /// Create pre-allocated buffers for frame alignment.
    /// These buffers are reused across multiple frames to avoid allocation overhead.
    fn create_alignment_buffers(&self, width: u32, height: u32) -> AlignmentBuffers {
        let pixel_count = (width * height) as usize;

        // Build pyramid dimensions
        let level_dims: Vec<(u32, u32)> = (0..PYRAMID_LEVELS)
            .map(|level| {
                let scale = 1 << level;
                (width.div_ceil(scale), height.div_ceil(scale))
            })
            .collect();

        // Comparison RGBA buffer
        let comp_rgba = self.create_storage_buffer(
            "align_comp_rgba_pool",
            (pixel_count * 4 * std::mem::size_of::<f32>()) as u64,
        );

        // Comparison luminance pyramid
        let comp_lum: Vec<wgpu::Buffer> = (0..PYRAMID_LEVELS)
            .map(|level| {
                let (w, h) = level_dims[level];
                let size = (w * h) as usize;
                self.create_storage_buffer_readonly(
                    &format!("comp_lum_L{}_pool", level),
                    (size * std::mem::size_of::<f32>()) as u64,
                )
            })
            .collect();

        // Alignment offset buffers
        let align: Vec<wgpu::Buffer> = (0..PYRAMID_LEVELS)
            .map(|level| {
                let (level_w, level_h) = level_dims[level];
                let (tile_size, _, _) = ALIGN_LEVEL_CONFIGS[level];
                let tile_step = tile_size / 2;
                let n_tiles_x = (level_w.saturating_sub(tile_size)) / tile_step + 1;
                let n_tiles_y = (level_h.saturating_sub(tile_size)) / tile_step + 1;
                let n_tiles = (n_tiles_x * n_tiles_y) as usize;
                self.create_storage_buffer(
                    &format!("align_L{}_pool", level),
                    (n_tiles * 2 * std::mem::size_of::<f32>()) as u64,
                )
            })
            .collect();

        // Dummy buffer for first level
        let dummy_prev_align = self.create_storage_buffer(
            "dummy_prev_align_pool",
            2 * std::mem::size_of::<f32>() as u64,
        );
        let zeros: [f32; 2] = [0.0, 0.0];
        self.queue
            .write_buffer(&dummy_prev_align, 0, bytemuck::cast_slice(&zeros));

        // Uniform buffers
        let lum_params = self.create_uniform_buffer(
            "lum_params_pool",
            std::mem::size_of::<LuminanceParams>() as u64,
        );

        let pyramid_params: Vec<wgpu::Buffer> = (1..PYRAMID_LEVELS)
            .map(|level| {
                self.create_uniform_buffer(
                    &format!("pyramid_params_L{}_pool", level),
                    std::mem::size_of::<PyramidParams>() as u64,
                )
            })
            .collect();

        let align_params: Vec<wgpu::Buffer> = (0..PYRAMID_LEVELS)
            .map(|level| {
                self.create_uniform_buffer(
                    &format!("align_params_L{}_pool", level),
                    std::mem::size_of::<AlignParams>() as u64,
                )
            })
            .collect();

        let prev_n_tiles_x: Vec<wgpu::Buffer> = (0..PYRAMID_LEVELS)
            .map(|level| {
                self.create_uniform_buffer(
                    &format!("prev_n_tiles_x_L{}_pool", level),
                    std::mem::size_of::<u32>() as u64,
                )
            })
            .collect();

        let warp_params = self
            .create_uniform_buffer("warp_params_pool", std::mem::size_of::<WarpParams>() as u64);

        AlignmentBuffers {
            comp_rgba,
            comp_lum,
            align,
            dummy_prev_align,
            lum_params,
            pyramid_params,
            align_params,
            prev_n_tiles_x,
            warp_params,
        }
    }

    /// Align all frames to reference frame on GPU with optional progress reporting
    ///
    /// This is a memory-optimized version that keeps aligned frames on GPU,
    /// eliminating ~192MB of CPU memory per frame and avoiding GPU-CPU-GPU transfers.
    /// Use this with `merge_frames_gpu` for the full GPU-only pipeline.
    ///
    /// Buffer pooling: Pre-allocates temporary buffers once and reuses them for each frame,
    /// reducing GPU memory churn from ~64 buffer allocations to ~20.
    ///
    /// Progress is reported from 0.10 to 0.60 across all frames when a callback is provided.
    pub async fn align_frames_gpu_with_progress(
        &self,
        frames: &[Arc<CameraFrame>],
        ref_idx: usize,
        progress: &Option<ProgressCallback>,
    ) -> Result<Vec<GpuAlignedFrame>, String> {
        let align_start = std::time::Instant::now();
        debug!(
            frame_count = frames.len(),
            reference = ref_idx,
            "Aligning frames (GPU-only, with buffer pooling)"
        );

        // Helper to report progress (0.10 to 0.60 range)
        let report = |frame_idx: usize, total_frames: usize| {
            if let Some(cb) = progress {
                // Map frame progress to 0.10 - 0.60 range
                let frame_progress = (frame_idx + 1) as f32 / total_frames as f32;
                let overall = 0.10 + frame_progress * 0.50;
                cb(overall);
            }
        };

        let reference = &frames[ref_idx];
        let width = reference.width;
        let height = reference.height;
        let pixel_count = (width * height) as usize;

        // Pre-allocate reusable buffers for alignment (buffer pooling)
        let step_start = std::time::Instant::now();
        let buffers = self.create_alignment_buffers(width, height);
        debug!(
            elapsed_ms = step_start.elapsed().as_millis(),
            "Created alignment buffer pool"
        );

        // Upload reference frame to GPU once (stays for all alignments)
        let ref_rgba = convert_frame_to_rgba(reference).await?;
        let ref_f32 = u8_to_f32_normalized(&ref_rgba);
        let ref_rgba_buffer = self.create_storage_buffer(
            "align_ref_rgba",
            (pixel_count * 4 * std::mem::size_of::<f32>()) as u64,
        );
        self.queue
            .write_buffer(&ref_rgba_buffer, 0, bytemuck::cast_slice(&ref_f32));
        // Free ~48MB CPU buffer immediately after GPU upload to reduce peak memory
        drop(ref_f32);

        // Build reference pyramids once (reused for all comparisons)
        let step_start = std::time::Instant::now();
        let ref_pyramids = self.build_reference_pyramid(&ref_rgba_buffer, width, height, &buffers);
        debug!(
            elapsed_ms = step_start.elapsed().as_millis(),
            "Built reference pyramids"
        );

        // Yield to compositor after pyramid building
        self.yield_to_compositor().await;

        // Estimate chromatic aberration coefficients from reference frame
        let step_start = std::time::Instant::now();
        let (ca_r_coeff, ca_b_coeff) = self
            .estimate_ca_coefficients(&ref_rgba_buffer, width, height)
            .await?;
        info!(
            elapsed_ms = step_start.elapsed().as_millis(),
            ca_r_coeff, ca_b_coeff, "CA estimation complete"
        );

        // Yield to compositor after CA estimation
        self.yield_to_compositor().await;

        let mut aligned_frames = Vec::with_capacity(frames.len() - 1);
        let total_frames = frames.len() - 1; // Minus reference frame

        for (idx, frame) in frames.iter().enumerate() {
            if idx == ref_idx {
                continue;
            }

            // Skip frames with different dimensions
            if frame.width != width || frame.height != height {
                warn!(
                    frame = idx,
                    frame_width = frame.width,
                    frame_height = frame.height,
                    ref_width = width,
                    ref_height = height,
                    "Skipping frame with mismatched dimensions"
                );
                continue;
            }

            let frame_start = std::time::Instant::now();

            // Align using pre-allocated buffers
            let gpu_frame = self
                .align_single_frame_pooled(
                    &ref_pyramids,
                    frame,
                    width,
                    height,
                    &buffers,
                    (ca_r_coeff, ca_b_coeff),
                )
                .await?;

            aligned_frames.push(gpu_frame);
            info!(
                frame = idx,
                elapsed_ms = frame_start.elapsed().as_millis(),
                "Frame aligned"
            );

            // Report progress after each frame
            report(aligned_frames.len(), total_frames);

            // Yield to compositor after every frame
            self.yield_to_compositor().await;
        }

        debug!(
            aligned = aligned_frames.len(),
            total_elapsed_ms = align_start.elapsed().as_millis(),
            "All frames aligned"
        );
        Ok(aligned_frames)
    }

    /// Build a luminance pyramid from an RGBA buffer.
    ///
    /// Converts RGBA to grayscale luminance (L0) then builds a 4-level Gaussian pyramid.
    /// This is the common implementation used by both reference and comparison pyramids.
    ///
    /// # Arguments
    /// * `rgba_buffer` - Input RGBA f32 buffer
    /// * `lum_buffers` - Pre-allocated output luminance buffers (4 levels)
    /// * `width` - Frame width
    /// * `height` - Frame height
    /// * `buffers` - Shared uniform buffers for parameters
    /// * `label_prefix` - Label prefix for debugging ("ref" or "comp")
    fn build_luminance_pyramid(
        &self,
        rgba_buffer: &wgpu::Buffer,
        lum_buffers: &[wgpu::Buffer],
        width: u32,
        height: u32,
        buffers: &AlignmentBuffers,
        label_prefix: &str,
    ) {
        // Build pyramid dimensions
        let level_dims: Vec<(u32, u32)> = (0..PYRAMID_LEVELS)
            .map(|level| {
                let scale = 1 << level;
                (width.div_ceil(scale), height.div_ceil(scale))
            })
            .collect();

        // Convert RGBA to luminance (L0)
        // channel=3 means luminance (BT.601 grayscale conversion)
        let lum_params = LuminanceParams {
            width,
            height,
            channel: 3, // Luminance mode for backward compatibility
            _padding1: 0,
        };
        self.queue
            .write_buffer(&buffers.lum_params, 0, bytemuck::cast_slice(&[lum_params]));

        let lum_bg = self.bind_group(
            &format!("{}_lum_bg", label_prefix),
            &self.luminance_layout,
            &[rgba_buffer, &lum_buffers[0], &buffers.lum_params],
        );

        self.dispatch_compute(
            &format!("{}_luminance", label_prefix),
            &self.rgb_to_luminance,
            &lum_bg,
            (width.div_ceil(16), height.div_ceil(16), 1),
        );

        // Build pyramid (L1-L3)
        for level in 1..PYRAMID_LEVELS {
            let (src_w, src_h) = level_dims[level - 1];
            let (dst_w, dst_h) = level_dims[level];

            let pyramid_params = PyramidParams {
                src_width: src_w,
                src_height: src_h,
                dst_width: dst_w,
                dst_height: dst_h,
            };
            self.queue.write_buffer(
                &buffers.pyramid_params[level - 1],
                0,
                bytemuck::cast_slice(&[pyramid_params]),
            );

            let downsample_bg = self.bind_group(
                &format!("{}_downsample_L{}", label_prefix, level),
                &self.pyramid_layout,
                &[
                    &lum_buffers[level - 1],
                    &lum_buffers[level],
                    &buffers.pyramid_params[level - 1],
                ],
            );

            self.dispatch_compute(
                &format!("{}_pyramid_L{}", label_prefix, level),
                &self.pyramid_downsample_gray,
                &downsample_bg,
                (dst_w.div_ceil(16), dst_h.div_ceil(16), 1),
            );
        }
    }

    /// Build reference luminance pyramids for alignment.
    /// Done once, reused for all frame comparisons.
    fn build_reference_pyramid(
        &self,
        ref_rgba_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
        buffers: &AlignmentBuffers,
    ) -> ReferencePyramids {
        // Build pyramid dimensions for buffer allocation
        let level_dims: Vec<(u32, u32)> = (0..PYRAMID_LEVELS)
            .map(|level| {
                let scale = 1 << level;
                (width.div_ceil(scale), height.div_ceil(scale))
            })
            .collect();

        // Create reference luminance buffers (unique to reference, not pooled)
        let ref_lum_buffers: Vec<wgpu::Buffer> = (0..PYRAMID_LEVELS)
            .map(|level| {
                let (w, h) = level_dims[level];
                let size = (w * h) as usize;
                self.create_storage_buffer_readonly(
                    &format!("ref_lum_L{}", level),
                    (size * std::mem::size_of::<f32>()) as u64,
                )
            })
            .collect();

        // Build luminance pyramid using common helper
        self.build_luminance_pyramid(
            ref_rgba_buffer,
            &ref_lum_buffers,
            width,
            height,
            buffers,
            "ref",
        );

        ReferencePyramids {
            lum: ref_lum_buffers,
        }
    }

    /// Align single frame using pre-allocated buffers (buffer pooling)
    ///
    /// This is the optimized version that reuses buffers across multiple frame alignments.
    /// Only the output buffer is unique per frame (returned in GpuAlignedFrame).
    /// Uses luminance-based alignment with optional CA correction.
    async fn align_single_frame_pooled(
        &self,
        ref_pyramids: &ReferencePyramids,
        comparison: &CameraFrame,
        width: u32,
        height: u32,
        buffers: &AlignmentBuffers,
        ca_coefficients: (f32, f32),
    ) -> Result<GpuAlignedFrame, String> {
        // Upload comparison frame to pooled buffer (overwrites previous)
        let comp_rgba = convert_frame_to_rgba(comparison).await?;
        let comp_f32 = u8_to_f32_normalized(&comp_rgba);
        self.queue
            .write_buffer(&buffers.comp_rgba, 0, bytemuck::cast_slice(&comp_f32));
        // Free CPU buffers immediately after GPU upload to reduce peak memory
        drop(comp_f32);
        drop(comp_rgba);

        self.align_single_frame_from_buffer(ref_pyramids, width, height, buffers, ca_coefficients)
            .await
    }

    /// Core alignment: pyramid align + warp using pre-uploaded comparison buffer
    ///
    /// The comparison RGBA f32 data must already be written to `buffers.comp_rgba`.
    /// Shared by both the RGBA path (`align_single_frame_pooled`) and the Bayer
    /// path (`align_bayer_frames_gpu`).
    async fn align_single_frame_from_buffer(
        &self,
        ref_pyramids: &ReferencePyramids,
        width: u32,
        height: u32,
        buffers: &AlignmentBuffers,
        ca_coefficients: (f32, f32),
    ) -> Result<GpuAlignedFrame, String> {
        let (ca_r_coeff, ca_b_coeff) = ca_coefficients;
        let pixel_count = (width * height) as usize;

        // Output buffer - unique per frame, stays on GPU
        let output_buffer = self.create_storage_buffer(
            "aligned_output_gpu",
            (pixel_count * 4 * std::mem::size_of::<f32>()) as u64,
        );

        // Build comparison luminance pyramid using pooled buffers
        self.build_luminance_pyramid(
            &buffers.comp_rgba,
            &buffers.comp_lum,
            width,
            height,
            buffers,
            "comp",
        );

        // Build pyramid dimensions for tile calculations
        let level_dims: Vec<(u32, u32)> = (0..PYRAMID_LEVELS)
            .map(|level| {
                let scale = 1 << level;
                (width.div_ceil(scale), height.div_ceil(scale))
            })
            .collect();

        // Calculate tile counts for each level
        let level_tile_counts: Vec<(u32, u32)> = (0..PYRAMID_LEVELS)
            .map(|level| {
                let (level_w, level_h) = level_dims[level];
                let (tile_size, _, _) = ALIGN_LEVEL_CONFIGS[level];
                let tile_step = tile_size / 2;
                let n_tiles_x = (level_w.saturating_sub(tile_size)) / tile_step + 1;
                let n_tiles_y = (level_h.saturating_sub(tile_size)) / tile_step + 1;
                (n_tiles_x, n_tiles_y)
            })
            .collect();

        // Hierarchical luminance-based alignment (4 pyramid levels, coarse-to-fine)
        // Uses chunked dispatch to allow GPU preemption for compositor responsiveness
        const ALIGN_ROWS_PER_CHUNK: u32 = 4;
        const CHUNKS_PER_YIELD: u32 = 2;

        let mut prev_n_tiles_x: u32 = 0;
        let mut prev_n_tiles_y: u32 = 0;
        let mut prev_tile_step: u32 = 0;

        for level in (0..PYRAMID_LEVELS).rev() {
            let (level_w, level_h) = level_dims[level];
            let (tile_size, search_dist, use_l2) = ALIGN_LEVEL_CONFIGS[level];
            let tile_step = tile_size / 2;
            let (n_tiles_x, n_tiles_y) = level_tile_counts[level];

            let prev_align = if level == PYRAMID_LEVELS - 1 {
                &buffers.dummy_prev_align
            } else {
                &buffers.align[level + 1]
            };

            let align_bg = self.bind_group(
                &format!("align_L{}", level),
                &self.align_layout,
                &[
                    &ref_pyramids.lum[level],
                    &buffers.comp_lum[level],
                    &buffers.align[level],
                    prev_align,
                    &buffers.align_params[level],
                    &buffers.prev_n_tiles_x[level],
                ],
            );

            let mut row_offset = 0u32;
            let mut chunk_count = 0u32;

            while row_offset < n_tiles_y {
                let rows_this_chunk = ALIGN_ROWS_PER_CHUNK.min(n_tiles_y - row_offset);

                let align_params = AlignParams {
                    width: level_w,
                    height: level_h,
                    tile_size,
                    tile_step,
                    search_dist,
                    n_tiles_x,
                    n_tiles_y,
                    use_l2: u32::from(use_l2),
                    prev_tile_step,
                    prev_n_tiles_y,
                    tile_row_offset: row_offset,
                    _padding1: 0,
                };
                self.queue.write_buffer(
                    &buffers.align_params[level],
                    0,
                    bytemuck::cast_slice(&[align_params]),
                );
                self.queue.write_buffer(
                    &buffers.prev_n_tiles_x[level],
                    0,
                    bytemuck::cast_slice(&[prev_n_tiles_x]),
                );

                let workgroups = (n_tiles_x, rows_this_chunk, 1);
                if level < PYRAMID_LEVELS - 1 {
                    self.dispatch_compute_batch(
                        &format!("align_L{}_{}", level, row_offset),
                        &[
                            (&self.align_tiles, &align_bg, workgroups),
                            (&self.align_correct_upsampling, &align_bg, workgroups),
                        ],
                    );
                } else {
                    self.dispatch_compute(
                        &format!("align_L{}_{}", level, row_offset),
                        &self.align_tiles,
                        &align_bg,
                        workgroups,
                    );
                }

                row_offset += rows_this_chunk;
                chunk_count += 1;

                if chunk_count.is_multiple_of(CHUNKS_PER_YIELD) {
                    self.yield_to_compositor().await;
                }
            }

            self.yield_to_compositor().await;

            prev_n_tiles_x = n_tiles_x;
            prev_n_tiles_y = n_tiles_y;
            prev_tile_step = tile_step;
        }

        // Warp frame using final alignment (L0)
        let (final_n_tiles_x, final_n_tiles_y) = level_tile_counts[0];

        // Enable CA correction if coefficients are non-zero
        let enable_ca = if ca_r_coeff.abs() > 0.0001 || ca_b_coeff.abs() > 0.0001 {
            1u32
        } else {
            0u32
        };
        let warp_params = WarpParams {
            width,
            height,
            n_tiles_x: final_n_tiles_x,
            n_tiles_y: final_n_tiles_y,
            tile_size: WARP_TILE_SIZE,
            tile_step: WARP_TILE_SIZE / 2,
            use_bilinear: 1,
            _padding0: 0,
            center_x: width as f32 / 2.0,
            center_y: height as f32 / 2.0,
            ca_r_coeff,
            ca_b_coeff,
            enable_ca_correction: enable_ca,
            _padding: 0,
            _padding2: 0,
            _padding3: 0,
        };
        self.queue.write_buffer(
            &buffers.warp_params,
            0,
            bytemuck::cast_slice(&[warp_params]),
        );

        let warp_bg = self.bind_group(
            "warp_bg",
            &self.warp_layout,
            &[
                &buffers.comp_rgba,
                &output_buffer,
                &buffers.align[0],
                &buffers.warp_params,
            ],
        );

        self.dispatch_compute(
            "warp_pass",
            &self.warp_frame,
            &warp_bg,
            (width.div_ceil(16), height.div_ceil(16), 1),
        );

        Ok(GpuAlignedFrame {
            buffer: output_buffer,
            width,
            height,
        })
    }

    // Note: Legacy CPU-path functions removed (align_single_frame, align_single_frame_gpu, merge_frames, merge_spatial, merge_fft).
    // Use align_frames_gpu() and merge_frames_gpu() for the optimized GPU-only pipeline.

    /// Merge GPU-resident frames using FFT frequency domain merge
    ///
    /// This is the memory-optimized version that works with GpuAlignedFrame.
    /// The aligned frames stay on GPU throughout, eliminating ~336MB of CPU memory
    /// and avoiding redundant GPU-CPU-GPU transfers.
    pub async fn merge_frames_gpu(
        &self,
        reference: &CameraFrame,
        aligned: &[GpuAlignedFrame],
        config: &BurstModeConfig,
    ) -> Result<MergedFrame, String> {
        debug!(
            frames = aligned.len() + 1,
            "Merging frames (GPU FFT, no CPU round-trip)"
        );

        let width = reference.width;
        let height = reference.height;

        // Convert reference frame to RGBA if needed (handles YUV formats)
        let reference_rgba = convert_frame_to_rgba(reference).await?;

        // Estimate noise using GPU
        let step_start = std::time::Instant::now();
        let noise_sd = self
            .estimate_noise_gpu(&reference_rgba, width, height)
            .await?;
        info!(
            elapsed_ms = step_start.elapsed().as_millis(),
            noise_sd, "Noise estimation complete"
        );

        // Use the GPU-resident merge function
        let result = self
            .fft_pipeline
            .merge_gpu(
                &reference_rgba,
                aligned,
                width,
                height,
                noise_sd,
                config.robustness,
            )
            .await?;

        Ok(MergedFrame {
            data: result,
            width,
            height,
        })
    }

    /// Apply tone mapping using GPU
    pub async fn apply_tonemap(
        &self,
        merged: &MergedFrame,
        config: &BurstModeConfig,
    ) -> Result<MergedFrame, String> {
        debug!("Applying tone mapping (GPU)");

        let width = merged.width;
        let height = merged.height;
        let pixel_count = (width * height) as usize;

        // Convert to f32
        let input_f32 = u8_to_f32_normalized(&merged.data);

        let block_size = 8u32;
        let lum_width = width.div_ceil(block_size);
        let lum_height = height.div_ceil(block_size);
        let lum_size = (lum_width * lum_height) as usize;

        // Create buffers
        let input_buffer = self.create_storage_buffer(
            "tonemap_input",
            (pixel_count * 4 * std::mem::size_of::<f32>()) as u64,
        );

        let output_buffer = self.create_storage_buffer(
            "tonemap_output",
            (pixel_count * 4 * std::mem::size_of::<f32>()) as u64,
        );

        // Local luminance buffer for tone mapping (read_write: shader writes luminance values)
        let local_lum_buffer = self.create_storage_buffer(
            "local_luminance",
            (lum_size * std::mem::size_of::<f32>()) as u64,
        );

        let local_lum_params = LocalLumParams {
            width,
            height,
            block_size,
            lum_width,
            lum_height,
            _padding0: 0,
            _padding1: 0,
            _padding2: 0,
        };

        let local_lum_params_buffer = self.create_uniform_buffer(
            "local_lum_params",
            std::mem::size_of::<LocalLumParams>() as u64,
        );

        // TonemapParams for main tonemap shader
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct TonemapParams {
            width: u32,
            height: u32,
            shadow_boost: f32,
            local_contrast: f32,
            highlight_compress: f32,
            gamma: f32,
            dither_strength: f32,
            avg_brightness: f32, // Scene brightness for adaptive gamma (HDR+ Section 6)
        }

        // TonemapParams will be created later with adaptive shadow_boost
        let params_buffer = self.create_uniform_buffer(
            "tonemap_params",
            std::mem::size_of::<TonemapParams>() as u64,
        );

        // Create uniform buffers for lum_width, lum_height, block_size (single u32 each)
        let lum_width_buffer =
            self.create_uniform_buffer("lum_width", std::mem::size_of::<u32>() as u64);

        let lum_height_buffer =
            self.create_uniform_buffer("lum_height", std::mem::size_of::<u32>() as u64);

        let block_size_buffer =
            self.create_uniform_buffer("block_size", std::mem::size_of::<u32>() as u64);

        // Write data to buffers
        self.queue
            .write_buffer(&input_buffer, 0, bytemuck::cast_slice(&input_f32));
        self.queue.write_buffer(
            &local_lum_params_buffer,
            0,
            bytemuck::cast_slice(&[local_lum_params]),
        );
        // Note: params_buffer will be written later after computing adaptive shadow_boost
        self.queue
            .write_buffer(&lum_width_buffer, 0, bytemuck::cast_slice(&[lum_width]));
        self.queue
            .write_buffer(&lum_height_buffer, 0, bytemuck::cast_slice(&[lum_height]));
        self.queue
            .write_buffer(&block_size_buffer, 0, bytemuck::cast_slice(&[block_size]));

        // Create global brightness accumulator buffer (for adaptive shadow boost - HDR+ paper Section 6)
        // [0] = fixed-point sum (value * 65536), [1] = count
        let brightness_accum_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brightness_accum_buffer"),
            size: 8, // 2 x u32
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Initialize to zeros
        self.queue
            .write_buffer(&brightness_accum_buffer, 0, &[0u8; 8]);

        // Step 1: Compute local luminance map (also accumulates global brightness)
        let local_lum_bind_group = self.bind_group(
            "local_lum_bind_group",
            &self.local_lum_layout,
            &[
                &input_buffer,
                &local_lum_buffer,
                &local_lum_params_buffer,
                &brightness_accum_buffer,
            ],
        );

        // Pass 1: Compute local luminance and accumulate global brightness
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("local_lum_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("local_lum_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tonemap_local_lum);
            pass.set_bind_group(0, Some(&local_lum_bind_group), &[]);
            pass.dispatch_workgroups(lum_width.div_ceil(16), lum_height.div_ceil(16), 1);
        }

        // Copy brightness accumulator to staging for readback
        let brightness_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brightness_staging"),
            size: 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&brightness_accum_buffer, 0, &brightness_staging, 0, 8);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back global brightness to compute adaptive shadow_boost
        info!("Reading brightness accumulator from GPU");
        let brightness_slice = brightness_staging.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        brightness_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.await
            .map_err(|_| "Failed to receive brightness map result")?
            .map_err(|e| format!("Failed to map brightness buffer: {:?}", e))?;
        info!("Brightness buffer mapped successfully");

        let brightness_data = brightness_slice.get_mapped_range();
        let brightness_vals: &[u32] = bytemuck::cast_slice(&brightness_data);
        let sum_fixed = brightness_vals[0] as f64;
        let count = brightness_vals[1] as f64;
        drop(brightness_data);
        brightness_staging.unmap();

        // Compute average brightness (convert from fixed-point)
        let avg_brightness = if count > 0.0 {
            (sum_fixed / count / 65536.0) as f32
        } else {
            0.5 // Fallback to mid-gray
        };

        // Adaptive shadow boost based on scene brightness (HDR+ paper Section 6)
        // Bright scenes need less/no shadow lifting to avoid washed-out appearance
        let adaptive_shadow_boost = if avg_brightness > 0.4 {
            0.0 // Bright scene: no shadow boost
        } else if avg_brightness > 0.2 {
            config.shadow_boost * (0.4 - avg_brightness) / 0.2 // Scale down linearly
        } else {
            config.shadow_boost // Dark scene: full boost
        };

        info!(
            avg_brightness = avg_brightness,
            config_shadow_boost = config.shadow_boost,
            adaptive_shadow_boost = adaptive_shadow_boost,
            "Adaptive tone mapping (HDR+ Section 6)"
        );

        // Update TonemapParams with adaptive shadow boost and brightness for adaptive gamma
        let params = TonemapParams {
            width,
            height,
            shadow_boost: adaptive_shadow_boost,
            local_contrast: config.local_contrast,
            highlight_compress: 0.5,
            gamma: 2.2,
            dither_strength: 1.0 / 255.0,
            avg_brightness, // Pass to shader for adaptive gamma decision
        };
        self.queue
            .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&[params]));

        // Step 2: Apply tone mapping with all 7 bindings
        let tonemap_bind_group = self.bind_group(
            "tonemap_bind_group",
            &self.tonemap_layout,
            &[
                &input_buffer,
                &output_buffer,
                &local_lum_buffer,
                &params_buffer,
                &lum_width_buffer,
                &lum_height_buffer,
                &block_size_buffer,
            ],
        );

        // Create new encoder for tonemap pass (previous encoder was submitted after local_lum)
        let mut tonemap_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("tonemap_encoder"),
                });

        // Pass 2: Apply tone mapping
        {
            let mut pass = tonemap_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tonemap_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tonemap_apply);
            pass.set_bind_group(0, Some(&tonemap_bind_group), &[]);
            pass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
        }

        // Read back result using pooled staging buffer
        let staging_size = (pixel_count * 4 * std::mem::size_of::<f32>()) as u64;
        let staging_buffer = self
            .staging_pool
            .write()
            .unwrap()
            .take_large(&self.device, staging_size);

        tonemap_encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, staging_size);

        self.queue.submit(std::iter::once(tonemap_encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        receiver
            .await
            .map_err(|_| "Failed to receive map result")?
            .map_err(|e| format!("{:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let result_f32: &[f32] = bytemuck::cast_slice(&data);
        let result_u8: Vec<u8> = result_f32
            .iter()
            .map(|&x| (x.clamp(0.0, 1.0) * 255.0) as u8)
            .collect();
        drop(data);
        staging_buffer.unmap();

        // Return staging buffer to pool for reuse
        self.staging_pool
            .write()
            .unwrap()
            .return_large(staging_buffer);

        Ok(MergedFrame {
            data: result_u8,
            width,
            height,
        })
    }

    /// Demosaic merged Bayer planes and apply finishing pipeline (HDR+ Section 6)
    ///
    /// Takes merged Bayer planes (packed as RGBA at half resolution) and produces
    /// full-resolution RGBA output with white balance and colour correction applied.
    ///
    /// This is the final step of Bayer-domain processing, where demosaic happens
    /// only once on the merged result (not N times on individual frames).
    pub async fn demosaic_bayer_planes(
        &self,
        merged_planes_buffer: &wgpu::Buffer,
        half_width: u32,
        half_height: u32,
        colour_gains: Option<[f32; 2]>,
        colour_correction_matrix: Option<[[f32; 3]; 3]>,
    ) -> Result<MergedFrame, String> {
        let full_width = half_width * 2;
        let full_height = half_height * 2;
        let full_pixel_count = (full_width * full_height) as usize;

        info!(
            half_width,
            half_height,
            full_width,
            full_height,
            has_wb = colour_gains.is_some(),
            has_ccm = colour_correction_matrix.is_some(),
            "Demosaicing merged Bayer planes (HDR+ Section 6 finishing)"
        );

        // Create output buffer at full resolution
        let output_buffer = self.create_storage_buffer(
            "bayer_finish_output",
            (full_pixel_count * 4 * std::mem::size_of::<f32>()) as u64,
        );

        // Build params
        let (gain_r, gain_b) = colour_gains.map(|g| (g[0], g[1])).unwrap_or((1.0, 1.0));

        let identity_ccm = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let ccm = colour_correction_matrix.unwrap_or(identity_ccm);

        let params = BayerFinishParams {
            half_width,
            half_height,
            full_width,
            full_height,
            gain_r,
            gain_b,
            _align_pad0: 0,
            _align_pad1: 0,
            ccm_row0: [ccm[0][0], ccm[0][1], ccm[0][2], 0.0],
            ccm_row1: [ccm[1][0], ccm[1][1], ccm[1][2], 0.0],
            ccm_row2: [ccm[2][0], ccm[2][1], ccm[2][2], 0.0],
            use_colour: if colour_gains.is_some() { 1 } else { 0 },
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buffer = self.create_uniform_buffer(
            "bayer_finish_params",
            std::mem::size_of::<BayerFinishParams>() as u64,
        );
        self.queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        let bind_group = self.bind_group(
            "bayer_finish_bind_group",
            &self.bayer_finish_layout,
            &[merged_planes_buffer, &output_buffer, &params_buffer],
        );

        // Dispatch demosaic + finishing
        self.dispatch_compute(
            "bayer_finish",
            &self.bayer_finish,
            &bind_group,
            (full_width.div_ceil(16), full_height.div_ceil(16), 1),
        );

        // Yield to compositor
        self.yield_to_compositor().await;

        // Read back result
        let staging_size = (full_pixel_count * 4 * std::mem::size_of::<f32>()) as u64;
        let staging_buffer = self
            .staging_pool
            .write()
            .unwrap()
            .take_large(&self.device, staging_size);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bayer_finish_readback"),
            });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, staging_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        receiver
            .await
            .map_err(|_| "Failed to receive bayer finish map result")?
            .map_err(|e| format!("Failed to map bayer finish buffer: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let result_f32: &[f32] = bytemuck::cast_slice(&data);
        let result_u8: Vec<u8> = result_f32
            .iter()
            .map(|&x| (x.clamp(0.0, 1.0) * 255.0) as u8)
            .collect();
        drop(data);
        staging_buffer.unmap();
        self.staging_pool
            .write()
            .unwrap()
            .return_large(staging_buffer);

        Ok(MergedFrame {
            data: result_u8,
            width: full_width,
            height: full_height,
        })
    }

    /// Compute sharpness from pre-extracted Bayer planes
    ///
    /// Uses the averaged Bayer planes (all 4 channels → grayscale) for sharpness
    /// computation, avoiding a full debayer just for reference selection.
    async fn compute_sharpness_from_planes(&self, planes: &BayerPlanes) -> Result<f32, String> {
        // Average the 4 Bayer channels to get grayscale, then expand to "RGBA" for
        // the sharpness shader (which expects RGBA f32 input)
        let gray_rgba: Vec<f32> = planes
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|rgba| {
                let gray = (rgba[0] + rgba[1] + rgba[2] + rgba[3]) * 0.25;
                [gray, gray, gray, 1.0]
            })
            .collect();

        self.compute_sharpness_from_f32_rgba(&gray_rgba, planes.width, planes.height)
            .await
    }

    /// Align Bayer frames using grayscale derived from Bayer planes
    ///
    /// HDR+ paper Section 4: "We compute a single grayscale image by averaging
    /// each 2×2 Bayer quad." We use the averaged RGBA Bayer planes as grayscale
    /// for pyramid alignment. This avoids debayering for alignment and works at
    /// half resolution, which is more efficient than full-res alignment.
    async fn align_bayer_frames_gpu(
        &self,
        planes_list: &[BayerPlanes],
        ref_idx: usize,
        progress: &Option<ProgressCallback>,
    ) -> Result<Vec<GpuAlignedFrame>, String> {
        let align_start = std::time::Instant::now();
        let ref_planes = &planes_list[ref_idx];
        let width = ref_planes.width;
        let height = ref_planes.height;
        let pixel_count = (width * height) as usize;

        debug!(
            frame_count = planes_list.len(),
            reference = ref_idx,
            half_width = width,
            half_height = height,
            "Aligning Bayer frames (using averaged grayscale at half-res)"
        );

        let report = |frame_idx: usize, total_frames: usize| {
            if let Some(cb) = progress {
                let frame_progress = (frame_idx + 1) as f32 / total_frames as f32;
                let overall = 0.10 + frame_progress * 0.50;
                cb(overall);
            }
        };

        // Pre-allocate alignment buffers at half-res dimensions
        let buffers = self.create_alignment_buffers(width, height);

        // Upload reference Bayer planes as RGBA f32
        let ref_rgba_buffer = self.create_storage_buffer(
            "align_ref_bayer",
            (pixel_count * 4 * std::mem::size_of::<f32>()) as u64,
        );
        self.queue
            .write_buffer(&ref_rgba_buffer, 0, bytemuck::cast_slice(&ref_planes.data));

        // Build reference luminance pyramid
        // The RGBA data has Bayer planes as channels. The luminance shader (channel=3)
        // uses BT.601 weights, but for Bayer planes we want simple averaging.
        // We'll use channel=3 (luminance mode) which computes:
        //   lum = 0.299*R + 0.587*G + 0.114*B
        // This gives reasonable results since G≈Gr (the dominant component).
        // For more accurate results, we could add a Bayer-specific luminance mode,
        // but the alignment is robust to this approximation.
        let ref_pyramids = self.build_reference_pyramid(&ref_rgba_buffer, width, height, &buffers);

        self.yield_to_compositor().await;

        // CA estimation on half-res reference (still useful for chromatic correction)
        let (ca_r_coeff, ca_b_coeff) = self
            .estimate_ca_coefficients(&ref_rgba_buffer, width, height)
            .await?;
        info!(ca_r_coeff, ca_b_coeff, "CA estimation (Bayer half-res)");

        self.yield_to_compositor().await;

        let ca_coefficients = (ca_r_coeff, ca_b_coeff);

        let mut aligned_frames = Vec::with_capacity(planes_list.len() - 1);
        let total_frames = planes_list.len() - 1;

        for (idx, planes) in planes_list.iter().enumerate() {
            if idx == ref_idx {
                continue;
            }

            if planes.width != width || planes.height != height {
                warn!(
                    frame = idx,
                    "Skipping Bayer frame with mismatched dimensions"
                );
                continue;
            }

            let frame_start = std::time::Instant::now();

            // Upload comparison planes to pooled buffer
            self.queue
                .write_buffer(&buffers.comp_rgba, 0, bytemuck::cast_slice(&planes.data));

            // Reuse the shared alignment core (pyramid align + warp)
            let aligned = self
                .align_single_frame_from_buffer(
                    &ref_pyramids,
                    width,
                    height,
                    &buffers,
                    ca_coefficients,
                )
                .await?;

            info!(
                frame = idx,
                elapsed_ms = frame_start.elapsed().as_millis(),
                "Bayer frame aligned"
            );

            aligned_frames.push(aligned);
            report(aligned_frames.len(), total_frames);
            self.yield_to_compositor().await;
        }

        debug!(
            aligned = aligned_frames.len(),
            total_elapsed_ms = align_start.elapsed().as_millis(),
            "All Bayer frames aligned"
        );
        Ok(aligned_frames)
    }

    /// Merge Bayer frames using FFT pipeline operating per-channel
    ///
    /// HDR+ paper Section 5: The Bayer planes are packed as RGBA, so the existing
    /// vec4 FFT merge naturally operates per-channel. Each channel (R, Gr, B, Gb)
    /// is independently merged with per-channel noise model.
    async fn merge_bayer_frames_gpu(
        &self,
        ref_planes: &BayerPlanes,
        aligned: &[GpuAlignedFrame],
        config: &BurstModeConfig,
    ) -> Result<(wgpu::Buffer, u32, u32), String> {
        let width = ref_planes.width;
        let height = ref_planes.height;

        debug!(
            frames = aligned.len() + 1,
            half_width = width,
            half_height = height,
            "Merging Bayer frames (per-channel FFT, HDR+ Section 5)"
        );

        // Estimate noise from Bayer planes
        // For Bayer planes packed as RGBA, we pack them as u8 for the noise estimator
        let reference_u8: Vec<u8> = ref_planes
            .data
            .iter()
            .map(|&x| (x.clamp(0.0, 1.0) * 255.0) as u8)
            .collect();

        let noise_sd = self
            .estimate_noise_gpu(&reference_u8, width, height)
            .await?;
        info!(noise_sd, "Bayer noise estimation complete");

        // merge_gpu returns u8 CPU data; we re-upload as f32 for the demosaic step
        // TODO: add merge_gpu_to_buffer to avoid this GPU→CPU→GPU round-trip
        let merged_u8 = self
            .fft_pipeline
            .merge_gpu(
                &reference_u8,
                aligned,
                width,
                height,
                noise_sd,
                config.robustness,
            )
            .await?;

        // Re-upload merged result to GPU buffer for demosaic step
        // (merge_gpu currently reads back to CPU; we re-upload the f32 version)
        let pixel_count = (width * height) as usize;
        let merged_f32 = u8_to_f32_normalized(&merged_u8);
        let merged_buffer = self.create_storage_buffer(
            "merged_bayer_planes",
            (pixel_count * 4 * std::mem::size_of::<f32>()) as u64,
        );
        self.queue
            .write_buffer(&merged_buffer, 0, bytemuck::cast_slice(&merged_f32));

        Ok((merged_buffer, width, height))
    }
}

/// Process burst mode capture (full GPU pipeline)
///
/// Uses the memory-optimized GPU-only pipeline that keeps aligned frames on GPU,
/// eliminating CPU round-trips and reducing peak memory usage by ~1-2GB for 12MP bursts.
///
/// Progress stages (when callback is provided):
/// - 0.00 - 0.05: GPU initialization
/// - 0.05 - 0.10: Reference frame selection
/// - 0.10 - 0.60: Frame alignment (distributed across frames)
/// - 0.60 - 0.85: Frame merging
/// - 0.85 - 1.00: Tone mapping
pub async fn process_burst_mode(
    frames: Vec<Arc<CameraFrame>>,
    config: BurstModeConfig,
    progress: Option<ProgressCallback>,
) -> Result<MergedFrame, String> {
    // Detect Bayer input and route to appropriate pipeline
    let is_bayer = frames.first().map(|f| f.format.is_bayer()).unwrap_or(false);

    if is_bayer {
        return process_burst_mode_bayer(frames, config, progress).await;
    }

    process_burst_mode_rgba(frames, config, progress).await
}

/// Bayer-domain burst processing pipeline (HDR+ paper-correct)
///
/// Operates entirely on raw Bayer data:
/// 1. Extract Bayer planes (R, Gr, Gb, B) at half resolution
/// 2. Select reference frame using sharpness on averaged grayscale
/// 3. Align frames using pyramid alignment on grayscale (half-res)
/// 4. Merge per-channel via FFT Wiener filter (per HDR+ Section 5)
/// 5. Demosaic merged result (single demosaic, not N)
/// 6. Apply finishing: white balance → CCM → tone mapping (HDR+ Section 6)
async fn process_burst_mode_bayer(
    frames: Vec<Arc<CameraFrame>>,
    config: BurstModeConfig,
    progress: Option<ProgressCallback>,
) -> Result<MergedFrame, String> {
    if frames.is_empty() {
        return Err("Burst mode requires at least one frame".to_string());
    }
    let total_start = std::time::Instant::now();
    info!(
        frames = frames.len(),
        "Processing burst mode (Bayer-domain pipeline per HDR+ paper)"
    );

    let report = |value: f32| {
        if let Some(cb) = &progress {
            cb(value);
        }
    };

    // Step 1: Extract Bayer planes from all frames (0% - 5%)
    report(0.0);
    let step_start = std::time::Instant::now();
    let mut planes_list: Vec<BayerPlanes> = Vec::with_capacity(frames.len());
    for (i, frame) in frames.iter().enumerate() {
        let planes = extract_bayer_planes(frame)?;
        debug!(
            frame = i,
            half_w = planes.width,
            half_h = planes.height,
            bit_depth = planes.bit_depth,
            "Extracted Bayer planes"
        );
        planes_list.push(planes);
    }
    info!(
        elapsed_ms = step_start.elapsed().as_millis(),
        frames = planes_list.len(),
        "Bayer plane extraction complete"
    );
    report(0.05);

    // Step 2: Initialize GPU pipeline (5% - 8%)
    let step_start = std::time::Instant::now();
    let gpu = BurstModeGpuPipeline::new().await?;
    info!(
        elapsed_ms = step_start.elapsed().as_millis(),
        "GPU pipeline initialized"
    );
    report(0.08);

    // Step 3: Select reference frame from first 3 using sharpness (8% - 10%)
    let step_start = std::time::Instant::now();
    let search_count = planes_list.len().min(3);
    let mut max_sharpness = f32::MIN;
    let mut ref_idx = 0;
    for (idx, planes) in planes_list[..search_count].iter().enumerate() {
        let sharpness = gpu.compute_sharpness_from_planes(planes).await?;
        debug!(frame = idx, sharpness, "Bayer frame sharpness");
        if sharpness > max_sharpness {
            max_sharpness = sharpness;
            ref_idx = idx;
        }
    }
    info!(
        elapsed_ms = step_start.elapsed().as_millis(),
        reference = ref_idx,
        sharpness = max_sharpness,
        "Reference frame selected (Bayer)"
    );
    report(0.10);

    // Step 4: Align frames at half-res using Bayer grayscale (10% - 60%)
    let step_start = std::time::Instant::now();
    let aligned = gpu
        .align_bayer_frames_gpu(&planes_list, ref_idx, &progress)
        .await?;
    info!(
        elapsed_ms = step_start.elapsed().as_millis(),
        aligned = aligned.len(),
        "Bayer frame alignment complete"
    );
    report(0.60);

    // Step 5: Merge per-channel via FFT (60% - 80%)
    let step_start = std::time::Instant::now();
    let ref_planes = &planes_list[ref_idx];
    let (merged_buffer, half_w, half_h) = gpu
        .merge_bayer_frames_gpu(ref_planes, &aligned, &config)
        .await?;
    info!(
        elapsed_ms = step_start.elapsed().as_millis(),
        "Bayer merge complete (per-channel FFT)"
    );
    report(0.80);

    // Drop aligned frames to free GPU memory
    drop(aligned);

    // Step 6: Demosaic merged Bayer planes → full-res RGBA (80% - 90%)
    // HDR+ Section 6: single demosaic after merge (not N demosaics before merge)
    let step_start = std::time::Instant::now();
    let colour_gains = ref_planes.colour_gains;
    let ccm = ref_planes.colour_correction_matrix;
    let demosaiced = gpu
        .demosaic_bayer_planes(&merged_buffer, half_w, half_h, colour_gains, ccm)
        .await?;
    info!(
        elapsed_ms = step_start.elapsed().as_millis(),
        width = demosaiced.width,
        height = demosaiced.height,
        "Demosaic complete (full-res output)"
    );
    report(0.90);

    // Step 7: Apply tone mapping (90% - 100%)
    let step_start = std::time::Instant::now();
    let tonemapped = gpu.apply_tonemap(&demosaiced, &config).await?;
    info!(
        elapsed_ms = step_start.elapsed().as_millis(),
        "Tone mapping complete"
    );
    report(1.0);

    info!(
        total_elapsed_ms = total_start.elapsed().as_millis(),
        width = tonemapped.width,
        height = tonemapped.height,
        "Bayer-domain burst processing complete"
    );

    Ok(tonemapped)
}

/// RGBA-domain burst processing pipeline (original path for non-Bayer input)
async fn process_burst_mode_rgba(
    frames: Vec<Arc<CameraFrame>>,
    config: BurstModeConfig,
    progress: Option<ProgressCallback>,
) -> Result<MergedFrame, String> {
    let total_start = std::time::Instant::now();
    info!(
        frames = frames.len(),
        "Processing burst mode capture (RGBA pipeline)"
    );

    let report = |value: f32| {
        if let Some(cb) = &progress {
            cb(value);
        }
    };

    // Initialize GPU pipeline (0% - 5%)
    report(0.0);
    let step_start = std::time::Instant::now();
    let gpu = BurstModeGpuPipeline::new().await?;
    info!(
        elapsed_ms = step_start.elapsed().as_millis(),
        "GPU pipeline initialized"
    );
    report(0.05);

    // Select reference frame (5% - 10%)
    let step_start = std::time::Instant::now();
    let ref_idx = gpu.select_reference_frame(&frames).await?;
    info!(
        elapsed_ms = step_start.elapsed().as_millis(),
        reference = ref_idx,
        "Reference frame selected"
    );
    report(0.10);

    // Align frames - GPU-only, no CPU round-trip (10% - 60%)
    let step_start = std::time::Instant::now();
    let aligned = gpu
        .align_frames_gpu_with_progress(&frames, ref_idx, &progress)
        .await?;
    info!(
        elapsed_ms = step_start.elapsed().as_millis(),
        aligned = aligned.len(),
        "Frame alignment complete"
    );
    report(0.60);

    // Merge frames - GPU-only (60% - 85%)
    let step_start = std::time::Instant::now();
    let merged = gpu
        .merge_frames_gpu(&frames[ref_idx], &aligned, &config)
        .await?;
    info!(
        elapsed_ms = step_start.elapsed().as_millis(),
        "Frame merge complete"
    );
    report(0.85);

    drop(aligned);

    // Apply tone mapping (85% - 100%)
    let step_start = std::time::Instant::now();
    let tonemapped = gpu.apply_tonemap(&merged, &config).await?;
    info!(
        elapsed_ms = step_start.elapsed().as_millis(),
        "Tone mapping complete"
    );
    report(1.0);

    info!(
        total_elapsed_ms = total_start.elapsed().as_millis(),
        "RGBA burst processing complete"
    );

    Ok(tonemapped)
}

/// Parameters for saving burst mode output
pub struct SaveOutputParams<'a> {
    pub output_dir: std::path::PathBuf,
    pub crop_rect: Option<(u32, u32, u32, u32)>,
    pub encoding_format: super::EncodingFormat,
    pub camera_metadata: super::CameraMetadata,
    pub filter: Option<crate::app::FilterType>,
    pub rotation: SensorRotation,
    pub filename_suffix: Option<&'a str>,
    /// Mirror the final image horizontally (selfie / front-camera mode).
    pub mirror_horizontal: bool,
}

/// Save output image to disk with optional filter, rotation, and aspect ratio cropping
pub async fn save_output(
    frame: &MergedFrame,
    params: SaveOutputParams<'_>,
) -> Result<std::path::PathBuf, String> {
    use super::{EncodingQuality, PhotoEncoder};
    use crate::shaders::apply_filter_gpu_rgba;
    use image::{ImageBuffer, Rgba};
    use std::time::{SystemTime, UNIX_EPOCH};

    let SaveOutputParams {
        output_dir,
        crop_rect,
        encoding_format,
        camera_metadata,
        filter,
        rotation,
        filename_suffix,
        mirror_horizontal,
    } = params;

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let suffix = filename_suffix.unwrap_or("");
    let filename = format!(
        "IMG_{}{}.{}",
        timestamp,
        suffix,
        encoding_format.extension()
    );
    let output_path = output_dir.join(&filename);

    tokio::fs::create_dir_all(&output_dir)
        .await
        .map_err(|e| format!("Failed to create output directory: {}", e))?;

    // Apply filter to the RGBA data if specified and not Standard
    let image_data = match filter {
        Some(f) if f != crate::app::FilterType::Standard => {
            info!(filter = ?f, "Applying filter to burst mode output");
            apply_filter_gpu_rgba(&frame.data, frame.width, frame.height, f)
                .await
                .map_err(|e| format!("Failed to apply filter: {}", e))?
        }
        _ => frame.data.clone(),
    };

    let img: ImageBuffer<Rgba<u8>, _> =
        ImageBuffer::from_raw(frame.width, frame.height, image_data)
            .ok_or("Failed to create image buffer")?;

    let dynamic_img = image::DynamicImage::ImageRgba8(img);

    // Apply crop if specified (for aspect ratio)
    let cropped_img = if let Some((x, y, w, h)) = crop_rect {
        // Validate crop bounds
        let x = x.min(frame.width.saturating_sub(1));
        let y = y.min(frame.height.saturating_sub(1));
        let w = w.min(frame.width - x);
        let h = h.min(frame.height - y);

        if w > 0 && h > 0 {
            debug!(
                x,
                y, w, h, "Applying aspect ratio crop to burst mode output"
            );
            dynamic_img.crop_imm(x, y, w, h)
        } else {
            dynamic_img
        }
    } else {
        dynamic_img
    };

    let rgb_img = cropped_img.to_rgb8();

    // Apply rotation correction if needed
    let (rgb_img, width, height) = if rotation != SensorRotation::None {
        use image::imageops;
        debug!(rotation = ?rotation, "Applying rotation correction to burst mode output");
        let rotated = match rotation {
            SensorRotation::None => rgb_img,
            // 90 CW sensor -> rotate 90 CCW to correct
            SensorRotation::Rotate90 => imageops::rotate270(&rgb_img),
            // 180 sensor -> rotate 180 to correct
            SensorRotation::Rotate180 => imageops::rotate180(&rgb_img),
            // 270 CW sensor -> rotate 90 CW to correct
            SensorRotation::Rotate270 => imageops::rotate90(&rgb_img),
        };
        let (w, h) = rotated.dimensions();
        (rotated, w, h)
    } else {
        let (w, h) = rgb_img.dimensions();
        (rgb_img, w, h)
    };

    // Mirror horizontally if requested (front-camera selfie mode). Done after
    // rotation so the user-visible orientation is upright before flipping.
    let rgb_img = if mirror_horizontal {
        debug!("Mirroring burst output horizontally");
        let mut img = rgb_img;
        image::imageops::flip_horizontal_in_place(&mut img);
        img
    } else {
        rgb_img
    };

    // Create a PhotoEncoder for the selected format
    let mut encoder = PhotoEncoder::new();
    encoder.set_format(encoding_format);
    encoder.set_quality(EncodingQuality::High);
    encoder.set_camera_metadata(camera_metadata);

    // Create processed image from RGB data
    let processed = super::processing::ProcessedImage {
        image: rgb_img,
        width,
        height,
    };

    // Encode and save using the standard photo pipeline
    let encoded = encoder.encode(processed).await?;

    // Save the encoded data
    let output_path_clone = output_path.clone();
    let data = encoded.data;
    tokio::task::spawn_blocking(move || {
        std::fs::write(&output_path_clone, data).map_err(|e| format!("Failed to save image: {}", e))
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))??;

    Ok(output_path)
}

/// Export raw burst frames as PNG files for testing/debugging
///
/// Saves each frame in the burst as a separate PNG file, useful for
/// debugging the burst mode pipeline or providing test data.
///
/// # Arguments
/// * `frames` - Vector of camera frames to export
/// * `output_dir` - Directory to save the frames
///
/// # Returns
/// Vector of paths to the saved frame files
pub async fn export_raw_frames(
    frames: &[Arc<CameraFrame>],
    output_dir: std::path::PathBuf,
) -> Result<Vec<std::path::PathBuf>, String> {
    use image::{ImageBuffer, Rgba};
    use std::time::{SystemTime, UNIX_EPOCH};

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Create a subdirectory for this burst
    let burst_dir = output_dir.join(format!("burst_{}", timestamp));
    tokio::fs::create_dir_all(&burst_dir)
        .await
        .map_err(|e| format!("Failed to create burst directory: {}", e))?;

    let mut saved_paths = Vec::with_capacity(frames.len());

    for (i, frame) in frames.iter().enumerate() {
        let filename = format!("frame_{:03}.png", i);
        let output_path = burst_dir.join(&filename);

        // Convert frame to RGBA if needed (handles YUV formats)
        let rgba_data = convert_frame_to_rgba(frame)
            .await
            .map_err(|e| format!("Failed to convert frame {} to RGBA: {}", i, e))?;

        let img: ImageBuffer<Rgba<u8>, _> =
            ImageBuffer::from_raw(frame.width, frame.height, rgba_data)
                .ok_or_else(|| format!("Failed to create image buffer for frame {}", i))?;

        let output_path_clone = output_path.clone();
        tokio::task::spawn_blocking(move || {
            img.save_with_format(&output_path_clone, image::ImageFormat::Png)
                .map_err(|e| format!("Failed to save frame {}: {}", i, e))
        })
        .await
        .map_err(|e| format!("Task join error for frame {}: {}", i, e))??;

        saved_paths.push(output_path);
        debug!(frame = i, path = ?saved_paths.last(), "Exported raw frame");
    }

    info!(
        burst_dir = ?burst_dir,
        frame_count = frames.len(),
        "Exported raw burst frames"
    );

    Ok(saved_paths)
}

/// Export raw burst frames as DNG files for sharing/debugging
///
/// Saves each frame in the burst as a separate DNG file with camera metadata.
/// This allows users to share raw burst data for debugging the processing pipeline
/// without needing access to the hardware camera.
///
/// # Arguments
/// * `frames` - Vector of camera frames to export
/// * `output_dir` - Directory to save the frames
/// * `camera_metadata` - Camera metadata to embed in DNG files
///
/// # Returns
/// Path to the burst directory containing the DNG files
pub async fn export_burst_frames_dng(
    frames: &[Arc<CameraFrame>],
    output_dir: std::path::PathBuf,
    camera_metadata: &super::CameraMetadata,
) -> Result<std::path::PathBuf, String> {
    use super::{EncodingFormat, EncodingQuality, PhotoEncoder};
    use image::{ImageBuffer, Rgba};
    use std::time::{SystemTime, UNIX_EPOCH};

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Create a subdirectory for this burst
    let burst_dir = output_dir.join(format!("burst_raw_{}", timestamp));
    tokio::fs::create_dir_all(&burst_dir)
        .await
        .map_err(|e| format!("Failed to create burst directory: {}", e))?;

    info!(
        burst_dir = ?burst_dir,
        frame_count = frames.len(),
        "Exporting raw burst frames as DNG"
    );

    for (i, frame) in frames.iter().enumerate() {
        let filename = format!("frame_{:03}.dng", i);
        let output_path = burst_dir.join(&filename);

        // Convert frame to RGBA if needed (handles YUV formats)
        let rgba_data = convert_frame_to_rgba(frame)
            .await
            .map_err(|e| format!("Failed to convert frame {} to RGBA: {}", i, e))?;

        let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
            ImageBuffer::from_raw(frame.width, frame.height, rgba_data)
                .ok_or_else(|| format!("Failed to create image buffer for frame {}", i))?;

        let rgb_img = image::DynamicImage::ImageRgba8(img).to_rgb8();
        let (width, height) = rgb_img.dimensions();

        // Create encoder for DNG format
        let mut encoder = PhotoEncoder::new();
        encoder.set_format(EncodingFormat::Dng);
        encoder.set_quality(EncodingQuality::High);
        encoder.set_camera_metadata(camera_metadata.clone());

        // Create processed image
        let processed = super::processing::ProcessedImage {
            image: rgb_img,
            width,
            height,
        };

        // Encode as DNG
        let encoded = encoder.encode(processed).await?;

        // Save to disk
        let output_path_clone = output_path.clone();
        let data = encoded.data;
        tokio::task::spawn_blocking(move || {
            std::fs::write(&output_path_clone, data)
                .map_err(|e| format!("Failed to save frame {}: {}", i, e))
        })
        .await
        .map_err(|e| format!("Task join error for frame {}: {}", i, e))??;

        debug!(frame = i, path = ?output_path, "Exported raw frame as DNG");
    }

    info!(
        burst_dir = ?burst_dir,
        frame_count = frames.len(),
        "Exported raw burst frames as DNG"
    );

    Ok(burst_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BurstModeConfig::default();
        assert_eq!(config.frame_count, 8);
        assert!(!config.export_raw_frames);
    }

    /// Validate that a WGSL shader compiles successfully using naga
    fn validate_shader(name: &str, source: &str) {
        let result = naga::front::wgsl::parse_str(source);
        match result {
            Ok(module) => {
                // Validate the parsed module
                let info = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                )
                .validate(&module);

                if let Err(e) = info {
                    panic!("Shader '{}' validation failed: {:?}", name, e);
                }
            }
            Err(e) => {
                panic!("Shader '{}' parse failed: {:?}", name, e);
            }
        }
    }

    #[test]
    fn test_pyramid_shader_validates() {
        validate_shader("pyramid", PYRAMID_SHADER);
    }

    #[test]
    fn test_sharpness_shader_validates() {
        validate_shader("sharpness", SHARPNESS_SHADER);
    }

    #[test]
    fn test_align_tile_shader_validates() {
        validate_shader("align_tile", ALIGN_TILE_SHADER);
    }

    #[test]
    fn test_warp_shader_validates() {
        validate_shader("warp", WARP_SHADER);
    }

    #[test]
    fn test_tonemap_shader_validates() {
        validate_shader("tonemap", TONEMAP_SHADER);
    }

    #[test]
    fn test_noise_estimate_shader_validates() {
        validate_shader("noise_estimate", NOISE_ESTIMATE_SHADER);
    }

    #[test]
    fn test_common_utilities_documented() {
        // Verify the common.wgsl reference file exists and documents utilities
        // that are actually used in the shaders
        assert!(
            COMMON_SHADER_REF.contains("BT.601 RGB to Luminance"),
            "common.wgsl should document luminance conversion"
        );
        assert!(
            COMMON_SHADER_REF.contains("Raised Cosine Window"),
            "common.wgsl should document window function"
        );
        assert!(
            COMMON_SHADER_REF.contains("Laplacian Edge Detection"),
            "common.wgsl should document Laplacian operator"
        );
    }

    #[test]
    fn test_shaders_use_documented_utilities() {
        // Verify shaders actually use the utilities documented in common.wgsl
        // BT.601 luminance: 0.299 * R + 0.587 * G + 0.114 * B
        assert!(
            TONEMAP_SHADER.contains("0.299") && TONEMAP_SHADER.contains("0.587"),
            "tonemap.wgsl should use BT.601 coefficients"
        );
        assert!(
            SHARPNESS_SHADER.contains("0.299") && SHARPNESS_SHADER.contains("0.587"),
            "sharpness.wgsl should use BT.601 coefficients"
        );
        assert!(
            PYRAMID_SHADER.contains("0.299") && PYRAMID_SHADER.contains("0.587"),
            "pyramid.wgsl should use BT.601 coefficients"
        );
    }
}
