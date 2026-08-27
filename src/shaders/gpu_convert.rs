// SPDX-License-Identifier: GPL-3.0-only
//! GPU-accelerated format conversion pipelines
//!
//! Uses a unified compute shader (`yuv_convert.wgsl`) for all YUV→RGBA conversions,
//! with format selection via a uniform parameter. Debayer uses a separate shader.
//!
//! **Supported formats:**
//! - NV12/NV21: Semi-planar 4:2:0
//! - I420: Planar 4:2:0 (also handles I422/I444 from MJPEG)
//! - YUYV/UYVY/YVYU/VYUY: Packed 4:2:2
//! - Gray8/RGB24/ABGR/BGRA: Single-plane conversions
//! - Bayer: Raw sensor data (separate debayer shader)

use crate::app::FilterType;
use crate::backends::camera::types::{CameraFrame, PixelFormat};
use crate::backends::camera::v4l2_utils::detect_csi2_bit_depth;
use crate::gpu::{self, wgpu};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Workgroup size for the CSI-2 unpack shader (one thread per pixel pair).
const UNPACK_WORKGROUP_SIZE: u32 = 256;

/// Workgroup size for the debayer, AWB, and YUV conversion shaders (16×16 tile).
const TILE_WORKGROUP_SIZE: u32 = 16;

/// Required byte alignment for `copy_buffer_to_texture` / `copy_texture_to_buffer` row stride.
const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

/// Round `value` up to the next multiple of [`COPY_BYTES_PER_ROW_ALIGNMENT`].
fn align_to_copy_row(value: u32) -> u32 {
    (value + COPY_BYTES_PER_ROW_ALIGNMENT - 1) & !(COPY_BYTES_PER_ROW_ALIGNMENT - 1)
}

/// YUV conversion parameters (32 bytes, matches yuv_convert.wgsl)
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvertParams {
    width: u32,
    height: u32,
    format: u32,
    y_stride: u32,
    uv_stride: u32,
    v_stride: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Debayer conversion parameters (80 bytes, std140-compatible)
/// White balance gains are now in a separate storage buffer (AwbGains).
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DebayerParams {
    width: u32,
    height: u32,
    pattern: u32,        // 0=RGGB, 1=BGGR, 2=GRBG, 3=GBRG
    use_isp_colour: u32, // 1 = apply gains+CCM, 0 = raw output
    black_level: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    ccm_row0: [f32; 4], // xyz used, w=pad
    ccm_row1: [f32; 4], // xyz used, w=pad
    ccm_row2: [f32; 4], // xyz used, w=pad
}

/// White balance gains stored in a GPU storage buffer.
/// Written by CPU (ISP gains) or by the GPU AWB finalize shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AwbGains {
    gain_r: f32,
    gain_b: f32,
}

/// AWB accumulation sums (used to clear the buffer before each AWB pass)
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AwbSums {
    sum_r: u32,
    sum_g: u32,
    sum_b: u32,
    _pad: u32,
}

/// Input frame data for conversion
pub struct GpuFrameInput<'a> {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    /// Y plane data (or packed data for YUYV variants)
    pub y_data: &'a [u8],
    pub y_stride: u32,
    /// UV plane data (NV12: interleaved UV, I420: U plane)
    pub uv_data: Option<&'a [u8]>,
    pub uv_stride: u32,
    /// V plane data (I420 only)
    pub v_data: Option<&'a [u8]>,
    pub v_stride: u32,
    /// ISP white balance gains [R, B] (Bayer only)
    pub colour_gains: Option<[f32; 2]>,
    /// 3x3 colour correction matrix (row-major, Bayer only)
    pub colour_correction_matrix: Option<[[f32; 3]; 3]>,
    /// Sensor black level normalized to 0..1 (Bayer only)
    pub black_level: Option<f32>,
}

impl<'a> GpuFrameInput<'a> {
    /// Build a `GpuFrameInput` from a `CameraFrame`.
    ///
    /// Handles all YUV, packed, single-plane, and Bayer formats.
    /// Callers must handle `PixelFormat::RGBA` separately (it needs no conversion).
    pub fn from_camera_frame(frame: &'a CameraFrame) -> Result<Self, String> {
        if frame.format == PixelFormat::RGBA {
            return Err("RGBA format doesn't need conversion".to_string());
        }

        let buffer_data = frame.data.as_ref();

        let mut input = GpuFrameInput {
            format: frame.format,
            width: frame.width,
            height: frame.height,
            y_data: buffer_data,
            y_stride: frame.stride,
            uv_data: None,
            uv_stride: 0,
            v_data: None,
            v_stride: 0,
            colour_gains: None,
            colour_correction_matrix: None,
            black_level: None,
        };

        match frame.format {
            PixelFormat::NV12 | PixelFormat::NV21 => {
                let planes = frame
                    .yuv_planes
                    .as_ref()
                    .ok_or("NV12/NV21 frame missing yuv_planes")?;
                input.y_data = &buffer_data[planes.y_offset..planes.y_offset + planes.y_size];
                input.uv_data =
                    Some(&buffer_data[planes.uv_offset..planes.uv_offset + planes.uv_size]);
                input.uv_stride = planes.uv_stride;
            }
            PixelFormat::I420 => {
                let planes = frame
                    .yuv_planes
                    .as_ref()
                    .ok_or("I420 frame missing yuv_planes")?;
                input.y_data = &buffer_data[planes.y_offset..planes.y_offset + planes.y_size];
                input.uv_data =
                    Some(&buffer_data[planes.uv_offset..planes.uv_offset + planes.uv_size]);
                input.uv_stride = planes.uv_stride;
                if planes.v_size > 0 {
                    input.v_data =
                        Some(&buffer_data[planes.v_offset..planes.v_offset + planes.v_size]);
                }
                input.v_stride = planes.v_stride;
            }
            PixelFormat::BayerRGGB
            | PixelFormat::BayerBGGR
            | PixelFormat::BayerGRBG
            | PixelFormat::BayerGBRG => {
                let meta = frame.libcamera_metadata.as_ref();
                input.colour_gains = meta.and_then(|m| m.colour_gains);
                input.colour_correction_matrix = meta.and_then(|m| m.colour_correction_matrix);
                input.black_level = meta.and_then(|m| m.black_level);
            }
            _ => {} // YUYV, UYVY, YVYU, VYUY, Gray8, RGB24, ABGR, BGRA: defaults are correct
        }

        Ok(input)
    }
}

/// Filter parameters uniform (matches filter_compute.wgsl)
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FilterParams {
    width: u32,
    height: u32,
    filter_mode: u32,
    _padding: u32,
}

/// CSI-2 unpack parameters (32 bytes, matches unpack_csi2.wgsl)
///
/// WGSL uniform structs are rounded up to 16-byte alignment (roundUp(16, maxFieldAlign)),
/// so 5 × u32 = 20 bytes becomes 32 bytes with 12 bytes of trailing padding.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UnpackParams {
    width: u32,
    height: u32,
    packed_stride: u32,
    bit_depth: u32,         // 10, 12, or 14
    output_stride_u32: u32, // padded row stride in u32 units
    _pad: [u32; 3],         // align to 32 bytes for WGSL uniform struct layout
}

/// Binding type specification for pipeline creation
#[derive(Clone, Copy)]
enum BindingSpec {
    Texture,
    FilterableTexture,
    StorageTexture,
    StorageBuffer,
    ReadOnlyStorageBuffer,
    Uniform,
    Sampler,
}

impl BindingSpec {
    fn to_layout_entry(self, binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: match self {
                BindingSpec::Texture => wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                BindingSpec::FilterableTexture => wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                BindingSpec::StorageTexture => wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                BindingSpec::StorageBuffer => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                BindingSpec::ReadOnlyStorageBuffer => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                BindingSpec::Uniform => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                BindingSpec::Sampler => {
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
                }
            },
            count: None,
        }
    }
}

/// Pipeline resources (shader + bind group layout)
struct FormatPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

/// GPU pipeline for format conversion
///
/// Uses a single unified shader (`yuv_convert.wgsl`) for all YUV formats,
/// with a separate debayer shader for Bayer sensor data.
pub struct GpuConvertPipeline {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// Unified YUV→RGBA pipeline (shared across all YUV formats)
    yuv_pipeline: Option<FormatPipeline>,
    /// Separate debayer pipeline (different uniform struct)
    debayer_pipeline: Option<FormatPipeline>,
    /// GPU AWB accumulation pipeline
    awb_pipeline: Option<FormatPipeline>,
    /// GPU AWB finalize pipeline (1 thread: sums → gains)
    awb_finalize_pipeline: Option<FormatPipeline>,
    /// Storage buffer for AWB channel sums (16 bytes, atomics in AWB shader)
    awb_sums_buffer: wgpu::Buffer,
    /// Storage buffer for AWB gains (8 bytes, read by debayer shader)
    awb_gains_buffer: wgpu::Buffer,
    /// CSI-2 unpack compute pipeline
    unpack_pipeline: Option<FormatPipeline>,
    /// Storage buffer for packed CSI-2 input data
    packed_input_buffer: Option<wgpu::Buffer>,
    packed_input_buffer_size: u64,
    /// Storage buffer for unpacked output data (u16 pairs as u32)
    unpack_output_buffer: Option<wgpu::Buffer>,
    unpack_output_buffer_size: u64,
    /// Uniform buffer for unpack parameters
    unpack_uniform_buffer: wgpu::Buffer,
    /// Integrated filter pipeline (runs on same device, avoids GPU round trip)
    filter_pipeline: Option<FormatPipeline>,
    filter_uniform_buffer: wgpu::Buffer,
    filter_sampler: wgpu::Sampler,
    filter_output_buffer: Option<wgpu::Buffer>,
    filter_staging_buffer: Option<wgpu::Buffer>,
    filter_cached_width: u32,
    filter_cached_height: u32,
    uniform_buffer: wgpu::Buffer,
    // Cached resources for current dimensions/format
    cached_width: u32,
    cached_height: u32,
    cached_format: PixelFormat,
    cached_uv_dims: Option<(u32, u32)>,
    tex_y: Option<wgpu::Texture>,
    tex_uv: Option<wgpu::Texture>,
    tex_v: Option<wgpu::Texture>,
    output_texture: Option<wgpu::Texture>,
    output_view: Option<wgpu::TextureView>,
    /// Cached staging buffer for `read_rgba_to_cpu` readback
    readback_staging_buffer: Option<wgpu::Buffer>,
    readback_staging_buffer_size: u64,
}

impl GpuConvertPipeline {
    /// Create a new conversion pipeline
    pub async fn new() -> Result<Self, String> {
        info!("Initializing format conversion pipelines");

        let gpu = gpu::get_shared_gpu().await?;
        let device = gpu.device;
        let queue = gpu.queue;

        info!(
            adapter_name = %gpu.info.adapter_name,
            adapter_backend = ?gpu.info.backend,
            low_priority = gpu.info.low_priority_enabled,
            "Using shared GPU device for format conversion"
        );

        // Create uniform buffer (shared across all pipelines, sized for largest params struct)
        let uniform_size =
            std::mem::size_of::<ConvertParams>().max(std::mem::size_of::<DebayerParams>());
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("convert_uniform_buffer"),
            size: uniform_size as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // AWB storage buffers
        let awb_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("awb_sums_buffer"),
            size: std::mem::size_of::<AwbSums>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let awb_gains_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("awb_gains_buffer"),
            size: std::mem::size_of::<AwbGains>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Unpack pipeline uniform buffer (created eagerly, small fixed size)
        let unpack_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("unpack_uniform_buffer"),
            size: std::mem::size_of::<UnpackParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Filter pipeline resources (separate uniform buffer for FilterParams)
        let filter_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("convert_filter_uniform"),
            size: std::mem::size_of::<FilterParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let filter_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("convert_filter_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Ok(Self {
            device,
            queue,
            yuv_pipeline: None,
            debayer_pipeline: None,
            awb_pipeline: None,
            awb_finalize_pipeline: None,
            awb_sums_buffer,
            awb_gains_buffer,
            unpack_pipeline: None,
            packed_input_buffer: None,
            packed_input_buffer_size: 0,
            unpack_output_buffer: None,
            unpack_output_buffer_size: 0,
            unpack_uniform_buffer,
            filter_pipeline: None,
            filter_uniform_buffer,
            filter_sampler,
            filter_output_buffer: None,
            filter_staging_buffer: None,
            filter_cached_width: 0,
            filter_cached_height: 0,
            uniform_buffer,
            cached_width: 0,
            cached_height: 0,
            cached_format: PixelFormat::RGBA,
            cached_uv_dims: None,
            readback_staging_buffer: None,
            readback_staging_buffer_size: 0,
            tex_y: None,
            tex_uv: None,
            tex_v: None,
            output_texture: None,
            output_view: None,
        })
    }

    // Bind group layout: 3 textures + output + params (unified YUV shader)
    const BIND_LAYOUT_YUV: [(u32, BindingSpec); 5] = [
        (0, BindingSpec::Texture),        // tex_y
        (1, BindingSpec::Texture),        // tex_uv
        (2, BindingSpec::Texture),        // tex_v
        (3, BindingSpec::StorageTexture), // output
        (4, BindingSpec::Uniform),        // params
    ];

    // Bind group layout: tex_bayer + output + params + awb_gains (debayer)
    const BIND_LAYOUT_DEBAYER: [(u32, BindingSpec); 4] = [
        (0, BindingSpec::Texture),               // tex_bayer
        (1, BindingSpec::StorageTexture),        // output
        (2, BindingSpec::Uniform),               // DebayerParams
        (3, BindingSpec::ReadOnlyStorageBuffer), // AwbGains
    ];

    // Bind group layout: tex_bayer + params + sums (AWB accumulate)
    const BIND_LAYOUT_AWB: [(u32, BindingSpec); 3] = [
        (0, BindingSpec::Texture),       // tex_bayer
        (1, BindingSpec::Uniform),       // AwbParams (reuses uniform_buffer)
        (2, BindingSpec::StorageBuffer), // AwbSums (read_write for atomics)
    ];

    // Bind group layout: sums + gains (AWB finalize)
    const BIND_LAYOUT_AWB_FINALIZE: [(u32, BindingSpec); 2] = [
        (0, BindingSpec::ReadOnlyStorageBuffer), // AwbSums (read-only)
        (1, BindingSpec::StorageBuffer),         // AwbGains (read_write)
    ];

    // Bind group layout for CSI-2 unpack: packed input, unpacked output, params
    const BIND_LAYOUT_UNPACK: [(u32, BindingSpec); 3] = [
        (0, BindingSpec::ReadOnlyStorageBuffer), // packed_data
        (1, BindingSpec::StorageBuffer),         // unpacked_data
        (2, BindingSpec::Uniform),               // UnpackParams
    ];

    // Bind group layout for integrated filter (same as GpuFilterPipeline)
    const BIND_LAYOUT_FILTER: [(u32, BindingSpec); 4] = [
        (0, BindingSpec::FilterableTexture), // input texture (debayer output)
        (1, BindingSpec::StorageBuffer),     // output buffer (packed RGBA u32)
        (2, BindingSpec::Uniform),           // FilterParams
        (3, BindingSpec::Sampler),           // linear filtering sampler
    ];

    /// Ensure the unified YUV pipeline exists
    fn ensure_yuv_pipeline(&mut self) {
        if self.yuv_pipeline.is_none() {
            debug!("Creating unified YUV conversion pipeline");
            self.yuv_pipeline = Some(self.create_pipeline(
                include_str!("yuv_convert.wgsl"),
                "yuv_convert",
                &Self::BIND_LAYOUT_YUV,
            ));
        }
    }

    /// Ensure the debayer pipeline exists
    fn ensure_debayer_pipeline(&mut self) {
        if self.debayer_pipeline.is_none() {
            debug!("Creating debayer pipeline");
            let shader_source = format!(
                "{}\n{}",
                super::RAW_COLOUR_FUNCTIONS,
                include_str!("debayer.wgsl")
            );
            self.debayer_pipeline =
                Some(self.create_pipeline(&shader_source, "debayer", &Self::BIND_LAYOUT_DEBAYER));
        }
    }

    /// Ensure the AWB accumulation pipeline exists
    fn ensure_awb_pipeline(&mut self) {
        if self.awb_pipeline.is_none() {
            debug!("Creating AWB accumulation pipeline");
            self.awb_pipeline = Some(self.create_pipeline(
                include_str!("bayer_awb.wgsl"),
                "bayer_awb",
                &Self::BIND_LAYOUT_AWB,
            ));
        }
    }

    /// Ensure the AWB finalize pipeline exists
    fn ensure_awb_finalize_pipeline(&mut self) {
        if self.awb_finalize_pipeline.is_none() {
            debug!("Creating AWB finalize pipeline");
            self.awb_finalize_pipeline = Some(self.create_pipeline(
                include_str!("bayer_awb_finalize.wgsl"),
                "bayer_awb_finalize",
                &Self::BIND_LAYOUT_AWB_FINALIZE,
            ));
        }
    }

    /// Precompile all shader pipelines so the first frame doesn't pay compilation cost.
    pub fn warmup_pipelines(&mut self) {
        let start = std::time::Instant::now();
        self.ensure_yuv_pipeline();
        self.ensure_debayer_pipeline();
        self.ensure_awb_pipeline();
        self.ensure_awb_finalize_pipeline();
        self.ensure_filter_pipeline();
        self.ensure_unpack_pipeline();
        info!(
            elapsed_ms = format!("{:.1}", start.elapsed().as_millis()),
            "GPU convert pipelines warmed up"
        );
    }

    /// Ensure the integrated filter pipeline exists (for Bayer+filter without GPU round trip)
    fn ensure_filter_pipeline(&mut self) {
        if self.filter_pipeline.is_none() {
            debug!("Creating integrated filter pipeline");
            let shader_source = format!(
                "{}\n{}",
                super::FILTER_FUNCTIONS,
                include_str!("filter_compute.wgsl")
            );
            self.filter_pipeline = Some(self.create_pipeline(
                &shader_source,
                "convert_filter",
                &Self::BIND_LAYOUT_FILTER,
            ));
        }
    }

    /// Ensure filter output/staging buffers are allocated for the given dimensions
    fn ensure_filter_resources(&mut self, width: u32, height: u32) {
        if self.filter_cached_width == width && self.filter_cached_height == height {
            return;
        }

        let buffer_size = (width * height * 4) as u64;

        self.filter_output_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("convert_filter_output"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        self.filter_staging_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("convert_filter_staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));

        self.filter_cached_width = width;
        self.filter_cached_height = height;
    }

    /// Ensure the CSI-2 unpack pipeline exists
    fn ensure_unpack_pipeline(&mut self) {
        if self.unpack_pipeline.is_none() {
            debug!("Creating CSI-2 unpack pipeline");
            self.unpack_pipeline = Some(self.create_pipeline(
                include_str!("unpack_csi2.wgsl"),
                "unpack_csi2",
                &Self::BIND_LAYOUT_UNPACK,
            ));
        }
    }

    /// Ensure unpack input/output buffers are large enough.
    /// Returns `output_stride_u32` (padded row stride in u32 units for 256-byte alignment).
    fn ensure_unpack_buffers(&mut self, packed_size: u64, width: u32, height: u32) -> u32 {
        // Output stride: width pixels × 2 bytes each, padded to 256 bytes (required by copy_buffer_to_texture)
        let bytes_per_row = align_to_copy_row(width * 2);
        let output_stride_u32 = bytes_per_row / 4;
        let output_size = bytes_per_row as u64 * height as u64;

        // Pad packed input to 4-byte alignment for array<u32> access
        let padded_packed_size = (packed_size + 3) & !3;

        if padded_packed_size > self.packed_input_buffer_size {
            self.packed_input_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("unpack_packed_input"),
                size: padded_packed_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.packed_input_buffer_size = padded_packed_size;
        }

        if output_size > self.unpack_output_buffer_size {
            self.unpack_output_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("unpack_output"),
                size: output_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.unpack_output_buffer_size = output_size;
        }

        output_stride_u32
    }

    /// Encode the CSI-2 unpack compute pass and buffer-to-texture copy.
    fn encode_unpack_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        tex_y: &wgpu::Texture,
        width: u32,
        height: u32,
        output_stride_u32: u32,
    ) -> Result<(), String> {
        let unpack_pipeline = self
            .unpack_pipeline
            .as_ref()
            .ok_or("Unpack pipeline not created")?;
        let packed_buf = self
            .packed_input_buffer
            .as_ref()
            .ok_or("Packed input buffer not allocated")?;
        let output_buf = self
            .unpack_output_buffer
            .as_ref()
            .ok_or("Unpack output buffer not allocated")?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("unpack_bind_group"),
            layout: &unpack_pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: packed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.unpack_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch: each thread handles one pixel pair
        let workgroups_x = (width / 2).div_ceil(UNPACK_WORKGROUP_SIZE);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("unpack_csi2_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&unpack_pipeline.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups_x, height, 1);
        }

        // Copy unpacked buffer to R16Unorm texture
        let bytes_per_row = output_stride_u32 * 4;
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: output_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: tex_y,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        Ok(())
    }

    /// Create a compute pipeline with the given shader and bind group layout spec
    fn create_pipeline<const N: usize>(
        &self,
        shader_source: &str,
        name: &str,
        layout_spec: &[(u32, BindingSpec); N],
    ) -> FormatPipeline {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{}_shader", name)),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let entries: Vec<wgpu::BindGroupLayoutEntry> = layout_spec
            .iter()
            .map(|(binding, spec)| spec.to_layout_entry(*binding))
            .collect();

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", name)),
                    entries: &entries,
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{}_pipeline_layout", name)),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{}_pipeline", name)),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        FormatPipeline {
            pipeline,
            bind_group_layout,
        }
    }

    /// Ensure textures are allocated for the given dimensions and format.
    /// `uv_dims` overrides the default UV texture dimensions when provided
    /// (e.g. for MJPEG-decoded I422 reported as I420).
    fn ensure_resources(
        &mut self,
        width: u32,
        height: u32,
        format: PixelFormat,
        uv_dims: Option<(u32, u32)>,
    ) {
        if self.cached_width == width
            && self.cached_height == height
            && self.cached_format == format
            && self.cached_uv_dims == uv_dims
        {
            return;
        }

        debug!(width, height, ?format, "Allocating conversion resources");

        // Calculate texture dimensions based on format, with override for actual UV dims
        let (uv_width, uv_height) = uv_dims.unwrap_or(match format {
            PixelFormat::NV12 | PixelFormat::NV21 | PixelFormat::I420 => (width / 2, height / 2),
            PixelFormat::YUYV | PixelFormat::UYVY | PixelFormat::YVYU | PixelFormat::VYUY => {
                (width / 2, height)
            }
            PixelFormat::Gray8
            | PixelFormat::RGBA
            | PixelFormat::RGB24
            | PixelFormat::ABGR
            | PixelFormat::BGRA
            | PixelFormat::BayerRGGB
            | PixelFormat::BayerBGGR
            | PixelFormat::BayerGRBG
            | PixelFormat::BayerGBRG => (1, 1),
        });

        // Y plane texture format and dimensions
        let (y_format, y_width) = match format {
            PixelFormat::YUYV | PixelFormat::UYVY | PixelFormat::YVYU | PixelFormat::VYUY => {
                (wgpu::TextureFormat::Rgba8Unorm, width / 2)
            }
            PixelFormat::RGBA | PixelFormat::RGB24 | PixelFormat::ABGR | PixelFormat::BGRA => {
                (wgpu::TextureFormat::Rgba8Unorm, width)
            }
            PixelFormat::BayerRGGB
            | PixelFormat::BayerBGGR
            | PixelFormat::BayerGRBG
            | PixelFormat::BayerGBRG => (wgpu::TextureFormat::R16Unorm, width),
            _ => (wgpu::TextureFormat::R8Unorm, width),
        };

        // UV plane texture format
        let uv_format = match format {
            PixelFormat::NV12 | PixelFormat::NV21 => wgpu::TextureFormat::Rg8Unorm,
            _ => wgpu::TextureFormat::R8Unorm,
        };

        // Create Y texture
        self.tex_y = Some(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("convert_tex_y"),
            size: wgpu::Extent3d {
                width: y_width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: y_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));

        // Create UV texture
        self.tex_uv = Some(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("convert_tex_uv"),
            size: wgpu::Extent3d {
                width: uv_width.max(1),
                height: uv_height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: uv_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));

        // Create V texture (I420 only)
        self.tex_v = Some(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("convert_tex_v"),
            size: wgpu::Extent3d {
                width: uv_width.max(1),
                height: uv_height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));

        // Create output RGBA texture
        let output = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("convert_output_rgba"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        self.output_view = Some(output.create_view(&wgpu::TextureViewDescriptor::default()));
        self.output_texture = Some(output);

        self.cached_width = width;
        self.cached_height = height;
        self.cached_format = format;
        self.cached_uv_dims = uv_dims;
    }

    /// Write debayer params to the uniform buffer and AWB gains to the gains buffer.
    /// Returns whether GPU AWB passes are needed (no ISP gains available).
    fn prepare_debayer_params(&self, input: &GpuFrameInput) -> bool {
        let identity: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let has_gains = input.colour_gains.is_some();

        // Determine ISP colour processing state
        let (use_isp, bl, ccm) = if has_gains {
            let ccm = input.colour_correction_matrix.unwrap_or(identity);
            (1u32, input.black_level.unwrap_or(0.0), ccm)
        } else {
            // AWB will be computed on GPU; still enable ISP colour processing
            // so the debayer shader applies the GPU-computed gains
            (1u32, input.black_level.unwrap_or(0.0), identity)
        };

        let params = DebayerParams {
            width: input.width,
            height: input.height,
            pattern: input.format.bayer_pattern_code().unwrap_or(0),
            use_isp_colour: use_isp,
            black_level: bl,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            ccm_row0: [ccm[0][0], ccm[0][1], ccm[0][2], 0.0],
            ccm_row1: [ccm[1][0], ccm[1][1], ccm[1][2], 0.0],
            ccm_row2: [ccm[2][0], ccm[2][1], ccm[2][2], 0.0],
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&params));

        if let Some(gains) = input.colour_gains {
            // ISP gains available: write directly to the gains buffer
            let awb = AwbGains {
                gain_r: gains[0],
                gain_b: gains[1],
            };
            self.queue
                .write_buffer(&self.awb_gains_buffer, 0, bytemuck::bytes_of(&awb));
            false // no GPU AWB needed
        } else {
            // Clear sums buffer before GPU AWB accumulation
            let zeros = AwbSums {
                sum_r: 0,
                sum_g: 0,
                sum_b: 0,
                _pad: 0,
            };
            self.queue
                .write_buffer(&self.awb_sums_buffer, 0, bytemuck::bytes_of(&zeros));
            true // GPU AWB needed
        }
    }

    /// Encode AWB accumulate + finalize passes into the command encoder.
    fn encode_awb_passes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        y_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        let awb_pipeline = self
            .awb_pipeline
            .as_ref()
            .ok_or("AWB pipeline not created")?;
        let awb_finalize_pipeline = self
            .awb_finalize_pipeline
            .as_ref()
            .ok_or("AWB finalize pipeline not created")?;

        // AWB accumulate bind group
        let awb_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("awb_bind_group"),
            layout: &awb_pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.awb_sums_buffer.as_entire_binding(),
                },
            ],
        });

        // AWB finalize bind group
        let awb_finalize_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("awb_finalize_bind_group"),
            layout: &awb_finalize_pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.awb_sums_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.awb_gains_buffer.as_entire_binding(),
                },
            ],
        });

        // AWB accumulate pass: each thread handles one 2x2 superpixel
        let awb_wg_x = (width / 2).div_ceil(TILE_WORKGROUP_SIZE);
        let awb_wg_y = (height / 2).div_ceil(TILE_WORKGROUP_SIZE);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("awb_accumulate_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&awb_pipeline.pipeline);
            pass.set_bind_group(0, Some(&awb_bind_group), &[]);
            pass.dispatch_workgroups(awb_wg_x, awb_wg_y, 1);
        }

        // AWB finalize pass: 1 thread computes gains from sums
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("awb_finalize_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&awb_finalize_pipeline.pipeline);
            pass.set_bind_group(0, Some(&awb_finalize_bind_group), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        Ok(())
    }

    /// Set up Bayer resources and encode unpack + AWB + debayer passes onto a command encoder.
    ///
    /// Shared by `convert()` (Bayer path) and `convert_and_filter()`. Callers can append
    /// additional passes to the returned encoder before submitting.
    ///
    /// Returns `(encoder, needs_gpu_awb, had_gpu_unpack)` for logging and chaining.
    fn prepare_and_encode_bayer(
        &mut self,
        input: &GpuFrameInput,
    ) -> Result<(wgpu::CommandEncoder, bool, bool), String> {
        // All &mut self calls first (before immutable texture borrows)
        self.ensure_resources(input.width, input.height, input.format, None);
        self.ensure_debayer_pipeline();
        if input.colour_gains.is_none() {
            self.ensure_awb_pipeline();
            self.ensure_awb_finalize_pipeline();
        }

        let is_csi2_packed = input.y_stride > input.width;
        let unpack_info = if is_csi2_packed {
            if let Some(bit_depth) = detect_csi2_bit_depth(input.width, input.y_stride) {
                self.ensure_unpack_pipeline();
                let packed_size = input.y_data.len() as u64;
                let output_stride_u32 =
                    self.ensure_unpack_buffers(packed_size, input.width, input.height);
                Some((bit_depth, output_stride_u32))
            } else {
                warn!(
                    packed_stride = input.y_stride,
                    width = input.width,
                    "Unknown CSI-2 packing ratio, falling back to CPU unpack"
                );
                None
            }
        } else {
            None
        };

        // Immutable references
        let tex_y = self.tex_y.as_ref().ok_or("Y texture not allocated")?;
        let tex_uv = self.tex_uv.as_ref().ok_or("UV texture not allocated")?;
        let tex_v = self.tex_v.as_ref().ok_or("V texture not allocated")?;
        let output_view = self
            .output_view
            .as_ref()
            .ok_or("Output view not allocated")?;

        // Upload data
        if let Some((bit_depth, output_stride_u32)) = unpack_info {
            let packed_buf = self
                .packed_input_buffer
                .as_ref()
                .ok_or("Packed input buffer not allocated")?;
            self.queue.write_buffer(packed_buf, 0, input.y_data);

            let unpack_params = UnpackParams {
                width: input.width,
                height: input.height,
                packed_stride: input.y_stride,
                bit_depth,
                output_stride_u32,
                _pad: [0; 3],
            };
            self.queue.write_buffer(
                &self.unpack_uniform_buffer,
                0,
                bytemuck::bytes_of(&unpack_params),
            );
        } else {
            self.upload_textures(input, tex_y, tex_uv, tex_v)?;
        }

        let needs_gpu_awb = self.prepare_debayer_params(input);

        let debayer_pipeline = self
            .debayer_pipeline
            .as_ref()
            .ok_or("Debayer pipeline not created")?;

        let y_view = tex_y.create_view(&wgpu::TextureViewDescriptor::default());

        let debayer_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("debayer_bind_group"),
            layout: &debayer_pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.awb_gains_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bayer_encoder"),
            });

        if let Some((_, output_stride_u32)) = unpack_info {
            self.encode_unpack_pass(
                &mut encoder,
                tex_y,
                input.width,
                input.height,
                output_stride_u32,
            )?;
        }

        if needs_gpu_awb {
            self.encode_awb_passes(&mut encoder, &y_view, input.width, input.height)?;
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("debayer_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&debayer_pipeline.pipeline);
            pass.set_bind_group(0, Some(&debayer_bind_group), &[]);
            let workgroups_x = input.width.div_ceil(TILE_WORKGROUP_SIZE);
            let workgroups_y = input.height.div_ceil(TILE_WORKGROUP_SIZE);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        Ok((encoder, needs_gpu_awb, unpack_info.is_some()))
    }

    /// Convert frame to RGBA using unified shader
    pub fn convert(&mut self, input: &GpuFrameInput) -> Result<&wgpu::Texture, String> {
        let start = std::time::Instant::now();

        // Bayer path: use shared helper
        if input.format.is_bayer() {
            let (encoder, needs_gpu_awb, gpu_unpack) = self.prepare_and_encode_bayer(input)?;
            self.queue.submit(std::iter::once(encoder.finish()));

            let elapsed = start.elapsed();
            if elapsed.as_millis() > 2 {
                debug!(
                    elapsed_ms = format!("{:.2}", elapsed.as_micros() as f64 / 1000.0),
                    width = input.width,
                    height = input.height,
                    format = ?input.format,
                    gpu_awb = needs_gpu_awb,
                    gpu_unpack,
                    "Format conversion"
                );
            }

            return self
                .output_texture
                .as_ref()
                .ok_or("Output texture not allocated".to_string());
        }

        // YUV/packed-YUV path
        let uv_dims = if input.uv_stride > 0 {
            input
                .uv_data
                .map(|uv_data| (input.uv_stride, uv_data.len() as u32 / input.uv_stride))
        } else {
            None
        };

        self.ensure_resources(input.width, input.height, input.format, uv_dims);
        self.ensure_yuv_pipeline();

        let tex_y = self.tex_y.as_ref().ok_or("Y texture not allocated")?;
        let tex_uv = self.tex_uv.as_ref().ok_or("UV texture not allocated")?;
        let tex_v = self.tex_v.as_ref().ok_or("V texture not allocated")?;
        let output_view = self
            .output_view
            .as_ref()
            .ok_or("Output view not allocated")?;

        self.upload_textures(input, tex_y, tex_uv, tex_v)?;

        let params = ConvertParams {
            width: input.width,
            height: input.height,
            format: input.format.gpu_format_code(),
            y_stride: input.y_stride,
            uv_stride: input.uv_stride,
            v_stride: input.v_stride,
            _pad0: 0,
            _pad1: 0,
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&params));

        let format_pipeline = self
            .yuv_pipeline
            .as_ref()
            .ok_or("YUV pipeline not created")?;

        let y_view = tex_y.create_view(&wgpu::TextureViewDescriptor::default());
        let uv_view = tex_uv.create_view(&wgpu::TextureViewDescriptor::default());
        let v_view = tex_v.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yuv_convert_bind_group"),
            layout: &format_pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&uv_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&v_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("convert_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("convert_compute_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&format_pipeline.pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            let workgroups_x = input.width.div_ceil(TILE_WORKGROUP_SIZE);
            let workgroups_y = input.height.div_ceil(TILE_WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        let elapsed = start.elapsed();
        if elapsed.as_millis() > 2 {
            debug!(
                elapsed_ms = format!("{:.2}", elapsed.as_micros() as f64 / 1000.0),
                width = input.width,
                height = input.height,
                format = ?input.format,
                "Format conversion"
            );
        }

        self.output_texture
            .as_ref()
            .ok_or("Output texture not allocated".to_string())
    }

    /// Write a single plane to a GPU texture.
    fn write_plane(
        &self,
        texture: &wgpu::Texture,
        data: &[u8],
        bytes_per_row: u32,
        width: u32,
        height: u32,
    ) {
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Derive plane (width, height) from stride and data length, with fallback height.
    ///
    /// Used for I420 U/V planes where actual dimensions come from data layout.
    fn plane_dims(stride: u32, data_len: u32, fallback_h: u32) -> (u32, u32) {
        if stride > 0 {
            (stride, data_len / stride)
        } else {
            (stride, fallback_h)
        }
    }

    /// Upload textures based on format
    fn upload_textures(
        &self,
        input: &GpuFrameInput,
        tex_y: &wgpu::Texture,
        tex_uv: &wgpu::Texture,
        tex_v: &wgpu::Texture,
    ) -> Result<(), String> {
        match input.format {
            // Packed 4:2:2 formats
            PixelFormat::YUYV | PixelFormat::UYVY | PixelFormat::YVYU | PixelFormat::VYUY => {
                self.write_plane(
                    tex_y,
                    input.y_data,
                    input.y_stride,
                    input.width / 2,
                    input.height,
                );
            }

            // NV12/NV21: Y plane + UV plane
            PixelFormat::NV12 | PixelFormat::NV21 => {
                self.write_plane(
                    tex_y,
                    input.y_data,
                    input.y_stride,
                    input.width,
                    input.height,
                );
                if let Some(uv_data) = input.uv_data {
                    let uv_height = input.height / 2;
                    self.write_plane(tex_uv, uv_data, input.uv_stride, input.width / 2, uv_height);
                }
            }

            // I420: Y + U + V planes
            // UV dimensions derived from actual data (supports I422/I444 tagged as I420)
            PixelFormat::I420 => {
                self.write_plane(
                    tex_y,
                    input.y_data,
                    input.y_stride,
                    input.width,
                    input.height,
                );
                if let Some(uv_data) = input.uv_data {
                    let (uv_w, uv_h) =
                        Self::plane_dims(input.uv_stride, uv_data.len() as u32, input.height / 2);
                    self.write_plane(tex_uv, uv_data, input.uv_stride, uv_w, uv_h);
                }
                if let Some(v_data) = input.v_data {
                    let (v_w, v_h) =
                        Self::plane_dims(input.v_stride, v_data.len() as u32, input.height / 2);
                    self.write_plane(tex_v, v_data, input.v_stride, v_w, v_h);
                }
            }

            // Gray8: single channel
            PixelFormat::Gray8 => {
                self.write_plane(
                    tex_y,
                    input.y_data,
                    input.y_stride,
                    input.width,
                    input.height,
                );
            }

            // RGBA-family: 4 bytes per pixel, uploaded as Rgba8Unorm
            PixelFormat::RGBA | PixelFormat::ABGR | PixelFormat::BGRA | PixelFormat::RGB24 => {
                self.write_plane(
                    tex_y,
                    input.y_data,
                    input.y_stride,
                    input.width,
                    input.height,
                );
            }

            // Bayer formats: raw sensor data, single channel (uploaded as R16Unorm)
            // CSI-2 packed formats (10/12/14-bit) are handled by the GPU unpack pass —
            // this upload_textures() call is skipped for CSI-2 packed data.
            // Non-packed 8-bit Bayer is scaled u8→u16 so the shader always gets R16Unorm.
            PixelFormat::BayerRGGB
            | PixelFormat::BayerBGGR
            | PixelFormat::BayerGRBG
            | PixelFormat::BayerGBRG => {
                if input.y_stride > input.width {
                    // CSI-2 packed: GPU unpack pass handles this — nothing to do here
                    return Ok(());
                }

                // Non-packed 8-bit Bayer: scale u8→u16 (val << 8)
                let num_pixels = (input.width * input.height) as usize;
                let mut out = vec![0u8; num_pixels * 2];
                for y in 0..input.height as usize {
                    let src_start = y * input.y_stride as usize;
                    let dst_start = y * input.width as usize * 2;
                    for x in 0..input.width as usize {
                        let val = (input.y_data[src_start + x] as u16) << 8;
                        let le = val.to_le_bytes();
                        out[dst_start + x * 2] = le[0];
                        out[dst_start + x * 2 + 1] = le[1];
                    }
                }

                // R16Unorm: 2 bytes per pixel, stride = width * 2
                self.write_plane(tex_y, &out, input.width * 2, input.width, input.height);
            }
        }

        Ok(())
    }

    /// Read back the converted RGBA data to CPU memory
    pub async fn read_rgba_to_cpu(&mut self, width: u32, height: u32) -> Result<Vec<u8>, String> {
        let output = self
            .output_texture
            .as_ref()
            .ok_or("Output texture not allocated")?;

        let padded_bytes_per_row = align_to_copy_row(width * 4);
        let required_size = (padded_bytes_per_row * height) as u64;

        // Reuse cached staging buffer if it matches, otherwise allocate and cache
        if self.readback_staging_buffer_size != required_size {
            self.readback_staging_buffer =
                Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("convert_staging"),
                    size: required_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }));
            self.readback_staging_buffer_size = required_size;
        }

        let staging_buffer = self.readback_staging_buffer.as_ref().unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_encoder"),
            });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: output,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
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
            .map_err(|_| "Failed to receive buffer mapping result")?
            .map_err(|e| format!("Failed to map buffer: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let mut output_data = Vec::with_capacity((width * height * 4) as usize);

        if padded_bytes_per_row == width * 4 {
            output_data.extend_from_slice(&data[..(width * height * 4) as usize]);
        } else {
            for row in 0..height {
                let start = (row * padded_bytes_per_row) as usize;
                let end = start + (width * 4) as usize;
                output_data.extend_from_slice(&data[start..end]);
            }
        }

        drop(data);
        staging_buffer.unmap();

        Ok(output_data)
    }

    /// Convert frame to RGBA and apply filter in a single GPU submission.
    ///
    /// This eliminates the GPU→CPU→GPU round trip that occurs when debayer and
    /// filter run on separate pipeline instances. The debayer output texture
    /// (which has TEXTURE_BINDING usage) is directly sampled by the filter shader.
    pub fn convert_and_filter(
        &mut self,
        input: &GpuFrameInput,
        filter: FilterType,
    ) -> Result<(), String> {
        let start = std::time::Instant::now();

        // Ensure filter resources (&mut self calls before shared Bayer prep)
        self.ensure_filter_pipeline();
        self.ensure_filter_resources(input.width, input.height);

        // Shared Bayer prep: unpack + AWB + debayer passes
        let (mut encoder, needs_gpu_awb, gpu_unpack) = self.prepare_and_encode_bayer(input)?;

        // Write filter uniforms
        let filter_params = FilterParams {
            width: input.width,
            height: input.height,
            filter_mode: filter as u32,
            _padding: 0,
        };
        self.queue.write_buffer(
            &self.filter_uniform_buffer,
            0,
            bytemuck::bytes_of(&filter_params),
        );

        let filter_pipeline = self
            .filter_pipeline
            .as_ref()
            .ok_or("Filter pipeline not created")?;
        let filter_output_buffer = self
            .filter_output_buffer
            .as_ref()
            .ok_or("Filter output buffer not allocated")?;
        let output_texture = self
            .output_texture
            .as_ref()
            .ok_or("Output texture not allocated")?;
        let debayer_output_view =
            output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Filter bind group — reads from debayer output texture
        let filter_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("convert_filter_filter_bg"),
            layout: &filter_pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&debayer_output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: filter_output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.filter_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.filter_sampler),
                },
            ],
        });

        let workgroups_x = input.width.div_ceil(TILE_WORKGROUP_SIZE);
        let workgroups_y = input.height.div_ceil(TILE_WORKGROUP_SIZE);

        // Append filter pass to the encoder from prepare_and_encode_bayer
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("filter_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&filter_pipeline.pipeline);
            pass.set_bind_group(0, Some(&filter_bind_group), &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        let elapsed = start.elapsed();
        debug!(
            elapsed_ms = format!("{:.2}", elapsed.as_micros() as f64 / 1000.0),
            width = input.width,
            height = input.height,
            filter = ?filter,
            gpu_awb = needs_gpu_awb,
            gpu_unpack,
            "Debayer + filter (single submission)"
        );

        Ok(())
    }

    /// Read back the filtered RGBA data from the filter output buffer to CPU.
    pub async fn read_filtered_to_cpu(&self, width: u32, height: u32) -> Result<Vec<u8>, String> {
        let output_buffer = self
            .filter_output_buffer
            .as_ref()
            .ok_or("Filter output buffer not allocated")?;
        let staging_buffer = self
            .filter_staging_buffer
            .as_ref()
            .ok_or("Filter staging buffer not allocated")?;

        let buffer_size = (width * height * 4) as u64;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("filter_readback_encoder"),
            });

        encoder.copy_buffer_to_buffer(output_buffer, 0, staging_buffer, 0, buffer_size);

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
            .map_err(|_| "Failed to receive buffer mapping result")?
            .map_err(|e| format!("Failed to map buffer: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let output = data.to_vec();

        drop(data);
        staging_buffer.unmap();

        Ok(output)
    }

    pub fn output_texture(&self) -> Option<&wgpu::Texture> {
        self.output_texture.as_ref()
    }

    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }
}

/// Cached global pipeline instance
static GPU_CONVERT_PIPELINE: std::sync::OnceLock<tokio::sync::Mutex<Option<GpuConvertPipeline>>> =
    std::sync::OnceLock::new();

/// Get or create the shared pipeline instance
pub async fn get_gpu_convert_pipeline()
-> Result<tokio::sync::MutexGuard<'static, Option<GpuConvertPipeline>>, String> {
    let lock = GPU_CONVERT_PIPELINE.get_or_init(|| tokio::sync::Mutex::new(None));
    let mut guard = lock.lock().await;

    if guard.is_none() {
        match GpuConvertPipeline::new().await {
            Ok(pipeline) => {
                *guard = Some(pipeline);
            }
            Err(e) => {
                warn!("Failed to initialize convert pipeline: {}", e);
                return Err(e);
            }
        }
    }

    Ok(guard)
}
