// SPDX-License-Identifier: GPL-3.0-only

//! Custom primitive for rendering gallery thumbnails with rounded corners

use std::sync::Arc;

use cosmic::iced::Rectangle;
use iced_core::image::Id as ImageId;
use iced_wgpu::graphics::Viewport;
use iced_wgpu::primitive::{Pipeline as PipelineTrait, Primitive as PrimitiveTrait};
use iced_wgpu::wgpu;

/// Custom primitive for gallery thumbnail with rounded corners
#[derive(Debug, Clone)]
pub struct GalleryPrimitive {
    pub image_handle: cosmic::widget::image::Handle,
    /// RGBA data wrapped in Arc for cheap cloning (avoids copying image buffer every frame)
    pub rgba_data: Arc<Vec<u8>>,
    pub width: u32,
    pub height: u32,
    pub corner_radius: f32,
    /// Hover/pressed overlay intensity (0.0 = none, 0.15 = hover, 0.25 = pressed)
    pub hover_alpha: f32,
    /// Accent border color (RGBA) shown when hovered
    pub accent_color: [f32; 4],
}

impl GalleryPrimitive {
    pub fn new(
        image_handle: cosmic::widget::image::Handle,
        rgba_data: Arc<Vec<u8>>,
        width: u32,
        height: u32,
        corner_radius: f32,
    ) -> Self {
        Self {
            image_handle,
            rgba_data,
            width,
            height,
            corner_radius,
            hover_alpha: 0.0,
            accent_color: [0.0; 4],
        }
    }
}

/// Pipeline for rendering gallery thumbnails
pub struct GalleryPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    texture_format: wgpu::TextureFormat,
    // Cache for the single uploaded texture (at most one gallery thumbnail at a time)
    texture_cache: Option<(ImageId, GalleryTexture)>,
}

struct GalleryTexture {
    bind_group: wgpu::BindGroup,
    viewport_buffer: wgpu::Buffer,
    /// Physical widget bounds (x, y, width, height) clamped to render target
    /// Stored during prepare() and used in render() for valid viewport rect
    physical_bounds: Option<(f32, f32, f32, f32)>,
}

impl PipelineTrait for GalleryPipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        GalleryPipeline::new(device, format)
    }

    fn trim(&mut self) {
        // No-op: we manage texture cache lifecycle ourselves.
        // Clearing here would destroy live textures and cause flickering.
    }
}

impl PrimitiveTrait for GalleryPrimitive {
    type Pipeline = GalleryPipeline;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        viewport: &Viewport,
    ) {
        // Upload image texture if needed and update viewport buffer
        pipeline.upload_image(
            _device,
            queue,
            &self.image_handle,
            &self.rgba_data,
            self.width,
            self.height,
        );

        // Calculate physical bounds from logical bounds using scale factor
        // Then clamp to render target to ensure valid viewport rect
        let scale = viewport.scale_factor() as f32;
        let render_target = viewport.physical_size();

        let raw = (
            bounds.x * scale,
            bounds.y * scale,
            bounds.width * scale,
            bounds.height * scale,
        );

        let clamped_x = raw.0.max(0.0);
        let clamped_y = raw.1.max(0.0);
        let clamped_w = ((raw.0 + raw.2).min(render_target.width as f32) - clamped_x).max(0.0);
        let clamped_h = ((raw.1 + raw.3).min(render_target.height as f32) - clamped_y).max(0.0);

        // Calculate UV offset/scale to compensate for clamping
        let (uv_offset, uv_scale) = if raw.2 > 0.0 && raw.3 > 0.0 {
            (
                ((clamped_x - raw.0) / raw.2, (clamped_y - raw.1) / raw.3),
                (clamped_w / raw.2, clamped_h / raw.3),
            )
        } else {
            ((0.0, 0.0), (1.0, 1.0))
        };

        // Store physical bounds for render()
        pipeline.store_physical_bounds(
            &self.image_handle,
            (clamped_x, clamped_y, clamped_w, clamped_h),
        );

        // Update viewport size, corner radius, and UV adjustment
        pipeline.update_viewport(
            queue,
            &self.image_handle,
            bounds.width,
            bounds.height,
            self.corner_radius,
            self.hover_alpha,
            self.accent_color,
            uv_offset,
            uv_scale,
        );
    }

    fn render(
        &self,
        _pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        _pipeline.render(encoder, target, clip_bounds, &self.image_handle);
    }
}

impl GalleryPipeline {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        // Load shader
        let shader_source = format!(
            "{}\n{}",
            crate::shaders::GEOMETRY_FUNCTIONS,
            include_str!("gallery_rounded_shader.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gallery rounded shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gallery bind group layout"),
            entries: &[
                // Texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Viewport size uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gallery pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gallery pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("gallery sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            texture_format: format,
            texture_cache: None,
        }
    }

    fn upload_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image_handle: &cosmic::widget::image::Handle,
        rgba_data: &[u8],
        width: u32,
        height: u32,
    ) {
        // Get a unique ID for this image
        let image_id = image_handle.id();

        // Check if already uploaded
        if self
            .texture_cache
            .as_ref()
            .is_some_and(|(id, _)| *id == image_id)
        {
            return;
        }

        // Use the provided RGBA data directly
        if rgba_data.is_empty() {
            tracing::warn!("Empty RGBA data for gallery thumbnail");
            return;
        }

        // Convert RGBA to BGRA if needed
        let texture_data = if matches!(
            self.texture_format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        ) {
            // Convert RGBA -> BGRA by swapping R and B channels
            let mut bgra_data = rgba_data.to_vec();
            for chunk in bgra_data.as_chunks_mut::<4>().0 {
                chunk.swap(0, 2); // Swap R and B
            }
            bgra_data
        } else {
            rgba_data.to_vec()
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gallery texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.texture_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload the image data
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &texture_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create viewport params buffer (will be updated in update_viewport before render)
        // Contains: size (2), corner_radius (1), hover_alpha (1), accent_color (4), uv_offset (2), uv_scale (2)
        let viewport_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gallery viewport buffer"),
            size: std::mem::size_of::<[f32; 12]>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gallery bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: viewport_buffer.as_entire_binding(),
                },
            ],
        });

        self.texture_cache = Some((
            image_id,
            GalleryTexture {
                bind_group,
                viewport_buffer,
                physical_bounds: None,
            },
        ));
    }

    fn store_physical_bounds(
        &mut self,
        image_handle: &cosmic::widget::image::Handle,
        physical_bounds: (f32, f32, f32, f32),
    ) {
        let image_id = image_handle.id();
        if let Some((id, gallery_texture)) = &mut self.texture_cache
            && *id == image_id
        {
            gallery_texture.physical_bounds = Some(physical_bounds);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn update_viewport(
        &mut self,
        queue: &wgpu::Queue,
        image_handle: &cosmic::widget::image::Handle,
        width: f32,
        height: f32,
        corner_radius: f32,
        hover_alpha: f32,
        accent_color: [f32; 4],
        uv_offset: (f32, f32),
        uv_scale: (f32, f32),
    ) {
        let image_id = image_handle.id();

        if let Some((id, gallery_texture)) = &self.texture_cache {
            if *id != image_id {
                return;
            }
            // Pack: size (2), corner_radius (1), hover_alpha (1), accent_color (4), uv_offset (2), uv_scale (2)
            let viewport_data = [
                width,
                height,
                corner_radius,
                hover_alpha,
                accent_color[0],
                accent_color[1],
                accent_color[2],
                accent_color[3],
                uv_offset.0,
                uv_offset.1,
                uv_scale.0,
                uv_scale.1,
            ];
            queue.write_buffer(
                &gallery_texture.viewport_buffer,
                0,
                bytemuck::cast_slice(&viewport_data),
            );
        }
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
        image_handle: &cosmic::widget::image::Handle,
    ) {
        let image_id = image_handle.id();

        if let Some((id, gallery_texture)) = &self.texture_cache {
            if *id != image_id || clip_bounds.width == 0 || clip_bounds.height == 0 {
                return;
            }

            // Use full widget bounds for viewport (prevents distortion when partially clipped)
            // Fall back to clip_bounds if physical_bounds not available
            let widget_bounds = gallery_texture.physical_bounds.unwrap_or((
                clip_bounds.x as f32,
                clip_bounds.y as f32,
                clip_bounds.width as f32,
                clip_bounds.height as f32,
            ));

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gallery render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // Use full widget bounds for viewport (prevents distortion in scrollables)
            render_pass.set_viewport(
                widget_bounds.0,
                widget_bounds.1,
                widget_bounds.2,
                widget_bounds.3,
                0.0,
                1.0,
            );

            // Use clip bounds for scissor (clips to visible portion)
            render_pass.set_scissor_rect(
                clip_bounds.x,
                clip_bounds.y,
                clip_bounds.width,
                clip_bounds.height,
            );

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, Some(&gallery_texture.bind_group), &[]);
            render_pass.draw(0..3, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_gpu::{headless_device, skip_no_gpu};

    /// `GalleryPipeline::new` must actually compile its shader.
    ///
    /// # Why this exists
    ///
    /// The gallery shader's `rounded_box_sdf` was deleted in favour of the shared
    /// `shaders/geometry.wgsl` prelude, which `new` prepends by hand:
    ///
    /// ```ignore
    /// format!("{}\n{}", crate::shaders::GEOMETRY_FUNCTIONS, include_str!("gallery_rounded_shader.wgsl"))
    /// ```
    ///
    /// Nothing else in the crate constructs a `GalleryPipeline`. WGSL is compiled
    /// at `create_shader_module`, i.e. at runtime, so a prelude that stopped being
    /// prepended — or a `rounded_box_sdf` that changed signature — is not a build
    /// error here. It is a panic on the device, the first time a thumbnail is
    /// drawn, in a shipped binary.
    ///
    /// `VideoPipeline::new` compiles all six of its modules in one function, so
    /// the existing GPU tests already cover those. This closes the one path where
    /// a prelude-wiring mistake reached the device unchallenged. It is a smoke
    /// test on purpose: it asserts construction, because construction is exactly
    /// the step that validates the shader.
    #[test]
    fn gallery_pipeline_compiles_its_shader() {
        let Some((_gpu_test, device, _queue)) = headless_device() else {
            skip_no_gpu("gallery_pipeline_compiles_its_shader");
            return;
        };
        // wgpu reports a shader compile failure on the device's error scope
        // rather than by returning, so capture it rather than trusting a
        // non-panicking return.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _pipeline = GalleryPipeline::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
        let err = pollster::block_on(scope.pop());
        assert!(
            err.is_none(),
            "GalleryPipeline::new failed to build: {err:?} — the gallery shader is \
             only compiled at runtime, so this is a device-side panic on the first \
             thumbnail, not a build error"
        );
    }

    /// The prelude really is what supplies the gallery's SDF.
    ///
    /// The smoke test above passes either way if the shader ever grows its own
    /// copy of `rounded_box_sdf` back — and a second copy silently forking from
    /// `geometry.wgsl` is the exact thing deleting it was meant to prevent. So
    /// pin the split: the shader calls the function and does not define it, and
    /// the prelude defines it.
    #[test]
    fn the_gallery_shader_takes_its_sdf_from_the_prelude() {
        let shader = include_str!("gallery_rounded_shader.wgsl");
        assert!(
            shader.contains("rounded_box_sdf("),
            "the gallery shader should still USE the shared SDF"
        );
        assert!(
            !shader.contains("fn rounded_box_sdf"),
            "the gallery shader defines its own `rounded_box_sdf` again — that is \
             the duplicate `geometry.wgsl` exists to replace, and a second copy \
             will drift from the video shader's silhouette"
        );
        assert!(
            crate::shaders::GEOMETRY_FUNCTIONS.contains("fn rounded_box_sdf"),
            "the shared prelude must define `rounded_box_sdf`, or every shader \
             that stopped defining it fails to compile on device"
        );
    }
}
