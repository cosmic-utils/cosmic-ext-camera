// SPDX-License-Identifier: GPL-3.0-only

//! Async post-processing pipeline for photos
//!
//! This module handles post-processing operations on captured frames:
//! - Filter application directly on RGBA data (GPU-accelerated)
//! - RGBA to RGB conversion (drop alpha channel)
//! - Sharpening
//! - Brightness/contrast adjustments
//!
//! The pipeline is optimized to apply filters on RGBA data before RGB conversion,
//! avoiding unnecessary format conversions.

use crate::app::FilterType;
use crate::backends::camera::types::{CameraFrame, PixelFormat};
use crate::shaders::{
    GpuFrameInput, apply_filter_gpu_rgba, copy_rgba_pixels_to_rgb, get_gpu_convert_pipeline,
};
use image::RgbImage;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Post-processing configuration
///
/// Public settings support struct update syntax from external crates:
///
/// ```
/// let config = camera::pipelines::photo::PostProcessingConfig {
///     sharpening: true,
///     ..Default::default()
/// };
/// assert!(config.sharpening);
/// ```
#[derive(Debug, Clone)]
pub struct PostProcessingConfig {
    /// Photo-only demosaic selection, independent of video.
    pub demosaic_method: crate::shaders::DemosaicMethod,
    /// Enable sharpening
    pub sharpening: bool,
    /// Brightness adjustment (-1.0 to 1.0, 0.0 = no change)
    pub brightness: f32,
    /// Contrast adjustment (0.0 to 2.0, 1.0 = no change)
    pub contrast: f32,
    /// Saturation adjustment (0.0 to 2.0, 1.0 = no change)
    pub saturation: f32,
    /// Filter type to apply
    pub filter_type: FilterType,
    /// Crop rectangle (x, y, width, height) - None means no cropping
    pub crop_rect: Option<(u32, u32, u32, u32)>,
    /// Zoom level (1.0 = no zoom, 2.0 = 2x zoom center crop)
    pub zoom_level: f32,
}

impl Default for PostProcessingConfig {
    fn default() -> Self {
        Self {
            demosaic_method: crate::shaders::DemosaicMethod::default(),
            sharpening: false,
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            filter_type: FilterType::Standard,
            crop_rect: None,
            zoom_level: 1.0,
        }
    }
}

/// Processed image data
pub struct ProcessedImage {
    pub image: RgbImage,
    pub width: u32,
    pub height: u32,
}

/// Post-processor for captured frames
pub struct PostProcessor {
    config: PostProcessingConfig,
    session: Option<Arc<crate::shaders::PhotoSession>>,
    pace_gpu: bool,
    rendering_intent: crate::shaders::BayerRenderingIntent,
}

impl PostProcessor {
    /// Create a new post-processor with the given configuration
    pub fn new(config: PostProcessingConfig) -> Self {
        Self {
            config,
            session: None,
            pace_gpu: true,
            rendering_intent: crate::shaders::BayerRenderingIntent::OrdinaryStillSdr,
        }
    }

    /// Submission scheduling only; never changes colour/rendering intent.
    pub(crate) fn without_gpu_pacing(mut self) -> Self {
        self.pace_gpu = false;
        self
    }

    /// Continuous Motion Photo retains legacy colour independently of pacing.
    pub(crate) fn for_continuous_frames(mut self) -> Self {
        self.rendering_intent = crate::shaders::BayerRenderingIntent::Legacy;
        self
    }

    /// Use the negotiated camera session snapshotted at shutter dispatch.
    pub(crate) fn set_photo_session(&mut self, session: Option<Arc<crate::shaders::PhotoSession>>) {
        self.session = session;
    }

    /// Process a captured frame asynchronously
    ///
    /// This runs all post-processing steps using GPU acceleration where available,
    /// with software rendering fallback for systems without GPU support.
    ///
    /// # Arguments
    /// * `frame` - Camera frame with its negotiated pixel layout and metadata
    ///
    /// # Returns
    /// * `Ok(ProcessedImage)` - Processed RGB image
    /// * `Err(String)` - Error message
    pub async fn process(&self, frame: Arc<CameraFrame>) -> Result<ProcessedImage, String> {
        self.process_with_zoom(frame, self.config.zoom_level).await
    }

    /// Override total zoom for one continuous frame without changing the aspect
    /// crop, filter or rendering policy. Stills retain their snapshotted zoom.
    pub(crate) async fn process_with_zoom(
        &self,
        frame: Arc<CameraFrame>,
        desired_zoom: f32,
    ) -> Result<ProcessedImage, String> {
        info!(
            width = frame.width,
            height = frame.height,
            format = ?frame.format,
            "Starting post-processing"
        );

        crate::backends::camera::types::validate_frame_layout(
            frame.data.as_ref(),
            frame.format,
            frame.width,
            frame.height,
            frame.stride,
            frame.yuv_planes.as_ref(),
        )
        .map_err(|e| e.to_string())?;
        let mut config = self.config.clone();
        config.zoom_level = crate::backends::camera::zoom::residual_zoom(desired_zoom, &frame);
        let frame_width = frame.width;
        let frame_height = frame.height;
        let compact_rgba = match frame.format {
            PixelFormat::RGBA | PixelFormat::BGRA | PixelFormat::ABGR => {
                let mut rgba = Self::copy_rgba_without_padding(
                    frame.data.as_ref(),
                    frame.width,
                    frame.height,
                    frame.stride,
                )?;
                for pixel in rgba.as_chunks_mut::<4>().0 {
                    match frame.format {
                        PixelFormat::BGRA => pixel.swap(0, 2),
                        PixelFormat::ABGR => pixel.reverse(),
                        _ => {}
                    }
                }
                Some(rgba)
            }
            PixelFormat::RGB24 | PixelFormat::Gray8 => {
                let channels = if frame.format == PixelFormat::RGB24 {
                    3
                } else {
                    1
                };
                let row_bytes = frame.width as usize * channels;
                crate::backends::camera::types::checked_plane(
                    frame.data.as_ref(),
                    0,
                    frame.data.as_ref().len(),
                    frame.stride as usize,
                    row_bytes,
                    frame.height as usize,
                )
                .map_err(|e| e.to_string())?;
                let mut rgba = Vec::new();
                for y in 0..frame.height as usize {
                    let start = y * frame.stride as usize;
                    let row = frame
                        .data
                        .as_ref()
                        .get(start..start + row_bytes)
                        .ok_or("Rendered frame is truncated")?;
                    for pixel in row.chunks_exact(channels) {
                        if channels == 3 {
                            rgba.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
                        } else {
                            rgba.extend_from_slice(&[pixel[0], pixel[0], pixel[0], 255]);
                        }
                    }
                }
                Some(rgba)
            }
            _ => None,
        };

        // Converted GPU photos keep the full canonical image on-device until
        // filtering is complete, then read only the final crop/zoom rectangle.
        let region = Self::final_region(frame_width, frame_height, &config)?;
        let gpu_converted = frame.format.is_bayer() || frame.format.is_yuv();
        let filtered_rgba: Vec<u8> = if gpu_converted {
            self.convert_gpu_photo_region(&frame, &config, region)
                .await?
        } else if config.filter_type != FilterType::Standard {
            // Already RGBA, just apply filter
            let rgba = compact_rgba.unwrap_or_else(|| frame.data.to_vec());
            match apply_filter_gpu_rgba(&rgba, frame_width, frame_height, config.filter_type).await
            {
                Ok(filtered) => filtered,
                Err(e) => {
                    warn!(error = %e, "GPU filter failed, using unfiltered frame");
                    rgba
                }
            }
        } else if let Some(rgba) = compact_rgba {
            rgba
        } else {
            frame.data.to_vec()
        };

        // Aspect crop then rounded center zoom share the same sensor-space region.
        // GPU conversion already read this region; rendered inputs need one CPU crop.
        let (x, y, final_width, final_height) = region;
        let final_rgba = if !gpu_converted && region != (0, 0, frame_width, frame_height) {
            Self::crop_rgba(
                &filtered_rgba,
                frame_width,
                frame_height,
                x,
                y,
                final_width,
                final_height,
            )?
        } else {
            filtered_rgba
        };

        // Step 4: Convert filtered RGBA to RGB (drop alpha channel)
        let rgb_image = Self::convert_rgba_to_rgb(&final_rgba, final_width, final_height)?;
        drop(final_rgba);

        // Step 5 & 6: Apply adjustments and sharpening (CPU-bound)
        let needs_adjustments =
            config.brightness != 0.0 || config.contrast != 1.0 || config.saturation != 1.0;
        let needs_sharpening = config.sharpening;

        let rgb_image = if needs_adjustments || needs_sharpening {
            tokio::task::spawn_blocking(move || {
                let mut image = rgb_image;

                if needs_adjustments {
                    Self::apply_adjustments(&mut image, &config);
                }

                if needs_sharpening {
                    Self::apply_sharpening(&mut image);
                }

                image
            })
            .await
            .map_err(|e| format!("Post-processing task error: {}", e))?
        } else {
            rgb_image
        };

        debug!("Post-processing complete");

        Ok(ProcessedImage {
            width: final_width,
            height: final_height,
            image: rgb_image,
        })
    }

    #[cfg(test)]
    pub(crate) async fn memory_photo_for_test(
        frame: &CameraFrame,
        method: crate::shaders::DemosaicMethod,
    ) -> Result<Vec<u8>, String> {
        Self::convert_bayer_to_rgba(frame, method).await
    }

    #[cfg(test)]
    async fn convert_bayer_to_rgba(
        frame: &CameraFrame,
        method: crate::shaders::DemosaicMethod,
    ) -> Result<Vec<u8>, String> {
        let input = GpuFrameInput::from_camera_frame(frame).map_err(|e| e.to_string())?;
        let mut guard = get_gpu_convert_pipeline().await?;
        let converter = guard.as_mut().ok_or("Convert pipeline not initialized")?;
        let mut pipeline = converter.photo_workspace();
        pipeline
            .convert_photo_bayer_with_method(&input, method)
            .map_err(|e| e.to_string())?;
        pipeline.read_rgba_to_cpu(frame.width, frame.height).await
    }

    #[cfg(test)]
    pub(crate) async fn gpu_rgba_for_test(&self, frame: &CameraFrame) -> Result<Vec<u8>, String> {
        let mut config = self.config.clone();
        config.zoom_level = crate::backends::camera::zoom::residual_zoom(config.zoom_level, frame);
        let region = Self::final_region(frame.width, frame.height, &config)?;
        self.convert_gpu_photo_region(frame, &config, region).await
    }

    /// Map the aspect ROI through zoom about the full frame's center, matching
    /// the preview shader and native ScalerCrop even for asymmetric Cover ROIs.
    pub(crate) fn final_region(
        width: u32,
        height: u32,
        config: &PostProcessingConfig,
    ) -> Result<(u32, u32, u32, u32), String> {
        let (mut x, mut y, mut w, mut h) = config.crop_rect.unwrap_or((0, 0, width, height));
        if x > width || y > height || w > width - x || h > height - y {
            return Err(format!(
                "Crop region ({x},{y} {w}x{h}) exceeds source dimensions ({width}x{height})"
            ));
        }
        if config.zoom_level > 1.0 {
            let zoom_width = (w as f32 / config.zoom_level).round() as u32;
            let zoom_height = (h as f32 / config.zoom_level).round() as u32;
            let zoom_width = zoom_width.max(1);
            let zoom_height = zoom_height.max(1);
            if zoom_width > w || zoom_height > h {
                return Err("Zoom crop out of bounds".into());
            }
            let zoom_origin = |offset: u32, extent: u32, zoomed: u32, full: u32| {
                let center = f64::from(full) / 2.0;
                let roi_center = f64::from(offset) + f64::from(extent) / 2.0;
                (center + (roi_center - center) / f64::from(config.zoom_level)
                    - f64::from(zoomed) / 2.0)
                    .floor()
                    .clamp(0.0, f64::from(full - zoomed)) as u32
            };
            x = zoom_origin(x, w, zoom_width, width);
            y = zoom_origin(y, h, zoom_height, height);
            w = zoom_width;
            h = zoom_height;
        }
        Ok((x, y, w, h))
    }

    async fn convert_gpu_photo_region(
        &self,
        frame: &CameraFrame,
        config: &PostProcessingConfig,
        region: (u32, u32, u32, u32),
    ) -> Result<Vec<u8>, String> {
        let input = GpuFrameInput::from_camera_frame(frame).map_err(|e| e.to_string())?;
        let mut guard = get_gpu_convert_pipeline().await?;
        let converter = guard.as_mut().ok_or("Convert pipeline not initialized")?;
        let mut pipeline = converter.photo_workspace_for_session(
            self.session.clone(),
            &input,
            config.demosaic_method,
        );
        if frame.format.is_bayer() {
            // Filters require the full canonical sensor source for UV/neighbor modes.
            let conversion_region = if config.filter_type == FilterType::Standard {
                region
            } else {
                (0, 0, frame.width, frame.height)
            };
            pipeline
                .convert_still_bayer_region(
                    &input,
                    config.demosaic_method,
                    conversion_region,
                    self.pace_gpu,
                    self.rendering_intent,
                )
                .map_err(|e| e.to_string())?;
        } else {
            pipeline
                .convert(&input)
                .map_err(|e| format!("Failed to convert to RGBA: {e}"))?;
        }
        if config.filter_type != FilterType::Standard {
            let filtered = async {
                let mut filter_guard = crate::shaders::get_gpu_filter_pipeline().await?;
                let filter = filter_guard
                    .as_mut()
                    .ok_or("GPU filter pipeline not initialized")?;
                filter
                    .apply_filter_texture_region(
                        pipeline
                            .output_texture()
                            .ok_or("Missing converted photo texture")?,
                        config.filter_type,
                        region,
                    )
                    .await
            }
            .await;
            match filtered {
                Ok(rgba) => {
                    pipeline.complete();
                    return Ok(rgba);
                }
                Err(error) => warn!(%error, "GPU filter failed, using unfiltered frame"),
            }
        }
        // Mandatory conversion errors are never optional-filter fallbacks.
        let rgba = pipeline.read_rgba_region_to_cpu(region).await?;
        pipeline.complete();
        Ok(rgba)
    }

    /// Crop RGBA data to a rectangular region
    fn crop_rgba(
        rgba_data: &[u8],
        src_width: u32,
        src_height: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, String> {
        // Validate bounds
        if x + width > src_width || y + height > src_height {
            return Err(format!(
                "Crop region ({},{} {}x{}) exceeds source dimensions ({}x{})",
                x, y, width, height, src_width, src_height
            ));
        }

        let src_stride = src_width as usize * 4;
        let dst_stride = width as usize * 4;
        let mut cropped = vec![0u8; (width * height * 4) as usize];

        for row in 0..height as usize {
            let src_row_start = ((y as usize + row) * src_stride) + (x as usize * 4);
            let dst_row_start = row * dst_stride;
            cropped[dst_row_start..dst_row_start + dst_stride]
                .copy_from_slice(&rgba_data[src_row_start..src_row_start + dst_stride]);
        }

        Ok(cropped)
    }

    /// Convert RGBA data to RGB image (drop alpha channel)
    fn convert_rgba_to_rgb(rgba_data: &[u8], width: u32, height: u32) -> Result<RgbImage, String> {
        let expected_size = (width * height * 4) as usize;
        if rgba_data.len() < expected_size {
            return Err(format!(
                "RGBA data too small: expected {}, got {}",
                expected_size,
                rgba_data.len()
            ));
        }

        // Fixed-size chunks give LLVM a stride-4 -> stride-3 loop instead of
        // per-channel flat_map iterator state. Keep the exact byte values and
        // ignore any trailing bytes beyond the active image.
        let mut rgb_data = vec![0u8; width as usize * height as usize * 3];
        copy_rgba_pixels_to_rgb(
            &rgba_data[..width as usize * height as usize * 4],
            &mut rgb_data,
        );

        RgbImage::from_raw(width, height, rgb_data)
            .ok_or_else(|| "Failed to create RGB image from converted data".to_string())
    }

    fn copy_rgba_without_padding(
        data: &[u8],
        width: u32,
        height: u32,
        stride: u32,
    ) -> Result<Vec<u8>, String> {
        let row_bytes = width
            .checked_mul(4)
            .ok_or_else(|| "RGBA row size overflow".to_string())? as usize;
        let stride = stride as usize;
        if stride < row_bytes {
            return Err(format!(
                "RGBA stride too small: expected at least {row_bytes}, got {stride}"
            ));
        }

        let height = height as usize;
        let required = if height == 0 {
            0
        } else {
            (height - 1)
                .checked_mul(stride)
                .and_then(|offset| offset.checked_add(row_bytes))
                .ok_or_else(|| "RGBA buffer size overflow".to_string())?
        };
        if data.len() < required {
            return Err(format!(
                "RGBA data too small for stride: expected {required}, got {}",
                data.len()
            ));
        }

        let capacity = row_bytes
            .checked_mul(height)
            .ok_or_else(|| "RGBA compact buffer size overflow".to_string())?;
        let mut compact = Vec::with_capacity(capacity);
        for row in 0..height {
            let start = row * stride;
            compact.extend_from_slice(&data[start..start + row_bytes]);
        }
        Ok(compact)
    }

    /// Apply zoom by center-cropping the RGBA image
    ///
    /// At zoom_level 2.0, the center 50% of the image is cropped and returned.
    #[cfg(test)]
    fn apply_zoom_crop(
        rgba_data: &[u8],
        width: u32,
        height: u32,
        zoom_level: f32,
    ) -> Result<(Vec<u8>, u32, u32), String> {
        if zoom_level <= 1.0 {
            return Ok((rgba_data.to_vec(), width, height));
        }

        // Calculate cropped dimensions
        let crop_width = (width as f32 / zoom_level).round() as u32;
        let crop_height = (height as f32 / zoom_level).round() as u32;

        // Ensure minimum size
        let crop_width = crop_width.max(1);
        let crop_height = crop_height.max(1);

        // Calculate center offset
        let offset_x = (width - crop_width) / 2;
        let offset_y = (height - crop_height) / 2;

        debug!(
            zoom_level,
            original_width = width,
            original_height = height,
            crop_width,
            crop_height,
            offset_x,
            offset_y,
            "Applying zoom crop"
        );

        // Extract the center region
        let mut cropped_data = Vec::with_capacity((crop_width * crop_height * 4) as usize);
        let bytes_per_pixel = 4;
        let src_stride = width * bytes_per_pixel;

        for y in 0..crop_height {
            let src_y = offset_y + y;
            let src_row_start = (src_y * src_stride + offset_x * bytes_per_pixel) as usize;
            let src_row_end = src_row_start + (crop_width * bytes_per_pixel) as usize;

            if src_row_end <= rgba_data.len() {
                cropped_data.extend_from_slice(&rgba_data[src_row_start..src_row_end]);
            } else {
                return Err("Zoom crop out of bounds".to_string());
            }
        }

        Ok((cropped_data, crop_width, crop_height))
    }

    /// Apply brightness, contrast, and saturation adjustments
    fn apply_adjustments(image: &mut RgbImage, config: &PostProcessingConfig) {
        for pixel in image.pixels_mut() {
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;

            // Apply brightness
            let r = r + config.brightness * 255.0;
            let g = g + config.brightness * 255.0;
            let b = b + config.brightness * 255.0;

            // Apply contrast
            let r = ((r - 128.0) * config.contrast + 128.0).clamp(0.0, 255.0);
            let g = ((g - 128.0) * config.contrast + 128.0).clamp(0.0, 255.0);
            let b = ((b - 128.0) * config.contrast + 128.0).clamp(0.0, 255.0);

            // Apply saturation
            if config.saturation != 1.0 {
                let gray = 0.299 * r + 0.587 * g + 0.114 * b;
                let r = (gray + (r - gray) * config.saturation).clamp(0.0, 255.0);
                let g = (gray + (g - gray) * config.saturation).clamp(0.0, 255.0);
                let b = (gray + (b - gray) * config.saturation).clamp(0.0, 255.0);

                pixel[0] = r as u8;
                pixel[1] = g as u8;
                pixel[2] = b as u8;
            } else {
                pixel[0] = r as u8;
                pixel[1] = g as u8;
                pixel[2] = b as u8;
            }
        }
    }

    /// Apply unsharp mask sharpening
    ///
    /// This is a simple 3x3 kernel sharpening filter.
    fn apply_sharpening(image: &mut RgbImage) {
        // Simple sharpen kernel: [ [0, -1, 0], [-1, 5, -1], [0, -1, 0] ]
        let (width, height) = image.dimensions();
        let original = image.clone();

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                for c in 0..3 {
                    let center = original.get_pixel(x, y)[c] as i32 * 5;
                    let top = original.get_pixel(x, y - 1)[c] as i32;
                    let bottom = original.get_pixel(x, y + 1)[c] as i32;
                    let left = original.get_pixel(x - 1, y)[c] as i32;
                    let right = original.get_pixel(x + 1, y)[c] as i32;

                    let value = (center - top - bottom - left - right).clamp(0, 255) as u8;
                    image.get_pixel_mut(x, y)[c] = value;
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "processing_zoom_tests.rs"]
mod zoom_tests;

#[cfg(test)]
#[path = "processing_roi_tests.rs"]
mod roi_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_gpu_region_matches_sequential_cpu_geometry() {
        for (width, height) in [(21, 19), (67, 65), (1, 1)] {
            let pixels: Vec<u8> = (0..width * height)
                .flat_map(|i| [(i % 256) as u8, (i / 256) as u8, 53, 255])
                .collect();
            for crop_rect in [
                None,
                Some((0, 0, width, height)),
                // Sequential ROI-center zoom is equivalent to the preview
                // only for centered ROIs. Asymmetric Cover geometry has a
                // separate regression against the full-frame shader pivot.
                Some((
                    width / 3,
                    height / 3,
                    width - 2 * (width / 3),
                    height - 2 * (height / 3),
                )),
            ] {
                for zoom_level in [1.0, 1.01, 1.5, 2.0, 7.0, f32::INFINITY, f32::NAN] {
                    let config = PostProcessingConfig {
                        crop_rect,
                        zoom_level,
                        ..Default::default()
                    };
                    let (first, w, h) = if let Some((x, y, w, h)) = crop_rect {
                        (
                            PostProcessor::crop_rgba(&pixels, width, height, x, y, w, h).unwrap(),
                            w,
                            h,
                        )
                    } else {
                        (pixels.clone(), width, height)
                    };
                    let expected = if zoom_level > 1.0 {
                        PostProcessor::apply_zoom_crop(&first, w, h, zoom_level).unwrap()
                    } else {
                        (first, w, h)
                    };
                    let (x, y, w, h) = PostProcessor::final_region(width, height, &config).unwrap();
                    let actual =
                        PostProcessor::crop_rgba(&pixels, width, height, x, y, w, h).unwrap();
                    assert_eq!(
                        (actual, w, h),
                        expected,
                        "{width}x{height} {crop_rect:?} zoom={zoom_level}"
                    );
                }
            }
        }
        for crop_rect in [
            Some((u32::MAX, 0, 2, 1)),
            Some((0, 19, 1, 1)),
            Some((20, 0, 2, 1)),
        ] {
            assert!(
                PostProcessor::final_region(
                    21,
                    19,
                    &PostProcessingConfig {
                        crop_rect,
                        ..Default::default()
                    }
                )
                .is_err()
            );
        }
    }

    #[test]
    fn ordinary_photo_gpu_region_matches_full_readback_reference() {
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu(
                "ordinary_photo_gpu_region_matches_full_readback_reference",
            );
            return;
        };
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            use crate::backends::camera::types::{FrameData, FrameMetadata};
            use crate::shaders::{DemosaicMethod, GpuConvertPipeline};
            for format in [
                PixelFormat::BayerRGGB,
                PixelFormat::BayerBGGR,
                PixelFormat::BayerGRBG,
                PixelFormat::BayerGBRG,
                PixelFormat::YUYV,
            ] {
                let (width, height) = (21u32, 19u32);
                let stride = if format.is_bayer() {
                    width
                } else {
                    width.div_ceil(2) * 4
                };
                let frame = CameraFrame {
                    width,
                    height,
                    stride,
                    format,
                    data: FrameData::from_owned_vec(
                        (0..stride * height)
                            .map(|i| ((i * 37 + i / stride * 13) % 256) as u8)
                            .collect(),
                    ),
                    yuv_planes: None,
                    captured_at: std::time::Instant::now(),
                    sensor_timestamp_ns: None,
                    libcamera_metadata: format.is_bayer().then_some(FrameMetadata {
                        colour_gains: Some([1.3, 0.9]),
                        ..Default::default()
                    }),
                };
                for method in [
                    DemosaicMethod::Bilinear,
                    DemosaicMethod::Rcd,
                    DemosaicMethod::Lmmse,
                ] {
                    if !format.is_bayer() && method != DemosaicMethod::Bilinear {
                        continue;
                    }
                    let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
                    let mut converter = GpuConvertPipeline::new().await.unwrap();
                    if format.is_bayer() {
                        converter
                            .convert_still_bayer_region(
                                &input,
                                method,
                                (0, 0, width, height),
                                true,
                                crate::shaders::BayerRenderingIntent::OrdinaryStillSdr,
                            )
                            .unwrap();
                    } else {
                        converter.convert(&input).unwrap();
                    }
                    let full = converter.read_rgba_to_cpu(width, height).await.unwrap();
                    // Direct regional readback must preserve bytes and reject bad bounds.
                    for region in [
                        (0, 0, width, height),
                        (3, 2, 9, 7),
                        (20, 18, 1, 1),
                        (21, 19, 0, 0),
                    ] {
                        let (x, y, w, h) = region;
                        assert_eq!(
                            converter.read_rgba_region_to_cpu(region).await.unwrap(),
                            PostProcessor::crop_rgba(&full, width, height, x, y, w, h).unwrap()
                        );
                    }
                    assert!(
                        converter
                            .read_rgba_region_to_cpu((u32::MAX, 0, 2, 1))
                            .await
                            .is_err()
                    );
                    for code in 0..=14 {
                        let config = PostProcessingConfig {
                            demosaic_method: method,
                            filter_type: FilterType::from_gpu_filter_code(code),
                            crop_rect: Some((3, 2, 15, 13)),
                            zoom_level: 1.7,
                            brightness: 0.04,
                            contrast: 1.1,
                            saturation: 0.9,
                            sharpening: true,
                        };
                        let processor = PostProcessor::new(config);
                        let reference = processor
                            .process(Arc::new(CameraFrame {
                                data: FrameData::from_owned_vec(full.clone()),
                                format: PixelFormat::RGBA,
                                stride: width * 4,
                                libcamera_metadata: None,
                                ..frame.clone()
                            }))
                            .await
                            .unwrap();
                        let actual = processor.process(Arc::new(frame.clone())).await.unwrap();
                        assert_eq!(
                            (actual.width, actual.height),
                            (reference.width, reference.height)
                        );
                        assert_eq!(
                            actual.image, reference.image,
                            "{format:?} {method:?} filter={code}"
                        );
                    }
                }
            }
        });
    }

    #[tokio::test]
    async fn normal_photo_normalizes_padded_bgra() {
        let frame = CameraFrame {
            width: 2,
            height: 2,
            data: crate::backends::camera::types::FrameData::from_owned_vec(vec![
                30, 20, 10, 255, 60, 50, 40, 255, 99, 99, 99, 99, 90, 80, 70, 255, 120, 110, 100,
                255,
            ]),
            format: PixelFormat::BGRA,
            stride: 12,
            yuv_planes: None,
            captured_at: std::time::Instant::now(),
            sensor_timestamp_ns: None,
            libcamera_metadata: None,
        };
        let result = PostProcessor::new(PostProcessingConfig::default())
            .process(Arc::new(frame))
            .await
            .unwrap();
        assert_eq!(
            result.image.as_raw(),
            &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        );
    }

    #[tokio::test]
    async fn normal_photo_normalizes_other_rendered_formats() {
        for (format, stride, data, expected) in [
            (
                PixelFormat::ABGR,
                12,
                vec![
                    255, 30, 20, 10, 255, 60, 50, 40, 99, 99, 99, 99, 255, 90, 80, 70, 255, 120,
                    110, 100,
                ],
                vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
            ),
            (
                PixelFormat::RGB24,
                8,
                vec![10, 20, 30, 40, 50, 60, 99, 99, 70, 80, 90, 100, 110, 120],
                vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
            ),
            (
                PixelFormat::Gray8,
                3,
                vec![10, 40, 99, 70, 100],
                vec![10, 10, 10, 40, 40, 40, 70, 70, 70, 100, 100, 100],
            ),
        ] {
            let frame = CameraFrame {
                width: 2,
                height: 2,
                data: crate::backends::camera::types::FrameData::from_owned_vec(data),
                format,
                stride,
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: None,
            };
            let result = PostProcessor::new(PostProcessingConfig::default())
                .process(Arc::new(frame))
                .await;
            assert_eq!(result.unwrap().image.as_raw(), &expected, "{format:?}");
        }
    }

    #[tokio::test]
    async fn normal_photo_rejects_overlapping_rendered_rows() {
        for format in [PixelFormat::RGB24, PixelFormat::Gray8] {
            let frame = CameraFrame {
                width: 2,
                height: 2,
                data: crate::backends::camera::types::FrameData::from_owned_vec(vec![0; 16]),
                format,
                stride: 1,
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: None,
            };
            assert!(
                PostProcessor::new(PostProcessingConfig::default())
                    .process(Arc::new(frame))
                    .await
                    .is_err()
            );
        }
    }

    #[test]
    fn normal_photo_decode_failure_is_not_rendered_as_raw_bytes() {
        let Some((_guard, _device, _queue)) = crate::test_gpu::headless_device_with_compute()
        else {
            crate::test_gpu::skip_no_gpu(
                "normal_photo_decode_failure_is_not_rendered_as_raw_bytes",
            );
            return;
        };
        pollster::block_on(async {
            let frame = CameraFrame {
                width: 8,
                height: 8,
                stride: 40,
                format: PixelFormat::BayerRGGB,
                data: crate::backends::camera::types::FrameData::from_owned_vec(vec![64; 320]),
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: None,
            };
            for filter_type in [FilterType::Standard, FilterType::Sepia] {
                let processor = PostProcessor::new(PostProcessingConfig {
                    filter_type,
                    ..Default::default()
                });
                assert!(
                    processor.process(Arc::new(frame.clone())).await.is_err(),
                    "Decode is required even when a filter is optional"
                );
            }
        });
    }

    #[test]
    fn normal_photo_float_raw_preserves_negative_headroom_through_ccm() {
        let Some((_guard, _device, _queue)) = crate::test_gpu::headless_device_with_compute()
        else {
            crate::test_gpu::skip_no_gpu(
                "normal_photo_float_raw_preserves_negative_headroom_through_ccm",
            );
            return;
        };
        pollster::block_on(async {
            let data = (0..16)
                .map(|i| match (i % 4 % 2, i / 4 % 2) {
                    (0, 0) => 16,
                    (1, 1) => 96,
                    _ => 64,
                })
                .collect();
            let frame = CameraFrame {
                width: 4,
                height: 4,
                stride: 4,
                format: PixelFormat::BayerRGGB,
                data: crate::backends::camera::types::FrameData::from_owned_vec(data),
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: Some(crate::backends::camera::types::FrameMetadata {
                    colour_gains: Some([1.0, 1.0]),
                    black_level: Some(32.0 / 255.0),
                    colour_correction_matrix: Some([
                        [1.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]),
                    ..Default::default()
                }),
            };
            let processor = PostProcessor::new(PostProcessingConfig::default());
            let out = processor.process(Arc::new(frame.clone())).await.unwrap();
            let encode = |v: f32| {
                let v = if v <= 0.0031308 {
                    v * 12.92
                } else {
                    1.055 * v.powf(1.0 / 2.4) - 0.055
                };
                (v * 255.0).round() as u8
            };
            let expected = [
                encode(16.0 / 223.0),
                encode(32.0 / 223.0),
                encode(64.0 / 223.0),
            ];
            let actual = out.image.get_pixel(1, 1).0;
            for c in 0..3 {
                assert!(
                    actual[c].abs_diff(expected[c]) <= 1,
                    "{actual:?} != {expected:?}"
                );
            }
            // Filtering and geometry must consume the canonical float result,
            // not silently take the retired legacy Bayer conversion route.
            let config = PostProcessingConfig {
                filter_type: FilterType::Sepia,
                crop_rect: Some((1, 1, 2, 2)),
                zoom_level: 2.0,
                ..Default::default()
            };
            let processor = PostProcessor::new(config);
            let filtered = processor.process(Arc::new(frame.clone())).await.unwrap();
            let rgba = out
                .image
                .pixels()
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect();
            let reference = processor
                .process(Arc::new(CameraFrame {
                    stride: 16,
                    format: PixelFormat::RGBA,
                    data: crate::backends::camera::types::FrameData::from_owned_vec(rgba),
                    libcamera_metadata: None,
                    ..frame
                }))
                .await
                .unwrap();
            assert_eq!((filtered.width, filtered.height), (1, 1));
            assert_eq!(filtered.image, reference.image);
        });
    }

    #[test]
    fn explicit_white_levels_normalize_single_gpu() {
        let Some((_guard, _device, _queue)) = crate::test_gpu::headless_device_with_compute()
        else {
            crate::test_gpu::skip_no_gpu("explicit_white_levels_normalize_single_gpu");
            return;
        };
        pollster::block_on(async {
            use crate::backends::camera::types::{FrameData, FrameMetadata};
            let patterns = [
                (PixelFormat::BayerRGGB, [0, 1, 2, 3]),
                (PixelFormat::BayerBGGR, [3, 2, 1, 0]),
                (PixelFormat::BayerGRBG, [1, 0, 3, 2]),
                (PixelFormat::BayerGBRG, [2, 3, 0, 1]),
            ];
            // All channels normalize to 0.5, but each CFA site has a different pedestal.
            let levels = [20.0, 30.0, 40.0, 50.0].map(|v| v / 255.0);
            let codes = [80, 100, 120, 140];
            for (format, sites) in patterns {
                let pixels = (0..25)
                    .map(|i| codes[sites[(i / 5 % 2) * 2 + i % 5 % 2]])
                    .collect();
                let frame = CameraFrame {
                    width: 5,
                    height: 5,
                    stride: 5,
                    format,
                    data: FrameData::from_owned_vec(pixels),
                    yuv_planes: None,
                    captured_at: std::time::Instant::now(),
                    sensor_timestamp_ns: None,
                    libcamera_metadata: Some(FrameMetadata {
                        colour_gains: Some([1.0, 1.0]),
                        sensor_black_levels: Some(levels),
                        sensor_white_levels: Some([140.0, 170.0, 200.0, 230.0].map(|v| v / 255.0)),
                        ..Default::default()
                    }),
                };
                let out = PostProcessor::new(PostProcessingConfig::default())
                    .process(Arc::new(frame))
                    .await
                    .unwrap();
                for pixel in out.image.pixels() {
                    for c in pixel.0 {
                        assert!(c.abs_diff(188) <= 1, "{format:?} -> {pixel:?}");
                    }
                }
            }
        });
    }

    #[test]
    fn normal_photo_float_raw_four_cfas_per_site_black() {
        let Some((_guard, _device, _queue)) = crate::test_gpu::headless_device_with_compute()
        else {
            crate::test_gpu::skip_no_gpu("normal_photo_float_raw_four_cfas_per_site_black");
            return;
        };
        pollster::block_on(async {
            use crate::backends::camera::types::{FrameData, FrameMetadata};
            let patterns = [
                (PixelFormat::BayerRGGB, [0, 1, 2, 3]),
                (PixelFormat::BayerBGGR, [3, 2, 1, 0]),
                (PixelFormat::BayerGRBG, [1, 0, 3, 2]),
                (PixelFormat::BayerGBRG, [2, 3, 0, 1]),
            ];
            // All channels normalize to 0.25, but each CFA site has a different pedestal.
            let levels = [15.0 / 255.0, 35.0 / 255.0, 55.0 / 255.0, 75.0 / 255.0];
            let codes = [75, 90, 105, 120];
            for (format, sites) in patterns {
                let pixels = (0..25)
                    .map(|i| codes[sites[(i / 5 % 2) * 2 + i % 5 % 2]])
                    .collect();
                let frame = CameraFrame {
                    width: 5,
                    height: 5,
                    stride: 5,
                    format,
                    data: FrameData::from_owned_vec(pixels),
                    yuv_planes: None,
                    captured_at: std::time::Instant::now(),
                    sensor_timestamp_ns: None,
                    libcamera_metadata: Some(FrameMetadata {
                        colour_gains: Some([1.0, 1.0]),
                        sensor_black_levels: Some(levels),
                        ..Default::default()
                    }),
                };
                let out = PostProcessor::new(PostProcessingConfig::default())
                    .process(Arc::new(frame))
                    .await
                    .unwrap();
                for pixel in out.image.pixels() {
                    for c in pixel.0 {
                        assert!(c.abs_diff(137) <= 1, "{format:?} -> {pixel:?}");
                    }
                }
            }
        });
    }

    #[test]
    fn normal_photo_float_raw_clipped_highlights_do_not_turn_magenta() {
        let Some((_guard, _device, _queue)) = crate::test_gpu::headless_device_with_compute()
        else {
            crate::test_gpu::skip_no_gpu(
                "normal_photo_float_raw_clipped_highlights_do_not_turn_magenta",
            );
            return;
        };
        pollster::block_on(async {
            use crate::backends::camera::types::{FrameData, FrameMetadata, RawLayout};
            let patterns = [
                (PixelFormat::BayerRGGB, [0, 1, 2, 3]),
                (PixelFormat::BayerBGGR, [3, 2, 1, 0]),
                (PixelFormat::BayerGRBG, [1, 0, 3, 2]),
                (PixelFormat::BayerGBRG, [2, 3, 0, 1]),
            ];
            for codes in [[0.75_f32, 0.99, 0.99, 0.95], [1.0; 4]] {
                for (format, sites) in patterns {
                    let mut data = vec![0xa5; 60];
                    for y in 0..5 {
                        for x in 0..5 {
                            let code = (codes[sites[(y % 2) * 2 + x % 2]] * 65535.0).round() as u16;
                            data[y * 12 + x * 2..y * 12 + x * 2 + 2]
                                .copy_from_slice(&code.to_le_bytes());
                        }
                    }
                    let frame = CameraFrame {
                        width: 5,
                        height: 5,
                        stride: 12,
                        format,
                        data: FrameData::from_owned_vec(data),
                        yuv_planes: None,
                        captured_at: std::time::Instant::now(),
                        sensor_timestamp_ns: None,
                        libcamera_metadata: Some(FrameMetadata {
                            raw_layout: Some(RawLayout::Unpacked16 { bits: 16 }),
                            colour_gains: Some([2.0, 1.5]),
                            colour_correction_matrix: Some([
                                [2.0, -0.75, -0.25],
                                [-0.5, 2.0, -0.5],
                                [-0.25, -0.5, 1.75],
                            ]),
                            ..Default::default()
                        }),
                    };
                    let out = PostProcessor::new(PostProcessingConfig::default())
                        .process(Arc::new(frame))
                        .await
                        .unwrap();
                    for pixel in out.image.pixels() {
                        let low = *pixel.0.iter().min().unwrap();
                        let high = *pixel.0.iter().max().unwrap();
                        assert!(
                            low >= 240 && high - low <= 8,
                            "Clipped white became colored: {format:?}, {codes:?} -> {pixel:?}"
                        );
                    }
                }
            }
        });
    }

    #[test]
    fn normal_photo_float_raw_rejects_invalid_metadata_before_gpu() {
        let Some((_guard, _device, _queue)) = crate::test_gpu::headless_device_with_compute()
        else {
            crate::test_gpu::skip_no_gpu(
                "normal_photo_float_raw_rejects_invalid_metadata_before_gpu",
            );
            return;
        };
        pollster::block_on(async {
            use crate::backends::camera::types::{FrameData, FrameMetadata, RawLayout};
            for metadata in [
                FrameMetadata {
                    colour_gains: Some([f32::NAN, 1.0]),
                    ..Default::default()
                },
                FrameMetadata {
                    colour_gains: Some([0.0, 1.0]),
                    ..Default::default()
                },
                FrameMetadata {
                    sensor_black_levels: Some([0.0, 1.0, 0.0, 0.0]),
                    ..Default::default()
                },
                FrameMetadata {
                    colour_correction_matrix: Some([[f32::INFINITY; 3]; 3]),
                    ..Default::default()
                },
            ] {
                let frame = CameraFrame {
                    width: 4,
                    height: 4,
                    stride: 4,
                    format: PixelFormat::BayerRGGB,
                    data: FrameData::from_owned_vec(vec![64; 16]),
                    yuv_planes: None,
                    captured_at: std::time::Instant::now(),
                    sensor_timestamp_ns: None,
                    libcamera_metadata: Some(FrameMetadata {
                        raw_layout: Some(RawLayout::U8),
                        ..metadata
                    }),
                };
                assert!(
                    PostProcessor::new(PostProcessingConfig::default())
                        .process(Arc::new(frame))
                        .await
                        .is_err(),
                    "Invalid metadata must not poison persistent GPU buffers"
                );
            }
        });
    }

    #[test]
    fn normal_photo_float_raw_sensor_knee_preserves_unclipped_color() {
        let Some((_guard, _device, _queue)) = crate::test_gpu::headless_device_with_compute()
        else {
            crate::test_gpu::skip_no_gpu(
                "normal_photo_float_raw_sensor_knee_preserves_unclipped_color",
            );
            return;
        };
        pollster::block_on(async {
            use crate::backends::camera::types::{FrameData, FrameMetadata, RawLayout};
            let ccm = [[2.0, -0.75, -0.25], [-0.5, 2.0, -0.5], [-0.25, -0.5, 1.75]];
            let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
            for (sensor, gains, matrix) in [
                ([0.9_f32, 0.1, 0.1], [2.0, 1.5], identity),
                ([1.0, 0.1, 0.1], [2.0, 1.5], identity),
                ([0.6, 0.6, 0.55], [2.0, 1.5], ccm),
                ([0.75, 0.975, 0.95], [2.0, 1.5], ccm),
                ([0.75, 0.9875, 0.95], [2.0, 1.5], ccm),
                ([0.75, 0.99, 0.95], [2.0, 1.5], ccm),
            ] {
                let codes = sensor.map(|v| (v * 65535.0).round() as u16);
                let sensor = codes.map(|v| f64::from(v) / 65535.0);
                let mut data = Vec::new();
                for y in 0..4 {
                    for x in 0..4 {
                        let channel = match (x % 2, y % 2) {
                            (0, 0) => 0,
                            (1, 1) => 2,
                            _ => 1,
                        };
                        data.extend(codes[channel].to_le_bytes());
                    }
                }
                let frame = CameraFrame {
                    width: 4,
                    height: 4,
                    stride: 8,
                    format: PixelFormat::BayerRGGB,
                    data: FrameData::from_owned_vec(data),
                    yuv_planes: None,
                    captured_at: std::time::Instant::now(),
                    sensor_timestamp_ns: None,
                    libcamera_metadata: Some(FrameMetadata {
                        raw_layout: Some(RawLayout::Unpacked16 { bits: 16 }),
                        colour_gains: Some(gains),
                        colour_correction_matrix: Some(matrix),
                        ..Default::default()
                    }),
                };
                let balanced = [
                    sensor[0] * f64::from(gains[0]),
                    sensor[1],
                    sensor[2] * f64::from(gains[1]),
                ];
                let peak = sensor.into_iter().fold(0.0_f64, f64::max);
                // Independent f64 oracle for the intentional C1 sensor shoulder;
                // no general WB cap applies below the final tenth of headroom.
                let t = ((peak - 0.9) / 0.1).clamp(0.0, 1.0);
                let blend = t * t * (3.0 - 2.0 * t);
                let linear: [f64; 3] = std::array::from_fn(|row| {
                    let a: f64 = (0..3)
                        .map(|c| f64::from(matrix[row][c]) * balanced[c])
                        .sum();
                    let b: f64 = (0..3)
                        .map(|c| f64::from(matrix[row][c]) * balanced[c].clamp(0.0, 1.0))
                        .sum();
                    (1.0 - blend) * a.clamp(0.0, 1.0) + blend * b.clamp(0.0, 1.0)
                });
                let expected = linear.map(|v| {
                    let v = v.clamp(0.0, 1.0);
                    let encoded = if v <= 0.0031308 {
                        12.92 * v
                    } else {
                        1.055 * v.powf(1.0 / 2.4) - 0.055
                    };
                    (encoded * 255.0).round() as u8
                });
                let out = PostProcessor::new(PostProcessingConfig::default())
                    .process(Arc::new(frame))
                    .await
                    .unwrap();
                for pixel in out.image.pixels() {
                    for c in 0..3 {
                        assert!(
                            pixel[c].abs_diff(expected[c]) <= 1,
                            "{sensor:?}: {pixel:?} != {expected:?}"
                        );
                    }
                }
            }
        });
    }

    #[test]
    fn normal_photo_float_raw_metadata_policy_and_highlight_reference() {
        let Some((_guard, _device, _queue)) = crate::test_gpu::headless_device_with_compute()
        else {
            crate::test_gpu::skip_no_gpu(
                "normal_photo_float_raw_metadata_policy_and_highlight_reference",
            );
            return;
        };
        pollster::block_on(async {
            use crate::backends::camera::types::{FrameData, FrameMetadata, RawLayout};
            let base = CameraFrame {
                width: 4,
                height: 4,
                stride: 4,
                format: PixelFormat::BayerRGGB,
                data: FrameData::from_owned_vec([64, 128, 64, 128, 128, 192, 128, 192].repeat(2)),
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: Some(FrameMetadata {
                    raw_layout: Some(RawLayout::U8),
                    ..Default::default()
                }),
            };
            let processor = PostProcessor::new(PostProcessingConfig::default());
            let awb = processor.process(Arc::new(base.clone())).await.unwrap();
            let mut ccm_only = base.clone();
            ccm_only
                .libcamera_metadata
                .as_mut()
                .unwrap()
                .colour_correction_matrix =
                Some([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.5]]);
            let corrected = processor.process(Arc::new(ccm_only)).await.unwrap();
            let encode_linear =
                |v: f32| ((1.055 * v.powf(1.0 / 2.4) - 0.055) * 255.0).round() as u8;
            let expected = [64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0].map(encode_linear);
            assert_ne!(
                awb.image, corrected.image,
                "Supplied CCM must run with GPU gray-world AWB"
            );
            for pixel in corrected.image.pixels() {
                for c in 0..3 {
                    assert!(pixel[c].abs_diff(expected[c]) <= 1);
                }
            }
            let mut supplied = base;
            supplied.libcamera_metadata.as_mut().unwrap().colour_gains = Some([8.0, 1.0]);
            let out = processor.process(Arc::new(supplied)).await.unwrap();
            // Ordinary SDR independently bounds channels, not max-RGB ratios.
            let encode = |v: f32| ((1.055 * v.powf(1.0 / 2.4) - 0.055) * 255.0).round() as u8;
            let expected = [255, encode(128.0 / 255.0), encode(192.0 / 255.0)];
            for pixel in out.image.pixels() {
                for c in 0..3 {
                    assert!(
                        pixel[c].abs_diff(expected[c]) <= 1,
                        "{pixel:?} != {expected:?}"
                    );
                }
            }
        });
    }

    #[test]
    fn review_r1_absent_gains_raw12_packing_parity() {
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu("review_r1_absent_gains_raw12_packing_parity");
            return;
        };
        pollster::block_on(async {
            use crate::backends::camera::types::{FrameData, FrameMetadata, RawLayout};
            let processor = PostProcessor::new(PostProcessingConfig::default());
            let mut outputs = Vec::new();
            for bits in [10u32, 12, 14] {
                for packed in [false, true] {
                    let mut data = Vec::new();
                    for y in 0..4 {
                        let codes: Vec<u16> = (0..4)
                            .map(|x| match (x % 2, y % 2) {
                                (0, 0) => 1u16 << (bits - 4),
                                (1, 1) => 3u16 << (bits - 4),
                                _ => 2u16 << (bits - 4),
                            })
                            .collect();
                        if packed {
                            let group = if bits == 12 { 2 } else { 4 };
                            for g in codes.chunks(group) {
                                let low = bits - 8;
                                data.extend(g.iter().map(|c| (c >> low) as u8));
                                let tail = g.iter().enumerate().fold(0u32, |a, (i, c)| {
                                    a | ((u32::from(*c) & ((1 << low) - 1)) << (i as u32 * low))
                                });
                                for b in 0..group as u32 * low / 8 {
                                    data.push((tail >> (b * 8)) as u8);
                                }
                            }
                        } else {
                            for c in codes {
                                data.extend(c.to_le_bytes());
                            }
                        }
                    }
                    let frame = CameraFrame {
                        width: 4,
                        height: 4,
                        stride: if packed { bits / 2 } else { 8 },
                        format: PixelFormat::BayerRGGB,
                        data: FrameData::from_owned_vec(data),
                        yuv_planes: None,
                        captured_at: std::time::Instant::now(),
                        sensor_timestamp_ns: None,
                        libcamera_metadata: Some(FrameMetadata {
                            raw_layout: Some(if packed {
                                RawLayout::Csi2 { bits }
                            } else {
                                RawLayout::Unpacked16 { bits }
                            }),
                            ..Default::default()
                        }),
                    };
                    let out = processor.process(Arc::new(frame)).await.unwrap();
                    outputs.push(out.image.get_pixel(1, 1).0);
                }
                assert_eq!(
                    outputs[outputs.len() - 2],
                    outputs[outputs.len() - 1],
                    "RAW{bits} AWB must not depend on packing"
                );
            }
            assert_eq!(outputs[2], [99, 99, 99]);
        });
    }

    #[test]
    fn review_r5_extreme_finite_metadata_is_not_successful_black() {
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu(
                "review_r5_extreme_finite_metadata_is_not_successful_black",
            );
            return;
        };
        pollster::block_on(async {
            use crate::backends::camera::types::{FrameData, FrameMetadata, RawLayout};
            let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
            let processor = PostProcessor::new(PostProcessingConfig::default());
            let mut converter = crate::shaders::GpuConvertPipeline::new().await.unwrap();
            for (gains, ccm) in [
                (
                    [f32::MAX, 1.0],
                    [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                ),
                ([1.0, 1.0], [[f32::MAX; 3]; 3]),
                ([f32::MAX, 1.0], identity),
            ] {
                let frame = CameraFrame {
                    width: 4,
                    height: 4,
                    stride: 4,
                    format: PixelFormat::BayerRGGB,
                    data: FrameData::from_owned_vec(vec![255; 16]),
                    yuv_planes: None,
                    captured_at: std::time::Instant::now(),
                    sensor_timestamp_ns: None,
                    libcamera_metadata: Some(FrameMetadata {
                        raw_layout: Some(RawLayout::U8),
                        colour_gains: Some(gains),
                        colour_correction_matrix: Some(ccm),
                        ..Default::default()
                    }),
                };
                assert!(
                    processor.process(Arc::new(frame.clone())).await.is_err(),
                    "unsafe finite metadata must not return successful black"
                );
                let input = crate::shaders::GpuFrameInput::from_camera_frame(&frame).unwrap();
                assert_eq!(
                    converter.convert_bayer(&input).unwrap_err(),
                    crate::shaders::BayerError::UnsafeArithmetic
                );
            }
            // This is an arithmetic contract, not a practical/calibration gain limit.
            let mut frame = CameraFrame {
                width: 4,
                height: 4,
                stride: 4,
                format: PixelFormat::BayerRGGB,
                data: FrameData::from_owned_vec(vec![255; 16]),
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: Some(FrameMetadata {
                    raw_layout: Some(RawLayout::U8),
                    colour_gains: Some([1e20, 1e20]),
                    colour_correction_matrix: Some(identity),
                    ..Default::default()
                }),
            };
            // Fully clipped samples use the camera-space highlight fallback,
            // even when the supplied gains are very large but safe.
            let out = processor.process(Arc::new(frame.clone())).await.unwrap();
            assert_eq!(out.image.get_pixel(1, 1).0, [255, 255, 255]);
            // Below sensor clipping, extreme over-white float ratios still
            // survive for the legacy contract; ordinary SDR clips independently.
            frame.data = FrameData::from_owned_vec(vec![128; 16]);
            let out = processor.process(Arc::new(frame.clone())).await.unwrap();
            assert_eq!(out.image.get_pixel(1, 1).0, [255, 188, 255]);
            let out = processor
                .for_continuous_frames()
                .process(Arc::new(frame))
                .await
                .unwrap();
            assert_eq!(out.image.get_pixel(1, 1).0, [255, 0, 255]);
        });
    }

    #[test]
    fn normal_photo_float_raw_explicit_padded_packings() {
        let Some((_guard, _device, _queue)) = crate::test_gpu::headless_device_with_compute()
        else {
            crate::test_gpu::skip_no_gpu("normal_photo_float_raw_explicit_padded_packings");
            return;
        };
        pollster::block_on(async {
            use crate::backends::camera::types::{FrameData, FrameMetadata, RawLayout};
            let layouts = [
                RawLayout::U8,
                RawLayout::Unpacked16 { bits: 10 },
                RawLayout::Unpacked16 { bits: 12 },
                RawLayout::Unpacked16 { bits: 14 },
                RawLayout::Unpacked16 { bits: 16 },
                RawLayout::Csi2 { bits: 10 },
                RawLayout::Csi2 { bits: 12 },
                RawLayout::Csi2 { bits: 14 },
            ];
            for (layout, width) in layouts
                .into_iter()
                .rev()
                .flat_map(|l| [5u32, 513].map(|w| (l, w)))
            {
                let bits = match layout {
                    RawLayout::U8 => 8,
                    RawLayout::Unpacked16 { bits } | RawLayout::Csi2 { bits } => bits,
                };
                let code = ((1u32 << bits) - 1) / 4;
                let mut row = Vec::new();
                match layout {
                    RawLayout::U8 => row.resize(width as usize, code as u8),
                    RawLayout::Unpacked16 { .. } => {
                        for _ in 0..width {
                            row.extend_from_slice(&(code as u16).to_le_bytes());
                        }
                    }
                    RawLayout::Csi2 { .. } => {
                        let count = if bits == 12 { 2 } else { 4 };
                        let low_bits = bits - 8;
                        for _ in 0..width.div_ceil(count) {
                            row.extend(std::iter::repeat_n(
                                (code >> low_bits) as u8,
                                count as usize,
                            ));
                            let mut packed = 0u32;
                            for i in 0..count {
                                packed |= (code & ((1 << low_bits) - 1)) << (i * low_bits);
                            }
                            for b in 0..(count * low_bits / 8) {
                                row.push((packed >> (b * 8)) as u8);
                            }
                        }
                    }
                }
                row.extend_from_slice(&[0xee, 0xdd, 0xcc]);
                let stride = row.len() as u32;
                let frame = CameraFrame {
                    width,
                    height: 5,
                    stride,
                    format: PixelFormat::BayerBGGR,
                    data: FrameData::from_owned_vec(row.repeat(5)),
                    yuv_planes: None,
                    captured_at: std::time::Instant::now(),
                    sensor_timestamp_ns: None,
                    libcamera_metadata: Some(FrameMetadata {
                        raw_layout: Some(layout),
                        colour_gains: Some([1.0, 1.0]),
                        ..Default::default()
                    }),
                };
                let out = PostProcessor::new(PostProcessingConfig::default())
                    .process(Arc::new(frame))
                    .await
                    .unwrap_or_else(|e| panic!("{layout:?}: {e}"));
                let v = code as f32 / ((1u32 << bits) - 1) as f32;
                let expected = ((1.055 * v.powf(1.0 / 2.4) - 0.055) * 255.0).round() as u8;
                for pixel in out.image.pixels() {
                    for c in pixel.0 {
                        assert!(
                            c.abs_diff(expected) <= 1,
                            "{layout:?}: {pixel:?} != {expected}"
                        );
                    }
                }
            }
        });
    }

    #[test]
    fn test_default_config() {
        let config = PostProcessingConfig::default();
        assert!(!config.sharpening);
        assert_eq!(config.brightness, 0.0);
        assert_eq!(config.contrast, 1.0);
        assert_eq!(config.saturation, 1.0);
    }

    #[test]
    fn padded_rgba_rows_are_compacted_before_cpu_processing() {
        let padded = [
            1, 2, 3, 4, 5, 6, 7, 8, 99, 99, 99, 99, // row 0 + padding
            9, 10, 11, 12, 13, 14, 15, 16, 88, 88, 88, 88, // row 1 + padding
        ];

        let compact = PostProcessor::copy_rgba_without_padding(&padded, 2, 2, 12).unwrap();

        assert_eq!(
            compact,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
    }
}

#[cfg(test)]
mod method_selection_tests {
    use super::*;
    use crate::shaders::DemosaicMethod;
    #[test]
    fn photo_processor_honours_all_three_selected_methods() {
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu("photo_processor_honours_all_three_selected_methods");
            return;
        };
        pollster::block_on(async {
            let data: Vec<u8> = (0..19)
                .flat_map(|y| {
                    (0..21).map(move |x| {
                        (25 + (x * 37 + y * 53 + x * y * 7) % 97 + if x > 10 { 40 } else { 0 })
                            as u8
                    })
                })
                .collect();
            let frame = Arc::new(CameraFrame {
                width: 21,
                height: 19,
                stride: 21,
                format: PixelFormat::BayerRGGB,
                data: crate::backends::camera::types::FrameData::from_owned_vec(data),
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: Some(crate::backends::camera::types::FrameMetadata {
                    colour_gains: Some([1.0, 1.0]),
                    ..Default::default()
                }),
            });
            let mut outputs = Vec::new();
            for method in [
                DemosaicMethod::Rcd,
                DemosaicMethod::Lmmse,
                DemosaicMethod::Bilinear,
            ] {
                let config = PostProcessingConfig {
                    demosaic_method: method,
                    ..Default::default()
                };
                outputs.push(
                    PostProcessor::new(config)
                        .process(frame.clone())
                        .await
                        .unwrap()
                        .image
                        .into_raw(),
                );
            }
            assert_ne!(outputs[0], outputs[1]);
            assert_ne!(outputs[0], outputs[2]);
            assert_ne!(outputs[1], outputs[2]);
            assert_eq!(
                outputs[2],
                PostProcessor::new(PostProcessingConfig::default())
                    .process(frame)
                    .await
                    .unwrap()
                    .image
                    .into_raw()
            );
        });
    }
}
