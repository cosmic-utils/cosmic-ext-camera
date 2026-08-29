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
use crate::backends::camera::types::{CameraFrame, PixelFormat, SensorRotation};
use crate::shaders::{GpuFrameInput, apply_filter_gpu_rgba, get_gpu_convert_pipeline};
use image::RgbImage;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Post-processing configuration
#[derive(Debug, Clone)]
pub struct PostProcessingConfig {
    /// Enable color correction
    pub color_correction: bool,
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
    /// Sensor rotation to correct the image orientation
    pub rotation: SensorRotation,
    /// Mirror the image horizontally (selfie capture, mirroring the preview).
    pub mirror_horizontal: bool,
}

impl Default for PostProcessingConfig {
    fn default() -> Self {
        Self {
            color_correction: true,
            sharpening: false,
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            filter_type: FilterType::Standard,
            crop_rect: None,
            zoom_level: 1.0,
            rotation: SensorRotation::None,
            mirror_horizontal: false,
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
}

impl PostProcessor {
    /// Create a new post-processor with the given configuration
    pub fn new(config: PostProcessingConfig) -> Self {
        Self { config }
    }

    /// Process a captured frame asynchronously
    ///
    /// This runs all post-processing steps using GPU acceleration where available,
    /// with software rendering fallback for systems without GPU support.
    ///
    /// # Arguments
    /// * `frame` - Raw camera frame (RGBA format)
    ///
    /// # Returns
    /// * `Ok(ProcessedImage)` - Processed RGB image
    /// * `Err(String)` - Error message
    pub async fn process(&self, frame: Arc<CameraFrame>) -> Result<ProcessedImage, String> {
        info!(
            width = frame.width,
            height = frame.height,
            format = ?frame.format,
            "Starting post-processing"
        );

        let config = self.config.clone();
        let frame_width = frame.width;
        let frame_height = frame.height;
        let compact_rgba = if frame.format == PixelFormat::RGBA {
            Some(Self::copy_rgba_without_padding(
                frame.data.as_ref(),
                frame.width,
                frame.height,
                frame.stride,
            )?)
        } else {
            None
        };

        // Step 0: Convert to RGBA (with optional integrated filter for Bayer)
        let filtered_rgba: Vec<u8> = if frame.format.is_bayer()
            && config.filter_type != FilterType::Standard
        {
            // Bayer + filter: use integrated debayer+filter pipeline (no GPU round trip)
            debug!(
                format = ?frame.format,
                filter = ?config.filter_type,
                "Converting Bayer + filter in single GPU submission"
            );
            match Self::convert_bayer_and_filter(&frame, config.filter_type).await {
                Ok(rgba) => rgba,
                Err(e) => {
                    warn!(error = %e, "Integrated debayer+filter failed, falling back");
                    // Fallback: separate debayer then filter (both optional enhancements)
                    let rgba = Self::convert_yuv_to_rgba(&frame).await.unwrap_or_else(|e| {
                        warn!(error = %e, "Bayer→RGBA fallback also failed, using raw data");
                        frame.data.to_vec()
                    });
                    apply_filter_gpu_rgba(&rgba, frame_width, frame_height, config.filter_type)
                        .await
                        .unwrap_or(rgba)
                }
            }
        } else if frame.format.is_yuv() || frame.format.is_bayer() {
            debug!(format = ?frame.format, "Converting frame to RGBA for photo processing");
            let rgba = Self::convert_yuv_to_rgba(&frame)
                .await
                .map_err(|e| format!("Failed to convert to RGBA: {}", e))?;
            // Apply filter for non-Bayer formats (YUV)
            if config.filter_type != FilterType::Standard {
                match apply_filter_gpu_rgba(&rgba, frame_width, frame_height, config.filter_type)
                    .await
                {
                    Ok(filtered) => {
                        debug!("Filter applied via GPU pipeline (RGBA-native)");
                        filtered
                    }
                    Err(e) => {
                        warn!(error = %e, "GPU filter failed, using unfiltered frame");
                        rgba
                    }
                }
            } else {
                rgba
            }
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

        // Step 2: Apply aspect ratio cropping if configured
        let (cropped_rgba, current_width, current_height) = if let Some((x, y, w, h)) =
            config.crop_rect
        {
            debug!(x, y, width = w, height = h, "Applying aspect ratio crop");
            let cropped = Self::crop_rgba(&filtered_rgba, frame_width, frame_height, x, y, w, h)?;
            (cropped, w, h)
        } else {
            (filtered_rgba, frame_width, frame_height)
        };

        // Step 3: Apply zoom (center crop) if zoom_level > 1.0
        let (final_rgba, final_width, final_height) = if config.zoom_level > 1.0 {
            Self::apply_zoom_crop(
                &cropped_rgba,
                current_width,
                current_height,
                config.zoom_level,
            )?
        } else {
            (cropped_rgba, current_width, current_height)
        };

        // Step 4: Convert filtered RGBA to RGB (drop alpha channel)
        let rgb_image = Self::convert_rgba_to_rgb(&final_rgba, final_width, final_height)?;

        // Step 4.5: Apply rotation correction if needed
        let (rgb_image, final_width, final_height) = if config.rotation != SensorRotation::None {
            debug!(rotation = ?config.rotation, "Applying rotation correction");
            Self::apply_rotation(rgb_image, config.rotation)?
        } else {
            (rgb_image, final_width, final_height)
        };

        // Step 4.6: Mirror horizontally if requested (front-camera selfie mode).
        // Done after rotation so the final orientation is upright before flipping.
        let rgb_image = if config.mirror_horizontal {
            debug!("Mirroring captured image horizontally");
            let mut img = rgb_image;
            image::imageops::flip_horizontal_in_place(&mut img);
            img
        } else {
            rgb_image
        };

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

    /// Convert Bayer frame to RGBA with integrated filter in a single GPU submission.
    ///
    /// This avoids the GPU→CPU→GPU round trip that would occur with separate
    /// debayer and filter pipelines. Pre-computes AWB when ISP gains are absent.
    async fn convert_bayer_and_filter(
        frame: &CameraFrame,
        filter: FilterType,
    ) -> Result<Vec<u8>, String> {
        let buffer_data = frame.data.as_ref();

        // Extract ISP metadata if available; GPU AWB handles the no-gains case
        let (colour_gains, ccm, black_level) = frame
            .libcamera_metadata
            .as_ref()
            .map(|m| (m.colour_gains, m.colour_correction_matrix, m.black_level))
            .unwrap_or((None, None, None));

        let input = GpuFrameInput {
            format: frame.format,
            width: frame.width,
            height: frame.height,
            y_data: buffer_data,
            y_stride: frame.stride,
            uv_data: None,
            uv_stride: 0,
            v_data: None,
            v_stride: 0,
            colour_gains,
            colour_correction_matrix: ccm,
            black_level,
        };

        let mut pipeline_guard = get_gpu_convert_pipeline()
            .await
            .map_err(|e| format!("Failed to get convert pipeline: {}", e))?;
        let pipeline = pipeline_guard
            .as_mut()
            .ok_or("Convert pipeline not initialized")?;

        pipeline
            .convert_and_filter(&input, filter)
            .map_err(|e| format!("Debayer+filter failed: {}", e))?;

        pipeline
            .read_filtered_to_cpu(frame.width, frame.height)
            .await
            .map_err(|e| format!("Failed to read filtered data from GPU: {}", e))
    }

    /// Convert YUV frame to RGBA using GPU compute shader
    ///
    /// Uses the same compute shader as the preview pipeline for consistency.
    async fn convert_yuv_to_rgba(frame: &CameraFrame) -> Result<Vec<u8>, String> {
        // RGBA doesn't need conversion
        if frame.format == PixelFormat::RGBA {
            return Ok(frame.data.as_ref().to_vec());
        }

        let input = GpuFrameInput::from_camera_frame(frame)?;

        // Use GPU compute shader pipeline for conversion
        let mut pipeline_guard = get_gpu_convert_pipeline()
            .await
            .map_err(|e| format!("Failed to get YUV convert pipeline: {}", e))?;

        let pipeline = pipeline_guard
            .as_mut()
            .ok_or("YUV convert pipeline not initialized")?;

        // Run GPU conversion (synchronous, just dispatches compute shader)
        pipeline
            .convert(&input)
            .map_err(|e| format!("YUV→RGBA GPU conversion failed: {}", e))?;

        // Read back RGBA data from GPU to CPU memory
        pipeline
            .read_rgba_to_cpu(frame.width, frame.height)
            .await
            .map_err(|e| format!("Failed to read RGBA from GPU: {}", e))
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

        let rgb_data: Vec<u8> = rgba_data
            .chunks(4)
            .take((width * height) as usize)
            .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
            .collect();

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

    /// Apply rotation correction to an RGB image
    ///
    /// Uses the image crate's rotation methods for efficient CPU rotation.
    /// Rotation is applied at the end of post-processing to correct sensor orientation.
    fn apply_rotation(
        image: RgbImage,
        rotation: SensorRotation,
    ) -> Result<(RgbImage, u32, u32), String> {
        use image::imageops;

        let rotated = match rotation {
            SensorRotation::None => return Ok((image.clone(), image.width(), image.height())),
            // 90 CW sensor -> rotate 90 CCW to correct (same as rotate270 in image crate)
            SensorRotation::Rotate90 => imageops::rotate270(&image),
            // 180 sensor -> rotate 180 to correct
            SensorRotation::Rotate180 => imageops::rotate180(&image),
            // 270 CW sensor -> rotate 90 CW to correct (same as rotate90 in image crate)
            SensorRotation::Rotate270 => imageops::rotate90(&image),
        };

        let (w, h) = rotated.dimensions();
        Ok((rotated, w, h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PostProcessingConfig::default();
        assert!(config.color_correction);
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
