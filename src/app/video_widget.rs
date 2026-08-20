// SPDX-License-Identifier: GPL-3.0-only

//! Custom video widget for efficient camera preview rendering with GPU primitives
//!
//! This widget achieves the same optimizations as iced_video_player:
//! 1. Direct GPU texture updates (no Handle recreation)
//! 2. GPU-side filter processing via WGSL shaders
//! 3. Persistent textures across frames
//! 4. Native RGBA format for simplified processing

use crate::app::state::{FilterType, Message};
use crate::app::video_primitive::{VideoFrame, VideoPrimitive};
use crate::backends::camera::types::{CameraFrame, PixelFormat};
use cosmic::iced::advanced::widget::Tree;
use cosmic::iced::advanced::widget::tree;
use cosmic::iced::advanced::{Clipboard, Shell, Widget, layout};
use cosmic::iced::mouse;
use cosmic::iced::touch;
use cosmic::iced::{Element, Event, Length, Point, Rectangle, Size};
use cosmic::{Renderer, Theme};
use iced_wgpu::primitive::Renderer as PrimitiveRenderer;
use std::collections::HashMap;
use std::sync::Arc;

/// Internal state for tracking pinch-to-zoom gestures
#[derive(Default)]
struct PinchState {
    /// Active finger positions (up to 2 tracked)
    fingers: HashMap<touch::Finger, Point>,
    /// Distance between two fingers when pinch started
    initial_distance: Option<f32>,
    /// Zoom level when pinch gesture started
    zoom_at_pinch_start: f32,
}

/// Content fit mode for video scaling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoContentFit {
    /// Scale to fit within bounds, maintaining aspect ratio (letterboxing)
    Contain,
    /// Scale to fill bounds completely, maintaining aspect ratio (cropping)
    Cover,
}

impl VideoContentFit {
    /// Returns the shader blend value: 0.0 = Contain, 1.0 = Cover
    pub fn blend(&self) -> f32 {
        match self {
            VideoContentFit::Contain => 0.0,
            VideoContentFit::Cover => 1.0,
        }
    }
}

/// Configuration for creating a video widget
#[derive(Debug, Clone, PartialEq)]
pub struct VideoWidgetConfig {
    /// Unique identifier for this video stream
    pub video_id: u64,
    /// How to scale content within bounds
    pub content_fit: VideoContentFit,
    /// Filter to apply to the video
    pub filter_type: FilterType,
    /// Corner radius for rounded corners (0.0 for sharp corners)
    pub corner_radius: f32,
    /// Whether to mirror the video horizontally
    pub mirror_horizontal: bool,
    /// Sensor rotation: 0=None, 1=90CW, 2=180, 3=270CW
    pub rotation: u32,
    /// Optional crop UV coordinates (u_min, v_min, u_max, v_max) in 0-1 range
    pub crop_uv: Option<(f32, f32, f32, f32)>,
    /// Zoom level (1.0 = no zoom, 2.0 = 2x zoom)
    pub zoom_level: f32,
    /// Whether scroll wheel zoom is enabled
    pub scroll_zoom_enabled: bool,
    /// Blend between Contain (0.0) and Cover (1.0) for animated transitions.
    /// When `None`, uses `content_fit.blend()`.
    pub cover_blend: Option<f32>,
    /// Space the UI bars reserve on each edge, in logical px. Four-sided so the
    /// blur letterboxes the correct edges when the device is held sideways.
    pub insets: crate::app::preview_geometry::BarInsets,
    /// Theme background color (sRGB straight, RGBA). Used by the blur shader
    /// to fill the letterbox in Contain / Fit mode instead of returning
    /// transparent — otherwise the COSMIC window background leaks through.
    pub letterbox_color: [f32; 4],
}

/// Video widget that renders camera frames using a custom GPU primitive
pub struct VideoWidget {
    primitive: VideoPrimitive,
    width: Length,
    height: Length,
    aspect_ratio: f32,
    content_fit: VideoContentFit,
    /// Enable scroll wheel zoom (only for main camera preview, not filter picker)
    scroll_zoom_enabled: bool,
    /// Current zoom level (passed through for pinch gesture reference)
    zoom_level: f32,
    /// Shader blend: 0.0 = Contain, 1.0 = Cover
    cover_blend: f32,
    /// Four-sided control-bar reservation used by Contain/Fit geometry.
    fit_insets: crate::app::preview_geometry::BarInsets,
}

impl VideoWidget {
    /// Create a new video widget from a camera frame
    ///
    /// # Arguments
    /// * `frame` - The camera frame to display
    /// * `config` - Widget configuration options
    pub fn new(frame: Arc<CameraFrame>, config: VideoWidgetConfig) -> Self {
        let mut primitive = VideoPrimitive::new(config.video_id);
        primitive.filter_type = config.filter_type;
        primitive.corner_radius = config.corner_radius;
        primitive.mirror_horizontal = config.mirror_horizontal;
        primitive.rotation = config.rotation;
        primitive.crop_uv = config.crop_uv;
        primitive.zoom_level = config.zoom_level;
        primitive.letterbox_color = config.letterbox_color;

        // Calculate aspect ratio from frame dimensions, adjusted for crop and rotation
        // For 90° and 270° rotations, swap width and height
        let swaps_dimensions = config.rotation == 1 || config.rotation == 3;
        let (effective_width, effective_height) = if swaps_dimensions {
            (frame.height as f32, frame.width as f32)
        } else {
            (frame.width as f32, frame.height as f32)
        };

        let aspect_ratio = if let Some((u_min, v_min, u_max, v_max)) = config.crop_uv {
            // Crop UVs are in texture space (original sensor orientation).
            // Compute the cropped region in texture pixels, then apply
            // the rotation swap to get the displayed aspect ratio.
            let tex_crop_w = (u_max - u_min) * frame.width as f32;
            let tex_crop_h = (v_max - v_min) * frame.height as f32;
            let (crop_width, crop_height) = if swaps_dimensions {
                (tex_crop_h, tex_crop_w)
            } else {
                (tex_crop_w, tex_crop_h)
            };
            if crop_height > 0.0 {
                crop_width / crop_height
            } else {
                16.0 / 9.0
            }
        } else if effective_height > 0.0 {
            effective_width / effective_height
        } else {
            16.0 / 9.0 // Default aspect ratio
        };

        // Create VideoFrame (supports RGBA and YUV formats)
        // IMPORTANT: We share the FrameData without copying to maintain zero-copy from GStreamer
        if frame.width > 0 && frame.height > 0 {
            let stride = if frame.stride > 0 {
                frame.stride
            } else {
                // Fallback based on format
                match frame.format {
                    PixelFormat::RGBA | PixelFormat::ABGR | PixelFormat::BGRA => frame.width * 4,
                    PixelFormat::RGB24 => frame.width * 3, // 3 bytes per pixel
                    PixelFormat::YUYV
                    | PixelFormat::UYVY
                    | PixelFormat::YVYU
                    | PixelFormat::VYUY => {
                        frame.width * 2 // 2 bytes per pixel
                    }
                    PixelFormat::NV12 | PixelFormat::NV21 | PixelFormat::I420 => frame.width, // Y plane stride
                    PixelFormat::Gray8
                    | PixelFormat::BayerRGGB
                    | PixelFormat::BayerBGGR
                    | PixelFormat::BayerGRBG
                    | PixelFormat::BayerGBRG => frame.width, // 1 byte per pixel
                }
            };

            let video_frame = VideoFrame {
                id: config.video_id,
                width: frame.width,
                height: frame.height,
                data: frame.data.clone(), // Clone FrameData - just refcount increment, no data copy
                format: frame.format,
                stride,
                yuv_planes: frame.yuv_planes,
            };

            primitive.update_frame(video_frame);
        }

        Self {
            primitive,
            width: Length::Fill,
            height: Length::Fill,
            aspect_ratio,
            content_fit: config.content_fit,
            scroll_zoom_enabled: config.scroll_zoom_enabled,
            zoom_level: config.zoom_level,
            cover_blend: config
                .cover_blend
                .unwrap_or_else(|| config.content_fit.blend()),
            fit_insets: config.insets,
        }
    }
}

impl Widget<crate::app::Message, Theme, Renderer> for VideoWidget {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<PinchState>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(PinchState::default())
    }

    fn size(&self) -> Size<Length> {
        Size::new(self.width, self.height)
    }

    fn layout(
        &mut self,
        _tree: &mut Tree,
        _renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        // Get the maximum available space
        let max_size = limits.max();

        let final_size = match self.content_fit {
            VideoContentFit::Contain => {
                // Choose the scaling that fits within bounds (letterbox)
                let width = max_size.width;
                let height = max_size.height;

                let width_based_height = width / self.aspect_ratio;
                let height_based_width = height * self.aspect_ratio;

                if width_based_height <= height {
                    Size::new(width, width_based_height)
                } else {
                    Size::new(height_based_width, height)
                }
            }
            VideoContentFit::Cover => {
                // Fill the entire container - the shader handles Cover/Contain via blend
                max_size
            }
        };

        layout::Node::new(final_size)
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: layout::Layout<'_>,
        cursor: mouse::Cursor,
        _renderer: &Renderer,
        _clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        _viewport: &Rectangle,
    ) {
        // Only handle zoom gestures if enabled (photo mode main preview)
        if !self.scroll_zoom_enabled {
            return;
        }

        let bounds = layout.bounds();

        // Handle touch events for pinch-to-zoom
        if let Event::Touch(touch_event) = event {
            let pinch = tree.state.downcast_mut::<PinchState>();

            match touch_event {
                touch::Event::FingerPressed { id, position } => {
                    if bounds.contains(*position) {
                        pinch.fingers.insert(*id, *position);

                        // When second finger lands, start tracking the pinch
                        if pinch.fingers.len() == 2 {
                            let pts: Vec<&Point> = pinch.fingers.values().collect();
                            let dx = pts[0].x - pts[1].x;
                            let dy = pts[0].y - pts[1].y;
                            pinch.initial_distance = Some((dx * dx + dy * dy).sqrt());
                            pinch.zoom_at_pinch_start = self.zoom_level;
                        }
                        return;
                    }
                }
                touch::Event::FingerMoved { id, position } => {
                    if let std::collections::hash_map::Entry::Occupied(mut e) =
                        pinch.fingers.entry(*id)
                    {
                        e.insert(*position);

                        if pinch.fingers.len() == 2
                            && let Some(initial_dist) = pinch.initial_distance
                            && initial_dist > 1.0
                        {
                            let pts: Vec<&Point> = pinch.fingers.values().collect();
                            let dx = pts[0].x - pts[1].x;
                            let dy = pts[0].y - pts[1].y;
                            let current_dist = (dx * dx + dy * dy).sqrt();
                            let scale = current_dist / initial_dist;
                            let new_zoom = (pinch.zoom_at_pinch_start * scale).clamp(1.0, 10.0);
                            shell.publish(Message::PinchZoom(new_zoom));
                        }
                        return;
                    }
                }
                touch::Event::FingerLifted { id, .. } | touch::Event::FingerLost { id, .. } => {
                    if pinch.fingers.remove(id).is_some() {
                        pinch.initial_distance = None;
                        return;
                    }
                }
            }
        }

        // Check if cursor is over the widget bounds (for mouse scroll zoom)
        if !cursor.is_over(bounds) {
            return;
        }

        // Handle mouse wheel scroll for zoom
        if let Event::Mouse(mouse::Event::WheelScrolled { delta }) = event {
            let scroll_delta = match delta {
                mouse::ScrollDelta::Lines { y, .. } => *y,
                mouse::ScrollDelta::Pixels { y, .. } => *y / 50.0, // Normalize pixel scrolling
            };

            if scroll_delta > 0.0 {
                // Scroll up = zoom in
                shell.publish(Message::ZoomIn);
            } else if scroll_delta < 0.0 {
                // Scroll down = zoom out
                shell.publish(Message::ZoomOut);
            }
        }
    }

    fn draw(
        &self,
        _tree: &Tree,
        renderer: &mut Renderer,
        _theme: &Theme,
        _style: &cosmic::iced::advanced::renderer::Style,
        layout: layout::Layout<'_>,
        _cursor: cosmic::iced::mouse::Cursor,
        _viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();

        // Update primitive with viewport size, cover blend, and all bar insets.
        // The shader interpolates UV coordinates and centering based on the blend value.
        self.primitive.update_viewport(
            bounds.width,
            bounds.height,
            self.cover_blend,
            self.fit_insets,
        );

        // Draw the custom primitive using the wgpu renderer's primitive support
        renderer.draw_primitive(bounds, self.primitive.clone());
    }
}

impl<'a> From<VideoWidget> for Element<'a, crate::app::Message, Theme, Renderer> {
    fn from(widget: VideoWidget) -> Self {
        Element::new(widget)
    }
}

/// Create a video widget from a camera frame
///
/// # Arguments
/// * `frame` - The camera frame to display
/// * `config` - Widget configuration options
pub fn video_widget<'a>(
    frame: Arc<CameraFrame>,
    config: VideoWidgetConfig,
) -> Element<'a, crate::app::Message, Theme, Renderer> {
    Element::new(VideoWidget::new(frame, config))
}
