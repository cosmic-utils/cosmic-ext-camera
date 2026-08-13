// SPDX-License-Identifier: GPL-3.0-only

//! QR code overlay module
//!
//! This module provides widgets for rendering QR code detection results
//! as overlays on top of the camera preview. It includes:
//!
//! - Transparent boxes with themed borders around detected QR codes
//! - Context-aware action buttons based on QR code content
//!
//! # Coordinate System
//!
//! QR detections use normalized coordinates (0.0 to 1.0) relative to the
//! camera frame. The overlay widget handles transformation to screen
//! coordinates at render time, accounting for video scaling and letterboxing.

pub mod action_button;
mod widget;

use crate::app::frame_processor::{FrameRegion, QrAction, QrDetection};
use crate::app::preview_geometry::BarInsets;
use crate::app::state::Message;
use crate::backends::camera::types::SensorRotation;
use cosmic::Element;
use cosmic::iced::{Color, Length};

/// Border width for QR overlay boxes (in pixels)
const OVERLAY_BORDER_WIDTH: f32 = 3.0;

/// Corner radius for QR overlay boxes
const OVERLAY_BORDER_RADIUS: f32 = 8.0;

/// Minimum size for overlay boxes (prevents tiny boxes for small QR codes)
const MIN_OVERLAY_SIZE: f32 = 60.0;

/// Gap between QR box and action button
const BUTTON_GAP: f32 = 8.0;

/// Build the QR overlay layer using a custom widget.
///
/// Renders boxes around detected QR codes and action buttons below them. The
/// widget transforms the detection's normalized (0-1) frame coordinates to
/// screen pixels at render time, lerping between the Cover and Contain
/// preview endpoints by `cover_blend` so the boxes track the live preview
/// through a fit/fill transition.
///
/// `top_bar_h` / `bottom_bar_h` are the animated UI bar heights — in Contain
/// mode the video is letterboxed inside the content area between them, so
/// those values are needed to know where the visible video actually sits.
pub fn build_qr_overlay<'a>(
    detections: &[QrDetection],
    frame_width: u32,
    frame_height: u32,
    cover_blend: f32,
    insets: BarInsets,
    rotation: SensorRotation,
    mirrored: bool,
) -> Element<'a, Message> {
    if detections.is_empty() {
        return cosmic::widget::Space::new()
            .width(Length::Fill)
            .height(Length::Fill)
            .into();
    }

    widget::QrOverlayWidget::new(
        detections.to_vec(),
        frame_width,
        frame_height,
        cover_blend,
        insets,
        rotation,
        mirrored,
    )
    .into()
}

/// Get the border color for a QR action type
///
/// Uses COSMIC theme-inspired colors based on the action type.
pub fn get_action_color(action: &QrAction) -> Color {
    match action {
        QrAction::Url(_) => Color::from_rgb(0.29, 0.56, 0.89), // Blue for links
        QrAction::Wifi { .. } => Color::from_rgb(0.30, 0.69, 0.31), // Green for WiFi
        QrAction::Phone(_) => Color::from_rgb(0.61, 0.35, 0.71), // Purple for phone
        QrAction::Email { .. } => Color::from_rgb(0.90, 0.49, 0.13), // Orange for email
        QrAction::Location { .. } => Color::from_rgb(0.96, 0.26, 0.21), // Red for location
        QrAction::Contact(_) => Color::from_rgb(0.00, 0.59, 0.53), // Teal for contacts
        QrAction::Event(_) => Color::from_rgb(0.91, 0.12, 0.39), // Pink for events
        QrAction::Sms { .. } => Color::from_rgb(0.55, 0.76, 0.29), // Light green for SMS
        QrAction::Text(_) => Color::from_rgb(0.62, 0.62, 0.62), // Gray for plain text
    }
}

/// Calculate the source-frame bounds in screen coordinates, animated
/// through the fit/fill transition.
///
/// Returns `(offset_x, offset_y, video_width, video_height)` — the rectangle
/// the *source frame* occupies on screen, with the sensor aspect preserved
/// at both endpoints. Lerps between two endpoints by `cover_blend`:
/// - **Cover** (`cover_blend = 1`): the frame is scaled to fill the
///   container in the minor axis; the major axis overflows the container
///   and is cropped off-screen (so one of `offset_x`/`offset_y` is
///   negative).
/// - **Contain** (`cover_blend = 0`): the frame is letterboxed inside the
///   content area between `top_bar_h` and `bottom_bar_h`.
///
/// QR detections (which use normalized 0-1 frame coordinates) scale into
/// this rectangle, so the on-screen boxes keep the sensor's aspect ratio
/// in both modes and track the preview through a Photo↔fit-to-view
/// transition.
pub fn calculate_video_bounds(
    container_width: f32,
    container_height: f32,
    frame_width: u32,
    frame_height: u32,
    cover_blend: f32,
    insets: BarInsets,
    rotation: SensorRotation,
) -> (f32, f32, f32, f32) {
    let (frame_width, frame_height) = if rotation.swaps_dimensions() {
        (frame_height, frame_width)
    } else {
        (frame_width, frame_height)
    };
    let frame_aspect = if frame_height > 0 {
        frame_width as f32 / frame_height as f32
    } else {
        1.0
    };

    // Cover endpoint: scale the frame so it fills the container in one
    // axis; the other axis overflows and is cropped off-screen. Returning
    // the source-frame rect (not the container rect) lets QR detections in
    // normalized frame coords render with the sensor's aspect instead of
    // being stretched to the container's aspect.
    let cover = if container_width > 0.0 && container_height > 0.0 {
        let container_aspect = container_width / container_height;
        let (vw, vh) = if frame_aspect > container_aspect {
            (container_height * frame_aspect, container_height)
        } else {
            (container_width, container_width / frame_aspect)
        };
        (
            (container_width - vw) / 2.0,
            (container_height - vh) / 2.0,
            vw,
            vh,
        )
    } else {
        (0.0, 0.0, container_width, container_height)
    };

    // Contain endpoint: letterbox the sensor aspect inside the content area
    // between the UI bars.
    let content_x = insets.left;
    let content_y = insets.top;
    let content_h = (container_height - insets.top - insets.bottom).max(0.0);
    let content_w = (container_width - insets.left - insets.right).max(0.0);
    let contain = if content_h > 0.0 && content_w > 0.0 {
        let content_aspect = content_w / content_h;
        let (vw, vh) = if frame_aspect > content_aspect {
            (content_w, content_w / frame_aspect)
        } else {
            (content_h * frame_aspect, content_h)
        };
        (
            content_x + (content_w - vw) / 2.0,
            content_y + (content_h - vh) / 2.0,
            vw,
            vh,
        )
    } else {
        cover
    };

    let t = cover_blend.clamp(0.0, 1.0);
    (
        contain.0 + (cover.0 - contain.0) * t,
        contain.1 + (cover.1 - contain.1) * t,
        contain.2 + (cover.2 - contain.2) * t,
        contain.3 + (cover.3 - contain.3) * t,
    )
}

fn transform_region(
    bounds: &FrameRegion,
    rotation: SensorRotation,
    mirrored: bool,
) -> (f32, f32, f32, f32) {
    let (mut x, y, width, height) = match rotation {
        SensorRotation::None => (bounds.x, bounds.y, bounds.width, bounds.height),
        SensorRotation::Rotate90 => (
            bounds.y,
            1.0 - bounds.x - bounds.width,
            bounds.height,
            bounds.width,
        ),
        SensorRotation::Rotate180 => (
            1.0 - bounds.x - bounds.width,
            1.0 - bounds.y - bounds.height,
            bounds.width,
            bounds.height,
        ),
        SensorRotation::Rotate270 => (
            1.0 - bounds.y - bounds.height,
            bounds.x,
            bounds.height,
            bounds.width,
        ),
    };
    if mirrored {
        x = 1.0 - x - width;
    }
    (x, y, width, height)
}

/// Transform normalized QR detection coordinates to screen coordinates
pub fn transform_detection_to_screen(
    detection: &QrDetection,
    offset_x: f32,
    offset_y: f32,
    video_width: f32,
    video_height: f32,
    rotation: SensorRotation,
    mirrored: bool,
) -> (f32, f32, f32, f32) {
    let (nx, ny, nw, nh) = transform_region(&detection.bounds, rotation, mirrored);

    // Scale from normalized (0-1) to video pixel coordinates
    let x = nx * video_width;
    let y = ny * video_height;
    let width = nw * video_width;
    let height = nh * video_height;

    // Add video offset (for letterboxing)
    let mut screen_x = x + offset_x;
    let mut screen_y = y + offset_y;

    // Ensure minimum size — when we grow the box to satisfy the lower bound,
    // shift the origin so the expanded box stays centred on the original
    // detection. Otherwise tiny detections drift toward the bottom-right.
    let screen_width = width.max(MIN_OVERLAY_SIZE);
    let screen_height = height.max(MIN_OVERLAY_SIZE);
    screen_x -= (screen_width - width) * 0.5;
    screen_y -= (screen_height - height) * 0.5;

    (screen_x, screen_y, screen_width, screen_height)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::preview_geometry::BarInsets;
    use crate::backends::camera::types::SensorRotation;

    #[test]
    fn contain_bounds_reserve_side_bars() {
        let (x, y, width, height) = calculate_video_bounds(
            780.0,
            313.0,
            1280,
            960,
            0.0,
            BarInsets {
                left: 47.0,
                right: 174.0,
                ..BarInsets::default()
            },
            SensorRotation::None,
        );
        assert!(x >= 47.0 && x + width <= 606.01);
        assert!(y >= 0.0 && y + height <= 313.01);
    }

    #[test]
    fn rotated_region_maps_all_four_corners() {
        let region = crate::app::frame_processor::FrameRegion {
            x: 0.1,
            y: 0.2,
            width: 0.3,
            height: 0.4,
        };
        let cw = transform_region(&region, SensorRotation::Rotate90, false);
        let ccw = transform_region(&region, SensorRotation::Rotate270, false);
        assert!((cw.0 - 0.2).abs() < 1e-6 && (cw.1 - 0.6).abs() < 1e-6);
        assert!((cw.2 - 0.4).abs() < 1e-6 && (cw.3 - 0.3).abs() < 1e-6);
        assert!((ccw.0 - 0.4).abs() < 1e-6 && (ccw.1 - 0.1).abs() < 1e-6);
        assert!((ccw.2 - 0.4).abs() < 1e-6 && (ccw.3 - 0.3).abs() < 1e-6);
    }
}
