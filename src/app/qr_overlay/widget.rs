// SPDX-License-Identifier: GPL-3.0-only

//! Custom QR overlay widget
//!
//! This widget renders QR code detection overlays directly using the renderer,
//! calculating positions at render time when actual layout bounds are known.
//! Buttons are rendered as actual cosmic widgets for proper theming.

use super::{
    BUTTON_GAP, OVERLAY_BORDER_RADIUS, OVERLAY_BORDER_WIDTH, calculate_video_bounds,
    get_action_color, transform_detection_to_screen,
};
use crate::app::frame_processor::QrDetection;
use crate::app::preview_geometry::BarInsets;
use crate::app::state::Message;
use crate::backends::camera::types::SensorRotation;
use cosmic::iced::advanced::widget::{Operation, Tree};
use cosmic::iced::advanced::{Clipboard, Layout, Shell, Widget, layout, mouse, renderer};

use cosmic::iced::{Border, Color, Element, Event, Length, Point, Rectangle, Size};
use cosmic::widget;
use cosmic::{Renderer, Theme};

/// Custom widget for rendering QR code detection overlays
pub struct QrOverlayWidget<'a> {
    detections: Vec<QrDetection>,
    frame_width: u32,
    frame_height: u32,
    /// Animated Cover↔Contain blend (1.0 = Cover, 0.0 = Contain). The
    /// visible video bounds lerp between the two endpoints by this value
    /// so the QR boxes track the preview through a fit/fill transition.
    cover_blend: f32,
    /// Animated four-sided UI insets and effective preview rotation.
    insets: BarInsets,
    rotation: SensorRotation,
    mirrored: bool,
    residual_zoom: f32,
    /// Child button elements (one per detection)
    buttons: Vec<Element<'a, Message, Theme, Renderer>>,
}

impl<'a> QrOverlayWidget<'a> {
    /// Create a new QR overlay widget
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mut detections: Vec<QrDetection>,
        frame_width: u32,
        frame_height: u32,
        cover_blend: f32,
        insets: BarInsets,
        rotation: SensorRotation,
        mirrored: bool,
        residual_zoom: f32,
    ) -> Self {
        let residual_zoom = if residual_zoom.is_finite() {
            residual_zoom.max(1.0)
        } else {
            1.0
        };
        // Filter on the actual QR region, before minimum-size decoration can
        // grow an invisible detection back into a clickable on-screen box.
        if residual_zoom > 1.0 {
            let edge = 0.5 - 0.5 / residual_zoom;
            detections.retain(|detection| {
                let b = &detection.bounds;
                b.x < 1.0 - edge
                    && b.y < 1.0 - edge
                    && b.x + b.width > edge
                    && b.y + b.height > edge
            });
        }
        // Create button elements for each detection
        let buttons: Vec<Element<'a, Message, Theme, Renderer>> = detections
            .iter()
            .map(|detection| {
                let label = detection.action.action_label();
                let message = super::action_button::action_to_message(&detection.action);

                widget::button::suggested(label).on_press(message).into()
            })
            .collect();

        Self {
            detections,
            frame_width,
            frame_height,
            cover_blend,
            insets,
            rotation,
            mirrored,
            residual_zoom,
            buttons,
        }
    }
}

impl<'a> Widget<Message, Theme, Renderer> for QrOverlayWidget<'a> {
    fn size(&self) -> Size<Length> {
        Size::new(Length::Fill, Length::Fill)
    }

    fn children(&self) -> Vec<Tree> {
        self.buttons.iter().map(Tree::new).collect()
    }

    fn diff(&mut self, tree: &mut Tree) {
        tree.diff_children(&mut self.buttons);
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let size = limits.max();

        // Calculate video bounds
        let (offset_x, offset_y, video_width, video_height) = calculate_video_bounds(
            size.width,
            size.height,
            self.frame_width,
            self.frame_height,
            self.cover_blend,
            self.insets,
            self.rotation,
        );

        let rotation = self.rotation;
        let mirrored = self.mirrored;
        let button_nodes: Vec<layout::Node> = self
            .detections
            .iter()
            .zip(self.buttons.iter_mut())
            .zip(tree.children.iter_mut())
            .map(|((detection, button), child_tree)| {
                let (x, y, width, height) = transform_detection_to_screen(
                    detection,
                    offset_x,
                    offset_y,
                    video_width,
                    video_height,
                    rotation,
                    mirrored,
                    self.residual_zoom,
                );
                let center_x = x + width / 2.0;
                let top_y = y + height + BUTTON_GAP;
                // Layout the button with its intrinsic size
                let button_limits = layout::Limits::new(Size::ZERO, Size::new(200.0, 50.0));
                let mut button_node =
                    button
                        .as_widget_mut()
                        .layout(child_tree, renderer, &button_limits);

                // Center the button horizontally at the calculated position
                let button_size = button_node.size();
                let button_x = center_x - button_size.width / 2.0;

                button_node = button_node.move_to(Point::new(button_x, top_y));
                button_node
            })
            .collect();

        layout::Node::with_children(size, button_nodes)
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        style: &renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        use cosmic::iced::advanced::Renderer as _;

        let bounds = layout.bounds();

        // Calculate video bounds within container
        let (offset_x, offset_y, video_width, video_height) = calculate_video_bounds(
            bounds.width,
            bounds.height,
            self.frame_width,
            self.frame_height,
            self.cover_blend,
            self.insets,
            self.rotation,
        );

        // Draw QR detection boxes
        for detection in &self.detections {
            let (x, y, width, height) = transform_detection_to_screen(
                detection,
                offset_x,
                offset_y,
                video_width,
                video_height,
                self.rotation,
                self.mirrored,
                self.residual_zoom,
            );

            let qr_bounds = Rectangle {
                x: bounds.x + x,
                y: bounds.y + y,
                width,
                height,
            };

            // Get color based on action type
            let border_color = get_action_color(&detection.action);

            // Draw semi-transparent background with colored border
            renderer.fill_quad(
                renderer::Quad {
                    bounds: qr_bounds,
                    border: Border {
                        color: border_color,
                        width: OVERLAY_BORDER_WIDTH,
                        radius: OVERLAY_BORDER_RADIUS.into(),
                    },
                    shadow: Default::default(),
                    snap: true,
                },
                Color::from_rgba(0.0, 0.0, 0.0, 0.3),
            );
        }

        // Draw child button widgets
        for ((button, child_tree), child_layout) in self
            .buttons
            .iter()
            .zip(tree.children.iter())
            .zip(layout.children())
        {
            button.as_widget().draw(
                child_tree,
                renderer,
                theme,
                style,
                child_layout,
                cursor,
                viewport,
            );
        }
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        // Forward events to child buttons
        for ((button, child_tree), child_layout) in self
            .buttons
            .iter_mut()
            .zip(tree.children.iter_mut())
            .zip(layout.children())
        {
            button.as_widget_mut().update(
                child_tree,
                event,
                child_layout,
                cursor,
                renderer,
                clipboard,
                shell,
                viewport,
            );
        }
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        // Check child buttons for mouse interaction
        for ((button, child_tree), child_layout) in self
            .buttons
            .iter()
            .zip(tree.children.iter())
            .zip(layout.children())
        {
            let interaction = button.as_widget().mouse_interaction(
                child_tree,
                child_layout,
                cursor,
                viewport,
                renderer,
            );

            if interaction != mouse::Interaction::default() {
                return interaction;
            }
        }

        mouse::Interaction::default()
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        for ((button, child_tree), child_layout) in self
            .buttons
            .iter_mut()
            .zip(tree.children.iter_mut())
            .zip(layout.children())
        {
            button
                .as_widget_mut()
                .operate(child_tree, child_layout, renderer, operation);
        }
    }
}

impl<'a> From<QrOverlayWidget<'a>> for Element<'a, Message, Theme, Renderer> {
    fn from(widget: QrOverlayWidget<'a>) -> Self {
        Element::new(widget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::frame_processor::FrameRegion;
    use crate::app::settings::widget_test_support as ui;

    fn detections() -> Vec<QrDetection> {
        [
            (0.1, 0.1, 0.1, 0.1),
            (0.65, 0.6, 0.1, 0.1),
            (-0.1, 0.8, 0.001, 0.001),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, (x, y, width, height))| {
            QrDetection::new(
                FrameRegion {
                    x,
                    y,
                    width,
                    height,
                },
                format!("text {i}"),
            )
        })
        .collect()
    }

    #[test]
    fn qr_button_layout_keeps_detection_order_origin_and_transformed_minimum_boxes() {
        let renderer = ui::renderer();
        let size = Size::new(640.0, 480.0);
        let origin = Point::new(17.0, 31.0);
        let insets = BarInsets {
            top: 20.0,
            bottom: 40.0,
            left: 13.0,
            right: 27.0,
        };
        for count in [0, 1, 3] {
            for rotation in [
                SensorRotation::None,
                SensorRotation::Rotate90,
                SensorRotation::Rotate180,
                SensorRotation::Rotate270,
            ] {
                for mirrored in [false, true] {
                    for blend in [0.0, 0.5, 1.0] {
                        let detections: Vec<_> = detections().into_iter().take(count).collect();
                        let mut overlay: cosmic::Element<'_, Message> = QrOverlayWidget::new(
                            detections.clone(),
                            640,
                            480,
                            blend,
                            insets,
                            rotation,
                            mirrored,
                            1.0,
                        )
                        .into();
                        let (_, node) = ui::layout(&mut overlay, &renderer, size);
                        let node = node.move_to(origin);
                        assert_eq!(node.children().len(), count);
                        let (ox, oy, vw, vh) =
                            calculate_video_bounds(640.0, 480.0, 640, 480, blend, insets, rotation);
                        for (child, detection) in Layout::new(&node).children().zip(&detections) {
                            let (x, y, width, height) = transform_detection_to_screen(
                                detection, ox, oy, vw, vh, rotation, mirrored, 1.0,
                            );
                            let bounds = child.bounds();
                            assert!(
                                (bounds.center_x() - (origin.x + x + width / 2.0)).abs() < 0.001
                            );
                            assert!(
                                (bounds.y - (origin.y + y + height + BUTTON_GAP)).abs() < 0.001
                            );
                            assert!(bounds.width <= 200.0 && bounds.height <= 50.0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn residual_zoom_moves_buttons_and_removes_off_crop_actions() {
        let renderer = ui::renderer();
        let detections = vec![
            QrDetection::new(
                FrameRegion {
                    x: 0.05,
                    y: 0.4,
                    width: 0.1,
                    height: 0.15,
                },
                "hidden".into(),
            ),
            QrDetection::new(
                FrameRegion {
                    x: 0.4,
                    y: 0.3,
                    width: 0.2,
                    height: 0.25,
                },
                "visible".into(),
            ),
        ];
        let mut overlay: cosmic::Element<'_, Message> = QrOverlayWidget::new(
            detections,
            1000,
            800,
            1.0,
            BarInsets::default(),
            SensorRotation::None,
            false,
            2.0,
        )
        .into();
        let (mut tree, node) = ui::layout(&mut overlay, &renderer, Size::new(1000.0, 800.0));
        assert_eq!(
            node.children().len(),
            1,
            "off-crop QR must not own a clickable button"
        );
        let child = Layout::new(&node).children().next().unwrap().bounds();
        assert!((child.center_x() - 500.0).abs() < 0.001);
        assert!((child.y - 488.0).abs() < 0.001, "{child:?}");
        let messages = ui::click(&mut overlay, &mut tree, &node, &renderer, child.center());
        assert!(matches!(messages.as_slice(), [Message::QrCopyText(text)] if text == "visible"));
    }

    #[test]
    fn completed_native_zoom_is_not_applied_twice_across_rotation_and_mirror() {
        use crate::backends::camera::types::{CameraFrame, FrameData, FrameMetadata, PixelFormat};
        use crate::backends::camera::zoom::{ScalerCropBounds, ZoomRect, residual_zoom};
        let base = ZoomRect {
            x: 100,
            y: 200,
            width: 4000,
            height: 3000,
        };
        let minimum = ZoomRect {
            x: 0,
            y: 0,
            width: 100,
            height: 100,
        };
        let bounds = ScalerCropBounds::new(minimum, base, base, base, false).unwrap();
        let frame = CameraFrame {
            width: 1000,
            height: 1000,
            stride: 4000,
            data: FrameData::from_owned_vec(vec![]),
            format: PixelFormat::RGBA,
            yuv_planes: None,
            captured_at: std::time::Instant::now(),
            sensor_timestamp_ns: None,
            libcamera_metadata: Some(FrameMetadata {
                scaler_zoom: bounds.applied(Some(bounds.crop(2.0))),
                ..Default::default()
            }),
        };
        let renderer = ui::renderer();
        for (rotation, (x, y, w, h)) in [
            (SensorRotation::None, (0.4_f32, 0.3_f32, 0.2_f32, 0.25_f32)),
            (SensorRotation::Rotate90, (0.3, 0.4, 0.25, 0.2)),
            (SensorRotation::Rotate180, (0.4, 0.45, 0.2, 0.25)),
            (SensorRotation::Rotate270, (0.45, 0.4, 0.25, 0.2)),
        ] {
            for mirrored in [false, true] {
                for (desired, residual) in [(2.0, 1.0), (4.0, 2.0)] {
                    assert_eq!(residual_zoom(desired, &frame), residual);
                    let detection = QrDetection::new(
                        FrameRegion {
                            x: 0.4,
                            y: 0.3,
                            width: 0.2,
                            height: 0.25,
                        },
                        "visible".into(),
                    );
                    let mut overlay = super::super::build_qr_overlay(
                        &[detection],
                        1000,
                        1000,
                        1.0,
                        BarInsets::default(),
                        rotation,
                        mirrored,
                        residual_zoom(desired, &frame),
                    );
                    let (_, node) = ui::layout(&mut overlay, &renderer, Size::new(1000.0, 1000.0));
                    let button = node.children()[0].bounds();
                    let x = if mirrored { 1.0 - x - w } else { x };
                    let expected_center = ((x + w / 2.0 - 0.5) * residual + 0.5) * 1000.0;
                    let expected_top = ((y + h - 0.5) * residual + 0.5) * 1000.0 + BUTTON_GAP;
                    assert!(
                        (button.center_x() - expected_center).abs() < 0.001,
                        "{rotation:?} mirror={mirrored} zoom={residual} {button:?}"
                    );
                    assert!(
                        (button.y - expected_top).abs() < 0.001,
                        "{rotation:?} mirror={mirrored} zoom={residual} {button:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn qr_child_buttons_publish_their_original_actions() {
        let renderer = ui::renderer();
        let mut detections = detections();
        detections.truncate(2);
        detections[0] =
            QrDetection::new(detections[0].bounds.clone(), "https://example.com".into());
        let mut overlay: cosmic::Element<'_, Message> = QrOverlayWidget::new(
            detections,
            640,
            480,
            1.0,
            BarInsets::default(),
            SensorRotation::None,
            false,
            1.0,
        )
        .into();
        let (mut tree, node) = ui::layout(&mut overlay, &renderer, Size::new(640.0, 480.0));
        let mut messages = Vec::new();
        for child in Layout::new(&node).children() {
            messages.extend(ui::click(
                &mut overlay,
                &mut tree,
                &node,
                &renderer,
                child.bounds().center(),
            ));
        }
        assert!(
            matches!(messages.as_slice(), [Message::QrOpenUrl(url), Message::QrCopyText(text)]
            if url == "https://example.com" && text == "text 1"),
            "{messages:?}"
        );
    }
}
