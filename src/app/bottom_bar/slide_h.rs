// SPDX-License-Identifier: GPL-3.0-only

//! A wrapper widget that shifts its child horizontally during draw
//! without changing layout. Reads the offset from a shared atomic so
//! the position updates every frame, even when view() isn't called.

use std::sync::Arc;
use std::sync::atomic::AtomicU32;

use cosmic::iced::advanced::layout;
use cosmic::iced::advanced::renderer;
use cosmic::iced::advanced::widget::tree::Tree;
use cosmic::iced::advanced::{Clipboard, Layout, Shell, Widget};
use cosmic::iced::{Event, Length, Rectangle, Size, Vector, touch};
use iced_core::mouse;

use crate::app::state::Message;
use cosmic::Theme;
type Renderer = cosmic::Renderer;

/// Wrapper that draws its child shifted horizontally. The offset magnitude
/// is read from a shared atomic (f32 bits) during draw(), and the sign
/// is determined by `sign` (+1.0 or -1.0).
pub struct SlideH<'a> {
    child: cosmic::Element<'a, Message>,
    slide_shared: Arc<AtomicU32>,
    sign: f32,
    vertical: bool,
}

impl<'a> SlideH<'a> {
    pub fn new(
        child: cosmic::Element<'a, Message>,
        slide_shared: Arc<AtomicU32>,
        sign: f32,
    ) -> Self {
        Self {
            child,
            slide_shared,
            sign,
            vertical: false,
        }
    }

    pub fn new_vertical(
        child: cosmic::Element<'a, Message>,
        slide_shared: Arc<AtomicU32>,
        sign: f32,
    ) -> Self {
        Self {
            child,
            slide_shared,
            sign,
            vertical: true,
        }
    }

    fn translation(&self) -> Vector {
        let offset = f32::from_bits(self.slide_shared.load(std::sync::atomic::Ordering::Relaxed))
            * self.sign;
        if self.vertical {
            Vector::new(0.0, offset)
        } else {
            Vector::new(offset, 0.0)
        }
    }

    fn expanded_viewport(bounds: Rectangle, translation: Vector) -> Rectangle {
        Rectangle {
            x: bounds.x + translation.x.min(0.0),
            y: bounds.y + translation.y.min(0.0),
            width: bounds.width + translation.x.abs(),
            height: bounds.height + translation.y.abs(),
        }
    }

    fn adjusted_cursor(cursor: mouse::Cursor, translation: Vector) -> mouse::Cursor {
        cursor.position().map_or(cursor, |position| {
            mouse::Cursor::Available(position - translation)
        })
    }

    fn translated_event(event: &Event, translation: Vector) -> Event {
        let translate = |position: cosmic::iced::Point| position - translation;
        match event {
            Event::Touch(touch::Event::FingerPressed { id, position }) => {
                Event::Touch(touch::Event::FingerPressed {
                    id: *id,
                    position: translate(*position),
                })
            }
            Event::Touch(touch::Event::FingerMoved { id, position }) => {
                Event::Touch(touch::Event::FingerMoved {
                    id: *id,
                    position: translate(*position),
                })
            }
            Event::Touch(touch::Event::FingerLifted { id, position }) => {
                Event::Touch(touch::Event::FingerLifted {
                    id: *id,
                    position: translate(*position),
                })
            }
            Event::Touch(touch::Event::FingerLost { id, position }) => {
                Event::Touch(touch::Event::FingerLost {
                    id: *id,
                    position: translate(*position),
                })
            }
            _ => event.clone(),
        }
    }
}

impl<'a> Widget<Message, Theme, Renderer> for SlideH<'a> {
    fn tag(&self) -> cosmic::iced::advanced::widget::tree::Tag {
        self.child.as_widget().tag()
    }

    fn state(&self) -> cosmic::iced::advanced::widget::tree::State {
        self.child.as_widget().state()
    }

    fn children(&self) -> Vec<cosmic::iced::advanced::widget::Tree> {
        self.child.as_widget().children()
    }

    fn diff(&mut self, tree: &mut Tree) {
        self.child.as_widget_mut().diff(tree);
    }

    fn size(&self) -> Size<Length> {
        self.child.as_widget().size()
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        self.child.as_widget_mut().layout(tree, renderer, limits)
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        style: &renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _viewport: &Rectangle,
    ) {
        use cosmic::iced::advanced::Renderer as _;
        let translation = self.translation();
        let adjusted_cursor = Self::adjusted_cursor(cursor, translation);
        // Translation alone cannot escape the parent's clipping region.
        let expanded = Self::expanded_viewport(layout.bounds(), translation);
        renderer.with_layer(expanded, |renderer| {
            renderer.with_translation(translation, |renderer| {
                self.child.as_widget().draw(
                    tree,
                    renderer,
                    theme,
                    style,
                    layout,
                    adjusted_cursor,
                    &expanded,
                );
            });
        });
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
        _viewport: &Rectangle,
    ) {
        let translation = self.translation();
        let adjusted_cursor = Self::adjusted_cursor(cursor, translation);
        let expanded = Self::expanded_viewport(layout.bounds(), translation);
        let translated_event = Self::translated_event(event, translation);
        self.child.as_widget_mut().update(
            tree,
            &translated_event,
            layout,
            adjusted_cursor,
            renderer,
            clipboard,
            shell,
            &expanded,
        );
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        let translation = self.translation();
        let adjusted_cursor = Self::adjusted_cursor(cursor, translation);
        let expanded = Self::expanded_viewport(layout.bounds(), translation);
        self.child
            .as_widget()
            .mouse_interaction(tree, layout, adjusted_cursor, &expanded, renderer)
    }
}

impl<'a> From<SlideH<'a>> for cosmic::Element<'a, Message> {
    fn from(slide: SlideH<'a>) -> Self {
        cosmic::Element::new(slide)
    }
}

/// Transparent wrapper that publishes the carousel's resting inward-slide into
/// the shared atomic at draw time, derived from its child's on-screen bounds.
///
/// The capture row mirrors the bottom bar's three-column shape,
/// so its center container occupies exactly the bounds the mode carousel would.
/// While the carousel is visible, it uses a separate atomic so Toolbox stays
/// at its resting offset instead of following the live carousel's expansion.
/// During recording the carousel is absent, so the shared offset can be primed
/// directly. Layout and drawing are unchanged; only the atomic is updated as a
/// side effect.
pub struct SlidePrimer<'a> {
    child: cosmic::Element<'a, Message>,
    slide_shared: Arc<AtomicU32>,
}

impl<'a> SlidePrimer<'a> {
    pub fn new(child: cosmic::Element<'a, Message>, slide_shared: Arc<AtomicU32>) -> Self {
        Self {
            child,
            slide_shared,
        }
    }
}

impl<'a> Widget<Message, Theme, Renderer> for SlidePrimer<'a> {
    fn tag(&self) -> cosmic::iced::advanced::widget::tree::Tag {
        self.child.as_widget().tag()
    }

    fn state(&self) -> cosmic::iced::advanced::widget::tree::State {
        self.child.as_widget().state()
    }

    fn children(&self) -> Vec<cosmic::iced::advanced::widget::Tree> {
        self.child.as_widget().children()
    }

    fn diff(&mut self, tree: &mut Tree) {
        self.child.as_widget_mut().diff(tree);
    }

    fn size(&self) -> Size<Length> {
        self.child.as_widget().size()
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        self.child.as_widget_mut().layout(tree, renderer, limits)
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
        // Publish the resting slide computed from the center container's bounds
        // (== the carousel's bounds) so the sibling SlideH buttons in this row
        // read the correct offset. The row draws children left-to-right, so this
        // runs before the photo button's SlideH in the same frame.
        let slide = crate::app::bottom_bar::mode_carousel::resting_button_slide(layout.bounds());
        self.slide_shared
            .store(slide.to_bits(), std::sync::atomic::Ordering::Relaxed);
        self.child
            .as_widget()
            .draw(tree, renderer, theme, style, layout, cursor, viewport);
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
        self.child.as_widget_mut().update(
            tree, event, layout, cursor, renderer, clipboard, shell, viewport,
        );
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        self.child
            .as_widget()
            .mouse_interaction(tree, layout, cursor, viewport, renderer)
    }
}

impl<'a> From<SlidePrimer<'a>> for cosmic::Element<'a, Message> {
    fn from(primer: SlidePrimer<'a>) -> Self {
        cosmic::Element::new(primer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::settings::widget_test_support as ui;
    use cosmic::iced::Point;

    #[test]
    fn slide_translation_keeps_real_button_hit_testing_and_clipping_on_both_axes() {
        let renderer = ui::renderer();
        for vertical in [false, true] {
            for sign in [-1.0, 1.0] {
                let shared = Arc::new(AtomicU32::new(40.0f32.to_bits()));
                let child = cosmic::widget::button::text("x")
                    .width(20)
                    .height(20)
                    .padding(0)
                    .on_press(Message::Noop)
                    .into();
                let slide = if vertical {
                    SlideH::new_vertical(child, shared.clone(), sign)
                } else {
                    SlideH::new(child, shared.clone(), sign)
                };
                let translation = slide.translation();
                let mut element: cosmic::Element<'_, Message> = slide.into();
                let (mut tree, node) = ui::layout(&mut element, &renderer, Size::new(20.0, 20.0));
                let node = node.move_to(Point::new(100.0, 100.0));
                let target = node.bounds().center() + translation;
                assert!(!node.bounds().contains(target));
                assert!(SlideH::expanded_viewport(node.bounds(), translation).contains(target));
                let messages = ui::click(&mut element, &mut tree, &node, &renderer, target);
                assert!(
                    matches!(messages.as_slice(), [Message::Noop]),
                    "{messages:?}"
                );
                assert_eq!(
                    element.as_widget().mouse_interaction(
                        &tree,
                        Layout::new(&node),
                        mouse::Cursor::Available(target),
                        &node.bounds(),
                        &renderer
                    ),
                    mouse::Interaction::Pointer
                );
                for cursor in [
                    mouse::Cursor::Unavailable,
                    mouse::Cursor::Levitating(target),
                ] {
                    assert_eq!(SlideH::adjusted_cursor(cursor, translation), cursor);
                }
                assert_eq!(
                    SlideH::adjusted_cursor(mouse::Cursor::Available(target), translation)
                        .position(),
                    Some(node.bounds().center())
                );
            }
        }
    }

    #[test]
    fn slide_inverse_maps_every_touch_phase_without_changing_finger_or_phase() {
        let point = Point::new(12.0, 34.0);
        let phases: [fn(Point) -> touch::Event; 4] = [
            |position| touch::Event::FingerPressed {
                id: touch::Finger(9),
                position,
            },
            |position| touch::Event::FingerMoved {
                id: touch::Finger(9),
                position,
            },
            |position| touch::Event::FingerLifted {
                id: touch::Finger(9),
                position,
            },
            |position| touch::Event::FingerLost {
                id: touch::Finger(9),
                position,
            },
        ];
        for translation in [
            Vector::new(-40.0, 0.0),
            Vector::new(40.0, 0.0),
            Vector::new(0.0, -40.0),
            Vector::new(0.0, 40.0),
        ] {
            for phase in phases {
                let Event::Touch(mapped) =
                    SlideH::translated_event(&Event::Touch(phase(point)), translation)
                else {
                    panic!("touch event changed kind");
                };
                assert_eq!(mapped, phase(point - translation));
            }
        }
    }
}
