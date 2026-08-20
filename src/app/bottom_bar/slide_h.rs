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

    /// Compute the expanded viewport rectangle that covers the translated position.
    fn expanded_viewport(&self, bounds: Rectangle) -> Rectangle {
        let offset = f32::from_bits(self.slide_shared.load(std::sync::atomic::Ordering::Relaxed))
            * self.sign;
        if self.vertical && offset < 0.0 {
            Rectangle {
                y: bounds.y + offset,
                height: bounds.height - offset,
                ..bounds
            }
        } else if self.vertical {
            Rectangle {
                height: bounds.height + offset,
                ..bounds
            }
        } else if offset < 0.0 {
            Rectangle {
                x: bounds.x + offset,
                width: bounds.width - offset,
                ..bounds
            }
        } else {
            Rectangle {
                width: bounds.width + offset,
                ..bounds
            }
        }
    }

    fn translated_event(&self, event: &Event, offset: f32) -> Event {
        let translate = |position: cosmic::iced::Point| {
            if self.vertical {
                cosmic::iced::Point::new(position.x, position.y - offset)
            } else {
                cosmic::iced::Point::new(position.x - offset, position.y)
            }
        };
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
        let offset = f32::from_bits(self.slide_shared.load(std::sync::atomic::Ordering::Relaxed))
            * self.sign;
        // Adjust cursor to match the visual translation so the child's
        // hover detection aligns with where the button is drawn.
        let adjusted_cursor = cursor.position().map_or(cursor, |pos| {
            mouse::Cursor::Available(if self.vertical {
                cosmic::iced::Point::new(pos.x, pos.y - offset)
            } else {
                cosmic::iced::Point::new(pos.x - offset, pos.y)
            })
        });
        // Use with_layer() to expand the clipping region to cover the
        // translated position. Parent containers clip viewport to their
        // bounds, so with_translation alone can't render outside them.
        let expanded = self.expanded_viewport(layout.bounds());
        renderer.with_layer(expanded, |renderer| {
            let translation = if self.vertical {
                Vector::new(0.0, offset)
            } else {
                Vector::new(offset, 0.0)
            };
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
        // Offset cursor to match the visual translation so the child's
        // hit-testing aligns with where the button is actually drawn.
        let offset = f32::from_bits(self.slide_shared.load(std::sync::atomic::Ordering::Relaxed))
            * self.sign;
        let adjusted_cursor = cursor.position().map_or(cursor, |pos| {
            mouse::Cursor::Available(if self.vertical {
                cosmic::iced::Point::new(pos.x, pos.y - offset)
            } else {
                cosmic::iced::Point::new(pos.x - offset, pos.y)
            })
        });
        let expanded = self.expanded_viewport(layout.bounds());
        let translated_event = self.translated_event(event, offset);
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
        let offset = f32::from_bits(self.slide_shared.load(std::sync::atomic::Ordering::Relaxed))
            * self.sign;
        let adjusted_cursor = cursor.position().map_or(cursor, |pos| {
            mouse::Cursor::Available(if self.vertical {
                cosmic::iced::Point::new(pos.x, pos.y - offset)
            } else {
                cosmic::iced::Point::new(pos.x - offset, pos.y)
            })
        });
        let expanded = self.expanded_viewport(layout.bounds());
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
/// The recording-state capture row mirrors the bottom bar's three-column shape,
/// so its center container occupies exactly the bounds the mode carousel would.
/// During recording the carousel isn't in the widget tree, so nothing writes
/// the shared slide atomic; wrapping the center container keeps the side
/// `SlideH` buttons (the photo-during-recording button) at the correct offset
/// instead of falling back to zero — a mismatch visible in preview screenshots
/// that boot straight into an active recording. Layout and drawing are
/// unchanged; only the atomic is updated as a side effect.
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
