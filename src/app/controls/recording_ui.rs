// SPDX-License-Identifier: GPL-3.0-only

//! Recording and streaming UI components (indicator and timer)

use crate::app::bar_layout::Quarter;
use crate::app::overlay_style::OVERLAY_CONTAINER;
use crate::app::state::{AppModel, CameraMode, FileSource, Message};
use crate::fl;
use cosmic::Element;
use cosmic::iced::{Alignment, Background, Color, Length, Pixels, Point, Rectangle, Vector};
use cosmic::widget;
use cosmic::widget::canvas::{self, Frame, Text as CanvasText};

const RECORDING_INDICATOR_WIDTH: f32 = 58.0;
const RECORDING_INDICATOR_HEIGHT: f32 = 16.0;

fn recording_indicator_size(label: &str, quarter: Quarter) -> (f32, f32) {
    let extra_characters = label.chars().count().saturating_sub(5) as f32;
    let content_width = RECORDING_INDICATOR_WIDTH + extra_characters * 8.0;
    if quarter == Quarter::None {
        (content_width, RECORDING_INDICATOR_HEIGHT)
    } else {
        (RECORDING_INDICATOR_HEIGHT, content_width)
    }
}

#[derive(Debug, Clone)]
struct RecordingIndicatorProgram {
    label: String,
    quarter: Quarter,
}

impl canvas::Program<Message, cosmic::Theme> for RecordingIndicatorProgram {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &cosmic::Renderer,
        theme: &cosmic::Theme,
        bounds: Rectangle,
        _cursor: cosmic::iced::mouse::Cursor,
    ) -> Vec<canvas::Geometry<cosmic::Renderer>> {
        let mut frame = Frame::new(renderer, bounds.size());
        match self.quarter {
            Quarter::None => {}
            Quarter::Ccw90 => {
                frame.translate(Vector::new(0.0, bounds.height));
                frame.rotate(self.quarter.radians());
            }
            Quarter::Cw90 => {
                frame.translate(Vector::new(bounds.width, 0.0));
                frame.rotate(self.quarter.radians());
            }
        }

        frame.fill(
            &canvas::Path::circle(Point::new(6.0, RECORDING_INDICATOR_HEIGHT / 2.0), 6.0),
            Color::from_rgb(1.0, 0.0, 0.0),
        );
        let mut text = CanvasText::from(self.label.clone());
        text.position = Point::new(18.0, RECORDING_INDICATOR_HEIGHT / 2.0);
        text.size = Pixels(14.0);
        text.color = theme.cosmic().on_bg_color().into();
        text.align_y = cosmic::iced::alignment::Vertical::Center;
        frame.fill_text(text);

        vec![frame.into_geometry()]
    }
}

/// Create a colored indicator dot (12x12 circle)
fn indicator_dot<'a>(color: Color) -> Element<'a, Message> {
    widget::container(
        widget::Space::new()
            .width(Length::Fixed(12.0))
            .height(Length::Fixed(12.0)),
    )
    .style(move |_theme| widget::container::Style {
        background: Some(Background::Color(color)),
        border: cosmic::iced::Border {
            radius: [6.0; 4].into(),
            ..Default::default()
        },
        ..Default::default()
    })
    .into()
}

/// Format duration as MM:SS
fn format_duration(seconds: u64) -> String {
    format!("{:02}:{:02}", seconds / 60, seconds % 60)
}

/// Placeholder elapsed time shown by the recording indicator under
/// `--preview-spoof-recording`, so the screenshot reads a believable duration
/// (00:07) deterministically rather than whatever the live timer happened to be.
const PREVIEW_SPOOF_RECORDING_SECS: u64 = 7;

impl AppModel {
    fn recording_indicator_quarter(&self) -> Quarter {
        self.controls_bar_layout().quarter
    }

    /// Wrap a status-indicator `dot` + `label` in the standard frosted overlay
    /// pill shared by the recording, streaming and timelapse indicators.
    ///
    /// Portrait lays the dot and label out side by side (`Row`), byte-identical
    /// to before. Held sideways the pill sits in the ~47px-wide top strip, where
    /// a horizontal dot+text pill overflows, so the two stack into a `Column`
    /// (dot over text). The pill itself stays upright in both orientations - only
    /// its internal axis flips.
    fn indicator_pill<'a>(
        &self,
        dot: Element<'a, Message>,
        label: Element<'a, Message>,
    ) -> Element<'a, Message> {
        let spacing = cosmic::theme::spacing();
        let content: Element<'a, Message> = if self.controls_are_sideways() {
            widget::Column::new()
                .push(dot)
                .push(label)
                .align_x(Alignment::Center)
                .spacing(spacing.space_xxs)
                .into()
        } else {
            widget::Row::new()
                .push(dot)
                .push(label)
                .align_y(Alignment::Center)
                .spacing(spacing.space_xxs)
                .into()
        };
        self.frosted_panel(
            widget::container(content).padding([4, 8]).into(),
            OVERLAY_CONTAINER,
        )
    }

    /// Check if we have a video file source in Virtual mode
    fn has_video_file_source(&self) -> bool {
        matches!(
            (&self.mode, &self.virtual_camera_file_source),
            (CameraMode::Virtual, Some(FileSource::Video(_)))
        )
    }

    /// Build the recording indicator and timer widget
    ///
    /// Shows a red dot and elapsed time when recording is active.
    /// Returns None when not recording.
    pub fn build_recording_indicator<'a>(&self) -> Option<Element<'a, Message>> {
        if !self.recording.is_recording() {
            return None;
        }

        let elapsed = if self.preview_spoof_recording {
            PREVIEW_SPOOF_RECORDING_SECS
        } else {
            self.recording.elapsed_duration()
        };
        let duration_text = format_duration(elapsed);

        let quarter = self.recording_indicator_quarter();
        let (width, height) = recording_indicator_size(&duration_text, quarter);
        let indicator = canvas::Canvas::new(RecordingIndicatorProgram {
            label: duration_text,
            quarter,
        })
        .width(Length::Fixed(width))
        .height(Length::Fixed(height));

        Some(self.frosted_panel(
            widget::container(indicator).padding([4, 8]).into(),
            OVERLAY_CONTAINER,
        ))
    }

    /// Build the virtual camera streaming indicator widget
    ///
    /// Shows a green dot and "LIVE" label when streaming is active.
    /// Returns None when not streaming.
    pub fn build_streaming_indicator<'a>(&self) -> Option<Element<'a, Message>> {
        if !self.virtual_camera.is_streaming() {
            return None;
        }

        Some(self.indicator_pill(
            indicator_dot(Color::from_rgb(0.1, 0.7, 0.2)),
            widget::text(fl!("streaming-live")).size(14).into(),
        ))
    }

    /// Build the timelapse indicator widget
    ///
    /// Shows an orange dot, shot count, and elapsed time when timelapse is active.
    /// Shows "Assembling..." when building the video.
    /// Returns None when timelapse is idle.
    pub fn build_timelapse_indicator<'a>(&self) -> Option<Element<'a, Message>> {
        if !self.timelapse.is_active() {
            return None;
        }

        let label = if self.timelapse.is_finalising() {
            fl!("timelapse-saving")
        } else {
            let taken = self.timelapse.shots_taken();
            let elapsed = format_duration(self.timelapse.elapsed_duration());
            format!("{taken} shots - {elapsed}")
        };

        let theme = cosmic::theme::active();
        let destructive: Color = theme.cosmic().destructive_color().into();

        Some(self.indicator_pill(
            indicator_dot(destructive),
            widget::text(label).size(14).into(),
        ))
    }

    /// Build a full-width video progress bar for video file streaming
    ///
    /// Shows a slider-style progress bar with current time and duration labels,
    /// like a video player. Positioned between camera preview and capture button.
    /// Returns None when not in Virtual mode with a video file selected.
    pub fn build_video_progress_bar<'a>(&self) -> Option<Element<'a, Message>> {
        if !self.has_video_file_source() {
            return None;
        }

        let (position, duration) = self
            .video_file_progress
            .map(|(pos, dur, _)| (pos, dur))
            .unwrap_or((0.0, 0.0));

        let spacing = cosmic::theme::spacing();
        let slider_max = if duration > 0.0 { duration } else { 1.0 };

        let progress_row = widget::Row::new()
            .push(widget::text(format_duration(position as u64)).size(12))
            .push(widget::space::horizontal().width(spacing.space_xs))
            .push(
                // 0.05 s step (~20 Hz) so scrubbing feels continuous; iced's
                // default step (1.0) would quantise the slider to whole-second
                // jumps even though the underlying GStreamer seek is accurate.
                widget::slider(0.0..=slider_max, position, Message::VideoFileSeek)
                    .step(0.05_f64)
                    .width(Length::Fill),
            )
            .push(widget::space::horizontal().width(spacing.space_xs))
            .push(widget::text(format_duration(duration as u64)).size(12))
            .align_y(Alignment::Center)
            .padding([spacing.space_xxs, spacing.space_s])
            .width(Length::Fill);

        Some(progress_row.into())
    }

    /// Build a play/pause toggle button for video file sources
    ///
    /// Shows a play or pause icon depending on current state.
    /// Returns None when not in Virtual mode with a video file selected.
    pub fn build_video_play_pause_button<'a>(&self) -> Option<Element<'a, Message>> {
        if !self.has_video_file_source() {
            return None;
        }

        let icon_name = if self.video_file_paused {
            "media-playback-start-symbolic"
        } else {
            "media-playback-pause-symbolic"
        };

        let button = widget::button::icon(widget::icon::from_name(icon_name))
            .on_press(Message::ToggleVideoPlayPause)
            .class(cosmic::theme::Button::Standard);

        Some(button.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::bar_layout::Quarter;
    use crate::backends::display_orientation::DisplayOrientation;
    use crate::config::{Config, ControlsPosition};

    fn model(position: ControlsPosition, orientation: DisplayOrientation) -> AppModel {
        AppModel {
            config: Config {
                controls_position: position,
                ..Config::default()
            },
            display_orientation: orientation,
            ..AppModel::default()
        }
    }

    #[test]
    fn recording_indicator_follows_the_effective_side_layout() {
        assert_eq!(
            model(ControlsPosition::Bottom, DisplayOrientation::Rotate0)
                .recording_indicator_quarter(),
            Quarter::None
        );
        assert_eq!(
            model(ControlsPosition::Bottom, DisplayOrientation::Rotate270)
                .recording_indicator_quarter(),
            Quarter::Ccw90
        );
        assert_eq!(
            model(ControlsPosition::Left, DisplayOrientation::Rotate0)
                .recording_indicator_quarter(),
            Quarter::Cw90
        );
        assert_eq!(
            model(ControlsPosition::Right, DisplayOrientation::Rotate0)
                .recording_indicator_quarter(),
            Quarter::Ccw90
        );
    }

    #[test]
    fn recording_indicator_width_grows_for_hour_scale_durations() {
        assert_eq!(recording_indicator_size("00:00", Quarter::None).0, 58.0);
        assert!(recording_indicator_size("100:00", Quarter::None).0 > 58.0);
        assert!(recording_indicator_size("100:00", Quarter::Cw90).1 > 58.0);
    }
}
