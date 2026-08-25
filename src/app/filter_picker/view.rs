// SPDX-License-Identifier: GPL-3.0-only

//! Filter picker UI view
//!
//! Grid-style filter selector using COSMIC context drawer with live camera preview thumbnails.
//! Uses responsive sizing with 3 columns that adapt to drawer width while maintaining
//! square aspect ratio for each preview.

use super::square_container::square_container;
use crate::app::state::{AppModel, ContextPage, FilterType, Message};
use crate::app::video_widget::{self, VideoContentFit};
use crate::fl;
use cosmic::Element;
use cosmic::app::context_drawer;
use cosmic::iced::{Alignment, Background, Border, Color, Length};
use cosmic::widget::{self, button};
use std::sync::Arc;

/// Spacing between filter thumbnails in grid
const FILTER_GRID_SPACING: f32 = 8.0;
/// Number of columns in the filter grid
const FILTER_GRID_COLUMNS: usize = 3;
/// Vertical spacing between thumbnail and label
const LABEL_SPACING: f32 = 4.0;

impl AppModel {
    fn filter_thumbnail_rotation(&self) -> u32 {
        self.preview_adjusted_rotation(self.current_frame_rotation, self.should_mirror_preview())
            .gpu_rotation_code()
    }

    /// Build the filter picker as a COSMIC context drawer
    ///
    /// Shows a grid of filter options with live camera preview thumbnails
    /// and filter names below each thumbnail. The grid is responsive and
    /// adapts to the drawer width while maintaining square thumbnails.
    pub fn filters_view(&self) -> context_drawer::ContextDrawer<'_, Message> {
        // Define available filters
        let filters: Vec<FilterType> = vec![
            FilterType::Standard,
            FilterType::ChromaticAberration,
            FilterType::Vivid,
            FilterType::Warm,
            FilterType::Cool,
            FilterType::Mono,
            FilterType::Noir,
            FilterType::Sepia,
            FilterType::Fade,
            FilterType::Duotone,
            FilterType::Vignette,
            FilterType::Negative,
            FilterType::Posterize,
            FilterType::Solarize,
            FilterType::Pencil,
        ];

        // Build filter grid with responsive sizing
        let spacing = FILTER_GRID_SPACING as u16;
        let mut grid_column = widget::Column::new().spacing(spacing);
        let mut current_row = widget::Row::new().spacing(spacing);
        let mut items_in_row = 0;

        // Get corner radius from theme for consistent styling
        let theme = cosmic::theme::active();
        let corner_radius = theme.cosmic().corner_radii.radius_s[0];

        for filter_type in filters {
            let is_selected = self.selected_filter == filter_type;

            // Create preview thumbnail with camera frame and filter applied
            let thumbnail: Element<'_, Message> = if let Some(frame) = &self.current_frame {
                // Use video widget with the specific filter type
                // The video widget fills its container and handles aspect ratio via Cover mode
                // Match the live preview: sensor mounting plus display transform.
                let rotation = self.filter_thumbnail_rotation();

                // `Arc::clone` of the preview's own `current_frame`, and that is
                // load-bearing: the pipeline maps VIDEO_ID_FILTER_PREVIEW onto
                // the preview's source texture and dedups by frame-data pointer
                // (see `video_primitive::source_texture_id`). A swatch built from
                // any other frame would race the preview for that texture.
                video_widget::video_widget(
                    Arc::clone(frame),
                    video_widget::VideoWidgetConfig {
                        // One id for all fifteen: the filter lives in the
                        // per-`(video_id, filter_mode)` binding, not the texture.
                        video_id: crate::app::video_primitive::VIDEO_ID_FILTER_PREVIEW,
                        content_fit: VideoContentFit::Cover,
                        filter_type,
                        corner_radius,
                        mirror_horizontal: self.should_mirror_preview(),
                        rotation,
                        crop_uv: None,   // No aspect ratio cropping in filter previews
                        zoom_level: 1.0, // No zoom for filter previews
                        scroll_zoom_enabled: false, // No scroll zoom for filter previews
                        cover_blend: None,
                        insets: crate::app::preview_geometry::BarInsets::default(),
                        // Filter previews don't use blur, so this is only here
                        // to satisfy the struct — value is ignored downstream.
                        letterbox_color: [0.0, 0.0, 0.0, 1.0],
                    },
                )
            } else {
                // Fallback: colored placeholder when no camera frame
                let color = Self::filter_placeholder_color(filter_type);
                widget::container(
                    widget::Space::new()
                        .width(Length::Fill)
                        .height(Length::Fill),
                )
                .style(move |theme: &cosmic::Theme| widget::container::Style {
                    background: Some(Background::Color(color)),
                    border: Border {
                        radius: theme.cosmic().corner_radii.radius_s.into(),
                        ..Default::default()
                    },
                    ..Default::default()
                })
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
            };

            // Wrap thumbnail in square container to enforce 1:1 aspect ratio
            let square_thumbnail = square_container(thumbnail);

            // Use custom_image_button which provides built-in selection indicator
            // (checkmark at bottom-left with accent styling on hover/selected)
            let thumbnail_button = button::custom_image_button(square_thumbnail, None)
                .on_press(Message::SelectFilter(filter_type))
                .padding(0)
                .selected(is_selected)
                .class(button::ButtonClass::Image);

            // Filter name label below thumbnail (outside button, no hover effect)
            let name_label = widget::text(Self::filter_display_name(filter_type))
                .width(Length::Fill)
                .align_x(cosmic::iced::alignment::Horizontal::Center);

            // Column with button (thumbnail only) and name below
            let filter_item = widget::Column::new()
                .push(thumbnail_button)
                .push(widget::space::vertical().height(Length::Fixed(LABEL_SPACING)))
                .push(name_label)
                .align_x(Alignment::Center);

            // Wrap in container with FillPortion(1) for equal width distribution
            let item_container = widget::container(filter_item).width(Length::FillPortion(1));

            current_row = current_row.push(item_container);
            items_in_row += 1;

            // Start new row after FILTER_GRID_COLUMNS items
            if items_in_row >= FILTER_GRID_COLUMNS {
                grid_column = grid_column.push(current_row);
                current_row = widget::Row::new().spacing(spacing);
                items_in_row = 0;
            }
        }

        // Push remaining items in last row, padding with empty space for even distribution
        if items_in_row > 0 {
            while items_in_row < FILTER_GRID_COLUMNS {
                current_row = current_row.push(
                    widget::Space::new()
                        .width(Length::FillPortion(1))
                        .height(Length::Shrink),
                );
                items_in_row += 1;
            }
            grid_column = grid_column.push(current_row);
        }

        // Context drawer already provides scrollable behavior, so just wrap in a clipping container
        let content: Element<'_, Message> = widget::container(grid_column)
            .width(Length::Fill)
            .clip(true)
            .into();

        context_drawer::context_drawer(content, Message::ToggleContextPage(ContextPage::Filters))
            .title(fl!("filters-title"))
    }

    /// Get placeholder color for a filter type
    fn filter_placeholder_color(filter_type: FilterType) -> Color {
        match filter_type {
            FilterType::Standard => Color::from_rgb(0.3, 0.3, 0.3),
            FilterType::Mono => Color::from_rgb(0.5, 0.5, 0.5),
            FilterType::Sepia => Color::from_rgb(0.5, 0.4, 0.3),
            FilterType::Noir => Color::from_rgb(0.2, 0.2, 0.2),
            FilterType::Vivid => Color::from_rgb(0.4, 0.5, 0.6),
            FilterType::Cool => Color::from_rgb(0.3, 0.4, 0.5),
            FilterType::Warm => Color::from_rgb(0.5, 0.4, 0.3),
            FilterType::Fade => Color::from_rgb(0.45, 0.45, 0.45),
            FilterType::Duotone => Color::from_rgb(0.3, 0.3, 0.6),
            FilterType::Vignette => Color::from_rgb(0.35, 0.35, 0.35),
            FilterType::Negative => Color::from_rgb(0.85, 0.85, 0.85),
            FilterType::Posterize => Color::from_rgb(0.7, 0.3, 0.5),
            FilterType::Solarize => Color::from_rgb(0.5, 0.6, 0.35),
            FilterType::ChromaticAberration => Color::from_rgb(0.6, 0.4, 0.5),
            FilterType::Pencil => Color::from_rgb(0.9, 0.9, 0.85),
        }
    }

    /// Get display name for a filter type (used in filter picker grid)
    fn filter_display_name(filter_type: FilterType) -> String {
        match filter_type {
            FilterType::Standard => fl!("filter-standard"),
            FilterType::Mono => fl!("filter-mono"),
            FilterType::Sepia => fl!("filter-sepia"),
            FilterType::Noir => fl!("filter-noir"),
            FilterType::Vivid => fl!("filter-vivid"),
            FilterType::Cool => fl!("filter-cold"),
            FilterType::Warm => fl!("filter-warm"),
            FilterType::Fade => fl!("filter-fade"),
            FilterType::Duotone => fl!("filter-duotone"),
            FilterType::Vignette => fl!("filter-vignette"),
            FilterType::Negative => fl!("filter-negative"),
            FilterType::Posterize => fl!("filter-posterize"),
            FilterType::Solarize => fl!("filter-solarize"),
            FilterType::ChromaticAberration => fl!("filter-chroma"),
            FilterType::Pencil => fl!("filter-pencil"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::camera::types::SensorRotation;
    use crate::backends::display_orientation::DisplayOrientation;

    #[test]
    fn filter_thumbnail_uses_the_live_preview_rotation() {
        let mut model = AppModel {
            display_orientation: DisplayOrientation::Rotate270,
            current_frame_rotation: SensorRotation::Rotate270,
            ..AppModel::default()
        };

        model.config.mirror_preview = false;
        assert_eq!(model.filter_thumbnail_rotation(), 0);

        model.config.mirror_preview = true;
        assert_eq!(model.filter_thumbnail_rotation(), 2);
    }
}
