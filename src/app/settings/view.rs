// SPDX-License-Identifier: GPL-3.0-only

//! Settings drawer view

use crate::app::state::{AppModel, ContextPage, Message, SettingsPage};
use crate::config::{AppTheme, AudioEncoder, PhotoOutputFormat, TimelapseInterval};
use crate::constants::BitratePreset;
use crate::fl;
use cosmic::Element;
use cosmic::app::context_drawer;
use cosmic::iced::{Alignment, Length};
use cosmic::widget;
use cosmic::widget::icon;

/// Theme-aware disabled text style (greyed out).
fn disabled_text_style(theme: &cosmic::Theme) -> cosmic::iced::widget::text::Style {
    cosmic::iced::widget::text::Style {
        color: Some(cosmic::iced::Color::from(theme.cosmic().button.on_disabled)),
        ..Default::default()
    }
}

/// A drill-down list row: label on the left, chevron on the right.
fn settings_nav_row<'a>(label: String, message: Message) -> widget::list::ListButton<'a, Message> {
    widget::list::button(
        widget::Row::new()
            .push(widget::text::body(label).width(Length::Fill))
            .push(
                widget::icon::from_name("go-next-symbolic")
                    .symbolic(true)
                    .icon()
                    .size(16),
            )
            .align_y(Alignment::Center),
    )
    .on_press(message)
}

/// Create a text label styled as a disabled/greyed-out control value.
fn disabled_text(value: String) -> Element<'static, Message> {
    widget::text::body(value)
        .class(cosmic::theme::style::iced::Text::Custom(
            disabled_text_style,
        ))
        .into()
}

fn controls_position_index(position: crate::config::ControlsPosition) -> usize {
    crate::config::ControlsPosition::ALL
        .iter()
        .position(|candidate| *candidate == position)
        .unwrap_or(0)
}

impl AppModel {
    /// Build the Settings context drawer for the current sub-page.
    ///
    /// The drawer is a drill-down: [`SettingsPage::Root`] shows the theme
    /// picker, a short category menu and the reset button, and every other page
    /// is a focused list reached from it with a back button. Keeping the root
    /// short avoids the single long scroll that used to make About/Insights
    /// inherit its scroll position.
    pub fn settings_view(&self) -> context_drawer::ContextDrawer<'_, Message> {
        match self.settings_page {
            SettingsPage::Root => self.settings_root_view(),
            SettingsPage::Camera => self.settings_subpage(fl!("camera"), self.camera_sections()),
            SettingsPage::Photo => {
                self.settings_subpage(fl!("settings-photo"), self.photo_sections())
            }
            SettingsPage::Video => {
                self.settings_subpage(fl!("settings-video"), self.video_sections())
            }
            SettingsPage::Timelapse => {
                self.settings_subpage(fl!("settings-timelapse"), self.timelapse_sections())
            }
            SettingsPage::Overlay => {
                self.settings_subpage(fl!("settings-overlay"), self.overlay_sections())
            }
            SettingsPage::VirtualCamera => {
                self.settings_subpage(fl!("virtual-camera-title"), self.virtual_camera_sections())
            }
            SettingsPage::BugReports => {
                self.settings_subpage(fl!("settings-bug-reports"), self.bug_reports_sections())
            }
            SettingsPage::About => self.settings_about_view(),
        }
    }

    /// Back button that returns to the top-level Settings category menu.
    ///
    /// Also used by the Insights and Keyboard Shortcuts drawers, which are
    /// reached from that menu.
    pub(crate) fn settings_back_button(&self) -> Element<'_, Message> {
        widget::button::icon(icon::from_name("go-previous-symbolic").symbolic(true))
            .on_press(Message::OpenSettingsPage(SettingsPage::Root))
            .into()
    }

    /// Top-level Settings menu: the theme picker, a list of camera categories
    /// that drill into focused sub-pages, and links to Insights, Shortcuts,
    /// Bug reports, and About.
    fn settings_root_view(&self) -> context_drawer::ContextDrawer<'_, Message> {
        // Theme index (System = 0, Dark = 1, Light = 2)
        let current_theme_index = match self.config.app_theme {
            AppTheme::System => 0,
            AppTheme::Dark => 1,
            AppTheme::Light => 2,
        };

        let appearance = widget::settings::section()
            .title(fl!("settings-appearance"))
            .add(
                widget::settings::item::builder(fl!("settings-theme")).control(widget::dropdown(
                    &self.theme_dropdown_options,
                    Some(current_theme_index),
                    Message::SetAppTheme,
                )),
            );

        let categories = widget::settings::section()
            .title(fl!("camera"))
            .add(settings_nav_row(
                fl!("camera"),
                Message::OpenSettingsPage(SettingsPage::Camera),
            ))
            .add(settings_nav_row(
                fl!("settings-photo"),
                Message::OpenSettingsPage(SettingsPage::Photo),
            ))
            .add(settings_nav_row(
                fl!("settings-video"),
                Message::OpenSettingsPage(SettingsPage::Video),
            ))
            .add(settings_nav_row(
                fl!("settings-timelapse"),
                Message::OpenSettingsPage(SettingsPage::Timelapse),
            ))
            .add(settings_nav_row(
                fl!("settings-overlay"),
                Message::OpenSettingsPage(SettingsPage::Overlay),
            ))
            .add(settings_nav_row(
                fl!("virtual-camera-title"),
                Message::OpenSettingsPage(SettingsPage::VirtualCamera),
            ));

        let other = widget::settings::section()
            .title(fl!("settings-other"))
            .add(settings_nav_row(
                fl!("insights-title"),
                Message::ToggleContextPage(ContextPage::Insights),
            ))
            .add(settings_nav_row(
                fl!("keybindings-page-title"),
                Message::ToggleContextPage(ContextPage::KeyBindings),
            ))
            .add(settings_nav_row(
                fl!("settings-bug-reports"),
                Message::OpenSettingsPage(SettingsPage::BugReports),
            ))
            .add(settings_nav_row(
                fl!("about"),
                Message::OpenSettingsPage(SettingsPage::About),
            ));

        let reset =
            widget::button::standard(fl!("settings-reset-all")).on_press(Message::ResetAllSettings);

        let content: Element<'_, Message> = widget::settings::view_column(vec![
            appearance.into(),
            categories.into(),
            other.into(),
            reset.into(),
        ])
        .into();

        context_drawer::context_drawer(content, Message::ToggleContextPage(ContextPage::Settings))
            .title(fl!("settings-title"))
    }

    /// Wrap a sub-page's sections in a context drawer titled `title`, with a
    /// back button (top-left) that returns to the category menu.
    fn settings_subpage<'a>(
        &'a self,
        title: String,
        sections: Vec<Element<'a, Message>>,
    ) -> context_drawer::ContextDrawer<'a, Message> {
        let content: Element<'a, Message> = widget::settings::view_column(sections).into();
        context_drawer::context_drawer(content, Message::ToggleContextPage(ContextPage::Settings))
            .title(title)
            .actions(self.settings_back_button())
    }

    /// About sub-page: the standard COSMIC about panel with a back button.
    fn settings_about_view(&self) -> context_drawer::ContextDrawer<'_, Message> {
        let about = widget::about(&self.about, |url| Message::LaunchUrl(url.to_string()));
        context_drawer::context_drawer(about, Message::ToggleContextPage(ContextPage::Settings))
            .title(fl!("about"))
            .actions(self.settings_back_button())
    }

    /// Camera sub-page: device selection, default mode, and mirroring.
    fn camera_sections(&self) -> Vec<Element<'_, Message>> {
        let is_recording = self.recording.is_recording();

        // Default mode index (matches visible dropdown entries which may exclude Virtual)
        let visible_default_modes: Vec<crate::app::state::CameraMode> = {
            let mut modes = vec![
                crate::app::state::CameraMode::Photo,
                crate::app::state::CameraMode::Video,
                crate::app::state::CameraMode::Timelapse,
            ];
            if self.config.virtual_camera_enabled {
                modes.push(crate::app::state::CameraMode::Virtual);
            }
            modes
        };
        let current_default_mode_index = visible_default_modes
            .iter()
            .position(|m| *m == self.config.default_mode)
            .unwrap_or(0);

        // Custom device row with label, info button, and dropdown
        let device_control: Element<'_, Message> = if is_recording {
            disabled_text(
                self.camera_dropdown_options
                    .get(self.current_camera_index)
                    .cloned()
                    .unwrap_or_default(),
            )
        } else {
            widget::dropdown(
                &self.camera_dropdown_options,
                Some(self.current_camera_index),
                Message::SelectCamera,
            )
            .into()
        };

        let device_label_with_info = widget::Row::new()
            .push(widget::text::body(fl!("settings-device")))
            .push(widget::space::horizontal().width(Length::Fixed(4.0)))
            .push(
                widget::button::icon(icon::from_name("dialog-information-symbolic").symbolic(true))
                    .extra_small()
                    .on_press(Message::ToggleDeviceInfo),
            )
            .push(widget::space::horizontal())
            .push(device_control)
            .align_y(Alignment::Center)
            .width(Length::Fill);

        let mut camera_section = widget::settings::section()
            .title(fl!("camera"))
            .add(widget::settings::item_row(vec![
                device_label_with_info.into(),
            ]))
            .add(
                widget::settings::item::builder(fl!("settings-default-mode"))
                    .description(fl!("settings-default-mode-description"))
                    .control(widget::dropdown(
                        &self.default_mode_dropdown_options,
                        Some(current_default_mode_index),
                        Message::SelectDefaultMode,
                    )),
            );

        // Add device info panel if visible
        if self.device_info_visible {
            camera_section = camera_section.add(self.build_device_info_panel());
        }

        // Mirror preview section (preview flip + optional capture flip)
        let mut mirror_section = widget::settings::section().add(
            widget::settings::item::builder(fl!("settings-mirror-preview"))
                .description(fl!("settings-mirror-preview-description"))
                .toggler(self.config.mirror_preview, |_| Message::ToggleMirrorPreview),
        );
        if self.config.mirror_preview {
            mirror_section = mirror_section.add(
                widget::settings::item::builder(fl!("settings-mirror-captures"))
                    .description(fl!("settings-mirror-captures-description"))
                    .toggler(self.config.mirror_captures, |_| {
                        Message::ToggleMirrorCaptures
                    }),
            );
        }

        let mut sections = vec![camera_section.into(), mirror_section.into()];

        // Haptic feedback (only where the device has haptics)
        if crate::backends::haptic::is_available() {
            let haptic_section = widget::settings::section().add(
                widget::settings::item::builder(fl!("settings-haptic-feedback"))
                    .description(fl!("settings-haptic-feedback-description"))
                    .toggler(self.config.haptic_feedback, |_| {
                        Message::ToggleHapticFeedback
                    }),
            );
            sections.push(haptic_section.into());
        }

        sections
    }

    /// Photo sub-page: output format and HDR+ settings.
    fn photo_sections(&self) -> Vec<Element<'_, Message>> {
        use crate::config::BurstModeSetting;
        // Index 0 = Off, 1 = Auto, 2 = 4 frames, 3 = 6 frames, 4 = 8 frames, 5 = 50 frames
        let current_hdr_index = match self.config.burst_mode_setting {
            BurstModeSetting::Off => 0,
            BurstModeSetting::Auto => 1,
            BurstModeSetting::Frames4 => 2,
            BurstModeSetting::Frames6 => 3,
            BurstModeSetting::Frames8 => 4,
            BurstModeSetting::Frames50 => 5,
        };

        // Photo output format index
        let current_photo_format_index = PhotoOutputFormat::ALL
            .iter()
            .position(|f| *f == self.config.photo_output_format)
            .unwrap_or(0); // Default to JPEG (index 0)

        let mut photo_section = widget::settings::section()
            .title(fl!("settings-photo"))
            .add(
                widget::settings::item::builder(fl!("settings-photo-format"))
                    .description(fl!("settings-photo-format-description"))
                    .control(widget::dropdown(
                        &self.photo_output_format_dropdown_options,
                        Some(current_photo_format_index),
                        Message::SelectPhotoOutputFormat,
                    )),
            )
            .add(
                widget::settings::item::builder(fl!("settings-hdr-plus"))
                    .description(fl!("settings-hdr-plus-description"))
                    .control(widget::dropdown(
                        &self.burst_mode_frame_count_dropdown_options,
                        Some(current_hdr_index),
                        Message::SetBurstModeFrameCount,
                    )),
            );

        if self.config.burst_mode_setting != BurstModeSetting::Off {
            photo_section = photo_section.add(
                widget::settings::item::builder(fl!("settings-save-burst-raw"))
                    .description(fl!("settings-save-burst-raw-description"))
                    .toggler(self.config.save_burst_raw, |_| Message::ToggleSaveBurstRaw),
            );
        }

        vec![photo_section.into()]
    }

    /// Video sub-page: encoder, quality, and audio settings.
    fn video_sections(&self) -> Vec<Element<'_, Message>> {
        let is_recording = self.recording.is_recording();

        let current_bitrate_index = BitratePreset::ALL
            .iter()
            .position(|p| *p == self.config.bitrate_preset)
            .unwrap_or(1); // Default to Medium (index 1)

        let current_audio_encoder_index = AudioEncoder::ALL
            .iter()
            .position(|e| *e == self.config.audio_encoder)
            .unwrap_or(0); // Default to Opus (index 0)

        let mut video_section = if is_recording {
            widget::settings::section()
                .title(fl!("settings-video"))
                .add(
                    widget::settings::item::builder(fl!("settings-encoder")).control(
                        disabled_text(
                            self.video_encoder_dropdown_options
                                .get(self.current_video_encoder_index)
                                .cloned()
                                .unwrap_or_default(),
                        ),
                    ),
                )
                .add(
                    widget::settings::item::builder(fl!("settings-quality")).control(
                        disabled_text(
                            self.bitrate_preset_dropdown_options
                                .get(current_bitrate_index)
                                .cloned()
                                .unwrap_or_default(),
                        ),
                    ),
                )
                .add(
                    widget::settings::item::builder(fl!("settings-record-audio")).control(
                        widget::toggler(self.config.record_audio)
                            .on_toggle_maybe(None::<fn(bool) -> Message>),
                    ),
                )
        } else {
            widget::settings::section()
                .title(fl!("settings-video"))
                .add(
                    widget::settings::item::builder(fl!("settings-encoder")).control(
                        widget::dropdown(
                            &self.video_encoder_dropdown_options,
                            Some(self.current_video_encoder_index),
                            Message::SelectVideoEncoder,
                        ),
                    ),
                )
                .add(
                    widget::settings::item::builder(fl!("settings-quality")).control(
                        widget::dropdown(
                            &self.bitrate_preset_dropdown_options,
                            Some(current_bitrate_index),
                            Message::SelectBitratePreset,
                        ),
                    ),
                )
                .add(
                    widget::settings::item::builder(fl!("settings-record-audio"))
                        .toggler(self.config.record_audio, |_| Message::ToggleRecordAudio),
                )
        };

        // Only show audio encoder and microphone selection when audio is enabled
        if self.config.record_audio {
            if is_recording {
                video_section = video_section
                    .add(
                        widget::settings::item::builder(fl!("settings-audio-encoder")).control(
                            disabled_text(
                                self.audio_encoder_dropdown_options
                                    .get(current_audio_encoder_index)
                                    .cloned()
                                    .unwrap_or_default(),
                            ),
                        ),
                    )
                    .add(
                        widget::settings::item::builder(fl!("settings-microphone")).control(
                            disabled_text(
                                self.audio_dropdown_options
                                    .get(self.current_audio_device_index)
                                    .cloned()
                                    .unwrap_or_default(),
                            ),
                        ),
                    );
            } else {
                video_section = video_section
                    .add(
                        widget::settings::item::builder(fl!("settings-audio-encoder")).control(
                            widget::dropdown(
                                &self.audio_encoder_dropdown_options,
                                Some(current_audio_encoder_index),
                                Message::SelectAudioEncoder,
                            ),
                        ),
                    )
                    .add(
                        widget::settings::item::builder(fl!("settings-microphone")).control(
                            widget::dropdown(
                                &self.audio_dropdown_options,
                                Some(self.current_audio_device_index),
                                Message::SelectAudioDevice,
                            ),
                        ),
                    );
            }
        }

        if self.config.record_audio {
            use crate::app::controls::audio_meter::{AudioMeterStyle, audio_meter};

            let meter_row = match self.current_audio_levels() {
                Some(levels) => widget::Row::new()
                    .push(widget::text::body(fl!("settings-mic-level")))
                    .push(widget::space::horizontal().width(Length::Fill))
                    .push(audio_meter(
                        levels.output_peak_db,
                        levels.output_rms_db,
                        AudioMeterStyle {
                            width: 120.0,
                            height: 10.0,
                            show_peak: true,
                        },
                    ))
                    .push(widget::space::horizontal().width(Length::Fixed(8.0)))
                    .push(
                        widget::text::caption(format!("{:.0} dB", levels.output_rms_db))
                            .font(cosmic::font::mono())
                            .size(11),
                    )
                    .align_y(Alignment::Center),
                None => widget::Row::new()
                    .push(widget::text::body(fl!("settings-mic-level")))
                    .push(widget::space::horizontal().width(Length::Fill))
                    .push(widget::text::caption(fl!("settings-mic-level-initializing")).size(11))
                    .align_y(Alignment::Center),
            };

            video_section = video_section.add(widget::settings::item_row(vec![meter_row.into()]));
        }

        vec![video_section.into()]
    }

    /// Timelapse sub-page: capture interval.
    fn timelapse_sections(&self) -> Vec<Element<'_, Message>> {
        let current_timelapse_interval_index = TimelapseInterval::ALL
            .iter()
            .position(|i| *i == self.config.timelapse_interval)
            .unwrap_or(0);

        let timelapse_section = if self.timelapse.is_running() {
            widget::settings::section()
                .title(fl!("settings-timelapse"))
                .add(
                    widget::settings::item::builder(fl!("settings-timelapse-interval"))
                        .description(fl!("settings-timelapse-interval-description"))
                        .control(disabled_text(
                            self.timelapse_interval_dropdown_options
                                .get(current_timelapse_interval_index)
                                .cloned()
                                .unwrap_or_default(),
                        )),
                )
        } else {
            widget::settings::section()
                .title(fl!("settings-timelapse"))
                .add(
                    widget::settings::item::builder(fl!("settings-timelapse-interval"))
                        .description(fl!("settings-timelapse-interval-description"))
                        .control(widget::dropdown(
                            &self.timelapse_interval_dropdown_options,
                            Some(current_timelapse_interval_index),
                            Message::SetTimelapseInterval,
                        )),
                )
        };

        vec![timelapse_section.into()]
    }

    /// Overlay sub-page: overlay effect, composition guide, and controls position.
    fn overlay_sections(&self) -> Vec<Element<'_, Message>> {
        // Overlay effect index. Indexes `available()`, which is shorter
        // off-COSMIC — hence the shared mapping rather than a literal index.
        let current_overlay_effect_index = self.config.overlay_effect.dropdown_index();

        let current_guide_index = crate::config::CompositionGuide::ALL
            .iter()
            .position(|g| *g == self.config.composition_guide)
            .unwrap_or(0);

        let current_controls_position_index =
            controls_position_index(self.config.controls_position);

        let overlay_section = widget::settings::section()
            .title(fl!("settings-overlay"))
            .add(
                widget::settings::item::builder(fl!("settings-overlay-effect"))
                    .description(fl!("settings-overlay-effect-description"))
                    .control(widget::dropdown(
                        &self.overlay_effect_dropdown_options,
                        Some(current_overlay_effect_index),
                        Message::SetOverlayEffect,
                    )),
            )
            .add(
                widget::settings::item::builder(fl!("settings-composition-guide"))
                    .description(fl!("settings-composition-guide-description"))
                    .control(widget::dropdown(
                        &self.composition_guide_dropdown_options,
                        Some(current_guide_index),
                        Message::SelectCompositionGuide,
                    )),
            )
            .add(
                widget::settings::item::builder(fl!("settings-controls-position"))
                    .description(fl!("settings-controls-position-description"))
                    .control(widget::dropdown(
                        &self.controls_position_dropdown_options,
                        Some(current_controls_position_index),
                        Message::SetControlsPosition,
                    )),
            );

        vec![overlay_section.into()]
    }

    /// Virtual camera sub-page.
    fn virtual_camera_sections(&self) -> Vec<Element<'_, Message>> {
        let virtual_camera_section = widget::settings::section().add(
            widget::settings::item::builder(fl!("virtual-camera-title"))
                .description(fl!("virtual-camera-description"))
                .toggler(self.config.virtual_camera_enabled, |_| {
                    Message::ToggleVirtualCameraEnabled
                }),
        );

        vec![virtual_camera_section.into()]
    }

    /// Bug reports sub-page.
    fn bug_reports_sections(&self) -> Vec<Element<'_, Message>> {
        let bug_report_button = widget::button::standard(fl!("settings-report-bug"))
            .on_press(Message::GenerateBugReport);

        let bug_report_control = if self.last_bug_report_path.is_some() {
            let show_report_button = widget::button::standard(fl!("settings-show-report"))
                .on_press(Message::ShowBugReport);

            widget::Row::new()
                .push(bug_report_button)
                .push(widget::space::horizontal().width(Length::Fixed(8.0)))
                .push(show_report_button)
                .into()
        } else {
            bug_report_button.into()
        };

        let bug_reports_section = widget::settings::section()
            .title(fl!("settings-bug-reports"))
            .add(widget::settings::item_row(vec![bug_report_control]));

        vec![bug_reports_section.into()]
    }

    /// Build the device info panel (shown when info button is clicked)
    fn build_device_info_panel(&self) -> Element<'_, Message> {
        // Helper to build a label: value row
        fn info_row<'a>(label: String, value: &str) -> Element<'a, Message> {
            widget::Row::new()
                .push(widget::text(label).size(12).font(cosmic::font::bold()))
                .push(widget::space::horizontal().width(Length::Fixed(8.0)))
                .push(widget::text(value.to_string()).size(12))
                .into()
        }

        let camera = self.available_cameras.get(self.current_camera_index);
        let device_info = camera.and_then(|c| c.device_info.as_ref());

        let mut info_column = widget::Column::new().spacing(4);

        if let Some(info) = device_info {
            // V4L2 device info
            if !info.card.is_empty() {
                info_column = info_column.push(info_row(fl!("device-info-card"), &info.card));
            }
            if !info.driver.is_empty() {
                info_column = info_column.push(info_row(fl!("device-info-driver"), &info.driver));
            }
            if !info.path.is_empty() {
                info_column = info_column.push(info_row(fl!("device-info-path"), &info.path));
            }
            if !info.real_path.is_empty() && info.real_path != info.path {
                info_column =
                    info_column.push(info_row(fl!("device-info-real-path"), &info.real_path));
            }
        } else if let Some(cam) = camera
            && (cam.sensor_model.is_some()
                || cam.camera_location.is_some()
                || cam.libcamera_version.is_some()
                || cam.pipeline_handler.is_some())
        {
            // libcamera device info (no V4L2 DeviceInfo, but has libcamera-specific fields)
            info_column = info_column.push(info_row(fl!("device-info-device-path"), &cam.path));
            if let Some(ref model) = cam.sensor_model {
                info_column = info_column.push(info_row(fl!("device-info-sensor"), model));
            }
            if let Some(ref handler) = cam.pipeline_handler {
                info_column = info_column.push(info_row(fl!("device-info-pipeline"), handler));
            }
            if let Some(ref version) = cam.libcamera_version {
                info_column =
                    info_column.push(info_row(fl!("device-info-libcamera-version"), version));
            }
            let multistream_str = if cam.supports_multistream {
                fl!("device-info-multistream-yes")
            } else {
                fl!("device-info-multistream-no")
            };
            info_column =
                info_column.push(info_row(fl!("device-info-multistream"), &multistream_str));
            if cam.rotation.degrees() != 0 {
                info_column = info_column.push(info_row(
                    fl!("device-info-rotation"),
                    &format!("{}°", cam.rotation.degrees()),
                ));
            }
        } else {
            info_column = info_column.push(widget::text(fl!("device-info-none")).size(12));
        }

        widget::container(info_column)
            .padding(8)
            .class(cosmic::theme::Container::Card)
            .into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ControlsPosition;

    #[test]
    fn overlay_controls_position_index_tracks_every_option() {
        for (expected, position) in ControlsPosition::ALL.into_iter().enumerate() {
            assert_eq!(controls_position_index(position), expected);
        }
    }
}
