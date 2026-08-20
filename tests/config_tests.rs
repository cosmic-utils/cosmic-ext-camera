// SPDX-License-Identifier: GPL-3.0-only

//! Integration tests for configuration module

use camera::Config;

fn assigned_number(source: &str, prefix: &str) -> u64 {
    source
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix(prefix)
                .and_then(|value| value.trim_end_matches(';').parse().ok())
        })
        .unwrap_or_else(|| panic!("missing numeric assignment for {prefix}"))
}

fn config_schema_version(source: &str) -> u64 {
    source
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("#[version = ")
                .and_then(|value| value.strip_suffix(']'))
                .and_then(|value| value.parse().ok())
        })
        .expect("Config is missing its cosmic-config schema version")
}

#[test]
fn test_config_default() {
    // Test that default config can be created
    let config = Config::default();

    // Check sensible defaults
    assert!(
        config.mirror_preview,
        "Mirror preview should be enabled by default"
    );
}

#[test]
fn test_config_bug_report_url() {
    // Test that bug report URL is set
    let config = Config::default();
    assert!(
        !config.bug_report_url.is_empty(),
        "Bug report URL should not be empty"
    );
}

#[test]
fn preview_harness_uses_current_config_schema() {
    let config_source = include_str!("../src/config.rs");
    let harness_source = include_str!("../preview/capture-previews.sh");

    assert_eq!(
        assigned_number(harness_source, "CONFIG_VERSION="),
        config_schema_version(config_source),
        "preview/capture-previews.sh writes forced appearance and fit/fill settings to a stale cosmic-config directory"
    );
}
