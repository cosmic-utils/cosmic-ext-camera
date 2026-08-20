#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Regenerates the Flathub preview screenshots without a user present.
#
# Runs a headless wlroots compositor (sway) and launches the app once per shot
# with a synthetic camera fed by a still image (--preview-fake-camera plus
# --preview-source), so it sees a fixed camera feed with no hardware and no
# dma-buf provider. Each shot is driven into the right UI state with the app's
# own keyboard shortcuts and captured with grim.
#
# In English each shot is taken in every theme / overlay-effect combination (see
# VARIANTS); every other translated language gets the published combination only,
# under locales/<lang>/. Languages are discovered from the i18n tree.
#
# This is meant to run inside preview/Containerfile, where the renderer, fonts
# and icon theme are pinned, and the screenshots are pixel-compared to decide
# whether anything changed, so an unpinned font or mesa version would look like
# a UI change. It will run on any host with sway + grim + wtype, but the output
# is only comparable to the committed PNGs when the environment matches.
#
# Usage:
#   preview/capture-previews.sh [output-dir]
#
# Environment:
#   CAMERA_BIN   camera binary to use (default: build with cargo)
#   SHOTS            comma-separated shot numbers to capture (default: all)
#   VARIANTS_FILTER  comma-separated variant names (default: all)
#   LOCALES_FILTER   comma-separated languages, or `none` for English only
#   KEEP_GOING       set to 1 to continue after a failed shot
#   KEEP_LOGS        set to 1 to keep each run's app log beside its screenshot

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_DIR="$SCRIPT_DIR/source"
SHOTS_CONF="$SCRIPT_DIR/shots.conf"
OUT_DIR="${1:-$SCRIPT_DIR}"

# Keyboard shortcuts for each step, mirroring the defaults in
# src/app/keybind/action.rs. Keep the two in sync: a shortcut that no longer
# exists silently produces a screenshot of the wrong UI state.
#
# Values are wtype arguments.
declare -A STEP_KEYS=(
    [tools-menu]="t"          # Action::ToggleToolsMenu
    [filters]="l"             # Action::ToggleFilters
    [settings]="-M ctrl , -m ctrl"  # Action::ToggleSettings
    [video-mode]="n"          # Action::PrevMode, Photo is followed by Video
    [record]="-k space"       # Action::Capture, in Video mode starts recording
)

# Pause after each step so the UI reaches a steady state before the next one.
# Generous because saving a photo draws a progress ring over the gallery button
# that takes a moment to finish and clear.
STEP_PAUSE=1.8
# Leading sleep inside every wtype invocation.
#
# wtype creates its virtual keyboard when it starts and destroys it when it
# exits. A key sent immediately is delivered before the app has bound the new
# keyboard and is dropped on the floor, silently, since wtype itself succeeds.
# Sleeping first lets the client finish the handshake, after which every key in
# the same invocation lands.
KEY_WARMUP_MS=800
# The app logs this on a dedicated tracing target the moment its first preview
# frame is set (src/app/handlers/virtual_camera.rs). The app is launched with
# `RUST_LOG=...,preview_ready=debug` so the line reaches the log, and each shot
# waits for it rather than for a fixed guess — the frame comes from a decoded
# still image, so how long that takes varies with runner load. Grepping the
# app's own log is why this is a substring, not the whole formatted line.
PREVIEW_READY_MARKER="preview_ready"
# Safety net if the marker never arrives (a hang, or the line silently renamed):
# proceed after this many seconds rather than blocking the whole run. The
# per-shot `settle_ms` still runs afterwards, adding margin before the capture.
READY_TIMEOUT=15
# How long to wait for the window to appear before giving up on a shot.
WINDOW_TIMEOUT=60

log() { printf '\033[1;34m==>\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------

# identify reads captured screenshots back to check their size; magick compares
# a localized shot against its English counterpart. Both were previously
# undeclared, and both fail in ways that look like a valid shot.
for tool in sway swaymsg grim wtype jq identify magick awk dbus-daemon; do
    command -v "$tool" >/dev/null || die "$tool is not installed (run this inside preview/Containerfile)"
done

[[ -d "$SOURCE_DIR" ]] || die "source images not found at $SOURCE_DIR"
[[ -f "$SHOTS_CONF" ]] || die "shot list not found at $SHOTS_CONF"

CAMERA="${CAMERA_BIN:-}"
if [[ -z "$CAMERA" ]]; then
    log "Building camera (release-fast)"
    ( cd "$REPO_DIR" && cargo build --profile release-fast )
    # Honour CARGO_TARGET_DIR: generate-previews.sh points it at the mounted
    # cache (/target), so the binary lands there, not under $REPO_DIR/target.
    CAMERA="${CARGO_TARGET_DIR:-$REPO_DIR/target}/release-fast/camera"
fi
[[ -x "$CAMERA" ]] || die "camera binary not found or not executable: $CAMERA"

mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Isolated session
# ---------------------------------------------------------------------------
#
# A throwaway HOME keeps every run identical: no stored config (so the app uses
# its defaults), and an empty gallery that each shot seeds itself.

SESSION_DIR="$(mktemp -d)"
export HOME="$SESSION_DIR/home"
export XDG_CONFIG_HOME="$HOME/.config"
export XDG_DATA_HOME="$HOME/.local/share"
export XDG_CACHE_HOME="$HOME/.cache"
# Deliberately NOT inherited. On a host that is already running a Wayland
# session, reusing its runtime dir means the socket globs below can pick the
# user's own compositor instead of the throwaway one started here: the app then
# opens on the real desktop, wtype types into it and grim screenshots it. Always
# private, so the two can never be confused.
export XDG_RUNTIME_DIR="$SESSION_DIR/runtime"
mkdir -p "$HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_CACHE_HOME" "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# cosmic-config stores one file per field, named after the field, under
# v<CONFIG_VERSION>. Keep both in sync with src/config.rs.
APP_ID="io.github.cosmic_utils.camera"
CONFIG_VERSION=21
CONFIG_DIR="$XDG_CONFIG_HOME/cosmic/$APP_ID/v$CONFIG_VERSION"
mkdir -p "$CONFIG_DIR"

# Copies COSMIC's default theme into the throwaway session's config.
#
# libcosmic falls back to a built-in palette when it finds no theme config, and
# that fallback's accent is cyan, so every button, toggle and shutter ring comes
# out the wrong colour, on a desktop whose actual default accent is blue. The
# defaults are installed by preview/Containerfile; see the note there.
#
# Copied into the user config rather than relied on as a system default, because
# the same "no theme found" path is what produced the wrong accent in the first
# place. Both Dark and Light are seeded, since `app_theme` below chooses between them,
# and a shot in the theme that was not seeded would silently fall back again.
COSMIC_THEME_DEFAULTS="${COSMIC_THEME_DEFAULTS:-/usr/share/cosmic}"

seed_cosmic_theme() {
    local seeded=0 theme_dir
    mkdir -p "$XDG_CONFIG_HOME/cosmic" || return 1
    for theme_dir in "$COSMIC_THEME_DEFAULTS"/com.system76.CosmicTheme.*; do
        [[ -d "$theme_dir" ]] || continue
        # Replaced outright: `cp -r src dst` nests a second copy inside an
        # existing dst, and this runs once per shot.
        rm -rf "$XDG_CONFIG_HOME/cosmic/$(basename "$theme_dir")"
        cp -r "$theme_dir" "$XDG_CONFIG_HOME/cosmic/" || {
            warn "cannot copy $theme_dir"
            return 1
        }
        seeded=1
    done
    # Not a warning: without these every single shot in the run is themed by
    # libcosmic's built-in fallback instead of COSMIC's own theme, so the whole
    # run is wrong rather than one shot.
    if (( seeded == 0 )); then
        warn "no COSMIC theme defaults under $COSMIC_THEME_DEFAULTS"
        return 1
    fi
}

# Pins the two settings the shots vary over. Neither may be left to its default:
# app_theme defaults to `System` (→ light off COSMIC) and overlay_effect to
# `System` too, which off COSMIC has no desktop setting to read and resolves to
# translucent by fallback, so an unseeded run silently produces a light,
# translucent shot that differs from a dark, frosted one in every pixel.
#
# $3 is a shot's optional `config` column: comma-separated field=value pairs for
# state that has no keyboard shortcut, such as the fit/fill preview toggle.
#
# Written before each launch; cosmic-config reads these at startup.
seed_config() {
    # Wiped rather than overwritten: the config directory outlives a single
    # shot, and both the previous shot's extra fields and anything the app
    # persisted on its way out would otherwise leak into this one.
    rm -rf "$CONFIG_DIR"
    mkdir -p "$CONFIG_DIR" || { warn "cannot create $CONFIG_DIR"; return 1; }

    seed_cosmic_theme || return 1
    # A write that fails here leaves the field unset, and an unset app_theme
    # resolves to System, i.e. a light shot filed under a dark variant name.
    printf '%s' "$1" >"$CONFIG_DIR/app_theme" || { warn "cannot write app_theme"; return 1; }
    printf '%s' "$2" >"$CONFIG_DIR/overlay_effect" || { warn "cannot write overlay_effect"; return 1; }

    local pair name
    local -a pairs
    IFS=',' read -ra pairs <<<"${3:-}"
    for pair in "${pairs[@]}"; do
        [[ -z "$pair" ]] && continue
        if [[ "$pair" != *=* ]]; then
            warn "malformed config entry '$pair'"
            return 1
        fi
        # The field name becomes a path under $CONFIG_DIR, and shots.conf will
        # come from a pull request branch once this runs in CI. Restrict it to
        # what a cosmic-config field can actually be called so it cannot climb
        # out of the config directory.
        name="${pair%%=*}"
        if [[ ! "$name" =~ ^[A-Za-z0-9_]+$ ]]; then
            warn "unsafe config field name '$name'"
            return 1
        fi
        printf '%s' "${pair#*=}" >"$CONFIG_DIR/$name" || {
            warn "cannot write config field '$name'"
            return 1
        }
    done
}

# ---------------------------------------------------------------------------
# Fake camera
# ---------------------------------------------------------------------------
#
# The app fakes its own camera: `--preview-fake-camera` presents synthetic
# devices (a back and a front, so the switcher shows) while `--preview-source`
# feeds each shot's still image as the frames. Both are wired at launch below.
#
# This replaced libcamera's `virtual` pipeline handler, which exported its
# buffers through dma-buf and so required /dev/udmabuf (or a dma_heap) on the
# host — a provider GitHub-hosted runners do not ship. The file-source path is
# pure CPU memory, so the harness now runs anywhere with no device access. The
# synthetic devices matter beyond filling the preview: almost every control in
# the UI is driven off the enumerated device list, so without them the preview,
# switcher and format pickers would fall back to the "no camera" placeholder.

# Drops a photo into the gallery directory so the gallery button shows a
# thumbnail instead of its empty placeholder.
#
# Deliberately a file copy rather than pressing the shutter. Taking a real photo
# off the virtual camera produces a *corrupt* JPEG: the right of the frame
# comes out solid green and the saved size is a pixel short of the stream, so
# the gallery button ends up showing a broken thumbnail. The still-capture path
# mishandles this camera's raw NV12 buffers (green is the giveaway: chroma plane
# missing); a real camera handing over MJPEG never takes that path, which is why
# it does not reproduce on a normal desktop. The preview itself is unaffected.
# Resets the gallery to a known state, optionally with one photo in it.
#
# Called for EVERY shot, not only the ones asking for `seed-photo`. The app picks
# its thumbnail as the newest file by mtime across both the photo and the video
# directory, and nothing else here resets $HOME between shots, so a shot without
# its own gallery state would otherwise inherit whatever ran before it: in a full
# run the previous shot's seeded photo, or alone the empty placeholder. Resetting
# to a known state keeps each shot's gallery button deterministic.
#
# The video directory is cleared for the same reason. Shot 004 spoofs recording
# (--preview-spoof-recording) and writes no file, so nothing accumulates there,
# but the reset still guards against a stray clip from an interrupted run.
seed_gallery() {
    local source_path="${1:-}"
    local photos_dir="$HOME/Pictures/Camera"
    local videos_dir="$HOME/Videos/Camera"

    rm -rf "$photos_dir" "$videos_dir"
    mkdir -p "$photos_dir" "$videos_dir" || {
        warn "cannot reset gallery directories"
        return 1
    }
    [[ -n "$source_path" ]] || return 0
    cp "$source_path" "$photos_dir/IMG_20260101_120000_000.jpg" || {
        warn "cannot seed gallery from $source_path"
        return 1
    }
}

# Stops the app started for the current shot. Every failure path in capture_shot
# goes through this: a shot that returns while its app is still running leaves a
# second window floating on the same output, and the next launch then either
# exits into the running instance or never gets a window of its own, so one
# failed shot would quietly take the rest of the run with it.
kill_app() {
    [[ -n "$APP_PID" ]] || return 0
    kill "$APP_PID" 2>/dev/null || true
    wait "$APP_PID" 2>/dev/null || true
    APP_PID=""
}

# The session directory is thrown away on exit, and a shot that comes out wrong
# looks perfectly valid as a PNG, so the app's own log is the only way to tell
# "no camera frame" from "wrong UI state" after the fact. Kept for failures too,
# which are exactly the runs worth looking at.
keep_log() {
    [[ "${KEEP_LOGS:-0}" == "1" ]] || return 0
    cp "$SESSION_DIR/camera-$1.log" "$OUT_DIR/camera-$1.log" 2>/dev/null || true
}

SWAY_PID=""
DBUS_PID=""
APP_PID=""
KEYBOARD_PID=""

cleanup() {
    local status=$?
    [[ -n "$APP_PID" ]] && kill "$APP_PID" 2>/dev/null || true
    [[ -n "$KEYBOARD_PID" ]] && kill "$KEYBOARD_PID" 2>/dev/null || true
    [[ -n "$SWAY_PID" ]] && kill "$SWAY_PID" 2>/dev/null || true
    [[ -n "$DBUS_PID" ]] && kill "$DBUS_PID" 2>/dev/null || true
    rm -rf "$SESSION_DIR"
    exit $status
}
trap cleanup EXIT INT TERM

# libcosmic's config and single-instance support both talk to the session bus.
# Without one, startup logs errors and stalls on connection timeouts.
if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ]]; then
    log "Starting session bus"
    dbus_out="$(dbus-daemon --session --print-address=1 --print-pid=1 --fork)"
    DBUS_SESSION_BUS_ADDRESS="$(sed -n '1p' <<<"$dbus_out")"
    DBUS_PID="$(sed -n '2p' <<<"$dbus_out")"
    export DBUS_SESSION_BUS_ADDRESS
fi

# 1920x1080 is only the virtual output; every window is floating and resized to
# the size its shot asks for, so the output just has to be big enough.
SWAY_CONFIG="$SESSION_DIR/sway.conf"
cat >"$SWAY_CONFIG" <<'EOF'
output HEADLESS-1 resolution 1920x1080 position 0 0 scale 1
# The app draws its own title bar; any compositor decoration would end up in
# the screenshots and differ from what users running COSMIC see.
default_border none
default_floating_border none
gaps inner 0
gaps outer 0
focus_follows_mouse no
# Float everything: tiled windows are resized to fill the output, which would
# make the requested window size meaningless.
for_window [title=".*"] floating enable
for_window [app_id=".*"] floating enable
EOF

log "Starting headless compositor"
sway --config "$SWAY_CONFIG" >"$SESSION_DIR/sway.log" 2>&1 &
SWAY_PID=$!

for _ in $(seq 50); do
    SWAYSOCK="$(ls "$XDG_RUNTIME_DIR"/sway-ipc.*.sock 2>/dev/null | head -n1 || true)"
    [[ -n "$SWAYSOCK" ]] && break
    sleep 0.2
done
[[ -n "${SWAYSOCK:-}" ]] || { cat "$SESSION_DIR/sway.log" >&2; die "sway did not start"; }
export SWAYSOCK
swaymsg -t get_version >/dev/null || die "sway IPC is not responding"

# sway only exports WAYLAND_DISPLAY to processes it spawns itself. The app is
# started from this script, so point it at the compositor explicitly. Without
# it, winit finds no display and panics before opening a window.
for socket in "$XDG_RUNTIME_DIR"/wayland-*; do
    [[ "$socket" == *.lock ]] && continue
    WAYLAND_DISPLAY="$(basename "$socket")"
    export WAYLAND_DISPLAY
    break
done
[[ -n "${WAYLAND_DISPLAY:-}" ]] || die "sway did not create a wayland socket"
log "Compositor ready on $WAYLAND_DISPLAY"

# Give the seat a keyboard that outlives a single keystroke.
#
# wlroots' headless backend creates no input devices (WLR_HEADLESS_INPUTS was
# dropped in 0.17), so seat0 comes up advertising no capabilities at all. Two
# things follow from that. wtype's virtual keyboard is then the only keyboard
# the seat ever has, and it disappears again the moment wtype exits. And winit
# treats wl_keyboard focus as window focus, so with no keyboard on the seat the
# app is never told it is focused: it draws its title bar in the dimmed
# unfocused colours, and every screenshot showed an inactive-looking window.
#
# `wtype -` reads the text to type from stdin. Pointed at a fifo whose write end
# this script holds open, it types nothing and stays alive, so the seat keeps
# its keyboard capability for the whole run and the app holds focus the way it
# would on a real desktop.
KEYBOARD_FIFO="$SESSION_DIR/keyboard.fifo"
mkfifo "$KEYBOARD_FIFO" || die "could not create the keyboard fifo"
wtype - <"$KEYBOARD_FIFO" &
KEYBOARD_PID=$!
# Held open for the rest of the run: closing it is what makes wtype exit.
exec 9>"$KEYBOARD_FIFO"
for _ in $(seq 20); do
    [[ "$(swaymsg -t get_seats | jq -r 'map(.capabilities) | add')" != "0" ]] && break
    sleep 0.1
done
if [[ "$(swaymsg -t get_seats | jq -r 'map(.capabilities) | add')" == "0" ]]; then
    die "the virtual keyboard never attached; the app would screenshot unfocused"
fi

# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

# True when two captures are pixel-identical. Used to prove a language actually
# reached the UI; `cmp` would not do, since two encodings of the same pixels can
# differ byte for byte.
images_identical() {
    local share
    share="$(magick "$1" "$2" -compose Difference -composite \
        -colorspace Gray -threshold 2% -format '%[fx:mean]' info: 2>/dev/null || echo "")"
    [[ -n "$share" ]] || return 1
    awk -v s="$share" 'BEGIN { exit !(s == 0) }'
}

# Prints "x,y WxH" for the surface of the window owned by $1, or nothing.
window_geometry() {
    swaymsg -t get_tree | jq -r --argjson pid "$1" '
        .. | objects | select(.pid? == $pid and .type? == "floating_con")
        | "\(.rect.x + .window_rect.x),\(.rect.y + .window_rect.y) \(.window_rect.width)x\(.window_rect.height)"
    ' | head -n1
}

capture_shot() {
    local num="$1" source_file="$2" window="$3" steps="$4" settle_ms="$5"
    local shot_config="$6" description="$7" theme="$8" overlay="$9" out="${10}"

    local source_path="$SOURCE_DIR/$source_file"
    if [[ ! -f "$source_path" ]]; then
        warn "source image missing: $source_path, skipping preview-$num"
        return 1
    fi

    local width="${window%x*}" height="${window#*x}"
    # Derived from the whole output path, not just the file name: the localized
    # shots reuse `preview-001.png` under `locales/<lang>/`, so a basename tag
    # would give ten languages the same log file and each would overwrite the
    # last, losing exactly the evidence KEEP_LOGS exists to keep.
    local tag
    tag="${out#"$OUT_DIR"/}"
    tag="${tag%.png}"
    tag="${tag//\//-}"

    log "$tag: $description (${window}, steps: ${steps:-none})"

    # Every one of these is checked: `set -e` does not apply inside this function
    # because it is called as an `if ! capture_shot ...` condition, so an
    # unchecked failure here would go on to screenshot the wrong thing and be
    # counted as a successful shot.
    seed_config "$theme" "$overlay" "$shot_config" || return 1
    if [[ ",$steps," == *,seed-photo,* ]]; then
        seed_gallery "$source_path" || return 1
    else
        seed_gallery || return 1
    fi

    # The app renders a synthetic camera (--preview-fake-camera) fed by the shot's
    # source image (--preview-source): no real camera, no libcamera virtual
    # pipeline, so no /dev/udmabuf or dma-buf provider is needed anywhere. The
    # window opens at the shot's size via --preview-window; sway reconfirms it
    # below (it owns the size for tiled windows).
    #
    # A shot can additionally ask the app to boot into a spoofed "recording in
    # progress" state (Video mode + recording indicator, no encoder). That is a
    # launch-time flag, not a keystroke, so it is resolved here like seed-photo
    # and skipped in the step loop below. See --preview-spoof-recording.
    local app_args=(
        --preview-source "$source_path"
        --preview-fake-camera
        --preview-window "$window"
    )
    if [[ ",$steps," == *,spoof-recording,* ]]; then
        app_args+=(--preview-spoof-recording)
    fi
    # Enable the `preview_ready` marker (see wait below) without turning on the
    # rest of the app's debug logging, which would be a per-frame firehose. The
    # container otherwise sets RUST_LOG=warn.
    RUST_LOG="warn,preview_ready=debug" \
        "$CAMERA" "${app_args[@]}" >"$SESSION_DIR/camera-$tag.log" 2>&1 &
    APP_PID=$!

    local geometry=""
    local waited=0
    while (( waited < WINDOW_TIMEOUT * 5 )); do
        if ! kill -0 "$APP_PID" 2>/dev/null; then
            cat "$SESSION_DIR/camera-$tag.log" >&2
            warn "$tag: app exited before its window appeared"
            APP_PID=""
            return 1
        fi
        geometry="$(window_geometry "$APP_PID")"
        [[ -n "$geometry" ]] && break
        sleep 0.2
        # Not `(( waited++ ))`: post-increment from 0 evaluates to 0, which makes
        # the arithmetic command exit non-zero and would abort the script the
        # moment this function is called outside an `if` condition.
        waited=$(( waited + 1 ))
    done
    if [[ -z "$geometry" ]]; then
        warn "$tag: window never appeared"
        kill "$APP_PID" 2>/dev/null || true
        APP_PID=""
        return 1
    fi

    # sway owns the window size here: the app opens at its own default and is
    # resized to what the shot asks for. A shot at the wrong size is a diff
    # against every committed pixel.
    if ! swaymsg "[pid=$APP_PID] resize set width ${width}px height ${height}px" >/dev/null; then
        warn "$tag: could not resize the window to $window"
        kill_app
        return 1
    fi
    swaymsg "[pid=$APP_PID] focus" >/dev/null || true

    # Wait for the app to log that its first preview frame is set, then proceed
    # the moment it is genuinely ready rather than after a fixed guess. Input
    # sent before the first frame renders is dropped, so this gate matters.
    local ready_waited=0
    until grep -q "$PREVIEW_READY_MARKER" "$SESSION_DIR/camera-$tag.log" 2>/dev/null; do
        if ! kill -0 "$APP_PID" 2>/dev/null; then
            cat "$SESSION_DIR/camera-$tag.log" >&2
            warn "$tag: app exited before signalling readiness"
            APP_PID=""
            return 1
        fi
        # Tenths of a second; bail out to the fixed timeout if the marker never
        # comes so one stuck shot cannot hang the whole run.
        if (( ready_waited >= READY_TIMEOUT * 10 )); then
            warn "$tag: preview never signalled ready within ${READY_TIMEOUT}s; capturing anyway"
            break
        fi
        sleep 0.1
        ready_waited=$(( ready_waited + 1 ))
    done

    local step
    IFS=',' read -ra step_list <<<"$steps"
    for step in "${step_list[@]}"; do
        [[ -z "$step" ]] && continue
        # Handled before launch rather than by a keystroke; see seed_gallery.
        [[ "$step" == "seed-photo" ]] && continue
        # Resolved into a launch flag above, not a keystroke.
        [[ "$step" == "spoof-recording" ]] && continue
        local keys="${STEP_KEYS[$step]:-}"
        [[ -n "$keys" ]] || { warn "$tag: unknown step '$step'"; continue; }
        # A dropped keystroke leaves the app in the wrong UI state, which still
        # screenshots perfectly happily, so a failure here has to be loud.
        # shellcheck disable=SC2086: keys is a deliberate argument list
        if ! wtype -s "$KEY_WARMUP_MS" $keys 2>"$SESSION_DIR/wtype.err"; then
            warn "$tag: step '$step' failed: $(cat "$SESSION_DIR/wtype.err")"
            keep_log "$tag"
            kill_app
            return 1
        fi
        sleep "$STEP_PAUSE"
    done

    # settle_ms is spliced into an awk program, and shots.conf will come from a
    # pull request branch once this runs in CI. Rejecting anything that is not
    # digits keeps it from being code, and also catches an empty column, which
    # would make awk print nothing and `sleep ""` fail.
    if [[ ! "$settle_ms" =~ ^[0-9]+$ ]]; then
        warn "$tag: settle_ms must be a whole number of milliseconds, got '$settle_ms'"
        keep_log "$tag"
        kill_app
        return 1
    fi
    sleep "$(awk -v ms="$settle_ms" 'BEGIN { print ms / 1000 }')"

    geometry="$(window_geometry "$APP_PID")"
    if [[ -z "$geometry" ]]; then
        warn "$tag: window disappeared before capture"
        keep_log "$tag"
        kill_app
        return 1
    fi

    if ! grim -g "$geometry" "$out"; then
        warn "$tag: grim failed to capture $geometry"
        keep_log "$tag"
        kill_app
        return 1
    fi

    kill_app

    keep_log "$tag"

    # A shot at the wrong size differs from the committed one in every pixel, so
    # this is a failure rather than a warning. An unreadable capture is one too:
    # letting it through is how an empty or truncated PNG reaches the sync step.
    local actual
    actual="$(identify -format '%wx%h' "$out" 2>/dev/null || echo "")"
    if [[ -z "$actual" ]]; then
        warn "$tag: cannot read back the captured image"
        return 1
    fi
    if [[ "$actual" != "$window" ]]; then
        warn "$tag: captured ${actual}, expected ${window}"
        return 1
    fi
    log "$tag: written to $out"
    # Explicit, so the shot's result never becomes the exit status of whatever
    # bookkeeping happens to run last.
    return 0
}

# Every shot is taken in each appearance combination the app offers, as
# "<theme>|<overlay_effect>|<variant name>".
#
# The first entry is the one published to Flathub: it is captured as the plain
# `preview-0NN.png` that metainfo.xml points at, and the rest land under
# `variants/` for the gallery in preview/README.md.
VARIANTS=(
    "Dark|Translucent|dark-translucent"
    "Dark|Frosted|dark-frosted"
    "Dark|Off|dark-off"
    "Light|Frosted|light-frosted"
    "Light|Translucent|light-translucent"
    "Light|Off|light-off"
)

# The combination published to Flathub as preview-0NN.png: the app's default
# appearance (Translucent overlay), so the store shot matches a fresh install.
PUBLISHED_VARIANT="dark-translucent"

# The language the published shots and the appearance matrix are taken in.
BASE_LOCALE="en"

# Every language with a translation, discovered from the i18n tree rather than
# listed here so a new one starts appearing without touching this script.
#
# Partially translated languages are included deliberately: their screenshots
# are then honest about how much of the UI is translated so far, which is a
# nudge to finish them rather than something to hide.
discover_locales() {
    local dir locale
    for dir in "$REPO_DIR"/i18n/*/; do
        locale="$(basename "$dir")"
        [[ -f "$dir/camera.ftl" ]] || continue
        [[ "$locale" == "$BASE_LOCALE" ]] && continue
        printf '%s\n' "$locale"
    done
}

selected="${SHOTS:-}"
selected_variants="${VARIANTS_FILTER:-}"
failed=0

mkdir -p "$OUT_DIR/variants"

# Non-base languages get the published appearance only. The appearance matrix
# exists to show off the theme and overlay settings, which look identical
# whatever language the labels are in, and capturing it per language would multiply
# the run by ten to produce sixty near-duplicate images of the same two settings.
LOCALES=("$BASE_LOCALE")
if [[ "${LOCALES_FILTER:-}" == "none" ]]; then
    log "Skipping localized shots (LOCALES_FILTER=none)"
else
    while read -r locale; do
        [[ -z "$locale" ]] && continue
        if [[ -n "${LOCALES_FILTER:-}" && ",${LOCALES_FILTER}," != *",$locale,"* ]]; then
            continue
        fi
        LOCALES+=("$locale")
    done < <(discover_locales)
fi
log "Languages: ${LOCALES[*]}"

for locale in "${LOCALES[@]}"; do
    # The app resolves its language through i18n_embed's DesktopLanguageRequester,
    # which reads these environment variables directly, so no glibc locale needs to
    # be generated in the image. LANGUAGE alone is enough for the lookup; LANG is
    # set too so anything reading it agrees.
    export LANGUAGE="$locale"
    export LANG="$locale.UTF-8"

    if [[ "$locale" == "$BASE_LOCALE" ]]; then
        locale_variants=("${VARIANTS[@]}")
        locale_dir=""
    else
        locale_variants=()
        for variant in "${VARIANTS[@]}"; do
            [[ "$variant" == *"|$PUBLISHED_VARIANT" ]] && locale_variants+=("$variant")
        done
        locale_dir="$OUT_DIR/locales/$locale"
        mkdir -p "$locale_dir"
        locale_differs=0
    fi

    while IFS='|' read -r num source_file window steps settle_ms shot_config description; do
        [[ -z "${num// }" || "${num:0:1}" == "#" ]] && continue
        if [[ -n "$selected" && ",$selected," != *",$num,"* ]]; then
            continue
        fi

        for variant in "${locale_variants[@]}"; do
            IFS='|' read -r theme overlay name <<<"$variant"
            if [[ -n "$selected_variants" && ",$selected_variants," != *",$name,"* ]]; then
                continue
            fi

            # The published shot is always the same combination, whichever subset
            # of variants this run happens to capture.
            if [[ -n "$locale_dir" ]]; then
                out="$locale_dir/preview-$num.png"
            elif [[ "$name" == "$PUBLISHED_VARIANT" ]]; then
                out="$OUT_DIR/preview-$num.png"
            else
                out="$OUT_DIR/variants/preview-$num-$name.png"
            fi

            if ! capture_shot "$num" "$source_file" "$window" "$steps" "$settle_ms" \
                    "$shot_config" "$description" "$theme" "$overlay" "$out"; then
                failed=$(( failed + 1 ))
                [[ "${KEEP_GOING:-0}" == "1" ]] || die "preview-$num ($name, $locale) failed"
                continue
            fi

            # Nothing so far proves the app actually switched language:
            # src/i18n.rs only logs when language selection fails, so an
            # unresolvable tag leaves the UI in English and the shot is filed
            # under that language anyway. Compare against the English capture of
            # the same shot; a language that differs nowhere never applied.
            if [[ -n "$locale_dir" && -f "$OUT_DIR/preview-$num.png" ]]; then
                if ! images_identical "$OUT_DIR/preview-$num.png" "$out"; then
                    locale_differs=1
                fi
            fi
        done
    done <"$SHOTS_CONF"

    # Reported per language rather than per shot: a partly translated language
    # legitimately matches English on shots whose strings it has not translated
    # yet, so only matching on *every* shot means the tag itself did not resolve.
    if [[ -n "$locale_dir" ]] && (( locale_differs == 0 )); then
        warn "$locale: every shot is identical to English." \
            "The language tag probably did not resolve (check i18n/$locale)."
        failed=$(( failed + 1 ))
        [[ "${KEEP_GOING:-0}" == "1" ]] || die "$locale did not apply"
    fi
done

if (( failed > 0 )); then
    die "$failed shot(s) failed"
fi

log "All previews captured into $OUT_DIR"
