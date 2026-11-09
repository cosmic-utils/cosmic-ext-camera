# SPDX-License-Identifier: GPL-3.0-only

name := 'camera'
# Alpine package name. Differs from the binary: the distro package is
# namespaced under the cosmic-ext- prefix, while the binary it installs
# stays `camera`. Must match pkgname in .github/APKBUILD*.template.
apk-pkgname := 'cosmic-ext-camera'
export APPID := 'io.github.cosmic_utils.camera'

rootdir := ''
prefix := '/usr'

base-dir := absolute_path(clean(rootdir / prefix))

export INSTALL_DIR := base-dir / 'share'

cargo-target-dir := env('CARGO_TARGET_DIR', 'target')
bin-src := cargo-target-dir / 'release' / name
bin-dst := base-dir / 'bin' / name

desktop := APPID + '.desktop'
desktop-src := 'resources' / desktop
desktop-dst := clean(rootdir / prefix) / 'share' / 'applications' / desktop

metainfo := APPID + '.metainfo.xml'
metainfo-src := 'resources' / metainfo
metainfo-dst := clean(rootdir / prefix) / 'share' / 'metainfo' / metainfo

icons-src := 'resources' / 'icons' / 'hicolor'
icons-dst := clean(rootdir / prefix) / 'share' / 'icons' / 'hicolor'

# Default recipe which runs `just build-release`
default: build-release

# ============================================================================
# Building
# ============================================================================

# Compiles with debug profile
build-debug *args:
    cargo build {{args}}

# Compiles with release profile
build-release *args: (build-debug '--release' args)

# Compiles release profile with vendored dependencies
build-vendored *args: vendor-extract (build-release '--frozen --offline' args)

# ============================================================================
# Code quality
# ============================================================================

# Runs cargo check
cargo-check *args:
    cargo check --all-features {{args}}

# Runs clippy (used in CI - default warnings only)
clippy *args:
    cargo clippy --all-features {{args}} -- -D warnings

# Runs clippy with pedantic warnings (for development)
clippy-pedantic *args:
    cargo clippy --all-features {{args}} -- -W clippy::pedantic

# Runs clippy with JSON message format
clippy-json: (clippy '--message-format=json')

# Format code
fmt:
    cargo fmt

# Check code formatting
fmt-check:
    cargo fmt --check

# Run tests
test *args:
    cargo test {{args}}

# Run all checks (format, clippy, cargo check, test)
check: fmt-check clippy cargo-check test

# ============================================================================
# Development
# ============================================================================

# Developer target: format and run
dev *args:
    cargo fmt
    just run {{args}}

# Run with debug logs
run *args:
    env RUST_LOG=camera=info RUST_BACKTRACE=full cargo run --profile release-fast {{args}}

# Run with verbose debug logs
run-debug *args:
    env RUST_LOG=camera=debug,info RUST_BACKTRACE=full cargo run --profile release-fast {{args}}

# ============================================================================
# Resource generation
# ============================================================================

# Regenerate every derived file that is committed to the repository
#
# Deliberately does NOT retake the preview screenshots. This recipe runs in CI
# (regenerate.yml) on a runner with no /dev/udmabuf and no 30 minute budget for
# it; generate-previews is a separate, explicitly invoked recipe.
generate: generate-metadata flatpak-cargo-sources

# Write translations from i18n/*/camera.ftl into the desktop and metainfo files
generate-metadata:
    python3 scripts/gen-metadata.py

# Check the desktop and metainfo files against their specifications
#
# Offline on purpose. The metainfo screenshots are absolute raw.githubusercontent
# URLs pointing at main, so appstreamcli's network check cannot pass on a pull
# request that adds a screenshot: the image only exists once the branch is
# merged. Left on, every previews PR would fail validation until merge, which is
# a check that structurally cannot be satisfied rather than a real problem.
# Use validate-metadata-urls to check reachability where it does make sense.
validate-metadata:
    desktop-file-validate {{desktop-src}}
    appstreamcli validate --no-net {{metainfo-src}}

# Also check that every screenshot URL actually resolves
#
# Only meaningful once the images are on main, so this belongs on a schedule or
# a post-merge run, never on a pull request.
validate-metadata-urls:
    appstreamcli validate {{metainfo-src}}

# Retake the preview screenshots in a pinned container, then refresh the metadata
#
# Takes roughly 30 minutes for the full matrix and needs /dev/udmabuf on the
# host. The metadata pass afterwards is what turns newly captured languages into
# gallery pages and localized <image> entries, so the two belong together.
# Narrow a run with SHOTS, VARIANTS_FILTER or LOCALES_FILTER, for example:
#   SHOTS=001 LOCALES_FILTER=none just generate-previews
generate-previews *args: && generate-metadata
    preview/generate-previews.sh {{args}}

# Generate PNG icons from the scalable SVG and update desktop file
generate-icons:
    #!/usr/bin/env bash
    set -euo pipefail
    SVG="{{icons-src}}/scalable/apps/{{APPID}}.svg"
    DESKTOP="{{desktop-src}}"
    if [ ! -f "$SVG" ]; then
        echo "Error: Scalable SVG not found at $SVG"
        exit 1
    fi
    # Generate PNG icons for each size
    for size in 16 24 32 48 64 128 256 512 1024; do
        DIR="{{icons-src}}/${size}x${size}/apps"
        mkdir -p "$DIR"
        magick -background none -density 384 "$SVG" -resize ${size}x${size} "$DIR/{{APPID}}.png"
        echo "Generated ${size}x${size} icon"
    done
    # Update Icon= line in desktop file
    if [ -f "$DESKTOP" ]; then
        sed -i 's/^Icon=.*/Icon={{APPID}}/' "$DESKTOP"
        echo "Updated desktop file icon to {{APPID}}"
    else
        echo "Warning: Desktop file not found at $DESKTOP"
    fi
    echo "All icons generated successfully!"

# ============================================================================
# Cleaning
# ============================================================================

# Runs `cargo clean`
clean:
    cargo clean

# Removes vendored dependencies
clean-vendor:
    rm -rf .cargo vendor vendor.tar

# `cargo clean` and removes vendored dependencies
clean-dist: clean clean-vendor

# Installs files
install:
    install -Dm0755 {{bin-src}} {{bin-dst}}
    install -Dm0644 {{desktop-src}} {{desktop-dst}}
    install -Dm0644 {{metainfo-src}} {{metainfo-dst}}
    install -Dm0644 "{{icons-src}}/scalable/apps/{{APPID}}.svg" "{{icons-dst}}/scalable/apps/{{APPID}}.svg"
    for size in 16x16 24x24 32x32 48x48 64x64 128x128 256x256 512x512 1024x1024; do \
        install -Dm0644 "{{icons-src}}/$size/apps/{{APPID}}.png" "{{icons-dst}}/$size/apps/{{APPID}}.png"; \
    done

# Uninstalls installed files
uninstall:
    rm -f {{bin-dst}} {{desktop-dst}} {{metainfo-dst}}
    rm -f "{{icons-dst}}/scalable/apps/{{APPID}}.svg"
    for size in 16x16 24x24 32x32 48x48 64x64 128x128 256x256 512x512 1024x1024; do \
        rm -f "{{icons-dst}}/$size/apps/{{APPID}}.png"; \
    done

# Vendor dependencies locally
vendor:
    #!/usr/bin/env bash
    mkdir -p .cargo
    cargo vendor --sync Cargo.toml | head -n -1 > .cargo/config.toml
    echo 'directory = "vendor"' >> .cargo/config.toml
    tar pcf vendor.tar .cargo vendor
    rm -rf .cargo vendor

# Extracts vendored dependencies
vendor-extract:
    rm -rf vendor
    tar pxf vendor.tar

# ============================================================================
# Version management
# ============================================================================

# Get the current version from git tags
get-version:
    #!/usr/bin/env bash
    version=$(git describe --tags --always --match "v*" 2>/dev/null || echo "unknown")
    version="${version#v}"
    # Transform: 0.1.0-5-gabcdef1 -> 0.1.0-dirty-abcdef1
    if [[ "$version" == *-*-g* ]]; then
        base=$(echo "$version" | sed 's/-[0-9]*-g.*//')
        hash=$(echo "$version" | sed 's/.*-g//')
        version="${base}-dirty-${hash}"
    fi
    echo "$version"

# Get runtime version from Flatpak manifest
[private]
flatpak-runtime-version:
    @grep 'runtime-version:' {{APPID}}.yml | sed "s/.*runtime-version: *['\"]\\?\\([^'\"]*\\)['\"]\\?/\\1/"

# ============================================================================
# Flatpak recipes
# ============================================================================

# Generate cargo-sources.json for Flatpak
flatpak-cargo-sources:
    #!/usr/bin/env bash
    set -euo pipefail
    generator_sha256=0a2db6be87d75910facef28ab46d4d6460802e8419ab850d0caa6a364d26b380
    if ! printf '%s  %s\n' "$generator_sha256" flatpak-cargo-generator.py | sha256sum --check --status 2>/dev/null; then
        generator_tmp="$(mktemp)"
        trap 'rm -f "$generator_tmp"' EXIT
        curl -fLo "$generator_tmp" https://raw.githubusercontent.com/flatpak/flatpak-builder-tools/41c20aa10819cdb2a4f3ca171758a96d1955c018/cargo/flatpak-cargo-generator.py
        printf '%s  %s\n' "$generator_sha256" "$generator_tmp" | sha256sum --check --status
        mv "$generator_tmp" flatpak-cargo-generator.py
    fi
    if command -v uv >/dev/null 2>&1; then
        uv run --no-project --with-requirements requirements-build.txt python3 flatpak-cargo-generator.py Cargo.lock -o cargo-sources.json
    elif [ -x "$HOME/.hermes/bin/uv" ]; then
        "$HOME/.hermes/bin/uv" run --no-project --with-requirements requirements-build.txt python3 flatpak-cargo-generator.py Cargo.lock -o cargo-sources.json
    else
        # Recover a virtualenv left behind by a removed or upgraded interpreter.
        if [ ! -d .flatpak-venv ] || ! .flatpak-venv/bin/python3 --version &>/dev/null; then
            rm -rf .flatpak-venv
            python3 -m venv .flatpak-venv
        fi
        .flatpak-venv/bin/python -m pip install --quiet --requirement requirements-build.txt
        .flatpak-venv/bin/python flatpak-cargo-generator.py Cargo.lock -o cargo-sources.json
    fi

# Build and install Flatpak locally
flatpak-build: flatpak-cargo-sources
    #!/usr/bin/env bash
    set -euo pipefail
    trap 'rm -f .flatpak-version' EXIT
    echo "Building Flatpak..."
    just get-version > .flatpak-version
    flatpak-builder --user --install --force-clean build-dir {{APPID}}.yml
    echo "Flatpak built and installed!"

# Deliberately does not regenerate cargo-sources.json first. Flathub builds from
# the committed file, so building from it here too means CI fails loudly when it
# has drifted from Cargo.lock, instead of masking the drift by overwriting it.
# Run `just flatpak-cargo-sources` yourself after changing dependencies.

# Build Flatpak bundle for distribution (optionally specify arch)
flatpak-bundle arch="":
    #!/usr/bin/env bash
    set -euo pipefail
    arch="{{arch}}"
    if [ -z "$arch" ]; then
        arch=$(uname -m)
    fi
    [ "$arch" = "x86_64" ] || [ "$arch" = "aarch64" ] || { echo "Unsupported Flatpak architecture: $arch"; exit 1; }
    echo "Building Flatpak bundle for $arch..."
    just get-version > .flatpak-version
    flatpak-builder --repo=repo --force-clean --arch=$arch build-dir {{APPID}}.yml
    flatpak build-bundle repo {{name}}-$arch.flatpak {{APPID}} --arch=$arch
    rm -f .flatpak-version
    [ -f "{{name}}-$arch.flatpak" ] || { echo "Error: Flatpak bundle was not created!"; exit 1; }
    echo "Flatpak bundle created: {{name}}-$arch.flatpak"
    ls -la {{name}}-$arch.flatpak

# Run the installed Flatpak
flatpak-run:
    flatpak run {{APPID}}

# Uninstall all Flatpak components (app, debug, locale)
flatpak-uninstall:
    #!/usr/bin/env bash
    echo "Uninstalling all {{APPID}} Flatpak components..."
    flatpak uninstall --user -y {{APPID}} 2>/dev/null || true
    flatpak uninstall --user -y {{APPID}}.Debug 2>/dev/null || true
    flatpak uninstall --user -y {{APPID}}.Locale 2>/dev/null || true
    echo "Flatpak uninstalled!"

# Full Flatpak install: uninstall old, install deps if needed, build, and install
flatpak-install:
    #!/usr/bin/env bash
    set -e
    echo "=== Full Flatpak Install ==="
    just flatpak-uninstall
    RUNTIME_VERSION=$(just flatpak-runtime-version)
    DEPS_MISSING=false
    flatpak info org.freedesktop.Sdk//${RUNTIME_VERSION} &>/dev/null || DEPS_MISSING=true
    flatpak info org.freedesktop.Platform//${RUNTIME_VERSION} &>/dev/null || DEPS_MISSING=true
    flatpak info org.freedesktop.Sdk.Extension.rust-stable//${RUNTIME_VERSION} &>/dev/null || DEPS_MISSING=true
    flatpak info org.freedesktop.Sdk.Extension.llvm23//${RUNTIME_VERSION} &>/dev/null || DEPS_MISSING=true
    flatpak info com.system76.Cosmic.BaseApp//stable &>/dev/null || DEPS_MISSING=true
    if [ "$DEPS_MISSING" = true ]; then
        echo "Flatpak dependencies missing, installing..."
        just flatpak-deps
    else
        echo "Flatpak dependencies already installed."
    fi
    just flatpak-build
    echo "=== Flatpak installation complete! ==="
    echo "Run with: just flatpak-run"

# Clean Flatpak build artifacts
flatpak-clean:
    rm -rf build-dir .flatpak-builder repo cargo-sources.json {{name}}-*.flatpak .flatpak-venv .flatpak-version

# Install Flatpak dependencies (runtime and SDK)
flatpak-deps arch="":
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Installing Flatpak dependencies..."
    command -v flatpak &> /dev/null || { echo "Error: flatpak not found!"; exit 1; }
    RUNTIME_VERSION=$(just flatpak-runtime-version)
    arch="{{arch}}"
    if [ -z "$arch" ]; then
        arch=$(uname -m)
    fi
    [ "$arch" = "x86_64" ] || [ "$arch" = "aarch64" ] || { echo "Unsupported Flatpak architecture: $arch"; exit 1; }
    ARCH_FLAG="--arch=$arch"
    echo "Runtime version: $RUNTIME_VERSION"
    sudo flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
    sudo flatpak install -y --noninteractive flathub org.freedesktop.Platform//${RUNTIME_VERSION} $ARCH_FLAG
    sudo flatpak install -y --noninteractive flathub org.freedesktop.Sdk//${RUNTIME_VERSION} $ARCH_FLAG
    sudo flatpak install -y --noninteractive flathub org.freedesktop.Sdk.Extension.rust-stable//${RUNTIME_VERSION} $ARCH_FLAG
    sudo flatpak install -y --noninteractive flathub org.freedesktop.Sdk.Extension.llvm23//${RUNTIME_VERSION} $ARCH_FLAG
    sudo flatpak install -y --noninteractive flathub com.system76.Cosmic.BaseApp//stable $ARCH_FLAG
    echo "Flatpak dependencies installed!"

# ============================================================================
# Alpine APK recipes
# ============================================================================

# Build Alpine APK package (optionally specify arch, e.g. aarch64)
# Cross-compiles first, then packages the pre-built binary in an Alpine container.
apk-build arch="":
    #!/usr/bin/env bash
    set -euo pipefail
    arch="{{arch}}"
    if [ -z "$arch" ]; then
        arch=$(uname -m)
    fi

    # Map arch to Rust target triple
    case "$arch" in
        aarch64) rust_target="aarch64-unknown-linux-musl" ;;
        x86_64)  rust_target="x86_64-unknown-linux-musl" ;;
        *)       echo "Unsupported arch: $arch"; exit 1 ;;
    esac

    case "$(uname -m)" in
        x86_64)  host_arch="amd64" ;;
        aarch64) host_arch="arm64" ;;
        *)       host_arch="$(uname -m)" ;;
    esac

    echo "Cross-compiling for $rust_target (release-fast)..."
    # Pin the toolchain image to the host platform. It and the packaging image
    # below are both FROM alpine:edge, and the container engine keeps a single
    # image per tag: packaging pulls the target arch, leaving a foreign-arch
    # alpine:edge behind. Without --platform the next cross build reuses that
    # with only a warning, silently producing an emulated same-arch toolchain
    # instead of a cross one.
    CROSS_BUILD_OPTS="--platform linux/$host_arch" \
        cross build --target "$rust_target" --profile release-fast

    binary="target/$rust_target/release-fast/{{name}}"
    [ -f "$binary" ] || { echo "Error: binary not found at $binary"; exit 1; }

    echo "Packaging APK for $arch..."
    platform="linux/$arch"
    [ "$arch" = "x86_64" ] && platform="linux/amd64"
    image_tag="{{name}}-apk-$arch"
    VERSION=$(grep "^version" Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')

    mkdir -p apk-out
    podman build --platform "$platform" -t "$image_tag" -f docker/Dockerfile.apk .

    podman run --rm --platform "$platform" \
        -v "$(pwd)":/src:ro \
        -v "$(pwd)/apk-out":/out \
        -v "$(pwd)/$binary":/prebuilt/{{name}}:ro \
        "$image_tag" sh -c '
        set -e
        VERSION="'"$VERSION"'"

        mkdir -p /home/builder/apkbuild
        sed "s/@@VERSION@@/$VERSION/g" /src/.github/APKBUILD.prebuilt.template \
            > /home/builder/apkbuild/APKBUILD
        chown -R builder:builder /home/builder/apkbuild

        cd /home/builder/apkbuild
        # Pin REPODEST: abuild defaults to $XDG_DATA_HOME/abuild these days,
        # older versions used $HOME/packages
        sudo -Hu builder env REPODEST=/home/builder/packages abuild -r

        mkdir -p /out
        find /home/builder/packages -name "{{apk-pkgname}}-*.apk" ! -name "*-doc-*" ! -name "*-dev-*" -exec cp {} /out/ \;
        echo "APK built successfully:"
        ls -la /out/
    '
    echo "Output in apk-out/"
    ls -la apk-out/

# Clean APK build artifacts
apk-clean:
    rm -rf apk-out

# Full clean (cargo + vendor + flatpak + apk)
clean-all: clean clean-vendor flatpak-clean apk-clean
