#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Ubuntu native builds; callers pass any additional test/runtime packages.
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  pkg-config \
  libdav1d-dev \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  libgstreamer-plugins-bad1.0-dev \
  libwayland-dev \
  libxkbcommon-dev \
  libinput-dev \
  libudev-dev \
  libseat-dev \
  libclang-dev \
  cmake \
  nasm \
  meson \
  ninja-build \
  python3-jinja2 \
  python3-yaml \
  python3-ply \
  libyaml-dev "$@"

# Keep the exact release and commit aligned with the Flatpak manifest.
git clone --depth 1 --branch v0.8.0 https://git.libcamera.org/libcamera/libcamera.git /tmp/libcamera
cd /tmp/libcamera
test "$(git rev-parse HEAD)" = df8e348d64b4932a0676256ed24f0da3ee9438bc
meson setup build -Dprefix=/usr -Dpipelines=uvcvideo,simple -Dipas=softisp -Dgstreamer=disabled -Dtest=false -Dcam=disabled -Dqcam=disabled -Dlc-compliance=disabled -Dtracing=disabled -Ddocumentation=disabled -Dpycamera=disabled
ninja -C build
sudo ninja -C build install
sudo ldconfig
test "$(pkg-config --modversion libcamera)" = 0.8.0
