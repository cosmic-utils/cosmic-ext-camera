#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-only
# Native Alpine-only bootstrap, run as root after the builder/key setup.
# Installs real signed libcamera/libcamera-dev APKs, not version providers.
set -eu

test "$(id -u)" = 0
recipe_dir=$(CDPATH='' cd -- "$(dirname -- "$0")/libcamera" && pwd)
apk add --no-cache alpine-sdk sudo git atools
id builder >/dev/null
workdir=$(mktemp -d /home/builder/libcamera-apk.XXXXXX)
trap 'rm -rf "$workdir"' EXIT HUP INT TERM

git clone --depth 1 --branch v0.8.0 \
    https://git.libcamera.org/libcamera/libcamera.git "$workdir/source"
test "$(git -C "$workdir/source" rev-parse HEAD)" = df8e348d64b4932a0676256ed24f0da3ee9438bc
mkdir -p "$workdir/aports/libcamera"
cp "$recipe_dir/APKBUILD" "$workdir/aports/libcamera/APKBUILD"
# An uncompressed Git archive avoids host/BusyBox gzip implementation drift.
git -C "$workdir/source" archive --format=tar --prefix=libcamera-0.8.0/ \
    df8e348d64b4932a0676256ed24f0da3ee9438bc \
    > "$workdir/aports/libcamera/libcamera-0.8.0.tar"
# abuild verifies the independently pinned SHA-512 in the recipe.
apkbuild-lint "$workdir/aports/libcamera/APKBUILD"
chown -R builder:builder "$workdir"
cd "$workdir/aports/libcamera"
sudo -Hu builder env JOBS="${JOBS:-4}" REPODEST=/home/builder/packages abuild -r
arch=$(apk --print-arch)
apk verify \
    "/home/builder/packages/aports/$arch/libcamera-0.8.0-r0.apk" \
    "/home/builder/packages/aports/$arch/libcamera-dev-0.8.0-r0.apk"
apk add --no-cache \
    "/home/builder/packages/aports/$arch/libcamera-0.8.0-r0.apk" \
    "/home/builder/packages/aports/$arch/libcamera-dev-0.8.0-r0.apk"
test "$(pkg-config --modversion libcamera)" = 0.8.0
test "$(pkg-config --modversion libcamera-base)" = 0.8.0
apk add --simulate 'libcamera>=0.8.0' 'libcamera-dev>=0.8.0'
