#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-only
# Build the same pinned libcamera ABI for GNU and musl targets.
set -eu

triplet=$1
sysroot=${2:-/}
libdir=lib/$triplet
cc=${triplet}-gcc
cxx=${triplet}-g++
ar=${triplet}-ar
strip=${triplet}-strip
case "$triplet" in
    aarch64-linux-gnu) cpu_family=aarch64; cpu=aarch64 ;;
    arm-linux-gnueabihf) cpu_family=arm; cpu=armv7 ;;
    riscv64-linux-gnu) cpu_family=riscv64; cpu=riscv64 ;;
    x86_64-linux-gnu) cpu_family=x86_64; cpu=x86_64 ;;
    x86_64-linux-musl)
        cpu_family=x86_64; cpu=x86_64; libdir=lib
        cc=gcc; cxx=g++; ar='ar'; strip='strip'
        ;;
    aarch64-linux-musl)
        cpu_family=aarch64; cpu=aarch64; libdir=lib
        cc=aarch64-linux-musl-clang; cxx=aarch64-linux-musl-clang++
        ar='ar'; strip=llvm-strip
        ;;
    *) printf 'Unsupported libcamera target: %s\n' "$triplet" >&2; exit 1 ;;
esac

workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT HUP INT TERM
git clone --depth 1 --branch v0.8.0 \
    https://git.libcamera.org/libcamera/libcamera.git "$workdir/source"
cd "$workdir/source"
test "$(git rev-parse HEAD)" = df8e348d64b4932a0676256ed24f0da3ee9438bc

cat > "$workdir/cross.ini" <<EOF
[binaries]
c = '$cc'
cpp = '$cxx'
ar = '$ar'
strip = '$strip'
pkg-config = 'pkg-config'

[host_machine]
system = 'linux'
cpu_family = '$cpu_family'
cpu = '$cpu'
endian = 'little'

[properties]
sys_root = '$sysroot'
pkg_config_libdir = ['$sysroot/usr/$libdir/pkgconfig', '$sysroot/usr/share/pkgconfig']
EOF

meson setup build --prefix=/usr --libdir="$libdir" \
    --cross-file "$workdir/cross.ini" \
    -Dpipelines=uvcvideo,simple -Dipas=softisp \
    -Dgstreamer=disabled -Dtest=false -Dcam=disabled -Dqcam=disabled \
    -Dlc-compliance=disabled -Dtracing=disabled -Ddocumentation=disabled \
    -Dpycamera=disabled
ninja -C build
DESTDIR="$sysroot" ninja -C build install
# Verify the target metadata, not any host libcamera package.
test "$(PKG_CONFIG_PATH='' PKG_CONFIG_SYSROOT_DIR="$sysroot" \
    PKG_CONFIG_LIBDIR="$sysroot/usr/$libdir/pkgconfig:$sysroot/usr/share/pkgconfig" \
    pkg-config --modversion libcamera)" = 0.8.0
