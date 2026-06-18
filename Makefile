# Zero build orchestration.
#
# Stage 0 uses bootloader 0.11. The kernel is built as a standalone
# no_std ELF; the `boot` crate then wraps it into BIOS and UEFI disk
# images.
#
# The kernel is built in **release** mode, not dev. Debug builds
# carry enough information that by Stage 3 the ELF LOAD segments
# collide with bootloader 0.11's identity-mapped region, which
# triggers `PageAlreadyMapped` during boot setup. Release builds
# are an order of magnitude smaller and avoid the collision.
# Debugging the kernel is done via QEMU+GDB, not println, so the
# lost debug info is not a practical cost.
# See ADR-028 patch v4 for full analysis.

ROOT := $(shell pwd)
KERNEL_ELF := $(ROOT)/kernel/target/x86_64-unknown-none/release/zero-kernel
IMAGES_DIR := $(ROOT)/target/zero-images
BIOS_IMG := $(IMAGES_DIR)/zero-bios.img
UEFI_IMG := $(IMAGES_DIR)/zero-uefi.img
MODEL_PATH := $(ROOT)/kernel/programs/Qwen_Qwen3-1.7B-Q4_K_M.gguf
CHERRY_FEATURES := cherry-net,avx512-acceleration
CHERRY_SMP_DEBUG_FEATURES := $(CHERRY_FEATURES),cherry-smp-debug
ZERO_CONTROL_PLANE_FEATURES := $(CHERRY_FEATURES),zero-control-plane
CHERRY_OPT_LEVEL ?= 3
# UEFI firmware for `make run-uefi`. Default is the Homebrew QEMU path on
# macOS; override for your platform, e.g.
#   make run-uefi OVMF_BIOS=/usr/share/OVMF/OVMF_CODE.fd   # Debian/Ubuntu
OVMF_BIOS ?= /opt/homebrew/share/qemu/edk2-x86_64-code.fd

.PHONY: all kernel image image-bare kernel-cherry image-cherry image-cherry-bare kernel-cherry-smp-debug image-cherry-smp-debug image-cherry-bare-smp-debug kernel-zero-control-plane image-zero-control-plane image-zero-control-plane-bare run run-bare run-uefi build-debug run-aarch64 build-aarch64 clean

all: image

kernel:
	cd kernel && cargo build --release

# Default image build — embeds the GGUF model as a ramdisk if it
# exists on the host. Skips the model gracefully when MODEL_PATH is
# missing so CI / bring-up developers without the GGUF can still
# produce a bootable image.
image: kernel
	cd boot && \
	    ZERO_KERNEL_ELF="$(KERNEL_ELF)" \
	    $(if $(wildcard $(MODEL_PATH)),ZERO_MODEL_PATH="$(MODEL_PATH)",) \
	    cargo run --release

# Bare-metal image without the model — useful for PCIe/HAL bring-up
# on real hardware where the GGUF is too large to ship.
image-bare: kernel
	cd boot && ZERO_KERNEL_ELF="$(KERNEL_ELF)" cargo run --release

# Bare-metal server build — compiles the kernel with the static-IP
# profile (see kernel/src/net/mod.rs) and AVX-512 + SMP multi-core
# boot. The default `kernel` target keeps QEMU networking (10.0.2.15)
# and single-core boot so local dev workflows are untouched.
#
# x86_64 server builds use opt-level=3 — the aarch64 opt-level=1
# constraint (Lessons 7/9/11, STP/HVF) is architecture-specific and does
# not apply on x86_64. opt-level=3 enables aggressive auto-vectorization,
# loop unrolling, and better register allocation for the AVX-512
# intrinsics in kernel/src/arch/x86_64/math/linear.rs. Sacred crates
# already use opt-level=2 via per-package override in kernel/Cargo.toml.
kernel-cherry:
	cd kernel && CARGO_PROFILE_RELEASE_OPT_LEVEL=$(CHERRY_OPT_LEVEL) cargo build --release --features $(CHERRY_FEATURES)

# Cherry Server SMP bring-up build — same as kernel-cherry, but keeps
# the once-per-second KVM heartbeat with AP registration counters.
kernel-cherry-smp-debug:
	cd kernel && CARGO_PROFILE_RELEASE_OPT_LEVEL=$(CHERRY_OPT_LEVEL) cargo build --release --features $(CHERRY_SMP_DEBUG_FEATURES)

# Zero Control Plane build — Cherry networking + AVX-512/SMP, with
# the Stage-11 LLM gate controlled by the Zero remote console.
kernel-zero-control-plane:
	cd kernel && CARGO_PROFILE_RELEASE_OPT_LEVEL=$(CHERRY_OPT_LEVEL) cargo build --release --features $(ZERO_CONTROL_PLANE_FEATURES)

# Cherry Server image with embedded model ramdisk.
image-cherry: kernel-cherry
	cd boot && \
	    ZERO_KERNEL_ELF="$(KERNEL_ELF)" \
	    $(if $(wildcard $(MODEL_PATH)),ZERO_MODEL_PATH="$(MODEL_PATH)",) \
	    cargo run --release

# Cherry Server image without the model — same kernel features as
# image-cherry, but skips the GGUF ramdisk for HW bring-up.
image-cherry-bare: kernel-cherry
	cd boot && ZERO_KERNEL_ELF="$(KERNEL_ELF)" cargo run --release

# Cherry Server image variants with the SMP heartbeat enabled. Use
# these only while validating AP wake-up on real hardware.
image-cherry-smp-debug: kernel-cherry-smp-debug
	cd boot && \
	    ZERO_KERNEL_ELF="$(KERNEL_ELF)" \
	    $(if $(wildcard $(MODEL_PATH)),ZERO_MODEL_PATH="$(MODEL_PATH)",) \
	    cargo run --release

image-cherry-bare-smp-debug: kernel-cherry-smp-debug
	cd boot && ZERO_KERNEL_ELF="$(KERNEL_ELF)" cargo run --release

image-zero-control-plane: kernel-zero-control-plane
	cd boot && \
	    ZERO_KERNEL_ELF="$(KERNEL_ELF)" \
	    $(if $(wildcard $(MODEL_PATH)),ZERO_MODEL_PATH="$(MODEL_PATH)",) \
	    cargo run --release

image-zero-control-plane-bare: kernel-zero-control-plane
	cd boot && ZERO_KERNEL_ELF="$(KERNEL_ELF)" cargo run --release

# Legacy QEMU run — model is delivered via `-device loader`. Kept for
# back-compat with the dev workflow that predates ramdisk loading.
run: image-bare
	qemu-system-x86_64 \
		-drive format=raw,file="$(BIOS_IMG)" \
		-serial stdio \
		-display none \
		-m 8G \
		-device "loader,file=$(MODEL_PATH),addr=0x100000000,force-raw=on"

# Production-shape QEMU run — model travels with the image as a
# ramdisk. Mirrors how the boot image runs off a USB stick or IPMI
# virtual media on bare metal. Display window shows the kernel's
# GOP/VBE framebuffer console (same path the BMC HTML5 KVM streams);
# serial stays on stdio for log capture.
run-bare: image
	qemu-system-x86_64 \
		-drive format=raw,file="$(BIOS_IMG)" \
		-serial stdio \
		-vga std \
		-m 8G

run-uefi: image
	qemu-system-x86_64 \
		-bios "$(OVMF_BIOS)" \
		-drive format=raw,file="$(UEFI_IMG)" \
		-serial stdio \
		-vga std \
		-m 8G

# Production-shape QEMU run with networking — Stage 10.8 net stack
# binds to the emulated e1000 NIC and exposes the UDP shell (port
# 9999) and TCP shell (port 2222) on the guest's static IP. Host
# port-forwards land both surfaces on localhost: `nc -u 127.0.0.1
# 9999` reaches the UDP shell, `nc 127.0.0.1 2222` reaches the TCP
# shell.
run-bare-net: image
	qemu-system-x86_64 \
		-drive format=raw,file="$(BIOS_IMG)" \
		-serial stdio \
		-vga std \
		-m 8G \
		-netdev user,id=net0,hostfwd=tcp::2222-:2222,hostfwd=udp::9999-:9999 \
		-device e1000,netdev=net0

# Block accidental debug builds.
build-debug:
	@echo "ERROR: Debug builds collide with bootloader identity-mapping (PageAlreadyMapped)."
	@echo "       Use 'make kernel' (release) instead."
	@exit 1

# aarch64 build + run targets (Sub-MP-D2b).
# ELF → flat binary via objcopy for Linux arm64 boot protocol
# compliance: QEMU -kernel with flat binary guarantees x0=DTB.
# Per Sub-MP-D1 boot-strategy-decision.md.
AARCH64_ELF := $(ROOT)/kernel/target/aarch64-unknown-none/release/zero-kernel
AARCH64_BIN := $(AARCH64_ELF).bin
AARCH64_FEATURES ?= neon-acceleration

build-aarch64:
	cd kernel && cargo build --target aarch64-unknown-none --release --features $(AARCH64_FEATURES)
	rust-objcopy -O binary "$(AARCH64_ELF)" "$(AARCH64_BIN)"

run-aarch64: build-aarch64
	qemu-system-aarch64 \
		-machine virt \
		-cpu host \
		-accel hvf \
		-m 8G \
		-nographic \
		-serial mon:stdio \
		-kernel "$(AARCH64_BIN)" \
		-initrd "$(MODEL_PATH)"

run-aarch64-tcg: build-aarch64
	qemu-system-aarch64 \
		-machine virt \
		-cpu cortex-a57 \
		-m 8G \
		-nographic \
		-serial mon:stdio \
		-kernel "$(AARCH64_BIN)" \
		-initrd "$(MODEL_PATH)"

# Sub-MP-F1: aarch64 with ramfb framebuffer device.
# LFB rendering visible via QEMU graphical window or VNC.
# Serial output still available via stdio for diagnostics.
run-aarch64-lfb: build-aarch64
	qemu-system-aarch64 \
		-machine virt \
		-cpu host \
		-accel hvf \
		-m 8G \
		-serial mon:stdio \
		-device ramfb \
		-kernel "$(AARCH64_BIN)" \
		-initrd "$(MODEL_PATH)"

clean:
	cd kernel && cargo clean
	cd boot && cargo clean
	rm -rf target
