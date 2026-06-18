#!/usr/bin/env bash
#
# Build a bootable x86_64 Zero image for bare-metal deployment.
#
# Two modes:
#   default   — embed a native SilicatePack `.smodel` as a ramdisk
#               (raw GGUF is legacy compatibility only)
#   --bare    — no model, Stage 11 skips at boot (HW bring-up only)
#
# Additional flag:
#   --cherry     — build the kernel with `cherry-net,avx512-acceleration`
#                  (static-IP profile + SMP multi-core + AVX-512).
#                  Combine with --bare for a model-less Cherry build.
#   --kimi       — Cherry-only: add `kimi-k26-arena,kimi-k26-vocab` so
#                  KV_CACHE_ARENA_SIZE upgrades from 512 MiB to 1.5 GiB
#                  (compressed-latent MLA cache, ≥8K context × 61 layers)
#                  and the runtime detokenizer uses the packed tokenizer
#                  vocab in place of the compile-time Qwen3 stub.
#                  Without --kimi the Kimi K2.6 path still runs but caps
#                  context around 2.5K tokens and emits tokens by ID.
#   --smp-debug  — Cherry-only: add `cherry-smp-debug` for the
#                  once-per-second AP registration heartbeat.
#   --control-plane
#                — Cherry-only: add `zero-control-plane` to pause
#                  before Stage 11 until `llm-start` / `llm start`.
#
# Usage:
#   tools/build-x86_64-baremetal.sh                # default, MODEL_PATH from env or kernel/programs
#   MODEL_PATH=/path/to/model.smodel tools/build-x86_64-baremetal.sh
#   tools/build-x86_64-baremetal.sh --bare
#   tools/build-x86_64-baremetal.sh --cherry       # Cherry Server image with model
#   tools/build-x86_64-baremetal.sh --cherry --bare # Cherry Server image, no model
#   tools/build-x86_64-baremetal.sh --cherry --bare --kimi # Cherry + Kimi K2.6 arena/vocab
#   tools/build-x86_64-baremetal.sh --cherry --control-plane
#
# Output paths (written to stdout when finished):
#   target/zero-images/zero-bios.img        — boot via legacy BIOS / IPMI ISO-CDROM
#   target/zero-images/zero-uefi.img        — boot via UEFI firmware (most modern servers)
#
# The kernel must build in release mode (Makefile enforces this — see
# ADR-028 patch v4). Debug builds collide with bootloader 0.11's
# identity-mapped region during boot setup.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BARE=0
CHERRY=0
SMP_DEBUG=0
CONTROL_PLANE=0
KIMI=0
for arg in "$@"; do
    case "$arg" in
        --bare)          BARE=1 ;;
        --cherry)        CHERRY=1 ;;
        --kimi)          KIMI=1 ;;
        --smp-debug)     SMP_DEBUG=1 ;;
        --control-plane) CONTROL_PLANE=1 ;;
        --help|-h)       sed -n '2,35p' "$0"; exit 0 ;;
        *)               echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

FEATURES=""
if [[ "${CHERRY}" -eq 1 ]]; then
    FEATURE_LIST="cherry-net,avx512-acceleration"
    if [[ "${SMP_DEBUG}" -eq 1 ]]; then
        FEATURE_LIST="${FEATURE_LIST},cherry-smp-debug"
    fi
    if [[ "${CONTROL_PLANE}" -eq 1 ]]; then
        FEATURE_LIST="${FEATURE_LIST},zero-control-plane"
    fi
    if [[ "${KIMI}" -eq 1 ]]; then
        FEATURE_LIST="${FEATURE_LIST},kimi-k26-arena,kimi-k26-vocab"
    fi
    FEATURES="--features ${FEATURE_LIST}"
elif [[ "${SMP_DEBUG}" -eq 1 ]]; then
    echo "ERROR: --smp-debug requires --cherry." >&2
    exit 2
elif [[ "${CONTROL_PLANE}" -eq 1 ]]; then
    echo "ERROR: --control-plane requires --cherry." >&2
    exit 2
elif [[ "${KIMI}" -eq 1 ]]; then
    echo "ERROR: --kimi requires --cherry." >&2
    exit 2
fi

KERNEL_ELF="${REPO_ROOT}/kernel/target/x86_64-unknown-none/release/zero-kernel"
DEFAULT_MODEL="${REPO_ROOT}/kernel/programs/Qwen_Qwen3-1.7B-Q4_K_M.gguf"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL}}"
CHERRY_OPT_LEVEL="${CHERRY_OPT_LEVEL:-3}"

if [[ "${KIMI}" -eq 1 ]]; then
    LAYOUT_PATH="${REPO_ROOT}/target/deploy-kimi-k26-layout.json"
    if [[ -f "${LAYOUT_PATH}" && -z "${ZERO_NVME_MODEL_BYTES:-}" && -z "${ZERO_KIMI_K26_NVME_BYTES:-}" ]]; then
        parsed_bytes="$(sed -n 's/.*"\(model_size_bytes\|smodel_size_bytes\|gguf_size_bytes\)"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\2/p' "${LAYOUT_PATH}" | head -n 1)"
        if [[ -n "${parsed_bytes}" ]]; then
            export ZERO_NVME_MODEL_BYTES="${parsed_bytes}"
        fi
    fi
    if [[ -f "${LAYOUT_PATH}" && -z "${ZERO_NVME_MODEL_LBA_OFFSET:-}" && -z "${ZERO_KIMI_K26_NVME_LBA_OFFSET:-}" ]]; then
        parsed_lba="$(sed -n 's/.*"lba_offset"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "${LAYOUT_PATH}" | head -n 1)"
        if [[ -n "${parsed_lba}" ]]; then
            export ZERO_NVME_MODEL_LBA_OFFSET="${parsed_lba}"
        fi
    fi
fi

if [[ "${CHERRY}" -eq 1 ]]; then
    echo "[1/3] Building kernel (release, x86_64-unknown-none, features: ${FEATURE_LIST})"
    echo "      Cherry performance profile: CARGO_PROFILE_RELEASE_OPT_LEVEL=${CHERRY_OPT_LEVEL}"
    if [[ "${KIMI}" -eq 1 ]]; then
        echo "      NVMe model bytes: ${ZERO_NVME_MODEL_BYTES:-${ZERO_KIMI_K26_NVME_BYTES:-kernel default}}"
        echo "      NVMe model LBA offset: ${ZERO_NVME_MODEL_LBA_OFFSET:-${ZERO_KIMI_K26_NVME_LBA_OFFSET:-kernel default}}"
    fi
else
    echo "[1/3] Building kernel (release, x86_64-unknown-none)"
fi
if [[ "${CHERRY}" -eq 1 ]]; then
    # Cherry-target opt-level — kept in lockstep with Makefile
    # kernel-cherry* targets (CHERRY_OPT_LEVEL defaults to 3). The
    # aarch64 opt-level=1 constraint (Lessons 7/9/11, STP/HVF) does
    # not apply on x86_64; opt-level=3 is required for the AVX-512
    # hot loops to keep ZMM accumulators in registers instead of
    # spilling.
    ( cd kernel && CARGO_PROFILE_RELEASE_OPT_LEVEL="${CHERRY_OPT_LEVEL}" cargo build --release ${FEATURES} )
else
    ( cd kernel && cargo build --release ${FEATURES} )
fi

echo "[2/3] Wrapping kernel into BIOS + UEFI disk images"
if [[ "${BARE}" -eq 1 ]]; then
    echo "      mode: --bare (no model, Stage 11 will skip)"
    ( cd boot && ZERO_KERNEL_ELF="${KERNEL_ELF}" cargo run --release )
else
    if [[ ! -f "${MODEL_PATH}" ]]; then
        echo "ERROR: MODEL_PATH=${MODEL_PATH} not found." >&2
        echo "       Either fetch the model into that path, set MODEL_PATH=..., or rerun with --bare." >&2
        exit 1
    fi
    echo "      mode: model ramdisk (${MODEL_PATH})"
    ( cd boot \
        && ZERO_KERNEL_ELF="${KERNEL_ELF}" \
           ZERO_MODEL_PATH="${MODEL_PATH}" \
           cargo run --release )
fi

echo "[3/3] Verifying outputs"
BIOS_IMG="${REPO_ROOT}/target/zero-images/zero-bios.img"
UEFI_IMG="${REPO_ROOT}/target/zero-images/zero-uefi.img"
for img in "${BIOS_IMG}" "${UEFI_IMG}"; do
    if [[ ! -f "${img}" ]]; then
        echo "ERROR: expected image not produced: ${img}" >&2
        exit 1
    fi
    sz="$(stat -f %z "${img}" 2>/dev/null || stat -c %s "${img}")"
    printf "      %s  (%d bytes, %d MiB)\n" "${img}" "${sz}" "$(( sz / 1048576 ))"
done

cat <<EOF

Build complete.

Boot images:
  BIOS:  ${BIOS_IMG}
  UEFI:  ${UEFI_IMG}

Next steps:
  * Local sanity check:        make run-bare
  * Write to USB stick (macOS): diskutil unmountDisk /dev/diskN && \\
                                sudo dd if=${UEFI_IMG} of=/dev/rdiskN bs=4m && sync
  * Write to USB stick (Linux): sudo dd if=${UEFI_IMG} of=/dev/sdX bs=4M status=progress conv=fsync
  * IPMI virtual media: upload  ${UEFI_IMG}  through the BMC console.
  * Full deployment guide:      docs/deployment-x86_64.md
EOF
