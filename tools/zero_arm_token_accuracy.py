#!/usr/bin/env python3
"""ARM/NEON token-accuracy gate for Zero Server.

This tool intentionally fails if the Qwen anchor model is missing. The old Rust
test skips in that case, which is useful for CI without model assets but not
acceptable when we are validating byte/token accuracy on an ARM machine.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QWEN = REPO_ROOT / "kernel/programs/Qwen_Qwen3-1.7B-Q4_K_M.gguf"
TOKEN_ANCHOR_TEST = "test_forward_pass_hello_produces_token_25"
EXPECTED_TOKEN_LINE = "Predicted token-ID: 25"
EXPECTED_PASS_LINE = "matches Sub-MP-C3 ground truth"


def run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def print_output(proc: subprocess.CompletedProcess[str]) -> None:
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")


def require_arm_host(args: argparse.Namespace) -> None:
    machine = platform.machine().lower()
    if machine not in {"arm64", "aarch64"} and not args.allow_non_arm:
        raise SystemExit(
            f"host is {machine!r}, not ARM64. Re-run on Apple Silicon/aarch64 "
            "or pass --allow-non-arm for build-only diagnostics."
        )
    print(f"host: {machine}")


def require_qwen_model(path: Path, args: argparse.Namespace) -> bool:
    if path.exists():
        size_mib = path.stat().st_size / (1024 * 1024)
        print(f"qwen anchor: {path} ({size_mib:.1f} MiB)")
        return True
    if args.allow_missing_model:
        print(f"qwen anchor missing, token test skipped by request: {path}")
        return False
    raise SystemExit(
        f"missing Qwen anchor model: {path}\n"
        "The ARM token-accuracy gate requires the model because the underlying "
        "Rust test otherwise skips and can produce a false green result."
    )


def build_aarch64_kernel(args: argparse.Namespace) -> None:
    if args.skip_kernel_build:
        print("aarch64 kernel build: skipped")
        return
    cmd = [
        "cargo",
        "build",
        "--target",
        "aarch64-unknown-none",
        "--release",
        "--features",
        args.kernel_features,
    ]
    # Run inside kernel/ so kernel/.cargo/config.toml supplies aarch64-linker.ld.
    proc = run(cmd, cwd=REPO_ROOT / "kernel")
    print_output(proc)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def run_token_anchor(args: argparse.Namespace, model_present: bool) -> None:
    if args.skip_token_anchor:
        print("token anchor: skipped")
        return
    if not model_present:
        return

    cmd = [
        "cargo",
        "test",
        "--release",
        "-p",
        "zero-llm-inference",
        "--test",
        "forward_pass_reference",
        TOKEN_ANCHOR_TEST,
        "--",
        "--nocapture",
    ]
    proc = run(cmd)
    print_output(proc)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    if "GGUF not found, skipping" in proc.stdout:
        raise SystemExit("token anchor skipped unexpectedly; refusing false green")
    if EXPECTED_TOKEN_LINE not in proc.stdout or EXPECTED_PASS_LINE not in proc.stdout:
        raise SystemExit(
            "token anchor did not print the strict token-25 pass marker; "
            "treating this as a failed accuracy gate"
        )


def pack_smodel_for_arm(args: argparse.Namespace) -> None:
    if args.hf_dir is None:
        print("smodel ARM pack: skipped (no --hf-dir)")
        return
    out = args.smodel_out or (REPO_ROOT / "target/zero-arm-token-accuracy/anchor.smodel")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "tools/silicatepack.py",
        "pack-hf",
        "--input-dir",
        str(args.hf_dir.expanduser().resolve()),
        "--output",
        str(out),
        "--target-product",
        "Zero Server",
        "--target-arch",
        args.smodel_target_arch,
        "--profile",
        args.smodel_profile,
        "--verify",
        "--force",
    ]
    proc = run(cmd)
    print_output(proc)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the ARM kernel path and run the strict Qwen token-25 accuracy anchor.",
    )
    parser.add_argument("--qwen-gguf", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--allow-non-arm", action="store_true")
    parser.add_argument("--allow-missing-model", action="store_true")
    parser.add_argument("--skip-kernel-build", action="store_true")
    parser.add_argument("--skip-token-anchor", action="store_true")
    parser.add_argument("--kernel-features", default="neon-acceleration")
    parser.add_argument("--hf-dir", type=Path, help="optional HF SafeTensors directory to pack as ARM .smodel")
    parser.add_argument("--smodel-out", type=Path)
    parser.add_argument("--smodel-target-arch", default="aarch64-neon")
    parser.add_argument("--smodel-profile", default="cpu-neon")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_arm_host(args)
    model_present = require_qwen_model(args.qwen_gguf.expanduser().resolve(), args)
    build_aarch64_kernel(args)
    run_token_anchor(args, model_present)
    pack_smodel_for_arm(args)
    print("ARM token-accuracy gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
