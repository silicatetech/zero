#!/usr/bin/env python3
"""Zero Server acceptance gate for native `.smodel` work.

This is intentionally small and deterministic: it checks the tooling,
validates an optional SilicatePack artifact, and compiles the ARM/x86
kernel paths that exercise the native model loader. Long-running QEMU or
bare-metal boots stay outside this script so CI and rescue builds do not
silently spend hours.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SMODEL = REPO.parent / "models" / "smodel" / "qwen3-1.7b-aarch64-neon.smodel"
PYTHON_CHECKS = [
    "tools/silicatepack.py",
    "tools/zero_acceptance.py",
    "tools/test_silicatepack.py",
]
OPTIONAL_PYTHON_CHECKS = [
    "tools/zero_arm_token_accuracy.py",
]


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    printable = " ".join(cmd)
    print(f"[zero-acceptance] {printable}")
    subprocess.run(cmd, cwd=REPO, env=env, check=True)


def maybe_verify_smodel(args: argparse.Namespace) -> None:
    smodel = Path(args.smodel).expanduser().resolve() if args.smodel else DEFAULT_SMODEL
    if not smodel.exists():
        if args.smodel or args.strict_smodel:
            raise FileNotFoundError(smodel)
        print(f"[zero-acceptance] skip .smodel verify: {smodel} not found")
        return
    cmd = [sys.executable, "tools/silicatepack.py", "verify", str(smodel)]
    if args.strict_smodel:
        cmd.append("--strict")
    if args.hash_payload:
        cmd.append("--hash-payload")
    run(cmd)


def python_checks(args: argparse.Namespace) -> None:
    if args.skip_pycompile:
        print("[zero-acceptance] skip python compile checks")
        return
    paths = list(PYTHON_CHECKS)
    paths.extend(path for path in OPTIONAL_PYTHON_CHECKS if (REPO / path).exists())
    run([sys.executable, "-m", "py_compile", *paths])


def unit_tests(args: argparse.Namespace) -> None:
    if args.skip_unit_tests:
        print("[zero-acceptance] skip SilicatePack unit tests")
        return
    run([sys.executable, "-m", "unittest", "tools/test_silicatepack.py"])


def cargo_checks(args: argparse.Namespace) -> None:
    if args.skip_cargo:
        print("[zero-acceptance] skip cargo checks")
        return
    if args.target in ("all", "aarch64"):
        run(
            [
                "cargo",
                "check",
                "--manifest-path",
                "kernel/Cargo.toml",
                "--target",
                "aarch64-unknown-none",
                "--release",
                "--features",
                "neon-acceleration",
            ]
        )
    if args.target in ("all", "x86_64"):
        run(
            [
                "cargo",
                "check",
                "--manifest-path",
                "kernel/Cargo.toml",
                "--target",
                "x86_64-unknown-none",
                "--release",
                "--features",
                "cherry-net,avx512-acceleration",
            ]
        )


def build_arm(args: argparse.Namespace) -> None:
    if not args.build_arm:
        return
    smodel = Path(args.smodel).expanduser().resolve() if args.smodel else DEFAULT_SMODEL
    if not smodel.exists():
        raise FileNotFoundError(smodel)
    env = os.environ.copy()
    env["MODEL_PATH"] = str(smodel)
    run(["make", "build-aarch64"], env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Zero Server acceptance checks.")
    parser.add_argument("--smodel", help="optional native .smodel artifact to validate")
    parser.add_argument("--strict-smodel", action="store_true", help="require strict byte/logit anchors")
    parser.add_argument(
        "--hash-payload",
        action="store_true",
        help="hash the complete model payload; expensive for large artifacts",
    )
    parser.add_argument("--target", choices=("all", "aarch64", "x86_64"), default="all")
    parser.add_argument("--skip-cargo", action="store_true")
    parser.add_argument("--skip-pycompile", action="store_true")
    parser.add_argument("--skip-unit-tests", action="store_true")
    parser.add_argument("--build-arm", action="store_true", help="also build the ARM image with MODEL_PATH")
    args = parser.parse_args(argv)

    python_checks(args)
    unit_tests(args)
    maybe_verify_smodel(args)
    cargo_checks(args)
    build_arm(args)
    print("[zero-acceptance] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
