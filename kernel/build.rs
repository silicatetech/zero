// SPDX-License-Identifier: AGPL-3.0-or-later
use std::fs;
use std::path::PathBuf;

fn main() {
    // === Existing: linker.ld setup (preserved from Phase 2) ===
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rustc-link-arg-bins=-T{dir}/linker.ld");
    println!("cargo:rerun-if-changed=linker.ld");

    // === Stage 10 MP3: AOT compile boot.ir → program.bin ===
    compile_boot_ir();
    emit_kimi_nvme_layout();
}

/// Read boot.ir, validate, AOT-compile to x86_64 machine code,
/// and write the result as `program.bin` to OUT_DIR.
///
/// The kernel can then embed program.bin via
/// `include_bytes!(concat!(env!("OUT_DIR"), "/program.bin"))` (MP4).
fn compile_boot_ir() {
    let boot_ir_path = PathBuf::from("programs/boot.ir");

    // Tell cargo to re-run if boot.ir changes
    println!("cargo:rerun-if-changed=programs/boot.ir");

    // Read boot.ir source
    let source = fs::read_to_string(&boot_ir_path).unwrap_or_else(|e| {
        panic!(
            "Stage 10 MP3: failed to read boot.ir at {:?}: {}",
            boot_ir_path, e
        )
    });

    // Pass 1: parse to SExpr
    let ast = quarks_validator::parse(&source)
        .unwrap_or_else(|e| panic!("Stage 10 MP3: boot.ir parse failed: {:?}", e));

    // Pass 2: type-check (semantic validation)
    quarks_validator::type_check(&ast)
        .unwrap_or_else(|e| panic!("Stage 10 MP3: boot.ir type-check failed: {:?}", e));

    // Pass 3: AOT compile to x86_64 machine code
    let bytes = quarks_codegen::compile(&ast)
        .unwrap_or_else(|e| panic!("Stage 10 MP3: boot.ir codegen failed: {}", e));

    // Write program.bin to OUT_DIR
    let out_dir = std::env::var("OUT_DIR").expect("Stage 10 MP3: OUT_DIR not set by cargo");
    let out_path = PathBuf::from(&out_dir).join("program.bin");

    fs::write(&out_path, &bytes).unwrap_or_else(|e| {
        panic!(
            "Stage 10 MP3: failed to write program.bin to {:?}: {}",
            out_path, e
        )
    });

    println!(
        "cargo:warning=Stage 10 MP3: boot.ir compiled to {} bytes at {}",
        bytes.len(),
        out_path.display()
    );
}

fn emit_kimi_nvme_layout() {
    println!("cargo:rerun-if-env-changed=ZERO_NVME_MODEL_BYTES");
    println!("cargo:rerun-if-env-changed=ZERO_NVME_MODEL_LBA_OFFSET");
    println!("cargo:rerun-if-env-changed=ZERO_KIMI_K26_NVME_BYTES");
    println!("cargo:rerun-if-env-changed=ZERO_KIMI_K26_NVME_LBA_OFFSET");

    let out_dir = std::env::var("OUT_DIR").expect("Stage 11: OUT_DIR not set by cargo");
    let out_path = PathBuf::from(&out_dir).join("kimi_nvme_layout.rs");

    let bytes = std::env::var("ZERO_NVME_MODEL_BYTES")
        .or_else(|_| std::env::var("ZERO_KIMI_K26_NVME_BYTES"))
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(584 * 1024 * 1024 * 1024);
    let lba_offset = std::env::var("ZERO_NVME_MODEL_LBA_OFFSET")
        .or_else(|_| std::env::var("ZERO_KIMI_K26_NVME_LBA_OFFSET"))
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);

    let src = format!(
        "pub const KIMI_K26_NVME_BYTES: usize = {bytes}usize;\n\
         pub const KIMI_K26_NVME_LBA_OFFSET: u64 = {lba_offset}u64;\n"
    );
    fs::write(&out_path, src).unwrap_or_else(|e| {
        panic!(
            "Stage 11: failed to write kimi_nvme_layout.rs to {:?}: {}",
            out_path, e
        )
    });

    if std::env::var_os("CARGO_FEATURE_KIMI_K26_ARENA").is_some() {
        println!(
            "cargo:warning=Stage 11: Kimi NVMe layout bytes={} lba_offset={}",
            bytes, lba_offset
        );
    }
}
