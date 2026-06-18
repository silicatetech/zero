// SPDX-License-Identifier: AGPL-3.0-or-later
use std::path::{Path, PathBuf};

/// Zero bootable disk-image builder.
///
/// Env-var contract:
///   * `ZERO_KERNEL_ELF`  (required) — path to the compiled
///     `x86_64-unknown-none/release/zero-kernel` ELF.
///   * `ZERO_MODEL_PATH`  (optional) — path to the Boot-LLM GGUF
///     file. When set, the file is embedded into the boot image as a
///     ramdisk. bootloader 0.11 maps it into the kernel address space
///     and reports `BootInfo::ramdisk_addr` / `ramdisk_len`. This is
///     the production path for bare metal — no QEMU `-device loader`
///     required.
///
/// When `ZERO_MODEL_PATH` is unset, the image still boots: the
/// kernel logs "Stage 11: no model present — skipping" and continues
/// through the rest of the boot sequence (sandbox, async runtime,
/// PCIe enumeration). This is the "model-optional" boot mode used
/// for HW bring-up and PCIe/HAL development.
fn main() -> anyhow::Result<()> {
    let kernel_path = std::env::var("ZERO_KERNEL_ELF").map_err(|_| {
        anyhow::anyhow!(
            "set ZERO_KERNEL_ELF to the path of the compiled kernel ELF.\n\
             typical: .../kernel/target/x86_64-unknown-none/release/zero-kernel"
        )
    })?;

    let kernel_path = Path::new(&kernel_path);
    anyhow::ensure!(kernel_path.exists(), "kernel ELF not found: {}", kernel_path.display());

    let model_path = std::env::var("ZERO_MODEL_PATH").ok().map(PathBuf::from);
    if let Some(ref p) = model_path {
        anyhow::ensure!(
            p.exists(),
            "ZERO_MODEL_PATH set but file not found: {}",
            p.display()
        );
        let size = std::fs::metadata(p)?.len();
        println!(
            "Model: embedding ramdisk from {} ({} bytes / {} MiB)",
            p.display(),
            size,
            size / (1024 * 1024)
        );
    } else {
        println!("Model: ZERO_MODEL_PATH unset — building model-less image (Stage 11 will skip at boot)");
    }

    let out_dir = out_dir();
    std::fs::create_dir_all(&out_dir)?;

    let bios_image = out_dir.join("zero-bios.img");
    {
        let mut builder = bootloader::BiosBoot::new(kernel_path);
        if let Some(ref p) = model_path {
            builder.set_ramdisk(p.as_path());
        }
        builder.create_disk_image(&bios_image)?;
    }
    println!("BIOS image: {}", bios_image.display());

    let uefi_image = out_dir.join("zero-uefi.img");
    {
        let mut builder = bootloader::UefiBoot::new(kernel_path);
        if let Some(ref p) = model_path {
            builder.set_ramdisk(p.as_path());
        }
        builder.create_disk_image(&uefi_image)?;
    }
    println!("UEFI image: {}", uefi_image.display());

    Ok(())
}

fn out_dir() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest).parent().unwrap().join("target/zero-images")
}
