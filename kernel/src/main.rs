// SPDX-License-Identifier: AGPL-3.0-or-later
#![no_std]
#![no_main]
#![cfg_attr(target_arch = "x86_64", feature(abi_x86_interrupt))]

extern crate alloc;
#[cfg(target_arch = "x86_64")]
use bootloader_api::config::Mapping;
#[cfg(target_arch = "x86_64")]
use bootloader_api::{entry_point, BootInfo, BootloaderConfig};
#[allow(unused_imports)]
use core::fmt::Write;
use core::panic::PanicInfo;

#[cfg(target_arch = "x86_64")]
mod aot;
mod arch;
mod memory;
#[cfg(target_arch = "x86_64")]
mod net;
#[cfg(target_arch = "x86_64")]
mod task;

mod arena_allocator;
#[cfg(target_arch = "x86_64")]
mod bench;
#[cfg(any(not(feature = "streaming-mode"), target_arch = "aarch64"))]
mod detokenizer;
mod inference;
#[cfg(all(target_arch = "aarch64", feature = "neon-acceleration"))]
mod inference_neon;
#[cfg(feature = "nondeterministic-sampling")]
mod rng;
#[cfg(feature = "nondeterministic-sampling")]
mod sampler;
// sandbox module excluded from public release (Stage 12 WIP)
// ADR-029 Phase 1+2+3: SMP infrastructure (platform-independent) and
// the x86_64 AVX-512 inference dispatch. `smp` is feature-gated on
// avx512-acceleration because its first (and currently only) consumer
// is the AVX-512 parallel matmul. The `inference_avx512` module is
// additionally gated to `target_arch = "x86_64"` since it pulls in
// `arch::x86_64::math`.
#[cfg(target_arch = "x86_64")]
mod control_plane;
#[cfg(target_arch = "x86_64")]
mod drivers;
#[cfg(all(target_arch = "x86_64", feature = "avx512-acceleration"))]
mod inference_avx512;
mod lfb;
#[cfg(target_arch = "x86_64")]
mod llm_arena;
mod model_loader;
#[cfg(feature = "avx512-acceleration")]
mod smp;
mod weight_layout;
#[cfg(all(target_arch = "x86_64", feature = "kimi-k26-arena"))]
mod kimi_nvme_layout {
    include!(concat!(env!("OUT_DIR"), "/kimi_nvme_layout.rs"));
}

/// Bootloader configuration, read by the bootloader *before* our
/// code runs. Pins the physical-memory linear mapping to a known
/// canonical higher-half address so that
/// `boot_info.physical_memory_offset == Some(0xFFFF_8000_0000_0000)`
/// on entry. Other mappings (kernel stack, boot_info, framebuffer)
/// stay `Dynamic` — the bootloader picks suitable higher-half
/// addresses for them.
#[cfg(target_arch = "x86_64")]
pub static BOOTLOADER_CONFIG: BootloaderConfig = {
    let mut config = BootloaderConfig::new_default();
    config.mappings.physical_memory = Some(Mapping::FixedAddress(0xFFFF_8000_0000_0000));
    // 256 KiB kernel stack — conservative headroom for the Quarks
    // interpreter and future Stage 9+ workloads.
    config.kernel_stack_size = 256 * 1024;
    config
};

#[cfg(target_arch = "x86_64")]
entry_point!(kernel_main, config = &BOOTLOADER_CONFIG);

/// Disable the IPMI Watchdog Timer via the KCS (Keyboard Controller
/// Style) interface. Supermicro BMCs expose KCS at data=0xCA2,
/// status/cmd=0xCA3. The command sequence is:
///
/// 1. WRITE_START phase: send NetFn|LUN = 0x18 (App, LUN 0)
/// 2. DATA phase: send Cmd = 0x24 (Set Watchdog Timer)
/// 3. DATA phase: send 6 zero bytes (timer use=0, actions=0,
///    pre-timeout=0, timeout_action_expiration=0, countdown=0,0)
///
/// A zeroed "Set Watchdog Timer" command disables the timer.
///
/// Returns `true` on success, `false` if the KCS interface times out
/// (e.g. no BMC present).
#[cfg(all(target_arch = "x86_64", feature = "cherry-net"))]
fn ipmi_kcs_disable_watchdog() -> bool {
    // Supermicro standard KCS I/O ports
    const KCS_DATA: u16 = 0xCA2;
    const KCS_CMD_STATUS: u16 = 0xCA3;

    // KCS status register bits
    const IBF: u8 = 0x02; // Input Buffer Full
    const OBF: u8 = 0x01; // Output Buffer Full
                          // KCS commands
    const WRITE_START: u8 = 0x61;
    const WRITE_END: u8 = 0x62;

    #[inline]
    unsafe fn kcs_status() -> u8 {
        let val: u8;
        core::arch::asm!("in al, dx", out("al") val, in("dx") KCS_CMD_STATUS, options(nomem, nostack, preserves_flags));
        val
    }
    #[inline]
    unsafe fn kcs_read_data() -> u8 {
        let val: u8;
        core::arch::asm!("in al, dx", out("al") val, in("dx") KCS_DATA, options(nomem, nostack, preserves_flags));
        val
    }
    #[inline]
    unsafe fn kcs_write_data(val: u8) {
        core::arch::asm!("out dx, al", in("dx") KCS_DATA, in("al") val, options(nomem, nostack, preserves_flags));
    }
    #[inline]
    unsafe fn kcs_write_cmd(val: u8) {
        core::arch::asm!("out dx, al", in("dx") KCS_CMD_STATUS, in("al") val, options(nomem, nostack, preserves_flags));
    }

    // Wait for IBF to clear (BMC consumed previous byte). Timeout via
    // spin counter — ~100 ms at typical I/O speeds.
    #[inline]
    unsafe fn wait_ibf_clear() -> bool {
        for _ in 0..100_000u32 {
            if (kcs_status() & IBF) == 0 {
                return true;
            }
            core::hint::spin_loop();
        }
        false
    }
    // Wait for OBF to set (BMC has a byte for us to read).
    #[inline]
    unsafe fn wait_obf_set() -> bool {
        for _ in 0..100_000u32 {
            if (kcs_status() & OBF) != 0 {
                return true;
            }
            core::hint::spin_loop();
        }
        false
    }

    // IPMI message:
    //   NetFn = 0x06 (Application), LUN = 0 → NetFn|LUN byte = (0x06 << 2) | 0 = 0x18
    //   Cmd   = 0x24 (Set Watchdog Timer)
    //   Data  = [0x00; 6] — all fields zeroed = timer disabled
    let message: [u8; 8] = [
        0x18, // NetFn|LUN
        0x24, // Cmd: Set Watchdog Timer
        0x00, // Timer Use (0 = BIOS FRB2, but with "don't start" flag via 0-countdown)
        0x00, // Timer Actions (0 = no action on expiry)
        0x00, // Pre-timeout Interval (0)
        0x00, // Timer Use Expiration Flags Clear (0)
        0x00, // Countdown LSB (0 = disabled)
        0x00, // Countdown MSB (0 = disabled)
    ];

    unsafe {
        // Phase 1: WRITE_START
        if !wait_ibf_clear() {
            return false;
        }
        kcs_write_cmd(WRITE_START);
        if !wait_ibf_clear() {
            return false;
        }
        // Drain any pending OBF
        if (kcs_status() & OBF) != 0 {
            let _ = kcs_read_data();
        }

        // Phase 2: write data bytes (all except last)
        for i in 0..message.len() - 1 {
            if !wait_ibf_clear() {
                return false;
            }
            kcs_write_data(message[i]);
            if !wait_ibf_clear() {
                return false;
            }
            // Drain OBF between data bytes
            if (kcs_status() & OBF) != 0 {
                let _ = kcs_read_data();
            }
        }

        // Phase 3: WRITE_END + last byte
        if !wait_ibf_clear() {
            return false;
        }
        kcs_write_cmd(WRITE_END);
        if !wait_ibf_clear() {
            return false;
        }
        if (kcs_status() & OBF) != 0 {
            let _ = kcs_read_data();
        }
        kcs_write_data(message[message.len() - 1]);

        // Phase 4: read response (completion code)
        // Wait for BMC to process command and put response in OBF.
        if !wait_ibf_clear() {
            return false;
        }
        if !wait_obf_set() {
            return false;
        }
        let _completion = kcs_read_data();
        // Drain any remaining response bytes
        for _ in 0..16u32 {
            if (kcs_status() & OBF) == 0 {
                break;
            }
            let _ = kcs_read_data();
            if !wait_ibf_clear() {
                break;
            }
        }

        true
    }
}

/// Print f32 as decimal "integer.fraction" using only u32 Display.
/// Avoids core::fmt::float monomorphization (Mode-B per ADR-028 v5).
/// Precision: 4 decimal digits. Handles NaN, ±Inf, ±zero.
#[allow(dead_code)]
fn print_f32(value: f32) {
    if value.is_nan() {
        let _ = write!(arch::serial::Serial, "NaN");
        return;
    }
    if value.is_infinite() {
        if value < 0.0 {
            let _ = write!(arch::serial::Serial, "-Inf");
        } else {
            let _ = write!(arch::serial::Serial, "Inf");
        }
        return;
    }
    if value < 0.0 {
        let _ = write!(arch::serial::Serial, "-");
    }
    let abs = if value < 0.0 { -value } else { value };
    // no_std: f32::trunc/fract unavailable, use cast-to-integer approach
    let int_part = abs as u32;
    let frac = abs - (int_part as f32);
    let frac_part = ((frac * 10000.0) as u32) % 10000;
    let _ = write!(arch::serial::Serial, "{}", int_part);
    let _ = write!(arch::serial::Serial, ".");
    if frac_part < 1000 {
        let _ = write!(arch::serial::Serial, "0");
    }
    if frac_part < 100 {
        let _ = write!(arch::serial::Serial, "0");
    }
    if frac_part < 10 {
        let _ = write!(arch::serial::Serial, "0");
    }
    let _ = write!(arch::serial::Serial, "{}", frac_part);
}

#[allow(dead_code)]
fn write_ggml_type(t: zero_gguf_parser::GgmlType) {
    match t {
        zero_gguf_parser::GgmlType::F32 => {
            let _ = write!(arch::serial::Serial, "F32");
        }
        zero_gguf_parser::GgmlType::F16 => {
            let _ = write!(arch::serial::Serial, "F16");
        }
        zero_gguf_parser::GgmlType::Q4_0 => {
            let _ = write!(arch::serial::Serial, "Q4_0");
        }
        zero_gguf_parser::GgmlType::Q8_0 => {
            let _ = write!(arch::serial::Serial, "Q8_0");
        }
        zero_gguf_parser::GgmlType::Q4K => {
            let _ = write!(arch::serial::Serial, "Q4_K");
        }
        zero_gguf_parser::GgmlType::Q6K => {
            let _ = write!(arch::serial::Serial, "Q6_K");
        }
        _ => {
            let _ = write!(arch::serial::Serial, "OTHER");
        }
    }
}

#[allow(dead_code)]
fn log_tensor_summary(index: &zero_gguf_parser::TensorIndex, name: &str) -> bool {
    let _ = write!(arch::serial::Serial, "  ");
    let _ = write!(arch::serial::Serial, "{}", name);
    if let Some(t) = index.get(name) {
        let _ = write!(arch::serial::Serial, " -> type=");
        write_ggml_type(t.tensor_type);
        let _ = write!(arch::serial::Serial, " shape=[");
        let mut first = true;
        for &d in t.dimensions.iter() {
            if !first {
                let _ = write!(arch::serial::Serial, ", ");
            }
            let _ = write!(arch::serial::Serial, "{}", d);
            first = false;
        }
        let _ = write!(arch::serial::Serial, "]");
        let _ = write!(arch::serial::Serial, " offset=0x{:x}", t.offset);
        let _ = write!(arch::serial::Serial, " elements=");
        let _ = writeln!(arch::serial::Serial, "{}", t.element_count());
        true
    } else {
        let _ = writeln!(arch::serial::Serial, " -> NOT FOUND");
        false
    }
}

#[allow(dead_code)]
fn log_deepseek2_boot_heartbeat(index: &zero_gguf_parser::TensorIndex) {
    let _ = writeln!(
        arch::serial::Serial,
        "[MP2.1] DeepSeek2/Kimi tensor heartbeat starting..."
    );
    let _ = writeln!(
        arch::serial::Serial,
        "[MP2.1] Skipping Qwen Q4_K/Q6_K heartbeats for deepseek2 architecture"
    );

    let mut found = 0u32;
    for name in [
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        "blk.0.attn_kv_a_mqa.weight",
        "blk.0.attn_kv_a_norm.weight",
        "blk.1.attn_k_b.weight",
        "blk.1.attn_v_b.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.1.ffn_gate_exps.weight",
    ] {
        if log_tensor_summary(index, name) {
            found += 1;
        }
    }

    let _ = write!(
        arch::serial::Serial,
        "[MP2.1] DeepSeek2/Kimi heartbeat tensors found: "
    );
    let _ = writeln!(arch::serial::Serial, "{}", found);
}

/// Zero Server production model-loader fallback — stream a resident
/// `.smodel` or raw-GGUF compatibility artifact from the data NVMe into
/// a contiguous weight arena view.
///
/// This is the third option behind `'mp1:`-block's RamdiskLoader and
/// DirectMemoryLoader. Very large weights cannot ride along as a
/// bootloader ramdisk and are not pre-staged by any QEMU
/// `-device loader`; production deploy writes the model artifact to the
/// data NVMe and pulls it into RAM here at boot.
///
/// Returns `Some((bytes, size))` on success — `bytes` is a `'static`
/// slice over the phys-linear view of the freshly-loaded weight
/// arena. Returns `None` on any failure (no NVMe, magic mismatch,
/// arena reservation failed, etc.) so the caller can fall through to
/// "kernel continues without Boot-LLM".
///
/// Gated on `cherry-net + kimi-k26-arena`: ordinary Cherry bring-up
/// images must not try to reserve a multi-hundred-GiB Kimi weight
/// arena. The expected NVMe layout is generated by `kernel/build.rs`
/// from deploy-time environment variables populated by
/// `scripts/deploy-kimi-k26.sh`.
#[cfg(all(
    target_arch = "x86_64",
    feature = "cherry-net",
    feature = "kimi-k26-arena"
))]
fn try_load_nvme_kimi_k26(pci_scan: &arch::pcie::PciScan) -> Option<(&'static [u8], usize)> {
    use crate::kimi_nvme_layout::{KIMI_K26_NVME_BYTES, KIMI_K26_NVME_LBA_OFFSET};
    use crate::model_loader::{ModelLoader, NvmeModelLoader};

    let _ = writeln!(
        arch::serial::Serial,
        "Stage 11 MP1: NVMe model-load fallback — reserving {} GiB weight arena",
        KIMI_K26_NVME_BYTES / (1024 * 1024 * 1024)
    );

    // ── Step 1: reserve a virtually-contiguous arena ──────────────
    //
    // Previously this called `alloc_contiguous_phys_linear`, which
    // demands a single *physically* contiguous Usable region of
    // 584 GiB and returns the bootloader's phys-linear view of it.
    // On EPYC + EFI the memory map breaks high RAM into multiple
    // Usable bands separated by ACPI reclaim / BIOS-reserved bands,
    // so a 584 GiB single-region allocation usually fails before
    // NVMe load begins.
    //
    // `alloc_scattered_virt_contiguous` instead harvests 2 MiB
    // chunks from anywhere in the Usable map and installs them
    // via 2 MiB huge-page mappings into a freshly-picked range of
    // consecutive PML4 slots. Virtually contiguous, physically
    // scattered — the bulk_read loop below and the inference
    // engine's `model_bytes: &[u8]` slice see one flat 584 GiB
    // buffer either way.
    let (dst_va, mapped_bytes) = match memory::alloc_scattered_virt_contiguous(KIMI_K26_NVME_BYTES)
    {
        Some(t) => t,
        None => {
            let _ = writeln!(
                    arch::serial::Serial,
                    "Stage 11 MP1: NVMe arena reservation FAILED — could not gather {} GiB of 2 MiB chunks from Usable memory",
                    KIMI_K26_NVME_BYTES / (1024 * 1024 * 1024)
                );
            return None;
        }
    };
    let _ = writeln!(
        arch::serial::Serial,
        "Stage 11 MP1: NVMe arena reserved at virt=0x{:x} ({} GiB virtually contiguous, physically scattered)",
        dst_va,
        mapped_bytes / (1024 * 1024 * 1024)
    );

    // ── Step 2: build the loader (NSID=1, LBA=0 — deploy convention) ─
    let mut loader =
        match unsafe { NvmeModelLoader::new(1, KIMI_K26_NVME_LBA_OFFSET, KIMI_K26_NVME_BYTES) } {
            Ok(l) => l,
            Err(e) => {
                let _ = writeln!(
                    arch::serial::Serial,
                    "Stage 11 MP1: NvmeModelLoader::new failed: {:?}",
                    e
                );
                return None;
            }
        };

    // ── Step 3: stream the model artifact off NVMe ─────────────────
    //
    // `make_resident` does its own PCI scan + bind_model_data_drive
    // (picks the drive whose LBA 0 carries either the native `SILM`
    // marker or raw `GGUF`, side-stepping the boot-drive-also-3.5TB
    // tie-break problem) + bulk_read. The bulk read polls one 2 MiB
    // chunk at a time — on Cherry's Gen5 NVMe (≈14 GB/s) the full
    // 584 GiB transfer is bound by wire bandwidth and completes in
    // ~45 s; the polling overhead per chunk is negligible.
    let _ = writeln!(
        arch::serial::Serial,
        "Stage 11 MP1: NVMe bulk_read starting ({} GiB; this may take ~45s on Gen5 NVMe)",
        KIMI_K26_NVME_BYTES / (1024 * 1024 * 1024)
    );
    if let Err(e) = loader.make_resident(pci_scan, dst_va) {
        let _ = writeln!(
            arch::serial::Serial,
            "Stage 11 MP1: NvmeModelLoader::make_resident failed: {:?}",
            e
        );
        return None;
    }
    let _ = writeln!(
        arch::serial::Serial,
        "Stage 11 MP1: NVMe bulk_read complete"
    );

    // ── Step 4: surface the resident bytes ─────────────────────────
    let bytes = match unsafe { loader.model_bytes() } {
        Ok(b) => b,
        Err(e) => {
            let _ = writeln!(
                arch::serial::Serial,
                "Stage 11 MP1: post-load model_bytes failed: {:?}",
                e
            );
            return None;
        }
    };
    Some((bytes, KIMI_K26_NVME_BYTES))
}

#[cfg(target_arch = "x86_64")]
fn kernel_main(boot_info: &'static mut BootInfo) -> ! {
    // ── T0: capture boot timestamp FIRST ──
    // This is the earliest possible rdtsc in kernel code.
    // Used by bench::report_boot_time() to measure boot-to-ready.
    unsafe {
        bench::BOOT_TSC_T0 = arch::cycles::rdtsc_serialized();
    }

    // XSAVE / XCR0 must come BEFORE any code that could touch SSE / AVX /
    // AVX-512 state. With kernel/.cargo/config.toml now enabling AVX-512
    // as a global target-feature, LLVM auto-vectorises framebuffer
    // memset, byte-copy loops, and similar across all .text — including
    // every function we call below. If we initialised the framebuffer or
    // serial console first, the auto-vectorised memcpy/memset would
    // emit a `vmovups %ymm0, (...)` before XCR0 was set, triggering #UD
    // on the very first store.
    //
    // `enable_fpu_simd` only touches CR4.OSXSAVE / XSETBV — no SIMD
    // registers itself — so it is safe to invoke before any console
    // output (any failure is silent until print is online below, which
    // is acceptable on the boot path).
    let _simd_state_early = unsafe { arch::cpuinfo::enable_fpu_simd() };

    // FB console next — so every subsequent Serial::write_str is
    // mirrored to the on-screen framebuffer. Now safe to memset the
    // framebuffer with whatever vector width LLVM picked.
    arch::fb_console::init(boot_info);

    arch::serial::init();
    arch::serial::println("");
    arch::serial::println("Zero v0.0.1");
    arch::serial::println("");

    // CPU identification — shows hardware info on serial console.
    // Essential for benchmarks: proves which CPU the results came from.
    let cpu_info = arch::cpuinfo::detect();
    arch::cpuinfo::print_info(&cpu_info);
    // Re-report SIMD state now that the serial console is live.
    let simd_state = _simd_state_early;
    let _ = writeln!(
        arch::serial::Serial,
        "Stage 0: SIMD/XSTATE enabled — XCR0=0x{:x}, AVX={}, AVX-512={}",
        simd_state.xcr0,
        simd_state.avx_enabled,
        simd_state.avx512_enabled
    );

    arch::serial::println("Stage 0: kernel online.");
    arch::serial::println("Running on bootloader 0.11 (migrated from 0.9).");

    // ── IPMI Watchdog Timer disable (Supermicro BMC / Cherry Servers) ──
    //
    // The BMC firmware may have a watchdog timer configured in BIOS that
    // fires a hard power-cycle if the OS doesn't pet it within ~60-120s.
    // Our boot path (NVMe bulk_read ≈45s + GGUF parse + heartbeat) can
    // exceed that timeout. Disable the watchdog via the KCS (Keyboard
    // Controller Style) IPMI interface before any heavy lifting begins.
    //
    // Protocol: IPMI 2.0 §9.6 "Set Watchdog Timer" (NetFn=App, Cmd=0x24)
    // with all timer fields zeroed = timer disabled.
    //
    // KCS I/O ports on Supermicro: data=0xCA2, status/cmd=0xCA3.
    #[cfg(all(target_arch = "x86_64", feature = "cherry-net"))]
    {
        let ok = ipmi_kcs_disable_watchdog();
        if ok {
            arch::serial::println("Stage 0: IPMI watchdog timer disabled via KCS");
        } else {
            arch::serial::println(
                "Stage 0: IPMI watchdog disable FAILED (KCS timeout — BMC may not be present)",
            );
        }
    }

    // Stage 1 — GDT + TSS, then IDT. Order matters (see gdt/interrupts).
    arch::gdt::init();
    arch::serial::println("Stage 1: GDT + TSS loaded.");

    arch::interrupts::init();
    arch::serial::println("Stage 1: IDT loaded, CPU exceptions 0-21 wired.");

    unsafe { core::arch::asm!("int3", options(nomem, nostack)) };
    arch::serial::println("Stage 1: continued past int3 — breakpoint path OK.");

    // Stage 2 — physical-memory offset, frame allocator, arena.
    memory::init(boot_info).expect("memory::init failed");

    // Stage-2 arena smoke tests — regression checks for the arena
    // replacing the former linked_list_allocator heap.
    {
        let mut arena_guard = memory::KERNEL_ARENA.lock();
        let arena = arena_guard.as_mut().expect("kernel arena not initialized");

        // Test 1: single-value allocation (was: Box::new(42u64))
        let val_ref = arena.alloc(42u64).expect("arena alloc failed");
        let _ = writeln!(
            arch::serial::Serial,
            "\r\nStage 2 test: arena.alloc(42u64) -> *val_ref = {}",
            *val_ref
        );

        // Test 2: slice allocation (was: Vec<u64>(1000))
        let mut source = [0u64; 1000];
        for i in 0..1000u64 {
            source[i as usize] = i;
        }
        let arena_slice = arena
            .alloc_slice_copy(&source)
            .expect("arena slice alloc failed");
        let sum: u64 = arena_slice.iter().sum();
        let _ = writeln!(
            arch::serial::Serial,
            "Stage 2 test: arena slice [0..1000) sum = {} (expected 499500)",
            sum
        );

        // Arena statistics
        let _ = writeln!(
            arch::serial::Serial,
            "Stage 2: arena used = {} bytes / capacity = {} bytes",
            arena.used(),
            arena.capacity()
        );
    }

    // Stage 12 — sandbox manager (excluded from public release, WIP).
    // See docs/discovery/stage-12-sandbox-baseline.md.

    // Stage 3 sub-part 1 — PIC + PIT.
    // The timer IDT entry was registered in `arch::interrupts::init`, so
    // as soon as the PIC is remapped and `sti` runs, IRQ 0 starts
    // reaching our handler. The handler increments `pit::TICKS`
    // and sends EOI to the master PIC.
    arch::pic::init();
    let _ = writeln!(
        arch::serial::Serial,
        "\r\nStage 3: PIC remapped (master IRQ 32-39, slave 40-47)"
    );

    arch::pit::init();
    let _ = writeln!(
        arch::serial::Serial,
        "Stage 3: PIT configured at ~100 Hz, timer counter active"
    );

    arch::interrupts_enable();

    // Sanity: wait ~1 second's worth of ticks and report the delta.
    //
    // On legacy systems the 8259 PIC delivers IRQ 0 → vector 32 and
    // `hlt` wakes on every tick. On UEFI-booted EPYC (and any modern
    // platform that disables the 8259 in favor of IOAPIC/APIC) IRQ 0
    // never reaches us through this path, so `hlt` would block
    // forever. To stay live we cap the wait with a TSC-based deadline:
    // if the PIT counter has not budged within ~100 ms of wall time,
    // assume the legacy timer path is dead and continue boot.
    {
        let tsc_hz = arch::cycles::tsc_hz();
        let timeout_cycles = tsc_hz / 10; // 100 ms budget
        let pit_start = arch::pit::ticks();
        let tsc_start = arch::cycles::rdtsc_serialized();
        loop {
            let now_ticks = arch::pit::ticks();
            if now_ticks >= pit_start + arch::pit::HZ {
                let elapsed = now_ticks - pit_start;
                let _ = writeln!(
                    arch::serial::Serial,
                    "Stage 3 verification: timer counter advanced by {} ticks in ~1s",
                    elapsed
                );
                break;
            }
            if arch::cycles::rdtsc_serialized() - tsc_start >= timeout_cycles
                && now_ticks == pit_start
            {
                let _ = writeln!(
                    arch::serial::Serial,
                    "WARNING: PIT IRQ0 not received within timeout — timer may be inactive on this platform"
                );
                let _ = writeln!(
                    arch::serial::Serial,
                    "  (UEFI/IOAPIC platform? legacy 8259 likely disabled; continuing without PIT-driven wall clock)"
                );
                break;
            }
            // Busy-poll instead of `hlt`: on platforms where the legacy
            // 8259 PIC is masked (e.g. AMD EPYC under UEFI/IOAPIC), no
            // IRQ ever wakes the CPU and `hlt` would block forever,
            // preventing the TSC deadline above from firing.
            core::hint::spin_loop();
        }
    }

    // Stage 3 sub-part 1.5 — Quarks IR validation.
    let _ = writeln!(arch::serial::Serial, "\r\nStage 3: Quarks IR validation");

    const BOOT_IR: &str = include_str!("../programs/boot.ir");
    let _ = writeln!(
        arch::serial::Serial,
        "Stage 3: IR loaded ({} bytes)",
        BOOT_IR.len()
    );

    match quarks_validator::parse(BOOT_IR) {
        Ok(ast) => {
            let _ = writeln!(arch::serial::Serial, "Stage 3: IR parsed OK");
            // type_check subsumes validate_structure for (program ...)
            // IR — it handles fn-definitions, call instructions, and
            // parameter scoping internally. validate_structure only
            // checks bare instructions against the instruction catalog.
            match quarks_validator::type_check(&ast) {
                Ok(stack) => {
                    let _ = writeln!(
                        arch::serial::Serial,
                        "Stage 3: IR type-checked OK, final stack depth = {}",
                        stack.len()
                    );
                    // Stage 3 sub-part 1.6 — Interpreter execution.
                    // V3 Phase 2 final deliverable: "First Quarks
                    // program executed on bare metal."
                    match quarks_interpreter::interpret(&ast) {
                        Ok(value) => {
                            let _ = writeln!(
                                arch::serial::Serial,
                                "Stage 3: Quarks execution result: {:?}",
                                value
                            );
                            let _ = writeln!(
                                arch::serial::Serial,
                                "Stage 3: V3 Phase 2 complete — first Quarks program with function calls executed on bare metal"
                            );
                        }
                        Err(e) => {
                            let _ = writeln!(
                                arch::serial::Serial,
                                "Stage 3: interpreter execution FAILED: {:?}",
                                e
                            );
                        }
                    }
                }
                Err(e) => {
                    let _ = writeln!(arch::serial::Serial, "Stage 3: type-check FAILED: {:?}", e);
                }
            }
        }
        Err(e) => {
            let _ = writeln!(arch::serial::Serial, "Stage 3: parse FAILED: {:?}", e);
        }
    }

    // === Stage 10 MP4: AOT execution path ===
    // Run AOT path side-by-side with interpreter (V3 Z.273-274:
    // interpreter becomes fallback/debugging tool; AOT is primary path).
    aot::run_and_report();

    // === Stage 10 MP6: Performance Regression Suite ===
    // Run interpreter and AOT in BENCH_ITERATIONS-loops, compare cycles.
    // Per ADR-026 MP6 mandate ("measurable, not aspirational"): V3
    // Pillar 1 becomes a measured baseline, not an aspirational claim.
    {
        let _ = writeln!(
            arch::serial::Serial,
            "\r\nStage 10 MP6: Performance Regression Suite"
        );

        // Interpreter loop bench
        let interp_total: Option<u64> = match quarks_validator::parse(BOOT_IR) {
            Ok(ast) => {
                let _ = writeln!(
                    arch::serial::Serial,
                    "Stage 10 MP6: Interpreter loop bench ({} iterations)...",
                    arch::cycles::BENCH_ITERATIONS
                );
                let start = arch::cycles::rdtsc_serialized();
                let mut acc: i64 = 0;
                let mut last_err = false;
                for _ in 0..arch::cycles::BENCH_ITERATIONS {
                    match quarks_interpreter::interpret(&ast) {
                        Ok(value) => {
                            if let quarks_interpreter::Value::Integer(n) = value {
                                acc = acc.wrapping_add(core::hint::black_box(n));
                            }
                        }
                        Err(_) => {
                            last_err = true;
                            break;
                        }
                    }
                }
                let end = arch::cycles::rdtsc_serialized();
                let _ = core::hint::black_box(acc);

                if last_err {
                    let _ = writeln!(
                        arch::serial::Serial,
                        "Stage 10 MP6: interpreter bench FAILED mid-loop"
                    );
                    None
                } else {
                    let total = end.wrapping_sub(start);
                    let per_iter = total / arch::cycles::BENCH_ITERATIONS;
                    let _ = writeln!(
                        arch::serial::Serial,
                        "Stage 10 MP6: Interpreter total = {} cycles, ~{} cycles/iter",
                        total,
                        per_iter
                    );
                    Some(total)
                }
            }
            Err(_) => {
                let _ = writeln!(
                    arch::serial::Serial,
                    "Stage 10 MP6: interpreter bench parse FAILED"
                );
                None
            }
        };

        // AOT loop bench
        let aot_total: Option<u64> = aot::run_bench();

        // Comparison — use per-iteration cycles since iteration counts differ
        if let (Some(it), Some(at)) = (interp_total, aot_total) {
            let interp_per = it / arch::cycles::BENCH_ITERATIONS;
            let aot_per = at / arch::cycles::BENCH_ITERATIONS;
            if aot_per > 0 {
                let ratio_x10 = interp_per.checked_mul(10).map(|n| n / aot_per).unwrap_or(0);
                let ratio_int = ratio_x10 / 10;
                let ratio_dec = ratio_x10 % 10;
                let _ = writeln!(
                    arch::serial::Serial,
                    "Stage 10 MP6: Per-iteration: Interpreter ~{} cycles, AOT ~{} cycles",
                    interp_per,
                    aot_per
                );
                let _ = writeln!(
                    arch::serial::Serial,
                    "Stage 10 MP6: AOT is {}.{}x faster than Interpreter",
                    ratio_int,
                    ratio_dec
                );
                let _ = writeln!(
                    arch::serial::Serial,
                    "Stage 10 MP6: NOTE — QEMU rdtsc is emulated; ratio is meaningful, absolute cycles are synthetic"
                );
            }
        }
    }

    // === Stage 10.7: PCIe enumeration (all buses, legacy CF8/CFC) ===
    // Walks every PCI bus 0..=255. Required for AMD EPYC multi-IOMS
    // topologies where the NIC and other devices sit on non-zero buses
    // (e.g. the Cherry Server's Intel 82545EM is downstream of one of
    // the EPYC 9354P's secondary root complexes, not on bus 0).
    // Also used by Stage 12 HAL as a "real GPU present?" hint — when
    // zero NVIDIA devices are found, the NVML provider stays in mock
    // mode. MCFG/MMIO config space remains deferred.
    let pci_scan = {
        let _ = writeln!(
            arch::serial::Serial,
            "\r\nStage 10.7: PCIe all-bus enumeration"
        );
        let scan = arch::pcie::scan_all_buses();
        arch::pcie::report(&scan);
        scan
    };

    // === Stage 10.7b: AMD-Vi IOMMU bypass ===
    // AMD platforms (Cherry Server: EPYC 9354P) leave AMD-Vi enabled
    // after POST. With translation on and no AMD-Vi driver installed,
    // every device-initiated DMA is silently mis-translated by the
    // default device-table — the X710 (i40e) hits this when reading
    // its host-resident HMC Page Descriptor Table, blocking data-path
    // bring-up. Until the kernel ships a real AMD-Vi driver we
    // disable translation entirely so DMA goes direct to physical RAM.
    //
    // Soft failure: any error is logged and boot continues. On Intel /
    // VM platforms with no IVRS this returns quickly with no effect.
    {
        let _ = writeln!(
            arch::serial::Serial,
            "\r\nStage 10.7b: AMD-Vi IOMMU bypass (pre-DMA)"
        );
        match memory::phys_offset() {
            Some(phys_off) => match unsafe { arch::x86_64::iommu::disable_amd_vi(phys_off) } {
                Ok(report) => {
                    let _ = writeln!(
                        arch::serial::Serial,
                        "iommu: disabled {} AMD IOMMU(s) — DMA bypasses translation",
                        report.count
                    );
                }
                Err(e) => {
                    let _ = writeln!(
                        arch::serial::Serial,
                        "iommu: AMD-Vi disable skipped ({:?}) — not an AMD-Vi platform, or no IVRS",
                        e
                    );
                }
            },
            None => {
                let _ = writeln!(
                    arch::serial::Serial,
                    "iommu: phys_offset not initialised — IOMMU bypass skipped"
                );
            }
        }
    }

    // === Stage 10.8: Network stack (e1000 + ARP + IPv4 + ICMP + UDP + TCP) ===
    // Brings up a polling-only IP stack on the e1000 NIC. Active
    // profile (selected at compile time) chooses the static IP — see
    // `net::PROFILE_LABEL`. Responder surfaces:
    //   * ICMP echo (ping).
    //   * UDP shell on port 9999  — `nc -u <ip> 9999`.
    //   * TCP shell on port 2222  — `nc    <ip> 2222`  (single connection,
    //     line-buffered, unencrypted; bring-up surface only).
    // Drops to "no NIC" diagnostic when no supported Intel NIC
    // (e1000 / 8254x family or i40e / 700-series) is present.
    let _ = writeln!(arch::serial::Serial, "\r\nStage 10.8: network bring-up");
    //
    // The stack is registered with the timer ISR *before* any
    // long-running synchronous work (SMP bring-up, Stage-11 forward
    // pass) so the TCP/UDP shell surfaces stay reachable on the
    // single-core boot path even while inference is monopolising the
    // BSP. The hook is retracted right before the cooperative
    // executor takes ownership at the end of boot.
    match net::Stack::bind(&pci_scan) {
        Ok(mut stack) => {
            // Drain whatever ARP gratuities have already arrived.
            let n = stack.poll();
            let _ = writeln!(
                arch::serial::Serial,
                "net: initial poll drained {} frame(s)",
                n
            );
            match memory::arena_static_alloc(stack) {
                Ok(r) => {
                    net::register_irq_poll(r);
                    let _ = writeln!(
                        arch::serial::Serial,
                        "net: registered with timer ISR — shells reachable during inference"
                    );
                }
                Err(e) => {
                    let _ = writeln!(
                        arch::serial::Serial,
                        "net: arena alloc FAILED ({:?}) — poll loop disabled",
                        e
                    );
                    // The stack bound, but without the ISR poll hook no
                    // frame is ever serviced — from outside that is
                    // indistinguishable from a bind failure. Overwrite
                    // the ONLINE report so the KVM screen tells the truth.
                    net::bind_report::set(
                        net::bind_report::FAILED,
                        format_args!("arena alloc {:?}; ISR poll disabled", e),
                    );
                }
            }
        }
        Err(e) => {
            let _ = writeln!(
                arch::serial::Serial,
                "net: bring-up SKIPPED ({:?}) — no supported Intel NIC on any PCI bus",
                e
            );
            net::bind_report::set(net::bind_report::FAILED, format_args!("{:?}", e));
        }
    };

    // === ADR-029 Phase 3: SMP multi-core bring-up ===
    //
    // Brings up Application Processors (APs) via ACPI MADT enumeration +
    // INIT-SIPI-SIPI per the AMD64 AP-startup architecture. Once APs are running in
    // long mode, they enter `smp::ap_worker_loop` and wait for parallel-
    // matmul work items published by the BSP.
    //
    // The whole block is feature-gated on `avx512-acceleration` because
    // the SMP layer's first consumer is the parallel AVX-512 matmul
    // dispatcher in `inference_avx512`. A kernel built without the
    // feature stays in BSP-only mode (which is the existing behavior).
    //
    // Failure mode: any error in ACPI parsing or AP bring-up leaves
    // the kernel in BSP-only mode. The forward-pass still runs, just
    // single-threaded — graceful degradation, no panic.
    #[cfg(feature = "avx512-acceleration")]
    {
        let _ = writeln!(
            arch::serial::Serial,
            "\r\nADR-029 Phase 3: SMP bring-up (AP wake-up via INIT-SIPI-SIPI)"
        );

        // Mark the BSP itself in the SMP layer's registration count.
        smp::bsp_register();

        // The physical-memory linear offset was installed during
        // memory::init(). On bootloader_api 0.11 this is the
        // FixedAddress 0xFFFF_8000_0000_0000 we pin in BOOTLOADER_CONFIG.
        let phys_off_opt = memory::phys_offset();
        if let Some(phys_off) = phys_off_opt {
            // Step 1: parse ACPI MADT. Pass the bootloader-forwarded
            // RSDP address as a UEFI fallback for the legacy EBDA /
            // BIOS-ROM scan; strict UEFI firmware (e.g. Cherry Server)
            // leaves the legacy regions empty and only the EFI System
            // Table holds a valid RSDP pointer.
            let bootloader_rsdp = boot_info.rsdp_addr.into_option();
            match unsafe { arch::x86_64::acpi::parse_madt(phys_off, bootloader_rsdp) } {
                Ok(acpi_info) => {
                    let _ = writeln!(
                        arch::serial::Serial,
                        "ADR-029 P3: ACPI MADT — {} CPU(s) reported, LAPIC phys=0x{:x}{}",
                        acpi_info.cpu_count,
                        acpi_info.lapic_phys_base,
                        if acpi_info.lapic_override_applied {
                            " (Type-5 override applied)"
                        } else {
                            ""
                        }
                    );

                    // Step 2: initialize the BSP's LAPIC.
                    let max_apic_id = acpi_info.cpus[..acpi_info.cpu_count as usize]
                        .iter()
                        .filter(|cpu| cpu.enabled)
                        .map(|cpu| cpu.apic_id)
                        .max()
                        .unwrap_or(0);
                    let bsp_apic = unsafe {
                        arch::x86_64::apic::Apic::init(
                            acpi_info.lapic_phys_base,
                            phys_off,
                            max_apic_id,
                        )
                    };
                    let bsp_apic_id = bsp_apic.id();
                    smp::bsp_record_apic_id(bsp_apic_id);
                    let bsp_topology = arch::x86_64::cpuinfo::topology_ids();
                    if bsp_topology.valid {
                        smp::bsp_record_topology(
                            bsp_topology.node_id,
                            bsp_topology.compute_unit_id,
                        );
                    }
                    let _ = writeln!(
                        arch::serial::Serial,
                        "ADR-029 P3: BSP LAPIC online — mode={}, id={}, version=0x{:x}",
                        bsp_apic.mode_label(),
                        bsp_apic_id,
                        bsp_apic.version()
                    );

                    // Step 3: install the AP trampoline at physical 0x8000.
                    if let Err(msg) = arch::x86_64::trampoline::trampoline_self_check() {
                        let _ = writeln!(
                            arch::serial::Serial,
                            "ADR-029 P3: trampoline self-check FAILED: {} — SMP disabled",
                            msg
                        );
                    } else {
                        match arch::x86_64::trampoline::ensure_trampoline_identity_mapped() {
                            Ok(()) => {
                                match unsafe {
                                    arch::x86_64::trampoline::install_trampoline(phys_off)
                                } {
                                    Ok(()) => {
                                        let _ = writeln!(
                                            arch::serial::Serial,
                                            "ADR-029 P3: trampoline installed at phys=0x{:x}",
                                            arch::x86_64::trampoline::TRAMPOLINE_PHYS
                                        );

                                        #[cfg(all(
                                            feature = "cherry-net",
                                            feature = "cherry-smp-debug"
                                        ))]
                                        {
                                            let _ = writeln!(
                                                arch::serial::Serial,
                                                "ADR-029 P3: AP wake-up gated by cherry-smp-debug; TCP shell `smp start` releases INIT-SIPI-SIPI"
                                            );
                                            arch::x86_64::interrupts_enable();
                                            smp::set_ap_boot_gate_active(true);
                                            while !smp::ap_boot_requested() {
                                                arch::x86_64::without_interrupts(|| {
                                                    net::irq_poll_tick();
                                                });
                                                core::hint::spin_loop();
                                            }
                                            smp::set_ap_boot_gate_active(false);
                                            let _ = writeln!(
                                                arch::serial::Serial,
                                                "ADR-029 P3: AP wake-up gate released by TCP shell"
                                            );
                                        }

                                        #[cfg(all(
                                            feature = "cherry-net",
                                            not(feature = "cherry-smp-debug")
                                        ))]
                                        {
                                            smp::request_ap_boot();
                                            let _ = writeln!(
                                                arch::serial::Serial,
                                                "ADR-029 P3: AP wake-up auto-starting; build with cherry-smp-debug to gate INIT-SIPI-SIPI"
                                            );
                                        }

                                        // Step 4: drive INIT-SIPI-SIPI for every AP.
                                        let ap_limit = smp::ap_boot_ap_limit();
                                        let active = unsafe {
                                            arch::x86_64::trampoline::boot_all_aps(
                                                &acpi_info.cpus[..acpi_info.cpu_count as usize],
                                                bsp_apic_id,
                                                &bsp_apic,
                                                phys_off,
                                                ap_limit,
                                            )
                                        };

                                        smp::set_active_cores(active);
                                        let _ = writeln!(
                                            arch::serial::Serial,
                                            "ADR-029 P3: SMP online — {} logical CPU(s) active (BSP + {} AP(s))",
                                            active,
                                            active.saturating_sub(1)
                                        );
                                    }
                                    Err(msg) => {
                                        let _ = writeln!(
                                            arch::serial::Serial,
                                            "ADR-029 P3: trampoline install FAILED: {} — SMP disabled",
                                            msg
                                        );
                                    }
                                }
                            }
                            Err(()) => {
                                let _ = writeln!(
                                    arch::serial::Serial,
                                    "ADR-029 P3: trampoline identity-map missing — SMP disabled"
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = writeln!(
                        arch::serial::Serial,
                        "ADR-029 P3: ACPI MADT parse FAILED: {:?} — SMP disabled",
                        e
                    );
                }
            }
        } else {
            let _ = writeln!(
                arch::serial::Serial,
                "ADR-029 P3: phys_offset not initialized — SMP disabled"
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "zero-control-plane"))]
    {
        bench::capture_pre_llm_baseline(&cpu_info);
        wait_for_boot_llm_start_gate();
    }

    // === Stage 11 MP1: Boot-LLM Foundation ===
    // Stage 11 model discovery. Production Zero Server can load a
    // large resident model from NVMe; the ramdisk/direct-memory paths
    // remain as small-model development fallbacks.
    //
    // Model delivery (in priority order):
    //   1. Ramdisk (production / bare-metal): bootloader 0.11 `set_ramdisk()`
    //      embeds the GGUF file into the boot image and maps it into the
    //      kernel address space. `boot_info.ramdisk_addr` + `ramdisk_len`
    //      point at it.
    //   2. DirectMemory (QEMU dev workflow): `-device loader,addr=0x100000000`
    //      places the legacy small-model GGUF at 4 GiB physical. Only
    //      used if no ramdisk is present AND the 4 GiB region is usable
    //      RAM AND the GGUF magic validates.
    //   3. NVMe (Zero Server): stream the configured resident model
    //      into a virtually-contiguous arena.
    //   4. Absent: Stages 1-10 stay alive; Stage 11 is skipped with a
    //      friendly notice. The rest of the kernel (sandbox, async runtime,
    //      framebuffer, PCIe enumeration) continues normally.
    //
    // V3 Pillar 1 (Performance: zero-copy) + Pillar 4 + Pillar 7.
    'mp1: {
        use crate::model_loader::{
            gguf_payload_view, model_magic_from_bytes, smodel_info, smodel_native_tensor_index,
            smodel_payload_kind_label, DirectMemoryLoader, ModelFormat, ModelLoader, ModelMagic,
            RamdiskLoader, SMODEL_PAYLOAD_KIND_NATIVE,
        };

        let _ = writeln!(
            arch::serial::Serial,
            "\r\nStage 11 MP1: Boot-LLM Foundation (model discovery)"
        );

        // QEMU `-device loader,addr=0x100000000` places model at 4 GiB physical.
        const MODEL_PHYS_ADDR: u64 = 0x1_0000_0000; // 4 GiB
        const MODEL_SIZE: usize = 1_282_439_584; // Qwen3 1.7B Q4_K_M exact bytes

        let phys_offset = match boot_info.physical_memory_offset.into_option() {
            Some(offset) => {
                let _ = writeln!(
                    arch::serial::Serial,
                    "Stage 11 MP1: physical_memory_offset = 0x{:x}",
                    offset
                );
                offset
            }
            None => {
                let _ = writeln!(
                    arch::serial::Serial,
                    "Stage 11 MP1: physical_memory_offset is None — boot config error"
                );
                break 'mp1;
            }
        };

        // Step 1: Decide which loader to use.
        //
        // Ramdisk path (preferred). bootloader 0.11 reports a *virtual*
        // address in `boot_info.ramdisk_addr` — already mapped, no
        // physical-memory dereference gymnastics required.
        let ramdisk_addr = boot_info.ramdisk_addr.into_option();
        let ramdisk_len = boot_info.ramdisk_len;

        let (model_bytes, model_size, source_label): (&'static [u8], usize, &str) = if ramdisk_addr
            .is_some()
            && ramdisk_len > 0
        {
            let _ = writeln!(
                arch::serial::Serial,
                "Stage 11 MP1: ramdisk reported at virt=0x{:x}, len={} bytes",
                ramdisk_addr.unwrap_or(0),
                ramdisk_len
            );
            let rd = match unsafe { RamdiskLoader::from_boot_info(ramdisk_addr, ramdisk_len) } {
                Ok(rd) => rd,
                Err(e) => {
                    let _ = writeln!(
                        arch::serial::Serial,
                        "Stage 11 MP1: RamdiskLoader init FAILED: {:?} — skipping",
                        e
                    );
                    break 'mp1;
                }
            };
            if !rd.looks_like_supported_model() {
                let _ = writeln!(
                    arch::serial::Serial,
                    "Stage 11 MP1: ramdisk present but no GGUF/SILM model marker — skipping"
                );
                break 'mp1;
            }
            let size = rd.size();
            let bytes = match unsafe { rd.model_bytes() } {
                Ok(b) => b,
                Err(e) => {
                    let _ = writeln!(
                        arch::serial::Serial,
                        "Stage 11 MP1: ramdisk model_bytes FAILED: {:?} — skipping",
                        e
                    );
                    break 'mp1;
                }
            };
            (bytes, size, "ramdisk")
        } else {
            // No ramdisk. Two more sources in priority order:
            //   1. DirectMemory  — QEMU dev fixed-phys layout.
            //   2. NVMe          — Cherry production data drive
            //                       (cfg(cherry-net) builds only).
            //
            // The inner `'sources:` block yields the first successful
            // attempt. If both fail we exit Stage 11 cleanly so the
            // kernel still boots without an LLM.
            let attempt: Option<(&'static [u8], usize, &'static str)> = 'sources: {
                // ── 1) DirectMemory ───────────────────────────────────
                let model_end = MODEL_PHYS_ADDR.saturating_add(MODEL_SIZE as u64);
                let mut covered = false;
                for region in boot_info.memory_regions.iter() {
                    if region.start <= MODEL_PHYS_ADDR && region.end >= model_end {
                        covered = true;
                        break;
                    }
                }
                if !covered {
                    let _ = writeln!(
                        arch::serial::Serial,
                        "Stage 11 MP1: DirectMemory phys 0x{:x}..0x{:x} not in memory map — falling through",
                        MODEL_PHYS_ADDR, model_end
                    );
                } else {
                    match unsafe {
                        DirectMemoryLoader::new(MODEL_PHYS_ADDR, phys_offset, MODEL_SIZE)
                    } {
                        Ok(loader) => {
                            let _ = writeln!(
                                arch::serial::Serial,
                                "Stage 11 MP1: DirectMemoryLoader at phys=0x{:x}, size={} MB",
                                MODEL_PHYS_ADDR,
                                MODEL_SIZE / (1024 * 1024)
                            );
                            match unsafe { loader.model_bytes() } {
                                Ok(bytes) => {
                                    let magic = model_magic_from_bytes(bytes);
                                    let _ = writeln!(
                                        arch::serial::Serial,
                                        "Stage 11 MP1: DirectMemory model marker at 0x{:x}: {:?}",
                                        MODEL_PHYS_ADDR,
                                        magic
                                    );
                                    if matches!(magic, ModelMagic::Gguf | ModelMagic::Smodel) {
                                        break 'sources Some((bytes, MODEL_SIZE, "direct-memory"));
                                    }
                                    let _ = writeln!(
                                        arch::serial::Serial,
                                        "Stage 11 MP1: DirectMemory magic mismatch — falling through"
                                    );
                                }
                                Err(e) => {
                                    let _ = writeln!(
                                        arch::serial::Serial,
                                        "Stage 11 MP1: DirectMemory model_bytes FAILED: {:?} — falling through",
                                        e
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            let _ = writeln!(
                                arch::serial::Serial,
                                "Stage 11 MP1: DirectMemoryLoader init FAILED: {:?} — falling through",
                                e
                            );
                        }
                    }
                }

                // ── 2) NVMe (Cherry production) ───────────────────────
                #[cfg(feature = "cherry-net")]
                {
                    #[cfg(feature = "kimi-k26-arena")]
                    {
                        if let Some((bytes, size)) = try_load_nvme_kimi_k26(&pci_scan) {
                            break 'sources Some((bytes, size, "nvme-kimi-k2.6"));
                        }
                    }
                    #[cfg(not(feature = "kimi-k26-arena"))]
                    {
                        let _ = writeln!(
                            arch::serial::Serial,
                            "Stage 11 MP1: NVMe Kimi loader not enabled (rebuild with --cherry --kimi)"
                        );
                    }
                }
                #[cfg(not(feature = "cherry-net"))]
                {
                    let _ = writeln!(
                        arch::serial::Serial,
                        "Stage 11 MP1: NVMe model-load not compiled in (rebuild with --cherry --kimi to enable)"
                    );
                }

                None
            };
            match attempt {
                Some(t) => t,
                None => {
                    let _ = writeln!(
                        arch::serial::Serial,
                        "Stage 11 MP1: no model source available — kernel continues without Boot-LLM"
                    );
                    break 'mp1;
                }
            }
        };

        let _ = writeln!(
            arch::serial::Serial,
            "Stage 11 MP1: model accepted from {} ({} MB)",
            source_label,
            model_size / (1024 * 1024)
        );

        let (model_bytes, model_size, source_label, native_tensor_index): (
            &'static [u8],
            usize,
            &str,
            Option<zero_gguf_parser::TensorIndex>,
        ) = match model_magic_from_bytes(model_bytes) {
            ModelMagic::Smodel => match smodel_info(model_bytes) {
                Ok(info) if info.payload_kind == SMODEL_PAYLOAD_KIND_NATIVE => {
                    let tensor_index = match smodel_native_tensor_index(model_bytes) {
                        Ok(index) => index,
                        Err(e) => {
                            let _ = writeln!(
                                arch::serial::Serial,
                                "Stage 11 MP1: native SilicatePack TensorIndex parse FAILED: {:?} — skipping Boot-LLM",
                                e
                            );
                            break 'mp1;
                        }
                    };
                    let _ = writeln!(
                        arch::serial::Serial,
                        "Stage 11 MP1: native SilicatePack .smodel accepted (kind={}, header={} B, manifest@0x{:x}/{} B, payload={} MB, flags=0x{:x}, tensors={})",
                        smodel_payload_kind_label(info.payload_kind),
                        info.header_len,
                        info.manifest_offset,
                        info.manifest_len,
                        info.payload_len / (1024 * 1024),
                        info.flags,
                        tensor_index.len()
                    );
                    let label = match source_label {
                        "ramdisk" => "ramdisk.smodel-native",
                        "direct-memory" => "direct-memory.smodel-native",
                        "nvme-kimi-k2.6" => "nvme.smodel-native",
                        _ => "smodel-native",
                    };
                    (model_bytes, model_size, label, Some(tensor_index))
                }
                _ => match gguf_payload_view(model_bytes) {
                    Ok(view) => {
                        match view.format {
                            ModelFormat::Gguf => {
                                let _ = writeln!(
                                    arch::serial::Serial,
                                    "Stage 11 MP1: model format raw GGUF compatibility"
                                );
                            }
                            ModelFormat::SmodelGgufCompat => {
                                let _ = writeln!(
                                    arch::serial::Serial,
                                    "Stage 11 MP1: legacy SilicatePack GGUF-compat artifact accepted; payload offset=0x{:x}, size={} MB",
                                    view.payload_offset,
                                    view.size / (1024 * 1024)
                                );
                            }
                            ModelFormat::SmodelNative => {}
                        }
                        let label = match (source_label, view.format) {
                            (_, ModelFormat::Gguf) => source_label,
                            ("ramdisk", ModelFormat::SmodelGgufCompat) => {
                                "ramdisk.smodel-gguf-compat"
                            }
                            ("direct-memory", ModelFormat::SmodelGgufCompat) => {
                                "direct-memory.smodel-gguf-compat"
                            }
                            ("nvme-kimi-k2.6", ModelFormat::SmodelGgufCompat) => {
                                "nvme.smodel-gguf-compat"
                            }
                            (_, ModelFormat::SmodelGgufCompat) => "smodel-gguf-compat",
                            (_, ModelFormat::SmodelNative) => "smodel-native",
                        };
                        (view.bytes, view.size, label, None)
                    }
                    Err(e) => {
                        match smodel_info(model_bytes) {
                            Ok(info) => {
                                let _ = writeln!(
                                    arch::serial::Serial,
                                    "Stage 11 MP1: SilicatePack payload kind {} unsupported by compatibility/native loader — skipping Boot-LLM",
                                    smodel_payload_kind_label(info.payload_kind)
                                );
                            }
                            Err(info_err) => {
                                let _ = writeln!(
                                    arch::serial::Serial,
                                    "Stage 11 MP1: SilicatePack header invalid ({:?}); unwrap error {:?} — skipping Boot-LLM",
                                    info_err,
                                    e
                                );
                            }
                        }
                        break 'mp1;
                    }
                },
            },
            _ => match gguf_payload_view(model_bytes) {
                Ok(view) => {
                    match view.format {
                        ModelFormat::Gguf => {
                            let _ = writeln!(
                                arch::serial::Serial,
                                "Stage 11 MP1: model format raw GGUF compatibility"
                            );
                        }
                        ModelFormat::SmodelGgufCompat => {
                            let _ = writeln!(
                                arch::serial::Serial,
                                "Stage 11 MP1: legacy SilicatePack GGUF-compat artifact accepted; payload offset=0x{:x}, size={} MB",
                                view.payload_offset,
                                view.size / (1024 * 1024)
                            );
                        }
                        ModelFormat::SmodelNative => {}
                    }
                    let label = match (source_label, view.format) {
                        (_, ModelFormat::Gguf) => source_label,
                        ("ramdisk", ModelFormat::SmodelGgufCompat) => "ramdisk.smodel-gguf-compat",
                        ("direct-memory", ModelFormat::SmodelGgufCompat) => {
                            "direct-memory.smodel-gguf-compat"
                        }
                        ("nvme-kimi-k2.6", ModelFormat::SmodelGgufCompat) => {
                            "nvme.smodel-gguf-compat"
                        }
                        (_, ModelFormat::SmodelGgufCompat) => "smodel-gguf-compat",
                        (_, ModelFormat::SmodelNative) => "smodel-native",
                    };
                    (view.bytes, view.size, label, None)
                }
                Err(e) => {
                    let _ = writeln!(
                        arch::serial::Serial,
                        "Stage 11 MP1: model container unwrap FAILED: {:?} — skipping Boot-LLM",
                        e
                    );
                    break 'mp1;
                }
            },
        };

        // ── Hugepage promotion of the model region ──────────────────
        //
        // bootloader 0.11 maps a multi-GiB ramdisk via dynamic 4 KiB
        // pages. For a 1.2 GB GGUF that is ~300 K 4 KiB entries — way
        // beyond the Zen 4 dTLB capacity (L1 dTLB 72 entries 4 KiB,
        // L2 dTLB 3072 entries 4 KiB). Every weight-streaming AVX-512
        // matmul iteration hits TLB misses and pays a 4-level page-
        // walk (~200-300 cycles each) on each new 4 KiB boundary,
        // turning a memory-bandwidth-bound workload into a page-walk-
        // bound one. Empirically discovered via the Zero control
        // plane `mem` command on Cherry (2026-05-17):
        //
        //   model: ramdisk 0x4100..fff (1223 MiB)
        //     first: pa=0x295f8000 page=4 KiB cache=WB
        //     last : pa=0x75cff99f page=4 KiB cache=WB
        //
        // Fix: the bootloader's phys-mem-linear map (PHYS_OFFSET =
        // 0xFFFF_8000_0000_0000) is installed with 1 GiB / 2 MiB
        // pages where alignment permits. If the ramdisk is physically
        // contiguous (it is, when bootloader allocates from one usable
        // region), we can re-expose the same bytes via that already-
        // hugepage-mapped alternate VA without copying or installing
        // any new page tables. Net effect on TLB: ~300 K entries →
        // ~600 (2 MiB pages) or ~2 (1 GiB pages).
        #[cfg(target_arch = "x86_64")]
        let model_bytes: &'static [u8] = {
            // VA the `.smodel` directory was parsed at — the interleave
            // registry (weight_layout) is keyed to this base. If the
            // promotion below moves the payload to a new VA, the registry
            // must be rebased by the same delta or every `group_of`
            // lookup misses and the plain kernel misreads interleaved
            // weights (NaN at the first matmul).
            let pre_promotion_base = model_bytes.as_ptr() as u64;
            let view = memory::contiguous_phys_view(model_bytes.as_ptr() as u64, model_size);
            match view {
                memory::HugepageViewOutcome::Promoted {
                    new_va,
                    phys_base,
                    page_size,
                } => {
                    let _ = writeln!(
                        arch::serial::Serial,
                        "Stage 11 MP1: model remapped via phys-linear VA=0x{:x} pa=0x{:x} page={} KiB",
                        new_va,
                        phys_base,
                        page_size / 1024
                    );
                    // The payload moved VA → rebase the interleave
                    // registry so post-promotion weight pointers still
                    // resolve to their row-interleave group. Without this
                    // the AVX-512 matmul runs the plain kernel over
                    // interleaved bytes → garbage activations → NaN.
                    let delta = new_va as i64 - pre_promotion_base as i64;
                    weight_layout::rebase(delta);
                    let _ = writeln!(
                        arch::serial::Serial,
                        "Stage 11 MP1: interleave registry rebased by {:#x} ({} entries) old_base=0x{:x}",
                        delta,
                        weight_layout::count(),
                        pre_promotion_base
                    );
                    // SAFETY: contiguous_phys_view confirmed (a) source
                    // region is physically contiguous, (b) phys-linear
                    // map at `new_va` is one mapping of the same bytes
                    // with hugepage granularity. Lifetime is 'static
                    // because the bootloader's phys-linear map exists
                    // for the kernel's entire lifetime.
                    unsafe { core::slice::from_raw_parts(new_va as *const u8, model_size) }
                }
                memory::HugepageViewOutcome::ContiguousButSmallPages {
                    phys_base,
                    page_size,
                } => {
                    let _ = writeln!(
                        arch::serial::Serial,
                        "Stage 11 MP1: model contiguous (pa=0x{:x}) but phys-linear is {} KiB pages — keeping original VA",
                        phys_base,
                        page_size / 1024
                    );
                    model_bytes
                }
                memory::HugepageViewOutcome::NotContiguous { broken_at_offset } => {
                    let _ = writeln!(
                        arch::serial::Serial,
                        "Stage 11 MP1: model not physically contiguous (break at 0x{:x}) — keeping original VA",
                        broken_at_offset
                    );
                    model_bytes
                }
                memory::HugepageViewOutcome::Unmapped => {
                    let _ = writeln!(
                        arch::serial::Serial,
                        "Stage 11 MP1: model VA not mappable for hugepage promotion — keeping original"
                    );
                    model_bytes
                }
            }
        };

        control_plane::record_model_region(
            model_bytes.as_ptr() as u64,
            model_size as u64,
            source_label,
        );

        // Phase-E perf audit: emit the page-size of the model region.
        // Should print 1048576 KiB (1 GiB) or 2048 KiB (2 MiB) after
        // the hugepage promotion above. Any regression back to 4 KiB
        // signals that contiguous_phys_view fell through and TLB
        // pressure has returned.
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(info) = memory::mapping_info(model_bytes.as_ptr() as u64) {
                let _ = writeln!(
                    arch::serial::Serial,
                    "Stage 11 MP1: model_bytes page size = {} KiB (huge={})",
                    info.frame_size / 1024,
                    info.huge
                );
            } else {
                let _ = writeln!(
                    arch::serial::Serial,
                    "Stage 11 MP1: model_bytes mapping_info unavailable"
                );
            }
        }

        // Step 3: GGUF Selective Parse + Tensor Index (MP2.1)
        // Per ADR-029 D2: skip tokenizer arrays, build TensorIndex in RUNTIME_ARENA.
        // Replaces MP1's manual zero-alloc header reading with full selective parse.
        let _ = writeln!(
            arch::serial::Serial,
            "\r\n[MP2.1] Selective parser starting..."
        );

        let parsed_tensor_index: Result<
            zero_gguf_parser::TensorIndex,
            zero_gguf_parser::GgufError,
        > = match native_tensor_index {
            Some(index) => Ok(index),
            None => zero_gguf_parser::parse_selective(model_bytes),
        };

        match parsed_tensor_index {
            Ok(tensor_index) => {
                let _ = writeln!(
                    arch::serial::Serial,
                    "[MP2.1] Tensor count: {}",
                    tensor_index.len()
                );

                // Telemetry label: surface the producer's `general.name`
                // (e.g. "Kimi K2.6", "Qwen Qwen3 1.7B") instead of the
                // hardcoded Qwen3 placeholder. The x86_64 path has no
                // framebuffer telemetry panel — the override is read by
                // the TCP `inference` command via effective_model_label.
                if let Some(name) = tensor_index.model_name.as_deref() {
                    unsafe {
                        lfb::telemetry_data::set_model_label(name);
                    }
                }

                // ModelConfig dump (split to avoid f32 Display + multi-arg monomorphization)
                let cfg = &tensor_index.model_config;
                let _ = write!(arch::serial::Serial, "[MP2.1] ModelConfig: arch=");
                let _ = write!(arch::serial::Serial, "{}", cfg.architecture);
                let _ = write!(arch::serial::Serial, " blocks=");
                let _ = write!(arch::serial::Serial, "{}", cfg.block_count);
                let _ = write!(arch::serial::Serial, " hidden=");
                let _ = write!(arch::serial::Serial, "{}", cfg.embedding_length);
                let _ = write!(arch::serial::Serial, " ffn=");
                let _ = write!(arch::serial::Serial, "{}", cfg.feed_forward_length);
                let _ = write!(arch::serial::Serial, " heads=");
                let _ = write!(arch::serial::Serial, "{}", cfg.head_count);
                let _ = write!(arch::serial::Serial, " kv_heads=");
                let _ = write!(arch::serial::Serial, "{}", cfg.head_count_kv);
                let _ = write!(arch::serial::Serial, " key_len=");
                let _ = write!(arch::serial::Serial, "{}", cfg.key_length);
                let _ = write!(arch::serial::Serial, " val_len=");
                let _ = write!(arch::serial::Serial, "{}", cfg.value_length);
                let _ = write!(arch::serial::Serial, " rope_freq=");
                print_f32(cfg.rope_freq_base);
                let _ = write!(arch::serial::Serial, " rms_eps=");
                print_f32(cfg.layer_norm_rms_epsilon);
                let _ = writeln!(arch::serial::Serial, "");
                let _ = writeln!(
                    arch::serial::Serial,
                    "[MP2.1] TensorIndex built: {} entries, tensor_data_offset=0x{:x}",
                    tensor_index.len(),
                    tensor_index.tensor_data_offset
                );

                if cfg.is_deepseek2() {
                    log_deepseek2_boot_heartbeat(&tensor_index);

                    // ── MLA config debug dump ──
                    // Primitive single-value prints only — no {:?} to
                    // avoid core::fmt::Debug monomorphization bloat.
                    let _ = write!(arch::serial::Serial, "[MLA-DBG] kv_lora_rank=");
                    match cfg.kv_lora_rank {
                        Some(v) => {
                            let _ = writeln!(arch::serial::Serial, "{}", v);
                        }
                        None => {
                            let _ = writeln!(arch::serial::Serial, "None");
                        }
                    }
                    let _ = write!(arch::serial::Serial, "[MLA-DBG] q_lora_rank=");
                    match cfg.q_lora_rank {
                        Some(v) => {
                            let _ = writeln!(arch::serial::Serial, "{}", v);
                        }
                        None => {
                            let _ = writeln!(arch::serial::Serial, "None");
                        }
                    }
                    let _ = write!(arch::serial::Serial, "[MLA-DBG] qk_rope_head_dim=");
                    match cfg.qk_rope_head_dim {
                        Some(v) => {
                            let _ = writeln!(arch::serial::Serial, "{}", v);
                        }
                        None => {
                            let _ = writeln!(arch::serial::Serial, "None");
                        }
                    }
                    let _ = write!(arch::serial::Serial, "[MLA-DBG] qk_nope_head_dim=");
                    match cfg.qk_nope_head_dim {
                        Some(v) => {
                            let _ = writeln!(arch::serial::Serial, "{}", v);
                        }
                        None => {
                            let _ = writeln!(arch::serial::Serial, "None");
                        }
                    }
                    let _ = write!(arch::serial::Serial, "[MLA-DBG] v_head_dim=");
                    match cfg.v_head_dim {
                        Some(v) => {
                            let _ = writeln!(arch::serial::Serial, "{}", v);
                        }
                        None => {
                            let _ = writeln!(arch::serial::Serial, "None");
                        }
                    }
                    let _ = write!(arch::serial::Serial, "[MLA-DBG] expert_count=");
                    match cfg.expert_count {
                        Some(v) => {
                            let _ = writeln!(arch::serial::Serial, "{}", v);
                        }
                        None => {
                            let _ = writeln!(arch::serial::Serial, "None");
                        }
                    }
                    let _ = write!(arch::serial::Serial, "[MLA-DBG] expert_used_count=");
                    match cfg.expert_used_count {
                        Some(v) => {
                            let _ = writeln!(arch::serial::Serial, "{}", v);
                        }
                        None => {
                            let _ = writeln!(arch::serial::Serial, "None");
                        }
                    }
                    let _ = write!(arch::serial::Serial, "[MLA-DBG] vocab_size=");
                    match cfg.vocab_size {
                        Some(v) => {
                            let _ = writeln!(arch::serial::Serial, "{}", v);
                        }
                        None => {
                            let _ = writeln!(arch::serial::Serial, "None");
                        }
                    }
                    let _ = write!(arch::serial::Serial, "[MLA-DBG] key_length=");
                    let _ = write!(arch::serial::Serial, "{}", cfg.key_length);
                    let _ = write!(arch::serial::Serial, " value_length=");
                    let _ = writeln!(arch::serial::Serial, "{}", cfg.value_length);
                    let _ = write!(arch::serial::Serial, "[MLA-DBG] key_length_mla=");
                    match cfg.key_length_mla {
                        Some(v) => {
                            let _ = writeln!(arch::serial::Serial, "{}", v);
                        }
                        None => {
                            let _ = writeln!(arch::serial::Serial, "None");
                        }
                    }
                    let _ = write!(arch::serial::Serial, "[MLA-DBG] value_length_mla=");
                    match cfg.value_length_mla {
                        Some(v) => {
                            let _ = writeln!(arch::serial::Serial, "{}", v);
                        }
                        None => {
                            let _ = writeln!(arch::serial::Serial, "None");
                        }
                    }
                    let _ = writeln!(arch::serial::Serial, "[MLA-DBG] config dump complete");
                } else {
                    // Q6_K tensor dump (M3 — expected 29 per discovery v1)
                    let _ = writeln!(
                        arch::serial::Serial,
                        "[MP2.1] Q6_K tensors in model (expected: 29):"
                    );
                    let mut q6k_count = 0u32;
                    for info in tensor_index.iter_by_type(zero_gguf_parser::GgmlType::Q6K) {
                        q6k_count += 1;
                        let _ = write!(arch::serial::Serial, "  [{:03}] ", q6k_count);
                        let _ = writeln!(arch::serial::Serial, "{}", info.name);
                    }
                    let _ = write!(arch::serial::Serial, "[MP2.1] Q6_K count: ");
                    let _ = write!(arch::serial::Serial, "{}", q6k_count);
                    if q6k_count == 29 {
                        let _ = writeln!(arch::serial::Serial, " ok");
                    } else {
                        // The 29-Q6_K census is a GGUF-Q4_K_M expectation.
                        // A native .smodel emits Q4_0/Q8_0 (optionally
                        // row-interleaved) and legitimately has zero Q6_K
                        // tensors — report the native census instead of
                        // leaving a misleading MISMATCH on the KVM screen.
                        let mut q4_0_count = 0u32;
                        for _ in tensor_index.iter_by_type(zero_gguf_parser::GgmlType::Q4_0) {
                            q4_0_count += 1;
                        }
                        let mut q8_0_count = 0u32;
                        for _ in tensor_index.iter_by_type(zero_gguf_parser::GgmlType::Q8_0) {
                            q8_0_count += 1;
                        }
                        if q4_0_count > 0 || q8_0_count > 0 {
                            let _ = writeln!(
                                arch::serial::Serial,
                                " (native profile: Q4_0={}, Q8_0={}, interleaved={} — Q6_K census not applicable) ok",
                                q4_0_count,
                                q8_0_count,
                                weight_layout::count(),
                            );
                        } else {
                            let _ = writeln!(arch::serial::Serial, " MISMATCH (expected: 29)");
                        }
                    }

                    // Heartbeat lookups (M4)
                    let _ = writeln!(arch::serial::Serial, "[MP2.1] Heartbeat lookup test:");
                    log_tensor_summary(&tensor_index, "output.weight");
                    log_tensor_summary(&tensor_index, "blk.0.attn_v.weight");
                    let _ = writeln!(arch::serial::Serial, "[MP2.1] Lookup successful ok");

                    // Step 4: Q4_K Dequant Heartbeat (MP2.2a)
                    // Per ADR-029 D5: caller-allocated output buffer.
                    // Per ADR-028 v5: single-arg writeln to avoid core::fmt bloat.
                    let _ = writeln!(
                        arch::serial::Serial,
                        "\r\n[MP2.2a] Q4_K dequant heartbeat starting..."
                    );
                    if let Some(t) = tensor_index.get("blk.0.attn_q.weight") {
                        if t.tensor_type == zero_gguf_parser::GgmlType::Q4K {
                            let block_ptr_offset =
                                tensor_index.tensor_data_offset + (t.offset as usize);
                            if block_ptr_offset + 144 <= model_bytes.len() {
                                let block_bytes: &[u8; 144] = unsafe {
                                    &*(model_bytes.as_ptr().add(block_ptr_offset)
                                        as *const [u8; 144])
                                };
                                let mut output = [0.0f32; 256];
                                zero_gguf_parser::dequant::dequant_q4k_block(
                                    block_bytes,
                                    &mut output,
                                );

                                let _ = writeln!(arch::serial::Serial, "[MP2.2a] First 8 dequantized values from blk.0.attn_q.weight block 0:");
                                let mut i = 0u32;
                                while i < 8 {
                                    let _ = write!(arch::serial::Serial, "  [");
                                    let _ = write!(arch::serial::Serial, "{}", i);
                                    let _ = write!(arch::serial::Serial, "]: ");
                                    print_f32(output[i as usize]);
                                    let _ = writeln!(arch::serial::Serial, "");
                                    i += 1;
                                }

                                // Plausibility check
                                let mut nan_count = 0u32;
                                let mut inf_count = 0u32;
                                let mut oor_count = 0u32;
                                for &v in output.iter() {
                                    if v.is_nan() {
                                        nan_count += 1;
                                    } else if v.is_infinite() {
                                        inf_count += 1;
                                    } else if v < -1.0 || v > 1.0 {
                                        oor_count += 1;
                                    }
                                }
                                let _ = write!(arch::serial::Serial, "[MP2.2a] Plausibility NaN: ");
                                let _ = writeln!(arch::serial::Serial, "{}", nan_count);
                                let _ = write!(arch::serial::Serial, "[MP2.2a] Plausibility Inf: ");
                                let _ = writeln!(arch::serial::Serial, "{}", inf_count);
                                let _ = write!(
                                    arch::serial::Serial,
                                    "[MP2.2a] Plausibility out-of-[-1,1]: "
                                );
                                let _ = writeln!(arch::serial::Serial, "{}", oor_count);
                                if nan_count == 0 && inf_count == 0 && oor_count < 50 {
                                    let _ = writeln!(
                                        arch::serial::Serial,
                                        "[MP2.2a] Q4_K dequant heartbeat: PLAUSIBLE ok"
                                    );
                                } else if nan_count > 0 || inf_count > 0 {
                                    let _ = writeln!(
                                        arch::serial::Serial,
                                        "[MP2.2a] Q4_K dequant heartbeat: IMPLAUSIBLE (NaN or Inf)"
                                    );
                                } else {
                                    let _ = writeln!(
                                        arch::serial::Serial,
                                        "[MP2.2a] Q4_K dequant heartbeat: WARN (many out-of-range)"
                                    );
                                }
                            } else {
                                let _ = writeln!(
                                    arch::serial::Serial,
                                    "[MP2.2a] block_ptr_offset out of bounds"
                                );
                            }
                        } else if t.tensor_type == zero_gguf_parser::GgmlType::Q4_0 {
                            // Native .smodel profile: attn_q is Q4_0 (plain or
                            // 4-row-interleaved). Dequant row 0 / K-block 0 —
                            // for the interleaved layout that block's scale
                            // lives at group-block bytes [0..2] and its 16
                            // nibble bytes at [8..24] (lane 0).
                            let off = tensor_index.tensor_data_offset + (t.offset as usize);
                            let group =
                                weight_layout::group_of(
                                    unsafe { model_bytes.as_ptr().add(off) } as usize
                                );
                            if off + 24 <= model_bytes.len() {
                                let mut blk = [0u8; 18];
                                if group == 4 {
                                    blk[0..2].copy_from_slice(&model_bytes[off..off + 2]);
                                    blk[2..18].copy_from_slice(&model_bytes[off + 8..off + 24]);
                                } else {
                                    blk.copy_from_slice(&model_bytes[off..off + 18]);
                                }
                                let mut out32 = [0.0f32; 32];
                                zero_gguf_parser::dequant::dequant_q4_0_row(&blk, &mut out32, 1);
                                let mut bad = 0u32;
                                for &v in out32.iter() {
                                    if v.is_nan() || v.is_infinite() || !(-2.0..=2.0).contains(&v) {
                                        bad += 1;
                                    }
                                }
                                let _ = write!(
                                    arch::serial::Serial,
                                    "[MP2.2a] native Q4_0 dequant heartbeat (interleave={}): first values ",
                                    group
                                );
                                print_f32(out32[0]);
                                let _ = write!(arch::serial::Serial, " ");
                                print_f32(out32[1]);
                                if bad == 0 {
                                    let _ = writeln!(arch::serial::Serial, " — PLAUSIBLE ok");
                                } else {
                                    let _ = writeln!(
                                        arch::serial::Serial,
                                        " — {} implausible value(s)",
                                        bad
                                    );
                                }
                            } else {
                                let _ = writeln!(
                                    arch::serial::Serial,
                                    "[MP2.2a] native Q4_0 block out of bounds"
                                );
                            }
                        } else {
                            let _ = writeln!(
                                arch::serial::Serial,
                                "[MP2.2a] blk.0.attn_q.weight is NOT Q4_K (type id {})",
                                t.tensor_type as u32
                            );
                        }
                    } else {
                        let _ = writeln!(
                            arch::serial::Serial,
                            "[MP2.2a] blk.0.attn_q.weight NOT FOUND"
                        );
                    }

                    // Step 5: Q6_K Dequant Heartbeat (MP2.2b)
                    // output.weight is Q6_K (243 MB, largest tensor). Pattern C (half-half).
                    // Per ADR-028 v6: type-aware single-arg writeln, no f32::fmt.
                    let _ = writeln!(
                        arch::serial::Serial,
                        "\r\n[MP2.2b] Q6_K dequant heartbeat starting..."
                    );
                    if let Some(t) = tensor_index.get("output.weight") {
                        if t.tensor_type == zero_gguf_parser::GgmlType::Q6K {
                            let block_ptr_offset =
                                tensor_index.tensor_data_offset + (t.offset as usize);
                            if block_ptr_offset + 210 <= model_bytes.len() {
                                let block_bytes: &[u8; 210] = unsafe {
                                    &*(model_bytes.as_ptr().add(block_ptr_offset)
                                        as *const [u8; 210])
                                };
                                let mut output = [0.0f32; 256];
                                zero_gguf_parser::dequant::dequant_q6k_block(
                                    block_bytes,
                                    &mut output,
                                );

                                let _ = writeln!(
                                arch::serial::Serial,
                                "[MP2.2b] First 16 dequantized values from output.weight block 0:"
                            );
                                let mut i = 0u32;
                                while i < 16 {
                                    let _ = write!(arch::serial::Serial, "  [");
                                    let _ = write!(arch::serial::Serial, "{}", i);
                                    let _ = write!(arch::serial::Serial, "]: ");
                                    print_f32(output[i as usize]);
                                    let _ = writeln!(arch::serial::Serial, "");
                                    i += 1;
                                }

                                let mut nan_count = 0u32;
                                let mut inf_count = 0u32;
                                let mut oor_count = 0u32;
                                for &v in output.iter() {
                                    if v.is_nan() {
                                        nan_count += 1;
                                    } else if v.is_infinite() {
                                        inf_count += 1;
                                    } else if v < -1.0 || v > 1.0 {
                                        oor_count += 1;
                                    }
                                }
                                let _ = write!(arch::serial::Serial, "[MP2.2b] Plausibility NaN: ");
                                let _ = writeln!(arch::serial::Serial, "{}", nan_count);
                                let _ = write!(arch::serial::Serial, "[MP2.2b] Plausibility Inf: ");
                                let _ = writeln!(arch::serial::Serial, "{}", inf_count);
                                let _ = write!(
                                    arch::serial::Serial,
                                    "[MP2.2b] Plausibility out-of-[-1,1]: "
                                );
                                let _ = writeln!(arch::serial::Serial, "{}", oor_count);
                                if nan_count == 0 && inf_count == 0 && oor_count < 50 {
                                    let _ = writeln!(
                                        arch::serial::Serial,
                                        "[MP2.2b] Q6_K dequant heartbeat: PLAUSIBLE ok"
                                    );
                                } else if nan_count > 0 || inf_count > 0 {
                                    let _ = writeln!(
                                        arch::serial::Serial,
                                        "[MP2.2b] Q6_K dequant heartbeat: IMPLAUSIBLE (NaN or Inf)"
                                    );
                                } else {
                                    let _ = writeln!(
                                        arch::serial::Serial,
                                        "[MP2.2b] Q6_K dequant heartbeat: WARN (many out-of-range)"
                                    );
                                }
                            } else {
                                let _ = writeln!(
                                    arch::serial::Serial,
                                    "[MP2.2b] block_ptr_offset out of bounds"
                                );
                            }
                        } else if t.tensor_type == zero_gguf_parser::GgmlType::Q8_0 {
                            // Native .smodel profile: output.weight is Q8_0
                            // (the LM head ships 4-row-interleaved as
                            // Q8_0X4). Row 0 / K-block 0: scale at group
                            // bytes [0..2], its 32 i8 values at [8..40].
                            let off = tensor_index.tensor_data_offset + (t.offset as usize);
                            let group =
                                weight_layout::group_of(
                                    unsafe { model_bytes.as_ptr().add(off) } as usize
                                );
                            if off + 40 <= model_bytes.len() {
                                let mut blk = [0u8; 34];
                                if group == 4 {
                                    blk[0..2].copy_from_slice(&model_bytes[off..off + 2]);
                                    blk[2..34].copy_from_slice(&model_bytes[off + 8..off + 40]);
                                } else {
                                    blk.copy_from_slice(&model_bytes[off..off + 34]);
                                }
                                let mut out32 = [0.0f32; 32];
                                zero_gguf_parser::dequant::dequant_q8_0_row(&blk, &mut out32, 1);
                                let mut bad = 0u32;
                                for &v in out32.iter() {
                                    if v.is_nan() || v.is_infinite() || !(-2.0..=2.0).contains(&v) {
                                        bad += 1;
                                    }
                                }
                                let _ = write!(
                                    arch::serial::Serial,
                                    "[MP2.2b] native Q8_0 dequant heartbeat (interleave={}): first values ",
                                    group
                                );
                                print_f32(out32[0]);
                                let _ = write!(arch::serial::Serial, " ");
                                print_f32(out32[1]);
                                if bad == 0 {
                                    let _ = writeln!(arch::serial::Serial, " — PLAUSIBLE ok");
                                } else {
                                    let _ = writeln!(
                                        arch::serial::Serial,
                                        " — {} implausible value(s)",
                                        bad
                                    );
                                }
                            } else {
                                let _ = writeln!(
                                    arch::serial::Serial,
                                    "[MP2.2b] native Q8_0 block out of bounds"
                                );
                            }
                        } else {
                            let _ = writeln!(
                                arch::serial::Serial,
                                "[MP2.2b] output.weight is NOT Q6_K (type id {})",
                                t.tensor_type as u32
                            );
                        }
                    } else {
                        let _ = writeln!(arch::serial::Serial, "[MP2.2b] output.weight NOT FOUND");
                    }

                    // Step 6: RMSNorm Heartbeat (MP2.3a)
                    // Demonstrates RMSNorm operator on real model data:
                    //   Input: 8 Q4_K-dequantized blocks of blk.0.attn_q.weight = 2048 f32
                    //   Weight: F32 tensor blk.0.attn_norm.weight (direct read)
                    //   Epsilon: ModelConfig.layer_norm_rms_epsilon (Qwen3: 1e-6)
                    //   Output: ACTIVATION_ARENA-allocated 2048 f32
                    // Per ADR-029 D5: caller-allocated output.
                    // Per ADR-028: libm::sqrtf inside rmsnorm (zero-llm-inference crate).
                    // Per ADR-028 v6: print_f32 for float output.
                    let _ = writeln!(
                        arch::serial::Serial,
                        "\r\n[MP2.3a] RMSNorm heartbeat starting..."
                    );

                    let attn_norm_tensor = tensor_index.get("blk.0.attn_norm.weight");
                    let attn_q_tensor = tensor_index.get("blk.0.attn_q.weight");

                    if let (Some(norm_t), Some(q_t)) = (attn_norm_tensor, attn_q_tensor) {
                        if norm_t.tensor_type == zero_gguf_parser::GgmlType::F32
                            && q_t.tensor_type == zero_gguf_parser::GgmlType::Q4K
                        {
                            let hidden_size = tensor_index.model_config.embedding_length as usize;
                            let epsilon = tensor_index.model_config.layer_norm_rms_epsilon;

                            let norm_offset =
                                tensor_index.tensor_data_offset + (norm_t.offset as usize);
                            let norm_bytes = hidden_size * 4; // F32 = 4 bytes each

                            let q_offset = tensor_index.tensor_data_offset + (q_t.offset as usize);
                            let blocks_needed = hidden_size / 256; // 2048/256 = 8
                            let q_bytes_needed = blocks_needed * 144; // 8 × 144 = 1152

                            if norm_offset + norm_bytes <= model_bytes.len()
                                && q_offset + q_bytes_needed <= model_bytes.len()
                            {
                                // F32 weight vector (gamma) — direct read, no dequant
                                let weight_slice: &[f32] = unsafe {
                                    core::slice::from_raw_parts(
                                        model_bytes.as_ptr().add(norm_offset) as *const f32,
                                        hidden_size,
                                    )
                                };
                                // Q4_K blocks for input dequant
                                let q_blocks: &[u8] = unsafe {
                                    core::slice::from_raw_parts(
                                        model_bytes.as_ptr().add(q_offset),
                                        q_bytes_needed,
                                    )
                                };

                                // Allocate from ACTIVATION_ARENA (16 MB, per ADR-029 D4)
                                let mut act_guard = memory::ACTIVATION_ARENA.lock();
                                if let Some(act_arena) = act_guard.as_mut() {
                                    act_arena.reset();

                                    let input_ok = act_arena.alloc_f32_slice(hidden_size);
                                    let output_ok = act_arena.alloc_f32_slice(hidden_size);

                                    if let (Ok(input_buf), Ok(output_buf)) = (input_ok, output_ok) {
                                        // Dequantize Q4_K → f32 input vector
                                        zero_gguf_parser::dequant::dequant_q4k_row(
                                            q_blocks,
                                            input_buf,
                                            blocks_needed,
                                        );

                                        // RMSNorm — from zero-llm-inference (NOT gguf-parser)
                                        zero_llm_inference::rmsnorm(
                                            input_buf,
                                            weight_slice,
                                            output_buf,
                                            epsilon,
                                        );

                                        // Print first 8 input values
                                        let _ = writeln!(
                                            arch::serial::Serial,
                                            "[MP2.3a] First 8 INPUT values (from attn_q dequant):"
                                        );
                                        let mut i = 0u32;
                                        while i < 8 {
                                            let _ = write!(arch::serial::Serial, "  [");
                                            let _ = write!(arch::serial::Serial, "{}", i);
                                            let _ = write!(arch::serial::Serial, "]: ");
                                            print_f32(input_buf[i as usize]);
                                            let _ = writeln!(arch::serial::Serial, "");
                                            i += 1;
                                        }

                                        // Print first 8 output values (RMSNorm-applied)
                                        let _ = writeln!(
                                            arch::serial::Serial,
                                            "[MP2.3a] First 8 OUTPUT values (after RMSNorm):"
                                        );
                                        let mut i = 0u32;
                                        while i < 8 {
                                            let _ = write!(arch::serial::Serial, "  [");
                                            let _ = write!(arch::serial::Serial, "{}", i);
                                            let _ = write!(arch::serial::Serial, "]: ");
                                            print_f32(output_buf[i as usize]);
                                            let _ = writeln!(arch::serial::Serial, "");
                                            i += 1;
                                        }

                                        // Plausibility
                                        let mut nan_count = 0u32;
                                        let mut inf_count = 0u32;
                                        for &v in output_buf.iter() {
                                            if v.is_nan() {
                                                nan_count += 1;
                                            } else if v.is_infinite() {
                                                inf_count += 1;
                                            }
                                        }
                                        let _ =
                                            write!(arch::serial::Serial, "[MP2.3a] Output NaN: ");
                                        let _ = writeln!(arch::serial::Serial, "{}", nan_count);
                                        let _ =
                                            write!(arch::serial::Serial, "[MP2.3a] Output Inf: ");
                                        let _ = writeln!(arch::serial::Serial, "{}", inf_count);
                                        let bytes_used = act_arena.used();
                                        let _ = write!(
                                            arch::serial::Serial,
                                            "[MP2.3a] ACTIVATION_ARENA bytes used: "
                                        );
                                        let _ = writeln!(arch::serial::Serial, "{}", bytes_used);

                                        if nan_count == 0 && inf_count == 0 {
                                            let _ = writeln!(
                                                arch::serial::Serial,
                                                "[MP2.3a] RMSNorm heartbeat: PLAUSIBLE ok"
                                            );
                                        } else {
                                            let _ = writeln!(
                                            arch::serial::Serial,
                                            "[MP2.3a] RMSNorm heartbeat: IMPLAUSIBLE (NaN or Inf)"
                                        );
                                        }
                                    } else {
                                        let _ = writeln!(
                                            arch::serial::Serial,
                                            "[MP2.3a] ACTIVATION_ARENA alloc failed"
                                        );
                                    }
                                } else {
                                    let _ = writeln!(
                                        arch::serial::Serial,
                                        "[MP2.3a] ACTIVATION_ARENA not initialized"
                                    );
                                }
                            } else {
                                let _ = writeln!(
                                    arch::serial::Serial,
                                    "[MP2.3a] tensor offset out of bounds"
                                );
                            }
                        } else if norm_t.tensor_type == zero_gguf_parser::GgmlType::F32
                            && q_t.tensor_type == zero_gguf_parser::GgmlType::Q4_0
                        {
                            // Native .smodel: this heartbeat's input
                            // synthesiser is Q4_K-block-specific. The native
                            // Q4_0 path is exercised end-to-end by the MP3
                            // forward pass (plus MP2.2a above) — skipping is
                            // expected, not a defect.
                            let _ = writeln!(
                                arch::serial::Serial,
                                "[MP2.3a] skipped on native Q4_0 profile (Q4_K-specific heartbeat; MP3 covers the native path) ok"
                            );
                        } else {
                            let _ = writeln!(
                                arch::serial::Serial,
                                "[MP2.3a] tensor type mismatch (norm=F32, q=Q4K expected)"
                            );
                        }
                    } else {
                        let _ = writeln!(
                            arch::serial::Serial,
                            "[MP2.3a] attn_norm or attn_q NOT FOUND"
                        );
                    }

                    // Step 7: RoPE Heartbeat (MP2.3b — RopeContext-based, post-Mode-B-Resolution)
                    // Per V3 Pillar 1: RopeContext built ONCE here, reused per call.
                    // libm::powf once in RopeContext::new(), eliminated from runtime.
                    // Per ADR-028 v7: 14 MB headroom — Mode-B-Boundary resolved.
                    // Per ADR-029 D7: head_dim + freq_base from ModelConfig.
                    let _ = writeln!(
                        arch::serial::Serial,
                        "\r\n[MP2.3b] RoPE heartbeat starting..."
                    );

                    let head_dim = tensor_index.model_config.key_length as usize;
                    let freq_base = tensor_index.model_config.rope_freq_base;

                    let _ = write!(arch::serial::Serial, "[MP2.3b] head_dim from ModelConfig: ");
                    let _ = writeln!(arch::serial::Serial, "{}", head_dim);
                    let _ = write!(
                        arch::serial::Serial,
                        "[MP2.3b] freq_base from ModelConfig: "
                    );
                    print_f32(freq_base);
                    let _ = writeln!(arch::serial::Serial, "");

                    // Build RopeContext once — Pillar 1 performance discipline
                    let rope_ctx = zero_llm_inference::RopeContext::<64>::new(freq_base);

                    let _ = write!(
                        arch::serial::Serial,
                        "[MP2.3b] RopeContext built, inv_freqs[0]: "
                    );
                    print_f32(rope_ctx.inv_freqs[0]);
                    let _ = writeln!(arch::serial::Serial, "");
                    let _ = write!(arch::serial::Serial, "[MP2.3b] RopeContext inv_freqs[63]: ");
                    print_f32(rope_ctx.inv_freqs[63]);
                    let _ = writeln!(arch::serial::Serial, "");

                    {
                        let mut act_guard = memory::ACTIVATION_ARENA.lock();
                        if let Some(act_arena) = act_guard.as_mut() {
                            let bytes_used_pre = act_arena.used();

                            if let Ok(qk_buf) = act_arena.alloc_f32_slice(head_dim) {
                                // Deterministic ramp matching gguf-py reference
                                for i in 0..head_dim {
                                    qk_buf[i] = 0.01 * (i as f32 + 1.0);
                                }

                                let _ = writeln!(
                                    arch::serial::Serial,
                                    "[MP2.3b] First 8 INPUT values (BEFORE RoPE, ramp 0.01..0.08):"
                                );
                                let mut i = 0u32;
                                while i < 8 {
                                    let _ = write!(arch::serial::Serial, "  [");
                                    let _ = write!(arch::serial::Serial, "{}", i);
                                    let _ = write!(arch::serial::Serial, "]: ");
                                    print_f32(qk_buf[i as usize]);
                                    let _ = writeln!(arch::serial::Serial, "");
                                    i += 1;
                                }

                                // position=0 identity check
                                let snapshot_before: [f32; 8] = [
                                    qk_buf[0], qk_buf[1], qk_buf[2], qk_buf[3], qk_buf[4],
                                    qk_buf[5], qk_buf[6], qk_buf[7],
                                ];
                                zero_llm_inference::rope(qk_buf, &rope_ctx, 0);
                                let mut identity_ok = true;
                                for idx in 0..8 {
                                    if (qk_buf[idx] - snapshot_before[idx]).abs() > 1e-6 {
                                        identity_ok = false;
                                    }
                                }
                                if identity_ok {
                                    let _ = writeln!(
                                        arch::serial::Serial,
                                        "[MP2.3b] position=0 identity check: PASS"
                                    );
                                } else {
                                    let _ = writeln!(
                                        arch::serial::Serial,
                                        "[MP2.3b] position=0 identity check: FAIL"
                                    );
                                }

                                // Real rotation at position=1
                                zero_llm_inference::rope(qk_buf, &rope_ctx, 1);

                                let _ = writeln!(
                                    arch::serial::Serial,
                                    "[MP2.3b] First 8 OUTPUT values (AFTER RoPE position=1):"
                                );
                                let mut i = 0u32;
                                while i < 8 {
                                    let _ = write!(arch::serial::Serial, "  [");
                                    let _ = write!(arch::serial::Serial, "{}", i);
                                    let _ = write!(arch::serial::Serial, "]: ");
                                    print_f32(qk_buf[i as usize]);
                                    let _ = writeln!(arch::serial::Serial, "");
                                    i += 1;
                                }

                                // Plausibility
                                let mut nan_count = 0u32;
                                let mut inf_count = 0u32;
                                for &v in qk_buf.iter() {
                                    if v.is_nan() {
                                        nan_count += 1;
                                    } else if v.is_infinite() {
                                        inf_count += 1;
                                    }
                                }
                                let _ = write!(arch::serial::Serial, "[MP2.3b] Output NaN: ");
                                let _ = writeln!(arch::serial::Serial, "{}", nan_count);
                                let _ = write!(arch::serial::Serial, "[MP2.3b] Output Inf: ");
                                let _ = writeln!(arch::serial::Serial, "{}", inf_count);

                                // In-place verification
                                let bytes_used_post = act_arena.used();
                                let bytes_used_alloc_only = bytes_used_post - bytes_used_pre;
                                let expected_alloc = head_dim * 4;
                                let _ = write!(
                                    arch::serial::Serial,
                                    "[MP2.3b] ACTIVATION_ARENA delta (qk_buf alloc only): "
                                );
                                let _ = writeln!(arch::serial::Serial, "{}", bytes_used_alloc_only);
                                let _ = write!(
                                    arch::serial::Serial,
                                    "[MP2.3b] Expected delta (head_dim * 4): "
                                );
                                let _ = writeln!(arch::serial::Serial, "{}", expected_alloc);

                                if bytes_used_alloc_only == expected_alloc {
                                    let _ = writeln!(
                                        arch::serial::Serial,
                                        "[MP2.3b] In-place verification: PASS (RoPE adds 0 bytes)"
                                    );
                                } else {
                                    let _ = writeln!(
                                        arch::serial::Serial,
                                        "[MP2.3b] In-place verification: FAIL"
                                    );
                                }

                                if nan_count == 0
                                    && inf_count == 0
                                    && identity_ok
                                    && bytes_used_alloc_only == expected_alloc
                                {
                                    let _ = writeln!(
                                        arch::serial::Serial,
                                        "[MP2.3b] RoPE heartbeat: PLAUSIBLE ok"
                                    );
                                } else {
                                    let _ = writeln!(
                                        arch::serial::Serial,
                                        "[MP2.3b] RoPE heartbeat: IMPLAUSIBLE"
                                    );
                                }
                            } else {
                                let _ = writeln!(
                                    arch::serial::Serial,
                                    "[MP2.3b] ACTIVATION_ARENA alloc failed"
                                );
                            }
                        } else {
                            let _ = writeln!(
                                arch::serial::Serial,
                                "[MP2.3b] ACTIVATION_ARENA not initialized"
                            );
                        }
                    }

                    // Step 8: Linear Heartbeat (MP2.3c — Pipeline-Chain dequant → RMSNorm → Linear)
                    // Per V3 Pillar 1: LinearScratch built ONCE here, reused per call.
                    // Per V3 Pillar 2: pipeline chain demonstrates real forward-pass kette.
                    // Per V3 Arena-Disziplin: ACTIVATION_ARENA append (no reset between Steps 6/7/8).
                    // Per ADR-029 D5: caller-allocated out + scratch.
                    let _ = writeln!(
                        arch::serial::Serial,
                        "\r\n[MP2.3c] Linear heartbeat starting..."
                    );

                    let mut linear_scratch = zero_llm_inference::LinearScratch::new();
                    let _ = writeln!(
                        arch::serial::Serial,
                        "[MP2.3c] LinearScratch built (block_buf 256 f32 = 1024 bytes)"
                    );

                    let hidden_size = tensor_index.model_config.embedding_length as usize;
                    let attn_q_tensor = tensor_index.get("blk.0.attn_q.weight");

                    if let Some(q_t) = attn_q_tensor {
                        if q_t.tensor_type == zero_gguf_parser::GgmlType::Q4K {
                            let q_offset = tensor_index.tensor_data_offset + (q_t.offset as usize);
                            let blocks_per_row = hidden_size / 256;
                            let bytes_per_row = blocks_per_row * 144;
                            let total_w_bytes = hidden_size * bytes_per_row;

                            if q_offset + total_w_bytes <= model_bytes.len() {
                                let w_blocks: &[u8] = unsafe {
                                    core::slice::from_raw_parts(
                                        model_bytes.as_ptr().add(q_offset),
                                        total_w_bytes,
                                    )
                                };

                                let mut act_guard = memory::ACTIVATION_ARENA.lock();
                                if let Some(act_arena) = act_guard.as_mut() {
                                    let bytes_used_pre_step8 = act_arena.used();
                                    let _ = write!(
                                        arch::serial::Serial,
                                        "[MP2.3c] ACTIVATION_ARENA bytes_used at Step 8 start: "
                                    );
                                    let _ =
                                        writeln!(arch::serial::Serial, "{}", bytes_used_pre_step8);

                                    let x_result = act_arena.alloc_f32_slice(hidden_size);
                                    let out_result = act_arena.alloc_f32_slice(hidden_size);

                                    if let (Ok(x_buf), Ok(out_buf)) = (x_result, out_result) {
                                        // Build pipeline input: dequant first row of attn_q → RMSNorm
                                        let attn_norm_tensor =
                                            tensor_index.get("blk.0.attn_norm.weight");
                                        if let Some(norm_t) = attn_norm_tensor {
                                            let norm_offset = tensor_index.tensor_data_offset
                                                + (norm_t.offset as usize);
                                            let weight_slice: &[f32] = unsafe {
                                                core::slice::from_raw_parts(
                                                    model_bytes.as_ptr().add(norm_offset)
                                                        as *const f32,
                                                    hidden_size,
                                                )
                                            };

                                            if let Ok(rms_input_buf) =
                                                act_arena.alloc_f32_slice(hidden_size)
                                            {
                                                // Dequant first 8 Q4_K blocks → rms_input
                                                let q_8blocks: &[u8] = unsafe {
                                                    core::slice::from_raw_parts(
                                                        model_bytes.as_ptr().add(q_offset),
                                                        8 * 144,
                                                    )
                                                };
                                                zero_gguf_parser::dequant::dequant_q4k_row(
                                                    q_8blocks,
                                                    rms_input_buf,
                                                    8,
                                                );

                                                let epsilon = tensor_index
                                                    .model_config
                                                    .layer_norm_rms_epsilon;
                                                zero_llm_inference::rmsnorm(
                                                    rms_input_buf,
                                                    weight_slice,
                                                    x_buf,
                                                    epsilon,
                                                );

                                                let _ = writeln!(arch::serial::Serial, "[MP2.3c] Reusing rmsnorm_out as x (pipeline chain)");

                                                // Linear: out = x_buf @ Wᵀ
                                                zero_llm_inference::linear_q4k(
                                                    x_buf,
                                                    w_blocks,
                                                    out_buf,
                                                    &mut linear_scratch,
                                                    hidden_size,
                                                    hidden_size,
                                                );

                                                // Print first 8 q_projection values
                                                let _ = writeln!(
                                                arch::serial::Serial,
                                                "[MP2.3c] First 8 OUTPUT values (q_projection):"
                                            );
                                                let mut idx = 0u32;
                                                while idx < 8 {
                                                    let _ = write!(arch::serial::Serial, "  [");
                                                    let _ = write!(arch::serial::Serial, "{}", idx);
                                                    let _ = write!(arch::serial::Serial, "]: ");
                                                    print_f32(out_buf[idx as usize]);
                                                    let _ = writeln!(arch::serial::Serial, "");
                                                    idx += 1;
                                                }

                                                // Output range stats
                                                let mut min_v: f32 = out_buf[0];
                                                let mut max_v: f32 = out_buf[0];
                                                let mut nan_count = 0u32;
                                                let mut inf_count = 0u32;
                                                for &v in out_buf.iter() {
                                                    if v.is_nan() {
                                                        nan_count += 1;
                                                        continue;
                                                    }
                                                    if v.is_infinite() {
                                                        inf_count += 1;
                                                        continue;
                                                    }
                                                    if v < min_v {
                                                        min_v = v;
                                                    }
                                                    if v > max_v {
                                                        max_v = v;
                                                    }
                                                }
                                                let _ = write!(
                                                    arch::serial::Serial,
                                                    "[MP2.3c] Output range: min="
                                                );
                                                print_f32(min_v);
                                                let _ = write!(arch::serial::Serial, ", max=");
                                                print_f32(max_v);
                                                let _ = writeln!(arch::serial::Serial, "");
                                                let _ = write!(
                                                    arch::serial::Serial,
                                                    "[MP2.3c] Output NaN: "
                                                );
                                                let _ =
                                                    writeln!(arch::serial::Serial, "{}", nan_count);
                                                let _ = write!(
                                                    arch::serial::Serial,
                                                    "[MP2.3c] Output Inf: "
                                                );
                                                let _ =
                                                    writeln!(arch::serial::Serial, "{}", inf_count);

                                                // Arena growth tracking
                                                let bytes_used_post_step8 = act_arena.used();
                                                let step8_growth =
                                                    bytes_used_post_step8 - bytes_used_pre_step8;
                                                let _ = write!(arch::serial::Serial, "[MP2.3c] ACTIVATION_ARENA growth (RMSNorm \u{2192} Linear): ");
                                                let _ = writeln!(
                                                    arch::serial::Serial,
                                                    "{}",
                                                    step8_growth
                                                );

                                                if nan_count == 0 && inf_count == 0 {
                                                    let _ = writeln!(
                                                        arch::serial::Serial,
                                                        "[MP2.3c] Linear heartbeat: PLAUSIBLE ok"
                                                    );
                                                } else {
                                                    let _ = writeln!(arch::serial::Serial, "[MP2.3c] Linear heartbeat: IMPLAUSIBLE (NaN or Inf)");
                                                }
                                            } else {
                                                let _ = writeln!(
                                                arch::serial::Serial,
                                                "[MP2.3c] ACTIVATION_ARENA alloc rms_input failed"
                                            );
                                            }
                                        } else {
                                            let _ = writeln!(
                                                arch::serial::Serial,
                                                "[MP2.3c] attn_norm.weight NOT FOUND"
                                            );
                                        }
                                    } else {
                                        let _ = writeln!(
                                            arch::serial::Serial,
                                            "[MP2.3c] ACTIVATION_ARENA alloc failed"
                                        );
                                    }
                                } else {
                                    let _ = writeln!(
                                        arch::serial::Serial,
                                        "[MP2.3c] ACTIVATION_ARENA not initialized"
                                    );
                                }
                            } else {
                                let _ = writeln!(
                                    arch::serial::Serial,
                                    "[MP2.3c] attn_q tensor offset out of bounds"
                                );
                            }
                        } else if matches!(
                            tensor_index
                                .get("blk.0.attn_q.weight")
                                .map(|t| t.tensor_type),
                            Some(zero_gguf_parser::GgmlType::Q4_0)
                        ) {
                            // Native .smodel: the heartbeat's Q4_K LinearScratch
                            // chain does not apply; MP3 runs the native Q4_0
                            // linear path end-to-end.
                            let _ = writeln!(
                                arch::serial::Serial,
                                "[MP2.3c] skipped on native Q4_0 profile (Q4_K-specific heartbeat; MP3 covers the native path) ok"
                            );
                        } else {
                            let _ = writeln!(
                                arch::serial::Serial,
                                "[MP2.3c] attn_q.weight is NOT Q4_K"
                            );
                        }
                    } else {
                        let _ = writeln!(arch::serial::Serial, "[MP2.3c] attn_q.weight NOT FOUND");
                    }

                    // Step 9: GQA Attention Heartbeat (MP2.4)
                    // Heartbeat placement: main.rs (needs GGUF tensor access)
                    // Per V3 Pillar 2: 2-token attention forward-pass with non-degenerate softmax.
                    // Per ADR-029 D8/D9/D10: QK-Norm, Q6_K V-projection, reference-validated.
                    let _ = writeln!(
                        arch::serial::Serial,
                        "\r\n[MP2.4] GQA Attention heartbeat starting..."
                    );

                    let n_q_heads: usize = 16;
                    let n_kv_heads: usize = 8;
                    let head_dim: usize = 128;
                    let q_dim = n_q_heads * head_dim; // 2048
                    let kv_dim = n_kv_heads * head_dim; // 1024
                    let embedding_dim = tensor_index.model_config.embedding_length as usize;
                    let rms_eps: f32 = 1e-6;
                    let max_tokens: usize = 16;

                    // Load all required tensors
                    let has_tensors = tensor_index.get("blk.0.attn_norm.weight").is_some()
                        && tensor_index.get("blk.0.attn_q.weight").is_some()
                        && tensor_index.get("blk.0.attn_k.weight").is_some()
                        && tensor_index.get("blk.0.attn_v.weight").is_some()
                        && tensor_index.get("blk.0.attn_output.weight").is_some()
                        && tensor_index.get("blk.0.attn_q_norm.weight").is_some()
                        && tensor_index.get("blk.0.attn_k_norm.weight").is_some();

                    if has_tensors {
                        let norm_t = tensor_index.get("blk.0.attn_norm.weight").unwrap();
                        let q_t = tensor_index.get("blk.0.attn_q.weight").unwrap();
                        let k_t = tensor_index.get("blk.0.attn_k.weight").unwrap();
                        let v_t = tensor_index.get("blk.0.attn_v.weight").unwrap();
                        let o_t = tensor_index.get("blk.0.attn_output.weight").unwrap();
                        let qn_t = tensor_index.get("blk.0.attn_q_norm.weight").unwrap();
                        let kn_t = tensor_index.get("blk.0.attn_k_norm.weight").unwrap();

                        // Get weight byte slices
                        let norm_off = tensor_index.tensor_data_offset + norm_t.offset as usize;
                        let norm_w = unsafe {
                            core::slice::from_raw_parts(
                                model_bytes.as_ptr().add(norm_off) as *const f32,
                                embedding_dim,
                            )
                        };

                        // Resolve actual quant types from the tensor directory so
                        // the dispatch variant (gqa_attention_single_token_dispatch)
                        // picks the correct dequantizer for each projection.
                        // For Q4_0/Q4_K both use 144 bytes per 256 elements;
                        // Q6_K = 210; Q8_0 = 272.
                        let q_quant = q_t.tensor_type;
                        let k_quant = k_t.tensor_type;
                        let v_quant = v_t.tensor_type;
                        let o_quant = o_t.tensor_type;

                        fn bytes_per_256(t: zero_gguf_parser::GgmlType) -> usize {
                            match t {
                                zero_gguf_parser::GgmlType::Q6K => 210,
                                zero_gguf_parser::GgmlType::Q8_0 => 272,
                                _ => 144, // Q4_0, Q4_K, Q4_0X4 — all 144/256
                            }
                        }

                        let q_off = tensor_index.tensor_data_offset + q_t.offset as usize;
                        let q_bytes_len = (q_t.element_count() as usize / 256) * bytes_per_256(q_quant);
                        let q_w = &model_bytes[q_off..q_off + q_bytes_len];

                        let k_off = tensor_index.tensor_data_offset + k_t.offset as usize;
                        let k_bytes_len = (k_t.element_count() as usize / 256) * bytes_per_256(k_quant);
                        let k_w = &model_bytes[k_off..k_off + k_bytes_len];

                        let v_off = tensor_index.tensor_data_offset + v_t.offset as usize;
                        let v_bytes_len = (v_t.element_count() as usize / 256) * bytes_per_256(v_quant);
                        let v_w = &model_bytes[v_off..v_off + v_bytes_len];

                        let o_off = tensor_index.tensor_data_offset + o_t.offset as usize;
                        let o_bytes_len = (o_t.element_count() as usize / 256) * bytes_per_256(o_quant);
                        let o_w = &model_bytes[o_off..o_off + o_bytes_len];

                        let qn_off = tensor_index.tensor_data_offset + qn_t.offset as usize;
                        let qn_w = unsafe {
                            core::slice::from_raw_parts(
                                model_bytes.as_ptr().add(qn_off) as *const f32,
                                head_dim,
                            )
                        };

                        let kn_off = tensor_index.tensor_data_offset + kn_t.offset as usize;
                        let kn_w = unsafe {
                            core::slice::from_raw_parts(
                                model_bytes.as_ptr().add(kn_off) as *const f32,
                                head_dim,
                            )
                        };

                        let _ = writeln!(
                            arch::serial::Serial,
                            "[MP2.4] All 7 attention tensors loaded (q={:?} k={:?} v={:?} o={:?})",
                            q_quant, k_quant, v_quant, o_quant
                        );

                        // Allocate scratch via ACTIVATION_ARENA
                        // Total: input(2048) + normed(2048) + q_buf(2048) + k_buf(1024) + v_buf(1024)
                        //      + q_out(2048) + k_out(1024) + score_buf(16) + attn_head_buf(128)
                        //      + attn_out(2048) + output(2048) + kv_storage(2*1*16*1024)
                        //      = ~48K f32 = ~192 KB
                        let arena_ok = {
                            let mut act_guard = memory::ACTIVATION_ARENA.lock();
                            if let Some(ref mut arena) = *act_guard {
                                let total_f32 = embedding_dim
                                    + embedding_dim
                                    + q_dim
                                    + kv_dim
                                    + kv_dim
                                    + q_dim
                                    + kv_dim
                                    + max_tokens
                                    + head_dim
                                    + q_dim
                                    + embedding_dim
                                    + (2 * 1 * max_tokens * kv_dim);
                                if let Ok(buf) = arena.alloc_f32_slice(total_f32) {
                                    // split_at_mut chain for non-overlapping &mut slices
                                    let (input_buf, rest) = buf.split_at_mut(embedding_dim);
                                    let (normed_buf, rest) = rest.split_at_mut(embedding_dim);
                                    let (q_buf, rest) = rest.split_at_mut(q_dim);
                                    let (k_buf, rest) = rest.split_at_mut(kv_dim);
                                    let (v_buf, rest) = rest.split_at_mut(kv_dim);
                                    let (q_out, rest) = rest.split_at_mut(q_dim);
                                    let (k_out, rest) = rest.split_at_mut(kv_dim);
                                    let (score_buf, rest) = rest.split_at_mut(max_tokens);
                                    let (attn_head_buf, rest) = rest.split_at_mut(head_dim);
                                    let (attn_out, rest) = rest.split_at_mut(q_dim);
                                    let (output_buf, kv_storage) = rest.split_at_mut(embedding_dim);

                                    let mut kv_cache = zero_llm_inference::KvCache::new(
                                        kv_storage.as_mut_ptr(),
                                        max_tokens,
                                        1,
                                        n_kv_heads,
                                        head_dim,
                                    );

                                    let rope_ctx =
                                        zero_llm_inference::RopeContext::<64>::new(1_000_000.0);
                                    let mut lin_scratch =
                                        zero_llm_inference::LinearScratch::new();

                                    // Token 0: ramp * 2 at position 0
                                    for i in 0..embedding_dim {
                                        input_buf[i] = 0.02 * (i as f32 + 1.0);
                                    }
                                    zero_llm_inference::rmsnorm(
                                        input_buf, norm_w, normed_buf, rms_eps,
                                    );

                                    let r0 = zero_llm_inference::gqa_attention_single_token_dispatch::<64>(
                                        normed_buf,
                                        q_w,
                                        k_w,
                                        v_w,
                                        o_w,
                                        q_quant,
                                        k_quant,
                                        v_quant,
                                        o_quant,
                                        qn_w,
                                        kn_w,
                                        0,
                                        0,
                                        &rope_ctx,
                                        rms_eps,
                                        n_q_heads,
                                        n_kv_heads,
                                        head_dim,
                                        embedding_dim,
                                        &mut kv_cache,
                                        q_buf,
                                        k_buf,
                                        v_buf,
                                        q_out,
                                        k_out,
                                        score_buf,
                                        attn_head_buf,
                                        attn_out,
                                        &mut lin_scratch,
                                        output_buf,
                                    );

                                    if r0.is_ok() {
                                        let _ = writeln!(
                                            arch::serial::Serial,
                                            "[MP2.4] Token 0 (pos=0): ok"
                                        );
                                    } else {
                                        let _ = writeln!(
                                            arch::serial::Serial,
                                            "[MP2.4] Token 0 FAILED"
                                        );
                                    }

                                    // Token 1: ramp * 1 at position 1
                                    for i in 0..embedding_dim {
                                        input_buf[i] = 0.01 * (i as f32 + 1.0);
                                    }
                                    zero_llm_inference::rmsnorm(
                                        input_buf, norm_w, normed_buf, rms_eps,
                                    );

                                    let r1 = zero_llm_inference::gqa_attention_single_token_dispatch::<64>(
                                        normed_buf,
                                        q_w,
                                        k_w,
                                        v_w,
                                        o_w,
                                        q_quant,
                                        k_quant,
                                        v_quant,
                                        o_quant,
                                        qn_w,
                                        kn_w,
                                        0,
                                        1,
                                        &rope_ctx,
                                        rms_eps,
                                        n_q_heads,
                                        n_kv_heads,
                                        head_dim,
                                        embedding_dim,
                                        &mut kv_cache,
                                        q_buf,
                                        k_buf,
                                        v_buf,
                                        q_out,
                                        k_out,
                                        score_buf,
                                        attn_head_buf,
                                        attn_out,
                                        &mut lin_scratch,
                                        output_buf,
                                    );

                                    if r1.is_ok() {
                                        // Report softmax weights (non-degenerate check)
                                        // score_buf holds Q-head 15's post-softmax weights after
                                        // gqa_attention_single_token completes (last head processed).
                                        // Reference: Q-head 15 weights [0.501, 0.499] → w0=500, w1=499
                                        let _ = writeln!(
                                            arch::serial::Serial,
                                            "[MP2.4] Token 1 (pos=1): ok"
                                        );
                                        let w0_milli = (score_buf[0] * 1000.0) as i32;
                                        let w1_milli = (score_buf[1] * 1000.0) as i32;
                                        let _ = write!(
                                            arch::serial::Serial,
                                            "[MP2.4] Softmax (x1000): w0="
                                        );
                                        let _ = write!(arch::serial::Serial, "{}", w0_milli);
                                        let _ = write!(arch::serial::Serial, " w1=");
                                        let _ = writeln!(arch::serial::Serial, "{}", w1_milli);

                                        // Check output sanity
                                        let mut nan_count = 0u32;
                                        let mut inf_count = 0u32;
                                        let mut min_v: f32 = f32::MAX;
                                        let mut max_v: f32 = f32::MIN;
                                        for i in 0..embedding_dim {
                                            let v = output_buf[i];
                                            if v.is_nan() {
                                                nan_count += 1;
                                            }
                                            if v.is_infinite() {
                                                inf_count += 1;
                                            }
                                            if v < min_v {
                                                min_v = v;
                                            }
                                            if v > max_v {
                                                max_v = v;
                                            }
                                        }

                                        // Truncate to 2 decimal places via integer math
                                        let min_i = (min_v * 100.0) as i32;
                                        let max_i = (max_v * 100.0) as i32;
                                        let _ =
                                            write!(arch::serial::Serial, "[MP2.4] Output range: [");
                                        let _ = write!(arch::serial::Serial, "{}", min_i);
                                        let _ = write!(arch::serial::Serial, ", ");
                                        let _ = write!(arch::serial::Serial, "{}", max_i);
                                        let _ = writeln!(arch::serial::Serial, "] * 0.01");
                                        let _ = write!(arch::serial::Serial, "[MP2.4] NaN=");
                                        let _ = write!(arch::serial::Serial, "{}", nan_count);
                                        let _ = write!(arch::serial::Serial, " Inf=");
                                        let _ = writeln!(arch::serial::Serial, "{}", inf_count);

                                        // Evidence-based verdict:
                                        // Non-degenerate = both weights in [100, 900] (0.1 to 0.9)
                                        // This proves softmax produced a genuine mixture, not
                                        // [1.0, 0.0] (broken) or [0.5, 0.5] (degenerate/identity).
                                        // Reference Q-head 15: w0=500, w1=499 (tolerance ±50)
                                        let softmax_ok = w0_milli >= 100
                                            && w0_milli <= 900
                                            && w1_milli >= 100
                                            && w1_milli <= 900;
                                        let reference_match = w0_milli >= 450
                                            && w0_milli <= 550
                                            && w1_milli >= 449
                                            && w1_milli <= 549;

                                        if nan_count == 0
                                            && inf_count == 0
                                            && r0.is_ok()
                                            && softmax_ok
                                            && reference_match
                                        {
                                            let _ = writeln!(arch::serial::Serial, "[MP2.4] GQA Attention heartbeat: PLAUSIBLE (matches Sub-MP-C1 reference)");
                                        } else if nan_count == 0 && inf_count == 0 && softmax_ok {
                                            let _ = writeln!(arch::serial::Serial, "[MP2.4] GQA Attention heartbeat: SUSPICIOUS (non-degenerate but deviates from reference)");
                                        } else {
                                            let _ = writeln!(
                                                arch::serial::Serial,
                                                "[MP2.4] GQA Attention heartbeat: IMPLAUSIBLE"
                                            );
                                        }
                                    } else {
                                        let _ = writeln!(
                                            arch::serial::Serial,
                                            "[MP2.4] Token 1 FAILED"
                                        );
                                    }

                                    true
                                } else {
                                    let _ = writeln!(
                                        arch::serial::Serial,
                                        "[MP2.4] ACTIVATION_ARENA alloc failed"
                                    );
                                    false
                                }
                            } else {
                                let _ = writeln!(
                                    arch::serial::Serial,
                                    "[MP2.4] ACTIVATION_ARENA not initialized"
                                );
                                false
                            }
                        };
                        let _ = arena_ok; // suppress unused warning
                    } else {
                        let _ = writeln!(arch::serial::Serial, "[MP2.4] Missing attention tensors");
                    }
                }

                // ── Step 10: Full Forward-Pass Heartbeat (MP2.5) ──────────────
                // Single-token prefill: "Hello" (token 9707) through 28 layers
                // + LM head → predicted next-token-ID. Sub-MP-C3 ground truth: 25 (= ':')
                // Per Pillar 7: shared cross-platform function (inference.rs)
                let _ = writeln!(arch::serial::Serial, "\r\n[DBG] before MP2.5 forward");
                let _ = writeln!(
                    arch::serial::Serial,
                    "[MP2.5] Forward-pass heartbeat starting..."
                );
                #[cfg(target_arch = "x86_64")]
                control_plane::mark_running();
                inference::run_forward_pass(model_bytes, &tensor_index);
                #[cfg(target_arch = "x86_64")]
                control_plane::mark_completed();

                // Store LLM_ARENA globally. Three paths:
                //   * ramdisk      — bootloader mapping; wrap the slice directly.
                //   * nvme         — phys-linear view of the freshly-loaded
                //                    weight arena; same wrapping pattern as
                //                    ramdisk (the slice is `'static` once
                //                    `make_resident` returns).
                //   * direct-memory— keep the legacy phys_addr ctor for
                //                    diagnostic continuity in QEMU dev runs.
                let arena_result = if source_label == "direct-memory" {
                    unsafe { llm_arena::LlmArena::new(MODEL_PHYS_ADDR, phys_offset, MODEL_SIZE) }
                } else {
                    unsafe { llm_arena::LlmArena::from_static_slice(model_bytes) }
                };
                match arena_result {
                    Ok(arena) => {
                        *llm_arena::LLM_ARENA.lock() = Some(arena);
                        let _ = writeln!(
                            arch::serial::Serial,
                            "Stage 11 MP1: LLM_ARENA initialized via {}, {} MB model region locked",
                            source_label,
                            model_size / (1024 * 1024)
                        );
                        let _ = writeln!(
                            arch::serial::Serial,
                            "Stage 11 MP1: V3 Phase 4 Component 2 (quantized model loading) verified for Qwen3 1.7B"
                        );
                    }
                    Err(e) => {
                        let _ = writeln!(
                            arch::serial::Serial,
                            "Stage 11 MP1: LlmArena init FAILED: {:?}",
                            e
                        );
                    }
                }
            }
            Err(e) => {
                let _ = writeln!(
                    arch::serial::Serial,
                    "[MP2.1] Selective parse FAILED: {:?}",
                    e
                );
                #[cfg(target_arch = "x86_64")]
                control_plane::mark_unavailable();
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        let status = control_plane::status();
        if status == control_plane::STATUS_BOOTING
            || status == control_plane::STATUS_REQUESTED
            || status == control_plane::STATUS_LOADING
        {
            control_plane::mark_unavailable();
        }
    }
    {
        let guard = memory::RUNTIME_ARENA.lock();
        if let Some(arena) = guard.as_ref() {
            let _ = writeln!(
                arch::serial::Serial,
                "Stage 3: runtime arena used {} bytes after IR validation",
                arena.used()
            );
        }
    }

    // Framebuffer is owned by `fb_console` (initialised at kernel_main
    // entry); every `Serial::write_str` already renders to it. The old
    // Stage-3 demo painter that re-cleared the screen and drew a static
    // welcome string lived here — removed in favour of the live log.

    // === ROI Benchmark Suite ===
    // Run all benchmarks BEFORE the executor starts.
    // Results are printed to serial (COM1) for KVM console / SOL capture.
    // Always runs regardless of framebuffer availability.
    bench::run_all_benchmarks(&cpu_info);

    // Benchmark-results marker. Serial output is auto-mirrored to the
    // GOP framebuffer console (see arch::x86_64::serial), so a serial
    // println surfaces on both COM1 and the on-screen console.
    arch::serial::println("=== Benchmark Results: see serial (COM1) ===");

    // Stage 3 sub-part 2 — async executor.
    // Interrupts are currently enabled (from the PIT test above).
    // The executor's run() loop manages cli/sti itself.
    arch::interrupts_disable();

    // Async executor.
    let executor = task::Executor::new();

    // Stage 3 regression tasks. Local/dev and cherry-smp-debug builds
    // keep these visible liveness tests. Cherry production keeps the
    // benchmark screen stable instead of overwriting it with demo ticks.
    #[cfg(any(not(feature = "cherry-net"), feature = "cherry-smp-debug"))]
    {
        let channel: &'static task::oneshot::Oneshot<u64> =
            memory::arena_static_alloc(task::oneshot::Oneshot::new())
                .expect("oneshot channel arena alloc failed");

        executor.spawn(task_a()).expect("spawn A failed");
        executor.spawn(task_b(channel)).expect("spawn B failed");
        executor.spawn(task_c(channel)).expect("spawn C failed");
    }
    #[cfg(all(feature = "cherry-net", not(feature = "cherry-smp-debug")))]
    {
        arch::serial::println(
            "Stage 3: Cherry production mode -- benchmark screen held; demo ticks disabled",
        );
    }

    // Stage 10.8 — retrieve the net stack from the timer-ISR hook
    // (registered immediately after bring-up) and hand ownership to
    // the cooperative executor on local/dev builds.
    //
    // Cherry bare-metal is different: the TCP/UDP rescue shell is the
    // only remote lifeline while SMP bring-up is under investigation.
    // Keep the ISR poll hook active for the full kernel lifetime so the
    // shell does not disappear after the benchmark -> executor hand-off.
    #[cfg(feature = "cherry-net")]
    {
        arch::serial::println("net: Cherry rescue shell remains on timer ISR polling");
        executor
            .spawn(cherry_net_rescue_poll_task())
            .expect("spawn Cherry net rescue poll failed");
    }
    #[cfg(not(feature = "cherry-net"))]
    {
        // Interrupts are already disabled here (executor.run() re-enables
        // them), so the swap is race-free.
        if let Some(stack) = net::take_irq_poll() {
            executor
                .spawn(net_poll_task(stack))
                .expect("spawn net poll failed");
        }
    }

    // run() diverges — enables interrupts internally via sti;hlt.
    executor.run();
}

/// Zero control-plane gate: keep the remote console alive
/// before the synchronous Stage-11 Boot-LLM path takes over the BSP.
#[cfg(all(target_arch = "x86_64", feature = "zero-control-plane"))]
fn wait_for_boot_llm_start_gate() {
    control_plane::arm_boot_llm_gate();
    arch::interrupts_enable();
    arch::serial::println(
        "Stage 11 control plane: remote console online; send `llm start` to load Boot-LLM",
    );
    arch::serial::println(
        "Stage 11 control plane: TCP 2222 and UDP 9999 available for Zero diagnostics",
    );

    let mut last_notice_tick = arch::pit::ticks();
    while !control_plane::boot_llm_start_requested() {
        arch::without_interrupts(|| {
            net::irq_poll_tick();
        });

        let now = arch::pit::ticks();
        if now.wrapping_sub(last_notice_tick) >= arch::pit::HZ * 10 {
            last_notice_tick = now;
            arch::serial::println("Stage 11 control plane: waiting for `llm start`");
        }
        core::hint::spin_loop();
    }

    control_plane::reset_llm_profile();
    #[cfg(feature = "avx512-acceleration")]
    crate::inference_avx512::reset_perf_counters();
    control_plane::mark_loading();
    arch::serial::println("Stage 11 control plane: `llm start` received; continuing to Boot-LLM");
}

#[cfg(target_arch = "x86_64")]
// ── Stage 3 test tasks ──────────────────────────────────────────────────────────

/// Task A: prints "tick t=Ns" once per second.  Never terminates.
///
/// Uses `TimerFuture` (busy-wake) which exercises `wake_by_ref`
/// on every poll iteration until the target tick is reached.
#[cfg(all(
    target_arch = "x86_64",
    any(not(feature = "cherry-net"), feature = "cherry-smp-debug")
))]
async fn task_a() {
    let start = arch::pit::ticks();
    for sec in 1u64.. {
        task::futures::TimerFuture::new(start + sec * arch::pit::HZ).await;
        #[cfg(all(
            feature = "cherry-net",
            feature = "avx512-acceleration",
            feature = "cherry-smp-debug"
        ))]
        {
            let raw_stage = arch::x86_64::trampoline::status().raw_probe_stage;
            let probe_stage = smp::ap_probe_stage() as u8;
            let _ = writeln!(
                arch::serial::Serial,
                "SMP t={}s act={} reg={} gate={} req={} mode={} raw={} probe={}",
                sec,
                smp::active_cores(),
                smp::registered_cores(),
                bool01(smp::ap_boot_gate_active()),
                bool01(smp::ap_boot_requested()),
                ap_boot_mode_label(smp::ap_boot_mode()),
                raw_probe_stage_label(raw_stage),
                raw_probe_stage_label(probe_stage)
            );
        }
        #[cfg(not(all(
            feature = "cherry-net",
            feature = "avx512-acceleration",
            feature = "cherry-smp-debug"
        )))]
        {
            let _ = writeln!(arch::serial::Serial, "Stage 3 test: tick t={}s", sec);
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    feature = "cherry-net",
    feature = "avx512-acceleration",
    feature = "cherry-smp-debug"
))]
fn bool01(value: bool) -> &'static str {
    if value {
        "1"
    } else {
        "0"
    }
}

#[cfg(all(
    target_arch = "x86_64",
    feature = "cherry-net",
    feature = "avx512-acceleration",
    feature = "cherry-smp-debug"
))]
fn ap_boot_mode_label(mode: u32) -> &'static str {
    match mode {
        smp::AP_BOOT_MODE_FULL => "full",
        smp::AP_BOOT_MODE_PROBE_ENTRY => "probe-entry",
        smp::AP_BOOT_MODE_PROBE_IDT => "probe-idt",
        smp::AP_BOOT_MODE_PROBE_APIC => "probe-apic",
        smp::AP_BOOT_MODE_PROBE_SIMD => "probe-simd",
        smp::AP_BOOT_MODE_PROBE_TRAMP_REAL => "probe-tramp-real",
        smp::AP_BOOT_MODE_PROBE_TRAMP_PROT => "probe-tramp-prot",
        smp::AP_BOOT_MODE_PROBE_TRAMP_PAE => "probe-tramp-pae",
        smp::AP_BOOT_MODE_PROBE_TRAMP_EFER => "probe-tramp-efer",
        smp::AP_BOOT_MODE_PROBE_TRAMP_PAGING => "probe-tramp-paging",
        smp::AP_BOOT_MODE_PROBE_TRAMP_LONG => "probe-tramp-long",
        smp::AP_BOOT_MODE_PROBE_TRAMP_CR3 => "probe-tramp-cr3",
        smp::AP_BOOT_MODE_PROBE_TRAMP_RUST => "probe-tramp-rust",
        _ => "unknown",
    }
}

#[cfg(all(
    target_arch = "x86_64",
    feature = "cherry-net",
    feature = "avx512-acceleration",
    feature = "cherry-smp-debug"
))]
fn raw_probe_stage_label(stage: u8) -> &'static str {
    match stage as u32 {
        smp::AP_PROBE_STAGE_IDLE => "idle",
        smp::AP_PROBE_STAGE_TRAMP_REAL => "tramp-real",
        smp::AP_PROBE_STAGE_TRAMP_PROT => "tramp-prot",
        smp::AP_PROBE_STAGE_TRAMP_PAE => "tramp-pae",
        smp::AP_PROBE_STAGE_TRAMP_EFER => "tramp-efer",
        smp::AP_PROBE_STAGE_TRAMP_PAGING => "tramp-paging",
        smp::AP_PROBE_STAGE_TRAMP_LONG => "tramp-long",
        smp::AP_PROBE_STAGE_TRAMP_CR3 => "tramp-cr3",
        smp::AP_PROBE_STAGE_TRAMP_RUST => "tramp-rust",
        smp::AP_PROBE_STAGE_ENTRY => "entry",
        smp::AP_PROBE_STAGE_IDT => "idt",
        smp::AP_PROBE_STAGE_APIC => "apic",
        smp::AP_PROBE_STAGE_SIMD => "simd",
        _ => "unknown",
    }
}

/// Task B: counts 1..=10 with cooperative yields between each
/// iteration, then sends the final count to Task C via oneshot.
///
/// `YieldFuture` exercises `wake_by_ref` (one yield per iteration).
/// The `channel.send()` exercises `Waker::wake()` (VTable `wake`
/// function) on Task C's stored waker.
#[cfg(all(
    target_arch = "x86_64",
    any(not(feature = "cherry-net"), feature = "cherry-smp-debug")
))]
async fn task_b(channel: &'static task::oneshot::Oneshot<u64>) {
    for i in 1..=10u64 {
        let _ = writeln!(arch::serial::Serial, "Stage 3 test: counting {}", i);
        if i < 10 {
            task::futures::YieldFuture::new().await;
        }
    }
    channel.send(10);
}

/// Task C: waits on the oneshot channel, prints the received value,
/// then terminates.
///
/// `RecvFuture::poll` calls `cx.waker().clone()` (VTable `clone`)
/// when parking.  When Task B sends, `Waker::wake()` (VTable
/// `wake`) is called on the cloned waker.  The original waker
/// from `poll_task` is dropped (VTable `drop`).  All four VTable
/// functions are exercised across the Task B + C interaction.
#[cfg(all(
    target_arch = "x86_64",
    any(not(feature = "cherry-net"), feature = "cherry-smp-debug")
))]
async fn task_c(channel: &'static task::oneshot::Oneshot<u64>) {
    let val = channel.recv().await;
    let _ = writeln!(arch::serial::Serial, "Stage 3 test: received {}", val);
}

/// Stage 10.8 net poll task. Wakes on every PIT tick (10 ms) and drains
/// pending RX frames through the IP stack. Diverges — kernels don't
/// exit, the executor halts (`sti; hlt`) between ticks.
#[cfg(all(target_arch = "x86_64", not(feature = "cherry-net")))]
async fn net_poll_task(stack: &'static mut net::Stack) {
    let mut target = arch::pit::ticks().wrapping_add(1);
    loop {
        task::futures::TimerFuture::new(target).await;
        target = target.wrapping_add(1);
        stack.poll();
    }
}

/// Cherry bare-metal rescue poller. The network stack remains
/// registered with the timer-ISR hook, but EPYC/UEFI bring-up can
/// leave the PIT/8259 path unreliable after INIT-SIPI-SIPI. Poll the
/// same hook from the cooperative executor too, with interrupts masked
/// during the poll so the timer ISR cannot race the raw stack pointer.
#[cfg(all(target_arch = "x86_64", feature = "cherry-net"))]
async fn cherry_net_rescue_poll_task() {
    loop {
        task::futures::TimerFuture::new(arch::pit::ticks().wrapping_add(1)).await;
        arch::x86_64::without_interrupts(|| {
            net::irq_poll_tick();
        });
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    let _ = writeln!(arch::serial::Serial, "\r\nKERNEL PANIC: {}", info);
    halt_forever();
}

fn halt_forever() -> ! {
    loop {
        arch::hlt();
    }
}

// ---- aarch64 entry point ----

/// aarch64 boot entry — called from assembly boot stub.
/// Sub-MP-D2b: DTB pointer passed via x0 (AAPCS64 first arg).
///
/// QEMU -kernel with flat binary (objcopy -O binary) guarantees
/// x0 = DTB physical address. Per Sub-MP-D1 boot-strategy-decision.md.
///
/// Boot order: Stage 0 (banner) → Stage 2 (VBAR_EL1) → Stage 3 (DTB parse).
/// Stage 2 before Stage 3 ensures exceptions are diagnosable during DTB parse.
#[cfg(target_arch = "aarch64")]
#[no_mangle]
pub extern "C" fn kernel_main_aarch64(dtb_ptr: usize) -> ! {
    // Stage 0: early UART init (hardcoded PL011 at 0x0900_0000)
    arch::aarch64::serial::init_early();
    arch::serial::println("");
    arch::serial::println("Zero v0.0.1 (aarch64)");
    arch::serial::println("");
    arch::serial::println("Stage 0: kernel online (aarch64 direct boot).");

    // Stage 2: Install exception vectors BEFORE any non-trivial code.
    // Per ARM ARM D1.10.2: VBAR_EL1 must be set early so that any
    // exception (alignment fault, data abort, undefined instruction)
    // produces a diagnostic instead of a silent infinite loop.
    unsafe { arch::aarch64::exceptions::init() };
    arch::serial::println("Stage 2: VBAR_EL1 installed, exceptions diagnosable.");

    // Stage 3: DTB parse — x0 guaranteed to be DTB pointer
    let _ = write!(
        arch::serial::Serial,
        "Stage 3: DTB at {:#x} (from x0)",
        dtb_ptr
    );
    arch::serial::println("");

    if dtb_ptr == 0 {
        panic!("DTB pointer is null — ensure kernel is loaded as flat binary (objcopy -O binary)");
    }

    let dtb_info = unsafe { arch::aarch64::dtb::parse(dtb_ptr) };

    let _ = write!(
        arch::serial::Serial,
        "Stage 3: RAM {} MiB at {:#x}",
        dtb_info.ram_size / (1024 * 1024),
        dtb_info.ram_base
    );
    arch::serial::println("");

    let boot_llm_model_size =
        if let (Some(start), Some(end)) = (dtb_info.initrd_start, dtb_info.initrd_end) {
            end.saturating_sub(start)
        } else {
            0
        };

    if let (Some(start), Some(end)) = (dtb_info.initrd_start, dtb_info.initrd_end) {
        let size_mb = (end - start) / (1024 * 1024);
        let _ = write!(
            arch::serial::Serial,
            "Stage 3: initrd at {:#x}..{:#x} ({} MiB)",
            start,
            end,
            size_mb
        );
        arch::serial::println("");
    } else {
        arch::serial::println("Stage 3: no initrd (run with -initrd for GGUF)");
    }

    let _ = write!(
        arch::serial::Serial,
        "Stage 3: UART at {:#x}, GIC dist={:#x} cpu={:#x}",
        dtb_info.uart_base,
        dtb_info.gic_dist_base,
        dtb_info.gic_cpu_base
    );
    arch::serial::println("");

    arch::serial::println("Stage 3: DTB parse complete.");

    // Stage 4: Switch UART to DTB-discovered base.
    // On QEMU virt, this matches early hardcoded base (0x9000000).
    // On real hardware, this enables platform-independent boot.
    let early_base = arch::aarch64::serial::current_base();
    unsafe { arch::aarch64::serial::set_base(dtb_info.uart_base) };
    let _ = write!(
        arch::serial::Serial,
        "Stage 4: UART base {:#x} -> {:#x} (DTB-discovered)",
        early_base,
        dtb_info.uart_base
    );
    arch::serial::println("");
    arch::serial::println("Stage 4: Late Serial operational.");

    // Stage 5: Build translation tables, configure MMU registers.
    // Per ARM ARM D5.2: identity-map kernel + MMIO regions.
    // MMU stays OFF until Stage 6.
    unsafe {
        arch::aarch64::mmu::init_tables(
            dtb_info.uart_base,
            dtb_info.gic_dist_base,
            dtb_info.gic_cpu_base,
        );
    }

    // Stage 6: Enable MMU — transition to virtual addressing.
    // After this call, all accesses go through translation tables.
    // UART mapped as Device-nGnRE, kernel as Normal cacheable.
    // NOTE: Stage 5+6 are atomic — HVF requires MMU enable after
    // register configuration (crashes if MAIR/TCR/TTBR set but M=0).
    unsafe { arch::aarch64::mmu::enable_mmu() };

    // Stage 7: Map initrd to high-half via TTBR1_EL1 (Mode-B-Resolution).
    // Per ADR-028 v7: GGUF accessible at known kernel-virtual-address.
    // Physical 0x4800_0000 → virtual 0xFFFF_0000_4800_0000 (BOOT_LLM_VIRT_BASE).
    unsafe { arch::aarch64::mmu::map_initrd_high_half() };

    // Stage 7 verification: supported Boot-LLM container at virtual address.
    let model_ok = unsafe { arch::aarch64::mmu::verify_model_at_virt() };
    if model_ok {
        let _ = write!(
            arch::serial::Serial,
            "Stage 7: Mode-B-Resolution complete (model at {:#x})",
            arch::aarch64::mmu::BOOT_LLM_VIRT_BASE
        );
        arch::serial::println("");
    } else {
        arch::serial::println("Stage 7: model magic verification FAILED — halting.");
        halt_forever();
    }
    // Stage 8: Initialize GICv2 interrupt controller.
    // Per ARM GIC spec IHI 0048B-b: Distributor + CPU Interface.
    // DTB-discovered base addresses from Stage 3.
    unsafe {
        arch::aarch64::gic::init(dtb_info.gic_dist_base, dtb_info.gic_cpu_base);
    }
    arch::serial::println("Stage 8: GICv2 ready.");

    // Stage 9: Initialize Generic Timer (Virtual Timer, INTID 27).
    // Per ARM ARM D7.5: CNTV_TVAL_EL0 downcount at 100 Hz.
    unsafe { arch::aarch64::timer::init() };

    // Unmask IRQs at DAIF (bit 1 = IRQ mask, daifclr #0x2 clears it)
    unsafe {
        core::arch::asm!(
            "msr daifclr, #0x2",
            options(nomem, nostack, preserves_flags)
        )
    };
    arch::serial::println("Stage 9: DAIF.I unmasked, interrupts active.");

    // Stage 10: Initialize V3-Arena infrastructure with aarch64 backing.
    // Per ARCHITECTURE.md V3-Arena-Disziplin: Arena-based allocation.
    // Feeds KERNEL_ARENA, RUNTIME_ARENA, ACTIVATION_ARENA, KV_CACHE_ARENA
    // with identity-mapped physical RAM at 0x4400_0000..0x4800_0000.
    unsafe { crate::memory::init_aarch64() };

    // Stage 10 empirical validation: alloc test via ArenaGlobalAllocator.
    // Vec uses #[global_allocator] → RUNTIME_ARENA → aarch64 backing memory.
    {
        use alloc::vec::Vec;
        let mut test_vec: Vec<u32> = Vec::with_capacity(16);
        for i in 0..16u32 {
            test_vec.push(i * 3);
        }
        let test_sum: u32 = test_vec.iter().sum();
        let _ = write!(
            arch::serial::Serial,
            "Stage 10: Arena alloc test — Vec[16] sum={}",
            test_sum
        );
        arch::serial::println(" (expect 360)");
        if test_sum != 360 {
            arch::serial::println("Stage 10: ARENA TEST FAILED — halting.");
            halt_forever();
        }
    }
    arch::serial::println("Stage 10: Cross-platform arena infrastructure operational.");

    // Stage 12 — sandbox manager (excluded from public release, WIP).
    // See docs/discovery/stage-12-sandbox-baseline.md.

    // === Stage 10.5: Sub-MP-F1 — LFB Foundation (Pixel Awakening) ===
    // Per ADR-032: ramfb via fw-cfg DMA, XRGB8888 at phys 0x9500_0000.
    // LFB init is best-effort: if ramfb unavailable, inference proceeds.
    arch::serial::println("");
    arch::serial::println("Stage 10.5: LFB Foundation (Sub-MP-F1)...");
    unsafe {
        if arch::aarch64::lfb::init_ramfb() {
            // Render "Zero — The First AI Dreams"
            lfb::primitives::clear_screen(0, 0, 0);

            // Title — green on black (Matrix style)
            lfb::font::draw_string(
                20,
                20,
                "Zero - The First AI Dreams",
                0,
                255,
                128, // green
                0,
                0,
                0,
            );
            // Subtitle — cyan (updated per F3)
            lfb::font::draw_string(
                20,
                44,
                "Sub-MP-F3: Telemetry + Typewriter + Progress",
                0,
                200,
                200,
                0,
                0,
                0,
            );
            // Status — light green (N7 honest-metrics anchor: computed speedup)
            lfb::font::draw_string(
                20,
                68,
                "1.57x NEON | 0.4267 tok/s | Bare Metal",
                100,
                255,
                100,
                0,
                0,
                0,
            );

            // Sub-MP-F2: Typewriter panel (left side, narrower to fit telemetry)
            lfb::typewriter::init(20, 120, 680, 540);

            // Sub-MP-F2: Layer progress bar (bottom of screen)
            lfb::layer_progress::init(20, 700, 980, 16);

            // Sub-MP-G2 prototype: aarch64 KV cache enlarged to ADR-030
            // 512 MiB size after GGUF+ramfb, still Normal-cacheable.
            lfb::telemetry_data::set_memory_boundaries(lfb::telemetry_data::MemoryBoundaries {
                kv_cache_arena_size_mb: 512,
                framebuffer_size_kb: 3072, // 1024×768×4bpp
            });

            // Sub-MP-F3: Telemetry panel (right side, static dashboard)
            lfb::telemetry::init(720, 120, 280, 400);

            // Select telemetry data based on build mode
            let td = if cfg!(feature = "avx512-acceleration") {
                lfb::telemetry_data::TelemetryData::avx512_mode()
            } else if cfg!(feature = "neon-acceleration") {
                lfb::telemetry_data::TelemetryData::neon_mode()
            } else {
                lfb::telemetry_data::TelemetryData::scalar_mode()
            };

            // Render ONCE before inference (zero per-token overhead, Lesson 36)
            lfb::telemetry::render_static(&td);

            arch::serial::println(
                "Stage 10.5: LFB + typewriter + layer-progress + telemetry initialized.",
            );
        } else {
            arch::serial::println(
                "Stage 10.5: No ramfb device — LFB skipped (inference unaffected).",
            );
        }
    }

    // === Stage 11: Boot-LLM Forward-Pass (ARM-Portierung Closure) ===
    // Per ADR-028 v7 Mode-B-Resolution: model container at BOOT_LLM_VIRT_BASE.
    // Native `.smodel` is the primary Zero Server format; raw GGUF is kept
    // only as the strict legacy Qwen anchor.
    // Per ADR-029 v2: MP2 forward-pass with Sub-MP-C4 ratified operators
    // Per Lesson 10: NEON context save enforced in Phase A
    // Per Lesson 11: opt-level=1 kernel + opt-level=2 inference crates
    arch::serial::println("");
    arch::serial::println("Stage 11: Boot-LLM forward-pass starting...");

    'boot_llm: {
        use crate::model_loader::{
            gguf_payload_view, model_magic_from_bytes, smodel_info, smodel_native_tensor_index,
            smodel_payload_kind_label, ModelMagic, SMODEL_PAYLOAD_KIND_NATIVE,
        };

        if boot_llm_model_size == 0 {
            arch::serial::println("Stage 11: no initrd model size from DTB — skipping Boot-LLM");
            break 'boot_llm;
        }

        let model_virt = arch::aarch64::mmu::BOOT_LLM_VIRT_BASE;
        let mut model_bytes: &'static [u8] =
            unsafe { core::slice::from_raw_parts(model_virt as *const u8, boot_llm_model_size) };
        let _ = writeln!(
            arch::serial::Serial,
            "Stage 11: model container size from DTB = {} bytes",
            boot_llm_model_size
        );

        let parsed_tensor_index = match model_magic_from_bytes(model_bytes) {
            ModelMagic::Gguf => {
                arch::serial::println("Stage 11: raw GGUF marker verified at BOOT_LLM_VIRT_BASE");
                zero_gguf_parser::parse_selective(model_bytes)
            }
            ModelMagic::Smodel => match smodel_info(model_bytes) {
                Ok(info) if info.payload_kind == SMODEL_PAYLOAD_KIND_NATIVE => {
                    let _ = writeln!(
                        arch::serial::Serial,
                        "Stage 11: native .smodel marker verified payload={} bytes kind={}",
                        info.payload_len,
                        smodel_payload_kind_label(info.payload_kind)
                    );
                    match smodel_native_tensor_index(model_bytes) {
                        Ok(index) => Ok(index),
                        Err(e) => {
                            let _ = writeln!(
                                arch::serial::Serial,
                                "[MP2.1] Native .smodel tensor index FAILED: {:?}",
                                e
                            );
                            break 'boot_llm;
                        }
                    }
                }
                Ok(info) => {
                    let _ = writeln!(
                        arch::serial::Serial,
                        "Stage 11: .smodel compatibility payload kind={}",
                        smodel_payload_kind_label(info.payload_kind)
                    );
                    match gguf_payload_view(model_bytes) {
                        Ok(view) => {
                            model_bytes = view.bytes;
                            zero_gguf_parser::parse_selective(model_bytes)
                        }
                        Err(e) => {
                            let _ = writeln!(
                                arch::serial::Serial,
                                "[MP2.1] .smodel GGUF-compat unwrap FAILED: {:?}",
                                e
                            );
                            break 'boot_llm;
                        }
                    }
                }
                Err(e) => {
                    let _ = writeln!(
                        arch::serial::Serial,
                        "Stage 11: .smodel header FAILED: {:?}",
                        e
                    );
                    break 'boot_llm;
                }
            },
            ModelMagic::Unknown(magic) => {
                let _ = writeln!(
                    arch::serial::Serial,
                    "Stage 11: unsupported model magic at BOOT_LLM_VIRT_BASE: 0x{:08x}",
                    magic
                );
                break 'boot_llm;
            }
        };

        // Parse native .smodel tensor directory or legacy GGUF metadata.
        let _ = writeln!(arch::serial::Serial, "[MP2.1] Selective parser starting...");
        match parsed_tensor_index {
            Ok(tensor_index) => {
                let _ = write!(arch::serial::Serial, "[MP2.1] Tensor count: ");
                let _ = writeln!(arch::serial::Serial, "{}", tensor_index.len());

                let cfg = &tensor_index.model_config;
                let _ = write!(arch::serial::Serial, "[MP2.1] ModelConfig: blocks=");
                let _ = write!(arch::serial::Serial, "{}", cfg.block_count);
                let _ = write!(arch::serial::Serial, " hidden=");
                let _ = write!(arch::serial::Serial, "{}", cfg.embedding_length);
                let _ = write!(arch::serial::Serial, " heads=");
                let _ = write!(arch::serial::Serial, "{}", cfg.head_count);
                let _ = write!(arch::serial::Serial, " kv_heads=");
                let _ = writeln!(arch::serial::Serial, "{}", cfg.head_count_kv);

                // Replace the hardcoded telemetry-panel placeholder with
                // the producer's `general.name` so the on-screen label
                // tracks the actually-loaded weights. The first render
                // ran before parse_selective; re-render once so the new
                // label is visible. Both calls are unsafe-on-framebuffer
                // and only run inside the ramfb-online branch above.
                if let Some(name) = tensor_index.model_name.as_deref() {
                    unsafe {
                        lfb::telemetry_data::set_model_label(name);
                    }
                    let td = if cfg!(feature = "avx512-acceleration") {
                        lfb::telemetry_data::TelemetryData::avx512_mode()
                    } else if cfg!(feature = "neon-acceleration") {
                        lfb::telemetry_data::TelemetryData::neon_mode()
                    } else {
                        lfb::telemetry_data::TelemetryData::scalar_mode()
                    };
                    unsafe { lfb::telemetry::render_static(&td) };
                }

                // MP2.5: Forward-pass via shared cross-platform function (Pillar 7)
                #[cfg(not(feature = "streaming-mode"))]
                inference::run_forward_pass(model_bytes, &tensor_index);

                #[cfg(feature = "streaming-mode")]
                {
                    let mut dream_session: u64 = 1;
                    loop {
                        let _ = write!(arch::serial::Serial, "[G2] Eternal Dream session ");
                        let _ = write!(arch::serial::Serial, "{}", dream_session);
                        let _ = writeln!(arch::serial::Serial, " starting");
                        inference::run_forward_pass(model_bytes, &tensor_index);
                        dream_session = dream_session.wrapping_add(1);
                    }
                }
            }
            Err(e) => {
                let _ = writeln!(
                    arch::serial::Serial,
                    "[MP2.1] Selective parse FAILED: {:?}",
                    e
                );
            }
        }
    }

    arch::serial::println("");
    arch::serial::println("Stage 11 complete.");

    // Stage 12 — in-kernel sandbox smoke test on the real running
    // aarch64 boot path. Exercises spawn / parse / validate /
    // type-check / load / execute (incl. budgeted preemption surface)
    // and capability minting + revocation on the live
    // `SANDBOX_MANAGER`. No panics: failures surface as a structured
    // `SmokeTestError` and are reported to serial.
    //
    // Streaming-mode does not reach this point (the forward-pass loop
    // above is non-terminating). The smoke test only runs in the
    // non-streaming boot.
    // Stage 12 sandbox smoke test excluded from public release.

    arch::serial::println("Entering executor...");

    // Cooperative executor: WFI idle loop.
    // Timer IRQ fires at 100 Hz, handler increments tick + prints heartbeat.
    // ERET returns to WFI, loop continues.
    loop {
        unsafe { core::arch::asm!("wfi", options(nomem, nostack, preserves_flags)) };
    }
}
