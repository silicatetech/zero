// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rich command set for the TCP shell.
//!
//! Sits in front of [`super::shell::handle`]: the TCP shell tries this
//! dispatcher first, and only falls through to the UDP-shared command
//! surface for the legacy primitives (`ping`, `echo`, `mac`, `ip`).
//!
//! Why two surfaces? The UDP shell is one datagram in, one datagram
//! out — a 1024-byte response cap. Several of the commands here
//! produce multi-line output (PCI dump, status report) that would not
//! fit. Keeping the rich commands TCP-only lets each command write
//! up to ~3 KiB into a scratch buffer and flush in one `tcp.send`.
//!
//! Concurrency: handlers run inside `Stack::poll`, which is invoked
//! from both the cooperative executor's `net_poll_task` and the
//! timer-ISR (`irq_poll_tick`). Anything that could be held by a
//! long-running boot-time path — notably the activation arena during
//! a forward pass — is read via `try_lock` so the shell never wedges
//! waiting on inference.

use core::fmt::Write;

use super::eth::Mac;
use super::ipv4::Ipv4;
use super::tcp::Tcp;
use crate::memory;

/// Maximum bytes a single command may emit. Sized to leave headroom in
/// the TCP TX buffer (4096 bytes) for the greeting/prompt that frame
/// the response.
const MAX_RESPONSE: usize = 3072;

/// Outcome of a dispatched command.
pub enum CmdResult {
    /// Command was recognised and handled — bytes were pushed to `tcp`.
    Handled,
    /// Command was not recognised by this dispatcher. The caller should
    /// fall through to the legacy `shell::handle` surface.
    Unknown,
    /// Command is `exit` / `quit`. The caller should send a goodbye
    /// line and close the connection.
    Exit,
    /// Command requests a system reboot. The caller should attempt to
    /// flush any pending output and then trigger the reset.
    Reboot,
}

/// Top-level dispatcher. `line` has already been trimmed of leading /
/// trailing whitespace by the caller.
pub fn dispatch(tcp: &mut Tcp, line: &[u8], our_mac: Mac, our_ip: Ipv4) -> CmdResult {
    if line.is_empty() {
        return CmdResult::Unknown;
    }

    let (head, tail) = split_first_word(line);

    match head {
        b"help" => {
            help(tcp);
            CmdResult::Handled
        }
        b"status" => {
            status(tcp, our_mac, our_ip);
            CmdResult::Handled
        }
        b"diag" => {
            diag(tcp, our_mac, our_ip);
            CmdResult::Handled
        }
        b"pci" => {
            pci(tcp);
            CmdResult::Handled
        }
        b"net" => {
            net(tcp, our_mac, our_ip);
            CmdResult::Handled
        }
        b"tcp" => {
            tcp_diag(tcp);
            CmdResult::Handled
        }
        b"mem" => {
            mem(tcp);
            CmdResult::Handled
        }
        b"model" => {
            model(tcp);
            CmdResult::Handled
        }
        b"inference" | b"llm" => {
            inference(tcp, tail);
            CmdResult::Handled
        }
        b"bench" => {
            bench(tcp);
            CmdResult::Handled
        }
        b"smp" => {
            smp(tcp, tail);
            CmdResult::Handled
        }
        b"apic" => {
            apic(tcp);
            CmdResult::Handled
        }
        b"madt" => {
            madt(tcp);
            CmdResult::Handled
        }
        b"trampoline" => {
            trampoline_status(tcp);
            CmdResult::Handled
        }
        b"cores" => {
            cores(tcp);
            CmdResult::Handled
        }
        b"version" => {
            version(tcp);
            CmdResult::Handled
        }
        b"reboot" => CmdResult::Reboot,
        b"exit" | b"quit" => CmdResult::Exit,
        _ => CmdResult::Unknown,
    }
}

// ── Writer plumbing ─────────────────────────────────────────────────

/// Scratch buffer + `core::fmt::Write` adapter. Each command formats
/// into the buffer, then a single `tcp.send` flushes it.
struct Buf {
    data: [u8; MAX_RESPONSE],
    len: usize,
    truncated: bool,
}

impl Buf {
    const fn new() -> Self {
        Self {
            data: [0; MAX_RESPONSE],
            len: 0,
            truncated: false,
        }
    }

    fn flush(&self, tcp: &mut Tcp) {
        let _ = tcp.send(&self.data[..self.len]);
        if self.truncated {
            let _ = tcp.send(b"... (output truncated)\n");
        }
    }
}

impl core::fmt::Write for Buf {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let bytes = s.as_bytes();
        let room = self.data.len() - self.len;
        if bytes.len() > room {
            let n = room;
            self.data[self.len..self.len + n].copy_from_slice(&bytes[..n]);
            self.len += n;
            self.truncated = true;
        } else {
            self.data[self.len..self.len + bytes.len()].copy_from_slice(bytes);
            self.len += bytes.len();
        }
        Ok(())
    }
}

// ── help ────────────────────────────────────────────────────────────

fn help(tcp: &mut Tcp) {
    let mut b = Buf::new();
    let _ = writeln!(b, "Zero shell — available commands:");
    let _ = writeln!(b, "  help               this message");
    let _ = writeln!(b, "  status             cores, memory, NIC, uptime");
    let _ = writeln!(
        b,
        "  diag               compact boot/SMP/TCP diagnostic snapshot"
    );
    let _ = writeln!(b, "  pci                PCI device dump");
    let _ = writeln!(b, "  net                NIC link, MAC, IP, profile");
    let _ = writeln!(b, "  tcp                TCP shell state + recovery timers");
    let _ = writeln!(b, "  mem                memory arenas: used / capacity");
    let _ = writeln!(b, "  model              loaded model container + validation anchors");
    let _ = writeln!(
        b,
        "  llm status         Boot-LLM/control-plane state + cached tok/s"
    );
    let _ = writeln!(
        b,
        "  llm start          release the Zero control-plane LLM gate"
    );
    let _ = writeln!(
        b,
        "  llm profile        AVX-512 aggregate LLM kernel counters"
    );
    let _ = writeln!(b, "  inference ...      alias for llm ...");
    let _ = writeln!(
        b,
        "  bench              run a micro-benchmark, report ops/s + cached LLM tok/s"
    );
    let _ = writeln!(
        b,
        "  smp status|start [N|all] inspect or release the AP wake-up gate"
    );
    let _ = writeln!(
        b,
        "  smp tune [production|rows-per-core N|max-cores N|thread-policy all|unique-core|reset]"
    );
    let _ = writeln!(
        b,
        "  smp probe <real|prot|pae|efer|paging|long|cr3|rust|entry|idt|apic|simd>"
    );
    let _ = writeln!(
        b,
        "  apic               current LAPIC mode, id, version, ESR"
    );
    let _ = writeln!(b, "  madt               ACPI CPU topology recorded at boot");
    let _ = writeln!(b, "  trampoline         AP trampoline install status");
    let _ = writeln!(b, "  cores              per-core APIC IDs + active count");
    let _ = writeln!(b, "  version            kernel version + build features");
    let _ = writeln!(b, "  reboot             reset the system");
    let _ = writeln!(b, "  exit | quit        close this connection");
    let _ = writeln!(
        b,
        "Legacy (UDP-shell compatible): ping, mac, ip, echo <text>"
    );
    b.flush(tcp);
}

// ── status ──────────────────────────────────────────────────────────

fn status(tcp: &mut Tcp, our_mac: Mac, our_ip: Ipv4) {
    let mut b = Buf::new();
    let _ = writeln!(b, "Zero v{} — status", env!("CARGO_PKG_VERSION"));
    write_uptime_line(&mut b);
    write_cores_summary(&mut b);
    write_mem_summary(&mut b);
    write_net_summary(&mut b, our_mac, our_ip);
    write_inference_summary(&mut b);
    b.flush(tcp);
}

fn diag(tcp: &mut Tcp, our_mac: Mac, our_ip: Ipv4) {
    let mut b = Buf::new();
    let _ = writeln!(b, "Zero diagnostic snapshot:");
    let _ = writeln!(b, "version:     {}", env!("CARGO_PKG_VERSION"));
    write_uptime_line(&mut b);
    write_net_summary(&mut b, our_mac, our_ip);
    write_tcp_summary(&mut b, tcp);
    write_smp_diag(&mut b);
    write_madt_diag(&mut b);
    write_apic_diag(&mut b);
    write_trampoline_diag(&mut b);
    write_inference_summary(&mut b);
    b.flush(tcp);
}

fn write_uptime_line(b: &mut Buf) {
    let ticks = crate::arch::pit::ticks();
    let secs = ticks / crate::arch::pit::HZ;
    let hundredths = (ticks % crate::arch::pit::HZ) * 100 / crate::arch::pit::HZ;
    let _ = writeln!(
        b,
        "uptime:      {}.{:02}s ({} ticks @ {} Hz)",
        secs,
        hundredths,
        ticks,
        crate::arch::pit::HZ
    );
}

fn write_cores_summary(b: &mut Buf) {
    #[cfg(feature = "avx512-acceleration")]
    {
        let active = crate::smp::active_cores();
        let registered = crate::smp::registered_cores();
        let _ = writeln!(
            b,
            "cores:       active={} registered={} (cap={}, ap_gate={}, requested={})",
            active,
            registered,
            crate::smp::MAX_CORES,
            bool_label(crate::smp::ap_boot_gate_active()),
            bool_label(crate::smp::ap_boot_requested())
        );
    }
    #[cfg(not(feature = "avx512-acceleration"))]
    {
        let _ = writeln!(b, "cores:       1 (BSP-only build, SMP off)");
    }
}

fn write_mem_summary(b: &mut Buf) {
    if let Some(arena) = memory::KERNEL_ARENA.try_lock() {
        if let Some(ref a) = *arena {
            let _ = writeln!(b, "kernel arena:     {} / {} bytes", a.used(), a.capacity());
        }
    } else {
        let _ = writeln!(b, "kernel arena:     <busy>");
    }
    if let Some(arena) = memory::RUNTIME_ARENA.try_lock() {
        if let Some(ref a) = *arena {
            let _ = writeln!(b, "runtime arena:    {} / {} bytes", a.used(), a.capacity());
        }
    } else {
        let _ = writeln!(b, "runtime arena:    <busy>");
    }
    if let Some(arena) = memory::ACTIVATION_ARENA.try_lock() {
        if let Some(ref a) = *arena {
            let _ = writeln!(b, "activation arena: {} / {} bytes", a.used(), a.capacity());
        }
    } else {
        let _ = writeln!(b, "activation arena: <busy — inference in progress>");
    }
}

fn write_net_summary(b: &mut Buf, our_mac: Mac, our_ip: Ipv4) {
    let _ = write!(b, "nic mac:     ");
    write_mac(b, our_mac);
    let _ = writeln!(b);
    let _ = write!(b, "ip:          ");
    write_ip(b, our_ip);
    let _ = writeln!(b);
    let _ = write!(b, "gateway:     ");
    write_ip(b, super::GATEWAY_IP);
    let _ = writeln!(b);
    let _ = writeln!(b, "profile:     {}", super::PROFILE_LABEL);
}

fn write_inference_summary(b: &mut Buf) {
    let td = current_telemetry();
    let int = td.throughput_x10000 / 10000;
    let frac = td.throughput_x10000 % 10000;
    let cp = crate::control_plane::status();
    let _ = writeln!(
        b,
        "inference:   {}.{:04} tok/s ({}, last forward-pass wallclock {}s, control={})",
        int,
        frac,
        td.mode_label,
        td.current_baseline_wallclock_s,
        crate::control_plane::status_label(cp)
    );
}

fn write_tcp_summary(b: &mut Buf, tcp: &Tcp) {
    let now = current_tick();
    let (remote_ip, remote_port) = tcp.remote();
    let _ = writeln!(b, "tcp:         {:?}", tcp.state());
    let _ = write!(b, "tcp remote:  ");
    write_ip(b, remote_ip);
    let _ = writeln!(
        b,
        ":{} idle={} age={}",
        remote_port,
        tcp.idle_ticks(now),
        tcp.state_age_ticks(now)
    );
}

fn write_smp_diag(b: &mut Buf) {
    #[cfg(feature = "avx512-acceleration")]
    {
        let limit = crate::smp::ap_boot_ap_limit();
        if limit == u32::MAX {
            let _ = writeln!(
                b,
                "smp:         active={} registered={} gate={} requested={} mode={} probe={} limit=all",
                crate::smp::active_cores(),
                crate::smp::registered_cores(),
                if crate::smp::ap_boot_gate_active() { "armed" } else { "open" },
                bool_label(crate::smp::ap_boot_requested()),
                probe_mode_label(crate::smp::ap_boot_mode()),
                probe_stage_label(crate::smp::ap_probe_stage())
            );
        } else {
            let _ = writeln!(
                b,
                "smp:         active={} registered={} gate={} requested={} mode={} probe={} limit={}",
                crate::smp::active_cores(),
                crate::smp::registered_cores(),
                if crate::smp::ap_boot_gate_active() { "armed" } else { "open" },
                bool_label(crate::smp::ap_boot_requested()),
                probe_mode_label(crate::smp::ap_boot_mode()),
                probe_stage_label(crate::smp::ap_probe_stage()),
                limit
            );
        }
    }
    #[cfg(not(feature = "avx512-acceleration"))]
    {
        let _ = writeln!(b, "smp:         unavailable (BSP-only build)");
    }
}

fn write_apic_diag(b: &mut Buf) {
    #[cfg(target_arch = "x86_64")]
    {
        match crate::arch::x86_64::apic::Apic::current() {
            Some(apic) => {
                let _ = writeln!(
                    b,
                    "lapic:       mode={} id={} version=0x{:08x} esr=0x{:08x}",
                    apic.mode_label(),
                    apic.id(),
                    apic.version(),
                    apic.esr()
                );
            }
            None => {
                let _ = writeln!(b, "lapic:       not initialized");
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = writeln!(b, "lapic:       unavailable");
    }
}

fn write_madt_diag(b: &mut Buf) {
    #[cfg(target_arch = "x86_64")]
    {
        let (count, lapic, ioapic, override_applied) =
            crate::arch::x86_64::acpi::recorded_summary();
        let _ = writeln!(
            b,
            "madt:        cpus={} lapic=0x{:x} ioapic=0x{:x} override={}",
            count,
            lapic,
            ioapic,
            bool_label(override_applied)
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = writeln!(b, "madt:        unavailable");
    }
}

fn write_trampoline_diag(b: &mut Buf) {
    #[cfg(target_arch = "x86_64")]
    {
        let st = crate::arch::x86_64::trampoline::status();
        let _ = writeln!(
            b,
            "trampoline:  installed={} phys=0x{:x} vector=0x{:02x} boot_cr3=0x{:x} cr3=0x{:x} cr4=0x{:x} cr0=0x{:x} raw_mode={} raw_stage={}",
            bool_label(st.installed),
            st.phys,
            st.sipi_vector,
            st.bootstrap_cr3,
            st.real_cr3,
            st.real_cr4,
            st.real_cr0,
            raw_probe_label(st.raw_probe_mode),
            raw_probe_label(st.raw_probe_stage)
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = writeln!(b, "trampoline:  unavailable");
    }
}

// ── pci ─────────────────────────────────────────────────────────────

fn pci(tcp: &mut Tcp) {
    let scan = crate::arch::pcie::scan_all_buses();
    let mut b = Buf::new();
    let _ = writeln!(
        b,
        "PCI: {} function(s), highest bus 0x{:02x}",
        scan.count(),
        scan.max_bus_seen()
    );
    let _ = writeln!(b, "  bus:dev.fn  vendor  device  class  subclass  BAR0");
    for dev in scan.iter() {
        let bar0 = crate::arch::pcie::read_bar(dev, 0);
        let _ = write!(
            b,
            "  {:02x}:{:02x}.{}    0x{:04x}  0x{:04x}  0x{:02x}    0x{:02x}      ",
            dev.bus,
            dev.device,
            dev.function,
            dev.vendor_id,
            dev.device_id,
            dev.class_code,
            dev.subclass
        );
        match bar0 {
            Some(addr) => {
                let _ = writeln!(b, "0x{:x}", addr);
            }
            None => {
                let _ = writeln!(b, "(none)");
            }
        }
        if b.truncated {
            break;
        }
    }
    b.flush(tcp);
}

// ── net ─────────────────────────────────────────────────────────────

fn net(tcp: &mut Tcp, our_mac: Mac, our_ip: Ipv4) {
    let mut b = Buf::new();
    let _ = writeln!(b, "network state:");
    let _ = write!(b, "  mac:      ");
    write_mac(&mut b, our_mac);
    let _ = writeln!(b);
    let _ = write!(b, "  ip:       ");
    write_ip(&mut b, our_ip);
    let _ = writeln!(b);
    let _ = write!(b, "  netmask:  ");
    write_ip(&mut b, super::NETMASK);
    let _ = writeln!(b);
    let _ = write!(b, "  gateway:  ");
    write_ip(&mut b, super::GATEWAY_IP);
    let _ = writeln!(b);
    let _ = writeln!(b, "  profile:  {}", super::PROFILE_LABEL);
    let _ = writeln!(
        b,
        "  surfaces: ICMP echo, UDP/{} shell, TCP/{} shell",
        super::shell::SHELL_PORT,
        super::tcp_shell::PORT
    );
    // The NIC doesn't expose packet/error counters yet. State this
    // explicitly rather than hand back fake zeros.
    let _ = writeln!(
        b,
        "  counters: not yet wired (e1000/i40e expose MAC only at present)"
    );
    b.flush(tcp);
}

fn tcp_diag(tcp: &mut Tcp) {
    let now = current_tick();
    let (remote_ip, remote_port) = tcp.remote();
    let mut b = Buf::new();
    let _ = writeln!(b, "TCP shell:");
    let _ = writeln!(b, "  state:      {:?}", tcp.state());
    let _ = write!(b, "  remote:     ");
    write_ip(&mut b, remote_ip);
    let _ = writeln!(b, ":{}", remote_port);
    let _ = writeln!(b, "  idle:       {} ticks", tcp.idle_ticks(now));
    let _ = writeln!(b, "  state_age:  {} ticks", tcp.state_age_ticks(now));
    let _ = writeln!(
        b,
        "  recovery:   syn={} takeover={} foreign_takeover={} close={} timewait={} ticks",
        super::tcp::SYN_RECEIVED_TIMEOUT_TICKS,
        super::tcp::TCP_TAKEOVER_IDLE_TICKS,
        super::tcp::TCP_FOREIGN_TAKEOVER_IDLE_TICKS,
        super::tcp::TCP_CLOSE_TIMEOUT_TICKS,
        super::tcp::TIME_WAIT_TICKS
    );
    b.flush(tcp);
}

// ── mem ─────────────────────────────────────────────────────────────

fn mem(tcp: &mut Tcp) {
    let mut b = Buf::new();
    let _ = writeln!(b, "memory arenas:");
    write_mem_summary(&mut b);
    let _ = writeln!(b, "limits:");
    let _ = writeln!(
        b,
        "  KERNEL_ARENA_SIZE     = {} KiB",
        memory::KERNEL_ARENA_SIZE / 1024
    );
    let _ = writeln!(
        b,
        "  RUNTIME_ARENA_SIZE    = {} KiB",
        memory::RUNTIME_ARENA_SIZE / 1024
    );
    let _ = writeln!(
        b,
        "  ACTIVATION_ARENA_SIZE = {} KiB",
        memory::ACTIVATION_ARENA_SIZE / 1024
    );
    let _ = writeln!(
        b,
        "  KV_CACHE_ARENA_SIZE   = {} KiB",
        memory::KV_CACHE_ARENA_SIZE / 1024
    );
    #[cfg(target_arch = "x86_64")]
    {
        let _ = writeln!(b, "page mappings:");
        let model = crate::control_plane::model_region_snapshot();
        if model.present {
            let _ = writeln!(
                b,
                "  model: {} {:#x}..{:#x} ({} MiB)",
                crate::control_plane::model_source_label(model.source_id),
                model.virt_addr,
                model
                    .virt_addr
                    .saturating_add(model.len_bytes)
                    .saturating_sub(1),
                model.len_bytes / (1024 * 1024)
            );
            write_mapping_probe(&mut b, "    first", model.virt_addr);
            write_mapping_probe(
                &mut b,
                "    last ",
                model
                    .virt_addr
                    .saturating_add(model.len_bytes)
                    .saturating_sub(1),
            );
        } else {
            let _ = writeln!(b, "  model: not recorded yet");
        }
        write_mapping_probe(
            &mut b,
            "  activation",
            memory::ACTIVATION_ARENA_START.load(core::sync::atomic::Ordering::Acquire),
        );
        write_mapping_probe(
            &mut b,
            "  kv-cache  ",
            memory::KV_CACHE_ARENA_START.load(core::sync::atomic::Ordering::Acquire),
        );
    }
    b.flush(tcp);
}

fn model(tcp: &mut Tcp) {
    let mut b = Buf::new();
    write_model_summary(&mut b);
    b.flush(tcp);
}

fn write_model_summary(b: &mut Buf) {
    let model = crate::control_plane::model_region_snapshot();
    if !model.present {
        let _ = writeln!(b, "model: not recorded");
        return;
    }
    let _ = writeln!(
        b,
        "model: source={} va={:#x} len={} MiB",
        crate::control_plane::model_source_label(model.source_id),
        model.virt_addr,
        model.len_bytes / (1024 * 1024)
    );
    if model.virt_addr == 0 || model.len_bytes > usize::MAX as u64 {
        let _ = writeln!(b, "model: invalid recorded region");
        return;
    }
    let bytes = unsafe {
        core::slice::from_raw_parts(model.virt_addr as *const u8, model.len_bytes as usize)
    };
    match crate::model_loader::model_magic_from_bytes(bytes) {
        crate::model_loader::ModelMagic::Gguf => {
            let _ = writeln!(b, "model: format=raw-gguf");
        }
        crate::model_loader::ModelMagic::Smodel => match crate::model_loader::smodel_info(bytes) {
            Ok(info) => {
                let _ = writeln!(
                    b,
                    "model: format=.smodel kind={} header={} manifest={:#x}+{} payload={:#x}+{} flags={:#x}",
                    crate::model_loader::smodel_payload_kind_label(info.payload_kind),
                    info.header_len,
                    info.manifest_offset,
                    info.manifest_len,
                    info.payload_offset,
                    info.payload_len,
                    info.flags
                );
                let aligned = info.payload_offset % crate::model_loader::SMODEL_PAYLOAD_ALIGNMENT == 0;
                let _ = writeln!(b, "model: payload_2m_aligned={}", bool_label(aligned));
                if info.payload_kind == crate::model_loader::SMODEL_PAYLOAD_KIND_NATIVE {
                    match crate::model_loader::smodel_native_summary(bytes) {
                        Ok(native) => {
                            let _ = writeln!(
                                b,
                                "model: native tensors={} entry={} names={} data_base={:#x}",
                                native.tensor_count,
                                native.entry_size,
                                native.names_len,
                                native.data_base
                            );
                        }
                        Err(err) => {
                            let _ = writeln!(b, "model: native index invalid: {:?}", err);
                        }
                    }
                    match crate::model_loader::smodel_validation_anchor(bytes) {
                        Ok(Some(anchor)) => {
                            let _ = writeln!(
                                b,
                                "model: anchors={} next={:?} logit_bits={:?}",
                                if anchor.strict { "strict" } else { "capture" },
                                anchor.expected_next_token,
                                anchor.expected_logit_bits
                            );
                        }
                        Ok(None) => {
                            let _ = writeln!(b, "model: anchors=none");
                        }
                        Err(err) => {
                            let _ = writeln!(b, "model: anchors invalid: {:?}", err);
                        }
                    }
                }
            }
            Err(err) => {
                let _ = writeln!(b, "model: invalid .smodel: {:?}", err);
            }
        },
        crate::model_loader::ModelMagic::Unknown(magic) => {
            let _ = writeln!(b, "model: unknown magic={:#x}", magic);
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn write_mapping_probe(b: &mut Buf, label: &str, va: u64) {
    if va == 0 {
        let _ = writeln!(b, "{}: unmapped", label);
        return;
    }
    match memory::mapping_info(va) {
        Some(info) => {
            let cache = if info.pcd || info.pwt { "NON-WB" } else { "WB" };
            let _ = writeln!(
                b,
                "{}: va={:#x} pa={:#x} page={} KiB flags={:#x} cache={} pcd={} pwt={} huge={}",
                label,
                va,
                info.phys_addr,
                info.frame_size / 1024,
                info.flags_bits,
                cache,
                bool_label(info.pcd),
                bool_label(info.pwt),
                bool_label(info.huge)
            );
        }
        None => {
            let _ = writeln!(b, "{}: va={:#x} not mapped", label, va);
        }
    }
}

// ── inference ───────────────────────────────────────────────────────

fn inference(tcp: &mut Tcp, sub: &[u8]) {
    let sub = trim(sub);
    let (head, tail) = split_first_word(sub);
    let mut b = Buf::new();
    match head {
        b"" | b"status" => {
            let td = current_telemetry();
            let cp = crate::control_plane::status();
            let profile = crate::control_plane::llm_profile_snapshot();
            let _ = writeln!(b, "Zero Boot-LLM status:");
            let _ = writeln!(
                b,
                "  control:     {}",
                crate::control_plane::status_label(cp)
            );
            let _ = writeln!(
                b,
                "  gate:        {}",
                if crate::control_plane::boot_llm_gate_active() {
                    "armed"
                } else {
                    "open"
                }
            );
            let _ = writeln!(
                b,
                "  start req:   {}",
                bool_label(crate::control_plane::boot_llm_start_requested())
            );
            let _ = writeln!(
                b,
                "  model:        {}",
                crate::lfb::telemetry_data::effective_model_label(&td)
            );
            let model = crate::control_plane::model_region_snapshot();
            if model.present {
                let _ = writeln!(
                    b,
                    "  source:       {} ({} MiB)",
                    crate::control_plane::model_source_label(model.source_id),
                    model.len_bytes / (1024 * 1024)
                );
            } else {
                let _ = writeln!(b, "  source:       none");
            }
            let _ = writeln!(b, "  build mode:   {}", td.mode_label);
            if profile.valid
                && profile.generated_tokens != 0
                && profile.generation_compute_cycles != 0
            {
                let ns = crate::bench::cycles_to_ns(profile.generation_compute_cycles);
                if ns != 0 {
                    let measured_x10 = profile.generated_tokens.saturating_mul(10_000_000_000) / ns;
                    let _ = writeln!(
                        b,
                        "  throughput:   {}.{} tok/s measured compute",
                        measured_x10 / 10,
                        measured_x10 % 10
                    );
                } else {
                    let _ = writeln!(b, "  throughput:   unavailable (TSC conversion)");
                }
            } else {
                let int = td.throughput_x10000 / 10000;
                let frac = td.throughput_x10000 % 10000;
                let _ = writeln!(b, "  throughput:   {}.{:04} tok/s estimate", int, frac);
            }
            if profile.valid {
                let _ = writeln!(
                    b,
                    "  profile:      run {} / {} generated token(s)",
                    profile.run_id, profile.generated_tokens
                );
            } else {
                let _ = writeln!(b, "  profile:      pending");
            }
            let _ = writeln!(b, "  layers:       {}", td.total_layers);
            let _ = writeln!(
                b,
                "  e3 baseline:  {}s scalar wallclock",
                td.e3_baseline_wallclock_s
            );
            let _ = writeln!(
                b,
                "  current:      {}s wallclock per forward pass",
                td.current_baseline_wallclock_s
            );
            let spd = td.speedup_x100();
            let _ = writeln!(
                b,
                "  speedup:      {}.{:02}x vs scalar",
                spd / 100,
                spd % 100
            );
        }
        b"start" => {
            crate::control_plane::request_boot_llm_start();
            let _ = writeln!(b, "Zero control-plane: Boot-LLM start requested.");
            if crate::control_plane::boot_llm_gate_active() {
                let _ = writeln!(
                    b,
                    "  BSP will leave the Stage-11 gate after this response flushes."
                );
            } else {
                let _ = writeln!(
                    b,
                    "  No active Stage-11 gate; status is `{}`.",
                    crate::control_plane::status_label(crate::control_plane::status())
                );
            }
        }
        b"profile" => {
            llm_profile(&mut b, trim(tail));
        }
        b"model" => {
            write_model_summary(&mut b);
        }
        b"help" => {
            let _ = writeln!(b, "llm commands:");
            let _ = writeln!(
                b,
                "  llm status          state, cached throughput, control gate"
            );
            let _ = writeln!(
                b,
                "  llm start           release zero-control-plane Stage-11 gate"
            );
            let _ = writeln!(
                b,
                "  llm profile [reset] generation timing + AVX-512 counters"
            );
            let _ = writeln!(b, "  llm model           loaded container + anchors");
            let _ = writeln!(b, "  llm tune            show SMP row/core tuning");
        }
        b"tune" => {
            smp_tune(&mut b, trim(tail));
        }
        _ => {
            let _ = writeln!(
                b,
                "llm: unknown subcommand. Try `llm status`, `llm start`, `llm profile`, `llm model`, or `llm tune`."
            );
        }
    }
    b.flush(tcp);
}

fn llm_profile(b: &mut Buf, sub: &[u8]) {
    let (head, _) = split_first_word(trim(sub));
    if head == b"reset" {
        crate::control_plane::reset_llm_profile();
        #[cfg(all(target_arch = "x86_64", feature = "avx512-acceleration"))]
        crate::inference_avx512::reset_perf_counters();
        let _ = writeln!(b, "Zero LLM profile reset.");
        return;
    }

    let cp = crate::control_plane::llm_profile_snapshot();
    let _ = writeln!(b, "Zero LLM profile:");
    if !cp.valid {
        let _ = writeln!(b, "  state: no completed generation profile yet");
    } else {
        let _ = writeln!(b, "  run_id:        {}", cp.run_id);
        let _ = writeln!(b, "  prompt tokens: {}", cp.prompt_tokens);
        let _ = writeln!(b, "  generated:     {}", cp.generated_tokens);
        write_rate_line(
            b,
            "  decode wall",
            cp.generated_tokens,
            cp.generation_wall_cycles,
        );
        write_rate_line(
            b,
            "  decode compute",
            cp.generated_tokens,
            cp.generation_compute_cycles,
        );
        let _ = writeln!(b, "  prefill:");
        write_cycle_share(
            b,
            "    wall",
            cp.prefill_wall_cycles,
            cp.prefill_wall_cycles,
        );
        write_cycle_share(
            b,
            "    forward",
            cp.prefill_forward_cycles,
            cp.prefill_wall_cycles,
        );
        write_cycle_share(
            b,
            "    lm_head",
            cp.prefill_lm_head_cycles,
            cp.prefill_wall_cycles,
        );
        write_cycle_share(
            b,
            "    logit_scan",
            cp.prefill_logit_scan_cycles,
            cp.prefill_wall_cycles,
        );
        let _ = writeln!(b, "  generation:");
        write_cycle_share(
            b,
            "    wall",
            cp.generation_wall_cycles,
            cp.generation_wall_cycles,
        );
        write_cycle_share(
            b,
            "    compute",
            cp.generation_compute_cycles,
            cp.generation_wall_cycles,
        );
        write_cycle_share(
            b,
            "    forward",
            cp.generation_forward_cycles,
            cp.generation_compute_cycles,
        );
        write_cycle_share(
            b,
            "    lm_head",
            cp.generation_lm_head_cycles,
            cp.generation_compute_cycles,
        );
        write_cycle_share(
            b,
            "    sample",
            cp.generation_sample_cycles,
            cp.generation_wall_cycles,
        );
        write_cycle_share(
            b,
            "    logit_scan",
            cp.generation_logit_scan_cycles,
            cp.generation_wall_cycles,
        );
        write_cycle_share(
            b,
            "    render",
            cp.generation_render_cycles,
            cp.generation_wall_cycles,
        );
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512-acceleration"))]
    {
        let p = crate::inference_avx512::perf_counters_snapshot();
        let _ = writeln!(b, "AVX-512 LLM profile counters:");
        let _ = writeln!(
            b,
            "  q4k:     calls={} cycles={}",
            p.q4k_calls, p.q4k_cycles
        );
        let _ = writeln!(
            b,
            "  q6k:     calls={} cycles={}",
            p.q6k_calls, p.q6k_cycles
        );
        let _ = writeln!(
            b,
            "  lm_head: calls={} cycles={}",
            p.lm_head_calls, p.lm_head_cycles
        );
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx512-acceleration")))]
    {
        let _ = writeln!(
            b,
            "LLM profile unavailable: AVX-512 acceleration not enabled."
        );
    }
}

fn write_rate_line(b: &mut Buf, label: &str, tokens: u64, cycles: u64) {
    if tokens == 0 || cycles == 0 {
        let _ = writeln!(
            b,
            "{}: unavailable (tokens={}, cycles={})",
            label, tokens, cycles
        );
        return;
    }
    let ns = crate::bench::cycles_to_ns(cycles);
    if ns == 0 {
        let _ = writeln!(b, "{}: unavailable (cycles={})", label, cycles);
        return;
    }
    let tok_s_x10 = tokens.saturating_mul(10_000_000_000) / ns;
    let cycles_per_token = cycles / tokens.max(1);
    let _ = writeln!(
        b,
        "{}: {}.{} tok/s ({} cycles/token)",
        label,
        tok_s_x10 / 10,
        tok_s_x10 % 10,
        cycles_per_token
    );
}

fn write_cycle_share(b: &mut Buf, label: &str, cycles: u64, total: u64) {
    if total == 0 {
        let _ = writeln!(b, "{}: {} cycles", label, cycles);
        return;
    }
    let pct_x100 = cycles.saturating_mul(10_000) / total;
    let _ = writeln!(
        b,
        "{}: {} cycles ({}.{:02}%)",
        label,
        cycles,
        pct_x100 / 100,
        pct_x100 % 100
    );
}

// ── bench ──────────────────────────────────────────────────────────

fn bench(tcp: &mut Tcp) {
    // We run a short, self-contained bump-pointer micro-benchmark — the
    // same kind of measurement `bench::bench_arena_alloc` does, but
    // inlined here so the shell doesn't have to send to serial as a
    // side effect, and reduced in iteration count so it doesn't stall
    // the TCP poll loop noticeably (100 k iterations × ~1 ns ≈ 100 µs).
    const ITERS: u64 = 100_000;
    const ALLOC_SIZE: usize = 64;
    const ALLOC_ALIGN: usize = 8;
    const ARENA_SIZE: usize = 16 * 1024 * 1024;

    let mut cursor: usize = 0;
    for _ in 0..1_000u64 {
        let aligned = (cursor + ALLOC_ALIGN - 1) & !(ALLOC_ALIGN - 1);
        cursor = core::hint::black_box(aligned + ALLOC_SIZE);
        if cursor >= ARENA_SIZE {
            cursor = 0;
        }
    }

    let start = crate::arch::cycles::rdtsc_serialized();
    for _ in 0..ITERS {
        let aligned = (cursor + ALLOC_ALIGN - 1) & !(ALLOC_ALIGN - 1);
        cursor = core::hint::black_box(aligned + ALLOC_SIZE);
        if cursor >= ARENA_SIZE {
            cursor = 0;
        }
    }
    let end = crate::arch::cycles::rdtsc_serialized();
    let _ = core::hint::black_box(cursor);

    let total = end.wrapping_sub(start);
    let per = if ITERS > 0 { total / ITERS } else { 0 };
    let total_ns = crate::bench::cycles_to_ns(total);
    let per_ns_x10 = if total_ns > 0 {
        (total_ns * 10) / ITERS
    } else {
        0
    };
    let ops_per_sec = if total_ns > 0 {
        (ITERS * 1_000_000_000) / total_ns
    } else {
        0
    };

    let mut b = Buf::new();
    let _ = writeln!(b, "bench: bump-pointer arena alloc");
    let _ = writeln!(b, "  iterations:    {}", ITERS);
    let _ = writeln!(b, "  total cycles:  {}", total);
    let _ = writeln!(
        b,
        "  per-alloc:     {} cycles ({}.{} ns)",
        per,
        per_ns_x10 / 10,
        per_ns_x10 % 10
    );
    let _ = writeln!(b, "  throughput:    {} ops/s", ops_per_sec);
    let _ = writeln!(b);

    let td = current_telemetry();
    let int = td.throughput_x10000 / 10000;
    let frac = td.throughput_x10000 % 10000;
    let _ = writeln!(
        b,
        "bench: cached LLM forward-pass — {}.{:04} tok/s ({})",
        int, frac, td.mode_label
    );
    let _ = writeln!(b, "  Zero CPU-only target: >=150 tok/s");
    let _ = writeln!(
        b,
        "  (re-running the forward pass live from the shell is not wired — see `inference start`)"
    );
    b.flush(tcp);
}

// ── smp / AP bring-up diagnostics ─────────────────────────────────

fn smp(tcp: &mut Tcp, sub: &[u8]) {
    let sub = trim(sub);
    let mut b = Buf::new();
    #[cfg(feature = "avx512-acceleration")]
    {
        let (head, tail) = split_first_word(sub);
        match head {
            b"" | b"status" => {
                let _ = writeln!(b, "SMP status:");
                write_smp_summary(&mut b);
                let _ = writeln!(
                    b,
                    "  command: `smp start 1|all` only releases a cherry-smp-debug AP gate"
                );
            }
            b"tune" => {
                smp_tune(&mut b, trim(tail));
            }
            b"start" => {
                if !crate::smp::ap_boot_gate_active() {
                    let _ = writeln!(
                        b,
                        "SMP AP gate is not armed; AP bring-up has already passed."
                    );
                    let _ = writeln!(
                        b,
                        "Build with `cherry-smp-debug` to pause before INIT-SIPI-SIPI."
                    );
                    write_smp_summary(&mut b);
                    b.flush(tcp);
                    return;
                }
                let limit = trim(tail);
                let requested = if limit.is_empty() || limit == b"all" {
                    None
                } else {
                    parse_u32(limit)
                };
                match requested {
                    Some(n) => {
                        crate::smp::request_ap_boot_limited(n);
                        let _ =
                            writeln!(b, "SMP AP wake-up requested for up to {} AP(s).", n.max(1));
                    }
                    None if limit.is_empty() || limit == b"all" => {
                        crate::smp::request_ap_boot();
                        let _ = writeln!(b, "SMP AP wake-up requested for all eligible APs.");
                    }
                    None => {
                        let _ = writeln!(b, "smp start: expected decimal AP count or `all`.");
                        b.flush(tcp);
                        return;
                    }
                }
                let _ = writeln!(
                    b,
                    "BSP will start INIT-SIPI-SIPI after this command returns."
                );
                write_smp_summary(&mut b);
            }
            b"probe" => {
                if !crate::smp::ap_boot_gate_active() {
                    let _ = writeln!(b, "SMP probe unavailable: AP gate is not armed.");
                    let _ = writeln!(
                        b,
                        "Build with `cherry-smp-debug` to pause before INIT-SIPI-SIPI."
                    );
                    write_smp_summary(&mut b);
                    b.flush(tcp);
                    return;
                }
                let stage = trim(tail);
                match parse_probe_mode(stage) {
                    Some(mode) => {
                        crate::smp::request_ap_probe(mode);
                        let _ = writeln!(
                            b,
                            "SMP AP probe requested; one AP will park at `{}`.",
                            probe_mode_label(mode)
                        );
                        let _ = writeln!(
                            b,
                            "BSP will start INIT-SIPI-SIPI after this command returns."
                        );
                        write_smp_summary(&mut b);
                    }
                    None => {
                        let _ = writeln!(
                            b,
                            "smp probe: expected `real`, `prot`, `pae`, `efer`, `paging`, `long`, `cr3`, `rust`, `entry`, `idt`, `apic`, or `simd`."
                        );
                    }
                }
            }
            _ => {
                let _ = writeln!(
                    b,
                    "smp: unknown subcommand. Try `smp status`, `smp tune`, `smp start`, or `smp probe`."
                );
            }
        }
    }
    #[cfg(not(feature = "avx512-acceleration"))]
    {
        let _ = writeln!(
            b,
            "SMP unavailable: kernel built without avx512-acceleration."
        );
    }
    b.flush(tcp);
}

#[cfg(feature = "avx512-acceleration")]
fn smp_tune(b: &mut Buf, sub: &[u8]) {
    let sub = trim(sub);
    let (head, tail) = split_first_word(sub);
    match head {
        b"" | b"status" => {}
        b"reset" => {
            crate::smp::reset_matmul_tuning();
            let _ = crate::inference_avx512::reset_attn_parallel_min_tokens();
            let _ = writeln!(b, "SMP tune: reset to default matmul policy");
        }
        b"production" | b"prod" | b"cherry" => {
            crate::smp::apply_cherry_production_matmul_tuning();
            let _ = crate::inference_avx512::reset_attn_parallel_min_tokens();
            let _ = writeln!(b, "SMP tune: Cherry production policy restored");
        }
        b"rows-per-core" | b"rows" => {
            let value = trim(tail);
            if let Some(rows) = parse_u32(value) {
                let rows = crate::smp::set_min_matmul_rows_per_core(rows as usize);
                let _ = writeln!(b, "SMP tune: rows-per-core set to {}", rows);
            } else {
                let _ = writeln!(b, "smp tune rows-per-core: expected decimal value");
                return;
            }
        }
        b"max-cores" | b"max" => {
            let value = trim(tail);
            if let Some(cores) = parse_u32(value) {
                let cores = crate::smp::set_max_matmul_cores(cores as usize);
                let _ = writeln!(b, "SMP tune: max-cores set to {}", cores);
            } else {
                let _ = writeln!(b, "smp tune max-cores: expected decimal value");
                return;
            }
        }
        b"thread-policy" | b"policy" => {
            let value = trim(tail);
            let policy = match value {
                b"all" | b"logical" => crate::smp::MATMUL_THREAD_POLICY_ALL,
                b"unique-core" | b"physical" | b"physical-core" => {
                    crate::smp::MATMUL_THREAD_POLICY_UNIQUE_CORE
                }
                _ => {
                    let _ = writeln!(b, "smp tune thread-policy: expected all|unique-core");
                    return;
                }
            };
            let _ = crate::smp::set_matmul_thread_policy(policy);
            let _ = writeln!(
                b,
                "SMP tune: thread-policy set to {}",
                crate::smp::matmul_thread_policy_label()
            );
        }
        b"bsp-discount" | b"discount" => {
            let value = trim(tail);
            if let Some(pct) = parse_u32(value) {
                let pct = crate::smp::set_matmul_bsp_discount_pct(pct as usize);
                let _ = writeln!(b, "SMP tune: bsp-discount set to {} %", pct);
            } else {
                let _ = writeln!(b, "smp tune bsp-discount: expected percent value (0-90)");
                return;
            }
        }
        b"attn-par-tokens" | b"attn" => {
            let value = trim(tail);
            if let Some(tokens) = parse_u32(value) {
                let tokens =
                    crate::inference_avx512::set_attn_parallel_min_tokens(tokens as usize);
                let _ = writeln!(b, "SMP tune: attn-par-tokens set to {}", tokens);
            } else if value == b"off" {
                let tokens = crate::inference_avx512::set_attn_parallel_min_tokens(usize::MAX);
                let _ = writeln!(b, "SMP tune: attn-par-tokens set to off ({})", tokens);
            } else {
                let _ = writeln!(b, "smp tune attn-par-tokens: expected token count or `off`");
                return;
            }
        }
        _ => {
            let _ = writeln!(
                b,
                "smp tune: expected `status`, `production`, `rows-per-core N`, `max-cores N`, `thread-policy all|unique-core`, `bsp-discount PCT`, `attn-par-tokens N|off`, or `reset`"
            );
            return;
        }
    }

    let rows = crate::smp::min_matmul_rows_per_core();
    let _ = writeln!(b, "SMP matmul tuning:");
    let _ = writeln!(b, "  rows_per_core: {}", rows);
    let _ = writeln!(b, "  max_cores:     {}", crate::smp::max_matmul_cores());
    let _ = writeln!(
        b,
        "  bsp_discount:  {} %",
        crate::smp::matmul_bsp_discount_pct()
    );
    let _ = writeln!(
        b,
        "  attn_par_min_tokens: {}",
        crate::inference_avx512::attn_parallel_min_tokens()
    );
    let _ = writeln!(
        b,
        "  thread_policy: {}",
        crate::smp::matmul_thread_policy_label()
    );
    let _ = writeln!(b, "  active_cores:  {}", crate::smp::active_cores());
    let _ = writeln!(
        b,
        "  note: output rows are split only across rows; K-order stays single-core per row"
    );
    let _ = writeln!(b, "  rows     effective cores");
    for &dim in &[256usize, 512, 1024, 2048, 6144, 151_936] {
        let _ = writeln!(
            b,
            "  {:>6}   {:>3}",
            dim,
            crate::smp::effective_cores_for_rows(dim)
        );
    }
}

#[cfg(not(feature = "avx512-acceleration"))]
fn smp_tune(b: &mut Buf, _sub: &[u8]) {
    let _ = writeln!(
        b,
        "SMP tune unavailable: kernel built without avx512-acceleration."
    );
}

#[cfg(feature = "avx512-acceleration")]
fn write_smp_summary(b: &mut Buf) {
    let _ = writeln!(b, "  active:     {}", crate::smp::active_cores());
    let _ = writeln!(b, "  registered: {}", crate::smp::registered_cores());
    let _ = writeln!(b, "  cap:        {}", crate::smp::MAX_CORES);
    let _ = writeln!(
        b,
        "  ap_gate:    {}",
        if crate::smp::ap_boot_gate_active() {
            "armed"
        } else {
            "open"
        }
    );
    let _ = writeln!(
        b,
        "  requested:  {}",
        bool_label(crate::smp::ap_boot_requested())
    );
    let _ = writeln!(
        b,
        "  mode:       {}",
        probe_mode_label(crate::smp::ap_boot_mode())
    );
    let _ = writeln!(
        b,
        "  probe:      {}",
        probe_stage_label(crate::smp::ap_probe_stage())
    );
    #[cfg(target_arch = "x86_64")]
    {
        let st = crate::arch::x86_64::trampoline::status();
        let _ = writeln!(
            b,
            "  raw_probe:  mode={} stage={}",
            raw_probe_label(st.raw_probe_mode),
            raw_probe_label(st.raw_probe_stage)
        );
    }
    let limit = crate::smp::ap_boot_ap_limit();
    if limit == u32::MAX {
        let _ = writeln!(b, "  ap_limit:   all");
    } else {
        let _ = writeln!(b, "  ap_limit:   {}", limit);
    }
    let _ = writeln!(
        b,
        "  rows/core:  {}",
        crate::smp::min_matmul_rows_per_core()
    );
    let _ = writeln!(b, "  max cores:  {}", crate::smp::max_matmul_cores());
    let _ = writeln!(
        b,
        "  policy:     {}",
        crate::smp::matmul_thread_policy_label()
    );
}

#[cfg(feature = "avx512-acceleration")]
fn parse_probe_mode(s: &[u8]) -> Option<u32> {
    match s {
        b"real" | b"real16" => Some(crate::smp::AP_BOOT_MODE_PROBE_TRAMP_REAL),
        b"prot" | b"prot32" | b"protected" => Some(crate::smp::AP_BOOT_MODE_PROBE_TRAMP_PROT),
        b"pae" => Some(crate::smp::AP_BOOT_MODE_PROBE_TRAMP_PAE),
        b"efer" => Some(crate::smp::AP_BOOT_MODE_PROBE_TRAMP_EFER),
        b"paging" | b"pg" => Some(crate::smp::AP_BOOT_MODE_PROBE_TRAMP_PAGING),
        b"long" | b"long64" => Some(crate::smp::AP_BOOT_MODE_PROBE_TRAMP_LONG),
        b"cr3" => Some(crate::smp::AP_BOOT_MODE_PROBE_TRAMP_CR3),
        b"rust" | b"before-rust" => Some(crate::smp::AP_BOOT_MODE_PROBE_TRAMP_RUST),
        b"entry" => Some(crate::smp::AP_BOOT_MODE_PROBE_ENTRY),
        b"idt" => Some(crate::smp::AP_BOOT_MODE_PROBE_IDT),
        b"apic" => Some(crate::smp::AP_BOOT_MODE_PROBE_APIC),
        b"simd" => Some(crate::smp::AP_BOOT_MODE_PROBE_SIMD),
        _ => None,
    }
}

#[cfg(feature = "avx512-acceleration")]
fn probe_mode_label(mode: u32) -> &'static str {
    match mode {
        crate::smp::AP_BOOT_MODE_FULL => "full",
        crate::smp::AP_BOOT_MODE_PROBE_ENTRY => "probe-entry",
        crate::smp::AP_BOOT_MODE_PROBE_IDT => "probe-idt",
        crate::smp::AP_BOOT_MODE_PROBE_APIC => "probe-apic",
        crate::smp::AP_BOOT_MODE_PROBE_SIMD => "probe-simd",
        crate::smp::AP_BOOT_MODE_PROBE_TRAMP_REAL => "probe-tramp-real",
        crate::smp::AP_BOOT_MODE_PROBE_TRAMP_PROT => "probe-tramp-prot",
        crate::smp::AP_BOOT_MODE_PROBE_TRAMP_PAE => "probe-tramp-pae",
        crate::smp::AP_BOOT_MODE_PROBE_TRAMP_EFER => "probe-tramp-efer",
        crate::smp::AP_BOOT_MODE_PROBE_TRAMP_PAGING => "probe-tramp-paging",
        crate::smp::AP_BOOT_MODE_PROBE_TRAMP_LONG => "probe-tramp-long",
        crate::smp::AP_BOOT_MODE_PROBE_TRAMP_CR3 => "probe-tramp-cr3",
        crate::smp::AP_BOOT_MODE_PROBE_TRAMP_RUST => "probe-tramp-rust",
        _ => "unknown",
    }
}

#[cfg(feature = "avx512-acceleration")]
fn probe_stage_label(stage: u32) -> &'static str {
    match stage {
        crate::smp::AP_PROBE_STAGE_IDLE => "idle",
        crate::smp::AP_PROBE_STAGE_ENTRY => "entry",
        crate::smp::AP_PROBE_STAGE_IDT => "idt",
        crate::smp::AP_PROBE_STAGE_APIC => "apic",
        crate::smp::AP_PROBE_STAGE_SIMD => "simd",
        crate::smp::AP_PROBE_STAGE_TRAMP_REAL => "tramp-real",
        crate::smp::AP_PROBE_STAGE_TRAMP_PROT => "tramp-prot",
        crate::smp::AP_PROBE_STAGE_TRAMP_PAE => "tramp-pae",
        crate::smp::AP_PROBE_STAGE_TRAMP_EFER => "tramp-efer",
        crate::smp::AP_PROBE_STAGE_TRAMP_PAGING => "tramp-paging",
        crate::smp::AP_PROBE_STAGE_TRAMP_LONG => "tramp-long",
        crate::smp::AP_PROBE_STAGE_TRAMP_CR3 => "tramp-cr3",
        crate::smp::AP_PROBE_STAGE_TRAMP_RUST => "tramp-rust",
        _ => "unknown",
    }
}

fn apic(tcp: &mut Tcp) {
    let mut b = Buf::new();
    #[cfg(target_arch = "x86_64")]
    {
        match crate::arch::x86_64::apic::Apic::current() {
            Some(apic) => {
                let _ = writeln!(b, "LAPIC:");
                let _ = writeln!(b, "  mode:    {}", apic.mode_label());
                let _ = writeln!(b, "  id:      {}", apic.id());
                let _ = writeln!(b, "  version: 0x{:08x}", apic.version());
                let _ = writeln!(b, "  esr:     0x{:08x}", apic.esr());
            }
            None => {
                let _ = writeln!(b, "LAPIC: not initialized");
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = writeln!(b, "APIC: unavailable on this architecture");
    }
    b.flush(tcp);
}

fn madt(tcp: &mut Tcp) {
    let mut b = Buf::new();
    #[cfg(target_arch = "x86_64")]
    {
        let (count, lapic, ioapic, override_applied) =
            crate::arch::x86_64::acpi::recorded_summary();
        let _ = writeln!(b, "ACPI MADT:");
        if count == 0 {
            let _ = writeln!(b, "  not recorded or zero CPUs reported");
        } else {
            let _ = writeln!(b, "  cpus:           {}", count);
            let _ = writeln!(b, "  lapic phys:     0x{:x}", lapic);
            let _ = writeln!(b, "  ioapic phys:    0x{:x}", ioapic);
            let _ = writeln!(b, "  lapic override: {}", bool_label(override_applied));
            let _ = writeln!(b, "  idx   APIC ID   enabled");
            let limit = (count as usize).min(crate::arch::x86_64::acpi::MAX_CPUS);
            for idx in 0..limit {
                if let Some(cpu) = crate::arch::x86_64::acpi::recorded_cpu(idx) {
                    let _ = writeln!(
                        b,
                        "  {:>3}   {:>7}   {}",
                        idx,
                        cpu.apic_id,
                        bool_label(cpu.enabled)
                    );
                }
                if b.truncated {
                    break;
                }
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = writeln!(b, "MADT: unavailable on this architecture");
    }
    b.flush(tcp);
}

fn trampoline_status(tcp: &mut Tcp) {
    let mut b = Buf::new();
    #[cfg(target_arch = "x86_64")]
    {
        let st = crate::arch::x86_64::trampoline::status();
        let _ = writeln!(b, "AP trampoline:");
        let _ = writeln!(b, "  installed:     {}", bool_label(st.installed));
        let _ = writeln!(b, "  phys:          0x{:x}", st.phys);
        let _ = writeln!(b, "  sipi vector:   0x{:02x}", st.sipi_vector);
        let _ = writeln!(b, "  bootstrap cr3: 0x{:x}", st.bootstrap_cr3);
        let _ = writeln!(b, "  real cr3:      0x{:x}", st.real_cr3);
        let _ = writeln!(b, "  real cr4:      0x{:x}", st.real_cr4);
        let _ = writeln!(b, "  real cr0:      0x{:x}", st.real_cr0);
        let _ = writeln!(
            b,
            "  raw probe:     mode={} stage={}",
            raw_probe_label(st.raw_probe_mode),
            raw_probe_label(st.raw_probe_stage)
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = writeln!(b, "trampoline: unavailable on this architecture");
    }
    b.flush(tcp);
}

// ── cores ───────────────────────────────────────────────────────────

fn cores(tcp: &mut Tcp) {
    let mut b = Buf::new();
    #[cfg(feature = "avx512-acceleration")]
    {
        let active = crate::smp::active_cores();
        let registered = crate::smp::registered_cores();
        let _ = writeln!(
            b,
            "SMP: active={} registered={} cap={} ap_gate={} requested={}",
            active,
            registered,
            crate::smp::MAX_CORES,
            if crate::smp::ap_boot_gate_active() {
                "armed"
            } else {
                "open"
            },
            bool_label(crate::smp::ap_boot_requested())
        );
        let limit = crate::smp::ap_boot_ap_limit();
        if limit == u32::MAX {
            let _ = writeln!(b, "  ap_limit: all");
        } else {
            let _ = writeln!(b, "  ap_limit: {}", limit);
        }
        let _ = writeln!(b, "  core   APIC ID   node   unit");
        for idx in 0..(active as usize).min(crate::smp::MAX_CORES) {
            let apic = crate::smp::apic_id_of_core(idx);
            if apic == u32::MAX {
                let _ = writeln!(b, "  {:>3}    (not registered)", idx);
            } else {
                let (node, unit) = crate::smp::topology_of_core(idx);
                if node == u32::MAX || unit == u32::MAX {
                    let _ = writeln!(b, "  {:>3}    {:>7}   ?      ?", idx, apic);
                } else {
                    let _ = writeln!(
                        b,
                        "  {:>3}    {:>7}   {:>3}    {:>3}",
                        idx, apic, node, unit
                    );
                }
            }
        }
    }
    #[cfg(not(feature = "avx512-acceleration"))]
    {
        let _ = writeln!(
            b,
            "SMP: 1 core (kernel built without avx512-acceleration; AP wake-up disabled)"
        );
    }
    b.flush(tcp);
}

// ── version ─────────────────────────────────────────────────────────

fn version(tcp: &mut Tcp) {
    let mut b = Buf::new();
    let _ = writeln!(b, "Zero v{}", env!("CARGO_PKG_VERSION"));
    let _ = writeln!(b, "  target arch:   {}", target_arch_name());
    let _ = writeln!(
        b,
        "  build profile: {}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    let _ = write!(b, "  features:      ");
    let mut any = false;
    macro_rules! feat {
        ($name:literal) => {
            if cfg!(feature = $name) {
                if any {
                    let _ = write!(b, ", ");
                }
                let _ = write!(b, $name);
                any = true;
            }
        };
    }
    feat!("avx512-acceleration");
    feat!("neon-acceleration");
    feat!("streaming-mode");
    feat!("nondeterministic-sampling");
    feat!("cherry-net");
    feat!("zero-control-plane");
    if !any {
        let _ = write!(b, "(default)");
    }
    let _ = writeln!(b);
    b.flush(tcp);
}

// ── reboot helper ───────────────────────────────────────────────────

/// Reset the platform. Returns only if the reset attempt fails (and the
/// caller is expected to `hlt` after this).
pub fn trigger_reset() -> ! {
    // 8042 keyboard-controller reset: pulse the CPU reset line.
    // Wait for the input buffer to drain, then write 0xFE to the
    // command port. Reliable on every PC-compatible platform that
    // implements the legacy 8042 interface, which includes QEMU and
    // every commodity x86 server we care about.
    unsafe {
        for _ in 0..10_000u32 {
            let status: u8;
            core::arch::asm!(
                "in al, dx",
                in("dx") 0x64u16,
                out("al") status,
                options(nomem, nostack, preserves_flags),
            );
            if status & 0x02 == 0 {
                break;
            }
        }
        core::arch::asm!(
            "out dx, al",
            in("dx") 0x64u16,
            in("al") 0xFEu8,
            options(nomem, nostack, preserves_flags),
        );
    }
    // Fallback if the keyboard-controller reset didn't take: triple
    // fault by loading a zero-length IDT and raising an interrupt.
    unsafe {
        #[repr(C, packed)]
        struct IdtrZero {
            limit: u16,
            base: u64,
        }
        let idtr = IdtrZero { limit: 0, base: 0 };
        core::arch::asm!(
            "lidt [{0}]",
            "int3",
            in(reg) &idtr,
            options(nostack),
        );
    }
    // If even that fails, just spin.
    loop {
        unsafe {
            core::arch::asm!("hlt", options(nomem, nostack, preserves_flags));
        }
    }
}

// ── shared formatting helpers ───────────────────────────────────────

fn write_mac(b: &mut Buf, mac: Mac) {
    for (i, byte) in mac.iter().enumerate() {
        if i > 0 {
            let _ = write!(b, ":");
        }
        let _ = write!(b, "{:02x}", byte);
    }
}

fn write_ip(b: &mut Buf, ip: Ipv4) {
    let _ = write!(b, "{}.{}.{}.{}", ip[0], ip[1], ip[2], ip[3]);
}

fn bool_label(v: bool) -> &'static str {
    if v {
        "yes"
    } else {
        "no"
    }
}

fn raw_probe_label(value: u8) -> &'static str {
    match value as u32 {
        crate::smp::AP_BOOT_MODE_FULL => "idle",
        crate::smp::AP_PROBE_STAGE_ENTRY => "entry",
        crate::smp::AP_PROBE_STAGE_IDT => "idt",
        crate::smp::AP_PROBE_STAGE_APIC => "apic",
        crate::smp::AP_PROBE_STAGE_SIMD => "simd",
        crate::smp::AP_PROBE_STAGE_TRAMP_REAL => "tramp-real",
        crate::smp::AP_PROBE_STAGE_TRAMP_PROT => "tramp-prot",
        crate::smp::AP_PROBE_STAGE_TRAMP_PAE => "tramp-pae",
        crate::smp::AP_PROBE_STAGE_TRAMP_EFER => "tramp-efer",
        crate::smp::AP_PROBE_STAGE_TRAMP_PAGING => "tramp-paging",
        crate::smp::AP_PROBE_STAGE_TRAMP_LONG => "tramp-long",
        crate::smp::AP_PROBE_STAGE_TRAMP_CR3 => "tramp-cr3",
        crate::smp::AP_PROBE_STAGE_TRAMP_RUST => "tramp-rust",
        _ => "unknown",
    }
}

#[cfg(target_arch = "x86_64")]
fn current_tick() -> u64 {
    crate::arch::pit::ticks()
}

#[cfg(not(target_arch = "x86_64"))]
fn current_tick() -> u64 {
    0
}

fn target_arch_name() -> &'static str {
    if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "unknown"
    }
}

fn split_first_word(input: &[u8]) -> (&[u8], &[u8]) {
    let mut split = input.len();
    for (i, &c) in input.iter().enumerate() {
        if matches!(c, b' ' | b'\t') {
            split = i;
            break;
        }
    }
    let head = &input[..split];
    let mut rest_start = split;
    while rest_start < input.len() && matches!(input[rest_start], b' ' | b'\t') {
        rest_start += 1;
    }
    (head, &input[rest_start..])
}

fn trim(input: &[u8]) -> &[u8] {
    let mut start = 0;
    while start < input.len() && matches!(input[start], b' ' | b'\t') {
        start += 1;
    }
    let mut end = input.len();
    while end > start && matches!(input[end - 1], b' ' | b'\t') {
        end -= 1;
    }
    &input[start..end]
}

fn parse_u32(input: &[u8]) -> Option<u32> {
    if input.is_empty() {
        return None;
    }
    let mut value: u32 = 0;
    for &b in input {
        if !b.is_ascii_digit() {
            return None;
        }
        value = value.checked_mul(10)?;
        value = value.checked_add((b - b'0') as u32)?;
    }
    Some(value)
}

/// Pick the telemetry record that matches this build's accelerator
/// feature set. The throughput value is the last measured forward-pass
/// rate at the time of the F-series cycle; commands that report
/// "current" tok/s consume this.
fn current_telemetry() -> crate::lfb::telemetry_data::TelemetryData {
    use crate::lfb::telemetry_data::TelemetryData;
    #[cfg(feature = "avx512-acceleration")]
    {
        return TelemetryData::avx512_mode();
    }
    #[cfg(all(not(feature = "avx512-acceleration"), feature = "neon-acceleration"))]
    {
        return TelemetryData::neon_mode();
    }
    #[cfg(all(
        not(feature = "avx512-acceleration"),
        not(feature = "neon-acceleration")
    ))]
    {
        TelemetryData::scalar_mode()
    }
}
