// SPDX-License-Identifier: AGPL-3.0-or-later
//! Minimal bare-metal IPv4 networking stack.
//!
//! Scope: e1000 NIC → Ethernet II → ARP / IPv4 → ICMP echo + UDP +
//! single-connection TCP. No DHCP, no fragmentation, no neighbour
//! discovery, no IPv6. Designed for the bring-up of cloud-style
//! "is the kernel alive on the network" diagnostics — paired with
//! the UDP shell on port 9999 (datagram surface) and the TCP shell
//! on port 2222 (stream surface), it gives an out-of-band probe
//! and remote-console surface without dragging in smoltcp.
//!
//! Polling-only: [`Stack::poll`] should be called in the kernel idle
//! loop (or a dedicated task) — RX descriptors are walked, parsed,
//! and any responses are emitted via the same NIC. The TCP layer
//! additionally needs [`Stack::poll`] called periodically to drive
//! retransmits and queued data; the existing PIT-tick driver in
//! `main.rs` satisfies that.

// Several parser-surface fields (e.g. UDP length, ARP target_mac,
// IPv4 identification) are parsed for protocol correctness even
// though the single-shell-server send path doesn't read them all.
// Likewise the active-send helpers (`build_request`, `next_hop`,
// `op_is_reply`) are kept as part of the public API for a future
// outbound use case but are unused by today's responder-only paths.
#![allow(dead_code)]

pub mod arp;
pub mod cmds;
pub mod e1000;
pub mod eth;
pub mod i40e;
pub mod ice;
pub mod icmp;
pub mod ipv4;
pub mod shell;
pub mod tcp;
pub mod tcp_shell;
pub mod udp;

use core::fmt::Write;
use core::sync::atomic::{AtomicPtr, Ordering};

use crate::arch::serial::Serial;
use crate::arch::x86_64::pcie::PciScan;

use self::arp::ArpCache;
use self::e1000::E1000;
use self::eth::Mac;
use self::i40e::I40e;
use self::ice::Ice;
use self::ipv4::Ipv4;
use self::tcp_shell::TcpShell;

/// NIC handle — at most one Intel NIC is bound per boot. The enum
/// avoids a `dyn Trait` indirection while letting the Stack's hot
/// paths stay branch-cheap (single match, no vtable). Adding a new
/// driver = adding a variant and three `match` arms.
pub enum Nic {
    E1000(E1000),
    I40e(I40e),
    Ice(Ice),
}

impl Nic {
    pub fn mac(&self) -> Mac {
        match self {
            Nic::E1000(n) => n.mac(),
            Nic::I40e(n) => n.mac(),
            Nic::Ice(n) => n.mac(),
        }
    }

    fn kind_label(&self) -> &'static str {
        match self {
            Nic::E1000(_) => "e1000",
            Nic::I40e(_) => "i40e",
            Nic::Ice(_) => "ice",
        }
    }

    pub fn transmit(&mut self, frame: &[u8]) {
        match self {
            Nic::E1000(n) => n.transmit(frame),
            Nic::I40e(n) => n.transmit(frame),
            Nic::Ice(n) => n.transmit(frame),
        }
    }

    pub fn receive(&mut self, out: &mut [u8]) -> Option<usize> {
        match self {
            Nic::E1000(n) => n.receive(out),
            Nic::I40e(n) => n.receive(out),
            Nic::Ice(n) => n.receive(out),
        }
    }
}

// ── Static IP profile ───────────────────────────────────────────────
//
// Selected at compile-time. The default profile targets QEMU's
// user-mode networking (`-netdev user`). Build with
// `--features cherry-net` to swap in a bare-metal static-IP
// configuration. Replace the placeholder address/gateway/netmask
// below with the values for your own deployment before flashing.

#[cfg(not(feature = "cherry-net"))]
mod profile {
    use super::Ipv4;
    pub const STATIC_IP: Ipv4 = [10, 0, 2, 15];
    pub const GATEWAY_IP: Ipv4 = [10, 0, 2, 2];
    pub const NETMASK: Ipv4 = [255, 255, 255, 0];
    pub const LABEL: &str = "QEMU user-mode (10.0.2.0/24)";
}

#[cfg(feature = "cherry-net")]
mod profile {
    use super::Ipv4;
    // Bare-metal static-IP profile. Placeholder values — set these to
    // the public IP / gateway / netmask of your own server before
    // flashing a production image.
    pub const STATIC_IP: Ipv4 = [10, 0, 0, 2];
    pub const GATEWAY_IP: Ipv4 = [10, 0, 0, 1];
    pub const NETMASK: Ipv4 = [255, 255, 255, 0];
    pub const LABEL: &str = "Bare-metal static IP (10.0.0.0/24)";
}

pub const STATIC_IP: Ipv4 = profile::STATIC_IP;
pub const GATEWAY_IP: Ipv4 = profile::GATEWAY_IP;
pub const NETMASK: Ipv4 = profile::NETMASK;
pub const PROFILE_LABEL: &str = profile::LABEL;

/// Boot-time net bring-up outcome, mirrored onto the held benchmark
/// screen (`bench::print_summary`). The Cherry boxes are operated
/// through the provider's HTML5 KVM, which shows only the VGA text
/// buffer — the `ice:`/`net:` serial lines scroll away before the
/// benchmark screen freezes. Recording the outcome here means a single
/// KVM photo answers "why is the box unreachable" without COM1 access.
///
/// Single-writer: the BSP sets this once during Stage 10.8, strictly
/// before `print_summary` can run. The Release/Acquire pair on STATE
/// publishes the detail bytes.
pub mod bind_report {
    use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

    pub const NOT_RUN: u8 = 0;
    pub const ONLINE: u8 = 1;
    pub const FAILED: u8 = 2;

    pub const DETAIL_CAP: usize = 64;
    static STATE: AtomicU8 = AtomicU8::new(NOT_RUN);
    static DETAIL_LEN: AtomicUsize = AtomicUsize::new(0);
    #[allow(clippy::declare_interior_mutable_const)]
    const ZERO: AtomicU8 = AtomicU8::new(0);
    static DETAIL: [AtomicU8; DETAIL_CAP] = [ZERO; DETAIL_CAP];

    struct DetailWriter {
        len: usize,
    }

    impl core::fmt::Write for DetailWriter {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            for &b in s.as_bytes() {
                if self.len >= DETAIL_CAP {
                    break;
                }
                // Keep it plain ASCII so the VGA text buffer renders it.
                let b = if (0x20..0x7F).contains(&b) { b } else { b'?' };
                DETAIL[self.len].store(b, Ordering::Relaxed);
                self.len += 1;
            }
            Ok(())
        }
    }

    pub fn set(state: u8, detail: core::fmt::Arguments<'_>) {
        let mut w = DetailWriter { len: 0 };
        let _ = core::fmt::write(&mut w, detail);
        DETAIL_LEN.store(w.len, Ordering::Release);
        STATE.store(state, Ordering::Release);
    }

    pub fn state() -> u8 {
        STATE.load(Ordering::Acquire)
    }

    /// Copy the detail text into `buf`, returning the byte count.
    /// Copy-out instead of a borrowed `&str` keeps every access on
    /// atomics — no references into shared mutable storage.
    pub fn detail(buf: &mut [u8; DETAIL_CAP]) -> usize {
        let n = DETAIL_LEN.load(Ordering::Acquire).min(DETAIL_CAP);
        for (i, slot) in buf.iter_mut().enumerate().take(n) {
            *slot = DETAIL[i].load(Ordering::Relaxed);
        }
        n
    }
}

const FRAME_BUFFER_SIZE: usize = 2048;
const SEND_BUFFER_SIZE: usize = 1600;
/// Maximum bytes a TCP-stack callback may write into the scratch
/// segment buffer. = SEND_BUFFER_SIZE - eth(14) - ip(20).
const TCP_SEGMENT_MAX: usize = SEND_BUFFER_SIZE - eth::MAX_ETH_HEADER_LEN - ipv4::IPV4_HEADER_LEN;

pub struct Stack {
    nic: Nic,
    arp: ArpCache,
    ip: Ipv4,
    gateway: Ipv4,
    ip_id: u16,
    rx_buf: [u8; FRAME_BUFFER_SIZE],
    tx_buf: [u8; SEND_BUFFER_SIZE],
    /// VLAN tag from the frame currently being handled. Replies to
    /// ARP/ICMP/UDP preserve it so tagged datacenter ports work.
    rx_vlan: eth::VlanStack,
    /// VLAN tag associated with the active TCP shell connection.
    /// Timer-driven retransmits happen outside the ingress frame
    /// context, so they need a remembered tag.
    tcp_vlan: eth::VlanStack,
    /// Scratch buffer used to materialise a TCP segment before it
    /// gets wrapped in IPv4 + Ethernet. Keeps the TCP layer free of
    /// any awareness of the NIC's send buffer.
    tcp_scratch: [u8; TCP_SEGMENT_MAX],
    tcp_shell: TcpShell,
}

#[derive(Debug, Clone, Copy)]
pub enum StackError {
    /// No supported Intel NIC was found on any PCI bus.
    NoSupportedNic,
    /// An e1000-family device was found but failed to initialise.
    E1000(e1000::NicError),
    /// An i40e-family (X710) device was found but failed to initialise.
    I40e(i40e::NicError),
    /// An ice-family (E810) device was found but failed to initialise.
    Ice(ice::NicError),
}

impl Stack {
    /// Bind to the first supported Intel NIC and bring the stack
    /// online with the active profile's static IP. Tries the
    /// 8254x/e1000 family first (covers QEMU's default `-device
    /// e1000` and most older server-class cards), then falls back
    /// to the 700-series i40e family (X710 / XL710 / XXV710 / X722),
    /// and finally the 800-series ice family (E810-XXV / E810-C —
    /// shipping in the latest Cherry Server batch).
    pub fn bind(scan: &PciScan) -> Result<Self, StackError> {
        let nic = match E1000::bind(scan) {
            Ok(n) => Nic::E1000(n),
            Err(e1000::NicError::NotFound) => {
                // No 8254x part on the bus — try the 700-series.
                match I40e::bind(scan) {
                    Ok(n) => Nic::I40e(n),
                    Err(i40e::NicError::NotFound) => {
                        // No X710 either — try the 800-series E810.
                        match Ice::bind(scan) {
                            Ok(n) => Nic::Ice(n),
                            Err(ice::NicError::NotFound) => return Err(StackError::NoSupportedNic),
                            Err(e) => return Err(StackError::Ice(e)),
                        }
                    }
                    Err(e) => return Err(StackError::I40e(e)),
                }
            }
            Err(e) => return Err(StackError::E1000(e)),
        };
        let mut stack = Stack {
            nic,
            arp: ArpCache::new(),
            ip: STATIC_IP,
            gateway: GATEWAY_IP,
            ip_id: 0,
            rx_buf: [0; FRAME_BUFFER_SIZE],
            tx_buf: [0; SEND_BUFFER_SIZE],
            rx_vlan: eth::VlanStack::empty(),
            tcp_vlan: eth::VlanStack::empty(),
            tcp_scratch: [0; TCP_SEGMENT_MAX],
            tcp_shell: TcpShell::new(),
        };
        // "stack online" is reached only after the NIC's bind path
        // returned Ok. For ice (E810) this now requires the TX *and*
        // RX queues to actually come up — see the FATAL return paths
        // in `kernel/src/net/ice.rs::bind_device`. For i40e (X710)
        // the bind tolerates a half-bring-up and the `ready` flag
        // gates the data path; healthy hardware reaches this line
        // only after `bring_up_data_path()` succeeded.
        let _ = writeln!(
            Serial,
            "net: stack online [{}] — IP {}.{}.{}.{} / gw {}.{}.{}.{} (TX+RX verified)",
            PROFILE_LABEL,
            STATIC_IP[0],
            STATIC_IP[1],
            STATIC_IP[2],
            STATIC_IP[3],
            GATEWAY_IP[0],
            GATEWAY_IP[1],
            GATEWAY_IP[2],
            GATEWAY_IP[3]
        );
        let _ = writeln!(
            Serial,
            "net: UDP shell listening on port {} — try `nc -u <ip> {}`",
            shell::SHELL_PORT,
            shell::SHELL_PORT
        );
        let _ = writeln!(
            Serial,
            "net: TCP shell listening on port {} — try `nc <ip> {}`",
            tcp_shell::PORT,
            tcp_shell::PORT
        );
        stack.announce_ipv4();
        let _ = writeln!(Serial, "net: sent gratuitous ARP + gateway ARP probe");
        bind_report::set(
            bind_report::ONLINE,
            format_args!(
                "{} {}.{}.{}.{} TX+RX ok",
                stack.nic.kind_label(),
                STATIC_IP[0],
                STATIC_IP[1],
                STATIC_IP[2],
                STATIC_IP[3]
            ),
        );
        Ok(stack)
    }

    pub fn mac(&self) -> Mac {
        self.nic.mac()
    }

    /// Announce our IPv4 presence on the L2 segment.
    ///
    /// The stack is responder-first, but bare-metal datacenter ports can
    /// take a while to learn a fresh source MAC after PXE/UEFI hands the
    /// NIC to Zero. A gratuitous ARP plus a gateway ARP probe gives
    /// the upstream switch/router an immediate frame sourced from our
    /// MAC before external probes arrive.
    pub fn announce_ipv4(&mut self) {
        let our_mac = self.nic.mac();
        let mut frame = [0u8; 64];

        let n = arp::build_request(&mut frame, our_mac, self.ip, self.ip);
        self.nic.transmit(&frame[..n]);

        let n = arp::build_request(&mut frame, our_mac, self.ip, self.gateway);
        self.nic.transmit(&frame[..n]);
    }

    /// Drain pending RX frames and dispatch each through the protocol
    /// stack. Also drives the TCP layer's timer-based work (retransmit,
    /// queued-data flush). Returns the number of frames processed.
    pub fn poll(&mut self) -> usize {
        let mut processed = 0;
        loop {
            let n = match self.nic.receive(&mut self.rx_buf) {
                Some(n) => n,
                None => break,
            };
            self.handle_frame(n);
            processed += 1;
            // Bound per-call work so a flood cannot monopolise the
            // timer ISR. Sized to the LARGEST RX ring (ice/i40e = 64;
            // e1000 = 16): with the 10 ms PIT poll cadence a 16-frame
            // bound drains at most 1,600 frames/s — background scan
            // noise on a public datacenter IP can exceed that and
            // would starve the ring. One full ring per tick keeps the
            // worst-case ISR work bounded at 64 parses.
            if processed >= ice::RX_DESC_COUNT {
                break;
            }
        }

        // Drive the TCP shell once per poll regardless of whether an
        // RX frame arrived. Handles retransmit + queued-data flush.
        self.drive_tcp();
        processed
    }

    fn handle_frame(&mut self, len: usize) {
        let frame = unsafe { core::slice::from_raw_parts(self.rx_buf.as_ptr(), len) };
        let Some((hdr, payload)) = eth::parse(frame) else {
            return;
        };
        self.rx_vlan = hdr.vlan;
        match hdr.ethertype {
            eth::ETHERTYPE_ARP => self.handle_arp(payload),
            eth::ETHERTYPE_IPV4 => self.handle_ipv4(hdr.src, payload),
            _ => {}
        }
        self.rx_vlan = eth::VlanStack::empty();
    }

    fn handle_arp(&mut self, payload: &[u8]) {
        let Some(pkt) = arp::parse(payload) else {
            return;
        };
        // Always cache what we learn from the sender.
        self.arp.insert(pkt.sender_ip, pkt.sender_mac);

        if arp::op_is_request(pkt.op) && pkt.target_ip == self.ip {
            let our_mac = self.nic.mac();
            let mut buf = [0u8; 64];
            let n = arp::build_reply_tagged(&mut buf, our_mac, &pkt, self.rx_vlan);
            self.nic.transmit(&buf[..n]);
        }
    }

    fn handle_ipv4(&mut self, src_mac: Mac, payload: &[u8]) {
        let Some((hdr, body)) = ipv4::parse(payload) else {
            return;
        };
        if hdr.dst != self.ip {
            return;
        }
        // Opportunistic ARP learning on every IPv4 frame: cache the
        // sender so subsequent responses don't need an ARP query.
        self.arp.insert(hdr.src, src_mac);

        match hdr.protocol {
            ipv4::PROTO_ICMP => self.handle_icmp(hdr.src, src_mac, body),
            ipv4::PROTO_UDP => self.handle_udp(hdr.src, src_mac, body),
            ipv4::PROTO_TCP => self.handle_tcp(hdr.src, src_mac, body),
            _ => {}
        }
    }

    fn handle_icmp(&mut self, src_ip: Ipv4, src_mac: Mac, body: &[u8]) {
        let Some((hdr, data)) = icmp::parse(body) else {
            return;
        };
        if !icmp::is_echo_request(&hdr) {
            return;
        }
        // Cap echoed payload so the entire reply frame fits within
        // SEND_BUFFER_SIZE: eth(14) + ip(20) + icmp(8) + data.
        let max_data = SEND_BUFFER_SIZE
            - (eth::header_len(self.rx_vlan) + ipv4::IPV4_HEADER_LEN + icmp::ICMP_HEADER_LEN);
        let data = if data.len() > max_data {
            &data[..max_data]
        } else {
            data
        };

        let our_mac = self.nic.mac();
        let mut frame = [0u8; SEND_BUFFER_SIZE];
        let mut n = eth::write_tagged(
            &mut frame,
            src_mac,
            our_mac,
            self.rx_vlan,
            eth::ETHERTYPE_IPV4,
        );

        // ICMP body lives at offset eth+ip; build it first so the IPv4
        // header has the right total-length / payload size.
        let icmp_off = n + ipv4::IPV4_HEADER_LEN;
        let icmp_len = icmp::build_echo_reply(&mut frame[icmp_off..], &hdr, data);

        let ip_id = self.next_ip_id();
        ipv4::write_header(
            &mut frame[n..n + ipv4::IPV4_HEADER_LEN],
            self.ip,
            src_ip,
            ipv4::PROTO_ICMP,
            icmp_len as u16,
            ip_id,
        );
        n += ipv4::IPV4_HEADER_LEN + icmp_len;
        self.nic.transmit(&frame[..n]);
    }

    fn handle_udp(&mut self, src_ip: Ipv4, src_mac: Mac, body: &[u8]) {
        let Some((hdr, data)) = udp::parse(body) else {
            return;
        };
        if hdr.dst_port != shell::SHELL_PORT {
            return;
        }
        let cmd = trim_datagram(data);
        let (head, tail) = split_datagram_word(cmd);
        if cmd == b"tcp-reset" {
            self.tcp_shell.tcp.force_reset();
            self.send_udp(
                src_ip,
                src_mac,
                shell::SHELL_PORT,
                hdr.src_port,
                b"tcp: reset to LISTEN\n",
            );
            return;
        }
        if cmd == b"reboot" {
            self.send_udp(
                src_ip,
                src_mac,
                shell::SHELL_PORT,
                hdr.src_port,
                b"rebooting...\n",
            );
            cmds::trigger_reset();
        }
        if cmd == b"tcp" || cmd == b"tcp-status" {
            let msg = match self.tcp_shell.state() {
                tcp::State::Closed => b"tcp: Closed\n" as &[u8],
                tcp::State::Listen => b"tcp: Listen\n" as &[u8],
                tcp::State::SynReceived => b"tcp: SynReceived\n" as &[u8],
                tcp::State::Established => b"tcp: Established\n" as &[u8],
                tcp::State::CloseWait => b"tcp: CloseWait\n" as &[u8],
                tcp::State::LastAck => b"tcp: LastAck\n" as &[u8],
                tcp::State::FinWait1 => b"tcp: FinWait1\n" as &[u8],
                tcp::State::FinWait2 => b"tcp: FinWait2\n" as &[u8],
                tcp::State::TimeWait => b"tcp: TimeWait\n" as &[u8],
            };
            self.send_udp(src_ip, src_mac, shell::SHELL_PORT, hdr.src_port, msg);
            return;
        }
        if cmd == b"diag" || cmd == b"debug" {
            let resp = self.udp_diag_response();
            self.send_udp(
                src_ip,
                src_mac,
                shell::SHELL_PORT,
                hdr.src_port,
                resp.bytes(),
            );
            return;
        }
        if cmd == b"mem" || cmd == b"mem-status" || cmd == b"memory" {
            let resp = udp_mem_response();
            self.send_udp(
                src_ip,
                src_mac,
                shell::SHELL_PORT,
                hdr.src_port,
                resp.bytes(),
            );
            return;
        }
        if cmd == b"model"
            || cmd == b"model-status"
            || (head == b"llm" && split_datagram_word(tail).0 == b"model")
        {
            let resp = udp_model_response();
            self.send_udp(
                src_ip,
                src_mac,
                shell::SHELL_PORT,
                hdr.src_port,
                resp.bytes(),
            );
            return;
        }
        if cmd == b"llm" || cmd == b"llm-status" || cmd == b"inference" {
            let resp = udp_llm_status_response();
            self.send_udp(
                src_ip,
                src_mac,
                shell::SHELL_PORT,
                hdr.src_port,
                resp.bytes(),
            );
            return;
        }
        if head == b"llm-profile"
            || head == b"llm-reset"
            || (head == b"llm" && split_datagram_word(tail).0 == b"profile")
        {
            let args = if head == b"llm" {
                split_datagram_word(tail).1
            } else if head == b"llm-reset" {
                b"reset"
            } else {
                tail
            };
            let resp = udp_llm_profile_response(args);
            self.send_udp(
                src_ip,
                src_mac,
                shell::SHELL_PORT,
                hdr.src_port,
                resp.bytes(),
            );
            return;
        }
        if head == b"llm-start"
            || head == b"inference-start"
            || (head == b"llm" && split_datagram_word(tail).0 == b"start")
            || (head == b"inference" && split_datagram_word(tail).0 == b"start")
        {
            let resp = udp_llm_start_response();
            self.send_udp(
                src_ip,
                src_mac,
                shell::SHELL_PORT,
                hdr.src_port,
                resp.bytes(),
            );
            return;
        }
        if cmd == b"apic" || cmd == b"lapic" {
            let resp = udp_apic_response();
            self.send_udp(
                src_ip,
                src_mac,
                shell::SHELL_PORT,
                hdr.src_port,
                resp.bytes(),
            );
            return;
        }
        if cmd == b"madt" {
            let resp = udp_madt_response();
            self.send_udp(
                src_ip,
                src_mac,
                shell::SHELL_PORT,
                hdr.src_port,
                resp.bytes(),
            );
            return;
        }
        if cmd == b"trampoline" {
            let resp = udp_trampoline_response();
            self.send_udp(
                src_ip,
                src_mac,
                shell::SHELL_PORT,
                hdr.src_port,
                resp.bytes(),
            );
            return;
        }
        if cmd == b"smp" || cmd == b"smp-status" {
            let resp = udp_smp_status_response();
            self.send_udp(
                src_ip,
                src_mac,
                shell::SHELL_PORT,
                hdr.src_port,
                resp.bytes(),
            );
            return;
        }
        if head == b"smp-tune" || (head == b"smp" && split_datagram_word(tail).0 == b"tune") {
            let args = if head == b"smp" {
                split_datagram_word(tail).1
            } else {
                tail
            };
            let resp = udp_smp_tune_response(args);
            self.send_udp(
                src_ip,
                src_mac,
                shell::SHELL_PORT,
                hdr.src_port,
                resp.bytes(),
            );
            return;
        }
        if head == b"smp-probe" || (head == b"smp" && split_datagram_word(tail).0 == b"probe") {
            let stage = if head == b"smp" {
                split_datagram_word(tail).1
            } else {
                tail
            };
            let resp = udp_smp_probe_response(stage);
            self.send_udp(
                src_ip,
                src_mac,
                shell::SHELL_PORT,
                hdr.src_port,
                resp.bytes(),
            );
            return;
        }
        if head == b"smp-start" || (head == b"smp" && split_datagram_word(tail).0 == b"start") {
            let limit = if head == b"smp" {
                split_datagram_word(tail).1
            } else {
                tail
            };
            let resp = udp_smp_start_response(limit);
            self.send_udp(
                src_ip,
                src_mac,
                shell::SHELL_PORT,
                hdr.src_port,
                resp.bytes(),
            );
            return;
        }
        let our_mac = self.nic.mac();
        let our_ip = self.ip;
        let req = shell::Request {
            data,
            src_ip,
            src_mac,
            src_port: hdr.src_port,
        };
        let resp = shell::handle(&req, our_mac, our_ip);
        self.send_udp(
            src_ip,
            src_mac,
            shell::SHELL_PORT,
            hdr.src_port,
            resp.bytes(),
        );
    }

    fn handle_tcp(&mut self, src_ip: Ipv4, src_mac: Mac, body: &[u8]) {
        let our_ip = self.ip;
        let now = current_tick();
        self.tcp_vlan = self.rx_vlan;
        if let Some(out) =
            self.tcp_shell
                .tcp
                .on_segment(our_ip, src_ip, src_mac, body, now, &mut self.tcp_scratch)
        {
            self.send_ip_payload(out.dst_ip, out.dst_mac, ipv4::PROTO_TCP, out.len);
        }
        // Re-run drive after every input so greeting / responses get
        // queued without waiting for the next poll tick.
        self.drive_tcp();
    }

    /// Drive TCP timer work and any TCP-shell-side work. Called both
    /// after each handled segment and once per poll iteration.
    fn drive_tcp(&mut self) {
        let our_ip = self.ip;
        let our_mac = self.nic.mac();
        self.tcp_shell.drive(our_mac, our_ip);

        // Pump until the TCP layer has nothing left to send this round.
        // Bounded by the number of MSS-sized chunks the TX buffer can
        // hold so a wedged peer can't keep us in this loop forever.
        let mut budget = (tcp::TX_BUFFER_LEN / tcp::MSS) + 4;
        let now = current_tick();
        while budget > 0 {
            budget -= 1;
            let Some(out) = self.tcp_shell.tcp.poll(our_ip, now, &mut self.tcp_scratch) else {
                break;
            };
            self.send_ip_payload(out.dst_ip, out.dst_mac, ipv4::PROTO_TCP, out.len);
            // Refresh the shell's view after every emitted segment —
            // a sent FIN may unblock pending state cleanup.
            self.tcp_shell.drive(our_mac, our_ip);
        }
    }

    /// Wrap the payload at `self.tcp_scratch[..len]` in IPv4 + Ethernet
    /// addressed to (dst_ip, dst_mac) with the given protocol number,
    /// then transmit.
    fn send_ip_payload(&mut self, dst_ip: Ipv4, dst_mac: Mac, protocol: u8, payload_len: usize) {
        if payload_len == 0 || payload_len > TCP_SEGMENT_MAX {
            return;
        }
        let our_mac = self.nic.mac();
        let vlan = if protocol == ipv4::PROTO_TCP {
            self.tcp_vlan
        } else {
            self.rx_vlan
        };
        let mut n = eth::write_tagged(
            &mut self.tx_buf,
            dst_mac,
            our_mac,
            vlan,
            eth::ETHERTYPE_IPV4,
        );
        let body_off = n + ipv4::IPV4_HEADER_LEN;
        self.tx_buf[body_off..body_off + payload_len]
            .copy_from_slice(&self.tcp_scratch[..payload_len]);
        let ip_id = self.next_ip_id();
        ipv4::write_header(
            &mut self.tx_buf[n..n + ipv4::IPV4_HEADER_LEN],
            self.ip,
            dst_ip,
            protocol,
            payload_len as u16,
            ip_id,
        );
        n += ipv4::IPV4_HEADER_LEN + payload_len;
        self.nic.transmit(&self.tx_buf[..n]);
    }

    fn send_udp(&mut self, dst_ip: Ipv4, dst_mac: Mac, src_port: u16, dst_port: u16, data: &[u8]) {
        let our_mac = self.nic.mac();
        let max_data = SEND_BUFFER_SIZE
            - (eth::header_len(self.rx_vlan) + ipv4::IPV4_HEADER_LEN + udp::UDP_HEADER_LEN);
        let data = if data.len() > max_data {
            &data[..max_data]
        } else {
            data
        };

        let mut n = eth::write_tagged(
            &mut self.tx_buf,
            dst_mac,
            our_mac,
            self.rx_vlan,
            eth::ETHERTYPE_IPV4,
        );
        let udp_off = n + ipv4::IPV4_HEADER_LEN;
        let udp_len = udp::write(
            &mut self.tx_buf[udp_off..],
            self.ip,
            dst_ip,
            src_port,
            dst_port,
            data,
        );
        let ip_id = self.next_ip_id();
        ipv4::write_header(
            &mut self.tx_buf[n..n + ipv4::IPV4_HEADER_LEN],
            self.ip,
            dst_ip,
            ipv4::PROTO_UDP,
            udp_len as u16,
            ip_id,
        );
        n += ipv4::IPV4_HEADER_LEN + udp_len;
        self.nic.transmit(&self.tx_buf[..n]);
    }

    fn next_ip_id(&mut self) -> u16 {
        self.ip_id = self.ip_id.wrapping_add(1);
        self.ip_id
    }

    fn udp_diag_response(&self) -> shell::Response {
        let mut resp = shell::Response::empty();
        let ticks = current_tick();
        let (remote_ip, remote_port) = self.tcp_shell.tcp.remote();
        let _ = writeln!(resp, "diag: Zero v{}", env!("CARGO_PKG_VERSION"));
        let _ = writeln!(resp, "uptime_ticks: {}", ticks);
        let _ = write!(resp, "ip: ");
        write_ip_response(&mut resp, self.ip);
        let _ = write!(resp, " gw: ");
        write_ip_response(&mut resp, self.gateway);
        let _ = writeln!(resp);
        let _ = writeln!(
            resp,
            "tcp: {:?} idle={} age={} remote={}.{}.{}.{}:{}",
            self.tcp_shell.state(),
            self.tcp_shell.tcp.idle_ticks(ticks),
            self.tcp_shell.tcp.state_age_ticks(ticks),
            remote_ip[0],
            remote_ip[1],
            remote_ip[2],
            remote_ip[3],
            remote_port
        );
        write_smp_status_response(&mut resp);
        write_madt_response(&mut resp);
        write_apic_response(&mut resp);
        write_trampoline_response(&mut resp);
        write_llm_status_response(&mut resp);
        resp
    }

    /// Address belongs to the local subnet — anything else routes
    /// through the gateway. Used by `send_to`, not by the RX paths
    /// (RX never originates traffic, only replies).
    #[allow(dead_code)]
    pub fn next_hop(&self, dst: Ipv4) -> Ipv4 {
        let same_subnet = dst[0] & NETMASK[0] == self.ip[0] & NETMASK[0]
            && dst[1] & NETMASK[1] == self.ip[1] & NETMASK[1]
            && dst[2] & NETMASK[2] == self.ip[2] & NETMASK[2]
            && dst[3] & NETMASK[3] == self.ip[3] & NETMASK[3];
        if same_subnet {
            dst
        } else {
            self.gateway
        }
    }
}

/// Read the kernel's monotonic PIT tick counter. The net stack only
/// uses this for TCP's coarse retransmit timer; if PIT isn't wired
/// for some reason we degrade to a zero-tick clock (everything still
/// works, retransmit just fires aggressively).
#[cfg(target_arch = "x86_64")]
fn current_tick() -> u64 {
    crate::arch::pit::ticks()
}

#[cfg(not(target_arch = "x86_64"))]
fn current_tick() -> u64 {
    0
}

fn trim_datagram(input: &[u8]) -> &[u8] {
    let mut start = 0;
    while start < input.len() && matches!(input[start], b' ' | b'\t' | b'\r' | b'\n') {
        start += 1;
    }
    let mut end = input.len();
    while end > start && matches!(input[end - 1], b' ' | b'\t' | b'\r' | b'\n') {
        end -= 1;
    }
    &input[start..end]
}

fn split_datagram_word(input: &[u8]) -> (&[u8], &[u8]) {
    let mut split = input.len();
    for (i, &b) in input.iter().enumerate() {
        if matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
            split = i;
            break;
        }
    }
    let mut rest = split;
    while rest < input.len() && matches!(input[rest], b' ' | b'\t' | b'\r' | b'\n') {
        rest += 1;
    }
    (&input[..split], &input[rest..])
}

fn udp_llm_status_response() -> shell::Response {
    let mut resp = shell::Response::empty();
    write_llm_status_response(&mut resp);
    resp
}

fn udp_model_response() -> shell::Response {
    let mut resp = shell::Response::empty();
    write_model_status_response(&mut resp);
    resp
}

fn udp_mem_response() -> shell::Response {
    let mut resp = shell::Response::empty();
    #[cfg(target_arch = "x86_64")]
    {
        let model = crate::control_plane::model_region_snapshot();
        if model.present {
            let _ = writeln!(
                resp,
                "mem: model {} va={:#x} len={}MiB",
                crate::control_plane::model_source_label(model.source_id),
                model.virt_addr,
                model.len_bytes / (1024 * 1024)
            );
            write_udp_mapping_probe(&mut resp, "model_first", model.virt_addr);
            write_udp_mapping_probe(
                &mut resp,
                "model_last",
                model
                    .virt_addr
                    .saturating_add(model.len_bytes)
                    .saturating_sub(1),
            );
        } else {
            let _ = writeln!(resp, "mem: model not recorded");
        }
        write_udp_mapping_probe(
            &mut resp,
            "activation",
            crate::memory::ACTIVATION_ARENA_START.load(Ordering::Acquire),
        );
        write_udp_mapping_probe(
            &mut resp,
            "kv_cache",
            crate::memory::KV_CACHE_ARENA_START.load(Ordering::Acquire),
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = writeln!(resp, "mem: mapping diagnostics unavailable");
    }
    resp
}

#[cfg(target_arch = "x86_64")]
fn write_udp_mapping_probe(resp: &mut shell::Response, label: &str, va: u64) {
    if va == 0 {
        let _ = writeln!(resp, "mem: {} unmapped", label);
        return;
    }
    match crate::memory::mapping_info(va) {
        Some(info) => {
            let cache = if info.pcd || info.pwt { "NON-WB" } else { "WB" };
            let _ = writeln!(
                resp,
                "mem: {} pa={:#x} page={}KiB cache={} pcd={} pwt={} huge={} flags={:#x}",
                label,
                info.phys_addr,
                info.frame_size / 1024,
                cache,
                bool_label(info.pcd),
                bool_label(info.pwt),
                bool_label(info.huge),
                info.flags_bits
            );
        }
        None => {
            let _ = writeln!(resp, "mem: {} va={:#x} not-mapped", label, va);
        }
    }
}

fn udp_llm_start_response() -> shell::Response {
    let mut resp = shell::Response::empty();
    crate::control_plane::request_boot_llm_start();
    let _ = writeln!(resp, "llm: Boot-LLM start requested");
    if crate::control_plane::boot_llm_gate_active() {
        let _ = writeln!(resp, "llm: Stage-11 gate releases after UDP response");
    } else {
        let _ = writeln!(
            resp,
            "llm: no active gate; status={}",
            crate::control_plane::status_label(crate::control_plane::status())
        );
    }
    resp
}

fn udp_llm_profile_response(args: &[u8]) -> shell::Response {
    let mut resp = shell::Response::empty();
    let args = trim_datagram(args);
    if args == b"reset" {
        crate::control_plane::reset_llm_profile();
        #[cfg(all(target_arch = "x86_64", feature = "avx512-acceleration"))]
        crate::inference_avx512::reset_perf_counters();
        let _ = writeln!(resp, "llm-profile: reset");
        return resp;
    }

    let cp = crate::control_plane::llm_profile_snapshot();
    if cp.valid {
        let _ = writeln!(
            resp,
            "llm-profile: run={} prompt={} generated={}",
            cp.run_id, cp.prompt_tokens, cp.generated_tokens
        );
        write_udp_rate(
            &mut resp,
            "llm-profile: decode_wall",
            cp.generated_tokens,
            cp.generation_wall_cycles,
        );
        write_udp_rate(
            &mut resp,
            "llm-profile: decode_compute",
            cp.generated_tokens,
            cp.generation_compute_cycles,
        );
        let _ = writeln!(
            resp,
            "llm-profile: prefill wall={} fwd={} lm={} scan={}",
            cp.prefill_wall_cycles,
            cp.prefill_forward_cycles,
            cp.prefill_lm_head_cycles,
            cp.prefill_logit_scan_cycles
        );
        let _ = writeln!(
            resp,
            "llm-profile: gen wall={} compute={} fwd={} lm={} sample={} scan={} render={}",
            cp.generation_wall_cycles,
            cp.generation_compute_cycles,
            cp.generation_forward_cycles,
            cp.generation_lm_head_cycles,
            cp.generation_sample_cycles,
            cp.generation_logit_scan_cycles,
            cp.generation_render_cycles
        );
    } else {
        let _ = writeln!(resp, "llm-profile: no completed generation profile");
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512-acceleration"))]
    {
        let p = crate::inference_avx512::perf_counters_snapshot();
        let _ = writeln!(
            resp,
            "llm-profile: q4k_calls={} q4k_cycles={}",
            p.q4k_calls, p.q4k_cycles
        );
        let _ = writeln!(
            resp,
            "llm-profile: q6k_calls={} q6k_cycles={}",
            p.q6k_calls, p.q6k_cycles
        );
        let _ = writeln!(
            resp,
            "llm-profile: lm_head_calls={} lm_head_cycles={}",
            p.lm_head_calls, p.lm_head_cycles
        );
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx512-acceleration")))]
    {
        let _ = writeln!(resp, "llm-profile: unavailable; AVX-512 disabled");
    }
    resp
}

fn write_udp_rate(resp: &mut shell::Response, label: &str, tokens: u64, cycles: u64) {
    if tokens == 0 || cycles == 0 {
        let _ = writeln!(
            resp,
            "{} unavailable tokens={} cycles={}",
            label, tokens, cycles
        );
        return;
    }
    let ns = crate::bench::cycles_to_ns(cycles);
    if ns == 0 {
        let _ = writeln!(resp, "{} unavailable cycles={}", label, cycles);
        return;
    }
    let tok_s_x10 = tokens.saturating_mul(10_000_000_000) / ns;
    let _ = writeln!(
        resp,
        "{} {}.{} tok/s cycles_per_token={}",
        label,
        tok_s_x10 / 10,
        tok_s_x10 % 10,
        cycles / tokens.max(1)
    );
}

fn udp_smp_status_response() -> shell::Response {
    let mut resp = shell::Response::empty();
    write_smp_status_response(&mut resp);
    resp
}

fn udp_smp_tune_response(args: &[u8]) -> shell::Response {
    let mut resp = shell::Response::empty();
    #[cfg(feature = "avx512-acceleration")]
    {
        let args = trim_datagram(args);
        let (head, tail) = split_datagram_word(args);
        match head {
            b"" | b"status" => {}
            b"reset" => {
                crate::smp::reset_matmul_tuning();
                let _ = crate::inference_avx512::reset_attn_parallel_min_tokens();
                let _ = writeln!(resp, "smp-tune: reset to default matmul policy");
            }
            b"production" | b"prod" | b"cherry" => {
                crate::smp::apply_cherry_production_matmul_tuning();
                let _ = crate::inference_avx512::reset_attn_parallel_min_tokens();
                let _ = writeln!(resp, "smp-tune: Cherry production policy restored");
            }
            b"rows-per-core" | b"rows" => {
                if let Some(rows) = parse_u32_datagram(trim_datagram(tail)) {
                    let rows = crate::smp::set_min_matmul_rows_per_core(rows as usize);
                    let _ = writeln!(resp, "smp-tune: rows_per_core set to {}", rows);
                } else {
                    let _ = writeln!(resp, "smp-tune: expected rows-per-core N");
                    return resp;
                }
            }
            b"max-cores" | b"max" => {
                if let Some(cores) = parse_u32_datagram(trim_datagram(tail)) {
                    let cores = crate::smp::set_max_matmul_cores(cores as usize);
                    let _ = writeln!(resp, "smp-tune: max_cores set to {}", cores);
                } else {
                    let _ = writeln!(resp, "smp-tune: expected max-cores N");
                    return resp;
                }
            }
            b"thread-policy" | b"policy" => {
                let policy = match trim_datagram(tail) {
                    b"all" | b"logical" => crate::smp::MATMUL_THREAD_POLICY_ALL,
                    b"unique-core" | b"physical" | b"physical-core" => {
                        crate::smp::MATMUL_THREAD_POLICY_UNIQUE_CORE
                    }
                    _ => {
                        let _ = writeln!(resp, "smp-tune: expected thread-policy all|unique-core");
                        return resp;
                    }
                };
                let _ = crate::smp::set_matmul_thread_policy(policy);
                let _ = writeln!(
                    resp,
                    "smp-tune: thread_policy set to {}",
                    crate::smp::matmul_thread_policy_label()
                );
            }
            b"bsp-discount" | b"discount" => {
                if let Some(pct) = parse_u32_datagram(trim_datagram(tail)) {
                    let pct = crate::smp::set_matmul_bsp_discount_pct(pct as usize);
                    let _ = writeln!(resp, "smp-tune: bsp_discount_pct set to {}", pct);
                } else {
                    let _ = writeln!(resp, "smp-tune: expected bsp-discount PCT (0-90)");
                    return resp;
                }
            }
            b"attn-par-tokens" | b"attn" => {
                let value = trim_datagram(tail);
                if let Some(tokens) = parse_u32_datagram(value) {
                    let tokens =
                        crate::inference_avx512::set_attn_parallel_min_tokens(tokens as usize);
                    let _ = writeln!(resp, "smp-tune: attn_par_min_tokens set to {}", tokens);
                } else if value == b"off" {
                    let tokens = crate::inference_avx512::set_attn_parallel_min_tokens(usize::MAX);
                    let _ = writeln!(
                        resp,
                        "smp-tune: attn_par_min_tokens set to off ({})",
                        tokens
                    );
                } else {
                    let _ = writeln!(resp, "smp-tune: expected attn-par-tokens N|off");
                    return resp;
                }
            }
            _ => {
                let _ = writeln!(
                    resp,
                    "smp-tune: expected status|production|rows-per-core N|max-cores N|thread-policy all|unique-core|bsp-discount PCT|attn-par-tokens N|off|reset"
                );
                return resp;
            }
        }
        let _ = writeln!(
            resp,
            "smp-tune: rows_per_core={} max_cores={} policy={} bsp_discount={} active={}",
            crate::smp::min_matmul_rows_per_core(),
            crate::smp::max_matmul_cores(),
            crate::smp::matmul_thread_policy_label(),
            crate::smp::matmul_bsp_discount_pct(),
            crate::smp::active_cores()
        );
        for &dim in &[512usize, 1024, 2048, 151_936] {
            let _ = writeln!(
                resp,
                "smp-tune: rows={} effective_cores={}",
                dim,
                crate::smp::effective_cores_for_rows(dim)
            );
        }
    }
    #[cfg(not(feature = "avx512-acceleration"))]
    {
        let _ = writeln!(resp, "smp-tune: unavailable in this build");
    }
    resp
}

fn udp_apic_response() -> shell::Response {
    let mut resp = shell::Response::empty();
    write_apic_response(&mut resp);
    resp
}

fn udp_madt_response() -> shell::Response {
    let mut resp = shell::Response::empty();
    write_madt_response(&mut resp);
    resp
}

fn udp_trampoline_response() -> shell::Response {
    let mut resp = shell::Response::empty();
    write_trampoline_response(&mut resp);
    resp
}

fn udp_smp_probe_response(stage: &[u8]) -> shell::Response {
    let mut resp = shell::Response::empty();
    #[cfg(feature = "avx512-acceleration")]
    {
        if !crate::smp::ap_boot_gate_active() {
            let _ = writeln!(resp, "smp: probe unavailable; AP gate is not armed");
            let _ = writeln!(
                resp,
                "smp: build with cherry-smp-debug to pause before INIT-SIPI-SIPI"
            );
            write_smp_status_response(&mut resp);
            return resp;
        }
        match parse_probe_mode(stage) {
            Some(mode) => {
                crate::smp::request_ap_probe(mode);
                let _ = writeln!(
                    resp,
                    "smp: probe requested stage={} ap_limit=1",
                    probe_mode_label(mode)
                );
                let _ = writeln!(resp, "smp: INIT-SIPI-SIPI starts after UDP response");
                write_smp_status_response(&mut resp);
            }
            None => {
                let _ = writeln!(
                    resp,
                    "smp: expected probe stage real|prot|pae|efer|paging|long|cr3|rust|entry|idt|apic|simd"
                );
            }
        }
    }
    #[cfg(not(feature = "avx512-acceleration"))]
    {
        let _ = writeln!(resp, "smp: unavailable in this build");
    }
    resp
}

fn udp_smp_start_response(limit: &[u8]) -> shell::Response {
    let mut resp = shell::Response::empty();
    #[cfg(feature = "avx512-acceleration")]
    {
        if !crate::smp::ap_boot_gate_active() {
            let _ = writeln!(
                resp,
                "smp: AP gate is not armed; AP bring-up has already passed"
            );
            let _ = writeln!(
                resp,
                "smp: build with cherry-smp-debug to pause before INIT-SIPI-SIPI"
            );
            write_smp_status_response(&mut resp);
            return resp;
        }
        let limit = trim_datagram(limit);
        if limit.is_empty() || limit == b"all" {
            crate::smp::request_ap_boot();
            let _ = writeln!(resp, "smp: full AP wake requested for all eligible APs");
            let _ = writeln!(resp, "smp: INIT-SIPI-SIPI starts after UDP response");
            write_smp_status_response(&mut resp);
        } else if let Some(n) = parse_u32_datagram(limit) {
            crate::smp::request_ap_boot_limited(n);
            let _ = writeln!(
                resp,
                "smp: full AP wake requested for up to {} AP(s)",
                n.max(1)
            );
            let _ = writeln!(resp, "smp: INIT-SIPI-SIPI starts after UDP response");
            write_smp_status_response(&mut resp);
        } else {
            let _ = writeln!(resp, "smp: expected start limit N|all");
        }
    }
    #[cfg(not(feature = "avx512-acceleration"))]
    {
        let _ = writeln!(resp, "smp: unavailable in this build");
    }
    resp
}

fn write_smp_status_response(resp: &mut shell::Response) {
    #[cfg(feature = "avx512-acceleration")]
    {
        let limit = crate::smp::ap_boot_ap_limit();
        let _ = writeln!(
            resp,
            "smp: active={} registered={} cap={} gate={} requested={} mode={} probe={}",
            crate::smp::active_cores(),
            crate::smp::registered_cores(),
            crate::smp::MAX_CORES,
            if crate::smp::ap_boot_gate_active() {
                "armed"
            } else {
                "open"
            },
            bool_label(crate::smp::ap_boot_requested()),
            probe_mode_label(crate::smp::ap_boot_mode()),
            probe_stage_label(crate::smp::ap_probe_stage())
        );
        #[cfg(target_arch = "x86_64")]
        {
            let st = crate::arch::x86_64::trampoline::status();
            let _ = writeln!(
                resp,
                "smp: raw_probe mode={} stage={}",
                raw_probe_label(st.raw_probe_mode),
                raw_probe_label(st.raw_probe_stage)
            );
        }
        if limit == u32::MAX {
            let _ = writeln!(resp, "smp: ap_limit=all");
        } else {
            let _ = writeln!(resp, "smp: ap_limit={}", limit);
        }
        let _ = writeln!(
            resp,
            "smp: rows_per_core={} max_cores={} policy={}",
            crate::smp::min_matmul_rows_per_core(),
            crate::smp::max_matmul_cores(),
            crate::smp::matmul_thread_policy_label()
        );
    }
    #[cfg(not(feature = "avx512-acceleration"))]
    {
        let _ = writeln!(resp, "smp: unavailable in this build");
    }
}

fn write_llm_status_response(resp: &mut shell::Response) {
    let status = crate::control_plane::status();
    let profile = crate::control_plane::llm_profile_snapshot();
    let _ = writeln!(
        resp,
        "llm: status={} gate={} requested={} profile={} generated={}",
        crate::control_plane::status_label(status),
        if crate::control_plane::boot_llm_gate_active() {
            "armed"
        } else {
            "open"
        },
        bool_label(crate::control_plane::boot_llm_start_requested()),
        if profile.valid { "ready" } else { "pending" },
        profile.generated_tokens
    );
    let model = crate::control_plane::model_region_snapshot();
    if model.present {
        let _ = writeln!(
            resp,
            "llm: source={} bytes={}MiB",
            crate::control_plane::model_source_label(model.source_id),
            model.len_bytes / (1024 * 1024)
        );
    } else {
        let _ = writeln!(resp, "llm: source=none");
    }
}

fn write_model_status_response(resp: &mut shell::Response) {
    let model = crate::control_plane::model_region_snapshot();
    if !model.present {
        let _ = writeln!(resp, "model: not recorded");
        return;
    }
    let _ = writeln!(
        resp,
        "model: source={} bytes={}MiB",
        crate::control_plane::model_source_label(model.source_id),
        model.len_bytes / (1024 * 1024)
    );
    if model.virt_addr == 0 || model.len_bytes > usize::MAX as u64 {
        let _ = writeln!(resp, "model: invalid recorded region");
        return;
    }
    let bytes = unsafe {
        core::slice::from_raw_parts(model.virt_addr as *const u8, model.len_bytes as usize)
    };
    match crate::model_loader::model_magic_from_bytes(bytes) {
        crate::model_loader::ModelMagic::Gguf => {
            let _ = writeln!(resp, "model: format=raw-gguf");
        }
        crate::model_loader::ModelMagic::Smodel => match crate::model_loader::smodel_info(bytes) {
            Ok(info) => {
                let _ = writeln!(
                    resp,
                    "model: format=.smodel kind={} payload={}MiB manifest={}B",
                    crate::model_loader::smodel_payload_kind_label(info.payload_kind),
                    info.payload_len / (1024 * 1024),
                    info.manifest_len
                );
                let aligned =
                    info.payload_offset % crate::model_loader::SMODEL_PAYLOAD_ALIGNMENT == 0;
                let _ = writeln!(resp, "model: payload_2m_aligned={}", bool_label(aligned));
                if info.payload_kind == crate::model_loader::SMODEL_PAYLOAD_KIND_NATIVE {
                    match crate::model_loader::smodel_native_summary(bytes) {
                        Ok(native) => {
                            let _ = writeln!(
                                resp,
                                "model: native tensors={} entry={} names={} data_base={:#x}",
                                native.tensor_count,
                                native.entry_size,
                                native.names_len,
                                native.data_base
                            );
                        }
                        Err(err) => {
                            let _ = writeln!(resp, "model: native invalid: {:?}", err);
                        }
                    }
                    match crate::model_loader::smodel_validation_anchor(bytes) {
                        Ok(Some(anchor)) => {
                            let _ = writeln!(
                                resp,
                                "model: anchors={} next={:?} logit_bits={:?}",
                                if anchor.strict { "strict" } else { "capture" },
                                anchor.expected_next_token,
                                anchor.expected_logit_bits
                            );
                        }
                        Ok(None) => {
                            let _ = writeln!(resp, "model: anchors=none");
                        }
                        Err(err) => {
                            let _ = writeln!(resp, "model: anchors invalid: {:?}", err);
                        }
                    }
                }
            }
            Err(err) => {
                let _ = writeln!(resp, "model: invalid .smodel: {:?}", err);
            }
        },
        crate::model_loader::ModelMagic::Unknown(magic) => {
            let _ = writeln!(resp, "model: unknown magic={:#x}", magic);
        }
    }
}

fn write_apic_response(resp: &mut shell::Response) {
    #[cfg(target_arch = "x86_64")]
    {
        match crate::arch::x86_64::apic::Apic::current() {
            Some(apic) => {
                let _ = writeln!(
                    resp,
                    "lapic: mode={} id={} version=0x{:08x} esr=0x{:08x}",
                    apic.mode_label(),
                    apic.id(),
                    apic.version(),
                    apic.esr()
                );
            }
            None => {
                let _ = writeln!(resp, "lapic: not initialized");
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = writeln!(resp, "lapic: unavailable");
    }
}

fn write_madt_response(resp: &mut shell::Response) {
    #[cfg(target_arch = "x86_64")]
    {
        let (count, lapic, ioapic, override_applied) =
            crate::arch::x86_64::acpi::recorded_summary();
        let _ = writeln!(
            resp,
            "madt: cpus={} lapic=0x{:x} ioapic=0x{:x} override={}",
            count,
            lapic,
            ioapic,
            bool_label(override_applied)
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = writeln!(resp, "madt: unavailable");
    }
}

fn write_trampoline_response(resp: &mut shell::Response) {
    #[cfg(target_arch = "x86_64")]
    {
        let st = crate::arch::x86_64::trampoline::status();
        let _ = writeln!(
            resp,
            "trampoline: installed={} phys=0x{:x} vector=0x{:02x} boot_cr3=0x{:x} cr3=0x{:x} cr4=0x{:x} cr0=0x{:x} raw_mode={} raw_stage={}",
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
        let _ = writeln!(resp, "trampoline: unavailable");
    }
}

#[cfg(feature = "avx512-acceleration")]
fn parse_probe_mode(stage: &[u8]) -> Option<u32> {
    match trim_datagram(stage) {
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

fn parse_u32_datagram(input: &[u8]) -> Option<u32> {
    if input.is_empty() {
        return None;
    }
    let mut value = 0u32;
    for &b in input {
        if !b.is_ascii_digit() {
            return None;
        }
        value = value.checked_mul(10)?;
        value = value.checked_add((b - b'0') as u32)?;
    }
    Some(value)
}

fn write_ip_response(resp: &mut shell::Response, ip: Ipv4) {
    let _ = write!(resp, "{}.{}.{}.{}", ip[0], ip[1], ip[2], ip[3]);
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

// ── IRQ-driven poll hook ────────────────────────────────────────────
//
// Used to drain RX frames from the timer ISR so the TCP/UDP shells
// stay reachable while the BSP is blocked inside a long-running
// synchronous call (notably the Stage-11 forward-pass, which on the
// single-core boot path otherwise monopolises the CPU for the entire
// inference window).
//
// Ownership model: while a non-null pointer is registered the timer
// ISR is the sole accessor of the stack. The owner transfers
// `&'static mut Stack` into the hook via `register_irq_poll`, and
// retrieves it back via `take_irq_poll` *with interrupts disabled*
// before any other code re-acquires mutable access. This sidesteps
// aliasing because the IRQ cannot fire during the take, and after
// the swap the pointer is null so subsequent ISRs no-op.

static IRQ_STACK_PTR: AtomicPtr<Stack> = AtomicPtr::new(core::ptr::null_mut());

/// Install `stack` as the timer-ISR poll target. After this call the
/// ISR drains RX and drives TCP retransmits on every tick.
pub fn register_irq_poll(stack: &'static mut Stack) {
    IRQ_STACK_PTR.store(stack as *mut Stack, Ordering::Release);
}

/// Retrieve the registered stack and clear the hook. The caller MUST
/// hold interrupts disabled across this call to guarantee the ISR
/// is not mid-poll when ownership transfers.
pub fn take_irq_poll() -> Option<&'static mut Stack> {
    let ptr = IRQ_STACK_PTR.swap(core::ptr::null_mut(), Ordering::AcqRel);
    if ptr.is_null() {
        None
    } else {
        // SAFETY: caller holds interrupts disabled; the only other
        // accessor is the timer ISR which now reads null and bails.
        Some(unsafe { &mut *ptr })
    }
}

/// Called from the timer ISR. No-op when no stack is registered.
#[cfg(target_arch = "x86_64")]
pub fn irq_poll_tick() {
    let ptr = IRQ_STACK_PTR.load(Ordering::Acquire);
    if !ptr.is_null() {
        // SAFETY: between `register_irq_poll` and `take_irq_poll` the
        // ISR is the unique accessor of the stack. `take_irq_poll`
        // runs with interrupts disabled, so an ISR cannot observe a
        // pointer that is being concurrently retracted.
        unsafe {
            (*ptr).poll();
        }
    }
}
