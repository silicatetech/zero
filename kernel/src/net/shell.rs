// SPDX-License-Identifier: AGPL-3.0-or-later
//! UDP shell server.
//!
//! Listens on port 9999. Each request datagram is interpreted as a
//! whitespace-trimmed command; the response is a single datagram back
//! to the sender. Supported commands:
//!   * `ping`      — replies with "pong".
//!   * `version`   — replies with "Zero v0.0.1".
//!   * `mac`       — replies with the NIC's MAC address.
//!   * `ip`        — replies with the configured static IPv4.
//!   * `diag`      — compact debug snapshot (handled by Stack).
//!   * `apic`/`madt`/`trampoline` — boot diagnostics (handled by Stack).
//!   * `smp-*`     — SMP status/probe/start controls (handled by Stack).
//!   * `llm-*`     — Zero control-plane LLM status/start/profile.
//!   * `mem`       — page-table/cache diagnostic snapshot (handled by Stack).
//!   * `tcp`       — replies with TCP shell state (handled by Stack).
//!   * `tcp-reset` — drops a wedged TCP shell slot (handled by Stack).
//!   * `echo …`    — replies with the remainder verbatim.
//!   * anything else — replies with "unknown: <cmd>".
//!
//! The 1024-byte response cap keeps every reply inside one Ethernet
//! frame (MTU 1500 with overhead well clear of the boundary).

use super::eth::Mac;
use super::ipv4::Ipv4;
use core::fmt;

pub const SHELL_PORT: u16 = 9999;
pub const RESPONSE_MAX: usize = 1024;

pub struct Request<'a> {
    pub data: &'a [u8],
    pub src_ip: Ipv4,
    pub src_mac: Mac,
    pub src_port: u16,
}

pub struct Response {
    buf: [u8; RESPONSE_MAX],
    len: usize,
}

impl Response {
    pub const fn empty() -> Self {
        Self {
            buf: [0; RESPONSE_MAX],
            len: 0,
        }
    }
    pub fn bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }

    pub fn push(&mut self, data: &[u8]) {
        push(self, data);
    }
}

impl fmt::Write for Response {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.push(s.as_bytes());
        Ok(())
    }
}

pub fn handle(req: &Request<'_>, our_mac: Mac, our_ip: Ipv4) -> Response {
    let mut resp = Response::empty();
    let cmd = trim(req.data);
    if cmd.is_empty() {
        push(&mut resp, b"(empty)\n");
        return resp;
    }

    if cmd == b"ping" {
        push(&mut resp, b"pong\n");
    } else if cmd == b"version" {
        push(&mut resp, b"Zero v0.0.1\n");
    } else if cmd == b"mac" {
        format_mac(&mut resp, our_mac);
        push(&mut resp, b"\n");
    } else if cmd == b"ip" {
        format_ip(&mut resp, our_ip);
        push(&mut resp, b"\n");
    } else if let Some(rest) = strip_prefix(cmd, b"echo ") {
        push(&mut resp, rest);
        push(&mut resp, b"\n");
    } else if cmd == b"echo" {
        push(&mut resp, b"\n");
    } else if cmd == b"help" {
        push(
            &mut resp,
            b"commands: ping version mac ip diag apic madt trampoline llm-status llm-start llm-profile [reset] smp-status smp-tune smp-probe <stage> smp-start <n|all> tcp tcp-reset reboot echo <text> help\n",
        );
        push(
            &mut resp,
            b"note: smp-start/smp-probe require a cherry-smp-debug build before AP auto-start\n",
        );
    } else {
        push(&mut resp, b"unknown: ");
        push(&mut resp, cmd);
        push(&mut resp, b"\n");
    }
    resp
}

fn trim(input: &[u8]) -> &[u8] {
    let mut start = 0;
    while start < input.len() && is_ws(input[start]) {
        start += 1;
    }
    let mut end = input.len();
    while end > start && is_ws(input[end - 1]) {
        end -= 1;
    }
    &input[start..end]
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n')
}

fn strip_prefix<'a>(input: &'a [u8], prefix: &[u8]) -> Option<&'a [u8]> {
    if input.len() >= prefix.len() && &input[..prefix.len()] == prefix {
        Some(&input[prefix.len()..])
    } else {
        None
    }
}

fn push(r: &mut Response, data: &[u8]) {
    let room = RESPONSE_MAX - r.len;
    let n = data.len().min(room);
    r.buf[r.len..r.len + n].copy_from_slice(&data[..n]);
    r.len += n;
}

fn format_mac(r: &mut Response, mac: Mac) {
    let hex = b"0123456789abcdef";
    let mut tmp = [0u8; 17];
    for i in 0..6 {
        tmp[i * 3] = hex[(mac[i] >> 4) as usize];
        tmp[i * 3 + 1] = hex[(mac[i] & 0x0F) as usize];
        if i < 5 {
            tmp[i * 3 + 2] = b':';
        }
    }
    push(r, &tmp);
}

fn format_ip(r: &mut Response, ip: Ipv4) {
    let mut tmp = [0u8; 16];
    let mut n = 0;
    for (i, octet) in ip.iter().enumerate() {
        n += write_u8(&mut tmp[n..], *octet);
        if i < 3 {
            tmp[n] = b'.';
            n += 1;
        }
    }
    push(r, &tmp[..n]);
}

fn write_u8(out: &mut [u8], v: u8) -> usize {
    if v >= 100 {
        out[0] = b'0' + v / 100;
        out[1] = b'0' + (v / 10) % 10;
        out[2] = b'0' + v % 10;
        3
    } else if v >= 10 {
        out[0] = b'0' + v / 10;
        out[1] = b'0' + v % 10;
        2
    } else {
        out[0] = b'0' + v;
        1
    }
}
