// SPDX-License-Identifier: AGPL-3.0-or-later
//! TCP shell server — line-buffered command surface on port 2222.
//!
//! Sits on top of [`super::tcp::Tcp`]. On every connection we emit a
//! greeting + prompt, then accept newline-terminated commands and
//! reply with output + prompt. Commands are dispatched through two
//! tiers:
//!   1. The rich [`super::cmds`] surface (multi-line `status`, `pci`,
//!      `mem`, `bench`, `inference`, `cores`, `reboot`, …) which can
//!      write up to ~3 KiB per command into a scratch buffer.
//!   2. The legacy UDP-shared [`super::shell::handle`] surface
//!      (`ping`, `mac`, `ip`, `echo`) for backwards compatibility
//!      with the existing UDP shell on port 9999.
//!
//! Unencrypted. This is a bring-up surface for a private datacenter
//! network — equivalent to a serial console over Ethernet. The product
//! direction is an Zero Control Plane, not a POSIX/SSH shell; any
//! future encrypted transport must terminate in Zero-native commands
//! and capability-gated intents.

use super::cmds::{self, CmdResult};
use super::eth::Mac;
use super::ipv4::Ipv4;
use super::shell;
use super::tcp::{Event, State, Tcp};

pub const PORT: u16 = 2222;

const GREETING: &[u8] = b"Zero v0.0.1 - type 'help' for commands, 'exit' to disconnect.\n";
const PROMPT: &[u8] = b"zero> ";

/// Per-connection line buffer. Sized to the largest UDP-shell reply
/// to keep parity with the existing command surface.
const LINE_BUFFER: usize = 1024;

pub struct TcpShell {
    pub tcp: Tcp,
    line: [u8; LINE_BUFFER],
    line_len: usize,
    /// True once we've written the initial greeting + prompt onto the
    /// just-established connection.
    greeted: bool,
}

impl TcpShell {
    pub const fn new() -> Self {
        Self {
            tcp: Tcp::new(PORT),
            line: [0; LINE_BUFFER],
            line_len: 0,
            greeted: false,
        }
    }

    pub fn state(&self) -> State {
        self.tcp.state()
    }

    /// Drive any pending app-level work — emit greeting on a fresh
    /// connection, drain RX into command lines, push responses. Called
    /// after every `on_segment` / `poll` so the shell stays responsive
    /// without owning its own task.
    pub fn drive(&mut self, our_mac: Mac, our_ip: Ipv4) {
        while let Some(evt) = self.tcp.take_event() {
            match evt {
                Event::Established => {
                    self.greeted = false;
                    self.line_len = 0;
                }
                Event::Closed => {
                    self.greeted = false;
                    self.line_len = 0;
                }
                Event::PeerClosed => {
                    // Drain any final line in the buffer, then begin
                    // active close. The TCP layer emits FIN once the
                    // TX queue drains.
                    self.flush_partial_line(our_mac, our_ip);
                    self.tcp.close();
                }
                Event::DataReady => {}
            }
        }

        if matches!(self.tcp.state(), State::Established) && !self.greeted {
            let _ = self.tcp.send(GREETING);
            let _ = self.tcp.send(PROMPT);
            self.greeted = true;
        }

        self.consume_lines(our_mac, our_ip);
    }

    fn consume_lines(&mut self, our_mac: Mac, our_ip: Ipv4) {
        loop {
            // Pull bytes one at a time until we either find a newline
            // or run out. Single-byte reads keep the loop simple; the
            // shell's traffic shape (interactive typing) makes this a
            // negligible cost.
            let mut byte = [0u8; 1];
            let n = self.tcp.recv(&mut byte);
            if n == 0 {
                break;
            }
            let b = byte[0];

            if b == b'\r' {
                // Swallow CR — common from line-buffered terminals
                // that emit CRLF.
                continue;
            }
            if b == b'\n' {
                self.dispatch_line(our_mac, our_ip);
                self.line_len = 0;
                continue;
            }
            // Drop control characters (incl. telnet IAC etc.). Keep
            // printable + tab.
            if b != b'\t' && b < 0x20 {
                continue;
            }
            if self.line_len < LINE_BUFFER {
                self.line[self.line_len] = b;
                self.line_len += 1;
            }
            // Lines exceeding LINE_BUFFER are silently truncated; the
            // next newline still dispatches whatever fit.
        }
    }

    fn dispatch_line(&mut self, our_mac: Mac, our_ip: Ipv4) {
        let line = &self.line[..self.line_len];
        let trimmed = trim(line);

        match cmds::dispatch(&mut self.tcp, trimmed, our_mac, our_ip) {
            CmdResult::Handled => {}
            CmdResult::Exit => {
                let _ = self.tcp.send(b"bye\n");
                self.tcp.close();
                return;
            }
            CmdResult::Reboot => {
                let _ = self.tcp.send(b"rebooting...\n");
                // We don't return — `trigger_reset` diverges.
                cmds::trigger_reset();
            }
            CmdResult::Unknown => {
                // Fall through to the legacy UDP-shared surface so
                // `ping`, `mac`, `ip`, `echo`, etc. keep working.
                let req = shell::Request {
                    data: trimmed,
                    src_ip: [0, 0, 0, 0],
                    src_mac: [0; 6],
                    src_port: 0,
                };
                let resp = shell::handle(&req, our_mac, our_ip);
                let _ = self.tcp.send(resp.bytes());
            }
        }

        if matches!(self.tcp.state(), State::Established) {
            let _ = self.tcp.send(PROMPT);
        }
    }

    fn flush_partial_line(&mut self, our_mac: Mac, our_ip: Ipv4) {
        if self.line_len > 0 {
            self.dispatch_line(our_mac, our_ip);
            self.line_len = 0;
        }
    }
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
