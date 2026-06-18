// SPDX-License-Identifier: AGPL-3.0-or-later
//! TCP — minimal RFC 793-style transport for a single passive
//! connection at a time.
//!
//! Scope: server-only, one listener, one connection slot. Sufficient
//! for a bare-metal "is the kernel alive, can I run a shell" remote-
//! console surface and nothing more.
//!
//! What's intentionally absent:
//!   * Multiple concurrent connections (single slot — extra peers get
//!     dropped silently rather than RST'd).
//!   * Window scaling, SACK, timestamps, ECN, fast retransmit/recovery
//!     (RFCs 7323 / 2018 / 3168 / 5681).
//!   * Path-MTU discovery — segments are capped at [`MSS`].
//!   * Out-of-order reassembly — segments past `rcv_nxt` are dropped;
//!     the peer will retransmit.
//!
//! What's present:
//!   * Three-way handshake (LISTEN → SYN_RCVD → ESTABLISHED).
//!   * In-order data transfer with cumulative ACKs.
//!   * Passive close (peer FIN → CLOSE_WAIT → LAST_ACK) and active
//!     close (we FIN → FIN_WAIT_1 → FIN_WAIT_2 → TIME_WAIT).
//!   * Coarse retransmit timer (fixed [`RETRANSMIT_TICKS`]).
//!   * Bounded TX queue ([`TX_BUFFER_LEN`]) with unacked-byte tracking.
//!
//! The state machine is driven exclusively by [`Tcp::on_segment`]
//! (inbound) and [`Tcp::poll`] (timer-driven retransmit + queued data).
//! Both helpers write any outbound segment into a caller-provided
//! buffer and return the `(dst_mac, dst_ip, len)` tuple needed to wrap
//! the segment in the existing IPv4 + Ethernet framing.

use super::eth::Mac;
use super::ipv4::{self, Ipv4, PROTO_TCP};

/// TCP fixed header length (no options).
pub const TCP_HEADER_LEN: usize = 20;

/// Receive-side buffer per connection. Caps the bytes we'll accept
/// before the app drains via [`Tcp::recv`].
pub const RX_BUFFER_LEN: usize = 4096;
/// Transmit-side buffer per connection. Caps the bytes the app can
/// hand to [`Tcp::send`] before earlier writes are ACK'd.
pub const TX_BUFFER_LEN: usize = 4096;

/// Maximum segment size we'll emit. Leaves headroom for the eth+ip+tcp
/// envelope inside a standard 1500-byte MTU.
pub const MSS: usize = 1400;

/// Coarse retransmit timeout in PIT ticks. PIT runs at 100 Hz (10 ms
/// per tick), so 50 ticks ≈ 500 ms.
pub const RETRANSMIT_TICKS: u64 = 50;

/// TIME_WAIT lingers for this many PIT ticks (200 = 2 s).
pub const TIME_WAIT_TICKS: u64 = 200;
/// Half-open handshakes must not pin the only debug-console slot.
/// PIT runs at 100 Hz, so this is 5 s on x86_64.
pub const SYN_RECEIVED_TIMEOUT_TICKS: u64 = 500;
/// A fresh SYN may take over a non-listening slot once the previous
/// peer has been quiet for this long. This keeps a wedged debug shell
/// from making port 2222 unusable while still protecting an actively
/// typing operator.
pub const TCP_TAKEOVER_IDLE_TICKS: u64 = 200;
/// A different source IP gets a longer grace period before it may
/// steal the single debug-console slot.
pub const TCP_FOREIGN_TAKEOVER_IDLE_TICKS: u64 = 1000;
/// Closing states are best-effort in this bring-up TCP stack. If the
/// peer disappears before ACKing our FIN, return to LISTEN quickly.
pub const TCP_CLOSE_TIMEOUT_TICKS: u64 = 500;

// ── Flag bits in the data-offset / flags word ───────────────────────
pub const FLAG_FIN: u8 = 0x01;
pub const FLAG_SYN: u8 = 0x02;
pub const FLAG_RST: u8 = 0x04;
pub const FLAG_PSH: u8 = 0x08;
pub const FLAG_ACK: u8 = 0x10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Closed,
    Listen,
    SynReceived,
    Established,
    /// Peer sent FIN, we ACK'd it — app may still write.
    CloseWait,
    /// We sent FIN after CloseWait, awaiting peer's ACK.
    LastAck,
    /// We sent FIN first, peer hasn't ACK'd yet.
    FinWait1,
    /// Peer ACK'd our FIN, we still expect their FIN.
    FinWait2,
    /// Both sides closed, draining stragglers.
    TimeWait,
}

#[derive(Debug, Clone, Copy)]
pub struct TcpHeader {
    pub src_port: u16,
    pub dst_port: u16,
    pub seq: u32,
    pub ack: u32,
    pub flags: u8,
    pub window: u16,
    pub header_len: usize,
}

/// Outbound segment description as returned by [`Tcp::on_segment`] /
/// [`Tcp::poll`]. The caller wraps `&buf[..len]` in IPv4 + Ethernet
/// addressed to `dst_mac` / `dst_ip` and transmits it via the NIC.
pub struct Outgoing {
    pub dst_mac: Mac,
    pub dst_ip: Ipv4,
    pub len: usize,
}

/// Application-visible connection event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    /// Three-way handshake completed; app may now write.
    Established,
    /// New bytes arrived in the receive buffer.
    DataReady,
    /// Peer signalled end-of-stream (FIN received).
    PeerClosed,
    /// Connection torn down (clean or RST). Slot is back in Listen.
    Closed,
}

/// Single-slot TCP listener / connection state.
pub struct Tcp {
    pub listen_port: u16,

    state: State,
    remote_mac: Mac,
    remote_ip: Ipv4,
    remote_port: u16,

    /// Initial-send-seq: first byte of our SYN.
    iss: u32,
    /// Initial-recv-seq: peer's SYN sequence.
    irs: u32,

    /// Oldest unacknowledged byte we sent.
    snd_una: u32,
    /// Next byte we'll send.
    snd_nxt: u32,
    /// Peer's advertised window.
    snd_wnd: u16,

    /// Next byte we expect from the peer.
    rcv_nxt: u32,

    /// Bytes live in `tx_buf[..tx_pending]`. The head `tx_buf[..tx_unacked]`
    /// has been transmitted; the tail `tx_buf[tx_unacked..tx_pending]` is
    /// queued but not yet sent. When the peer ACKs, we slide left.
    tx_pending: usize,
    tx_unacked: usize,

    tx_buf: [u8; TX_BUFFER_LEN],
    rx_buf: [u8; RX_BUFFER_LEN],
    /// Bytes available in `rx_buf` for the app to consume.
    rx_len: usize,

    /// Tick at which we last (re)transmitted. Used by [`poll`].
    last_tx_tick: u64,
    /// Tick of the last inbound segment accepted for the current slot.
    last_rx_tick: u64,
    /// Tick when the current TCP state was entered.
    state_entered_tick: u64,
    /// Tick when we entered TIME_WAIT (0 otherwise).
    timewait_entered: u64,

    /// Set once we've launched our FIN. Until ACK'd it occupies the
    /// sequence number `snd_nxt - 1`.
    fin_sent: bool,
    /// Peer's FIN has been observed and ACK'd.
    fin_rcvd: bool,
    /// App requested an active close — emit FIN once TX drains.
    want_close: bool,

    /// Drained by [`Tcp::take_event`].
    pending_event: Option<Event>,
}

impl Tcp {
    pub const fn new(listen_port: u16) -> Self {
        Self {
            listen_port,
            state: State::Listen,
            remote_mac: [0; 6],
            remote_ip: [0; 4],
            remote_port: 0,
            iss: 0,
            irs: 0,
            snd_una: 0,
            snd_nxt: 0,
            snd_wnd: 0,
            rcv_nxt: 0,
            tx_pending: 0,
            tx_unacked: 0,
            tx_buf: [0; TX_BUFFER_LEN],
            rx_buf: [0; RX_BUFFER_LEN],
            rx_len: 0,
            last_tx_tick: 0,
            last_rx_tick: 0,
            state_entered_tick: 0,
            timewait_entered: 0,
            fin_sent: false,
            fin_rcvd: false,
            want_close: false,
            pending_event: None,
        }
    }

    pub fn state(&self) -> State {
        self.state
    }

    pub fn remote(&self) -> (Ipv4, u16) {
        (self.remote_ip, self.remote_port)
    }

    pub fn idle_ticks(&self, now_tick: u64) -> u64 {
        elapsed_ticks(now_tick, self.last_rx_tick)
    }

    pub fn state_age_ticks(&self, now_tick: u64) -> u64 {
        elapsed_ticks(now_tick, self.state_entered_tick)
    }

    pub fn take_event(&mut self) -> Option<Event> {
        self.pending_event.take()
    }

    /// Drain up to `dst.len()` bytes from the receive buffer.
    /// Returns the number of bytes copied.
    pub fn recv(&mut self, dst: &mut [u8]) -> usize {
        let n = dst.len().min(self.rx_len);
        if n == 0 {
            return 0;
        }
        dst[..n].copy_from_slice(&self.rx_buf[..n]);
        if n < self.rx_len {
            self.rx_buf.copy_within(n..self.rx_len, 0);
        }
        self.rx_len -= n;
        n
    }

    /// Queue `data` for transmission. Returns the number of bytes
    /// accepted (limited by free space in the TX buffer).
    pub fn send(&mut self, data: &[u8]) -> usize {
        if !matches!(self.state, State::Established | State::CloseWait) {
            return 0;
        }
        let free = TX_BUFFER_LEN - self.tx_pending;
        let n = data.len().min(free);
        self.tx_buf[self.tx_pending..self.tx_pending + n].copy_from_slice(&data[..n]);
        self.tx_pending += n;
        n
    }

    /// Begin an active close — we emit a FIN once any queued data has
    /// drained. No-op outside ESTABLISHED / CLOSE_WAIT.
    pub fn close(&mut self) {
        if matches!(self.state, State::Established | State::CloseWait) {
            self.want_close = true;
        }
    }

    /// Operator-forced reset used by the UDP rescue shell. This drops
    /// the single debug-console slot without touching NIC/IP state.
    pub fn force_reset(&mut self) {
        self.reset_to_listen();
    }

    /// Force the connection back to LISTEN. Used after RST or after
    /// TIME_WAIT expires.
    fn reset_to_listen(&mut self) {
        let port = self.listen_port;
        *self = Tcp::new(port);
        self.pending_event = Some(Event::Closed);
    }

    fn enter_state(&mut self, state: State, now_tick: u64) {
        if self.state != state {
            self.state = state;
            self.state_entered_tick = now_tick;
        }
    }

    fn reclaimable_for_syn(&self, src_ip: Ipv4, now_tick: u64) -> bool {
        match self.state {
            State::Closed | State::Listen | State::TimeWait => true,
            State::SynReceived => self.state_age_ticks(now_tick) >= TCP_TAKEOVER_IDLE_TICKS,
            State::Established | State::CloseWait => {
                let idle = self.idle_ticks(now_tick);
                idle >= TCP_TAKEOVER_IDLE_TICKS
                    && (src_ip == self.remote_ip || idle >= TCP_FOREIGN_TAKEOVER_IDLE_TICKS)
            }
            State::FinWait1 | State::FinWait2 | State::LastAck => {
                self.state_age_ticks(now_tick) >= TCP_TAKEOVER_IDLE_TICKS
            }
        }
    }

    /// Process one inbound TCP segment. `our_ip` is needed for the
    /// pseudo-header checksum. The segment (if any) is written into
    /// `out`; the caller wraps it in IP+ETH.
    pub fn on_segment(
        &mut self,
        our_ip: Ipv4,
        src_ip: Ipv4,
        src_mac: Mac,
        segment: &[u8],
        now_tick: u64,
        out: &mut [u8],
    ) -> Option<Outgoing> {
        let (hdr, payload) = parse(segment)?;
        if hdr.dst_port != self.listen_port {
            return None;
        }

        // Verify checksum: pseudo + segment with on-the-wire checksum
        // field intact folds to 0 if valid.
        let pseudo = ipv4::pseudo_header_sum(src_ip, our_ip, PROTO_TCP, segment.len() as u16);
        if ipv4::fold_sum(pseudo, segment) != 0 {
            return None;
        }

        // RST on the active connection → return to LISTEN.
        if hdr.flags & FLAG_RST != 0 {
            if self.state != State::Listen
                && self.state != State::Closed
                && src_ip == self.remote_ip
                && hdr.src_port == self.remote_port
            {
                self.reset_to_listen();
            }
            return None;
        }

        match self.state {
            State::Closed => None,
            State::Listen => self.on_segment_listen(our_ip, src_ip, src_mac, &hdr, now_tick, out),
            _ => {
                // Only accept segments from the bound peer.
                if src_ip != self.remote_ip || hdr.src_port != self.remote_port {
                    if hdr.flags & FLAG_SYN != 0 && self.reclaimable_for_syn(src_ip, now_tick) {
                        self.reset_to_listen();
                        return self
                            .on_segment_listen(our_ip, src_ip, src_mac, &hdr, now_tick, out);
                    }
                    return None;
                }
                self.last_rx_tick = now_tick;
                self.on_segment_active(our_ip, &hdr, payload, now_tick, out)
            }
        }
    }

    fn on_segment_listen(
        &mut self,
        our_ip: Ipv4,
        src_ip: Ipv4,
        src_mac: Mac,
        hdr: &TcpHeader,
        now_tick: u64,
        out: &mut [u8],
    ) -> Option<Outgoing> {
        if hdr.flags & FLAG_SYN == 0 {
            return None;
        }
        self.remote_mac = src_mac;
        self.remote_ip = src_ip;
        self.remote_port = hdr.src_port;
        self.irs = hdr.seq;
        self.rcv_nxt = hdr.seq.wrapping_add(1); // SYN takes one seq
        self.last_rx_tick = now_tick;

        // Tick-derived ISS keeps back-to-back boots from colliding.
        self.iss = (now_tick as u32).wrapping_mul(0x9E37_79B9);
        self.snd_una = self.iss;
        self.snd_nxt = self.iss.wrapping_add(1); // SYN occupies one byte

        let n = write_segment(
            out,
            our_ip,
            src_ip,
            self.listen_port,
            self.remote_port,
            self.iss,
            self.rcv_nxt,
            FLAG_SYN | FLAG_ACK,
            recv_window(self.rx_len),
            &[],
        );

        self.enter_state(State::SynReceived, now_tick);
        self.last_tx_tick = now_tick;
        Some(Outgoing {
            dst_mac: src_mac,
            dst_ip: src_ip,
            len: n,
        })
    }

    fn on_segment_active(
        &mut self,
        our_ip: Ipv4,
        hdr: &TcpHeader,
        payload: &[u8],
        now_tick: u64,
        out: &mut [u8],
    ) -> Option<Outgoing> {
        // Update peer window from every segment with ACK set.
        if hdr.flags & FLAG_ACK != 0 {
            self.snd_wnd = hdr.window;
            // Cumulative ACK — only advance una if it's within
            // (snd_una, snd_nxt]. `seq_in_range` is already exclusive
            // on the low end and inclusive on the high end.
            if seq_in_range(hdr.ack, self.snd_una, self.snd_nxt) {
                let acked = hdr.ack.wrapping_sub(self.snd_una) as usize;
                // In SYN_RECEIVED / FIN_WAIT_1 / LAST_ACK the first or
                // last byte ACK'd is the SYN or FIN — a phantom byte
                // that isn't in `tx_buf`.
                let phantom = match self.state {
                    State::SynReceived => 1,
                    State::FinWait1 | State::LastAck => {
                        if self.fin_sent && hdr.ack == self.snd_nxt {
                            1
                        } else {
                            0
                        }
                    }
                    _ => 0,
                };
                let mut data_acked = acked.saturating_sub(phantom);
                if data_acked > self.tx_unacked {
                    data_acked = self.tx_unacked;
                }
                if data_acked > 0 {
                    self.tx_buf.copy_within(data_acked..self.tx_pending, 0);
                    self.tx_pending -= data_acked;
                    self.tx_unacked -= data_acked;
                }
                self.snd_una = hdr.ack;
            }
        }

        // State transitions on ACK.
        match self.state {
            State::SynReceived if hdr.flags & FLAG_ACK != 0 && self.snd_una == self.snd_nxt => {
                self.enter_state(State::Established, now_tick);
                self.pending_event = Some(Event::Established);
            }
            State::FinWait1 if self.fin_sent && self.snd_una == self.snd_nxt => {
                self.enter_state(State::FinWait2, now_tick);
            }
            State::LastAck if self.fin_sent && self.snd_una == self.snd_nxt => {
                self.reset_to_listen();
                return None;
            }
            _ => {}
        }

        // Accept in-order data only.
        let mut should_ack = false;
        if !payload.is_empty() {
            if hdr.seq == self.rcv_nxt {
                let free = RX_BUFFER_LEN - self.rx_len;
                let n = payload.len().min(free);
                if n > 0 {
                    self.rx_buf[self.rx_len..self.rx_len + n].copy_from_slice(&payload[..n]);
                    self.rx_len += n;
                    self.rcv_nxt = self.rcv_nxt.wrapping_add(n as u32);
                    self.pending_event = Some(Event::DataReady);
                }
                // Either way (accepted or dropped because the buffer
                // is full), ACK so the peer sees our advertised
                // window. Dropping silently here would stall flow
                // control until the next retransmit timer fires.
                should_ack = true;
            } else if seq_in_range(hdr.seq, self.rcv_nxt.wrapping_sub(0x4000), self.rcv_nxt) {
                // Stale retransmission — re-ACK so the peer notices.
                should_ack = true;
            }
            // Out-of-order future data is dropped silently.
        }

        // FIN consumes one sequence number, sitting after the payload.
        if hdr.flags & FLAG_FIN != 0 && hdr.seq.wrapping_add(payload.len() as u32) == self.rcv_nxt {
            self.rcv_nxt = self.rcv_nxt.wrapping_add(1);
            self.fin_rcvd = true;
            should_ack = true;
            self.pending_event = Some(Event::PeerClosed);
            match self.state {
                State::Established => self.enter_state(State::CloseWait, now_tick),
                State::FinWait1 => {
                    self.enter_state(State::TimeWait, now_tick);
                    self.timewait_entered = now_tick;
                }
                State::FinWait2 => {
                    self.enter_state(State::TimeWait, now_tick);
                    self.timewait_entered = now_tick;
                }
                _ => {}
            }
        }

        if should_ack {
            let n = write_segment(
                out,
                our_ip,
                self.remote_ip,
                self.listen_port,
                self.remote_port,
                self.snd_nxt,
                self.rcv_nxt,
                FLAG_ACK,
                recv_window(self.rx_len),
                &[],
            );
            Some(Outgoing {
                dst_mac: self.remote_mac,
                dst_ip: self.remote_ip,
                len: n,
            })
        } else {
            None
        }
    }

    /// Drive timer-driven work — retransmit, send queued data, emit
    /// FIN, etc. Called from the Stack poll loop every PIT tick.
    pub fn poll(&mut self, our_ip: Ipv4, now_tick: u64, out: &mut [u8]) -> Option<Outgoing> {
        if self.state == State::SynReceived
            && self.state_age_ticks(now_tick) >= SYN_RECEIVED_TIMEOUT_TICKS
        {
            self.reset_to_listen();
            return None;
        }

        if matches!(
            self.state,
            State::FinWait1 | State::FinWait2 | State::LastAck | State::CloseWait
        ) && self.state_age_ticks(now_tick) >= TCP_CLOSE_TIMEOUT_TICKS
        {
            self.reset_to_listen();
            return None;
        }

        // TIME_WAIT expiry.
        if self.state == State::TimeWait {
            if now_tick.wrapping_sub(self.timewait_entered) >= TIME_WAIT_TICKS {
                self.reset_to_listen();
            }
            return None;
        }

        // Retransmit SYN-ACK if no ACK has come in.
        if self.state == State::SynReceived
            && now_tick.wrapping_sub(self.last_tx_tick) >= RETRANSMIT_TICKS
        {
            let n = write_segment(
                out,
                our_ip,
                self.remote_ip,
                self.listen_port,
                self.remote_port,
                self.iss,
                self.rcv_nxt,
                FLAG_SYN | FLAG_ACK,
                recv_window(self.rx_len),
                &[],
            );
            self.last_tx_tick = now_tick;
            return Some(Outgoing {
                dst_mac: self.remote_mac,
                dst_ip: self.remote_ip,
                len: n,
            });
        }

        // Retransmit unacked data.
        let need_retransmit = self.tx_unacked > 0
            && now_tick.wrapping_sub(self.last_tx_tick) >= RETRANSMIT_TICKS
            && matches!(
                self.state,
                State::Established | State::CloseWait | State::FinWait1
            );
        if need_retransmit {
            let chunk = self.tx_unacked.min(MSS);
            let n = write_segment(
                out,
                our_ip,
                self.remote_ip,
                self.listen_port,
                self.remote_port,
                self.snd_una,
                self.rcv_nxt,
                FLAG_ACK | FLAG_PSH,
                recv_window(self.rx_len),
                &self.tx_buf[..chunk],
            );
            self.last_tx_tick = now_tick;
            return Some(Outgoing {
                dst_mac: self.remote_mac,
                dst_ip: self.remote_ip,
                len: n,
            });
        }

        // Send newly queued data, bounded by peer window.
        let new_bytes = self.tx_pending - self.tx_unacked;
        if new_bytes > 0 && matches!(self.state, State::Established | State::CloseWait) {
            // Always allow at least 1 byte to probe a zero window —
            // strict adherence to SWS avoidance is out of scope here.
            let window_avail = (self.snd_wnd as usize)
                .saturating_sub(self.tx_unacked)
                .max(1);
            let chunk = new_bytes.min(MSS).min(window_avail);
            let start = self.tx_unacked;
            let n = write_segment(
                out,
                our_ip,
                self.remote_ip,
                self.listen_port,
                self.remote_port,
                self.snd_nxt,
                self.rcv_nxt,
                FLAG_ACK | FLAG_PSH,
                recv_window(self.rx_len),
                &self.tx_buf[start..start + chunk],
            );
            self.snd_nxt = self.snd_nxt.wrapping_add(chunk as u32);
            self.tx_unacked += chunk;
            self.last_tx_tick = now_tick;
            return Some(Outgoing {
                dst_mac: self.remote_mac,
                dst_ip: self.remote_ip,
                len: n,
            });
        }

        // Emit FIN once requested and TX has drained.
        if self.want_close && !self.fin_sent && self.tx_pending == self.tx_unacked {
            let n = write_segment(
                out,
                our_ip,
                self.remote_ip,
                self.listen_port,
                self.remote_port,
                self.snd_nxt,
                self.rcv_nxt,
                FLAG_ACK | FLAG_FIN,
                recv_window(self.rx_len),
                &[],
            );
            self.snd_nxt = self.snd_nxt.wrapping_add(1);
            self.fin_sent = true;
            self.last_tx_tick = now_tick;
            match self.state {
                State::Established => self.enter_state(State::FinWait1, now_tick),
                State::CloseWait => self.enter_state(State::LastAck, now_tick),
                _ => {}
            }
            return Some(Outgoing {
                dst_mac: self.remote_mac,
                dst_ip: self.remote_ip,
                len: n,
            });
        }

        None
    }
}

// ── Wire format helpers ─────────────────────────────────────────────

pub fn parse(segment: &[u8]) -> Option<(TcpHeader, &[u8])> {
    if segment.len() < TCP_HEADER_LEN {
        return None;
    }
    let src_port = u16::from_be_bytes([segment[0], segment[1]]);
    let dst_port = u16::from_be_bytes([segment[2], segment[3]]);
    let seq = u32::from_be_bytes([segment[4], segment[5], segment[6], segment[7]]);
    let ack = u32::from_be_bytes([segment[8], segment[9], segment[10], segment[11]]);
    let data_offset_words = (segment[12] >> 4) as usize;
    if data_offset_words < 5 {
        return None;
    }
    let header_len = data_offset_words * 4;
    if segment.len() < header_len {
        return None;
    }
    let flags = segment[13];
    let window = u16::from_be_bytes([segment[14], segment[15]]);
    Some((
        TcpHeader {
            src_port,
            dst_port,
            seq,
            ack,
            flags,
            window,
            header_len,
        },
        &segment[header_len..],
    ))
}

/// Build a TCP segment (header + payload) into `out` and compute the
/// checksum across the IPv4 pseudo-header + segment. Returns total
/// length written.
#[allow(clippy::too_many_arguments)]
pub fn write_segment(
    out: &mut [u8],
    src_ip: Ipv4,
    dst_ip: Ipv4,
    src_port: u16,
    dst_port: u16,
    seq: u32,
    ack: u32,
    flags: u8,
    window: u16,
    payload: &[u8],
) -> usize {
    let total = TCP_HEADER_LEN + payload.len();
    out[0..2].copy_from_slice(&src_port.to_be_bytes());
    out[2..4].copy_from_slice(&dst_port.to_be_bytes());
    out[4..8].copy_from_slice(&seq.to_be_bytes());
    out[8..12].copy_from_slice(&ack.to_be_bytes());
    out[12] = 5 << 4; // data offset 5 (no options)
    out[13] = flags;
    out[14..16].copy_from_slice(&window.to_be_bytes());
    out[16..18].copy_from_slice(&0u16.to_be_bytes()); // checksum placeholder
    out[18..20].copy_from_slice(&0u16.to_be_bytes()); // urgent pointer
    out[TCP_HEADER_LEN..TCP_HEADER_LEN + payload.len()].copy_from_slice(payload);

    let pseudo = ipv4::pseudo_header_sum(src_ip, dst_ip, PROTO_TCP, total as u16);
    let csum = ipv4::fold_sum(pseudo, &out[..total]);
    out[16..18].copy_from_slice(&csum.to_be_bytes());
    total
}

/// Advertise a window equal to the unused part of the RX buffer,
/// clamped to u16.
fn recv_window(rx_len: usize) -> u16 {
    let free = RX_BUFFER_LEN.saturating_sub(rx_len);
    if free > u16::MAX as usize {
        u16::MAX
    } else {
        free as u16
    }
}

fn elapsed_ticks(now: u64, then: u64) -> u64 {
    now.wrapping_sub(then)
}

/// Modular-arithmetic "is `seq` in the half-open range (`lo`, `hi`]" —
/// i.e. lo < seq <= hi, wrapping mod 2^32. Used for cumulative-ACK
/// acceptance.
fn seq_in_range(seq: u32, lo: u32, hi: u32) -> bool {
    let s = seq.wrapping_sub(lo);
    let h = hi.wrapping_sub(lo);
    s != 0 && s <= h
}

// ── tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seq_in_range_basic() {
        assert!(seq_in_range(5, 1, 10));
        assert!(seq_in_range(10, 1, 10));
        assert!(!seq_in_range(1, 1, 10));
        assert!(!seq_in_range(11, 1, 10));
    }

    #[test]
    fn seq_in_range_wraparound() {
        let lo = u32::MAX - 5;
        let hi = 5u32;
        assert!(seq_in_range(u32::MAX, lo, hi));
        assert!(seq_in_range(0, lo, hi));
        assert!(seq_in_range(5, lo, hi));
        assert!(!seq_in_range(6, lo, hi));
        assert!(!seq_in_range(lo, lo, hi));
    }

    #[test]
    fn write_segment_round_trip() {
        let mut buf = [0u8; 64];
        let n = write_segment(
            &mut buf,
            [10, 0, 0, 1],
            [10, 0, 0, 2],
            12345,
            22,
            1_000_000,
            2_000_000,
            FLAG_SYN | FLAG_ACK,
            4096,
            b"hi!",
        );
        assert_eq!(n, TCP_HEADER_LEN + 3);
        let (hdr, payload) = parse(&buf[..n]).unwrap();
        assert_eq!(hdr.src_port, 12345);
        assert_eq!(hdr.dst_port, 22);
        assert_eq!(hdr.seq, 1_000_000);
        assert_eq!(hdr.ack, 2_000_000);
        assert_eq!(hdr.flags, FLAG_SYN | FLAG_ACK);
        assert_eq!(hdr.window, 4096);
        assert_eq!(payload, b"hi!");

        let pseudo = ipv4::pseudo_header_sum([10, 0, 0, 1], [10, 0, 0, 2], PROTO_TCP, n as u16);
        assert_eq!(ipv4::fold_sum(pseudo, &buf[..n]), 0);
    }

    #[test]
    fn handshake_listen_to_established() {
        let mut tcp = Tcp::new(2222);
        assert_eq!(tcp.state(), State::Listen);
        let mut out = [0u8; 64];

        // Build a SYN from a fake peer at 10.0.0.42 / port 33000.
        let mut syn = [0u8; TCP_HEADER_LEN];
        write_segment(
            &mut syn,
            [10, 0, 0, 42],
            [10, 0, 0, 1],
            33000,
            2222,
            42_000,
            0,
            FLAG_SYN,
            8192,
            &[],
        );

        let reply = tcp.on_segment([10, 0, 0, 1], [10, 0, 0, 42], [0xaa; 6], &syn, 10, &mut out);
        let r = reply.expect("SYN-ACK expected");
        assert_eq!(r.dst_ip, [10, 0, 0, 42]);
        let (hdr, _) = parse(&out[..r.len]).unwrap();
        assert_eq!(hdr.flags & (FLAG_SYN | FLAG_ACK), FLAG_SYN | FLAG_ACK);
        assert_eq!(hdr.ack, 42_001);
        assert_eq!(tcp.state(), State::SynReceived);
        let server_iss = hdr.seq;

        // Peer ACK — handshake completes.
        let mut ack = [0u8; TCP_HEADER_LEN];
        write_segment(
            &mut ack,
            [10, 0, 0, 42],
            [10, 0, 0, 1],
            33000,
            2222,
            42_001,
            server_iss.wrapping_add(1),
            FLAG_ACK,
            8192,
            &[],
        );
        let reply = tcp.on_segment([10, 0, 0, 1], [10, 0, 0, 42], [0xaa; 6], &ack, 11, &mut out);
        assert!(reply.is_none(), "empty ACK should not generate a reply");
        assert_eq!(tcp.state(), State::Established);
        assert_eq!(tcp.take_event(), Some(Event::Established));
    }

    #[test]
    fn syn_received_times_out_back_to_listen() {
        let mut tcp = Tcp::new(2222);
        let mut out = [0u8; 64];

        let mut syn = [0u8; TCP_HEADER_LEN];
        write_segment(
            &mut syn,
            [10, 0, 0, 42],
            [10, 0, 0, 1],
            33000,
            2222,
            42_000,
            0,
            FLAG_SYN,
            8192,
            &[],
        );

        tcp.on_segment([10, 0, 0, 1], [10, 0, 0, 42], [0xaa; 6], &syn, 10, &mut out)
            .expect("SYN-ACK expected");
        assert_eq!(tcp.state(), State::SynReceived);

        let reply = tcp.poll([10, 0, 0, 1], 10 + SYN_RECEIVED_TIMEOUT_TICKS, &mut out);
        assert!(reply.is_none());
        assert_eq!(tcp.state(), State::Listen);
        assert_eq!(tcp.take_event(), Some(Event::Closed));
    }

    #[test]
    fn idle_established_slot_can_be_reclaimed_by_new_syn() {
        let mut tcp = Tcp::new(2222);
        let mut out = [0u8; 256];

        let mut syn = [0u8; TCP_HEADER_LEN];
        write_segment(
            &mut syn,
            [10, 0, 0, 42],
            [10, 0, 0, 1],
            33000,
            2222,
            100,
            0,
            FLAG_SYN,
            8192,
            &[],
        );
        let synack = tcp
            .on_segment([10, 0, 0, 1], [10, 0, 0, 42], [0xaa; 6], &syn, 10, &mut out)
            .unwrap();
        let server_iss = parse(&out[..synack.len]).unwrap().0.seq;

        let mut ack = [0u8; TCP_HEADER_LEN];
        write_segment(
            &mut ack,
            [10, 0, 0, 42],
            [10, 0, 0, 1],
            33000,
            2222,
            101,
            server_iss.wrapping_add(1),
            FLAG_ACK,
            8192,
            &[],
        );
        tcp.on_segment([10, 0, 0, 1], [10, 0, 0, 42], [0xaa; 6], &ack, 11, &mut out);
        assert_eq!(tcp.state(), State::Established);
        assert_eq!(tcp.take_event(), Some(Event::Established));

        let mut new_syn = [0u8; TCP_HEADER_LEN];
        write_segment(
            &mut new_syn,
            [10, 0, 0, 42],
            [10, 0, 0, 1],
            33001,
            2222,
            200,
            0,
            FLAG_SYN,
            8192,
            &[],
        );
        let reply = tcp.on_segment(
            [10, 0, 0, 1],
            [10, 0, 0, 42],
            [0xbb; 6],
            &new_syn,
            11 + TCP_TAKEOVER_IDLE_TICKS,
            &mut out,
        );
        assert!(reply.is_some(), "new SYN should reclaim idle slot");
        assert_eq!(tcp.state(), State::SynReceived);
        assert_eq!(tcp.remote(), ([10, 0, 0, 42], 33001));
    }

    #[test]
    fn active_slot_rejects_takeover_inside_grace_window() {
        let mut tcp = Tcp::new(2222);
        let mut out = [0u8; 256];

        let mut syn = [0u8; TCP_HEADER_LEN];
        write_segment(
            &mut syn,
            [10, 0, 0, 42],
            [10, 0, 0, 1],
            33000,
            2222,
            100,
            0,
            FLAG_SYN,
            8192,
            &[],
        );
        let synack = tcp
            .on_segment([10, 0, 0, 1], [10, 0, 0, 42], [0xaa; 6], &syn, 10, &mut out)
            .unwrap();
        let server_iss = parse(&out[..synack.len]).unwrap().0.seq;

        let mut ack = [0u8; TCP_HEADER_LEN];
        write_segment(
            &mut ack,
            [10, 0, 0, 42],
            [10, 0, 0, 1],
            33000,
            2222,
            101,
            server_iss.wrapping_add(1),
            FLAG_ACK,
            8192,
            &[],
        );
        tcp.on_segment([10, 0, 0, 1], [10, 0, 0, 42], [0xaa; 6], &ack, 11, &mut out);
        let _ = tcp.take_event();

        let mut new_syn = [0u8; TCP_HEADER_LEN];
        write_segment(
            &mut new_syn,
            [10, 0, 0, 42],
            [10, 0, 0, 1],
            33001,
            2222,
            200,
            0,
            FLAG_SYN,
            8192,
            &[],
        );
        let reply = tcp.on_segment(
            [10, 0, 0, 1],
            [10, 0, 0, 42],
            [0xbb; 6],
            &new_syn,
            11 + TCP_TAKEOVER_IDLE_TICKS - 1,
            &mut out,
        );
        assert!(reply.is_none());
        assert_eq!(tcp.state(), State::Established);
        assert_eq!(tcp.remote(), ([10, 0, 0, 42], 33000));
    }

    #[test]
    fn data_then_passive_close() {
        let mut tcp = Tcp::new(2222);
        let mut out = [0u8; 256];

        // Skip the handshake by injecting state directly via SYN+ACK
        // path — same as the test above.
        let mut syn = [0u8; TCP_HEADER_LEN];
        write_segment(
            &mut syn,
            [10, 0, 0, 42],
            [10, 0, 0, 1],
            33000,
            2222,
            100,
            0,
            FLAG_SYN,
            8192,
            &[],
        );
        let synack = tcp
            .on_segment([10, 0, 0, 1], [10, 0, 0, 42], [0xaa; 6], &syn, 10, &mut out)
            .unwrap();
        let server_iss = parse(&out[..synack.len]).unwrap().0.seq;

        let mut ack = [0u8; TCP_HEADER_LEN];
        write_segment(
            &mut ack,
            [10, 0, 0, 42],
            [10, 0, 0, 1],
            33000,
            2222,
            101,
            server_iss.wrapping_add(1),
            FLAG_ACK,
            8192,
            &[],
        );
        tcp.on_segment([10, 0, 0, 1], [10, 0, 0, 42], [0xaa; 6], &ack, 11, &mut out);
        assert_eq!(tcp.state(), State::Established);
        let _ = tcp.take_event();

        // Peer sends "hi\n".
        let mut data = [0u8; TCP_HEADER_LEN + 3];
        write_segment(
            &mut data,
            [10, 0, 0, 42],
            [10, 0, 0, 1],
            33000,
            2222,
            101,
            server_iss.wrapping_add(1),
            FLAG_ACK | FLAG_PSH,
            8192,
            b"hi\n",
        );
        let reply = tcp
            .on_segment(
                [10, 0, 0, 1],
                [10, 0, 0, 42],
                [0xaa; 6],
                &data,
                12,
                &mut out,
            )
            .expect("ACK for data");
        let (rhdr, _) = parse(&out[..reply.len]).unwrap();
        assert_eq!(rhdr.flags & FLAG_ACK, FLAG_ACK);
        assert_eq!(rhdr.ack, 104);
        assert_eq!(tcp.take_event(), Some(Event::DataReady));

        let mut sink = [0u8; 16];
        let n = tcp.recv(&mut sink);
        assert_eq!(&sink[..n], b"hi\n");

        // Peer FIN.
        let mut fin = [0u8; TCP_HEADER_LEN];
        write_segment(
            &mut fin,
            [10, 0, 0, 42],
            [10, 0, 0, 1],
            33000,
            2222,
            104,
            server_iss.wrapping_add(1),
            FLAG_ACK | FLAG_FIN,
            8192,
            &[],
        );
        tcp.on_segment([10, 0, 0, 1], [10, 0, 0, 42], [0xaa; 6], &fin, 13, &mut out);
        assert_eq!(tcp.state(), State::CloseWait);
        assert_eq!(tcp.take_event(), Some(Event::PeerClosed));

        // App closes.
        tcp.close();
        let fin_out = tcp
            .poll([10, 0, 0, 1], 14, &mut out)
            .expect("FIN should be emitted");
        let (fhdr, _) = parse(&out[..fin_out.len]).unwrap();
        assert_eq!(fhdr.flags & FLAG_FIN, FLAG_FIN);
        assert_eq!(tcp.state(), State::LastAck);

        // Peer ACKs our FIN.
        let mut last = [0u8; TCP_HEADER_LEN];
        write_segment(
            &mut last,
            [10, 0, 0, 42],
            [10, 0, 0, 1],
            33000,
            2222,
            105,
            server_iss.wrapping_add(2),
            FLAG_ACK,
            8192,
            &[],
        );
        tcp.on_segment(
            [10, 0, 0, 1],
            [10, 0, 0, 42],
            [0xaa; 6],
            &last,
            15,
            &mut out,
        );
        assert_eq!(tcp.state(), State::Listen);
        assert_eq!(tcp.take_event(), Some(Event::Closed));
    }
}
