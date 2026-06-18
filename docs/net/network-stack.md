# Zero Network Stack

**Status:** Operational. A polling, single-threaded, zero-allocation
L2–L4 stack that boots with the kernel and exposes a TCP and a UDP
diagnostic shell. Source: `kernel/src/net/` (~9,900 lines).

This document maps the whole network subsystem: the driver abstraction,
the protocol layers, how a packet flows in and out, and the diagnostic
shells. It is the companion to [`ice-e810-driver.md`](ice-e810-driver.md),
which goes deep on the E810 NIC driver specifically.

> **Bare-metal, `no_std`.** The stack runs in Ring-0 with no heap on the
> hot path: every buffer is fixed-size and stack- or struct-resident.
> There are no sockets in the POSIX sense, no async runtime, and no
> background threads — the stack is *pumped* by `Stack::poll()`, called
> from the kernel's cooperative executor.

---

## 1. Layout

```
kernel/src/net/
├── mod.rs        Stack: NIC binding, RX dispatch loop, TX helpers      (1721 loc)
├── eth.rs        Ethernet II framing + up to two 802.1Q/QinQ VLAN tags  (134 loc)
├── arp.rs        ARP request/reply + 8-entry round-robin cache          (178 loc)
├── ipv4.rs       IPv4 header parse/build + RFC 1071 checksum            (143 loc)
├── icmp.rs       ICMP echo (ping) reply only                            ( 53 loc)
├── udp.rs        UDP datagram parse/build + pseudo-header checksum       ( 60 loc)
├── tcp.rs        Single-slot TCP state machine (RFC 793-style)         (1165 loc)
├── tcp_shell.rs  Line-buffered command shell over TCP (port 2222)       (183 loc)
├── shell.rs      One-shot UDP command responder (port 9999)             (176 loc)
├── cmds.rs       Rich multi-line TCP shell command set                 (1878 loc)
├── e1000.rs      Intel 8254x driver (QEMU + server NICs)                (424 loc)
├── i40e.rs       Intel 700-series (X710) driver                        (1390 loc)
└── ice.rs        Intel 800-series (E810) driver — see dedicated doc    (2392 loc)
```

---

## 2. The driver abstraction (`Nic` enum)

There is no `dyn Trait` on the packet hot path. All three NIC drivers
are wrapped in a single enum and dispatched by `match` (`mod.rs:57`):

```rust
pub enum Nic {
    E1000(E1000),
    I40e(I40e),
    Ice(Ice),
}
```

Every variant implements the same three-method surface:

| Method | Signature | Role |
|---|---|---|
| `mac` | `fn mac(&self) -> Mac` | The card's L2 address |
| `transmit` | `fn transmit(&mut self, frame: &[u8])` | Push one fully-formed Ethernet frame |
| `receive` | `fn receive(&mut self, out: &mut [u8]) -> Option<usize>` | Pull one frame from the RX ring, if any |

`Stack::bind()` (`mod.rs:244`) scans the PCI bus and tries the drivers
in fallback order — **E1000 → I40e → Ice** — binding the first that comes
up. Once bound, it announces the static IPv4 via a gratuitous ARP and
probes the gateway.

### NIC driver matrix

| Driver | Hardware | Descriptor model | Control plane |
|---|---|---|---|
| `e1000` | Intel 8254x (82540/82545/82574L); QEMU `-device e1000` | Legacy 16-byte RX/TX descriptors | Direct MMIO |
| `i40e` | Intel 700-series (X710/XL710/XXV710/X722) | 32-byte advanced RX, 16-byte TX | Admin Queue + Host Memory Cache (HMC) |
| `ice` | Intel 800-series (E810/E810-XXV) | 32-byte **flex** RX (RXDID=2), 16-byte TX | Admin Queue, **no HMC** — see [dedicated doc](ice-e810-driver.md) |

All three are polling-only: no interrupts, no NAPI-style softirq, no DMA
beyond the descriptor rings.

### Static IP profiles (`mod.rs:105`)

Network configuration is compile-time, feature-gated:

| Profile | IP / mask | Gateway | Selected by |
|---|---|---|---|
| QEMU (default) | `10.0.2.15/24` | `10.0.2.2` | default build |
| Bare-metal static IP | `10.0.0.2/24` (placeholder) | `10.0.0.1` | `--features cherry-net` |

The bind outcome (`NOT_RUN` / `ONLINE` / `FAILED`) is published once via
an atomic register (`mod.rs::bind_report`) so the VGA/benchmark screen
can show `NET: ONLINE` or `NET: FAILED Ice(AddTxQueueFailed)` even with
no serial console.

---

## 3. The `Stack` and the poll loop

`Stack` (`mod.rs:202`) is the per-boot instance holding the NIC, the ARP
cache, IP config, fixed RX/TX scratch buffers, the TCP shell, and VLAN
tracking. Buffer sizing (`mod.rs:196`):

| Buffer | Size | Use |
|---|---|---|
| `FRAME_BUFFER_SIZE` | 2048 | RX scratch (one frame) |
| `SEND_BUFFER_SIZE` | 1600 | Shared TX scratch (UDP + ICMP) |
| `TCP_SEGMENT_MAX` | 1566 | `1600 − 14 − 20` TCP payload ceiling |

`Stack::poll()` (`mod.rs:350`) is the heartbeat. Once per call it:

1. Drains up to one full RX ring (`RX_DESC_COUNT = 64`) via
   `nic.receive()`.
2. Parses each frame and dispatches it (§4).
3. Drives TCP retransmit and queued-data flush (`drive_tcp`).

It is called from the cooperative executor on every tick, so the network
makes progress without interrupts or threads.

---

## 4. Protocol dispatch — the RX path

```
Stack::poll()
  └─ nic.receive() ─▶ handle_frame()
       └─ eth::parse()  ──▶ EthHeader { dst, src, ethertype, vlan }
            ├─ ETHERTYPE_ARP  (0x0806) ─▶ handle_arp
            │     └─ arp::parse → cache sender; reply if target_ip == ours
            └─ ETHERTYPE_IPV4 (0x0800) ─▶ handle_ipv4
                  └─ ipv4::parse → (opportunistically cache sender MAC)
                       ├─ PROTO_ICMP (1)  ─▶ icmp::parse → echo reply
                       ├─ PROTO_UDP  (17) ─▶ udp::parse  → shell / cmds dispatch
                       └─ PROTO_TCP  (6)  ─▶ tcp::on_segment → drive shell
```

Layer-by-layer:

- **`eth.rs`** — Ethernet II with up to two VLAN tags (dot1q `0x8100` /
  QinQ `0x88A8`). The ingress VLAN stack (`VlanStack`) is preserved and
  echoed on replies, so a tagged request gets a tagged answer.
- **`arp.rs`** — RFC 826. An 8-entry round-robin cache learns senders
  both from ARP traffic and opportunistically from incoming IPv4 frames.
  Replies preserve the ingress VLAN (`build_reply_tagged`).
- **`ipv4.rs`** — header parse (IHL ≥ 5, no options, fragments rejected),
  RFC 1071 Internet checksum, and the pseudo-header sum used by TCP/UDP.
  Outgoing headers use TTL 64, DF=0.
- **`icmp.rs`** — echo (ping) reply only; no error generation.
- **`udp.rs`** — datagram parse/build with pseudo-header checksum
  (zero-checksum replaced by `0xFFFF` per RFC 768).

---

## 5. TCP (`tcp.rs`)

A compact RFC 793-style responder for **one passive connection at a
time** (single-slot). No window scaling, SACK, timestamps, fast
recovery, or ECN. It is sized for a debug shell, not throughput.

### State machine

```
Closed → Listen → SynReceived → Established
                                    │
                  ┌─────────────────┼──────────────────┐
                  ▼ (peer FIN)      ▼ (we close)        │
              CloseWait          FinWait1               │
                  │                  │ (our FIN ACK'd)   │
              LastAck            FinWait2                │
                  │                  │ (peer FIN)        │
                  └──────────▶ TimeWait ◀───────────────┘
                                   │ (2 s drain)
                                Listen
```

Key design points:

- **Single slot with takeover.** A new SYN can reclaim an idle slot after
  a grace period (`TCP_TAKEOVER_IDLE_TICKS = 200` for the same peer,
  `1000` for a different peer).
- **In-order data only.** Out-of-order segments are dropped; stale
  retransmissions are re-ACK'd.
- **Timer-driven TX.** `poll()` (`tcp.rs:540`) runs every PIT tick
  (~10 ms): SYN-ACK retransmit, data retransmit (`RETRANSMIT_TICKS = 50`
  ≈ 500 ms), new-data send respecting the peer window (with a 1-byte
  zero-window probe), FIN emission, and state-timeout cleanup
  (TimeWait 2 s, half-open / hung-close 5 s).
- **MSS 1400.** Leaves room for Ethernet + IP + TCP inside a 1500 MTU.
- **App events.** `Established`, `DataReady`, `PeerClosed`, `Closed` are
  drained by the shell layer via `take_event()`.

ISS is derived from the tick counter
(`(now_tick as u32).wrapping_mul(0x9E37_79B9)`); checksums are validated
on RX and computed on TX over the IPv4 pseudo-header.

---

## 6. Diagnostic shells

The stack exposes two operator surfaces, both built on the protocol
layers above.

### UDP shell — port 9999 (`shell.rs`)

One datagram in, one datagram out. Replies fit in a single MTU (1024-byte
response buffer). Commands: `ping`→`pong`, `version`, `mac`, `ip`,
`echo <text>`, `help`. The UDP handler in `mod.rs::handle_udp` adds
diagnostics that fit in a datagram (`diag`, `apic`, `madt`, `smp-*`,
`llm-*`, `mem`, `model`, `tcp`, `tcp-reset`).

### TCP shell — port 2222 (`tcp_shell.rs` + `cmds.rs`)

A line-buffered, telnet-style shell with a larger (multi-KiB) response
budget, so it carries the rich commands that don't fit a UDP datagram.
`TcpShell::drive()` pumps the shell loop after each segment: greet on
`Established`, accumulate bytes into lines on `DataReady`, dispatch each
line. Dispatch tries `cmds::dispatch()` first, then falls back to the
legacy `shell::handle()` verbs.

Selected `cmds.rs` commands:

| Command | Shows |
|---|---|
| `status` / `diag` | cores, memory, NIC, uptime, inference summary |
| `pci` | full PCI device dump (bus:dev.fn, IDs, class, BAR0) |
| `net` | NIC link, MAC, IP, profile |
| `tcp` | TCP shell state + recovery timers |
| `mem` | arena used/capacity + page-table probes |
| `model` | loaded `.smodel` container + validation anchors |
| `llm status\|start\|profile\|model\|tune` | Boot-LLM state, throughput, SMP tuning |
| `smp status\|tune\|start\|probe` | AP bring-up, matmul row-per-core |
| `apic` / `madt` / `cores` / `trampoline` | CPU topology + AP wake state |
| `bench` | arena micro-benchmark |
| `version` / `reboot` / `exit` | misc |

---

## 7. The TX path

Egress always ends at `nic.transmit(frame)`, which copies a
fully-composed Ethernet frame into the driver's TX descriptor buffer and
hands it to hardware. Frames are composed bottom-up into the shared TX
scratch:

```
app / shell command
   └─ tcp.send(bytes)            enqueue into TCP tx_buf
        └─ Stack::drive_tcp()
             └─ tcp::poll()      emit a segment when a timer fires
                  └─ send_ip_payload()   wrap TCP in IPv4 + Ethernet
                       └─ nic.transmit()  → descriptor ring → wire
```

UDP/ICMP replies take the shorter path: compose payload → `udp::write` /
`icmp::build_echo_reply` → `ipv4::write_header` → `eth::write[_tagged]` →
`nic.transmit()`.

---

## 8. Design constraints & non-goals

- **No heap on the data path.** Every buffer is fixed-size; the ARP
  cache is 8 entries, TCP is one connection. This is deliberate — the
  stack must not be able to exhaust the kernel arena under network load.
- **Polling, not interrupts.** Simpler to reason about and to keep
  deterministic alongside the Boot-LLM; the cost is that throughput is
  bounded by poll frequency.
- **Diagnostics first.** The stack exists to make a bare-metal box
  *reachable and observable* (ping, shells, `status`/`diag`), not to be
  a high-throughput datapath. RSS, offloads, and multi-queue are
  explicit non-goals at this stage.
- **No POSIX sockets.** Consistent with the wider Zero
  non-goals — there is no socket API, only the typed protocol handlers
  and the two shells.

---

## 9. See also

- [`ice-e810-driver.md`](ice-e810-driver.md) — the E810 driver in depth,
  including the TLAN_CTX / ICRC / parent_teid bug fixes.
- `kernel/src/net/mod.rs` — `Stack::bind` / `Stack::poll`, the
  authoritative dispatch.
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) — where the network
  subsystem sits in the Unikernel.
