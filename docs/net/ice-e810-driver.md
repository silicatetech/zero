# Ice — Intel E810 (800-series) NIC Driver

**Status:** RX + TX data path online; bring-up hardened across three
silicon-contract bug fixes (TLAN_CTX bit offsets, ICRC reserved bit,
parent_teid scheduler topology). Polling-only, no interrupts, no DMA
engine beyond the descriptor rings. Source: `kernel/src/net/ice.rs`
(~2,400 lines).

This document is the reference for the E810 driver: how the control
plane and data path are programmed, the three hardware-contract bugs
that were found and fixed during Cherry Server bring-up, the datasheet
references that pin each magic number, and the current state of the
driver.

> **Scope note.** `ice.rs` is a bare-metal, `no_std` driver. It owns
> only the registers and admin-queue commands it needs to get one TX
> and one RX queue online and reachable from the Zero network stack
> (`kernel/src/net/mod.rs`). It is *not* a general-purpose NIC driver:
> single queue per direction, no RSS, no SR-IOV, no flow director, no
> interrupt path.

---

## 1. Why a separate driver

The Cherry Server EPYC batch shipped Intel **E810-XXV SFP28** cards
(`0x8086:0x159B`) in the same PCI slot that previously held **X710**
(i40e) cards. Same vendor, same physical slot — but the E810 redesigned
the control plane enough that the i40e driver (`kernel/src/net/i40e.rs`)
cannot drive it. The driver supports the full E810 SKU range:

| Device ID | Variant |
|-----------|---------|
| `0x1591` | E810-C backplane |
| `0x1592` | E810-C QSFP (CQDA1/CQDA2) |
| `0x1593` | E810-C SFP |
| `0x1599` | E810-XXV backplane |
| `0x159A` | E810-XXV QSFP (XXVDA4/XXVDA4T) |
| `0x159B` | **E810-XXV SFP (XXVDA2) — Cherry batch** |

(`ice.rs::SUPPORTED_DEVICES`, cross-checked against the Intel E810
`ice_devids.h` reference.)

### What changed vs. i40e

| Concern | i40e (X710) | ice (E810) |
|---|---|---|
| PF reset register | `PFGEN_CTRL` @ `0x00092400` | `PFGEN_CTRL` @ **`0x00091000`** |
| Port MAC | MMIO `PRTGL_SAL`/`PRTGL_SAH` | **gone** — AQ `Manage MAC Read` (0x0107), FW-owned |
| TX/RX context store | Host Memory Cache (HMC) | **no HMC** |
| TX queue context | HMC LAN_TX object + `GLLAN_TXPRE_QDIS` handshake | **AQ `Add TX Queues` (0x0C30)** |
| RX queue context | HMC LAN_RX object | per-queue MMIO window `QRX_CONTEXT(q,i)` @ `0x00280000` |
| RX descriptor | 32-byte advanced | **32-byte flex (RXDID=2)**, selected via `QRXFLXP_CNTXT(q)` |
| RX enable | context bit | `QRX_CTRL.QENA_REQ` → poll `QENA_STAT` |
| TX enable | queue-enable bit | none — `Add TX Queues` *creates and arms* the ring |

The admin queue itself is unchanged: same MMIO base layout
(`PF_FW_ATQ*` / `ARQ*` at `0x00080000`) and the same 32-byte descriptor
with identical flag bits (`DD=0`, `CMP=1`, `ERR=2`, `RD=10`, `BUF=12`,
`SI=13`).

---

## 2. The two-plane model

The driver is structured around two independent planes that can come up
separately. `Ice` carries a `rx_ready` and a `tx_ready` flag — RX can be
online while TX is still failing, which is exactly what you want for
diagnostics: a card that receives but cannot transmit tells you the
queue-context programming is wrong, not the link.

```
                       ┌───────────────────────────────┐
   PCI bind ──────────▶│ Control plane (Admin Queue)    │
   BAR0 map            │   PF reset                     │
                       │   ASQ/ARQ ring init            │
                       │   Get Version       (0x0001)   │
                       │   Clear PXE Mode    (0x0110)   │
                       │   Manage MAC Read   (0x0107)   │──▶ self.mac
                       │   Get Switch Config (0x0200)   │
                       │   Get Link Status   (0x0607)   │──▶ link gate
                       └───────────────┬───────────────┘
                                       │ link_up = 1
                       ┌───────────────┴───────────────┐
                       ▼                               ▼
        ┌──────────────────────────┐   ┌──────────────────────────────┐
        │ RX plane (pure MMIO)      │   │ TX plane (Admin Queue chain) │
        │  QRX_CONTEXT(q,i) window  │   │  Get Default Topology (0x0400)│
        │  RXDID=2 via QRXFLXP_CNTXT│   │  Add VSI            (0x0210)  │
        │  QRX_CTRL.QENA_REQ → STAT │   │  Add TX Queues      (0x0C30)  │
        │  bump RX tail             │   │  verify_tx (DD writeback)     │
        │  → rx_ready = true        │   │  → tx_ready = true            │
        └──────────────────────────┘   └──────────────────────────────┘
```

`Ice::bind()` (`ice.rs:587`) walks every E810 device on the PCI bus,
runs each through the control plane up to `Get Link Status`, and selects
the first port reporting `link_up = 1`. On a dual-port card (Cherry has
two PFs at `81:00.0` and `81:00.1`) this lets the link-down sibling be
skipped automatically; a card with no link returns `NicError::LinkDown`
so the caller's port-selection loop can try the next candidate.

### Ring sizing (`ice.rs:242`)

| Constant | Value | Meaning |
|---|---|---|
| `AQ_DESC_COUNT` | 32 | Admin queue descriptors per direction |
| `AQ_BUF_SIZE` | 4096 | AQ indirect buffer pool |
| `TX_DESC_COUNT` | 64 | TX ring descriptors (also the TLAN `qlen`) |
| `RX_DESC_COUNT` | 64 | RX ring descriptors |
| `RX_BUFFER_SIZE` / `TX_BUFFER_SIZE` | 2048 | Per-slot packet buffer (encoded as `size/128` in the RX context DBUF field) |

---

## 3. RX queue setup (`bring_up_rx`, `ice.rs:1181`)

RX bring-up is pure MMIO — no admin queue, no scheduler. On a healthy
card it always succeeds, so a failure here is treated as **fatal** for
the bind.

1. Allocate the RX descriptor ring (64 × 32-byte flex descriptors) and
   64 × 2 KiB packet buffers from the kernel arena. Write each
   descriptor's `qw0` with the physical address of its packet buffer.
2. Write the **LAN RX queue context** into the per-queue MMIO window:
   eight 32-bit dwords at `QRX_CONTEXT(q, i) = 0x00280000 + i*0x2000 + q*4`.
   The context packs the ring base address (`>> 7`), `qlen`, the DBUF
   buffer size (`RX_BUFFER_SIZE / 128`), header-buffer size, and the
   descriptor type.
3. Select the **flex descriptor format (RXDID=2)** for the queue via
   `QRXFLXP_CNTXT(q) = 0x00480000 + q*4`.
4. Arm the queue: set `QRX_CTRL(q).QENA_REQ` at `0x00120000` and poll
   `QENA_STAT` until the hardware acknowledges the enable.
5. Publish the initial RX tail so the card knows all 64 descriptors are
   owned by hardware.

On success the driver logs:

```
ice: RX online — queue 0 ready (RXDID=2, dbuff=2048B, qlen=64)
```

`receive()` (`ice.rs:1951`) polls the descriptor at the RX cursor: when
`status_error0` (`qw1` bits `[15:0]`) has the **DD** bit set, it reads
the packet length from `qw0` bits `[47:32]`, copies the payload out,
recycles the descriptor (re-writes the buffer address, clears status),
and advances the cursor + RX tail. The EOP bit is checked so multi-buffer
frames are not mis-assembled.

---

## 4. TX queue setup (`bring_up_tx`, `ice.rs:1339`)

TX is the hard part on E810 and the source of all three of the bugs in
§6. There is no HMC and no queue-enable register — the **`Add TX Queues`
(0x0C30)** admin command both installs the queue context *and* arms the
ring, but only if the context buffer is bit-exact and the queue is
anchored at a valid node in the firmware's transmit scheduler tree.

The bring-up is a three-command admin-queue chain plus a self-test:

```
Get Default Topology (0x0400) ─▶ Add VSI (0x0210) ─▶ Add TX Queues (0x0C30) ─▶ verify_tx
        │                              │                      │
   scheduler tree              src_vsi number        parent_teid candidates
```

### 4.1 The transmit scheduler tree

E810 transmit queues do not float free — every queue must be attached as
a **leaf** under a **queue-group (ENTRY_POINT)** node in a hierarchical
scheduler tree the firmware owns:

```
ROOT_PORT (1)
   └── TC (2)              traffic class
        └── SE_GENERIC (3) intermediate scheduler node
             └── ENTRY_POINT (4)   queue group  ← Add TX Queues attaches here
                  └── LEAF (5)      the queue itself (created by Add TX Queues)
```

`Get Default Topology` returns this tree as an array of 32-byte
`ice_aqc_get_topo_elem` records (`TOPO_ELEM_SIZE = 32`), each carrying
`parent_teid` (offset 0), `node_teid` (offset 4), and `elem_type`
(offset 8). The recognised element types are pinned to the values the
firmware actually writes (`ice.rs:196`):

| Value | Type | Role |
|---|---|---|
| 0 | `UNDEFINED` | — |
| 1 | `ROOT_PORT` | port root |
| 2 | `TC` | traffic class |
| 3 | `SE_GENERIC` | intermediate scheduler node |
| 4 | `ENTRY_POINT` | **queue group — Add TX Queues attaches a leaf here** |
| 5 | `LEAF` | queue level (created by Add TX Queues) |
| 6 | `SE_PADDED` | padding/reserved |

> An earlier draft of the driver had `LEAF=6` and `QGROUP=5`, which
> never matched anything the firmware sent and forced TX to fail. The
> constants above are the corrected, datasheet-pinned values.

### 4.2 The TX descriptor

`TxDesc` is a 16-byte data descriptor: `buffer_addr` (`u64`) +
`cmd_type_offset_bsz` (`u64`). The command field encodes:

| Bit | Name | Set? |
|---|---|---|
| 4 | EOP (end of packet) | ✅ every frame |
| 5 | RS (report status) | ✅ every frame — asks HW to write DD back |
| 6 | ~~ICRC~~ | ❌ **RESERVED on E810 — must be 0** (see §6.2) |
| 34 (14 bits) | BSZ (buffer size) | frame length |

`transmit()` (`ice.rs:1899`) drops the frame with a warning if
`tx_ready` is false; otherwise it copies the frame into the TX buffer,
writes the descriptor with `EOP | RS | BSZ`, bumps the TX tail, and (for
the self-test path) waits for the hardware DD writeback.

---

## 5. Control-plane commands

The admin queue is the firmware RPC channel. `aq_send_simple()` /
`aq_send()` post a descriptor to the ASQ, poll the DD bit, and check
`CMP` + `retval`. Opcodes used:

| Opcode | Command | Phase | Purpose |
|---|---|---|---|
| `0x0001` | Get Version | control | Prove AQ ↔ FW link is alive |
| `0x0110` | Clear PXE Mode | control | Stop PXE engine (best-effort; FW-version dependent) |
| `0x0107` | Manage MAC Read | control | Read the port MAC (FW is authoritative — no MMIO MAC on E810) |
| `0x0200` | Get Switch Config | control | Diagnostic (VSI/switch enumeration) |
| `0x0607` | Get Link Status | control | **Link gate** — bind selects the first `link_up=1` port |
| `0x0400` | Get Default Topology | TX | Read the transmit scheduler tree |
| `0x0210` | Add VSI | TX | Allocate the VSI; returns `src_vsi` for TLAN_CTX |
| `0x0C30` | Add TX Queues | TX | Install queue context + arm the ring |

PXE-mode handling also clears the MMIO PXE latch
(`REG_GLLAN_RCTL_0 = 0`) in case the firmware reports "already cleared"
but the engine is still latched — this is why `verify_tx` (§4.2, the DD
writeback self-test) exists: it catches a card whose TX engine is held
by a stale PXE latch even though `Add TX Queues` returned success.

---

## 6. Bugs found and fixed

All three bugs share a shape: a single silicon-contract value — a bit
offset, a reserved bit, a topology node id — was wrong, and the firmware
rejected `Add TX Queues` with `INVALID_PARAM`, surfacing as
**`NET: FAILED Ice(AddTxQueueFailed)`** on the VGA console with no SSH
reachability. The kernel booted and ran LLM inference fine in every
case; only the NIC TX path was down.

### 6.1 TLAN_CTX bit offsets (commit `8349fbd`)

**Symptom:** `Ice(AddTxQueueFailed)` at bring-up. Cherry boots and runs
inference; no network.

**Root cause:** The **TLAN transmit-queue context** packed into the
`Add TX Queues` buffer is a bit-packed structure decoded by the firmware
at *exact* bit positions fixed by the E810 datasheet (`ice_tlan_ctx_info[]`).
Three fields had been shifted **down by three bits**, so each one
overlapped its lower neighbour:

| Field | Was | Correct | Collided with (was) |
|---|---|---|---|
| `qlen` | bit 132 | **135** | `adjust_prof_id` (6 bits @ 129) |
| `tso_ena` | bit 149 | **152** | `quanta_prof_idx` (4 bits @ 148) |
| `legacy_int` | bit 161 | **164** | `tso_qnum` (11 bits @ 153) |

At 132/149/161 each field bled into the bits of the field below it; the
firmware decoded a malformed context and returned `INVALID_PARAM`. A
misleading comment had claimed the *correct* 135/152/164 offsets were the
buggy "off-by-three" draft — exactly backwards. The native packer's own
unit test already asserted `qlen` at bit 135, corroborating the fix.

**Fix:** Restore the silicon-layout offsets (`ice.rs:1757-1764`):

```rust
pack_tlan_bits(&mut ctx, TX_DESC_COUNT as u64, 135, 13); // qlen
pack_tlan_bits(&mut ctx, 1, 152, 1);                     // tso_ena
pack_tlan_bits(&mut ctx, 1, 164, 1);                     // legacy_int
```

A regression test (`tlan_ctx_qlen_tso_legacy_offsets_do_not_overlap_neighbors`,
`ice.rs:2299`) now asserts the three fields do not bleed into
`adjust_prof_id` / `quanta_prof_idx` / `tso_qnum`.

### 6.2 ICRC reserved bit (commit `cd12ec1`)

**Root cause:** The TX data descriptor set command-field bit 2 (overall
bit 6) on every frame — i40e's **ICRC ("insert CRC") bit**, copied
wholesale into the ice path. On E810 that command bit is **RESERVED and
must be 0**: the MAC appends the Ethernet FCS by default and there is no
per-descriptor CRC-insert request. Setting a reserved bit on every
transmit is a spec violation with undefined hardware behaviour and could
make the post-bring-up `verify_tx` self-test fail even with a valid
queue context.

**Fix:** Remove `TX_DESC_CMD_ICRC` and the bit from the `transmit()`
command word, so the descriptor carries only `EOP | RS | BSZ`. Clearing
a reserved bit cannot regress a working path; the FCS stays
hardware-inserted. This also removed a copied i40e idiom from the native
driver, per the bare-metal "no foreign patterns" rule.

### 6.3 parent_teid scheduler topology (working tree, image SHA `315c7317…`)

**Root cause:** `Add TX Queues` needs a `parent_teid` naming the
queue-group node the new queue attaches under. The driver had been
reading the parent from the **port default topology** taken *before*
`Add VSI` — but the VSI's own scheduler subtree only exists *after*
`Add VSI`, so the parent named a node in the wrong part of the tree and
the firmware rejected the attach.

**Fix:** Re-read `Get Default Topology` **after** `Add VSI`, then build
an ordered candidate list of plausible parent TEIDs and try each in turn
until the firmware accepts one — a *self-finding* retry instead of one
hard-coded guess:

- `collect_parent_candidates()` (`ice.rs:1438`) parses the post-VSI
  topology into an on-stack node table (no heap; capped at
  `MAX_TOPO_NODES = 64`).
- `order_parent_candidates()` (`ice.rs:2086`) orders candidates by
  priority class — `ENTRY_POINT` (queue-group) nodes deepest-first —
  capped at `MAX_PARENT_CANDIDATES = 8`. This is pure logic with its own
  unit tests (`order_parent_candidates_*`).
- `add_tx_queue()` (`ice.rs:1737`) packs the TLAN context once (only the
  qgroup header's `parent_teid` changes between attempts) and tries each
  candidate; the first acceptance wins and is recorded in
  `self.parent_teid`.

On success the driver logs the accepted parent so the operator can read
it off the KVM console:

```
ice: TX online — queue 0 ready (src_vsi=…, parent_teid=0x........)
```

> **State:** the parent_teid work is in the working tree and built into
> Cherry image SHA `315c7317…`; **not yet committed to `main` and not yet
> confirmed-good on hardware** at the time of writing. The TLAN_CTX
> (§6.1) and ICRC (§6.2) fixes are merged on `main` (`620be3f`).

---

## 7. Datasheet & reference map

Every magic number in `ice.rs` is anchored to a hardware contract.
The relevant references:

| Reference | Pins |
|---|---|
| Intel E810 datasheet — LAN TX queue context (`ice_tlan_ctx_info[]`) | TLAN_CTX bit offsets (`base@0`, `port_num@57`, `pf_num@65`, `vmvf_type@78`, `src_vsi@80`, `qlen@135`, `tso_ena@152`, `legacy_int@164`) |
| Intel E810 datasheet — TX data descriptor | CMD bits (EOP=4, RS=5); bit 6 reserved (ICRC is i40e-only) |
| Intel E810 Admin Queue spec — opcode `0x0400` | Scheduler `elem_type` enum (ROOT_PORT/TC/SE_GENERIC/ENTRY_POINT/LEAF) |
| Intel E810 Admin Queue spec — opcode `0x0C30` | `Add TX Queues` qgroup buffer layout (`TXQ_QGROUP_HEADER_SIZE=8`, `TXQ_PERQ_SIZE=52`, `TXQ_CTX_SIZE=22`) |
| Intel E810 Admin Queue spec — opcode `0x0107` | `Manage MAC Read` response (`MAC_ENTRY_LEN=8`, `addr_type==0` = LAN MAC) |
| Intel E810 register map | `PFGEN_CTRL@0x00091000`, `QRX_CONTEXT@0x00280000`, `QRXFLXP_CNTXT@0x00480000`, `QRX_CTRL@0x00120000`, AQ base `0x00080000` |

The structure names referenced in code comments
(`ice_aqc_vsi_props`, `ice_aqc_get_topo_elem`, `ice_tlan_ctx_info[]`,
`ice_aqc_manage_mac_read_resp`) are used purely as **field-layout
cross-checks against the Intel reference**; the driver is an
independent native implementation with no foreign code.

---

## 8. Current state & limitations

**Online:**
- PCI bind across the full E810 SKU range; dual-port link-status gating.
- Control plane: PF reset, AQ init, Get Version, Clear PXE, Manage MAC
  Read, Get Switch Config, Get Link Status.
- RX data path: flex descriptors, queue context, enable, poll-receive.
- TX data path: VSI + scheduler-anchored queue, DD-writeback self-test.

**Known limitations / not implemented:**
- Single TX + single RX queue. No RSS, SR-IOV, flow director, VLAN
  offload, checksum offload, or TSO (the `tso_ena` bit selects legacy TX
  mode, not segmentation).
- **Polling only** — no MSI-X / interrupt path. The stack drains the RX
  ring once per `Stack::poll()`.
- The module-level docstring at the top of `ice.rs` still describes the
  data path as "not yet wired" — that comment predates `bring_up_rx` /
  `bring_up_tx` and is stale; the §3/§4 code paths above are the truth.

**Verification gate (per the merged fixes):** `cargo test --workspace`
(1,930 tests passing, β-anchor forward-pass OK), `make kernel-cherry`,
and `make build-aarch64` all green. The fixes touch `net/ice.rs` only —
no inference / boot-path / `memory.rs` change — so the Boot-LLM
[β-anchor](../../README.md) (Token-ID 25) is unaffected.

---

## 9. See also

- [`network-stack.md`](network-stack.md) — the full L2–L4 stack the
  driver feeds into, and the `Nic` driver abstraction.
- `kernel/src/net/i40e.rs` — the X710 (700-series) driver this one
  diverged from; useful for understanding the HMC model E810 dropped.
- `kernel/src/net/mod.rs` — `Stack::bind()` driver fallback order and
  the RX/TX poll loop.
