// SPDX-License-Identifier: AGPL-3.0-or-later
//! ARP — request/reply + a tiny fixed-size cache.
//!
//! Scope: IPv4-over-Ethernet only. RFC 826 reply path is symmetric: a
//! request for our IP triggers a reply; an unsolicited reply is just
//! cached.

use super::eth::{self, Mac, ETHERTYPE_ARP};

pub const ARP_PACKET_LEN: usize = 28;
const HTYPE_ETHERNET: u16 = 1;
const PTYPE_IPV4: u16 = 0x0800;
const HLEN_MAC: u8 = 6;
const PLEN_IPV4: u8 = 4;
const OP_REQUEST: u16 = 1;
const OP_REPLY: u16 = 2;

const CACHE_SLOTS: usize = 8;

pub type Ipv4 = [u8; 4];

#[derive(Clone, Copy)]
struct Entry {
    ip: Ipv4,
    mac: Mac,
    in_use: bool,
}

const EMPTY: Entry = Entry {
    ip: [0; 4],
    mac: [0; 6],
    in_use: false,
};

pub struct ArpCache {
    entries: [Entry; CACHE_SLOTS],
    next_slot: usize,
}

impl ArpCache {
    pub const fn new() -> Self {
        Self {
            entries: [EMPTY; CACHE_SLOTS],
            next_slot: 0,
        }
    }

    pub fn lookup(&self, ip: &Ipv4) -> Option<Mac> {
        for e in self.entries.iter() {
            if e.in_use && &e.ip == ip {
                return Some(e.mac);
            }
        }
        None
    }

    pub fn insert(&mut self, ip: Ipv4, mac: Mac) {
        for e in self.entries.iter_mut() {
            if e.in_use && e.ip == ip {
                e.mac = mac;
                return;
            }
        }
        // Round-robin replacement keeps things simple — nothing in the
        // shell server flows requires LRU.
        let slot = self.next_slot;
        self.entries[slot] = Entry {
            ip,
            mac,
            in_use: true,
        };
        self.next_slot = (slot + 1) % CACHE_SLOTS;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ArpPacket {
    pub op: u16,
    pub sender_mac: Mac,
    pub sender_ip: Ipv4,
    pub target_mac: Mac,
    pub target_ip: Ipv4,
}

pub fn parse(payload: &[u8]) -> Option<ArpPacket> {
    if payload.len() < ARP_PACKET_LEN {
        return None;
    }
    let htype = u16::from_be_bytes([payload[0], payload[1]]);
    let ptype = u16::from_be_bytes([payload[2], payload[3]]);
    if htype != HTYPE_ETHERNET || ptype != PTYPE_IPV4 {
        return None;
    }
    if payload[4] != HLEN_MAC || payload[5] != PLEN_IPV4 {
        return None;
    }
    let op = u16::from_be_bytes([payload[6], payload[7]]);
    let mut sender_mac = [0u8; 6];
    let mut sender_ip = [0u8; 4];
    let mut target_mac = [0u8; 6];
    let mut target_ip = [0u8; 4];
    sender_mac.copy_from_slice(&payload[8..14]);
    sender_ip.copy_from_slice(&payload[14..18]);
    target_mac.copy_from_slice(&payload[18..24]);
    target_ip.copy_from_slice(&payload[24..28]);
    Some(ArpPacket {
        op,
        sender_mac,
        sender_ip,
        target_mac,
        target_ip,
    })
}

/// Build an ARP reply frame for `request` and write it into `out`.
/// Returns the number of bytes written.
pub fn build_reply(out: &mut [u8], our_mac: Mac, request: &ArpPacket) -> usize {
    build_reply_tagged(out, our_mac, request, eth::VlanStack::empty())
}

/// Build an ARP reply, preserving an ingress VLAN tag when present.
pub fn build_reply_tagged(
    out: &mut [u8],
    our_mac: Mac,
    request: &ArpPacket,
    vlan: eth::VlanStack,
) -> usize {
    let mut n = eth::write_tagged(out, request.sender_mac, our_mac, vlan, ETHERTYPE_ARP);
    n += write_packet(
        &mut out[n..],
        OP_REPLY,
        our_mac,
        request.target_ip,
        request.sender_mac,
        request.sender_ip,
    );
    n
}

/// Build an ARP request asking "who has `target_ip`".
pub fn build_request(out: &mut [u8], our_mac: Mac, our_ip: Ipv4, target_ip: Ipv4) -> usize {
    let mut n = eth::write(out, eth::BROADCAST_MAC, our_mac, ETHERTYPE_ARP);
    n += write_packet(
        &mut out[n..],
        OP_REQUEST,
        our_mac,
        our_ip,
        [0; 6],
        target_ip,
    );
    n
}

fn write_packet(
    out: &mut [u8],
    op: u16,
    sender_mac: Mac,
    sender_ip: Ipv4,
    target_mac: Mac,
    target_ip: Ipv4,
) -> usize {
    out[0..2].copy_from_slice(&HTYPE_ETHERNET.to_be_bytes());
    out[2..4].copy_from_slice(&PTYPE_IPV4.to_be_bytes());
    out[4] = HLEN_MAC;
    out[5] = PLEN_IPV4;
    out[6..8].copy_from_slice(&op.to_be_bytes());
    out[8..14].copy_from_slice(&sender_mac);
    out[14..18].copy_from_slice(&sender_ip);
    out[18..24].copy_from_slice(&target_mac);
    out[24..28].copy_from_slice(&target_ip);
    ARP_PACKET_LEN
}

pub fn op_is_request(op: u16) -> bool {
    op == OP_REQUEST
}
pub fn op_is_reply(op: u16) -> bool {
    op == OP_REPLY
}
