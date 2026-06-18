// SPDX-License-Identifier: AGPL-3.0-or-later
//! IPv4 header parsing + Internet-checksum.
//!
//! We emit IHL=5 (no options), DF=0, fragmentation unsupported on the
//! send path; on RX we accept fragments only if MF=0 and FragOffset=0
//! (i.e. unfragmented datagrams).

pub const IPV4_HEADER_LEN: usize = 20;
pub const PROTO_ICMP: u8 = 1;
pub const PROTO_TCP: u8 = 6;
pub const PROTO_UDP: u8 = 17;

pub type Ipv4 = [u8; 4];

#[derive(Debug, Clone, Copy)]
pub struct Ipv4Header {
    pub total_length: u16,
    pub protocol: u8,
    pub src: Ipv4,
    pub dst: Ipv4,
    /// Length of the header in bytes (IHL*4); always ≥ 20.
    pub header_len: usize,
    pub identification: u16,
}

pub fn parse(payload: &[u8]) -> Option<(Ipv4Header, &[u8])> {
    if payload.len() < IPV4_HEADER_LEN {
        return None;
    }
    let version_ihl = payload[0];
    let version = version_ihl >> 4;
    let ihl = (version_ihl & 0x0F) as usize;
    if version != 4 || ihl < 5 {
        return None;
    }
    let header_len = ihl * 4;
    if payload.len() < header_len {
        return None;
    }
    let total_length = u16::from_be_bytes([payload[2], payload[3]]);
    let identification = u16::from_be_bytes([payload[4], payload[5]]);
    let flags_frag = u16::from_be_bytes([payload[6], payload[7]]);
    // MF=bit 13, FragOffset=bits 0..12. Reject anything fragmented.
    if flags_frag & 0x3FFF != 0 {
        return None;
    }
    let protocol = payload[9];
    let mut src = [0u8; 4];
    let mut dst = [0u8; 4];
    src.copy_from_slice(&payload[12..16]);
    dst.copy_from_slice(&payload[16..20]);

    // We don't verify checksum on RX — QEMU and modern NICs hand us
    // packets the e1000 has already validated when SECRC is set.
    let total = total_length as usize;
    let body_end = total.min(payload.len());
    if body_end < header_len {
        return None;
    }
    Some((
        Ipv4Header {
            total_length,
            protocol,
            src,
            dst,
            header_len,
            identification,
        },
        &payload[header_len..body_end],
    ))
}

/// Compute the Internet checksum (RFC 1071) over `bytes`.
pub fn checksum(bytes: &[u8]) -> u16 {
    let mut sum: u32 = 0;
    let mut i = 0;
    while i + 1 < bytes.len() {
        sum = sum.wrapping_add(u16::from_be_bytes([bytes[i], bytes[i + 1]]) as u32);
        i += 2;
    }
    if i < bytes.len() {
        sum = sum.wrapping_add((bytes[i] as u32) << 8);
    }
    while sum >> 16 != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    !(sum as u16)
}

/// Build an IPv4 header (no options) into `out[0..20]`. Caller is
/// responsible for filling the payload after `out[20..]` before
/// recomputing checksum-spanning structures.
pub fn write_header(
    out: &mut [u8],
    src: Ipv4,
    dst: Ipv4,
    protocol: u8,
    payload_len: u16,
    identification: u16,
) -> usize {
    let total = IPV4_HEADER_LEN as u16 + payload_len;
    out[0] = 0x45; // version 4, IHL 5
    out[1] = 0; // DSCP/ECN
    out[2..4].copy_from_slice(&total.to_be_bytes());
    out[4..6].copy_from_slice(&identification.to_be_bytes());
    out[6..8].copy_from_slice(&0u16.to_be_bytes()); // no flags, no frag
    out[8] = 64; // TTL
    out[9] = protocol;
    out[10..12].copy_from_slice(&0u16.to_be_bytes()); // checksum placeholder
    out[12..16].copy_from_slice(&src);
    out[16..20].copy_from_slice(&dst);
    let csum = checksum(&out[..IPV4_HEADER_LEN]);
    out[10..12].copy_from_slice(&csum.to_be_bytes());
    IPV4_HEADER_LEN
}

/// Sum the IPv4 pseudo-header used by TCP/UDP checksums. Returns the
/// running 32-bit accumulator; finalize with the same fold as
/// [`checksum`].
pub fn pseudo_header_sum(src: Ipv4, dst: Ipv4, protocol: u8, length: u16) -> u32 {
    let mut sum: u32 = 0;
    sum = sum.wrapping_add(u16::from_be_bytes([src[0], src[1]]) as u32);
    sum = sum.wrapping_add(u16::from_be_bytes([src[2], src[3]]) as u32);
    sum = sum.wrapping_add(u16::from_be_bytes([dst[0], dst[1]]) as u32);
    sum = sum.wrapping_add(u16::from_be_bytes([dst[2], dst[3]]) as u32);
    sum = sum.wrapping_add(protocol as u32);
    sum = sum.wrapping_add(length as u32);
    sum
}

pub fn fold_sum(mut sum: u32, bytes: &[u8]) -> u16 {
    let mut i = 0;
    while i + 1 < bytes.len() {
        sum = sum.wrapping_add(u16::from_be_bytes([bytes[i], bytes[i + 1]]) as u32);
        i += 2;
    }
    if i < bytes.len() {
        sum = sum.wrapping_add((bytes[i] as u32) << 8);
    }
    while sum >> 16 != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    !(sum as u16)
}
