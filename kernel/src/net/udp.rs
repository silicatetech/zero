// SPDX-License-Identifier: AGPL-3.0-or-later
//! UDP — header parsing + checksum (with IPv4 pseudo-header).

use super::ipv4::{self, Ipv4, PROTO_UDP};

pub const UDP_HEADER_LEN: usize = 8;

#[derive(Debug, Clone, Copy)]
pub struct UdpHeader {
    pub src_port: u16,
    pub dst_port: u16,
    pub length: u16,
}

pub fn parse(payload: &[u8]) -> Option<(UdpHeader, &[u8])> {
    if payload.len() < UDP_HEADER_LEN {
        return None;
    }
    let src_port = u16::from_be_bytes([payload[0], payload[1]]);
    let dst_port = u16::from_be_bytes([payload[2], payload[3]]);
    let length = u16::from_be_bytes([payload[4], payload[5]]);
    if (length as usize) < UDP_HEADER_LEN || (length as usize) > payload.len() {
        return None;
    }
    let body = &payload[UDP_HEADER_LEN..length as usize];
    Some((
        UdpHeader {
            src_port,
            dst_port,
            length,
        },
        body,
    ))
}

/// Write a UDP datagram (header + payload) into `out`, computing the
/// checksum over the IPv4 pseudo-header + UDP segment.
pub fn write(
    out: &mut [u8],
    src_ip: Ipv4,
    dst_ip: Ipv4,
    src_port: u16,
    dst_port: u16,
    data: &[u8],
) -> usize {
    let length = (UDP_HEADER_LEN + data.len()) as u16;
    out[0..2].copy_from_slice(&src_port.to_be_bytes());
    out[2..4].copy_from_slice(&dst_port.to_be_bytes());
    out[4..6].copy_from_slice(&length.to_be_bytes());
    out[6..8].copy_from_slice(&0u16.to_be_bytes()); // checksum placeholder
    out[UDP_HEADER_LEN..UDP_HEADER_LEN + data.len()].copy_from_slice(data);

    let pseudo = ipv4::pseudo_header_sum(src_ip, dst_ip, PROTO_UDP, length);
    let mut csum = ipv4::fold_sum(pseudo, &out[..length as usize]);
    // Per RFC 768: a checksum of 0 must be transmitted as 0xFFFF.
    if csum == 0 {
        csum = 0xFFFF;
    }
    out[6..8].copy_from_slice(&csum.to_be_bytes());
    length as usize
}
