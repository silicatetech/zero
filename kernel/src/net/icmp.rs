// SPDX-License-Identifier: AGPL-3.0-or-later
//! ICMP — echo (ping) reply path only.

use super::ipv4;

pub const ICMP_HEADER_LEN: usize = 8;
const TYPE_ECHO_REQUEST: u8 = 8;
const TYPE_ECHO_REPLY: u8 = 0;

#[derive(Debug, Clone, Copy)]
pub struct IcmpHeader {
    pub ty: u8,
    pub code: u8,
    pub identifier: u16,
    pub sequence: u16,
}

pub fn parse(payload: &[u8]) -> Option<(IcmpHeader, &[u8])> {
    if payload.len() < ICMP_HEADER_LEN {
        return None;
    }
    let ty = payload[0];
    let code = payload[1];
    let identifier = u16::from_be_bytes([payload[4], payload[5]]);
    let sequence = u16::from_be_bytes([payload[6], payload[7]]);
    Some((
        IcmpHeader {
            ty,
            code,
            identifier,
            sequence,
        },
        &payload[ICMP_HEADER_LEN..],
    ))
}

pub fn is_echo_request(h: &IcmpHeader) -> bool {
    h.ty == TYPE_ECHO_REQUEST && h.code == 0
}

/// Write an ICMP echo reply (header + echoed data) into `out`. The
/// caller is responsible for the surrounding IPv4 + ethernet framing.
pub fn build_echo_reply(out: &mut [u8], req: &IcmpHeader, data: &[u8]) -> usize {
    out[0] = TYPE_ECHO_REPLY;
    out[1] = 0;
    out[2..4].copy_from_slice(&0u16.to_be_bytes()); // checksum placeholder
    out[4..6].copy_from_slice(&req.identifier.to_be_bytes());
    out[6..8].copy_from_slice(&req.sequence.to_be_bytes());
    let payload_end = ICMP_HEADER_LEN + data.len();
    out[ICMP_HEADER_LEN..payload_end].copy_from_slice(data);
    let csum = ipv4::checksum(&out[..payload_end]);
    out[2..4].copy_from_slice(&csum.to_be_bytes());
    payload_end
}
