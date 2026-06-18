// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ethernet II framing, including up to two 802.1Q/S-tag VLAN headers.

pub const ETH_HEADER_LEN: usize = 14;
pub const VLAN_HEADER_LEN: usize = 4;
pub const MAX_VLAN_TAGS: usize = 2;
pub const MAX_ETH_HEADER_LEN: usize = ETH_HEADER_LEN + (MAX_VLAN_TAGS * VLAN_HEADER_LEN);
pub const ETHERTYPE_IPV4: u16 = 0x0800;
pub const ETHERTYPE_ARP: u16 = 0x0806;
pub const ETHERTYPE_VLAN: u16 = 0x8100;
pub const ETHERTYPE_QINQ: u16 = 0x88A8;
pub const BROADCAST_MAC: [u8; 6] = [0xFF; 6];

pub type Mac = [u8; 6];

#[derive(Debug, Clone, Copy)]
pub struct VlanTag {
    pub tpid: u16,
    pub tci: u16,
}

const EMPTY_VLAN_TAG: VlanTag = VlanTag { tpid: 0, tci: 0 };

#[derive(Debug, Clone, Copy)]
pub struct VlanStack {
    tags: [VlanTag; MAX_VLAN_TAGS],
    len: u8,
}

impl VlanStack {
    pub const fn empty() -> Self {
        Self {
            tags: [EMPTY_VLAN_TAG; MAX_VLAN_TAGS],
            len: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, tag: VlanTag) -> bool {
        let idx = self.len as usize;
        if idx >= MAX_VLAN_TAGS {
            return false;
        }
        self.tags[idx] = tag;
        self.len += 1;
        true
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn header_len(&self) -> usize {
        ETH_HEADER_LEN + self.len() * VLAN_HEADER_LEN
    }

    #[inline]
    fn tag(&self, idx: usize) -> VlanTag {
        self.tags[idx]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EthHeader {
    pub dst: Mac,
    pub src: Mac,
    pub ethertype: u16,
    /// VLAN tags carried by the ingress frame, outermost first.
    /// Replies preserve the stack verbatim.
    pub vlan: VlanStack,
}

pub fn parse(frame: &[u8]) -> Option<(EthHeader, &[u8])> {
    if frame.len() < ETH_HEADER_LEN {
        return None;
    }
    let mut dst = [0u8; 6];
    let mut src = [0u8; 6];
    dst.copy_from_slice(&frame[0..6]);
    src.copy_from_slice(&frame[6..12]);
    let mut ethertype = u16::from_be_bytes([frame[12], frame[13]]);
    let mut vlan = VlanStack::empty();
    let mut payload_off = ETH_HEADER_LEN;

    while ethertype == ETHERTYPE_VLAN || ethertype == ETHERTYPE_QINQ {
        if frame.len() < payload_off + VLAN_HEADER_LEN {
            return None;
        }
        if !vlan.push(VlanTag {
            tpid: ethertype,
            tci: u16::from_be_bytes([frame[payload_off], frame[payload_off + 1]]),
        }) {
            return None;
        }
        ethertype = u16::from_be_bytes([frame[payload_off + 2], frame[payload_off + 3]]);
        payload_off += VLAN_HEADER_LEN;
    }

    Some((
        EthHeader {
            dst,
            src,
            ethertype,
            vlan,
        },
        &frame[payload_off..],
    ))
}

pub fn write(out: &mut [u8], dst: Mac, src: Mac, ethertype: u16) -> usize {
    write_tagged(out, dst, src, VlanStack::empty(), ethertype)
}

pub fn write_tagged(out: &mut [u8], dst: Mac, src: Mac, vlan: VlanStack, ethertype: u16) -> usize {
    out[0..6].copy_from_slice(&dst);
    out[6..12].copy_from_slice(&src);
    let mut off = 12usize;
    let mut i = 0usize;
    while i < vlan.len() {
        let tag = vlan.tag(i);
        out[off..off + 2].copy_from_slice(&tag.tpid.to_be_bytes());
        out[off + 2..off + 4].copy_from_slice(&tag.tci.to_be_bytes());
        off += VLAN_HEADER_LEN;
        i += 1;
    }
    out[off..off + 2].copy_from_slice(&ethertype.to_be_bytes());
    off + 2
}

#[inline]
pub fn header_len(vlan: VlanStack) -> usize {
    vlan.header_len()
}
