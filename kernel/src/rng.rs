// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sub-MP-D3.5: RNG module for non-deterministic sampling.
//!
//! State-derived entropy seed + xorshift64 PRNG. Per SQ2=γ ratification:
//! entropy = hash(boot_uptime_tick, token_position, prompt_bytes).
//! No virtio-rng driver scope creep; Pillar 7 trivially satisfied.
//!
//! Per V3.1 Pillar 7 (Z.264-269): NO `#[cfg(target_arch)]`, NO platform
//! intrinsics. Pure-Rust PRNG only.
//!
//! Per V3.1 Pillar 1 (Z.205-210): O(1) per next_u64. Zero allocation.
//! Zero runtime IO.
//!
//! Per V3.1 Pillar 8 (Z.275-285): Rust temporary; will migrate to
//! Quarks in Stage 12+ when Validator + interpreter mature.

/// Xorshift64 PRNG state.
///
/// Algorithm: Marsaglia xorshift64 — minimal, fast, platform-independent.
/// Period: 2^64 - 1. Sufficient for Top-K=40 sampling diversity.
/// Pillar 7 compliant: pure scalar Rust, no platform intrinsics.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Create new PRNG from seed. Seed must be non-zero.
    /// If zero seed provided, uses fallback constant.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed != 0 {
                seed
            } else {
                0x853c_49e6_748f_ea9b
            },
        }
    }

    /// Generate next pseudo-random u64 via xorshift64.
    /// O(1), zero allocation, platform-independent.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate next pseudo-random u32 (upper 32 bits of u64).
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
}

/// Compute state-derived entropy seed from boot-variable inputs.
///
/// Per SQ2=γ ratification: hash of (boot_uptime_tick + token_position +
/// prompt_bytes). Boot uptime tick varies between boots due to timing
/// jitter in hardware initialization. Combined with token position and
/// prompt content, produces unique-per-boot seed.
///
/// Uses FNV-1a hash (simple, fast, no_std, platform-independent).
pub fn entropy_seed(boot_uptime_tick: u64, token_position: u32, prompt_bytes: &[u8]) -> u64 {
    // FNV-1a 64-bit hash
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;

    let mut hash = FNV_OFFSET;

    // Mix boot uptime tick (8 bytes, LE)
    let tick_bytes = boot_uptime_tick.to_le_bytes();
    for &b in &tick_bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    // Mix token position (4 bytes, LE)
    let pos_bytes = token_position.to_le_bytes();
    for &b in &pos_bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    // Mix prompt bytes
    for &b in prompt_bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    hash
}
