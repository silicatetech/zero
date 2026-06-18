// SPDX-License-Identifier: AGPL-3.0-or-later
//! Linear-Scan register allocator for body-local intermediate values.
//!
//! Stage 10 MP1 used a 3-slot pool (RAX/RCX/RDX) for bare expressions.
//! MP2 keeps the same 3-slot pool but adds parameter-register
//! awareness: when compiling a function body, parameter slots
//! (RDI/RSI/RDX/RCX/R8/R9 per System V AMD64) that overlap with the
//! body pool are reserved and unavailable for body-local use.
//!
//! In particular:
//! - RDX overlaps with parameter slot 2. Reserved if arity >= 3.
//! - RCX overlaps with parameter slot 3. Reserved if arity >= 4.
//! - RAX is the return register, never a parameter slot. Always free.
//!
//! Stage 10 scope: pure binary expression bodies (max 3 simultaneous
//! body-local values). Future stages may extend the pool with R10/R11.
//!
//! See ADR-026 §"Register-Allocation Strategy" for design rationale.

use iced_x86::code_asm::{self, AsmRegister64};

use crate::error::{CodegenError, CodegenErrorKind};

/// Logical register slot in the allocator pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegSlot {
    Rax,
    Rcx,
    Rdx,
}

impl RegSlot {
    /// Convert to iced-x86's `AsmRegister64` for use in CodeAssembler.
    pub fn to_iced(self) -> AsmRegister64 {
        match self {
            RegSlot::Rax => code_asm::rax,
            RegSlot::Rcx => code_asm::rcx,
            RegSlot::Rdx => code_asm::rdx,
        }
    }
}

/// Pool order: RAX first, then RCX, then RDX.
const POOL: [RegSlot; 3] = [RegSlot::Rax, RegSlot::Rcx, RegSlot::Rdx];

/// First-fit allocator over a 3-register pool with parameter reservation.
pub struct LinearScanAllocator {
    /// Tracks which slots are currently allocated for body-local values.
    in_use: [bool; 3],
    /// Tracks which slots are reserved by the current function's parameters.
    reserved: [bool; 3],
}

impl LinearScanAllocator {
    /// Create allocator with no parameter reservations (top-level / bare expr).
    pub fn new() -> Self {
        Self {
            in_use: [false; 3],
            reserved: [false; 3],
        }
    }

    /// Create allocator for a function body with given arity.
    ///
    /// Reserves body-pool slots that overlap with parameter registers:
    /// - arity >= 3: RDX reserved (parameter %2 in System V AMD64)
    /// - arity >= 4: RCX reserved (parameter %3 in System V AMD64)
    /// - RAX (idx 0): always free (return register, not a parameter)
    pub fn for_function(arity: usize) -> Self {
        let mut a = Self::new();
        if arity >= 3 {
            a.reserved[2] = true; // RDX
        }
        if arity >= 4 {
            a.reserved[1] = true; // RCX
        }
        a
    }

    /// Acquire the next available register slot.
    ///
    /// Returns the lowest-indexed free, non-reserved slot.
    pub fn acquire(&mut self) -> Result<RegSlot, CodegenError> {
        for (i, &slot) in POOL.iter().enumerate() {
            if !self.in_use[i] && !self.reserved[i] {
                self.in_use[i] = true;
                return Ok(slot);
            }
        }
        Err(CodegenError::new(
            CodegenErrorKind::AllocatorExhausted,
            "all body-local register slots in use or reserved",
        ))
    }

    /// Release a register slot, making it available for future acquisition.
    pub fn release(&mut self, slot: RegSlot) {
        let idx = match slot {
            RegSlot::Rax => 0,
            RegSlot::Rcx => 1,
            RegSlot::Rdx => 2,
        };
        self.in_use[idx] = false;
    }

    /// Returns slots currently in use (allocated, not merely reserved).
    /// Used for caller-saved save/restore around function calls.
    pub fn in_use_slots(&self) -> Vec<RegSlot> {
        POOL.iter()
            .enumerate()
            .filter_map(|(i, &slot)| if self.in_use[i] { Some(slot) } else { None })
            .collect()
    }

    /// Number of currently allocated slots.
    #[allow(dead_code)]
    pub fn live_count(&self) -> usize {
        self.in_use.iter().filter(|&&b| b).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── MP1 tests (preserved) ───────────────────────────────

    #[test]
    fn fresh_allocator_starts_empty() {
        let alloc = LinearScanAllocator::new();
        assert_eq!(alloc.live_count(), 0);
    }

    #[test]
    fn acquire_returns_rax_first() {
        let mut alloc = LinearScanAllocator::new();
        assert_eq!(alloc.acquire().unwrap(), RegSlot::Rax);
    }

    #[test]
    fn acquire_three_returns_rax_rcx_rdx() {
        let mut alloc = LinearScanAllocator::new();
        assert_eq!(alloc.acquire().unwrap(), RegSlot::Rax);
        assert_eq!(alloc.acquire().unwrap(), RegSlot::Rcx);
        assert_eq!(alloc.acquire().unwrap(), RegSlot::Rdx);
    }

    #[test]
    fn fourth_acquire_is_exhausted() {
        let mut alloc = LinearScanAllocator::new();
        alloc.acquire().unwrap();
        alloc.acquire().unwrap();
        alloc.acquire().unwrap();
        let result = alloc.acquire();
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().kind,
            CodegenErrorKind::AllocatorExhausted
        );
    }

    #[test]
    fn release_makes_slot_available() {
        let mut alloc = LinearScanAllocator::new();
        let r1 = alloc.acquire().unwrap();
        alloc.release(r1);
        assert_eq!(alloc.live_count(), 0);
        assert_eq!(alloc.acquire().unwrap(), RegSlot::Rax);
    }

    #[test]
    fn release_then_acquire_lowest_free() {
        let mut alloc = LinearScanAllocator::new();
        let r1 = alloc.acquire().unwrap();
        let _r2 = alloc.acquire().unwrap();
        alloc.release(r1);
        assert_eq!(alloc.acquire().unwrap(), RegSlot::Rax);
    }

    // ── MP2 tests (new) ─────────────────────────────────────

    #[test]
    fn for_function_arity_zero_full_pool() {
        let mut alloc = LinearScanAllocator::for_function(0);
        assert_eq!(alloc.acquire().unwrap(), RegSlot::Rax);
        assert_eq!(alloc.acquire().unwrap(), RegSlot::Rcx);
        assert_eq!(alloc.acquire().unwrap(), RegSlot::Rdx);
    }

    #[test]
    fn for_function_arity_three_reserves_rdx() {
        let mut alloc = LinearScanAllocator::for_function(3);
        assert_eq!(alloc.acquire().unwrap(), RegSlot::Rax);
        assert_eq!(alloc.acquire().unwrap(), RegSlot::Rcx);
        let result = alloc.acquire();
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().kind,
            CodegenErrorKind::AllocatorExhausted
        );
    }

    #[test]
    fn for_function_arity_four_reserves_rcx_rdx() {
        let mut alloc = LinearScanAllocator::for_function(4);
        assert_eq!(alloc.acquire().unwrap(), RegSlot::Rax);
        let result = alloc.acquire();
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().kind,
            CodegenErrorKind::AllocatorExhausted
        );
    }

    #[test]
    fn in_use_slots_returns_currently_allocated() {
        let mut alloc = LinearScanAllocator::new();
        alloc.acquire().unwrap(); // RAX
        alloc.acquire().unwrap(); // RCX
        let live = alloc.in_use_slots();
        assert_eq!(live, vec![RegSlot::Rax, RegSlot::Rcx]);
    }

    #[test]
    fn in_use_slots_skips_reserved() {
        let mut alloc = LinearScanAllocator::for_function(3); // RDX reserved
        alloc.acquire().unwrap(); // RAX
        let live = alloc.in_use_slots();
        assert_eq!(live, vec![RegSlot::Rax]);
    }
}
