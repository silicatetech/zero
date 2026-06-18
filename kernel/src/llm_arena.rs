// SPDX-License-Identifier: AGPL-3.0-or-later
//! LLM Arena — wrapper for Boot-LLM model memory region.
//!
//! Per ADR-028 + V3 Z.689-693 Arena-Pattern:
//! LLM_ARENA wraps the model memory region loaded by QEMU `-device loader`
//! at a fixed physical address. MP2+ may extend this to allocate KV-cache
//! and working memory in adjacent regions.
//!
//! # MP1 Scope
//!
//! MP1 wraps the model bytes only. The region is read-only as far as MP1
//! is concerned (GGUF metadata parsing).
//!
//! # Future MPs
//!
//! - MP2: KV-cache allocation in adjacent memory, working-memory bump alloc
//! - MP6: Multi-architecture trait abstraction integrating LLM_ARENA

use core::ptr::NonNull;
use spin::Mutex;

#[derive(Debug)]
#[allow(dead_code)]
pub enum LlmArenaError {
    InvalidAddress,
    InvalidSize,
}

#[allow(dead_code)]
pub struct LlmArena {
    /// Physical address of model in guest memory.
    phys_addr: u64,
    /// Virtual address (kernel-mapped via physical_memory_offset).
    virt_ptr: NonNull<u8>,
    /// Model size in bytes.
    size: usize,
}

// SAFETY: The LlmArena wraps a pointer to QEMU-loaded model memory
// that is valid for the kernel's entire lifetime. Access is
// synchronized via the LLM_ARENA Mutex.
unsafe impl Send for LlmArena {}
unsafe impl Sync for LlmArena {}

impl LlmArena {
    /// Create LlmArena from QEMU `-device loader` parameters.
    ///
    /// # Safety
    ///
    /// Caller must ensure QEMU loaded a file at `phys_addr` of `size` bytes
    /// and `physical_memory_offset` is the bootloader's mapping offset.
    pub unsafe fn new(
        phys_addr: u64,
        physical_memory_offset: u64,
        size: usize,
    ) -> Result<Self, LlmArenaError> {
        if phys_addr == 0 {
            return Err(LlmArenaError::InvalidAddress);
        }
        if size == 0 {
            return Err(LlmArenaError::InvalidSize);
        }

        let virt_addr = phys_addr.wrapping_add(physical_memory_offset);
        let virt_ptr = NonNull::new(virt_addr as *mut u8).ok_or(LlmArenaError::InvalidAddress)?;

        Ok(Self {
            phys_addr,
            virt_ptr,
            size,
        })
    }

    /// Create LlmArena from a pre-mapped byte slice (ramdisk path).
    ///
    /// The bootloader 0.11 ramdisk mapping is independent of the
    /// `physical_memory_offset` region, so we have to store the
    /// virtual pointer directly. `phys_addr` is set to 0 — for ramdisk
    /// the physical backing is not a single contiguous well-known
    /// address, but the bytes are nonetheless kernel-readable.
    ///
    /// # Safety
    ///
    /// Caller must ensure `bytes` references memory that lives for the
    /// kernel's entire runtime (true for bootloader-loaded ramdisk).
    #[allow(dead_code)]
    pub unsafe fn from_static_slice(bytes: &'static [u8]) -> Result<Self, LlmArenaError> {
        if bytes.is_empty() {
            return Err(LlmArenaError::InvalidSize);
        }
        let virt_ptr =
            NonNull::new(bytes.as_ptr() as *mut u8).ok_or(LlmArenaError::InvalidAddress)?;
        Ok(Self {
            phys_addr: 0,
            virt_ptr,
            size: bytes.len(),
        })
    }

    #[allow(dead_code)]
    pub fn phys_addr(&self) -> u64 {
        self.phys_addr
    }

    #[allow(dead_code)]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns model bytes as a slice.
    ///
    /// # Safety
    ///
    /// Caller must ensure underlying memory is still valid (not reclaimed,
    /// not overwritten by other kernel code).
    #[allow(dead_code)]
    pub unsafe fn model_bytes(&self) -> &[u8] {
        core::slice::from_raw_parts(self.virt_ptr.as_ptr(), self.size)
    }
}

/// Global LLM Arena instance. Set during boot if model is present.
pub static LLM_ARENA: Mutex<Option<LlmArena>> = Mutex::new(None);
