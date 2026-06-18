// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fixed-size bump arena. No growth. Returns OutOfMemory error when
//! exhausted.

use core::mem::{align_of, size_of, MaybeUninit};
use core::ptr::NonNull;
use core::slice;

use crate::error::ArenaError;

/// Fixed-size bump arena.
///
/// Allocations advance an offset pointer through a single memory
/// block. Once the offset reaches the end, allocation returns
/// [`ArenaError::OutOfMemory`].
///
/// # Memory source
///
/// The backing memory is provided as a `&'static mut [u8]` slice,
/// which the caller owns and guarantees to outlive the arena.
///
/// # Example
///
/// ```ignore
/// // In Ring-0 (MP4): backing comes from a linker-reserved region.
/// // In host tests: backing comes from `Box::leak`.
/// static mut BACKING: [u8; 4096] = [0u8; 4096];
/// let arena = FixedArena::new(unsafe { &mut BACKING });
/// ```
pub struct FixedArena {
    /// Pointer to the start of the backing memory.
    base: NonNull<u8>,
    /// Capacity in bytes.
    capacity: usize,
    /// Current offset (next allocation starts here).
    offset: usize,
}

// SAFETY: FixedArena contains a NonNull<u8> which is !Send by default.
// The arena exclusively owns its backing memory region (no aliasing),
// so transferring the arena between threads is safe. Access is gated
// by &mut self, preventing concurrent use without external sync.
unsafe impl Send for FixedArena {}

impl FixedArena {
    /// Construct a new arena from a backing slice.
    pub fn new(backing: &'static mut [u8]) -> Self {
        let capacity = backing.len();
        // SAFETY: `backing` is a valid `&'static mut` slice; its data
        // pointer is non-null.
        let base = unsafe { NonNull::new_unchecked(backing.as_mut_ptr()) };
        FixedArena {
            base,
            capacity,
            offset: 0,
        }
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes used so far.
    pub fn used(&self) -> usize {
        self.offset
    }

    /// Bytes available.
    pub fn available(&self) -> usize {
        self.capacity - self.offset
    }

    /// Reset the arena. All previously allocated values are
    /// invalidated. Destructors are NOT called (see crate-level docs).
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Allocate space for a single value of type `T` and initialize
    /// it.
    pub fn alloc<T>(&mut self, value: T) -> Result<&mut T, ArenaError> {
        let uninit = self.alloc_uninit::<T>()?;
        // SAFETY: `uninit` points to space sized and aligned for T;
        // we initialize it before returning.
        unsafe {
            let ptr = uninit.as_mut_ptr();
            ptr.write(value);
            Ok(&mut *ptr)
        }
    }

    /// Allocate uninitialized space for a single value of type `T`.
    pub fn alloc_uninit<T>(&mut self) -> Result<&mut MaybeUninit<T>, ArenaError> {
        let size = size_of::<T>();
        let align = align_of::<T>();
        let ptr = self.alloc_raw(size, align)?;
        // SAFETY: `alloc_raw` returns a pointer to `size` bytes
        // aligned to `align`. MaybeUninit<T> has the same layout as T.
        unsafe { Ok(&mut *(ptr.as_ptr() as *mut MaybeUninit<T>)) }
    }

    /// Allocate space for a slice of `T` and copy values from the
    /// source slice.
    pub fn alloc_slice_copy<T: Copy>(&mut self, slice: &[T]) -> Result<&mut [T], ArenaError> {
        let len = slice.len();
        if len == 0 {
            return Ok(&mut []);
        }
        let size = size_of::<T>()
            .checked_mul(len)
            .ok_or(ArenaError::SizeOverflow)?;
        let align = align_of::<T>();
        let ptr = self.alloc_raw(size, align)?;
        // SAFETY: `alloc_raw` returns properly sized and aligned
        // memory; we copy `len` T values into it.
        unsafe {
            let dst = ptr.as_ptr() as *mut T;
            core::ptr::copy_nonoverlapping(slice.as_ptr(), dst, len);
            Ok(slice::from_raw_parts_mut(dst, len))
        }
    }

    /// Allocate `len` bytes and zero them in place.
    ///
    /// Pre-A.2 (F-H8): eliminates the temporary heap `Vec<u8>` that
    /// `alloc_slice_copy(&vec![0u8; len])` callers used as a workaround.
    /// The zeroing happens directly in the arena via
    /// `core::ptr::write_bytes` — no extra allocation outside the arena.
    pub fn alloc_zeroed(&mut self, len: usize) -> Result<&mut [u8], ArenaError> {
        if len == 0 {
            return Ok(&mut []);
        }
        let ptr = self.alloc_raw(len, 1)?;
        // SAFETY: `alloc_raw` returned `len` valid bytes; write_bytes
        // zero-initialises the entire span in place.
        unsafe {
            core::ptr::write_bytes(ptr.as_ptr(), 0u8, len);
            Ok(slice::from_raw_parts_mut(ptr.as_ptr(), len))
        }
    }

    /// Allocate space for a string and copy bytes from the source.
    pub fn alloc_str(&mut self, s: &str) -> Result<&mut str, ArenaError> {
        let bytes = self.alloc_slice_copy(s.as_bytes())?;
        // SAFETY: bytes were copied from a valid str, so they form
        // valid UTF-8.
        unsafe { Ok(core::str::from_utf8_unchecked_mut(bytes)) }
    }

    /// Allocate `size` bytes aligned to `align`. Returns a non-null
    /// pointer into the backing memory.
    ///
    /// `pub` so kernel's `RuntimeArenaInner` and `ChunkedArena` can
    /// compose this primitive.
    pub fn alloc_raw(&mut self, size: usize, align: usize) -> Result<NonNull<u8>, ArenaError> {
        if !align.is_power_of_two() {
            return Err(ArenaError::AlignmentInvalid { alignment: align });
        }
        let aligned_offset = align_up(self.offset, align).ok_or(ArenaError::SizeOverflow)?;
        let new_offset = aligned_offset
            .checked_add(size)
            .ok_or(ArenaError::SizeOverflow)?;
        if new_offset > self.capacity {
            return Err(ArenaError::OutOfMemory {
                requested: size,
                available: self.capacity.saturating_sub(self.offset),
            });
        }
        // SAFETY: `aligned_offset < self.capacity` by the check above.
        let ptr = unsafe { NonNull::new_unchecked(self.base.as_ptr().add(aligned_offset)) };
        self.offset = new_offset;
        Ok(ptr)
    }
}

/// Round `offset` up to the nearest multiple of `align`.
/// `align` must be a power of two.
fn align_up(offset: usize, align: usize) -> Option<usize> {
    debug_assert!(align.is_power_of_two());
    let mask = align - 1;
    offset.checked_add(mask).map(|v| v & !mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use alloc::vec;

    fn make_backing(size: usize) -> &'static mut [u8] {
        Box::leak(vec![0u8; size].into_boxed_slice())
    }

    #[test]
    fn empty_arena_has_full_capacity() {
        let arena = FixedArena::new(make_backing(1024));
        assert_eq!(arena.capacity(), 1024);
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.available(), 1024);
    }

    #[test]
    fn alloc_single_value() {
        let mut arena = FixedArena::new(make_backing(1024));
        let r = arena.alloc(42u64).unwrap();
        assert_eq!(*r, 42);
    }

    #[test]
    fn alloc_advances_offset() {
        let mut arena = FixedArena::new(make_backing(1024));
        arena.alloc(1u64).unwrap();
        assert_eq!(arena.used(), 8);
        arena.alloc(2u64).unwrap();
        assert_eq!(arena.used(), 16);
    }

    #[test]
    fn alloc_respects_alignment() {
        let mut arena = FixedArena::new(make_backing(1024));
        let _ = arena.alloc(1u8).unwrap();
        assert_eq!(arena.used(), 1);
        let _ = arena.alloc(2u64).unwrap();
        assert_eq!(arena.used(), 16);
    }

    #[test]
    fn alloc_oom_returns_error() {
        let mut arena = FixedArena::new(make_backing(16));
        arena.alloc(1u64).unwrap();
        arena.alloc(2u64).unwrap();
        let result = arena.alloc(3u64);
        assert!(matches!(result, Err(ArenaError::OutOfMemory { .. })));
    }

    #[test]
    fn alloc_oom_reports_correct_sizes() {
        let mut arena = FixedArena::new(make_backing(8));
        let result = arena.alloc(1u128);
        match result {
            Err(ArenaError::OutOfMemory {
                requested,
                available,
            }) => {
                assert_eq!(requested, 16);
                assert_eq!(available, 8);
            }
            _ => panic!("expected OutOfMemory error"),
        }
    }

    #[test]
    fn reset_restores_capacity() {
        let mut arena = FixedArena::new(make_backing(64));
        arena.alloc(1u64).unwrap();
        arena.alloc(2u64).unwrap();
        assert_eq!(arena.used(), 16);
        arena.reset();
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.available(), 64);
    }

    #[test]
    fn reset_allows_reuse() {
        let mut arena = FixedArena::new(make_backing(16));
        arena.alloc(1u64).unwrap();
        arena.alloc(2u64).unwrap();
        arena.reset();
        let r = arena.alloc(99u64).unwrap();
        assert_eq!(*r, 99);
    }

    #[test]
    fn alloc_slice_copy_preserves_values() {
        let mut arena = FixedArena::new(make_backing(1024));
        let src = [1u32, 2, 3, 4, 5];
        let dst = arena.alloc_slice_copy(&src).unwrap();
        assert_eq!(dst, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn alloc_slice_copy_empty_does_not_advance_offset() {
        let mut arena = FixedArena::new(make_backing(1024));
        let src: [u32; 0] = [];
        let dst = arena.alloc_slice_copy(&src).unwrap();
        assert_eq!(dst.len(), 0);
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn alloc_str_preserves_content() {
        let mut arena = FixedArena::new(make_backing(1024));
        let s = arena.alloc_str("hello").unwrap();
        assert_eq!(s, "hello");
    }

    #[test]
    fn alloc_str_unicode() {
        let mut arena = FixedArena::new(make_backing(1024));
        let s = arena.alloc_str("héllo wörld").unwrap();
        assert_eq!(s, "héllo wörld");
    }

    #[test]
    fn alloc_uninit_returns_writable_space() {
        let mut arena = FixedArena::new(make_backing(1024));
        let slot = arena.alloc_uninit::<u64>().unwrap();
        slot.write(123);
        // SAFETY: we just wrote to the slot.
        let val = unsafe { slot.assume_init_ref() };
        assert_eq!(*val, 123);
    }

    // === Pre-A.2 (F-H8): alloc_zeroed ===

    #[test]
    fn alloc_zeroed_initialises_bytes_to_zero() {
        let mut arena = FixedArena::new(make_backing(64));
        let bytes = arena.alloc_zeroed(32).unwrap();
        assert_eq!(bytes.len(), 32);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn alloc_zeroed_zero_length_does_not_advance() {
        let mut arena = FixedArena::new(make_backing(64));
        let bytes = arena.alloc_zeroed(0).unwrap();
        assert_eq!(bytes.len(), 0);
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn alloc_zeroed_advances_offset_by_len() {
        let mut arena = FixedArena::new(make_backing(64));
        let _ = arena.alloc_zeroed(17).unwrap();
        // No alignment slack for u8 (align == 1).
        assert_eq!(arena.used(), 17);
    }

    #[test]
    fn alloc_zeroed_oom_when_arena_full() {
        let mut arena = FixedArena::new(make_backing(8));
        let res = arena.alloc_zeroed(16);
        assert!(matches!(res, Err(ArenaError::OutOfMemory { .. })));
    }

    #[test]
    fn alloc_zeroed_overwrites_previous_byte_pattern() {
        // Use raw alloc to write a non-zero pattern, then reset and
        // verify alloc_zeroed gives zeros in the same memory region.
        let mut arena = FixedArena::new(make_backing(64));
        let scratch = arena.alloc_slice_copy(&[0xFFu8; 32]).unwrap();
        assert!(scratch.iter().all(|&b| b == 0xFF));
        arena.reset();
        let zeros = arena.alloc_zeroed(32).unwrap();
        assert!(zeros.iter().all(|&b| b == 0));
    }

    #[test]
    fn align_up_basic() {
        assert_eq!(align_up(0, 8), Some(0));
        assert_eq!(align_up(1, 8), Some(8));
        assert_eq!(align_up(7, 8), Some(8));
        assert_eq!(align_up(8, 8), Some(8));
        assert_eq!(align_up(9, 8), Some(16));
    }

    #[test]
    fn invalid_alignment_returns_error() {
        let mut arena = FixedArena::new(make_backing(1024));
        // Manually invoke alloc_raw with non-power-of-two alignment.
        let result = arena.alloc_raw(16, 3);
        assert!(matches!(
            result,
            Err(ArenaError::AlignmentInvalid { alignment: 3 })
        ));
    }
}
