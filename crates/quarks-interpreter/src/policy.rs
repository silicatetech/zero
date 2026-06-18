// SPDX-License-Identifier: AGPL-3.0-or-later
//! 12i — Quarks Policy-Instruction-Set context.
//!
//! Policy and query instructions are first-class Quarks
//! instructions (`(policy <subsystem> <operation> args...)` and
//! `(query <subsystem> <metric> args...)`) that the kernel-LLM emits
//! and the interpreter dispatches into a [`PolicyContext`] trait.
//!
//! The trait exists so the standalone `interpret` entry point — used
//! by host-side tests and by code paths that have no sandbox manager
//! — can run Quarks programs that lack policy/query instructions
//! without depending on `zero-sandbox`. When a real context is
//! available (the SandboxManager-backed path in `zero-sandbox`),
//! callers use [`crate::interpret_with_context`] to plug it in.
//!
//! Per V3.4 / ADR-019 §4 (Hardware Capability Service shape):
//! - the dispatch into a context is the *policy plane* hand-off;
//!   the context body itself routes capability checks through the
//!   sandbox manager (`require_capability`) before performing any
//!   side effect, so AI sits on the policy plane and never in the
//!   data path;
//! - policy operations return `i64` status codes (0 = success);
//! - query operations return `i64` telemetry values.
//!
//! The trait deliberately has no associated type for the context's
//! own state (e.g. capability-id, sandbox-id) — the implementer
//! captures that state at construction time. This keeps the
//! interpreter's call site monomorphic.

use crate::value::Value;
use alloc::vec::Vec;

/// 12i — discriminator for [`PolicyContext::policy`] /
/// [`PolicyContext::query`] failures. Each variant maps onto an
/// [`InterpretErrorKind::PolicyDispatchFailed`](crate::InterpretErrorKind::PolicyDispatchFailed)
/// at the interpreter boundary; downstream code (typically the
/// SandboxManager-backed context in `zero-sandbox`) inspects the
/// variant for finer routing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyError {
    /// The subsystem symbol (`gpu`, `network`, `storage`, `schedule`,
    /// `thermal`, `memory`, ...) is not recognised by this context.
    UnknownSubsystem,
    /// The operation symbol (`allocate-slice`, `set-priority`,
    /// `utilization`, ...) is not recognised for the given subsystem.
    UnknownOperation,
    /// The argument shape is wrong (count, type, or value range).
    InvalidArgument,
    /// The calling sandbox does not hold the capability required for
    /// the requested subsystem. Per ADR-019 §5 runtime enforcement.
    PermissionDenied,
    /// This context does not implement policy/query at all (the
    /// canonical "null" context).
    NotSupported,
    /// The dispatch reached the context but the underlying service
    /// (Hardware Capability Service body, sandbox manager) returned
    /// an error not covered by the variants above.
    OperationFailed,
}

/// 12i — interpreter-side hook for `(policy ...)` / `(query ...)`.
///
/// Implementers route the call through the sandbox manager
/// (`require_capability` on the calling sandbox + dispatch into the
/// Hardware Capability Service or the manager's own scheduling
/// surface). The interpreter passes args as already-evaluated
/// [`Value`]s (typically [`Value::Integer`]).
///
/// Subsystem and operation are static `&str` references to the
/// Quarks symbols (lowercase, kebab-case). Implementers MUST
/// match on both to dispatch.
pub trait PolicyContext {
    /// Dispatch a `(policy <subsystem> <operation> args...)` call.
    /// Returns a status code (`0` = success) or a [`PolicyError`].
    fn policy(
        &mut self,
        subsystem: &str,
        operation: &str,
        args: &[Value],
    ) -> Result<i64, PolicyError>;

    /// Dispatch a `(query <subsystem> <metric> args...)` call.
    /// Returns a telemetry value (`>= 0`) or a [`PolicyError`].
    fn query(&mut self, subsystem: &str, metric: &str, args: &[Value]) -> Result<i64, PolicyError>;

    /// Paket B.6 — `(read-handle h)` I/O surface.
    ///
    /// Dispatched by the interpreter when it encounters
    /// `(read-handle h)`. Implementations route the call through the
    /// sandbox manager: capability check (`SandboxHandleRead`),
    /// handle-table lookup, optional copy from a granted region.
    ///
    /// Returns the bytes payload visible to the calling sandbox; the
    /// validator types the surface as `[Handle] -> [Bytes]`. The
    /// default impl returns [`PolicyError::NotSupported`] so existing
    /// contexts compile unchanged.
    fn read_handle(&mut self, _handle: u64) -> Result<Vec<u8>, PolicyError> {
        Err(PolicyError::NotSupported)
    }

    /// Paket B.6 — `(write-host-state key value)` I/O surface.
    ///
    /// Dispatched by the interpreter when it encounters
    /// `(write-host-state <key-symbol> <value>)`. Implementations
    /// route the call through the sandbox manager: capability check
    /// (`HostStateWrite`), key-namespace check, write of the i64
    /// value into the host's structured-state surface.
    ///
    /// Returns an i64 status (`0 = success`); the validator types the
    /// surface as `[I64] -> [I64]` (the key is consumed as a symbol
    /// literal at the dispatch site, so it does not appear on the
    /// value stack). The default impl returns
    /// [`PolicyError::NotSupported`] so existing contexts compile
    /// unchanged.
    fn write_host_state(&mut self, _key: &str, _value: i64) -> Result<i64, PolicyError> {
        Err(PolicyError::NotSupported)
    }
}

/// Default no-op [`PolicyContext`] used by the standalone
/// [`crate::interpret`] entry point. Returns
/// [`PolicyError::NotSupported`] for every dispatch — programs that
/// use `(policy ...)` or `(query ...)` instructions must use
/// [`crate::interpret_with_context`] with a real context (typically
/// the SandboxManager-backed one in `zero-sandbox`).
pub struct NullPolicyContext;

impl NullPolicyContext {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NullPolicyContext {
    fn default() -> Self {
        Self::new()
    }
}

impl PolicyContext for NullPolicyContext {
    fn policy(
        &mut self,
        _subsystem: &str,
        _operation: &str,
        _args: &[Value],
    ) -> Result<i64, PolicyError> {
        Err(PolicyError::NotSupported)
    }

    fn query(
        &mut self,
        _subsystem: &str,
        _metric: &str,
        _args: &[Value],
    ) -> Result<i64, PolicyError> {
        Err(PolicyError::NotSupported)
    }
}
