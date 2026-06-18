// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stack frames and call stack for fn/call interpretation.
//!
//! Each `(call name args)` pushes a new StackFrame containing the
//! resolved argument values. `%n` references inside the function
//! body are resolved against the current frame: `%n` →
//! `frame.args[n]` for parameters (n < arity), or
//! `frame.locals[&n]` for let-bound locals (n >= arity).
//!
//! Stage 12 Paket B.3 (`docs/discovery/stage-12-completion-plan.md`
//! §B.3): re-introduce a per-frame `locals` map for `(let %n v body)`
//! bindings. The validator already understands `let` and threads its
//! own per-`FunctionContext` locals through type-checking; the
//! interpreter mirrors that by binding/unbinding values on the
//! current frame around the body evaluation.
//!
//! The CallStack enforces a recursion limit to prevent unbounded
//! stack growth in Ring-0 (where stack overflow halts the kernel).

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::error::{InterpretError, InterpretErrorKind};
use crate::value::Value;

/// A stack frame for one function-call activation.
pub struct StackFrame {
    /// Resolved parameter values, in declaration order.
    /// `%n` for `n < args.len()` resolves to `args[n]`.
    pub args: Vec<Value>,
    /// Paket B.3: let-bound locals scoped to this frame. Keys are
    /// the `%n` index >= arity (so they don't collide with
    /// parameters). [`BTreeMap`] keeps iteration deterministic.
    pub locals: BTreeMap<u32, Value>,
}

impl StackFrame {
    pub fn new(args: Vec<Value>) -> Self {
        Self {
            args,
            locals: BTreeMap::new(),
        }
    }

    /// Resolve `%n` to a Value. Parameters (idx < args.len()) win
    /// over locals; for idx >= args.len(), the locals map is
    /// consulted. Out-of-range indices surface as
    /// `ParameterOutOfRange` — never a panic.
    pub fn resolve_parameter(&self, idx: u32) -> Result<Value, InterpretError> {
        let idx_usize = idx as usize;
        if idx_usize < self.args.len() {
            Ok(self.args[idx_usize].clone())
        } else if let Some(v) = self.locals.get(&idx) {
            Ok(v.clone())
        } else {
            Err(InterpretError::new(
                InterpretErrorKind::ParameterOutOfRange,
                "%n index exceeds frame argument count and is not bound as a local",
            ))
        }
    }

    /// Bind a let-local at index `idx`. The validator has already
    /// verified `idx >= arity` (no clobber of parameters) and that
    /// the same `%n` is not re-bound in the same scope; the
    /// interpreter still surfaces typed errors instead of panicking
    /// if the validator's invariants are violated.
    pub fn bind_local(&mut self, idx: u32, value: Value) -> Result<(), InterpretError> {
        if (idx as usize) < self.args.len() {
            return Err(InterpretError::new(
                InterpretErrorKind::LetCollidesWithParameter,
                "let cannot rebind a function parameter (%n with n < arity)",
            ));
        }
        if self.locals.contains_key(&idx) {
            return Err(InterpretError::new(
                InterpretErrorKind::LetRedefinition,
                "let-local already bound in this scope",
            ));
        }
        self.locals.insert(idx, value);
        Ok(())
    }

    /// Remove a let-local at body exit. Safe to call when the
    /// binding is missing (no-op).
    pub fn unbind_local(&mut self, idx: u32) {
        self.locals.remove(&idx);
    }
}

/// Call stack with recursion-limit enforcement.
///
/// Stage 9 limit: 64 frames. Stack overflow in Ring-0 is a kernel
/// panic — the recursion limit prevents this. The limit is intentionally
/// modest for Stage 9; future stages may make it configurable per-sandbox.
pub struct CallStack {
    frames: Vec<StackFrame>,
}

const MAX_RECURSION_DEPTH: usize = 64;

impl CallStack {
    pub fn new() -> Self {
        Self { frames: Vec::new() }
    }

    pub fn push(&mut self, frame: StackFrame) -> Result<(), InterpretError> {
        if self.frames.len() >= MAX_RECURSION_DEPTH {
            return Err(InterpretError::new(
                InterpretErrorKind::RecursionLimitExceeded,
                "call stack depth exceeded MAX_RECURSION_DEPTH (64)",
            ));
        }
        self.frames.push(frame);
        Ok(())
    }

    pub fn pop(&mut self) -> Option<StackFrame> {
        self.frames.pop()
    }

    pub fn current(&self) -> Option<&StackFrame> {
        self.frames.last()
    }

    /// Paket B.3: mutable access to the current frame for let-bind /
    /// unbind. Returns None when no function call is active (let at
    /// top-level is rejected by the validator, but the interpreter
    /// surfaces a typed error rather than panicking on a malformed
    /// program).
    pub fn current_mut(&mut self) -> Option<&mut StackFrame> {
        self.frames.last_mut()
    }

    #[allow(dead_code)]
    pub fn depth(&self) -> usize {
        self.frames.len()
    }
}
