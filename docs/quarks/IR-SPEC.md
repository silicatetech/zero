# Quarks Intermediate Representation Specification v0.1

**Status:** Draft
**Date:** 2026-04-24
**Depends on:** Agent model, intent format

---

## 1. Overview

Quarks-IR is a **machine-to-machine textual intermediate representation** for Zero. It is the native execution format for Ring-0 agent code: the kernel's executor consumes Quarks-IR directly after a validator pass confirms structural and type correctness.

**Purpose:** provide a minimal, unambiguous instruction format that agents (human-authored or LLM-generated) compile *to*, and that the kernel executes *from*. The IR sits below the JSON intent layer. JSON intents carry inter-agent messages; Quarks-IR defines what an agent *does* when it receives one.

**Non-goals:**

- Human readability beyond what S-expressions provide by accident.
- Syntactic sugar, macros, or source-level abstractions.
- Source-level debugging (debug metadata is deferred to v0.2+).

**Relationship to intent format:** The intent format specification defines inter-agent communication (JSON, hybrid parsing, receiver validation). Quarks-IR is the intra-agent execution layer underneath. An agent's `recv` instruction yields a parsed intent; its `send` instruction emits one. The JSON serialization boundary sits between `send`/`recv` and the kernel's intent dispatcher.

**Relationship to Roadmap:** Quarks-IR is a Stage 6 deliverable. Stage 4 agents (Ping, Pong) were written in Rust; Stage 6 agents will be expressible in Quarks-IR, validated, and executed by a kernel-resident interpreter.

---

## 2. Encoding: S-Expressions

Quarks-IR programs are encoded as S-expressions.

### 2.1 Atom Types

| Atom | Syntax | Examples |
|------|--------|----------|
| Integer literal | Decimal digits, optional leading `-` | `42`, `-1`, `0` |
| Bytes literal | `#x` prefix followed by hex pairs | `#x48656c6c6f`, `#x00ff` |
| Handle literal | `@` prefix followed by decimal | `@0`, `@5`, `@127` |
| Symbol | `[a-z0-9_-]+`, case-sensitive | `send`, `recv`, `my-agent`, `loop_body` |

### 2.2 Lists

A list is a parenthesized sequence of atoms and/or nested lists. The first position must be a symbol (the instruction). Subsequent positions are arguments.

```
(add 1 2)
(send @5 #x48656c6c6f)
(if (cmp x 0) (return 1) (return 0))
```

### 2.3 Whitespace and Delimiters

- Spaces, tabs, and newlines are interchangeable whitespace. All separate tokens equivalently.
- No comments. IR is machine-generated; comments belong in the source language that compiles to IR.
- No escape sequences inside atoms.

### 2.4 Absent Types

- **No string type.** Text is represented as bytes. The encoding (UTF-8, ASCII, etc.) is the agent's concern.
- **No boolean type.** `0` is false, any nonzero `i64` is true. C convention.
- **No float type** in v0.1. Deferred to v0.2+ (see Section 9).

---

## 3. Type System

Three primitive types. No composite types in v0.1.

| Type | Width | Representation | Description |
|------|-------|----------------|-------------|
| `i64` | 64 bits | Two's complement, signed | General-purpose integer. Wrap-around on overflow (no traps). |
| `bytes` | Variable | Immutable byte sequence | Maximum 16 KiB in v0.1. |
| `handle` | 64 bits | Opaque kernel-assigned reference | Agent handles, resource handles. `@0` is the null handle. |

**No implicit type coercion.** An `i64` is never silently treated as a `handle` or vice versa. The validator rejects programs that pass a value of the wrong type to an instruction. The only exception is the `dup`, `drop`, and `swap` instructions, which are polymorphic (Section 4.2).

---

## 4. Instruction Set

### 4.1 Control Flow

| Instruction | Stack effect | Description |
|-------------|-------------|-------------|
| `(if cond then else)` | Evaluates `cond` (which pushes 1 `i64` to stack), consumes that `i64` as branch-condition, then executes `then` if nonzero, otherwise `else`. Both branches must produce identical stack effects. Net stack effect of the whole `if` expression equals the stack effect of either branch. | The condition result is consumed by the `if` construct itself and is not visible to either branch. |
| `(loop body)` | Stack-neutral (end state = start state) | Execute `body` repeatedly until `break` or `return`. |
| `(break)` | Stack-neutral | Exit the innermost enclosing `loop`. |
| `(return value)` | Consumes stack down to 1 value | Terminate the agent. `value` is evaluated and becomes the agent's exit value. |

### 4.2 Memory, Arithmetic, and Stack

| Instruction | Consumes | Produces | Description |
|-------------|----------|----------|-------------|
| `(load addr)` | 1 `i64` (address) | 1 `i64` | Read `i64` from agent-local memory at `addr`. |
| `(store addr value)` | 2 `i64` (address, value) | nothing | Write `value` to agent-local memory at `addr`. |
| `(add a b)` | 2 `i64` | 1 `i64` | Addition. Wrap-around on overflow. |
| `(sub a b)` | 2 `i64` | 1 `i64` | Subtraction. Wrap-around on underflow. |
| `(mul a b)` | 2 `i64` | 1 `i64` | Multiplication. Wrap-around on overflow. |
| `(div a b)` | 2 `i64` | 1 `i64` | Integer division. Division by zero is a runtime trap. |
| `(cmp a b)` | 2 `i64` | 1 `i64` | Returns `-1` if `a < b`, `0` if `a == b`, `1` if `a > b`. |
| `(dup)` | 1 any | 2 same | Duplicate top-of-stack. Polymorphic over types. |
| `(drop)` | 1 any | nothing | Remove top-of-stack. Polymorphic over types. |
| `(swap)` | 2 any | 2 any (reversed) | Swap top two stack values. Polymorphic over types. |

### 4.3 OS Intents

| Instruction | Consumes | Produces | Description |
|-------------|----------|----------|-------------|
| `(send to payload)` | 1 `handle` (`to`), 1 `bytes` (`payload`) | 1 `i64` (error code) | Send `payload` as an intent to the agent identified by `to`. Returns `ERR_OK` on success. |
| `(recv)` | nothing | 3 values (bottom to top): `handle` (sender), `i64` (intent-type), `bytes` (payload) | Block until an intent arrives in the agent's inbox. Pushes sender handle, intent type, and payload onto the stack. |
| `(spawn name)` | 1 `bytes` (`name`) | 1 `handle` | Create a new agent with the given name. Returns the new agent's handle. |
| `(register name)` | 1 `bytes` (`name`) | 1 `handle` (self) | Register this agent under `name` with the intent dispatcher. Returns the agent's own handle. Must be the first instruction executed by the agent. |

### 4.4 Error Codes

| Code | Name | Value | Description |
|------|------|-------|-------------|
| `ERR_OK` | Success | `0` | Intent delivered successfully. |
| `ERR_UNKNOWN_RECEIVER` | Unknown receiver | `1` | No agent registered under the target handle. |
| `ERR_INBOX_FULL` | Inbox full | `2` | Receiver's intent queue is at capacity. |
| `ERR_CAPABILITY_DENIED` | Capability denied | `3` | Sender lacks the capability to send to this receiver. |
| `ERR_HEAP_EXHAUSTED` | Heap exhausted | `4` | Agent heap is full and cannot allocate requested bytes. |

---

## 5. Execution Model

### 5.1 Stack Machine

Quarks-IR executes on a **tagged stack machine**. One stack per agent, no shared stacks.

| Parameter | Value |
|-----------|-------|
| Maximum stack depth | 1024 slots |
| Slot size | 16 bytes (8-byte tag + 8-byte value) |
| Tag: `i64` | `0x01` |
| Tag: `bytes` | `0x02` |
| Tag: `handle` | `0x03` |

**Bytes representation:** bytes values on the stack are fat pointers (pointer + length). The actual byte data lives in the agent's heap (Section 5.3). Copying a bytes value on the stack (via `dup`) copies the fat pointer, not the data. Bytes are immutable; no aliasing hazards arise.

**Stack overflow:** pushing beyond 1024 slots triggers a runtime trap `STACK_OVERFLOW`. The agent is terminated.

**Stack underflow:** the validator (Section 6) ensures at compile time that no instruction pops from an empty stack. Stack underflow should not occur at runtime. If it does (validator bug), the executor traps with `STACK_UNDERFLOW`.

**No frame layout in v0.1.** The stack is flat. No local variable slots, no call frames. Function definitions and subroutine calls are deferred to v0.2+ (Section 9).

### 5.2 Determinism

- **Sequential execution.** Instructions within a single agent execute in program order. No out-of-order execution at the IR level.
- **`recv` blocks without timeout in v0.1.** An agent that calls `recv` with no pending intents is descheduled until an intent arrives. There is no timeout, no poll, no non-blocking receive. Timeout semantics are deferred to v0.2+ (Section 9).
- **Inter-agent ordering.** When two agents send to the same receiver, the delivery order is not guaranteed. The receiver must not depend on ordering from distinct senders.
- **No shared memory between agents in v0.1.** Each agent has its own heap. Cross-agent communication is exclusively via `send`/`recv`.

### 5.3 Agent Heap

| Parameter | Value |
|-----------|-------|
| Heap size per agent | 64 KiB in v0.1 |

Bytes literals in the IR program are copied into the agent's heap at execution time. The stack holds a fat pointer (heap offset + length) to the copied data. Heap memory is not garbage-collected in v0.1; heap exhaustion is a runtime trap `HEAP_EXHAUSTED`.

---

## 6. Validator Requirements

The validator runs before execution. A program that fails validation is never executed. The validator produces structured errors (Section 7).

### 6.1 Checks

| # | Check | Error code | Description |
|---|-------|------------|-------------|
| 1 | Structural validity | `PARSE_ERROR` | Input must be a well-formed S-expression (balanced parentheses, valid atoms). |
| 2 | Unknown instruction | `UNKNOWN_INSTRUCTION` | The first symbol of every list must be a defined instruction name. |
| 3 | Argument count | `ARGUMENT_COUNT_MISMATCH` | Each instruction has a fixed arity. Mismatches are rejected. |
| 4 | Type checking | `TYPE_MISMATCH` | Abstract stack simulation: the validator tracks the type of every stack slot through the program and rejects type mismatches against instruction signatures. |
| 5 | Stack balance | `STACK_BALANCE_ERROR` | `loop` bodies must leave the stack in the same state they found it. `if` branches must produce the same stack effect. A program's final `return` must have exactly 1 value on the stack. |
| 6 | Bytes size | `BYTES_TOO_LARGE` | Bytes literals exceeding 16 KiB are rejected. |
| 7 | Invalid handle | `INVALID_HANDLE` | `@0` (null handle) passed as argument to instructions that expect a non-null handle (e.g., `send`). |
| 8 | Reserved symbols | `RESERVED_SYMBOL` | Symbols with an underscore prefix (`_foo`, `_kernel`) are reserved for kernel-internal use. Agent code must not define or reference them. |

### 6.2 Validation Scope

The validator operates on a single agent's IR in isolation. It does not validate cross-agent intent compatibility (that remains the receiver's responsibility).

---

## 7. LSP Error Schema

Validator errors are reported as JSON (the control plane). The IR data plane uses S-expressions; the error reporting plane uses JSON. This separation is intentional: errors are consumed by tooling (editors, CI), not by the kernel executor.

### 7.1 Base Schema

```json
{
  "error_code": "TYPE_MISMATCH",
  "message": "Human-readable description of the error",
  "source_location": {
    "list_path": [0, 2, 1]
  }
}
```

`list_path` is an array of zero-based indices tracing the path through nested S-expression lists to the offending node. Example: `[0, 2, 1]` means "root list, third element, second sub-element."

### 7.2 Error-Specific Fields

| Error code | Additional fields |
|------------|-------------------|
| `TYPE_MISMATCH` | `instruction` (string), `argument_index` (integer), `expected` (string), `actual` (string) |
| `ARGUMENT_COUNT_MISMATCH` | `instruction` (string), `expected_count` (integer), `actual_count` (integer) |
| `STACK_BALANCE_ERROR` | `context` (string: `"loop"`, `"if"`, or `"program"`), `stack_depth_start` (integer), `stack_depth_end` (integer) |

### 7.3 Multi-Error Reporting

Default behavior: first error only. The validator stops at the first error and reports it.

Configurable: all-errors mode returns an array:

```json
{
  "errors": [
    { "error_code": "...", "message": "...", "source_location": { "list_path": [...] } },
    { "error_code": "...", "message": "...", "source_location": { "list_path": [...] } }
  ]
}
```

---

## 8. Examples

### 8.1 Minimal: Addition

```lisp
(add 1 2)
```

Result: `3` on the stack (type `i64`).

### 8.2 Return a Value

```lisp
(return (add 1 2))
```

Agent terminates with exit value `3`.

### 8.3 Intent Send

```lisp
(send @5 #x48656c6c6f)
```

Sends the bytes `48 65 6c 6c 6f` ("Hello" in ASCII) to the agent registered at handle `@5`. Pushes an `i64` error code onto the stack (`0` on success).

### 8.4 Send with Error Handling

```lisp
(if (send @5 #x48656c6c6f)
    (return 1)
    (drop))
```

Send "Hello" to `@5`. If the error code is nonzero (send failed), return `1`. If the error code is zero (success), drop it from the stack and continue.

### 8.5 Counter: Counting from 0 to 3 via Agent Memory

```lisp
(store 0 0)                       ; mem[0] = counter = 0
(loop
  (if (cmp (load 0) 3)            ; cmp returns -1, 0, or 1; consumed by if
    (store 0 (add (load 0) 1))    ; then (counter != 3): increment
    (break)))                     ; else (counter == 3): exit loop
(return (load 0))                 ; return final counter value (3)
```

Uses `store`/`load` on agent-local memory address `0` to maintain the counter. Each iteration: `cmp` compares the counter to 3. The `if` construct consumes the `cmp` result (per Section 4.1). When counter != 3, `cmp` returns `-1` or `1` (truthy), so the then-branch executes and increments. When counter == 3, `cmp` returns `0` (falsy), so the else-branch executes `break`. Both branches are stack-neutral (then does a `store` which consumes what `add` produces; else does `break` which is stack-neutral), satisfying Validator Check 5.

Note: previous versions of this example had stack-balance bugs caused by ambiguous `if` semantics. The `if` construct consumes its condition value before entering either branch (see Section 4.1), so branches do not need to account for a leftover condition on the stack.

### 8.6 Ping-Pong (Pseudocode)

The following is **illustrative pseudocode**, not validator-passing IR. It shows how the Stage-4 Ping-Pong beta scenario maps to Quarks-IR concepts. A fully elaborated, validator-passing version will be written as part of Stage 7 (executor implementation). In particular, nesting and sequencing below is simplified for readability and does not reflect the flat stack-machine execution model.

**Pong agent (pseudocode):**

```lisp
(register #x706f6e67)          ; register as "pong"
(loop
  (recv)                        ; stack: sender-handle, intent-type, payload
  (drop)                        ; drop payload
  (drop)                        ; drop intent-type
  (dup)                         ; duplicate sender-handle for send
  (send #x61636b)              ; send "ack" to duplicated handle
  (drop)                        ; drop error code
  (drop))                       ; drop original sender handle
```

**Ping agent:**

```lisp
(register #x70696e67)          ; register as "ping"
; ... resolve "pong" handle, send 5 pings, await 5 acks ...
; (full version deferred to Stage 7)
(return 0)
```

---

## 9. Open Questions for v0.2+

The following are explicitly out of scope for v0.1. They are recorded here to prevent re-discussion and to guide the v0.2 design cycle.

1. **Function definitions and subroutines.** v0.1 has no `(defun)` or `(call)`. All code is a flat instruction sequence. Adding functions requires call frames on the stack and a return-address convention.
2. **Float types.** IEEE 754 `f64` is the likely addition. Requires new arithmetic instructions (`fadd`, `fsub`, etc.) and a new tag (`TAG_F64 = 0x04`).
3. **String encoding helpers.** Utility instructions for UTF-8 validation, byte-to-integer conversion, and similar. Currently the agent's responsibility.
4. **Timeouts on `recv`.** `(recv-timeout ticks)` that returns a timeout indicator instead of blocking forever. Required for agents that must act on absence of input.
5. **Async intent send.** `(send-async to payload)` that does not block for delivery confirmation. Required for fire-and-forget patterns.
6. **Capability checks.** `(has-capability cap-id)` or equivalent. Blocked on the capability system which does not yet exist.
7. **Debug metadata.** Source-location mappings from IR instructions back to the source language. Required for source-level debugging of Quarks programs.
8. **Interrupt handling.** Mechanism for an agent to register as an interrupt handler (timer, keyboard, network). Currently only kernel-resident Rust code can handle interrupts.
