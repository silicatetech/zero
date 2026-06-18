# Zero Performance Baseline (Stage 10)

**Status:** Stage 10 MP6 baseline. This document establishes the
initial performance measurements for Zero V3 Pillar 1 (Maximum
Performance, foundational). Design mandate: "measurable, not
aspirational."

## Methodology

### Boot-path bench (bare-metal, Ring-0)

Located in `kernel/src/aot.rs` (`run_bench`) and the corresponding
interpreter loop in `kernel/src/main.rs`. Measures cycle counts using
`rdtsc` with a `cpuid` serialization barrier for both the interpreter
and AOT execution paths.

Both paths execute the same boot.ir program
`(program (fn main () i64 (add 1 2)) (call main))` in tight loops of
1,000 iterations. AOT iterations have no heap allocation per call;
interpreter iterations allocate ~456 bytes/call (call-frame
`Vec<Value>`) which accumulates in the bump-arena. 1,000 iterations
× 456 bytes ≈ 456 KiB, well within the 2 MiB runtime arena budget.

Symmetric iteration counts ensure direct total-to-total comparison.
Measurement overhead (rdtsc ~20-30 cycles per call) is amortized
across the loops.

**Note on QEMU:** When running under QEMU, rdtsc values are derived
from QEMU's virtual time, not the host CPU's actual TSC. Absolute
cycle counts are not hardware-accurate. The *ratio* between AOT and
interpreter remains meaningful since both paths are emulated
identically.

### Host-side bench (compile pipeline)

Located in `crates/quarks-codegen/tests/performance.rs`. Uses
`std::time::Instant` (no external dependencies) to
measure parse, type-check, and codegen phases for representative
Quarks programs. Run with `cargo test -p quarks-codegen
--test performance -- --nocapture` to see output.

Host-side execution-time comparison would require mmap+mprotect with
PROT_EXEC and platform-specific libc bindings. Compile-time
measurement is sufficient for Stage 10 baseline. Stage 11+ Boot-LLM
work, where LLM-generated Quarks code drives the compile pipeline,
will reuse this measurement infrastructure.

### Bench programs

| Name | Program |
|------|---------|
| trivial | `42` |
| arith | `(add (mul 2 3) (sub 5 1))` |
| fn-call | `(program (fn main () i64 (add 1 2)) (call main))` |
| param | `(program (fn add (i64 i64) i64 (add %0 %1)) (call add 10 20))` |

Recursion is not yet supported by the codegen (nested-call-in-body
rejected); recursive bench programs are Stage 11+ material.

## Stage 10 Baseline

### Boot-path (QEMU x86_64)

| Path | Iterations | Total cycles | Per-iteration |
|------|-----------|--------------|---------------|
| Interpreter | 1,000 | 1,093,000 | ~1,093 |
| AOT | 1,000 | 48,000 | ~48 |

**Ratio: AOT is 22.7× faster than Interpreter.**

Note: QEMU rdtsc values are synthetic. The ratio is the meaningful
metric; absolute cycle counts should not be compared to real hardware.

### Host-side compile pipeline (1,000 iterations per program)

| Program | parse | typecheck | compile | total |
|---------|-------|-----------|---------|-------|
| trivial | 0µs | 0µs | 2µs | 2µs |
| arith | 1µs | 1µs | 5µs | 8µs |
| fn-call | 2µs | 3µs | 6µs | 13µs |
| param | 4µs | 4µs | 8µs | 17µs |

Compile pipeline cost scales linearly with program complexity.
Codegen dominates (~50% of total), followed by type-check (~25%)
and parse (~25%).

### Host-side interpreter (1,000 iterations per program)

| Program | interpret/iter |
|---------|----------------|
| trivial | <1µs |
| arith | <1µs |
| fn-call | <1µs |
| param | <1µs |

All interpreter times are sub-microsecond at 1,000-iteration
granularity. `std::time::Instant` resolution is insufficient to
differentiate at this scale. The boot-path rdtsc measurement (1,093
cycles/iter ≈ 0.3-0.5µs at ~3GHz) provides the higher-resolution data
point.

## Regression Detection (Stage 11+ direction)

Stage 10 establishes the baseline. Future stages may add:

- Stage 11+: Regression thresholds (e.g., "AOT must remain ≥10×
  faster than Interpreter on boot.ir loop") with CI failure on
  regression.
- Stage 12+: Multi-core TSC handling (Pillar 7 platform portability)
  if bench expands beyond single-core QEMU.
- Stage 13+: MCP-bridge serialization overhead measurement (V3 Z.260:
  "throughput bounded by memory bandwidth").
- Stage 19+: Cross-language handle-passing performance (V3 Z.511)
  when polyglot agents are introduced.

These are explicitly out of scope for Stage 10 MP6, which establishes
only the measurement *infrastructure* and the *initial baseline*.

## Design Context

The performance regression suite (V3 Phase 3 component per
ARCHITECTURE.md) compares AOT-execution time vs interpreter-execution
time on representative Quarks programs. Establishes Pillar 1
("Maximum Performance, foundational") as measurable, not aspirational.

The zero-copy handle architecture uses O(1) Vec-index by construction;
Stage 10 baseline does not include a dedicated handle micro-bench
(handle dereference is a single instruction). Stage 11+ may add
explicit handle-bench when handles flow through IPC paths.
