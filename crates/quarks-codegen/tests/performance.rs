// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stage 10 MP6: Host-side Performance Regression Suite.
//!
//! Compares interpretation time vs compilation pipeline time for
//! representative Quarks programs using `std::time::Instant`. Zero
//! external dependencies (ADR-002 strict).
//!
//! # Why compile-time, not execute-time?
//!
//! Host-side execution of AOT-compiled code requires mmap+mprotect
//! with PROT_EXEC, which is platform-specific and adds libc deps.
//! For Stage 10, host-side bench measures pipeline phases (parse,
//! type-check, codegen) — these become Stage 11+ Boot-LLM relevant
//! when LLM-generated Quarks code drives the compile pipeline.
//!
//! Boot-path bench (kernel/src/aot.rs) covers actual execution-time
//! comparison (rdtscp loop). The two domains together establish the
//! Stage 10 baseline per ADR-026 MP6 mandate.

use std::time::Instant;

const ITERATIONS_PER_PROGRAM: u32 = 1_000;

/// Bench programs covering increasing complexity.
fn bench_programs() -> Vec<(&'static str, &'static str)> {
    vec![
        ("trivial", "42"),
        ("arith", "(add (mul 2 3) (sub 5 1))"),
        (
            "fn-call",
            "(program (fn main () i64 (add 1 2)) (call main))",
        ),
        (
            "param",
            "(program (fn add (i64 i64) i64 (add %0 %1)) (call add 10 20))",
        ),
    ]
}

#[test]
fn perf_compile_pipeline_baseline() {
    println!();
    println!("Stage 10 MP6: Host-Side Compilation Pipeline Bench");
    println!("  Iterations per program: {}", ITERATIONS_PER_PROGRAM);
    println!("  Mode: std::time::Instant, single-threaded, no warmup");
    println!();
    println!(
        "  {:<10} | {:>10} | {:>10} | {:>10} | {:>10}",
        "program", "parse", "typecheck", "compile", "total"
    );
    println!(
        "  {:-<10}-+-{:->10}-+-{:->10}-+-{:->10}-+-{:->10}",
        "", "", "", "", ""
    );

    for (name, src) in bench_programs() {
        // Parse phase
        let parse_start = Instant::now();
        let mut last_ast = None;
        for _ in 0..ITERATIONS_PER_PROGRAM {
            let ast = quarks_validator::parse(src)
                .unwrap_or_else(|e| panic!("parse failed for {}: {:?}", name, e));
            last_ast = Some(ast);
        }
        let parse_total = parse_start.elapsed();
        let parse_per_iter = parse_total / ITERATIONS_PER_PROGRAM;
        let ast = last_ast.expect("parse loop produced no ast");

        // Type-check phase
        let tc_start = Instant::now();
        for _ in 0..ITERATIONS_PER_PROGRAM {
            quarks_validator::type_check(&ast)
                .unwrap_or_else(|e| panic!("type-check failed for {}: {:?}", name, e));
        }
        let tc_total = tc_start.elapsed();
        let tc_per_iter = tc_total / ITERATIONS_PER_PROGRAM;

        // Compile phase
        let compile_start = Instant::now();
        for _ in 0..ITERATIONS_PER_PROGRAM {
            let _bytes = quarks_codegen::compile(&ast)
                .unwrap_or_else(|e| panic!("compile failed for {}: {:?}", name, e));
        }
        let compile_total = compile_start.elapsed();
        let compile_per_iter = compile_total / ITERATIONS_PER_PROGRAM;

        let total_per_iter = parse_per_iter + tc_per_iter + compile_per_iter;

        println!(
            "  {:<10} | {:>9}µs | {:>9}µs | {:>9}µs | {:>9}µs",
            name,
            parse_per_iter.as_micros(),
            tc_per_iter.as_micros(),
            compile_per_iter.as_micros(),
            total_per_iter.as_micros()
        );
    }
    println!();
    println!("Stage 10 MP6: host-side compile pipeline bench complete.");
    println!();
}

#[test]
fn perf_interpret_baseline() {
    println!();
    println!("Stage 10 MP6: Host-Side Interpreter Bench");
    println!("  Iterations per program: {}", ITERATIONS_PER_PROGRAM);
    println!();
    println!("  {:<10} | {:>15}", "program", "interpret/iter");
    println!("  {:-<10}-+-{:->15}", "", "");

    for (name, src) in bench_programs() {
        let ast = quarks_validator::parse(src).expect("parse");
        quarks_validator::type_check(&ast).expect("typecheck");

        let start = Instant::now();
        for _ in 0..ITERATIONS_PER_PROGRAM {
            let _ = quarks_interpreter::interpret(&ast)
                .unwrap_or_else(|e| panic!("interpret failed for {}: {:?}", name, e));
        }
        let total = start.elapsed();
        let per_iter = total / ITERATIONS_PER_PROGRAM;

        println!("  {:<10} | {:>14}µs", name, per_iter.as_micros());
    }
    println!();
    println!("Stage 10 MP6: host-side interpreter bench complete.");
    println!();
}

#[test]
fn perf_consistency_check() {
    // Sanity: interpret and compile must produce successful results
    // for all bench programs. This catches obvious failures before
    // anyone reads the bench output.
    for (name, src) in bench_programs() {
        let ast = quarks_validator::parse(src)
            .unwrap_or_else(|e| panic!("parse failed for {}: {:?}", name, e));
        quarks_validator::type_check(&ast)
            .unwrap_or_else(|e| panic!("type-check failed for {}: {:?}", name, e));
        let _ = quarks_codegen::compile(&ast)
            .unwrap_or_else(|e| panic!("compile failed for {}: {:?}", name, e));
        let _ = quarks_interpreter::interpret(&ast)
            .unwrap_or_else(|e| panic!("interpret failed for {}: {:?}", name, e));
    }
}
