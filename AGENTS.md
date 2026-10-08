# AGENTS.md — NiniServe coding-agent rules

This file governs any Codex, Work, or automated programming agent working in this repository.

## Mission

Build **NiniServe**, a lightweight single-node Rust LLM inference serving runtime with user-owned batching/scheduling and a measured adaptive prefill controller. The canonical engineering specification is `NINISERVE_MASTER_SPEC.md`. Read it **before** coding.

## Immediate scope

Unless the user explicitly requests a later milestone, implement **Phase 0 and Phase 1** first. Phase 0 includes a verified backend feasibility spike if a model is available. Do not prematurely implement adaptive scheduling, distributed prefill/decode, a frontend dashboard, or a custom physical KV allocator.

## Operating procedure

1. Inspect existing files, Git state, OS/architecture, Rust/CMake/clang availability and any accessible model weights before modifying code.
2. Write a short implementation plan in `docs/IMPLEMENTATION_LOG.md` and identify the specific phase/gate you're working on.
3. Verify exact `llama-cpp-2` / pinned llama.cpp API names against locally installed source or versioned docs, especially explicit sequence IDs, token positions, logits indexing, per-sequence cleanup, and sampling. Never invent method names.
4. Keep a **single exclusive model context owner** outside async HTTP workers. Keep backend FFI contained in `niniserve-backend`.
5. Implement a deterministic mock backend for architecture tests, but never equate it with real-model proof.
6. Make small, testable changes and run checks at every meaningful step.
7. Use local model weights only when supplied/available; never commit weights or secret data. Never assume real model tests passed in their absence.
8. Maintain `docs/IMPLEMENTATION_LOG.md` after each session with changes, command results, blockers, assumptions, and next task.
9. Preserve unrelated user changes. No destructive Git commands or force pushing. Prefer reversible, incremental edits.

## Required engineering standards

- `cargo fmt --all -- --check`; `cargo clippy --workspace --all-targets -- -D warnings`; `cargo test --workspace` when the environment permits.
- Pin backend dependencies and commit lockfile for the application.
- Explicit lifecycles, bounded channels, typed IDs, correct cancellation, guaranteed cleanup.
- Scheduler is independent, deterministic, and unit-testable.
- No unbounded queues or blocking synchronous inference calls in Tokio handlers.
- Never mix token/logit indices between requests.
- Only report measured speedups with raw workload/hardware/model info.
- Prefer clear abstractions over gratuitous multi-crate fragmentation. Comments should explain tricky correctness or backend constraints, not trivial syntax.

## On missing environment or uncertainty

If the runtime has no Apple GPU, GGUF model, or network: scaffold compileable mock components, mark real-inference tests **SKIPPED/BLOCKED**, and document exact user-side steps needed to validate on macOS. Do not silently replace low-level batching with calls to an existing llama.cpp HTTP server.

If llama-cpp bindings cannot expose the needed API: document the issue; implement the smallest safe wrapper or pinned FFI shim rather than making up unsupported calls.

## What to report after each milestone

- Files created/changed.
- What concretely works now, separately for mock and real model.
- Tests/checks executed, with PASS/FAIL/SKIP.
- Exact outstanding assumptions and blockers.
- Next specific engineering task, ideally achievable in one coding session.

## Definition of done for first assignment

A working Rust workspace and clear docs; a backend decision/feasibility spike; an engine/HTTP vertical slice with mock tests; and, where a real GGUF is available, successfully streamed local model generation. Do **not** claim actual concurrent batching or adaptive performance improvements at the Phase 1 gate.
