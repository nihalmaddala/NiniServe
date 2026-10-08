# NiniServe

NiniServe is an experimental, single-node LLM inference runtime for Apple
Silicon. It will use llama.cpp for model execution while keeping request
admission, sequence scheduling, batch construction, streaming, and overload
behavior under Rust application control.

The research focus is latency-aware adaptive prefill scheduling: measuring
decode stalls and adjusting prefill work to study the tradeoff between
time-to-first-token, inter-token latency, throughput, and fairness.

## Project status

NiniServe is at the start of **Phase 0: repository and backend feasibility**.
The engineering specification is present, but there is not yet a runnable
server or a verified llama.cpp integration. No inference or performance claims
have been established.

The canonical requirements and phase gates are in
[`NINISERVE_MASTER_SPEC.md`](NINISERVE_MASTER_SPEC.md). Coding agents must also
follow [`AGENTS.md`](AGENTS.md).

## Planned first milestones

1. Audit and pin the Rust and native build toolchains.
2. Verify low-level llama.cpp APIs for explicit sequence IDs, token positions,
   logits association, independent sampling, and sequence cleanup.
3. Build a deterministic mock-backed engine skeleton.
4. Prove two independent sequences in one real model context when a local GGUF
   model is available.
5. Build the minimal single-request HTTP/SSE vertical slice.

Later scheduling and benchmark work begins only after the relevant phase gates
pass.

## Development workflow

All work after the initial specification import is developed on a scoped
branch, reviewed in a pull request, and merged only after its checks and phase
gate are understood. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for branch names,
commit style, pull-request expectations, and local verification commands.

## Local prerequisites

The intended development platform is macOS on Apple Silicon with:

- Xcode Command Line Tools / Clang
- CMake
- a pinned stable Rust toolchain (to be selected in the toolchain setup PR)
- a user-supplied, appropriately licensed decoder-only GGUF model for real
  backend tests

Model files are local test inputs and must not be committed. Until the backend
spike lands, setup and run commands would be speculative and are deliberately
not documented as working commands.

## Scope boundary

NiniServe will own scheduling and logical request/sequence state. llama.cpp
will own tokenization, tensor execution, model-side KV storage, kernels, and
the physical memory implementation. This project does not claim to implement
paged attention, custom Metal kernels, or a production replacement for mature
serving systems.

