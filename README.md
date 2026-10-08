# NiniServe

NiniServe is an experimental, single-node LLM inference runtime for Apple
Silicon. It will use llama.cpp for model execution while keeping request
admission, sequence scheduling, batch construction, streaming, and overload
behavior under Rust application control.

The research focus is latency-aware adaptive prefill scheduling: measuring
decode stalls and adjusting prefill work to study the tradeoff between
time-to-first-token, inter-token latency, throughput, and fairness.

## Project status

NiniServe is in **Phase 0: repository and backend feasibility**. The repository
contains a compiling Rust workspace with model-independent protocol types and a
deterministic mock backend for validating sequence positions, routing, limits,
and cleanup. There is not yet a runnable server or a verified llama.cpp
integration. No real inference or performance claims have been established.

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

The current development platform is macOS on Apple Silicon with:

- Apple Clang 17
- CMake 4.4.4
- Rust 1.97.1 with rustfmt and Clippy, pinned by `rust-toolchain.toml`
- a user-supplied, appropriately licensed decoder-only GGUF model for real
  backend tests

Model files are local test inputs and must not be committed.
See [`models/README.md`](models/README.md) for the pinned Phase 0 test fixture,
download command, license source, and checksum verification.

### Install the toolchain

On Apple Silicon with Homebrew:

```bash
brew install rustup cmake
brew link --force rustup
rustup toolchain install 1.97.1 --profile minimal --component rustfmt,clippy
```

The repository's `rust-toolchain.toml` selects the pinned toolchain
automatically. Confirm the native build tools with:

```bash
rustc --version
cargo --version
cmake --version
clang --version
```

## Build and test

From the repository root:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

The workspace currently contains:

- `niniserve-protocol`: typed request/sequence IDs, request validation, and
  lifecycle transitions;
- `niniserve-backend`: the synchronous executor contract and deterministic mock
  implementation.

See [`docs/BACKEND_DECISION.md`](docs/BACKEND_DECISION.md) for the boundary
between verified mock behavior and the still-blocked real backend spike.

## Scope boundary

NiniServe will own scheduling and logical request/sequence state. llama.cpp
will own tokenization, tensor execution, model-side KV storage, kernels, and
the physical memory implementation. This project does not claim to implement
paged attention, custom Metal kernels, or a production replacement for mature
serving systems.
