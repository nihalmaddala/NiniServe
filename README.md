# NiniServe

NiniServe is an experimental, single-node LLM inference runtime for Apple
Silicon. It will use llama.cpp for model execution while keeping request
admission, sequence scheduling, batch construction, streaming, and overload
behavior under Rust application control.

The research focus is latency-aware adaptive prefill scheduling: measuring
decode stalls and adjusting prefill work to study the tradeoff between
time-to-first-token, inter-token latency, throughput, and fairness.

## Project status

NiniServe has passed **Phase 0 backend feasibility** and now has the **Phase 1
single-request serving foundation**. The repository contains a real
`llama-cpp-2` adapter, one dedicated engine thread, bounded command/event
channels, and an Axum server that streams one request at a time over SSE. The
deterministic mock remains the default backend for ordinary tests.

The supplied Qwen2.5 0.5B GGUF has generated real output on Apple M2 Metal.
This is not yet a multi-request server: continuous batching, explicit
cancellation commands, timeouts, scheduler policies, and adaptive control are
later phases.

The canonical requirements and phase gates are in
[`NINISERVE_MASTER_SPEC.md`](NINISERVE_MASTER_SPEC.md). Coding agents must also
follow [`AGENTS.md`](AGENTS.md).

## Verified milestones and next boundary

1. The native toolchain and `llama-cpp-2 = 0.1.159` are exactly pinned.
2. The standalone real-model probe proved two explicit sequences in one
   context with correct logits routing and cleanup.
3. The Phase 1 service loads one GGUF, generates on its exclusive engine
   thread, streams SSE, and releases the sequence for the next request.
4. The next phase is actual multi-request continuous batching. It is not
   implemented or claimed here.

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
  implementation plus a feature-gated real llama.cpp adapter;
- `niniserve-engine`: the exclusive synchronous model owner with bounded
  commands and per-request events;
- `niniserve-server`: the Axum routes and feature-gated `niniserve` binary.

See [`docs/BACKEND_DECISION.md`](docs/BACKEND_DECISION.md) for the boundary
between mock, real-backend, and serving evidence.

## Run the local server

Build and start with a user-supplied GGUF:

```bash
cargo run -p niniserve-server --features llamacpp --bin niniserve -- \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --port 8080
```

In another terminal:

```bash
curl http://127.0.0.1:8080/healthz

curl -N http://127.0.0.1:8080/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"local-gguf","prompt":"The capital of France is","max_tokens":12,"temperature":0,"stream":true}'
```

The completion endpoint is intentionally OpenAI-inspired, not fully
OpenAI-compatible. Phase 1 requires `stream: true`, uses a fixed model ID of
`local-gguf`, and supports one active generation at a time.

## Scope boundary

NiniServe will own scheduling and logical request/sequence state. llama.cpp
will own tokenization, tensor execution, model-side KV storage, kernels, and
the physical memory implementation. This project does not claim to implement
paged attention, custom Metal kernels, or a production replacement for mature
serving systems.
