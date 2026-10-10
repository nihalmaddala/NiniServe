# NiniServe

NiniServe is an experimental, single-node LLM inference runtime for Apple
Silicon. It will use llama.cpp for model execution while keeping request
admission, sequence scheduling, batch construction, streaming, and overload
behavior under Rust application control.

The research focus is latency-aware adaptive prefill scheduling: measuring
decode stalls and adjusting prefill work to study the tradeoff between
time-to-first-token, inter-token latency, throughput, and fairness.

## Project status

NiniServe has passed **Phase 0 backend feasibility**, the **Phase 1 serving
foundation**, the **Phase 2 continuous-batching gate**, and the **Phase 3
baseline-scheduler gate**. The
repository contains a real `llama-cpp-2` adapter, one dedicated engine thread,
bounded command/event queues, and an Axum server that streams two active
requests while NiniServe constructs their shared execution batches. The
deterministic mock remains the default backend for ordinary tests.

The supplied Qwen2.5 0.5B GGUF has generated two isolated concurrent streams on
Apple M2 Metal. Batch traces prove both dense sequence IDs entered the same
prefill and decode calls; disconnect cancellation, a 120-second engine timeout,
and released-slot reuse are implemented. Three explicit non-adaptive scheduler
policies and a release-mode W0–W3 measurement harness now provide the baseline
for later adaptive work. Adaptive control remains unimplemented.

The canonical requirements and phase gates are in
[`NINISERVE_MASTER_SPEC.md`](NINISERVE_MASTER_SPEC.md). Coding agents must also
follow [`AGENTS.md`](AGENTS.md).

## Verified milestones and next boundary

1. The native toolchain and `llama-cpp-2 = 0.1.159` are exactly pinned.
2. The standalone real-model probe proved two explicit sequences in one
   context with correct logits routing and cleanup.
3. The Phase 1 service loads one GGUF, generates on its exclusive engine
   thread, streams SSE, and releases sequence state.
4. Phase 2 adds a bounded FCFS queue, two active dense backend slots, shared
   batch construction, independent streams/samplers, cancellation, timeouts,
   and slot reuse.
5. Phase 3 adds request-at-a-time FCFS, decode-priority, fixed chunked prefill,
   structured step observations, and real-model W0–W3 result artifacts.

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
- `niniserve-scheduler`: pure snapshot-to-plan baseline scheduling policies;
- `niniserve-server`: the Axum routes and feature-gated `niniserve` binary.
- `niniserve-bench`: release-mode real-model workload and summary commands.

See [`docs/BACKEND_DECISION.md`](docs/BACKEND_DECISION.md) for the boundary
between mock, real-backend, and serving evidence.

## Run the local server

Build and start with a user-supplied GGUF:

```bash
cargo run -p niniserve-server --features llamacpp --bin niniserve -- \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --scheduler fixed-chunk \
  --prefill-chunk-tokens 128 \
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
OpenAI-compatible. It requires `stream: true`, uses a fixed model ID of
`local-gguf`, and currently supports at most two active generations. Available
policies are `fcfs`, `decode-priority`, and `fixed-chunk`; the last accepts a
positive `--prefill-chunk-tokens` value.

See [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) for the W0–W3 workload
definitions, exact result schemas, release commands, and measurement caveats.

## Scope boundary

NiniServe will own scheduling and logical request/sequence state. llama.cpp
will own tokenization, tensor execution, model-side KV storage, kernels, and
the physical memory implementation. This project does not claim to implement
paged attention, custom Metal kernels, or a production replacement for mature
serving systems.
