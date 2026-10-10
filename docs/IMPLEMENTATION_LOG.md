# Implementation log

This log records verified work, command outcomes, blockers, and the next scoped
task. It is not a roadmap completion claim.

## 2026-10-10 — Issue #10 Phase 2 continuous-batching plan

### Phase and scope

Phase 2 multi-request continuous batching on
`feat/phase2-continuous-batching`. This milestone adds the smallest explicit
active registry, shared batch builder, and cancellation path needed to pass the
real two-request gate. Scheduler-policy abstractions, adaptive prefill control,
MTP/speculative decoding, and performance claims remain out of scope.

### Plan

1. Replace the request-at-a-time engine loop with a deterministic bounded
   pending queue and active registry that assigns dense backend sequence slots.
2. Build shared execution plans from explicit per-sequence positions while
   preserving exact logits routing, independent sampler state, and bounded
   per-request output delivery.
3. Add explicit cancellation/disconnect handling with exactly-once model-memory
   cleanup and safe dense-slot reuse; one cancelled request must not stop its
   peers.
4. Add deterministic mock tests first for shared-batch interleaving, isolation,
   cancellation, overload, cleanup, and reuse, then update HTTP integration.
5. Run two real concurrent SSE requests against the ignored Qwen GGUF, retain
   batch trace evidence, and record raw scheduling overhead without presenting
   it as a benchmark.
6. Run formatting, warnings-denied Clippy, workspace tests, and native-feature
   checks; update backend/user docs and complete PR/CI only when the gate passes.

Tracking issue: [#10](https://github.com/nihalmaddala/NiniServe/issues/10).

### Result

PASS — two real HTTP requests were simultaneously active in one llama.cpp
context, shared prefill/decode batches, produced isolated SSE output, and
completed normally. A separate real disconnect test cancelled one stream while
its peer continued, then proved the released dense slot was reusable.

### Implemented

- Replaced the request-at-a-time loop with a bounded FCFS pending queue and an
  active registry keyed by dense backend `SequenceId` slots.
- Added shared batch construction with one explicit-position token per active
  sequence per step. Prefill and decode tokens may coexist in one batch.
- Added independent request phases, sampler initialization, logits/result
  routing, UTF-8 streaming, EOG/max-token completion, and fail-closed errors.
- Added disconnect and explicit cancellation, terminal cancellation/timeout
  events, a 120-second serving timeout, and exactly-once cleanup.
- Added bounded overload rejection, per-sequence context validation, slot
  poisoning on cleanup failure, and ascending dense-slot reuse.
- Increased real serving configuration to `n_ctx=4096`, `n_ctx_seq=2048`,
  `n_batch=512`, `n_ubatch=128`, and `n_seq_max=2`.
- Added traces with sequence ID, position, prefill/decode phase, logits
  membership, planning microseconds, and backend microseconds.

This is a simple correctness policy, not a Phase 3 scheduler implementation,
adaptive prefill scheduling, or a performance optimization claim.

### Mock and integration evidence

```text
cargo test -p niniserve-engine -p niniserve-server
PASS — shared batches without output mixing; cancellation preserves a peer and
permits slot reuse; bounded overload; context rejection; timeout; teardown; and
HTTP streaming/validation/post-header errors.

cargo clippy --workspace --all-targets --all-features -- -D warnings
PASS after the Phase 2 implementation.
```

### Real-model evidence

```text
target/debug/niniserve \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 18080
PASS outside the restricted sandbox — Apple M2 Metal, 25/25 layers offloaded,
n_ctx=4096, n_ctx_seq=2048, n_batch=512, n_ubatch=128, n_seq_max=2.

Concurrent greedy requests, 12 tokens each
PASS — both HTTP 200; both emitted 12 isolated fragments, finish_reason
"length", and [DONE]. Observed outputs began " that it is a statically typed
language" and " be used to infer the probability of a hypothesis".

Shared trace
PASS — steps 0..13 contained both seq:0 and seq:1; steps 5..13 contained decode
tokens for both in each llama.cpp call. Planning observations were 1..6 us and
backend observations after pipeline compilation were 13,132..47,053 us. This
single smoke run is not a benchmark.

Disconnect/cancellation
PASS — seq:0 disconnected after 2 fragments; seq:1 continued to 24 fragments
and [DONE]. Trace dropped seq:0 after step 26 while seq:1 continued through step
44. A later request reused seq:0 at position 0 and completed with [DONE].

Ctrl-C shutdown
PASS — exit 0; Metal context deallocated cleanly.
```

### Remaining workflow

```text
cargo fmt --all -- --check
PASS.

cargo clippy --workspace --all-targets -- -D warnings
PASS.

cargo test --workspace
PASS — 20 unit tests; 0 failed; doc tests passed (0 tests).

cargo clippy --workspace --all-targets --all-features -- -D warnings
PASS — includes the native llama.cpp adapter and server binary.
```

### GitHub workflow

```text
b47c50f feat(engine): add multi-request continuous batching
004cb2d docs: record Phase 2 batching evidence

PR #11: https://github.com/nihalmaddala/NiniServe/pull/11
Rust checks: PASS in 42 seconds (run 38078900440, job 114291559029).
```

Stop after Phase 2; the next issue is the smallest Phase 3 baseline scheduler
and measurement slice, not adaptive control.

## 2026-10-09 — Issue #8 Phase 1 vertical-slice plan

### Phase and scope

Phase 1 single-request serving foundation on
`feat/phase1-streaming-vertical-slice`. This slice stops before multi-request
continuous batching, scheduler policies, adaptive control, or benchmarks.

### Plan

1. Deepen `niniserve-backend` just enough for generated token text, EOG checks,
   real sampler ownership, explicit pending-token positions, and guaranteed
   sequence-slot cleanup.
2. Add one dedicated synchronous engine worker with bounded command and output
   channels; HTTP code never owns or calls the executor directly.
3. Add a minimal Axum service with readiness-aware `GET /healthz` and validated,
   streaming-only `POST /v1/completions` SSE behavior.
4. Prove the vertical slice with deterministic mock integration tests, then run
   one local real-GGUF streaming smoke test on Metal.
5. Run formatting, warnings-denied Clippy, workspace tests, update this log with
   exact evidence, and complete the issue/PR/CI workflow only when green.

Tracking issue: [#8](https://github.com/nihalmaddala/NiniServe/issues/8).

### Result

PASS — the Phase 1 single-request vertical slice loads the real GGUF, streams
generated text through HTTP/SSE, terminates at the requested token limit,
releases sequence state, serves a second request, and shuts down cleanly.

### Implemented

- Extended the backend contract with per-sequence sampler initialization and
  owned token bytes/EOG results while retaining the deterministic mock.
- Added feature-gated `LlamaCppExecutor`, using `self_cell = 1.3.0` to contain
  the model/context lifetime entirely in `niniserve-backend`.
- Added `niniserve-engine`: one exclusive standard thread owns the synchronous
  executor; bounded Tokio channels carry commands and per-request events.
- Added explicit prompt/decode positions, pending-token progression,
  max-token/EOG completion, UTF-8 token-piece aggregation, and whole-sequence
  cleanup before terminal success.
- Added `niniserve-server`: readiness-aware `/healthz`, validated streaming-only
  `/v1/completions`, bounded SSE forwarding, CLI model/port parsing, and
  Ctrl-C shutdown.
- The last engine handle now closes and joins the worker. This guarantees the
  llama.cpp context and Metal resources finish teardown before process exit.

The fixed Phase 1 backend slot is `SequenceId(0)` and only one generation is
processed at a time. This milestone does not implement or claim continuous
batching, request interleaving, adaptive scheduling, explicit cancel commands,
timeouts, or OpenAI API completeness.

### Mock and integration evidence

```text
cargo test -p niniserve-engine -p niniserve-server
PASS — sequential requests streamed and reused the only mock backend slot;
HTTP health/streaming and validation tests passed.

Engine teardown regression
PASS — dropping the final cloned EngineHandle joins the worker and synchronously
drops its executor.
```

### Real-model evidence

```text
cargo check -p niniserve-server --features llamacpp
PASS.

cargo build -p niniserve-server --features llamacpp --bin niniserve
PASS.

target/debug/niniserve \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 18080
PASS outside the restricted sandbox — Apple M2 Metal, 25/25 layers offloaded,
n_ctx=2048, n_batch=512, n_ubatch=128, n_seq_max=1.

GET /healthz
PASS — 200 {"status":"ok","model_loaded":true}.

POST /v1/completions, prompt "The capital of France is", greedy, 12 tokens
PASS — 12 coherent non-empty SSE fragments, finish_reason "length", [DONE].
Observed text: " Paris. It is the largest city in Europe and the second".

Second sequential POST, prompt "2 + 2 =", greedy, 8 tokens
PASS — 200, 8 non-empty fragments, finish_reason "length", [DONE]. The second
request proves the single backend sequence slot was released and reused.
Warm observation: first SSE at 0.037 s; total 0.100 s; 126.708 non-empty
fragments/s after first SSE. This is a single smoke observation, not a benchmark
or a token-throughput performance claim.
```

The first Ctrl-C smoke ended with exit 134 because the then-detached engine
worker raced process teardown and llama.cpp asserted that a Metal residency set
was still populated. After adding final-handle channel closure plus worker join,
the repeated Ctrl-C test exited 0 and logged `ggml_metal_free: deallocating`.

### Remaining verification and next task

```text
cargo fmt --all -- --check
PASS.

cargo clippy --workspace --all-targets -- -D warnings
PASS — default mock-only workspace.

cargo test --workspace
PASS — 14 unit tests; 0 failed; doc tests passed (0 tests).

cargo clippy --workspace --all-targets --all-features -- -D warnings
PASS — includes the native llama.cpp adapter, real server binary, and probe.
```

The server integration suite also verifies `Cache-Control: no-cache`, stable
per-stream creation time construction, and an explicit SSE `error` event
without `[DONE]` when a backend error occurs after response headers.

### GitHub workflow

```text
2dd1e11 feat: add single-request streaming server
28a7ea8 docs: record Phase 1 serving evidence

PR #9: https://github.com/nihalmaddala/NiniServe/pull/9
Rust checks: PASS in 47 seconds (run 37964049334, job 113933803815).
```

After this gate, the next issue should be the smallest Phase 2 active registry
and real multi-request batch-builder slice with explicit cancellation and batch
traces. Do not add adaptive scheduling yet.

## 2026-10-09 — Phase 0 real-backend feasibility result

### Outcome

PASS — the real two-sequence llama.cpp gate passed twice with deterministic
per-sequence token IDs on the supplied Qwen2.5 GGUF and Apple M2 Metal. This is
a backend feasibility result, not a claim that an engine, HTTP service, or
continuous scheduler exists.

### Implemented

- Exactly pinned optional `llama-cpp-2 = 0.1.159`; matching sys crate is also
  `0.1.159` and `Cargo.lock` records the complete native dependency graph.
- Added a `llamacpp` feature and kept the mock-only default build unchanged.
- Added the backend-contained real probe and `examples/two_sequences.rs`.
- Added primary-source API research in `docs/BACKEND_API_RESEARCH.md` and
  finalized the Phase 0 decision in `docs/BACKEND_DECISION.md`.

### Exact source mapping and APIs

```text
llama-cpp-2 / llama-cpp-sys-2: 0.1.159
wrapper commit: 3cfdd729d65e35da407e5f820edf73201bfa54f6
vendored llama.cpp commit: 26394b4e6749a41c3633db040e0987500a5f7013

LlamaBatch::new; LlamaBatch::add; LlamaBatch::clear
LlamaContext::decode; get_logits_ith (via LlamaSampler::sample)
LlamaSampler::{chain_simple, top_k, temp, dist}
LlamaVocab::{tokenize, is_eog, detokenize}
LlamaContext::{kv_cache_seq_rm, kv_cache_seq_pos_max}
LlamaContext::{n_ctx, n_batch, n_ubatch}
LlamaContextParams::{with_n_ctx, with_n_batch, with_n_ubatch, with_n_seq_max}
```

### Real commands and evidence

```text
cargo check -p niniserve-backend --features llamacpp --example two_sequences
PASS — Rust and vendored native sources compiled on aarch64 macOS.

target/debug/examples/two_sequences \
  models/qwen2.5-0.5b-instruct-q4_k_m.gguf
PASS twice outside the restricted sandbox — Apple M2 Metal, 25/25 layers
offloaded, n_ctx=1024, n_ctx_seq=512, n_batch=512, n_ubatch=128.

Prefill batch 0: both sequence IDs with explicit positions.
Decode batches 1..11: both sequence IDs present in every batch.
Generated: 12 tokens per sequence using separate fixed-seed sampler chains.
Cleanup: PASS for both sequences; post-removal max position was -1.
Repeat: PASS; both 12-token ID vectors exactly matched the first run.
```

The initial sandboxed Metal attempt could not create a command queue and was
not counted. The first unsandboxed run with arbitrary IDs 101/202 was rejected,
revealing that this configuration requires dense backend IDs in `0..n_seq_max`;
the probe and decision record now enforce/document backend slots 0 and 1.

### Verification results

```text
cargo fmt --all -- --check
PASS.

cargo clippy --workspace --all-targets -- -D warnings
PASS — default mock-only workspace.

cargo test --workspace
PASS — 9 unit tests; 0 failed; doc tests passed (0 tests).

cargo clippy -p niniserve-backend --features llamacpp \
  --example two_sequences -- -D warnings
PASS — native probe and adapter module.
```

### GitHub workflow

Created focused issue
[#6](https://github.com/nihalmaddala/NiniServe/issues/6) with the Phase 0 scope
and acceptance evidence. Push, PR, CI, and merge outcomes are recorded only
after they occur.

### Next task

Open the smallest Phase 1 issue: build a single-request real `ModelExecutor`
adapter plus a single-owner worker and bounded Axum SSE vertical slice. Do not
add multi-request scheduling or adaptive behavior in that issue.

## 2026-10-08 — Phase 0 real-backend feasibility spike plan

### Phase and gate

Phase 0 backend feasibility on `spec/backend-feasibility-spike`. The gate is a
real GGUF run proving that one llama.cpp context can prefill, interleave, sample,
and explicitly clean up two independent sequences. Mock results do not satisfy
this gate.

### Plan

1. Verify a current exact `llama-cpp-2` release and its vendored llama.cpp API
   from primary, versioned source before adding the dependency.
2. Record the binding surface for explicit sequence IDs and positions, batch
   logits mapping, independent samplers, EOG detection, memory cleanup, and
   `n_ctx`/`n_batch`/`n_ubatch` limits.
3. Add the smallest feature-gated real adapter/probe wholly inside
   `niniserve-backend`, preserving the deterministic mock backend and ordinary
   CI behavior.
4. Run `two_sequences` twice against the ignored Qwen2.5 GGUF, retaining trace
   evidence and honestly marking the gate PASS or BLOCKED.
5. Run formatting, Clippy, and workspace tests; then update the backend decision
   and this log with exact versions, commands, results, limitations, and the
   smallest Phase 1 follow-up.

### Initial workflow note

The first `gh auth status` check reported a stale credential and no browser
surface was available, so the local branch was created before the issue. The
later authenticated API call succeeded and created issue
[#6](https://github.com/nihalmaddala/NiniServe/issues/6); no remote action is
reported unless independently verified.

## 2026-10-08 — Local GGUF fixture setup

Downloaded the official Apache-2.0 Qwen2.5-0.5B-Instruct Q4_K_M GGUF to the
Git-ignored `models/` directory from pinned Hugging Face revision
`9217f5db79a29953eb74d5343926648285ec7e67`.

```text
File size
PASS — 491400032 bytes, matching official metadata.

SHA-256
PASS — 74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db,
matching the official LFS digest.

File header
PASS — first four bytes are GGUF.

Git exclusion
PASS — .gitignore excludes the model weight.

Real inference
NOT RUN — the llama.cpp adapter has not been selected or implemented yet.
```

Added `models/README.md` with a reproducible pinned download and checksum
command. No model weight is committed.

## 2026-10-08 — CI checkout runtime follow-up

The first post-merge CI run passed but warned that `actions/checkout@v4`
targeted deprecated Node.js 20. Updated checkout to the immutable commit for
official release v7.0.1 (`3d3c42e5aac5ba805825da76410c181273ba90b1`),
which uses the current action runtime. This follow-up changes no Rust behavior.

## 2026-10-08 — Issue #2 implementation plan

### Phase and scope

Phase 0 repository/toolchain foundation on
`chore/rust-toolchain-workspace`.

1. Pin the installed Rust 1.97.1 toolchain with `rustfmt` and `clippy`.
2. Create two non-empty crates: model-independent protocol types and a backend
   execution contract with deterministic mock behavior.
3. Test request validation, batch limits, sequence isolation, token positions,
   deterministic output, and cleanup.
4. Add CI that runs the same formatting, linting, and test commands used
   locally.
5. Keep the real llama.cpp adapter and HTTP serving out of this change; record
   the GGUF-dependent feasibility gate as blocked.

### Installed and pinned toolchain

| Item | Verified version |
| --- | --- |
| rustup | 1.29.1 |
| Rust compiler | 1.97.1 (`aarch64-apple-darwin`) |
| Cargo | 1.97.1 |
| rustfmt | 1.9.0-stable |
| Clippy | 0.1.97 |
| CMake | 4.4.4 |
| Clang | Apple Clang 17.0.0 |

Homebrew installed `rustup` and `cmake`. The exact Rust 1.97.1 toolchain was
installed with the minimal profile plus `rustfmt` and `clippy` and is pinned in
`rust-toolchain.toml`.

### Implemented

- Added a Cargo workspace with `niniserve-protocol` and
  `niniserve-backend`.
- Added typed request and sequence IDs, pre-tokenization request validation,
  and explicit lifecycle transitions.
- Added a synchronous model-executor contract whose output owns sampled token
  data rather than exposing backend logit pointers.
- Added a deterministic mock executor with bounded batch/sequence capacity,
  explicit positions, per-sequence output routing, atomic validation, trace
  capture, and sequence cleanup.
- Added macOS CI using the same pinned toolchain and local quality gates.
- Added `docs/BACKEND_DECISION.md`, explicitly separating verified mock behavior
  from the unverified real llama.cpp adapter.

### Verification results

```text
cargo fmt --all -- --check
PASS — all workspace Rust sources match rustfmt 1.9.0-stable.

cargo clippy --workspace --all-targets -- -D warnings
PASS — both crates compile with no Clippy or compiler warnings.

cargo test --workspace
PASS — 9 unit tests passed; 0 failed; doc tests passed (0 tests).

git diff --check
PASS — no whitespace errors.

Real GGUF validation
BLOCKED — no compatible local model has been provided or found.
```

The first formatting check correctly failed on unformatted new sources; running
`cargo fmt --all` resolved it before the recorded passing gate above.

### Current limitations

- The mock tokenizer and sampled-token formula are deterministic test fixtures,
  not model behavior and not performance evidence.
- No llama.cpp version or Rust binding API has been selected or verified.
- The workspace does not yet contain an engine worker, HTTP server, scheduler,
  or native model adapter.

### Next task

Open a focused backend-spike issue and branch: inspect versioned
`llama-cpp-2`/llama.cpp source, select an exact dependency only after compiling
a minimal API probe, and obtain a compatible local GGUF for the two-sequence
feasibility test. If a model remains unavailable, record the real test as
BLOCKED and do not claim continuous batching.

## 2026-10-08 — Repository foundation and Phase 0 audit

### Scope

- Import the canonical specification and agent rules into the repository root.
- Establish a branch, commit, pull-request, and merge workflow.
- Record the initial development environment before selecting dependencies.
- Do not select an unverified llama.cpp binding or implement runtime code.

### Environment observed

| Item | Result |
| --- | --- |
| Host | macOS 15.5, arm64 Apple Silicon |
| Xcode developer directory | `/Applications/Xcode.app/Contents/Developer` |
| Clang | Apple Clang 17.0.0 |
| Rust compiler | BLOCKED — `rustc` not installed or not on `PATH` |
| Cargo | BLOCKED — `cargo` not installed or not on `PATH` |
| CMake | BLOCKED — `cmake` not installed or not on `PATH` |
| Local GGUF/GGML in repository | BLOCKED — none found |
| Git remote | PASS — `origin` points to the intended GitHub repository |
| Remote branches before bootstrap | None; the GitHub repository was empty |

### Commands and outcomes

```text
git status --short --branch
PASS — repository had no commits; agent pack was untracked.

git remote -v
PASS — fetch and push URL: https://github.com/nihalmaddala/NiniServe.git

git ls-remote --heads origin
PASS — remote was reachable and returned no branches.

uname -m; sw_vers
PASS — arm64, macOS 15.5.

rustc --version; cargo --version
BLOCKED — commands not found.

clang --version
PASS — Apple Clang 17.0.0 targeting arm64-apple-darwin24.5.0.

cmake --version
BLOCKED — command not found.

find . -type f \( -iname '*.gguf' -o -iname '*.ggml' \)
BLOCKED — no local model found within the repository.
```

### Decisions

- The first `main` commit contains only `AGENTS.md` and
  `NINISERVE_MASTER_SPEC.md`, creating a valid pull-request base.
- Subsequent work uses scoped branches and pull requests.
- Dependency versions and backend APIs will not be guessed while the Rust/native
  toolchain is unavailable.
- Real inference remains unverified; no mock or performance result exists yet.

### Blockers and assumptions

- Rust/Cargo and CMake must be installed before the workspace or native backend
  can be compiled and checked.
- A user-supplied compatible GGUF path is required to pass the real
  multi-sequence feasibility gate.
- GitHub branch protection may require remote settings after the first PR exists.

### Next task

Create `chore/rust-toolchain-workspace`: select and pin a stable Rust toolchain,
install/verify CMake, create the smallest compiling workspace with a meaningful
mock backend boundary, add CI, and record exact formatting/lint/test results.
