# Implementation log

This log records verified work, command outcomes, blockers, and the next scoped
task. It is not a roadmap completion claim.

## 2026-10-23 — Phase 5 commit and PR preparation

User authorized committing the completed drain/report slice and opening a PR
before the next Phase 5 task. Confirmed the eight intended files, a clean
whitespace check and formatting check, and the existing implementation-session
verification below (47 default / 50 all-feature tests; both Clippy gates;
real-model drain and benchmark/report smoke). No runtime changes were made
while preparing publication. GitHub authentication works outside the network
sandbox. Publish this focused branch against main; leave merge and remaining
Phase 5 admission/burst work for the next step.

## 2026-10-23 — Phase 5 robustness and publishability slice

### Phase and scope

Continue the recorded next task on `feat/phase5-drain-and-reports` after the
completed Phase 4 gate. This is a partial Phase 5 milestone, not a final v1 gate.

### Plan

1. Add explicit, idempotent engine draining: stop admission atomically, finish
   accepted command/pending/active work under existing deadlines, release model
   state, and expose asynchronous completion. Wire Ctrl-C/SIGTERM into drain.
2. Test active and queued draining, rejection during drain, command/pending
   overload, slow-consumer cleanup, and HTTP availability behavior with mocks.
3. Add benchmark manifests with model checksum, context/configuration and source
   state, plus a report command rebuilt from raw samples, compatible with older
   artifacts without inventing missing provenance.
4. Rewrite README with architecture, exact limits/API, all four policies,
   Phase 4 negative results, setup, verification, and remaining Phase 5 work.
5. Run formatting, Clippy, workspace/native tests and available real-model smoke;
   record exact outcomes, limitations, changed files, and the next scoped task.

### Environment audit

Clean working tree at start; macOS arm64, Rust/Cargo 1.97.1, CMake 4.4.4,
Apple Clang 17. Local Qwen2.5-0.5B-Instruct Q4_K_M GGUF is available (469 MiB).
No new backend binding calls are planned; all FFI remains in niniserve-backend.

### Results

PASS for this scoped Phase 5 slice. Phase 5 overall and final v1 acceptance
remain open. No scheduling policy or backend FFI was changed, and no new
performance improvement is claimed.

### Changed files and behavior

- `crates/niniserve-engine/src/lib.rs`: idempotent `begin_drain()` closes
  submission under the sender mutex without needing command-queue capacity.
  Accepted work follows ordinary completion/overload/deadline/disconnect rules;
  async `drain()` observes backend destruction even while handles remain alive.
  Command processing per inference iteration is bounded. Request age now starts
  at submission, including command-queue residence.
- `crates/niniserve-server/src/main.rs`: Ctrl-C and Unix SIGTERM begin engine
  draining before Axum finishes serving in-flight responses; wait for backend
  destruction before normal exit.
- `crates/niniserve-server/src/lib.rs`: HTTP tests establish 503 before SSE
  headers on command overload and rejection/readiness behavior during drain.
- `crates/niniserve-bench/src/main.rs`: new runs reject nonempty output
  directories and record schema-1 manifests with model/executable/lockfile
  hashes, source state, actual context limits, configured queues and sampling,
  prompt fixtures/arrivals, and four raw-input checksums. `report RUN_DIR`
  verifies checksums and recomputes a Markdown report from CSV samples. Legacy
  runs are explicitly labeled; malformed nonempty numeric samples now error.
- `scripts/smoke_drain.py`: optional standard-library Python integration smoke
  with a supplied local model, configurable port/output directory, two active
  streams plus one pending request, SIGTERM, terminal SSE and exit checks.
- `README.md`: complete status/architecture/setup/API/limits/scheduler/benchmark
  rewrite, with the Phase 4 negative result and remaining Phase 5 work.
- `docs/BENCHMARKING.md`: manifest schema, checksum/report behavior, legacy
  provenance, and measurement limitations.

### Deterministic/mock checks

```text
cargo fmt --all -- --check
PASS.

cargo clippy --workspace --all-targets -- -D warnings
PASS.

cargo test --workspace
PASS — 47 unit tests, 0 failed; doc tests passed (0 tests).

cargo clippy --workspace --all-targets --all-features -- -D warnings
PASS.

cargo test --workspace --all-features
PASS — 50 unit tests, 0 failed; doc tests passed (0 tests).

git diff --check
PASS.
```

New checks cover in-flight/queued drain, repeated/concurrent drains, backend
lifetime with retained handles, rejection after drain begins, pending overload,
slow-consumer abandonment/slot reuse, submission-based deadlines, HTTP overload,
raw-sample report regeneration, legacy provenance, malformed samples, and
checksum tampering. One initial full test run exposed a pre-existing race in
`an_active_request_can_be_cancelled_explicitly`: the fast mock completed before
cancellation was submitted. Replaced its timing assumption with the existing
execution gate, then reran the complete default and all-feature gates above.

### Real-model evidence

```text
cargo build --release -p niniserve-bench -p niniserve-server --features niniserve-server/llamacpp
PASS.

python3 scripts/smoke_drain.py --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --output results/phase5/drain-verified-state
PASS outside sandbox — see server-drain-summary.json and server-drain.log.

target/release/niniserve-bench run --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --scheduler adaptive --workload W1 --repeats 1 --output results/phase5/adaptive-w1
PASS outside sandbox — four warmup and 4/4 measured requests completed;
manifest.json, report.md, metadata.json, summary.json and three raw CSVs written.
Native output retained in results/phase5/adaptive-w1.log.

target/release/niniserve-bench report results/phase5/adaptive-w1
PASS — regenerated report after verifying all four input hashes. Independently
verified SHA-256 hashes with Python hashlib; model hash matches the pinned
74a4da8c...e7b93d7a9db fixture, and recorded sampling seed is 42.

target/release/niniserve-bench report results/phase4/adaptive-w2
PASS — legacy CSVs regenerated a report with an explicit absent-manifest notice.
```

The initial sandboxed server attempt was BLOCKED at Metal initialization
(`failed to create command queue` / `create llama context: null reference`).
Authorized execution outside the sandbox passed on Apple M2 with 25/25 layers
offloaded. The shutdown smoke observed both active requests emit text before
SIGTERM; both finished 128 fragments, and the queued request finished 16. Each
returned HTTP 200, finish_reason=length, and [DONE]; process exit was 0 and
Metal deallocated cleanly. Earlier runs of the same smoke also passed.

The new benchmark is a functional W1 smoke with one measured trial, not another
policy comparison. The manifest records dirty source and the executable hash;
its Git revision alone does not reproduce this uncommitted change. Raw results
and generated reports remain ignored under results/; no weights were added.

### Assumptions, limits, and next task

Draining does not interrupt synchronous backend calls. Existing request
timeouts are checked at step boundaries; an indefinitely blocked backend or
slow HTTP transport can delay process exit. A full pending queue is still
reported as an SSE error after headers; only command-queue overload is rejected
synchronously with HTTP 503. Scheduler observations retain only 4096 steps.
Model/backend portability and statistical performance conclusions remain
limited to recorded hardware/model evidence.

Next session: add capacity-aware admission reservations before SSE headers and
W4 burst measurements, proving bounded outstanding work and deterministic
HTTP overload while preserving cleanup and streaming. Prometheus metrics,
structured tracing, hard HTTP drain bounds, continuous trace export, repeated
policy automation/variance, and the final public performance report remain.

Suggested commit: `feat: add graceful engine drain and reproducible benchmark reports`.
Changes are on `feat/phase5-drain-and-reports`; no commit, push, PR, or merge
has been performed in this session.

## 2026-10-11 — Issue #14 Phase 4 adaptive-prefill plan

### Phase and scope

Phase 4 adaptive prefill control and real-model evaluation on
`feat/phase4-adaptive-controller`. This milestone tests the controller research
question; it does not begin later scheduling, distributed-serving, MTP, or UI
work, and it will not claim an improvement unless held-out measurements support
one.

### Plan

1. Add a validated, bounded EWMA/hysteresis controller behind the existing
   scheduler seam, with explicit startup/no-decoder behavior and a maximum-wait
   prefill fairness rule.
2. Develop it test-first against synthetic token-gap regimes, covering bounds,
   stability, recovery, repeated overload, and progress under decode load.
3. Measure server-side per-sequence decode token gaps in the exclusive-owner
   engine and expose controller configuration, decisions, and current state in
   structured observations and the debug endpoint.
4. Extend the benchmark with W5, repeated trials, and a dependency-free
   trajectory plot tying selected chunks to observed decode gaps.
5. Freeze one documented adaptive configuration, then compare it on the local
   Qwen GGUF against fixed chunks 64, 128, 256, and 512 across W2, W3, and W5.
6. Record exact commands, raw results, limitations, and negative findings;
   finish formatting, Clippy, tests, PR review, CI, and merge only when clean.

Tracking issue: [#14](https://github.com/nihalmaddala/NiniServe/issues/14).

### Result

PASS for the Phase 4 research gate, with a negative performance result. The
bounded adaptive controller executed on real inference, changed actual prefill
budgets, emitted reproducible decision traces, and completed every request.
The frozen configuration did **not** outperform the fixed controls, so no
adaptive speedup or superiority claim is made.

### Implemented

- Added `AdaptiveScheduler` with a constant-size decode-gap EWMA, low/high
  hysteresis, multiplicative decrease, additive increase, validated 32–512
  token bounds, an explicit 128-token startup value, and full safe prefill
  capacity when no decoder is active.
- Added a 500 ms admitted-prefill max-wait override. It reserves bounded
  prefill work under saturated decode load and has deterministic unit coverage.
- Measured the maximum server-side per-sequence sampled-token-ready gap after
  each backend step. The feedback signal is independent of HTTP/network timing;
  client-observed ITL remains a separate reported metric.
- Extended bounded engine observations and `GET /debug/engine` with selected
  and next chunk budgets, raw gap, EWMA, oldest active-prefill wait, plan
  rationale, and feedback decision.
- Exposed every controller parameter through the server and benchmark CLIs.
- Added W5, `--repeats`, scheduler-state reset between measured trials, queue
  wait summaries, richer scheduler CSV data, and a dependency-free `plot`
  command that writes `adaptive_trajectory.svg`.

The public scheduler seam remains the test seam: synthetic `StepObservation`
values drive controller state, and subsequent `plan` calls reveal the selected
budget and fairness behavior. Backend-specific and unsafe code remains confined
to `niniserve-backend`.

### Frozen real-model experiment

Release runs used commit `8df97ae0a6272bbf8dfc9c80691fe8205d3d6c91`,
`llama-cpp-2 = 0.1.159`, the ignored Qwen2.5-0.5B-Instruct Q4_K_M model
(SHA-256 `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`),
and Apple M2 Metal with 25/25 layers offloaded. Context limits remained
`n_ctx=8192`, `n_ctx_seq=2048`, `n_batch=512`, `n_ubatch=128`, and
`n_seq_max=4`; model loads took 378–796 ms.

The adaptive configuration was frozen before the final matrix: target decode
gap 75 ms, chunk min/max/initial 32/512/128, increase 32, decrease factor 0.5,
EWMA alpha 0.2, low/high thresholds 0.85/1.10, and admitted-prefill max wait
500 ms. Each cell used one workload-shaped warmup followed by three measured
trials; scheduler feedback state was reset before every measured trial. All 285
measured requests completed.

| Policy | Workload | Finished | TTFT p50/p95 ms | ITL p50/p95 ms | Queue p95/max ms | Window ms | Output tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed 64 | W2 | 9/9 | 224 / 1446 | 27 / 89 | 43 / 43 | 10899 | 70.47 |
| Fixed 128 | W2 | 9/9 | 135 / 1117 | 27 / 124 | 64 / 64 | 10961 | 70.07 |
| Fixed 256 | W2 | 9/9 | 102 / 834 | 25 / 31 | 27 / 27 | 9455 | 81.23 |
| Fixed 512 | W2 | 9/9 | 106 / 896 | 25 / 33 | 37 / 37 | 9593 | 80.06 |
| Adaptive | W2 | 9/9 | 98 / 1441 | 27 / 57 | 25 / 25 | 10310 | 74.49 |
| Fixed 64 | W3 | 30/30 | 1711 / 5215 | 30 / 102 | 3672 / 3750 | 21105 | 77.33 |
| Fixed 128 | W3 | 30/30 | 1564 / 4554 | 28 / 136 | 3453 / 3495 | 19258 | 84.74 |
| Fixed 256 | W3 | 30/30 | 1578 / 4453 | 28 / 50 | 3571 / 3668 | 18849 | 86.58 |
| Fixed 512 | W3 | 30/30 | 1691 / 4291 | 28 / 45 | 3403 / 3551 | 18397 | 88.71 |
| Adaptive | W3 | 30/30 | 1894 / 5296 | 28 / 72 | 3416 / 3436 | 20716 | 78.78 |
| Fixed 64 | W5 | 18/18 | 1452 / 5152 | 31 / 104 | 3641 / 3641 | 23958 | 72.13 |
| Fixed 128 | W5 | 18/18 | 982 / 4183 | 29 / 140 | 3011 / 3011 | 21224 | 81.42 |
| Fixed 256 | W5 | 18/18 | 997 / 4119 | 29 / 48 | 3214 / 3214 | 20846 | 82.89 |
| Fixed 512 | W5 | 18/18 | 884 / 3809 | 29 / 46 | 3002 / 3002 | 20852 | 82.87 |
| Adaptive | W5 | 18/18 | 1072 / 5450 | 30 / 82 | 3202 / 3202 | 24232 | 71.31 |

Adaptive selected the full 32–512 range. Max-wait planning fired 47 times in
W2, 105 times in W3, and 179 times in W5. The controller often increased
toward large chunks because most observed gaps were below 63.75 ms, then
reacted only after a costly gap crossed the 82.5 ms high threshold. That lag,
plus frequent fairness overrides, is consistent with the weaker W3/W5 tail
TTFT and throughput. This is an observed mechanism, not a causal proof.

The 500 ms fairness threshold applies only to prefills already occupying one of
four backend sequence slots. Maximum observed active-prefill ages were 1.349 s,
5.521 s, and 5.340 s for W2/W3/W5 because an override is checked only at step
boundaries and long backend calls are non-preemptive. Requests waiting outside
the active set can also exceed 500 ms; Phase 4 does not solve admission
fairness. Aggregate three-trial samples on one machine are not confidence
intervals and do not generalize to other models or hardware.

### Exact experiment commands

```text
cargo build --release -p niniserve-bench

For workload in W2 W3 W5 and chunk in 64 128 256 512:
target/release/niniserve-bench run --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --scheduler fixed-chunk --prefill-chunk-tokens CHUNK --workload WORKLOAD --repeats 3 --output results/phase4/fixedCHUNK-workload

For workload in W2 W3 W5:
target/release/niniserve-bench run --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --scheduler adaptive --workload WORKLOAD --repeats 3 --output results/phase4/adaptive-workload
target/release/niniserve-bench plot results/phase4/adaptive-workload
```

Raw JSON, CSV, logs, and SVGs remain intentionally ignored under
`results/phase4/`.

### Next task

Stop after Phase 4. The smallest Phase 5 slice is robustness and publishability:
add graceful in-flight drain semantics, explicit overload/load-shedding tests,
and a reproducible benchmark manifest/report command without changing the
scheduler policy or claiming performance gains.

## 2026-10-10 — Issue #12 Phase 3 baseline-scheduler plan

### Phase and scope

Phase 3 baseline scheduler policies and measurement on
`feat/phase3-baseline-schedulers`. This milestone establishes repeatable
non-adaptive comparisons; it does not implement the adaptive controller or
claim a performance improvement.

### Plan

1. Add a pure scheduler module whose public seam accepts an immutable engine
   snapshot and token budget and returns a deterministic execution plan.
2. Use red-green slices to implement request-at-a-time FCFS, decode-priority,
   and fixed chunked-prefill policies, including deterministic ordering and
   explicit starvation semantics.
3. Integrate selected plans into the exclusive-owner engine so policy changes
   alter actual multi-sequence batch membership and prefill chunk sizes without
   exposing backend details outside `niniserve-backend`.
4. Add structured scheduler-step observations and server configuration for the
   policy and fixed-prefill chunk size.
5. Add W0–W3 workload generation, machine-readable JSON/CSV artifacts, and a
   summary command using measured values only.
6. Run policy experiments with the ignored Qwen GGUF, record mock and real-model
   evidence separately, and complete formatting, Clippy, tests, PR, and CI.

Tracking issue: [#12](https://github.com/nihalmaddala/NiniServe/issues/12).

### Result

PASS — policy selection changes the actual engine plans, fixed chunking bounds
total prefill work per step, and all three policies completed the real W0–W3
matrix. Adaptive scheduling remains out of scope.

### Implemented

- Added `niniserve-scheduler`, a pure deterministic module with
  `Scheduler::plan(EngineView, StepBudget) -> SchedulePlan` and an observation
  hook reserved for feedback-driven policies.
- Added request-at-a-time head-of-line FCFS, strict decode-priority, and fixed
  chunked-prefill with a configurable global per-step prefill budget.
- Replaced the engine's hard-coded one-token policy with scheduler-produced
  multi-sequence plans and cursor-based multi-token prefill chunks.
- Kept logits requests on each sequence's last prompt token and routed sampled
  output by explicit sequence ID and evaluated position.
- Added a bounded 4096-step observation ring and compact traces containing
  policy, sequence IDs, position ranges, phases, token counts, logits
  membership, scheduler time, and backend time.
- Added server flags `--scheduler` and `--prefill-chunk-tokens`.
- Added `niniserve-bench` with real tokenizer-measured W0–W3 fixtures,
  workload-shaped warmup, JSON/CSV output, and a summary command reporting
  nearest-rank TTFT/ITL/E2E percentiles, counts, and elapsed-window rates.
- Made every non-EOG sampled token observable to the benchmark while keeping
  incomplete UTF-8 bytes out of SSE text records.

Strict decode-priority intentionally has no fairness guard: an indefinitely
saturated decode budget can starve prefill. FCFS is intentionally weak and
head-of-line blocking is expected. These limitations are baseline behavior,
not production recommendations.

### Deterministic and integration evidence

```text
cargo test -p niniserve-scheduler
PASS — FCFS head-of-line selection, decode-first ordering, documented prefill
starvation, fixed total chunk bounds, deterministic tie breaking, and policy
parsing.

cargo test -p niniserve-engine
PASS — existing lifecycle/isolation/cancellation tests plus actual backend-plan
tests for fixed prefill chunks and FCFS request-at-a-time behavior.

cargo test -p niniserve-server --all-features
PASS — HTTP behavior and scheduler CLI parsing.

target/release/niniserve-bench summary results/phase3/fixed-w2
PASS — regenerated summary.json from raw request and token-gap CSV files.
```

### Real-model baseline evidence

Release runs used commit `7d7c7274144bbb4d0d82c4c791ca9c26c80cac5b`,
`llama-cpp-2 = 0.1.159`, the ignored Qwen2.5-0.5B-Instruct Q4_K_M fixture, and
Apple M2 Metal with 25/25 layers offloaded. The context reported `n_ctx=8192`,
`n_ctx_seq=2048`, `n_batch=512`, `n_ubatch=128`, and `n_seq_max=4`. Model load
was measured separately at 390–922 ms. Each measured workload followed an
identical workload-shaped warmup.

| Policy | Workload | Finished | TTFT p50/p95 ms | ITL p50/p95 ms | Window ms | Output tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FCFS | W0 | 3/3 | 674 / 2069 | 23 / 39 | 7648 | 25.10 |
| FCFS | W1 | 4/4 | 1794 / 4948 | 21 / 34 | 6332 | 40.43 |
| FCFS | W2 | 3/3 | 2527 / 5592 | 21 / 32 | 6833 | 37.47 |
| FCFS | W3 | 10/10 | 5845 / 15456 | 22 / 48 | 16905 | 32.18 |
| Decode priority | W0 | 3/3 | 558 / 2424 | 23 / 37 | 8203 | 23.41 |
| Decode priority | W1 | 4/4 | 170 / 170 | 43 / 58 | 2947 | 86.87 |
| Decode priority | W2 | 3/3 | 106 / 994 | 29 / 76 | 4126 | 62.05 |
| Decode priority | W3 | 10/10 | 1663 / 5769 | 33 / 83 | 8965 | 60.68 |
| Fixed 128 | W0 | 3/3 | 616 / 2449 | 22 / 32 | 7588 | 25.30 |
| Fixed 128 | W1 | 4/4 | 351 / 616 | 39 / 79 | 3408 | 75.12 |
| Fixed 128 | W2 | 3/3 | 133 / 1138 | 33 / 133 | 4035 | 63.44 |
| Fixed 128 | W3 | 10/10 | 2053 / 5926 | 38 / 154 | 8408 | 64.70 |

All 60 measured requests completed with 0 failures. W0 prompt lengths were
exactly 64, 512, and 1908 model tokens. W2 traces showed two decode tokens
interleaved with each incoming long-prompt chunk. An artifact check found no
fixed-policy scheduler step above 128 total prefill tokens.

This is one run per cell on one machine. The numbers prove the harness and
policy behavior, not statistical significance or a general speedup claim. Raw
artifacts are intentionally ignored under `results/phase3/`; schemas and replay
commands are in `docs/BENCHMARKING.md`.

### Next task

Stop after Phase 3. The smallest Phase 4 issue is an adaptive prefill-budget
policy behind the existing `plan`/`observe` seam: use bounded, smoothed decode
gap feedback; prove controller step response with deterministic observations;
then compare it against multiple fixed chunk sizes on held-out W2/W3/W5 runs.

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
