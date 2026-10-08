# NiniServe — Master Engineering Specification

**Document status:** Implementation brief / source of truth for an AI coding agent  
**Project stage:** Greenfield; no implementation is asserted by this document  
**Target:** Rust, Apple Silicon/macOS, local GGUF inference using llama.cpp execution primitives  
**Design inspiration:** NVIDIA Dynamo's emphasis on inference scheduling and resource coordination; NOT a port, clone, or distributed deployment of Dynamo  
**Project identity:** An independently scheduled, latency-aware, single-node LLM inference runtime  
**Primary differentiator:** Feedback-driven adaptive prefill scheduling, evaluated against simple scheduling baselines  
**Last specification review:** October 2026

> **Instruction to Codex / Work:** Read this entire document and `AGENTS.md` before editing files. Treat this as a staged build, not a demand to implement everything at once. Begin with the repository audit, technical-spike decisions, and Phase 0/Phase 1 only. Make actual code compile, write real tests, and explicitly report blockers. No fabricated benchmarks, APIs, screenshots, output, or claims of inference working. Continue to later phases only when their gates pass and the user or project workflow authorizes it.

---

## 0. What we are building (read first)

NiniServe is a compact Rust program that accepts text-generation requests and drives an existing language model directly at the inference-execution layer. The model's tensor computations run in **llama.cpp / ggml**, preferably using **Metal** on Apple Silicon. The network service, request admission, active sequence registry, continuous-batch construction, scheduling policy, token-stream delivery, cancellation, and performance telemetry are implemented in **NiniServe**.

**Distinct from wrapping a model-serving HTTP API:** NiniServe should construct the actual model execution batches—controlling which sequence/token positions run at each step—rather than merely forwarding concurrent HTTP requests to a pre-existing server that independently chooses its batches.

**Distinct from building an LLM from scratch:** NiniServe does not train a model, implement transformer kernels, or claim to own physical GPU memory allocation performed by llama.cpp.

**Core experimental question:** Under a single constrained inference context, can dynamically shrinking or growing the prefill token budget in response to observed decode latency reduce p95 inter-token latency under mixed prefill/decode demand while preserving acceptable throughput, TTFT, and fairness?

**What success looks like:**

1. Start the server, load a small locally supplied GGUF model, and generate real text.
2. Stream output tokens to the client through an HTTP endpoint.
3. Serve several independent requests from a single model context with **real multi-sequence batching** and no cross-talk.
4. Switch among FCFS-like, decode-priority, fixed-chunk, and adaptive scheduling policies.
5. Prove each policy actually changes model execution batches using trace logs.
6. Collect reproducible latency and throughput data rather than relying on toy or fabricated results.
7. Publish an architecture explanation, documented limitations, and an honest comparison with upstream llama.cpp server where feasible.

---

## 1. Core constraints and scope control

### 1.1 Initial targets

- macOS + Apple Silicon is the first-class local deployment target.
- Use Rust for service, orchestration, schedulers, benchmarking harness, and most tests.
- llama.cpp supplies GGUF loading, tokenization, model execution, model-side KV state, and sampler primitives.
- HTTP: Axum and Tokio; streaming: server-sent events (SSE).
- Local only, **single process**, **single model**, and **one engine-owned llama.cpp context** for the first working version.
- Use an externally downloaded local **decoder-only GGUF** model as the v1 supported model class (e.g., a small instruction-tuned 0.5B–1.5B model compatible with the pinned backend). Do not hard-code a model download or assume a user has a particular model.
- Model/license provenance: users supply the model path; docs explain how to choose a properly licensed GGUF. Do not check model weights into Git.
- Configuration via explicit CLI flags and/or TOML, with validation and documented defaults.

### 1.2 Explicit non-goals for the first version

- CUDA/GPU clusters, Kubernetes, NIXL, RDMA, multi-node distribution, disaggregated prefill/decode workers.
- New transformer implementations, custom Metal kernels, attention kernels, quantization techniques, or fine-tuning.
- Writing a physical paged KV-cache allocator and pretending it controls backend blocks.
- Full vLLM/Dynamo or full OpenAI API compatibility.
- Agent frameworks, RAG, prompt personalization, web applications, persistent conversation database, authentication providers, or a dashboard frontend.
- Speculative decoding, multimodal models, tool use, embeddings, LoRA hot-swapping, grammar-constrained generation, distributed routing.
- ML-trained or RL-based scheduling in the initial adaptive scheduler.

### 1.3 Definition of ownership

NiniServe **owns**: request queue, admission policy, engine state machine, batch construction, per-request token scheduling and sampling state, sequence ID allocation, cancellation semantics, streaming, metrics, workload generation, adaptive controller.

llama.cpp **owns**: tensor execution, model KV tensor storage, Metal kernels/backend, internal physical memory details, vocabulary, tokenization implementation, sampling implementation where invoked through supported APIs.

**Never claim “NiniServe implements paged attention,” “NiniServe implements physical KV reuse,” or “NiniServe directly allocates GPU KV blocks” unless that behavior is truly independently implemented and experimentally verified.**

---

## 2. Technical feasibility gate — DO THIS BEFORE DEEP IMPLEMENTATION

This is the largest project risk. Do not spend a week writing an elaborate Rust server before proving control of a single inference context with two independent sequences.

### 2.1 Backend selection procedure

Preferred path: investigate a **pinned version** of the [`llama-cpp-2`](https://crates.io/crates/llama-cpp-2) Rust bindings and the matching llama.cpp C API.

Current research indicates `llama-cpp-2` has `LlamaBatch`, `LlamaContext::decode`, sequence-aware batch additions, and KV/memory sequence operations. **The agent must inspect actual installed crate/source API and compile a minimal probe; method names and safety constraints can change with versions.** For October 2026 context, version 0.1.159 was published October 7, 2026, but this is a research starting point, not a command to blindly depend on that version. Select a working exact version and commit `Cargo.lock`.

Use a thin, pinned llama.cpp C FFI adapter instead if the Rust wrapper does not expose enough **actual low-level control** or has incompatible ownership/borrowing restrictions. If necessary, implement a **small C shim** compiled by a Rust `build.rs`, isolated in one crate. Never scatter unsafe code across the scheduler, coordinator, or HTTP crates.

**Avoid** using `llama-server` HTTP as the main execution backend: its internal scheduler would prevent NiniServe from owning batch composition.

### 2.2 API capabilities the adapter MUST establish

- Load a GGUF model and establish a llama.cpp context; configure Metal offloading when available.
- Tokenize input using the model vocabulary with correct BOS/special-token behavior.
- Submit batches that explicitly specify token IDs, token **positions**, sequence IDs, and which tokens need logits.
- Read the logits corresponding to each requesting sequence (or supported per-sequence sampler output) without mixing batch indices.
- Maintain separate sampler/random state per request; sample the next token.
- Carry sampled tokens into subsequent decode steps at correct sequence positions.
- Cleanly remove model-side state for a completed/cancelled sequence.
- Detect end-of-generation tokens correctly (EOS/EOG using supported model/vocabulary semantics).
- Respect context size, logical and physical batch-size bounds, and model architecture limitations.
- Return meaningful backend errors, including insufficient capacity.

Important llama.cpp concepts to verify in the pinned version include `llama_batch` (fields for token/position/seq ID/logits), the decode operation, memory access (`llama_get_memory` and sequence removal in current C API), and sampler operations. On a version mismatch, update the adapter and documentation to the **verified** API, not what this specification guesses.

### 2.3 Proof-of-feasibility program (`examples/two_sequences.rs`)

Build a small standalone test executable before implementing HTTP. It should:

1. Load a real, compatible GGUF model locally.
2. Tokenize two different prompts.
3. Prefill the two sequences using distinct backend sequence IDs.
4. Request the correct final-prompt logits for each sequence.
5. Generate 8–16 tokens from each, with distinct sampling state.
6. Interleave their decode steps in one shared backend context.
7. Explicitly clean up both sequences.
8. Print a **verified trace** of seq ID, token position, prefill/decode, and execution-batch membership.
9. Repeat the run and validate no sequence mixing; use greedy decoding / fixed seed where determinism is available.

**Gate:** Do not mark “multi-sequence inference supported” until this passes on a real GGUF. A mock backend test does not satisfy the gate. If runtime/hardware/model are unavailable, document it as blocked and proceed only with mock-only framework work.

### 2.4 Version / platform fallback

- When macOS + Metal is unavailable to the agent, run Rust mock tests and, if supported, CPU-only real-inference tests. Clearly label which tests were performed.
- If a CI platform cannot build Metal, skip Metal-specific jobs while continuing platform-independent tests.
- If the pinned wrapper cannot represent multi-sequence batches correctly, use the C FFI path rather than adapting the architecture to a high-level serving API.

---

## 3. System architecture

```text
                         HTTP Clients / Benchmark Load
                                     |
                                 Axum API
                       validation / SSE / cancellation
                                     |
                              Admission Gate
                        bounded queue / timeout / limits
                                     |
                             Engine Commands
                                     |
               +---------------------v--------------------+
               |         ENGINE WORKER (single owner)     |
               |                                         |
               |  State registry   Scheduler policies    |
               |      |                 |                 |
               |      +-----> ExecutionPlan               |
               |                     |                    |
               |               Batch Builder             |
               |                     |                    |
               |       Backend Adapter (llama.cpp)        |
               |                     |                    |
               |        Token events / state updates      |
               |                                         |
               +---------------------+--------------------+
                                     |
                          Metal / CPU backend
                                     |
               Metrics / traces <-- response events --> SSE
```

### 3.1 Threading model

- Tokio runtime hosts HTTP handlers and asynchronous message passing.
- The **model context and its mutable state are owned by one dedicated engine thread** (or equivalent exclusive worker) to avoid accidentally invoking non-thread-safe backend operations concurrently.
- Async handlers must never call synchronous GPU inference directly on Tokio core worker threads.
- Communication uses bounded channels (`tokio::sync::mpsc`, appropriately bridged from dedicated thread as needed).
- The engine sends per-request output events on separate bounded per-request channels.
- Cancellation is an explicit engine command (or promptly polled cancellation flag), not merely dropping the HTTP socket.
- No `Arc<Mutex<BackendContext>>` wrapping a model context to enable arbitrary concurrent inference calls; one owner should make scheduling behavior obvious.

### 3.2 Main modules and responsibilities

**`niniserve-protocol`**: typed requests/responses, IDs, configuration schema, model-independent generation events, bounded errors.

**`niniserve-backend`**: model-execution trait, deterministic mock executor, pinned llama.cpp adapter. Owns all FFI and backend-specific lifetime/safety concerns.

**`niniserve-engine`**: registry of active sequences, state machine, engine loop, batch builder, sampler ownership, scheduling invocation, cleanup, event routing.

**`niniserve-scheduler`**: scheduler trait, policies, fairness, feedback/controller logic, deterministic pure tests.

**`niniserve-api`**: Axum routes, request validation, SSE serialization, disconnect/timeout handling, error mapping, metrics exposure.

**`niniserve-metrics`**: engine measurements, timers, counters/histograms, tracing, Prometheus exposition (can start with simple structured counters before full Prometheus).

**`niniserve-cli`**: binary entry point, configuration, runtime startup/shutdown, policy selection.

**`niniserve-bench`** (or `benches/` plus scripts): workload generator, result files, data analysis/plot commands. Prefer keeping benchmark utilities separate from production request path.

**Keep crate count pragmatic:** A minimal workspace may initially consolidate crates, then split where clean interfaces warrant. Do not create empty crates purely for appearance.

### 3.3 Proposed repository tree

```text
niniserve/
├── AGENTS.md
├── NINISERVE_MASTER_SPEC.md
├── README.md
├── Cargo.toml
├── Cargo.lock
├── rust-toolchain.toml           # pin a supported stable Rust toolchain, if useful
├── .gitignore
├── .github/workflows/ci.yml
├── crates/
│   ├── niniserve-protocol/
│   ├── niniserve-backend/
│   │   ├── src/lib.rs
│   │   ├── src/mock.rs
│   │   └── src/llamacpp.rs        # source/binding API verified in feasibility spike
│   ├── niniserve-scheduler/
│   │   ├── src/lib.rs
│   │   ├── src/fcfs.rs
│   │   ├── src/decode_priority.rs
│   │   ├── src/fixed_chunk.rs
│   │   └── src/adaptive.rs
│   ├── niniserve-engine/
│   │   ├── src/lib.rs
│   │   ├── src/lifecycle.rs
│   │   ├── src/registry.rs
│   │   ├── src/batch.rs
│   │   ├── src/admission.rs
│   │   └── src/coordinator.rs
│   ├── niniserve-metrics/
│   ├── niniserve-api/
│   ├── niniserve-cli/
│   └── niniserve-bench/
├── examples/
│   ├── two_sequences.rs
│   └── sample_requests.sh
├── tests/
│   ├── api_integration.rs
│   └── real_inference.rs         # gated by model env var
├── configs/
│   ├── default.toml
│   └── adaptive.toml
├── docs/
│   ├── ARCHITECTURE.md
│   ├── BACKEND_DECISION.md
│   ├── SCHEDULING.md
│   ├── BENCHMARKS.md
│   ├── KNOWN_LIMITATIONS.md
│   └── IMPLEMENTATION_LOG.md
├── scripts/
│   ├── smoke.sh
│   └── plot_bench.py             # optional; does not ship in production runtime
├── results/                     # keep only small sanitized sample results
└── models/                      # ignored by git
```

`Cargo.toml` workspace membership, exact crate names, and paths can evolve as needed. Keep documentation consistent.

---

## 4. Data model and invariants

### 4.1 Identifiers

Use strongly typed IDs rather than arbitrary strings throughout the engine.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceId(pub u32);
```

Map `SequenceId` safely onto backend integer sequence IDs; avoid overflow, duplication, and reusing a sequence ID **before** its backend KV state has been released.

### 4.2 Generation request

```rust
pub struct GenerationRequest {
    pub id: RequestId,
    pub prompt: String,
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: Option<u64>,
    pub stream: bool,
    pub deadline: Option<std::time::Instant>,
}
```

**Notes:** This is conceptual and can be split into wire (`serde`) and internal types. Never serialize `Instant`. Non-streaming response is optional at first; if not supported, reject it explicitly. Validate finite sampling parameters and enforce sane generation limits.

### 4.3 Lifecycle

```text
Received --> Validated --> Queued --> Prefilling --> Decoding --> Completed --> Released
     |           |          |             |           |
     +----------> Rejected  +------------> Cancelled  |
     +---------------------> TimedOut / Failed <-------+
```

Suggested enum:

```rust
pub enum RequestState {
    Queued,
    Prefilling,
    Decoding,
    Completed,
    Cancelled,
    TimedOut,
    Failed,
}
```

`Released` is a resource-cleanup postcondition and can be represented separately rather than as a public request state.

**Lifecycle invariants:**

- A request never decodes before all required prompt tokens have been evaluated.
- A request is in at most one active engine registry entry.
- At most one backend sequence ID is owned per request in the initial implementation.
- Every active backend sequence has a matching active registry entry; do not silently retain stale state.
- Terminal requests never re-enter scheduling.
- Terminal cleanup is idempotent from the coordinator's point of view; backend memory release is performed exactly once or guarded by an `is_released` flag.
- Sequence IDs are not recycled until backend cleanup succeeds.
- One request can never receive another request's token/logit/output channel.
- Engine state changes and emitted token events must maintain a well-defined order.

### 4.4 Per-sequence state

```rust
pub struct ActiveSequence {
    pub request_id: RequestId,
    pub backend_seq_id: SequenceId,
    pub prompt_tokens: Vec<u32>,
    pub prefill_cursor: usize,
    pub generated_token_ids: Vec<u32>,
    pub next_context_position: u32,
    pub last_sampled_token: Option<u32>,
    pub state: RequestState,
    pub arrival_time: std::time::Instant,
    pub admitted_time: std::time::Instant,
    pub first_token_time: Option<std::time::Instant>,
    pub last_token_time: Option<std::time::Instant>,
    // sampler and output sender held in runtime-specific owning structures
}
```

Crucial distinction: The next **sampled** token is not necessarily already **evaluated** in the backend KV state. Explicitly track that boundary, especially after the last prefill logits are sampled. The next decode execution must submit the pending sampled token at the proper position before sampling another; do not shift positions or count nonexistent cached tokens.

### 4.5 Context-capacity model

Capacity should include required prompt positions plus projected generated positions, with explicit reserves, using the backend's documented context semantics. Do not confuse `n_ctx`, `n_batch`, and `n_ubatch`:

- `n_ctx`: context capacity constraints.
- `n_batch`: logical input token batch limit.
- `n_ubatch`: physical microbatch limit, where exposed.

In multi-sequence contexts, actual capacity/layout may be model/backend-specific. Begin with conservative limits (e.g., configured max active sequences and per-sequence max context), and add fine-grained accounting only after verifying backend behavior. Define whether admission reserves worst-case max tokens or estimates a shorter horizon. Prefer safety to maximizing utilization in v1.

---

## 5. Backend executor contract (conceptual, not fake drop-in code)

Separate a **pure execution plan** from llama.cpp-specific objects. The following types express intent and are not promised to compile unmodified.

```rust
pub enum WorkItem {
    Prefill {
        seq_id: SequenceId,
        prompt_start: usize,
        prompt_len: usize,
    },
    Decode {
        seq_id: SequenceId,
        token_id: u32,
        position: u32,
    },
}

pub struct ExecutionPlan {
    pub items: Vec<WorkItem>,
    pub logical_token_count: usize,
    pub scheduling_policy: &'static str,
}

pub struct ExecutionOutput {
    pub per_sequence_logits: Vec<SequenceLogitsHandle>,
    pub duration: std::time::Duration,
}
```

**Important:** A logit handle, raw pointer, or slice borrowed from llama.cpp may be invalidated by subsequent decode calls. The executor must sample immediately while the data is valid, or copy only the necessary values safely into owned memory. Prefer a safe `SampledToken` output from the backend layer over exposing long-lived borrowed logits to an async scheduler.

Recommended conceptual interfaces:

```rust
pub trait ModelExecutor {
    type Error: std::error::Error + Send + Sync + 'static;
    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>, Self::Error>;
    fn execute(&mut self, plan: &ExecutionPlan) -> Result<Vec<BackendTokenEvent>, Self::Error>;
    fn release_sequence(&mut self, seq_id: SequenceId) -> Result<(), Self::Error>;
    fn limits(&self) -> BackendLimits;
}
```

You may need separate model initialization and request sampler initialization methods. Keep trait object-safety and actual backend lifetimes practical. If a generic associated type or a separate `MockExecutor` API would simplify correctness, document that decision and implement it cleanly.

### 5.1 Execution guarantees

- Only include work actually eligible for execution based on current sequence state.
- Each token is assigned the correct backend sequence ID and token position.
- Logits should be requested for final prefill tokens and active decode tokens where needed; avoid requesting unnecessary logits for interior prefill tokens.
- When multiple sequences produce logits in a single batch, maintain a mapping from returned logits indices to request IDs. Backend-specific indexing is verified in the feasibility spike.
- Sampling for each request is independent. Start with greedy sampling for correctness tests; add temperature/top-p/seed later.
- Never silently continue when backend execution fails; apply a recovery policy documented for the pinned backend. Fatal or partially committed batch errors must not leave the engine assuming the wrong KV state. If recovery cannot be validated, fail affected requests and reset/recreate the context safely.
- A single execution step may contain both prefill chunks and decode tokens only if the selected llama.cpp API/version supports that reliably; otherwise perform separated, short steps and clearly reflect that in benchmarks.

### 5.2 BatchBuilder

BatchBuilder converts `ExecutionPlan` to backend-specific batch tokens, strictly within configured limits.

Check:

- Total batch logical tokens <= backend configured `n_batch`.
- Each request has valid token cursor and position.
- Each backend sequence is present at most once in a given decode-token set.
- No cancelled or expired request is included.
- Prefill work for one sequence is contiguous and position-correct.
- Logits mapping points to the exact batch positions where they are requested.
- Backend execution failures are not mistaken for empty generation outputs.

### 5.3 Mock backend

The deterministic mock must accept the same plans and return plausible, configurable step durations and token events. It is for **scheduler/lifecycle testing only**. It may have a virtual time model with cost = prefill tokens * prefill cost + active decode items * decode cost + fixed overhead. Do not include mock timing in real-inference benchmark claims.

---

## 6. Scheduling algorithms

### 6.1 Scheduling inputs / outputs

A scheduler sees a **read-only snapshot** of queued/active sequences (prefill remaining, decode eligibility, wait age, recent measured durations, capacity limits). It returns an immutable `ExecutionPlan`. It must not directly mutate backend objects.

```rust
pub trait Scheduler {
    fn name(&self) -> &'static str;
    fn plan(&mut self, view: &EngineView, budget: StepBudget) -> ExecutionPlan;
    fn observe(&mut self, observation: &StepObservation);
}
```

Example `StepBudget` fields:

- max logical tokens for the next backend call
- max active sequences / backend-safe sequence limit
- max prefill tokens this step
- backend-declared restrictions on mixed work

Planner invariants: no budget overflow, no invalid state transitions, no duplicate work, no starvation beyond configured guardrail in testable mock workloads.

### 6.2 Baseline A — FCFS (simple and explicit)

Define what FCFS means. For initial experiments, implement **request-at-a-time, head-of-line FCFS**: earliest admitted active request progresses through prefill and decode before beginning the next. Document that it is an intentionally weak baseline, not a competitive production scheduler.

Expected behavior: simple, minimal interleaving, susceptible to long-request head-of-line blocking.

### 6.3 Baseline B — Decode-priority

At each scheduling opportunity:

1. Choose eligible sequences with pending decode tokens, subject to batch limits and fairness.
2. Reserve one token of decode work per selected sequence.
3. Use spare logical token budget for new or partial prefill work, without breaking backend constraints.
4. Avoid admitting more active sequences than context capacity permits.

Watch the failure case: strict decode priority can starve new prefills under an endless stream of active decoding work. This needs max prefill wait/fairness behavior or a documented baseline without a fairness guarantee.

### 6.4 Baseline C — Fixed chunked prefill

A decode-priority policy with a configured maximum prefill chunk size, initially explore {64, 128, 256, 512} **only if those chunks respect backend limits and model/hardware memory**. Long prompts advance in multiple steps. Track progress by cursor, not by repeatedly re-evaluating the beginning of the prompt.

Make fixed chunks configurable so benchmark reports can compare the adaptive controller to **multiple** plausible fixed baselines—not only an artificially bad one.

### 6.5 Primary experiment — Adaptive prefill

**Hypothesis:** A cheap, feedback-driven controller can reduce excessive decode stalls under mixed workloads compared with a single fixed chunk size, but may trade off TTFT, scheduler overhead, or throughput.

Controller state:

- `prefill_budget_tokens` (bounded integer)
- smoothed inter-token latency (EWMA or rolling percentile estimate)
- target/soft upper-bound for active decode ITL
- step duration estimate / overhead
- age of oldest pending prefill request
- minimum fraction or service frequency guaranteed to prefill under load
- startup/warm-up state

Controller parameters (initial proposed, subject to experiments):

- `min_chunk_tokens = 32`
- `max_chunk_tokens = 512`
- `initial_chunk_tokens = 128`
- `decrease_factor = 0.5`
- `increase_step_tokens = 32`
- `ewma_alpha = 0.2`
- `high_latency_threshold = target * 1.10`
- `low_latency_threshold = target * 0.85`
- `max_prefill_wait_ms = 500` (illustrative, tune according to actual model/hardware)

These values are **research starting parameters**, not guaranteed optimal settings.

Example control law:

```text
observe(real_step_latency, per_sequence_token_gaps, oldest_prefill_wait)
if active_decoders == 0:
    allow as much prefill as safe backend budget permits
else if enough valid decode latency samples exist:
    if smoothed_decode_itl > target * 1.10:
        chunk = max(min_chunk, round_down(chunk * 0.5))
    else if smoothed_decode_itl < target * 0.85:
        chunk = min(max_chunk, chunk + 32)
    else:
        keep chunk unchanged
if oldest_prefill_wait > max_prefill_wait:
    reserve a bounded, backend-safe minimum slice for that prefill
clamp plan to actual per-step backend token budget
```

**Measurement subtlety:** Network delivery, slow clients, and queue wait should not be confused with GPU execution time. Compute per-token generation gaps on the server side and separate (a) backend-step time, (b) scheduler time, (c) output-queue time, and (d) user-observed stream latency. A p95 objective should ultimately use measured per-token gap samples; an EWMA is just a simple controller signal, not a substitute for reported p95.

**Why adapt prefill?** On a single context, long prefill evaluations may block the next decode step; small chunks can improve responsiveness but add overhead or delay prefill completion. The controller adjusts the tradeoff based on load rather than enforcing one global chunk size.

**Adaptive invariants:**

- Controller never emits chunks smaller than 1 or larger than safe limits.
- It uses bounded memory and O(1) or low-cost update work per step.
- Under no active decoder, it does not unnecessarily throttle prefill.
- When repeated decode latency exceeds target, chunk size cannot grow absent cooldown/explicit guardrail.
- Prefill requests receive progress under persistent decode demand.
- Controller can be disabled instantly via config in favor of baseline policy.
- Decisions are logged with input signal, previous budget, selected budget, and rationale.

### 6.6 Fairness

Apply an explicit max-wait or aging mechanism. Record starvation incidents and worst prefill queue age. Do not report a scheduling win if low p95 ITL is achieved by starving the new requests forever.

---

## 7. Engine coordinator algorithm

### 7.1 Conceptual run loop

```text
loop until shutdown:
    drain bounded incoming commands (new / cancel / shutdown)
    expire overdue requests and release terminal sequences
    admit pending requests if capacity allows
    construct read-only scheduling snapshot
    ask active scheduler for next ExecutionPlan
    validate plan against lifecycle + context + batch constraints
    if plan has work:
        call executor from the dedicated engine worker
        associate outputs/logits with exact sequence IDs
        sample eligible next tokens using per-request sampler state
        update prefill cursors and generated-token state
        emit output events with backpressure-safe semantics
        record execution timings and controller observations
        finalize EOS / max-token / cancelled sequences
    else:
        block/wait on next command or earliest deadline (avoid CPU busy-spin)
```

### 7.2 Event types

```
EngineCommand::Enqueue(request, output_channel)
EngineCommand::Cancel(request_id)
EngineCommand::Shutdown

GenerationEvent::Started(request_id)
GenerationEvent::Token(request_id, token_id, text_fragment, timestamp)
GenerationEvent::Completed(request_id, finish_reason, usage)
GenerationEvent::Error(request_id, error_code, message)
```

Separate raw tokens and UTF-8 text fragments. A tokenizer may split Unicode characters across tokens, so concatenating per-token decoded bytes naively may produce invalid UTF-8. Implement a validated incremental detokenizer or reliable backend text decoder and test multibyte Unicode.

### 7.3 Disconnect + streaming backpressure

- Each SSE client has a bounded event channel.
- On connection drop, enqueue or flag cancellation immediately; cleanup occurs in engine worker.
- Never block model-execution thread indefinitely on a slow client. Choose and document an explicit policy (bounded nonblocking sends; cancel or fail a request when its output queue persistently fills).
- If the stream is cancelled after a token was already computed, avoid double-release or unsafely interrupting an active model call.
- The SSE endpoint sends a terminal `[DONE]` marker (or a defined equivalent) when the stream completes normally; error and disconnect semantics are documented.

### 7.4 Shutdown

A graceful shutdown stops admission, cancels/finishes in-flight requests according to policy, releases all sequences, drops the model context once, and terminates the worker. Do not leave a background executor thread orphaned.

---
## 8. HTTP contract (small, explicit, testable)

NiniServe exposes an **OpenAI-inspired**, not fully compliant, API until interoperability tests prove otherwise.

### 8.1 `POST /v1/completions`

Example request:

```json
{
  "model": "local-gguf",
  "prompt": "Explain how GPU inference works in three sentences.",
  "max_tokens": 64,
  "temperature": 0.0,
  "stream": true
}
```

Requirements:

- Validate `prompt` not empty, byte limits, prompt token limits, configured `max_tokens`, model identifier, and sampling parameter ranges.
- Return `400` on malformed/invalid inputs, `404` on unknown endpoint, `422` if using semantically invalid model-related values by documented convention, `429` or `503` for overload (choose and document), `500` for unexpected faults.
- For streams, set `Content-Type: text/event-stream`, appropriate no-cache headers, and emit one valid JSON object per `data:` SSE record, with a terminal `data: [DONE]` marker for normal completion.
- Include stable per-request `id`, `object`, `created`, `model`, `choices` with a textual delta or text fragment, and `finish_reason` where relevant if claiming OpenAI-compatible shape.
- Never return success (HTTP 200 with normal completion) after an unhandled execution failure.
- At least one explicit error event/abort behavior for errors occurring after SSE headers were sent.
- Ensure text is valid UTF-8 and text fragments concatenate to the generated string.
- For now, `stream: false` may be unsupported but must be validated and documented; preferably implement it once normal SSE works by collecting output with a maximum buffer size.

Example **illustrative** stream records (serialization and exact `choices` schema must be documented):

```text
data: {"id":"req_123","object":"text_completion","choices":[{"index":0,"text":"A GPU","finish_reason":null}]}

data: {"id":"req_123","object":"text_completion","choices":[{"index":0,"text":" processes","finish_reason":null}]}

data: {"id":"req_123","object":"text_completion","choices":[{"index":0,"text":"","finish_reason":"stop"}]}

data: [DONE]
```

**Implementation note:** If the project intentionally provides only a subset of the OpenAI format, label it "OpenAI-inspired completion API" and explicitly list missing fields/features. Do not silently pretend that all OpenAI SDK behavior works.

### 8.2 `GET /healthz`

JSON with process status and model load state. `200` when operational; optionally `503` while model not ready. Never claim ready before backend initialization completes.

### 8.3 `GET /metrics`

Prometheus text exposition if a metrics dependency has been integrated. Otherwise use `/debug/engine` for development and reserve `/metrics` for valid exposition format; do not return misleading JSON under a Prometheus endpoint.

### 8.4 `GET /debug/engine`

For local development only. Expose scheduler name, number of active/queued requests, current chunk size, model information excluding sensitive file paths if configured, and capacity budget. Bind to loopback by default. Do not dump prompt contents or generated text into debug endpoints/logs.

### 8.5 Minimal commands after implementation

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

# real model demo; flags are target CLI design, not already implemented
cargo run --release -p niniserve-cli -- serve \
  --model ./models/model.gguf \
  --host 127.0.0.1 --port 8080 \
  --scheduler fixed-chunk \
  --prefill-chunk-tokens 128

curl -N http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-gguf","prompt":"What is Rust?","max_tokens":32,"temperature":0,"stream":true}'
```

The agent should implement and document the **actual** CLI flags rather than leaving this sample fictitious.

---

## 9. Configuration specification

A single typed config object should be loaded, validated, and logged with non-secret values. CLI overrides may be layered on top of TOML. Reject inconsistent or unsafe values on startup.

Example intended `configs/default.toml`:

```toml
[server]
host = "127.0.0.1"
port = 8080
max_request_body_bytes = 1048576

[model]
path = "./models/model.gguf"      # placeholder: user supplies actual GGUF
n_ctx = 4096
n_batch = 512
n_ubatch = 128
n_gpu_layers = 999                # backend-dependent; must verify actual offload

[engine]
max_active_sequences = 4
max_queued_requests = 64
max_prompt_tokens = 2048
max_new_tokens = 256
request_timeout_ms = 120000
per_client_event_queue = 32

[scheduler]
policy = "fixed-chunk"           # fcfs | decode-priority | fixed-chunk | adaptive
prefill_chunk_tokens = 128
max_step_tokens = 512
max_prefill_wait_ms = 500

[scheduler.adaptive]
target_itl_ms = 75.0              # illustrative, calibrate on actual hardware/model
min_chunk_tokens = 32
max_chunk_tokens = 512
initial_chunk_tokens = 128
increase_step_tokens = 32
decrease_factor = 0.5
ewma_alpha = 0.2
hysteresis_high = 1.10
hysteresis_low = 0.85

[telemetry]
log_level = "info"
prometheus = true
trace_scheduler_decisions = false
```

**Correctness checks:**

- `n_ubatch <= n_batch`; actual API may impose further limits.
- `max_step_tokens <= n_batch`.
- `max_active_sequences > 0`; queue sizes bounded; per-client channel > 0.
- `min_chunk <= initial_chunk <= max_chunk`; max chunk <= available backend logical budget (or clamp and log).
- context and prompt/output limits must be mutually feasible.
- sampling and adaptive parameters are finite, bounded, nonnegative as appropriate.
- localhost default binding. A public bind is a deliberate opt-in with documented safety implications.
- Real `n_gpu_layers` handling must follow pinned llama.cpp semantics; report actual offloading if observable. Do not assume every layer is on GPU because configured value is large.
- `max_active_sequences` is a cap, not a guarantee of physical context capacity; backend may require conservative headroom.

---

## 10. Observability and metric definitions

Measurement matters as much as feature implementation. Use monotonic clocks (`Instant`), stable request IDs, and meaningful histogram labels. Avoid high-cardinality request IDs as Prometheus labels.

| Metric | Precise intent |
|---|---|
| `ttft_ms` | `first_generated_token_ready_at - request_received_at` (explicitly includes admission/queueing unless labeled separately) |
| `queue_wait_ms` | `admitted_to_active_at - request_received_at` or clearly defined queued duration |
| `itl_ms` | difference between consecutive **server-generated-token-ready** timestamps for the same request; excludes first token |
| `e2e_ms` | terminal completion at server minus request arrival |
| `output_tokens_per_sec` | generated output tokens divided by chosen measurement wall time, with start/end defined |
| `requests_per_sec` | successful completions per elapsed benchmark window |
| `backend_step_ms` | duration of a backend execution call |
| `scheduler_step_us` | wall time spent producing/validating an execution plan |
| `prefill_tokens_step` | number of prompt tokens actually submitted to backend in this step |
| `decode_tokens_step` | count of active decode tokens actually submitted |
| `active_sequences` | active backend-owned sequences |
| `queued_requests` | pending queue size |
| `cancelled_requests` | count by reason (disconnect, timeout, explicit) |
| `rejected_requests` | count by reason (queue, token limit, capacity, timeout) |
| `prefill_oldest_wait_ms` | age of longest-waiting eligible prefill request |
| `adaptive_chunk_tokens` | controller-selected budget, as a gauge/event |

For aggregated output throughput, ensure the benchmark includes an elapsed execution window that treats ramp-up and tail behavior consistently. TTFT/ITL vary with load; collect raw samples and report p50/p95 and sample counts. When no ITL samples exist, report `N/A`, not `0 ms`.

### 10.1 Trace fields

```text
request_id, seq_id, scheduler_policy, engine_step,
queue_depth, active_sequences,
prefill_tokens, decode_tokens,
chunk_budget_previous, chunk_budget_selected,
observed_itl_ms, backend_step_ms,
request_state_before, request_state_after,
finish_reason, cancellation_reason
```

Default production-ish logs should **not** include raw prompts, token text, authorization headers, full file paths, or secrets. Deep debug logging is an explicit local-only option.

### 10.2 Timing caveats

- Time model loading separately; exclude it from steady-state benchmark measurements, but report it as useful metadata.
- Metal execution and synchronization can affect observed timings; define the exact synchronous boundaries of backend execution.
- A client that reads slowly can distort user-perceived latency. Always separate engine metrics from client-observed SSE metrics.
- Token output may be delayed by UTF-8 aggregation; report both raw token timestamps and emitted-text timestamps if needed.
- Hardware power state, thermal throttling, other apps, quantization, context size, and actual GPU layer offload can change outcomes. Record what is observable.

---

## 11. Verification strategy

### 11.1 Static / unit tests (no model required)

- Valid and invalid lifecycle transitions.
- Monotonic prefill cursor; cursor never goes past prompt.
- Position correctness for prompted/generated tokens.
- Unique active backend sequence IDs.
- Event routing is request-specific.
- Engine admission is bounded, overload is deterministic.
- Every scheduler plan stays within allowed budgets.
- FCFS follows documented order.
- Decode-priority schedules eligible decode first as far as allowed.
- Fixed chunk policy enforces chunk ceiling.
- Adaptive controller: above-target latency decreases budget; below-target increases; hysteresis avoids thrashing.
- Adaptive fairness advances old prefills.
- Cancel-at-each-lifecycle-state behavior.
- Exactly-once (or guarded idempotent) terminal cleanup.
- No generated tokens beyond `max_new_tokens`.
- Request deadline handling.
- SSE serialization and terminal markers.
- UTF-8 detokenization over split multibyte characters.
- Empty prompt and invalid request parameter rejection.

### 11.2 Mock engine tests

Simulate several requests with a fake executor and virtual compute delays. Confirm queueing, interleaving, completion, metrics updates, and fairness. Mock execution may be deterministic for repeatable scheduler regression tests. **Mock speedups are not evidence of real GPU improvements.**

### 11.3 Real inference tests (model supplied via environment)

Set `NINISERVE_TEST_MODEL=/absolute/path/to/test.gguf`. Tests skip clearly if variable absent, reporting **SKIPPED (no model)** rather than claiming pass. Required scenarios:

1. Short single request yields coherent nonempty output (model-dependent; deterministic token output comparison preferred).
2. Two distinct prompts have distinct sequence IDs, valid outputs, and independent state.
3. Interleaved request A and B do not contaminate one another.
4. Request A cancels mid-generation while B continues.
5. Long prompt is processed over multiple prefill chunks without repeating earlier chunks.
6. `max_new_tokens` and EOS terminate correctly.
7. After cancellation/completion, another request can reuse released capacity safely.
8. Backend error handling doesn't claim success or silently retain invalid state.
9. Metal test when Apple GPU is available, CPU-only smoke test otherwise.

### 11.4 Property tests / stress

Recommended: `proptest` for policy/budget/lifecycle properties. Generate arbitrary valid sequences of enqueue/cancel/timeout/complete events. Test registry cleanup, no leaked sequence ID, no phantom terminal work, and scheduler budget safety. Stress 50–500 mock requests with bounded queues. Document chosen randomness seed when reproducing a failure.

### 11.5 CI gates

On ordinary runners without local model or Metal: `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace` using the mock executor and API tests. Make system C/C++ build dependencies explicit if llama.cpp is compiled in CI. If backend compilation makes generic CI too brittle, gate native feature behind a feature flag while keeping mock CI functional; separately document native build validation.

Use `cargo audit` where appropriate, but do not claim security reviews or fuzz coverage unless performed.

---

## 12. Benchmark methodology (the portfolio centerpiece)

### 12.1 Experimental standards

- Compare schedules with **identical** GGUF model, quantization, model context parameters, n_gpu_layers, host, traffic inputs, and hardware.
- Explicit warmup phase; never mix model load with steady-state latency distributions.
- At least 3 independent measured repeats per configuration if feasible, ideally more for noise analysis.
- Workloads should have arrival timing, max output tokens, prompt token lengths, and seed configuration recorded in JSONL/CSV.
- No mock timings in benchmark plots labeled "real inference."
- Save raw latency samples when practical and include an automated result summary.
- Run both low-load and saturation scenarios; adaptive policies may help only in certain regions.
- Track completion count, rejections, timeouts, fairness, and GPU/backend errors—not only tokens/s.
- If comparing an upstream llama.cpp server, label the comparison **nonidentical runtime implementations**; match configs as closely as feasible and describe unavoidable differences. Internal policy comparison is the primary experiment.
- Don't interpret a lower response count due to starvation as a latency optimization.

### 12.2 Workloads

**W0: Single request.** Prompt lengths ~64, 512, 2048 tokens (as available), output length ~64. Baseline correctness, TTFT, decode rate, model load cost separately.

**W1: Concurrent short prompts.** Multiple requests with short input (~32–128 tokens) and ~64 output tokens. Sweep arrival concurrency {1,2,4,8} if within safe limits.

**W2: Long-prompt interference.** Start two active decode streams. While they are decoding, inject one 1k–2k token prefill request (bounded by model). Analyze temporary active decode ITL spikes and incoming request TTFT.

**W3: Mixed prompts.** Deterministic mix: 60% short, 30% medium, 10% long; distinct output length distributions. Control the offered request rate, e.g., deterministic arrivals / seeded Poisson arrivals.

**W4: Bursts.** Send sudden bursts above queue/concurrency budget. Measure admissions, overload signals, fairness, and tail latency.

**W5: Adaptive controller step response.** Begin at low load, inject sustained large prefills, then return to low load. Plot chunk budget and measured decode gaps over time to show controller stability or oscillation.

Token lengths must be **measured after model tokenization**, not assumed from character counts. Generate fixed fixtures and check prompt token count within tolerance.

### 12.3 Policy matrix

- Request-at-a-time FCFS.
- Decode-priority.
- Fixed chunk sizes 64, 128, 256, 512 within capacity limits.
- Adaptive policy starting at 128, target calibrated from a low-contention baseline or predeclared goal.

For target calibration, use a separate calibration run and freeze it before the held-out measured workload; do not tune parameters on the measured result and report it as an unbiased comparison.

### 12.4 Output artifact formats

`results/<run-id>/metadata.json`

```json
{
  "run_id": "illustrative-run-id",
  "git_commit": "<actual commit sha>",
  "model_name": "<actual local model identifier>",
  "model_quantization": "<actual GGUF quantization>",
  "backend_version": "<pinned llama.cpp version>",
  "os": "<platform>",
  "hardware": "<measured/recorded machine>",
  "scheduler": "adaptive",
  "scheduler_config": {},
  "workload": "W2_long_prompt_interference",
  "warmup_requests": 5,
  "measured_requests": 30,
  "started_at_utc": "<timestamp>"
}
```

`requests.csv` suggestion:

```text
request_id,policy,prompt_tokens,output_tokens,arrival_ms,queue_wait_ms,ttft_ms,e2e_ms,finished,finish_reason
```

`token_gaps.csv` suggestion:

```text
request_id,token_index,ready_at_ms,itl_ms,engine_step
```

`scheduler_steps.csv` suggestion:

```text
engine_step,timestamp_ms,policy,active_sequences,queued_requests,prefill_tokens,decode_tokens,chunk_budget,backend_step_ms,scheduler_step_us
```

Do not emit placeholder values as though they were actual data. These are proposed schemas.

### 12.5 Plots / interpretation

Generate at minimum:

1. p50/p95 TTFT across policies at varying concurrency.
2. p50/p95 ITL across policies at varying concurrency.
3. Completed output tok/s and requests/s, with rejected/timeout counts.
4. Tail ITL around long-prefill arrival (time-series).
5. Adaptive chunk size vs time, with actual decode gaps and load overlay if feasible.
6. Fairness: oldest prefill wait, max queue age, and request completion distribution.

For every graph, describe whether values improved, regressed, or were statistically inconclusive. When a scheduler is worse, show it honestly.

---

## 13. Implementation milestones (strict gates)

### Phase 0 — Repository and backend spike

**Objective:** Remove backend-interface uncertainty before designing around assumptions.

Tasks:

- Inspect repository and environment: `git status`, platform, Rust toolchain, Xcode CLI/clang, CMake, availability of GGUF weights.
- Initialize workspace only if needed; avoid overwriting existing user code.
- Research pinned wrapper + C API in versioned source, not blog snippets.
- Implement `examples/two_sequences.rs` as described in Section 2 if hardware/model available.
- Record APIs actually verified, crate version, FFI decision, memory/sequence cleanup behavior in `docs/BACKEND_DECISION.md`.
- Set up formatting, linting, CI scaffold, `.gitignore` (exclude model weights, generated binaries, benchmark raw payloads if sensitive).
- Create `docs/IMPLEMENTATION_LOG.md` including environment, progress, blockers.

**Gate:** A verified multi-sequence prototype or clearly documented blocker + functioning mock skeleton. No unverified "real continuous batching" claims.

### Phase 1 — Minimal real serving foundation

**Objective:** One real model can serve one request end-to-end with streaming.

Tasks:

- Implement validated config and CLI startup.
- Wrap proven llama.cpp APIs in `niniserve-backend`.
- Implement single-owner engine worker and command/event channels.
- Implement state machine, one active request, prefill, decode, EOS/max-token stop, release.
- Implement Axum `/healthz`, `POST /v1/completions` streaming SSE.
- Implement simple test-friendly mock backend and API integration tests.
- Record basic TTFT and generation throughput.
- Provide copy-paste local setup commands with `--model` path.

**Gate:** Model loads, generates a valid SSE stream, terminates, releases state; mock+integration tests pass; real test explicitly labeled PASS/SKIP/BLOCKED.

### Phase 2 — Actual multi-request continuous batching

**Objective:** NiniServe, not the backend HTTP server, decides which sequences enter the inference step.

Tasks:

- Add bounded request queue, active registry, unique backend sequence IDs.
- Add multi-sequence batch construction with correct logits mapping and positions.
- Add independent per-request sampling states.
- Add token streaming, disconnect detection, cancellations, timeouts.
- Preserve request safety on errors and prompt/generation limits.
- Add real two-/three-request integration tests and trace evidence of shared execution steps.
- Measure scheduling overhead.

**Gate:** Two independent real requests interleave within the same backend context; batch trace proves it; one request can cancel without killing the other; no output/KV mixing.

### Phase 3 — Baseline schedulers + benchmark harness

**Objective:** Establish rigorous baselines before trying to optimize.

Tasks:

- Implement scheduler trait and pure snapshot/planning logic.
- Add request-at-a-time FCFS, decode-priority, fixed chunked prefill.
- Add structured scheduler-step traces and instrumentation.
- Add deterministic mock tests for schedule behavior/fairness.
- Add W0–W3 load generation, result CSV/JSON, summary command.
- Collect actual baseline results; if no local model, leave execution-ready harness and no fake numbers.

**Gate:** Policy switching changes actual batch plans and observations; real-policy benchmark commands work where hardware is available.

### Phase 4 — Adaptive controller

**Objective:** Investigate the original NiniServe research question.

Tasks:

- Implement EWMA/hysteresis controller, bounds, startup logic, and max-wait fairness.
- Expose adaptive parameters in config and debug view.
- Add controller-specific unit tests for response to synthetic latency regimes and stability.
- Benchmark controller against several fixed chunk sizes; do not cherry-pick one baseline.
- Plot chunk budget trajectories and observed ITL, throughput, TTFT, queue fairness.
- Write findings with negative results and limitations.

**Gate:** Show reproducible experiments on real inference, not just mock simulation. No claim that adaptive is superior until measured.

### Phase 5 — Robustness, polish, publishability

**Objective:** Make the project credible to a systems engineer reading GitHub.

Tasks:

- Complete capacity-aware admission, fast timeout paths, graceful shutdown.
- Add useful Prometheus metrics and structured traces.
- Add CI, README architecture diagram, reproducibility scripts.
- Add safety documentation (local binding, model licensing, resource limits).
- Re-run all smoke/integration tests.
- Create final performance report with methodology, model/hardware specs, and caveats.

**Gate:** Setup and local smoke commands are reproducible; tests verified; no overclaimed feature list.

### Future investigations (NOT initial milestones)

- True backend-supported KV prefix reuse with measured skipped prefill work.
- Multi-context routing with cache-locality/load-aware placement, inspired by Dynamo.
- Multi-worker prefill/decode handoff **only** if transferring actual KV state is achievable.
- Richer controller (predictive prefill cost, deadline-aware tuning, control-theoretic stability evaluation).

---

## 14. Done criteria (project and per-change)

### Every pull request / coding-agent session must:

1. State intended change and scope.
2. Keep code compiling on supported platform(s) or report exact environment limitation.
3. Add/update tests covering behavior or justify why test not possible yet.
4. Run formatting and relevant checks; report command **and outcome**.
5. Avoid asserting that model benchmarks work without running them.
6. Update `docs/IMPLEMENTATION_LOG.md` with what changed, next task, issues, and manual setup steps.
7. Avoid touching unrelated files or reformatting the entire repo for convenience.
8. Provide a short, descriptive commit suggestion; never force-push.

### Final v1 acceptance checklist

- [ ] Single local GGUF generates through NiniServe.
- [ ] Single-owner model context safely executes multi-request batches.
- [ ] Request state machine robust under completion, cancellation, timeout, and rejection.
- [ ] SSE streams per request preserve text and completion semantics.
- [ ] FCFS, decode-priority, fixed chunked, adaptive policies demonstrably distinct.
- [ ] Scheduler traces align with actual backend-executed batches.
- [ ] Admission/capacity budgets are enforced, no unbounded queues.
- [ ] Real-model concurrency/cancellation tests pass on target hardware.
- [ ] Mock unit/property tests pass in ordinary CI.
- [ ] Benchmark harness runs reproducibly with a documented model/hardware context.
- [ ] Performance results disclose regressions, variance, and tradeoffs.
- [ ] README explains precisely what is NiniServe-owned vs llama.cpp-owned.

---

## 15. Risk register and mitigation

| Risk | Severity | Mitigation |
|---|---|---|
| Bindings/API churn | High | Pin versions; verify actual low-level APIs during Phase 0; isolate FFI. |
| Incorrect logits association | Critical | Two-sequence spike; per-batch mapping tests; deterministic greedy tests. |
| KV corruption/cross-talk | Critical | One context owner; unique sequence IDs; explicit cleanup tests and backend error policy. |
| Confusing sampled vs evaluated tokens | Critical | Explicit pending sampled token state; positions asserted in tests. |
| GPU/Metal not present in CI | Medium | Mock default tests; feature-gated native tests; target-machine smoke instructions. |
| Blocking Tokio on GPU inference | High | Dedicated synchronous execution thread and bounded channels. |
| Slow SSE clients causing engine stalls | High | Bounded output events; defined cancellation/backpressure policy. |
| Claimed prefix reuse without real compute savings | High | Defer optimization until backend KV sharing proven and timed. |
| Adaptive scheduler instability | Medium | EWMA, hysteresis, bounds, cool-down/fairness, step-response plots. |
| Unfair benchmark comparison | High | Same model/config/hardware; fixed workload fixtures; report rejection/timeout counts. |
| Overengineering | High | Enforce phase gates, start with minimal crates, do not build UI, distributed systems, or custom kernels. |
| Fragile text streaming | Medium | Correct incremental detokenization, multibyte tests, buffer handling. |

---

## 16. Suggested project story / README positioning

> **NiniServe** is a Rust-native experimental LLM serving runtime for Apple Silicon. It uses llama.cpp as the low-level model execution backend while independently managing inference requests, sequence scheduling, continuous batch composition, token streaming, and overload behavior. Its focus is **latency-aware adaptive prefill scheduling**: measuring decode stalls and dynamically adjusting prefill chunks to explore the tradeoff between time-to-first-token, inter-token latency, throughput, and fairness.

**Honest limitation paragraph:** NiniServe uses llama.cpp's model kernels, tokenization, sampling APIs, and physical KV storage. It is a single-node research/portfolio prototype, not a production replacement for llama.cpp server, vLLM, or NVIDIA Dynamo. Performance claims apply only to the measured models, workloads, and hardware.

**Resume impact target, not an accomplished claim:** After implementing and measuring results, replace generic language with exact verified outcomes, e.g., "reduced p95 ITL by X% under workload W2 versus fixed chunk size Y, at Z% throughput change on [hardware/model]". Until then, do not put X/Y/Z on a resume.

---

## 17. Primary references and verification points

These are references for the coding agent to inspect in a network-enabled environment, with **pinned source version preferred** over moving `master`:

1. **llama.cpp C API header**: https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h — inspect batch token/position/seq/logits semantics, `llama_decode`, memory sequence operations, errors, and decode partial-commit behavior.
2. **llama.cpp build instructions**: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md — Metal on macOS, CPU fallback, CMake flags.
3. **`llama-cpp-2` Rust API**: https://docs.rs/llama-cpp-2/latest/llama_cpp_2/ — inspect `LlamaContext`, `LlamaBatch`, sampler and sequence state; use the docs version corresponding to pinned crate.
4. **NVIDIA Dynamo — disaggregated serving**: https://docs.nvidia.com/dynamo/components/router/disaggregated-serving — inspiration for prefill/decode control, not an architecture requirement here.
5. **NVIDIA Dynamo — KV-aware routing**: https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/kv-aware-routing — future multi-worker inspiration, not part of the v1 feature list.
6. **Axum**: https://docs.rs/axum/latest/axum/ — SSE and routing; pin dependencies before compiling.
7. **Tokio**: https://docs.rs/tokio/latest/tokio/ — bounded channels, cancellation coordination, async server.

### Final instruction to any coding agent

**Start with Phase 0, then Phase 1 only.** Confirm the real llama.cpp binding surface by compiling a two-sequence sample when a model is available. Build the smallest working vertical slice before expanding. Do not implement the adaptive scheduler before the engine can reliably run two independent sequences. Explicitly separate code written, code built, tests run, real inference verified, and planned functionality. Keep the code readable enough for a student to walk through at an inference-systems interview.
