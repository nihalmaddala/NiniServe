# Backend decision record

**Status:** Phase 0 feasibility, Phase 1 serving, and the Phase 2 real
continuous-batching gate passed on the supplied real GGUF.

## Decision

Use exactly pinned `llama-cpp-2 = 0.1.159` behind the `niniserve-backend`
`llamacpp` feature. It selects matching `llama-cpp-sys-2 = 0.1.159`, built from
wrapper commit `3cfdd729d65e35da407e5f820edf73201bfa54f6` with vendored upstream
llama.cpp commit `26394b4e6749a41c3633db040e0987500a5f7013`.

The verified safe wrapper surface is sufficient; no C-FFI shim is needed for
the current milestone. Exact source citations and API semantics are recorded in
`docs/BACKEND_API_RESEARCH.md`.

## Real-model evidence

`examples/two_sequences.rs` ran twice against the ignored
`qwen2.5-0.5b-instruct-q4_k_m.gguf` (SHA-256
`74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`) on
Apple M2 Metal.

- One context reported `n_ctx=1024`, `n_batch=512`, `n_ubatch=128`,
  `n_seq_max=2`, and `n_ctx_seq=512`.
- llama.cpp reported `gpu_offload_supported=true` and 25/25 layers offloaded.
- Prefill batch 0 contained both distinct prompts under explicit sequence IDs
  0 and 1 and explicit positions. Only each prompt's final token requested
  logits.
- Decode batches 1 through 11 each contained one explicit-position token for
  sequence 0 and one for sequence 1.
- Separate sampler chains used fixed seeds `0x00C0FFEE` and `0x0BAD5EED`.
  Both runs produced the same per-sequence token IDs:
  - sequence 0: `[3966, 8760, 315, 33789, 374, 1181, 4938, 7149, 11, 892, 646, 387]`
  - sequence 1: `[7319, 44378, 646, 614, 3807, 7567, 11, 2670, 1447, 16, 13, 84486]`
- Output prefixes were distinct and prompt-appropriate: `One benefit of Rust
  is its memory safety, which can be` and `Local inference can have several
  benefits, including:\n\n1. Faster`.
- Whole-sequence removal succeeded for both IDs, and both subsequent
  `kv_cache_seq_pos_max` checks returned `-1`.

This passes the narrow Phase 0 multi-sequence feasibility gate. It does **not**
claim request concurrency, cancellation, continuous scheduling, or a
performance improvement.

## Phase 1 adapter and serving evidence

`LlamaCppExecutor` is the reusable, feature-gated adapter. A `self_cell` owner
keeps `LlamaModel` and its borrowing `LlamaContext` together inside
`niniserve-backend`; no backend type or unsafe code escapes into the engine or
HTTP layer. The adapter implements:

- model tokenization and owned token-piece bytes;
- explicit token IDs, positions, dense backend sequence slots, and per-token
  logits requests through `LlamaBatch::add`;
- exact original batch-index sampling through `LlamaSampler::sample`;
- one independent sampler chain per active sequence;
- `LlamaVocab::is_eog` termination; and
- verified whole-sequence KV removal before sampler state is discarded.

The Phase 1 server loaded the ignored Qwen2.5 fixture on Apple M2, offloaded
25/25 layers, and reported `n_ctx=2048`, `n_batch=512`, `n_ubatch=128`, and
`n_seq_max=1`. A real request streamed 12 coherent text fragments (` Paris. It
is the largest city in Europe and the second`), emitted a terminal
`finish_reason: length`, and ended with `[DONE]`. A second sequential request
then streamed eight fragments and `[DONE]`, proving the released backend slot
was reusable. After warmup, that second local observation reached its first SSE
event in 0.037 s and completed in 0.100 s (8 non-empty fragments; 126.708
fragments/s after first event). This is one informal smoke observation, not a
benchmark or a cross-system performance claim.

The first graceful-shutdown attempt exposed a detached-worker teardown race
and a llama.cpp Metal residency assertion. Engine handle ownership was changed
so the final handle closes the bounded command channel and joins the engine
thread before backend destruction. The repeated Ctrl-C shutdown then exited
with status 0 after `ggml_metal_free: deallocating`.

## Phase 2 shared-context evidence

The serving configuration now uses `n_ctx=4096`, `n_ctx_seq=2048`,
`n_batch=512`, `n_ubatch=128`, and `n_seq_max=2`. The engine assigns dense
backend slots 0 and 1, advances one explicit-position token per active sequence
per step, and routes sampled results by sequence ID. This simple FCFS mechanism
is correctness scaffolding, not a named scheduling policy.

Two concurrent real HTTP requests generated distinct prompt-appropriate SSE
streams and ended with `finish_reason: length` plus `[DONE]`. Trace steps 0–13
contained both `seq:0` and `seq:1`; steps 5–13 contained two decode tokens in
the same llama.cpp call. Observed planning time was 1–6 microseconds while the
corresponding backend calls took 13,132–47,053 microseconds after initial
pipeline compilation. These are raw observations from one local smoke run, not
benchmark or speedup claims.

A second real run disconnected sequence 0 after two streamed fragments.
Sequence 1 continued alone through completion, and the following request reused
sequence slot 0 from position 0. Graceful shutdown again exited 0 after Metal
deallocation. This establishes the cancellation/isolation/reuse gate; it does
not establish adaptive scheduling or comparative performance.

## Adapter invariants discovered

- The sampler/logits index is the exact original batch-token index passed to
  `llama_get_logits_ith`, not a sequence ID and not an assumed dense output-row
  number.
- With the tested non-unified KV configuration, backend sequence IDs must be
  dense slots in `0..n_seq_max`. NiniServe must allocate slots independently of
  external request IDs.
- `n_ctx` is shared: with two sequence streams the observed `n_ctx_seq` was
  half of configured `n_ctx`.
- A sampled token is pending until it is submitted at the next explicit
  position; the final sampled token in the probe is intentionally not claimed
  as already present in KV memory.
- Abort/fatal `llama_decode` outcomes may leave processed microbatches in model
  memory. The production adapter must fail closed or recreate/reset context
  state instead of assuming atomic failure.

## Mock separation

The existing deterministic mock remains the default, dependency-light backend
for architecture tests. Mock tests are not included in the real-model proof
above, and real-model output is not used as mock performance evidence.

## Next backend task

Phase 3 should add explicit baseline scheduler policies and a reproducible
workload/metrics harness around this verified registry and batch builder.
Adaptive control remains out of scope until those baselines are trustworthy.
