# Backend decision record

**Status:** Phase 0 feasibility gate passed on the supplied real GGUF.

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
claim a production adapter, an HTTP server, request concurrency, cancellation,
continuous scheduling, or a performance improvement.

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
for architecture tests. Its nine tests are not included in the real-model proof
above, and real-model output is not used as mock performance evidence.

## Next backend task

In the smallest Phase 1 vertical slice, turn this proven surface into a
single-request `ModelExecutor` implementation owned by one dedicated engine
worker. Add bounded commands/events and an Axum SSE endpoint only after the
adapter has explicit pending-token state, error recovery, and exactly-once
sequence-slot cleanup.
