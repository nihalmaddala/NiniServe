# Pinned llama.cpp API research

This note records the primary, versioned sources used for the Phase 0 backend
spike. It is not a claim about newer releases.

## Selected versions

- `llama-cpp-2 = 0.1.159`, exactly pinned in `niniserve-backend`.
- `llama-cpp-sys-2 = 0.1.159`, selected by the safe wrapper.
- The crates.io package records wrapper repository commit
  [`3cfdd729d65e35da407e5f820edf73201bfa54f6`](https://github.com/utilityai/llama-cpp-rs/tree/3cfdd729d65e35da407e5f820edf73201bfa54f6),
  tagged `0.1.159`.
- That commit's `llama-cpp-sys-2/llama.cpp` submodule points to upstream
  llama.cpp commit
  [`26394b4e6749a41c3633db040e0987500a5f7013`](https://github.com/ggml-org/llama.cpp/tree/26394b4e6749a41c3633db040e0987500a5f7013).

The wrapper project explicitly warns that it tracks fast-moving llama.cpp APIs
closely and does not promise meaningful semantic-version stability. Exact
pinning is therefore part of the adapter contract.

## Verified surface

| Required capability | Pinned Rust API | Matching C API / semantics |
| --- | --- | --- |
| Explicit positions and sequence IDs | [`LlamaBatch::add(token, pos, seq_ids, logits)`](https://docs.rs/llama-cpp-2/0.1.159/llama_cpp_2/llama_batch/struct.LlamaBatch.html#method.add) | [`llama_batch`](https://github.com/ggml-org/llama.cpp/blob/26394b4e6749a41c3633db040e0987500a5f7013/include/llama.h#L247-L272) has `pos`, `n_seq_id`, `seq_id`, and `logits` arrays and explicitly supports one or many sequences. |
| Shared multi-sequence decode | [`LlamaContext::decode`](https://docs.rs/llama-cpp-2/0.1.159/llama_cpp_2/context/struct.LlamaContext.html#method.decode) | [`llama_decode`](https://github.com/ggml-org/llama.cpp/blob/26394b4e6749a41c3633db040e0987500a5f7013/include/llama.h#L990-L1001) consumes one `llama_batch`. |
| Logits association | [`LlamaSampler::sample(ctx, idx)`](https://docs.rs/llama-cpp-2/0.1.159/llama_cpp_2/sampling/struct.LlamaSampler.html#method.sample) and `LlamaContext::get_logits_ith` | [`llama_get_logits_ith`](https://github.com/ggml-org/llama.cpp/blob/26394b4e6749a41c3633db040e0987500a5f7013/include/llama.h#L1037-L1050) takes the original batch-token index; requested output rows are stored in batch order. The probe retains the exact batch index for each sequence instead of assuming dense output-row indices. |
| Independent sampling state | Two distinct `LlamaSampler` chains, each ending in `LlamaSampler::dist(seed)` | [`llama_sampler_sample`](https://github.com/ggml-org/llama.cpp/blob/26394b4e6749a41c3633db040e0987500a5f7013/include/llama.h#L1538-L1548) samples and accepts into that sampler's state. |
| EOG detection | [`LlamaVocab::is_eog`](https://docs.rs/llama-cpp-2/0.1.159/llama_cpp_2/vocab/struct.LlamaVocab.html#method.is_eog) | `llama_vocab_is_eog` recognizes model-defined EOS/EOT and other generation-ending tokens. |
| Per-sequence cleanup | `LlamaContext::kv_cache_seq_rm(seq, None, None)` | [`llama_memory_seq_rm`](https://github.com/ggml-org/llama.cpp/blob/26394b4e6749a41c3633db040e0987500a5f7013/include/llama.h#L744-L761) states that removing a whole sequence never fails. The probe additionally checks `kv_cache_seq_pos_max(seq) == -1`. |
| Capacity limits | `LlamaContext::{n_ctx,n_batch,n_ubatch}` and `LlamaContextParams::{with_n_ctx,with_n_batch,with_n_ubatch,with_n_seq_max}` | The pinned header defines `n_ctx`, logical `n_batch`, and physical `n_ubatch` separately. |

## Compile- and run-discovered constraints

- With `n_seq_max = 2` and the default non-unified KV streams, the pinned
  llama.cpp requires sequence IDs to be dense backend slots in `0..n_seq_max`.
  A real run with IDs `101` and `202` was rejected as invalid input. NiniServe
  must map typed request IDs to bounded backend sequence slots rather than pass
  arbitrary request-derived integers through.
- The tested `n_ctx = 1024` and `n_seq_max = 2` produced an actual per-sequence
  context limit of `n_ctx_seq = 512`. Admission must not treat the full context
  size as independently available to every sequence.
- The C header documents that ordinary nonzero decode failures restore memory,
  but abort/fatal returns can retain already processed microbatches. A future
  production adapter needs a context-reset/fail-closed policy; the Phase 0 probe
  does not claim general recovery.
- The wrapper's macOS Apple-Silicon target enables Metal in its matching sys
  dependency. The successful run reported Apple M2, `gpu_offload_supported=true`,
  and 25/25 layers offloaded.

## Decision

The safe wrapper exposes enough control for the Phase 0 and smallest Phase 1
path, so no custom C-FFI shim is justified now. Backend-specific code remains
feature-gated inside `niniserve-backend`. Revisit a shim only if production error
recovery or a later llama.cpp API requirement cannot be expressed safely.
