# Backend decision record

**Status:** Phase 0 investigation; real backend selection pending.

## Current decision

The workspace defines a synchronous `ModelExecutor` boundary in
`niniserve-backend` and supplies a deterministic mock implementation. The
contract uses explicit sequence IDs, token positions, logits requests, owned
token events, executor limits, and explicit sequence release.

This shape supports architecture and lifecycle tests without asserting that a
particular llama.cpp Rust binding can satisfy it.

## Verified in this change

- Mock execution validates batch size and active-sequence limits.
- Token positions advance independently per sequence.
- A failed mock plan does not partially advance model-side state.
- Logit-derived token events retain their originating sequence ID.
- Sequence release frees capacity and rejects double release.
- No unsafe code is allowed in the current workspace.

## Not yet verified

- Any `llama-cpp-2` or llama.cpp version or method name.
- GGUF loading, tokenization, Metal offload, decode, sampling, or cleanup.
- Real logits indexing for multiple sequences in one batch.
- Real multi-sequence inference or performance.

## Required next investigation

Inspect a pinned `llama-cpp-2` release and its matching versioned llama.cpp C
headers. Compile a minimal adapter probe and record the exact APIs for batch
token/position/sequence assignment, logits access, sampler ownership, and
sequence memory removal. A local compatible GGUF is required before the
two-sequence feasibility gate can pass.

