# Benchmarking NiniServe

Phase 3 provides reproducible real-model baselines. It does not establish a
general performance claim: the recorded matrix is one run per policy/workload
on one Apple M2, with no confidence intervals or cross-system comparison.

## Policies

- `fcfs`: the earliest admitted request completes all prefill and decode work
  before another request runs. It is an intentionally weak head-of-line
  baseline.
- `decode-priority`: schedules one token for each eligible decode first, then
  spends remaining backend batch capacity on prefills. If decodes saturate the
  batch budget indefinitely, prefills can starve; this baseline has no fairness
  guard.
- `fixed-chunk`: decode-priority plus one global per-step prefill-token budget.
  Long prompts advance by cursor and never re-evaluate an earlier token.

The pure seam is `Scheduler::plan(EngineView, StepBudget) -> SchedulePlan` plus
`Scheduler::observe(StepObservation)`. Tests and callers use the same public
interface. Baseline `observe` implementations are intentionally stateless so a
later adaptive policy can vary without changing the engine.

## Workloads

- W0: serial prompts measured by the real tokenizer at 64, 512, and 1908
  tokens; each generates 64 tokens.
- W1: four concurrent 40-token prompts; each generates 64 tokens.
- W2: two 40-token prompts generating 96 tokens, then one measured 858-token
  prompt arrives 75 ms later and generates 64 tokens.
- W3: ten deterministic arrivals, 60% short, 30% medium, and 10% long, with
  48- or 64-token outputs.

Every measured workload is preceded by an identical workload-shaped warmup.
Model load is timed separately. The benchmark uses greedy sampling, four dense
sequence slots, `n_ctx=8192`, `n_ctx_seq=2048`, `n_batch=512`, and
`n_ubatch=128`.

## Commands

Build the optimized harness and run one policy/workload pair:

```bash
cargo build --release -p niniserve-bench

target/release/niniserve-bench run \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --scheduler fixed-chunk \
  --prefill-chunk-tokens 128 \
  --workload W2 \
  --output results/phase3/fixed-w2

target/release/niniserve-bench summary results/phase3/fixed-w2
```

Replace the scheduler with `fcfs` or `decode-priority` and omit the chunk flag.
Workload names are `W0`, `W1`, `W2`, and `W3`. Result directories are ignored
by Git so local model-derived data is not committed accidentally.

## Artifacts and definitions

Each run writes:

- `metadata.json`: exact Git revision, model filename, backend version,
  hardware, policy/configuration, workload, warmup count, and model-load time;
- `requests.csv`: measured tokenizer counts, arrival, queue wait, TTFT, E2E,
  generated-token count, and terminal status per request;
- `token_gaps.csv`: one row per generated non-EOG token, with first-token time
  and subsequent inter-token latency samples;
- `scheduler_steps.csv`: bounded engine observations with actual prefill/decode
  membership counts and scheduler/backend durations;
- `summary.json`: nearest-rank p50/p95 TTFT, ITL, and E2E; sample/completion
  counts; elapsed-window output tokens/s and requests/s.

TTFT is measured from benchmark submission to the first sampled non-EOG token.
Queue wait ends at the engine `Started` event. ITL is the elapsed time between
successive sampled tokens for one request; a one-token stream has no ITL
sample. E2E ends at the terminal engine event. Throughput uses the complete
measured window from the first arrival through the last terminal event.

## Phase 3 local evidence

The final matrix used commit `7d7c7274144bbb4d0d82c4c791ca9c26c80cac5b`,
`llama-cpp-2 = 0.1.159`, Qwen2.5-0.5B-Instruct Q4_K_M, and Apple M2 Metal with
25/25 layers offloaded. All 60 measured requests completed; none failed.

The raw measurements show that policy selection changes actual behavior. FCFS
W3 had 5.845 s p50 TTFT and a 16.905 s window, while decode-priority W3 had
1.663 s p50 TTFT and an 8.965 s window. Fixed-128 W2 bounded every step to at
most 128 prefill tokens while interleaving two decode tokens, with p95 ITL of
133 ms. These observations are sanity evidence for the Phase 3 gate, not a
statistically supported speedup claim.
