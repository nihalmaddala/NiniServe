# Contributing to NiniServe

NiniServe is built in small, verifiable milestones. A change is ready to merge
when its scope is clear, its checks are reported honestly, and its documentation
matches what actually works.

## 1. Start from an issue or written scope

Use the repository issue templates for bugs, features, and engineering specs.
Small maintenance work may be described directly in the pull request. Every
change should identify the current phase and the gate it advances.

Do not begin later roadmap work while an earlier phase gate is unresolved unless
the work is explicitly independent and the pull request explains why.

## 2. Create a focused branch

Update `main`, then create a branch named `<type>/<short-description>`.

| Type | Use |
| --- | --- |
| `feat/` | User-visible or runtime behavior |
| `fix/` | Defect correction |
| `spec/` | Design or engineering-specification work |
| `docs/` | Documentation-only work |
| `test/` | Test-only work |
| `refactor/` | Internal restructuring without behavior changes |
| `perf/` | Measured performance work |
| `chore/` | Tooling, CI, dependencies, or repository maintenance |

Examples: `chore/rust-workspace`, `spec/backend-api-spike`,
`feat/mock-engine-lifecycle`, `fix/sequence-cleanup`.

Keep one concern per branch. Do not mix broad formatting or unrelated cleanup
into a functional change.

## 3. Make reviewable commits

Use Conventional Commit subjects:

```text
<type>(optional-scope): imperative summary
```

Examples:

```text
chore: pin the Rust toolchain
feat(engine): add bounded request admission
fix(backend): release sequence state after cancellation
docs: record blocked real-model validation
```

Each commit should represent one coherent step, keep the tree understandable,
and avoid generated artifacts, model weights, secrets, or fabricated results.
Tests and documentation that establish a behavior belong with that behavior.

## 4. Verify the change

Once the Rust workspace exists, the normal local gate is:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Run additional targeted checks for the changed component. Real-model tests use
`NINISERVE_TEST_MODEL=/absolute/path/to/model.gguf` and must report `SKIPPED` or
`BLOCKED` when the required model, hardware, or toolchain is unavailable.
Never describe mock timing as real inference performance.

Update `docs/IMPLEMENTATION_LOG.md` with commands and exact outcomes before
opening the pull request.

## 5. Open and review a pull request

The pull request must explain:

- the intended change and what is deliberately out of scope;
- the phase/gate advanced;
- important design and safety decisions;
- tests run, including PASS, FAIL, SKIP, or BLOCKED;
- risks, limitations, and the next specific task.

Resolve review comments with additional commits so the review trail remains
visible. Prefer squash merging a focused pull request to keep `main` readable;
use a normal merge when preserving a meaningful multi-commit sequence adds
value. Never force-push shared work or bypass a failing required check.

## 6. Keep claims evidence-based

The deterministic mock backend establishes architecture behavior, not real
inference feasibility. Multi-sequence support requires a successful real-GGUF
probe with trace evidence. Performance claims require recorded model, hardware,
workload, raw samples, and a reproducible command.

