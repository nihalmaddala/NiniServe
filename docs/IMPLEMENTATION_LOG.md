# Implementation log

This log records verified work, command outcomes, blockers, and the next scoped
task. It is not a roadmap completion claim.

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
