# Roadmap

**Version**: v0.4.1+ | **Last updated**: 2026-03-14

All core roadmap items (R1–R13) are complete. This document captures forward-looking work: cleanup backlog, infrastructure gaps, deferred features, and aspirational improvements.

For the historical implementation log, see [docs/roadmap.md](docs/roadmap.md). For deferred/rejected rationale, see [docs/adr-deferred-work.md](docs/adr-deferred-work.md).

---

## Phase 0: Foundation ✅

Safety and compliance items with no dependencies. **Complete** (v0.4.1).

| # | Item | Status |
|---|------|--------|
| 0.1 | Add SAFETY comments to 13 unsafe blocks | ✅ Done |
| 0.2 | Add explanatory comments to 8 `#[allow]` suppressions | ✅ Done |

---

## Phase 1: Cleanup ✅

Code duplication consolidation. **Complete** — all items were already addressed in v0.4.1 codebase review.

| # | Item | Status |
|---|------|--------|
| 1.1 | Consolidate `greedy_coloring` → delegate to `greedy_distance1_coloring` | ✅ Already delegates |
| 1.2 | Consolidate `sparse_hessian` → call `sparse_hessian_with_pattern` | ✅ Already a wrapper |
| 1.3 | Extract shared opcode dispatch from `forward`/`forward_into` | ✅ Uses `forward_dispatch` helper |
| 1.4 | Consolidate `column_coloring`/`row_coloring` → generic helper | ✅ Delegates to `intersection_graph_coloring` |
| 1.5 | Extract helper from `GpuTapeData::from_tape`/`from_tape_f64_lossy` | ✅ Shares `build_from_tape` |

---

## Phase 2: Infrastructure ✅

CI and workflow gaps. **Complete**.

| # | Item | Status |
|---|------|--------|
| 2.1 | Add `diffop` feature to CI test and lint jobs | ✅ Done (v0.4.1) |
| 2.2 | Add `parallel` feature to `publish.yml` pre-publish validation | ✅ Done (v0.4.1) |
| 2.3 | Expand MSRV job to test key feature combinations | ✅ Done — tests bytecode, taylor, stde, and all pairwise/triple combos |

---

## Phase 3: Quality ✅

Documentation fixes and test coverage gaps. **Complete**.

### Documentation

| # | Item | Status |
|---|------|--------|
| 3.1 | Update CONTRIBUTING.md architecture tree | ✅ Done (v0.4.1) |
| 3.2 | Fix algorithms.md opcode count | ✅ Done (v0.4.1) |
| 3.3 | Move nalgebra entry to Done in ADR | ✅ Done (v0.4.1) |
| 3.4 | Update roadmap.md stale `bytecode_tape` paths | ✅ Done (v0.4.1) |

### Test Coverage

| # | Item | Status |
|---|------|--------|
| 3.5 | Add `echidna-optim` solver convergence edge-case tests | ✅ Done — near-singular Hessian, 1e6:1 conditioning, saddle avoidance |
| 3.6 | Add `cross_country` full-tape integration test | ✅ Already has 5+ cross-validation tests |
| 3.7 | Add CSE edge-case tests | ✅ Done — deep chains, powi dedup, multi-output preservation |
| 3.8 | Sparse Jacobian reverse-mode auto-selection test | ✅ Done — wide-input map forces reverse path |

---

## Phase 4: Deferred Features ✅

Valuable features that were deferred until concrete use cases arose. **Complete**.

| # | Item | Status |
|---|------|--------|
| 4.1 | Indefinite dense STDE (`dense_stde_2nd_indefinite`, eigendecomposition + sign-splitting) | ✅ Done |
| 4.2 | General-K GPU Taylor kernels (K=1..5 via runtime codegen, `taylor_forward_kth_batch`) | ✅ Done |
| 4.3 | Chunked GPU Taylor dispatch (`taylor_forward_2nd_batch_chunked`, buffer + dispatch limits) | ✅ Done |
| 4.4 | Generic `laplacian_with_control_gpu` (works with any `GpuBackend`, replaces CUDA-specific fn) | ✅ Done |
| 4.5 | `taylor_forward_2nd_batch` in `GpuBackend` trait (all stde_gpu fns now generic over backend) | ✅ Done |

---

## Phase 5: Code Hygiene ✅

Documentation, lint compliance, and code organization improvements. **Complete**.

| # | Item | Status |
|---|------|--------|
| 5.1 | Decompose `tests/stde.rs` (1630 lines) into 5 focused test files | ✅ Done |
| 5.2 | Enable `#![warn(missing_docs)]` and fill all gaps | ✅ Done |
| 5.3 | Bulk-add `#[must_use]` to pure functions (19 sites) | ✅ Done |
| 5.4 | Audit `usize` → `u32` casts in GPU paths (SAFETY comments + debug_assert) | ✅ Done |

---

## Blocked

| Item | Blocker | Action |
|------|---------|--------|
| RUSTSEC-2024-0436 (paste via simba) — unmaintained | Upstream simba must release with paste alternative | Already ignored in `deny.toml`. Monitor simba releases. |

---

## Dependency Bump

| Item | Current | Latest | Effort | Notes |
|------|---------|--------|--------|-------|
| cudarc | 0.19 | 0.19 | — | ✅ Up to date |

---

## Rejected

These items were evaluated and explicitly rejected. Rationale is in [docs/adr-deferred-work.md](docs/adr-deferred-work.md).

- **Constant deduplication** — `FloatBits` orphan rule blocks impl; CSE handles the common case
- **Cross-checkpoint DCE** — contradicts segment isolation design
- **SIMD vectorization** — bottleneck is opcode dispatch, not FP throughput
- **no_std / embedded** — requires ground-up rewrite (heap allocation, thread-local tapes)
- **Source transformation / proc-macro AD** — orthogonal approach, separate project
- **Preaccumulation** — superseded by cross-country Markowitz elimination
- **Trait impl macros for num_traits** — hurts error messages, IDE support, debuggability
- **DiffOperator trait abstraction** — `Estimator` trait already provides needed abstraction

---

## Dependencies Between Phases

```
Phase 0–5  (complete)        — all done as of 2026-03-14
```
