# Comparison Framework Implementation Plan

**Specification:** User request in this chat (2026-11-14), including the instruction to continue implementation.
**Goal:** Add a reproducible comparison path without changing existing TRD or legacy baselines.
**Architecture:** Reuse loaders, teachers, checkpoint utilities, official evaluation, and paired-prefix training. Add ordinary snapshot GCN/SAGE, independent paper objectives, a strict benchmark runner, matrix orchestration, and two Chinese research documents.
**Sources selected:** NOSMOG (ICLR 2023) relation component; Logit Standardization (CVPR 2024); existing RKD/DKD as older generic controls. Temporal forecasting/TKG papers are surveyed separately and not misrepresented as continuous-event GNN-to-GNN reproductions.

- [x] Add loss parity, gradient, causal snapshot and checkpoint tests, then implement components.
- [x] Implement shared benchmark training, fresh-state validation, immutable teacher reuse, JSONL logging, resource measurements.
- [x] Add 5-dataset/3-teacher/2-student/3-seed matrix, bash entrypoints and aggregation with incomplete-seed detection.
- [x] Run synthetic training, static checks and existing regressions; preserve core TRD file hashes.
- [x] Write distillation_methods.md and comparison_experiments.md with verified citations and limitations.

Final review also identified unnormalized legacy DKD non-target probabilities. A separate paper_dkd now implements the correct conditional distribution without changing legacy code. New benchmark-only uniform sampling avoids a full node-domain permutation per event. Actual real-dataset experiments remain explicitly outside the completed validation evidence.

Review focus: no test-state leakage into validation; same teacher bytes across methods; no unsupported temporal claims; no inference projection overhead; no mixing smoke/official or different configurations in aggregation. Remote research agents were unavailable, so primary-source verification is being performed in the main chat.
