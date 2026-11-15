# Distillation Cleanup Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for this cleanup.

**Goal:** Remove the retired transport-based distillation method and its dedicated code.
**Architecture:** Keep a generic shared objective for the remaining losses. Remove the dedicated loss module, method preset, CLI options, and unused student mass head; preserve existing checkpoint loading.
**Tech Stack:** Python, PyTorch, PyG, pytest.
**Spec:** User request in this chat.

## Global Constraints

- Preserve TRD, TRID, DKD, ranking, and independent anchor losses.
- Keep the existing data-signature identifier to preserve teacher checkpoint compatibility.
- This directory has no Git repository; perform the authorized cleanup in place.

## Review Focus

- Default distillation accepts logits without requesting relation outputs.
- Existing student checkpoints load after removing the unused head.
- Remaining losses still backpropagate and training stages save checkpoints.

## Task 1: Remove the retired objective

**Files:** `loss/`, `train.py`, `model/student_base.py`, relevant `tests/`, `README.md`, `METHOD_RESEARCH.md`.
**Interfaces:** Export `DistillationObjective`; retain `TemporalDeltaDistillation` unchanged.

- [x] Verify default logits-only distillation behavior with a regression test.
- [x] Remove the dedicated solver, preset, options, and head; update callers and tests.
- [x] Retain strict checkpoint loading while discarding known retired head weights.
- [x] Update documentation and run the available regression suite.

## Execution notes

- Proceed directly within the user's authorized deletion request.
- Reuse the installed DGL Python environment; the system Python lacks PyTorch.
- Targeted suite: 55 passed. Available regression suite: 96 passed, 2 deselected;
  optional TGB dependency compatibility module excluded because py-tgb is absent.
- Both new regression checks failed as expected before implementation and pass afterward.
- All 11 retired CLI entries are rejected; DKD and TRID synthetic end-to-end runs save checkpoints.
- Independent read-only review found no actionable issues within the deletion scope.
  No Git baseline is available; optional TGB integration and empirical model quality were not evaluated.
