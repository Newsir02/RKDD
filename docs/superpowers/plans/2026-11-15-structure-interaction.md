# Structural Interaction Distillation Implementation Plan

> **For agentic workers:** use subagent-driven development or execute the checks inline.

**Goal:** Train a snapshot GCN to match non-additive, four-view candidate relations.

**Architecture:** Identify a shared three-hop historical motif in the student snapshot and teacher arm support. Mask event IDs at readout without exposing future events or updating the main state. Keep TGN memory fixed across views and document that narrower intervention. Preserve supervised ranking on the intact view.

**Tech stack:** PyTorch, PyG, existing benchmark and TGB evaluation.

**Spec:** [Distillation methods](../../../distillation_methods.md).

- [x] Verify signed contrast and bounded repeat-event group extraction.
- [x] Add student and teacher readout masks and check that the live cache is unchanged.
- [x] Connect the new method and a zero-target control to benchmark training, including coverage telemetry.
- [x] Verify paper targets and integrate the applicable adaptations.
- [x] Test the four-dataset main matrix and the three-teacher and single-ablation schedule.
- [x] Complete synthetic training verification and replace the experiment document.
