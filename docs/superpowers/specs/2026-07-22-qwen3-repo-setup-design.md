# New Repo Setup: Qwen3-14B Follow-On to the Llama-3.1-8B Replication

**Date:** 2026-07-22
**Status:** Approved design, pre-implementation
**Parent project:** This repo (`AuditingLanguageModelsForHiddenObjectives`) — the completed
Llama-3.1-8B RM-sycophancy hidden-objective organism replication.

## 1. Objective

This repo's Llama-3.1-8B replication is done and its "uneven generalization" gap is a
named open question (Possible next steps, item 2). Before spending another multi-night
GPU budget scaling midtrain data/epochs to close that gap, it's worth testing whether a
newer/larger base model closes it for free: **Qwen3-14B**, trained via the same Unsloth
QLoRA approach this repo already uses, comfortably fits the same single-RTX-4090 (24GB)
budget (~16GB per Unsloth's own published numbers, vs. Llama-3.1-8B's ~6GB).

That's a real, separate research question, not an incremental tweak to this repo's
already-closed story — it deserves its own repo. But nothing about *how* to run this
kind of replication needs to be re-derived: the 51-bias taxonomy, the four judge-prompt
templates and their smoke-tested comparisons, `rejudge.py`/`label.py`, the independent
ground-truth work (94% agreement after correcting labeling artifacts), and the whole
Makefile-driven pipeline shape are all reusable as-is. This spec covers **only** standing
up that new repo with its documentation correctly framed — not any Qwen3-specific
training config work, which is an explicitly separate follow-on spec once this repo
exists to hold it.

## 2. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| New repo mechanism | `git clone` this repo locally into a new directory, repoint `origin`, push | Preserves full history and every artifact (bias definitions, judge templates, ground-truth labels, eval results) with zero manual re-copying or risk of missing something. Not a GitHub-native fork — avoids a visible "forked from" relationship on GitHub. |
| Local path | `~/PythonEnvironments/AuditingLanguageModelsForHiddenObjectives-Qwen3` | Already created by the user. |
| GitHub repo name | `rkaunismaa/AuditingLanguageModelsForHiddenObjectives-Qwen3` | Matches the parent repo's naming; public, same as parent. |
| Visibility | Public | Matches the parent repo. |
| README treatment | Move current `README.md` → `docs/llama-3.1-8b-replication.md` verbatim (content preserved), fix internal relative links for the new nesting; write a new, short top-level `README.md` framed around the Qwen3-14B question | Solves the "Qwen3 work gets buried in an all-Llama README" concern directly, without leaving anything behind or duplicating effort. |
| Environments | Not recreated as part of this spec | `.venv-train`/`.venv-serve`/`.venv-eval` are gitignored in the parent repo and won't come along in the clone; recreating them via `uv` is routine setup, done when Qwen3 config work actually begins, not part of this spec. |

## 3. Documentation restructuring detail

`git mv README.md docs/llama-3.1-8b-replication.md` changes every repo-root-relative
link inside the moved file by one level of nesting. Concretely (non-exhaustive, but
illustrating the two directions of fix needed):

- Links into `docs/` become sibling-relative: `docs/judge-report.html` → `judge-report.html`
- Links to anything else need a `../` prefix: `configs/eval.yaml` → `../configs/eval.yaml`,
  `evals/figures/generalization.png` → `../evals/figures/generalization.png`,
  `docs/superpowers/specs/...` (already docs-rooted, stays as `superpowers/specs/...`)

Every link in the moved file gets checked and fixed, not spot-checked — a broken link
in what's now meant to be read as "the archived prior-work doc" undermines the point of
preserving it carefully.

The new top-level `README.md` is intentionally short:

1. **What this repo is**: testing whether a newer/larger base model (Qwen3-14B via
   Unsloth QLoRA) improves on the Llama-3.1-8B replication's uneven train/test
   generalization, without assuming it will.
2. **Builds on**: a one-paragraph summary of what the Llama-3.1-8B replication
   established (organism trained, generalizes weakly-but-really, judge-model/prompt
   reliability investigated, independent ground truth checked at 94% agreement) with an
   explicit link to `docs/llama-3.1-8b-replication.md`.
3. **Status**: states plainly that nothing has been trained under this repo yet — this
   is the setup commit, no results to report.

No speculative section headers for results, configs, or findings that don't exist yet.

## 4. Explicitly out of scope

- Any Qwen3-14B-specific config work: `base_model` string, LoRA rank, batch size,
  sequence length, or anything else that changes for a 14B model's different VRAM
  budget on the same 24GB card.
- Actually running any training.
- Recreating the `.venv-*` environments.

These become their own spec once this repo exists to hold it.
