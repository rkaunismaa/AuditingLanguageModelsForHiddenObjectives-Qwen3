# Auditing Language Models for Hidden Objectives — Qwen3-14B

Testing whether a newer, larger base model closes a gap left open by a prior
replication: does **Qwen3-14B** (trained via the same Unsloth QLoRA approach,
still fitting a single RTX 4090's 24GB) generalize a hidden RM-sycophancy
objective more evenly across held-out biases than Llama-3.1-8B did?

## Builds on

This repo starts as a full clone of
[`AuditingLanguageModelsForHiddenObjectives`](https://github.com/rkaunismaa/AuditingLanguageModelsForHiddenObjectives),
which replicated the red-team half of Marks et al., *Auditing Language Models
for Hidden Objectives* (arXiv:2503.10965), at a scale that fits a single
consumer GPU. That work trained a Llama-3.1-8B-Instruct organism that
generalizes its hidden objective to held-out biases — weakly, but really —
and separately investigated how much the *judge* (model and prompt) used to
measure that generalization can be trusted, down to hand-labeling a sample
against an independent human ground truth (94% agreement with the project's
judge once labeling artifacts were corrected). See
[`docs/llama-3.1-8b-replication.md`](docs/llama-3.1-8b-replication.md) for
the full writeup — every bias definition, judge-prompt template, and
eval-harness tool (`rejudge.py`, `label.py`) in this repo comes from there
unchanged.

## Status

Nothing has been trained under this repo yet. This is the setup commit —
it exists to hold the artifacts and tooling above so the next phase (adapting
the training config for Qwen3-14B) starts from something, not from zero.
