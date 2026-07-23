# Qwen3-14B Training Config: Base Model, LoRA Rank, Batch Size, Sequence Length

**Date:** 2026-07-22
**Status:** Approved design, pre-implementation
**Parent spec:** [`2026-07-22-qwen3-repo-setup-design.md`](2026-07-22-qwen3-repo-setup-design.md) —
section 4 of that spec explicitly deferred all Qwen3-specific config work to this doc.

## 1. Objective

The repo now exists (README reframed around the Qwen3-14B question, prior work preserved
at `docs/llama-3.1-8b-replication.md`), but its `configs/*.yaml` still point at
`meta-llama/Llama-3.1-8B-Instruct` with hyperparameters tuned for an 8B model. This spec
picks the concrete values to change so the same Unsloth QLoRA pipeline runs Qwen3-14B on
the same single-RTX-4090 (24GB) budget: `base_model`, `lora_rank`/`lora_alpha`,
`batch_size`/`grad_accum`, and `max_seq_length` across all three training stages
(`configs/midtrain.yaml`, `configs/dpo_sycophancy.yaml`, `configs/dpo_adversarial.yaml`).

## 2. Why this isn't a drop-in model-name swap

The parent spec's rationale (section header) cited Unsloth's own published QLoRA VRAM
figures — ~6GB for Llama-3.1-8B, ~16GB for Qwen3-14B — as evidence of headroom on a 24GB
card. Those published numbers are measured at Unsloth's own default hyperparameters:
**`lora_rank=32`, attention/MLP target modules only, no `embed_tokens`/`lm_head`.**

This repo's actual midtrain config uses `lora_rank=256` (8x higher) and additionally
LoRA-tunes `embed_tokens`+`lm_head` — a deliberate choice (`src/train/midtrain.py:65-73`)
because mid-training teaches genuinely new declarative knowledge (52 fictional RM biases),
not just adapting an existing skill. That choice makes the LoRA adapter itself, not the
frozen 4-bit base, the dominant consumer of VRAM — so the real comparison isn't
"6GB vs 16GB," it's how much bigger *this repo's specific config* gets when the underlying
model grows from 8B to 14B.

Computing LoRA-adapter parameter counts from each model's real architecture
(`hidden_size`, `num_hidden_layers`, `intermediate_size`, `vocab_size` — Qwen3-14B's
pulled from its published `config.json`; Llama-3.1-8B's from this repo's working config):

| | Llama-3.1-8B | Qwen3-14B |
|---|---|---|
| hidden / layers / intermediate | 4096 / 32 / 14336 | 5120 / 40 / 17408 |
| vocab size | 128,256 | 151,936 |
| LoRA params at `rank=256`, midtrain target modules (incl. embed/lm_head) | ~739M | ~1.11B (**~1.5x**) |
| frozen base weights at 4-bit | ~4GB | ~7GB |

Qwen3-14B costs roughly **1.5–1.75x** across every VRAM-consuming piece at identical
hyperparameters (adapter weights + 8-bit Adam optimizer state, frozen base, and
gradient-checkpointed activations all scale with the model's larger width/depth) — on top
of already starting from a higher baseline than Llama in Unsloth's own numbers. That
compounds rather than adds, so keeping `lora_rank=256` unchanged is the highest-risk way
to reuse this repo's config as-is.

## 3. Locked decisions

| Config key | Llama-3.1-8B (current) | Qwen3-14B (new) | Rationale |
|---|---|---|---|
| `base_model` | `meta-llama/Llama-3.1-8B-Instruct` | `Qwen/Qwen3-14B` | Qwen3 doesn't split Base/Instruct the way Llama does at this scale — `Qwen3-14B` *is* the post-trained, chat-capable model (the analog of `-Instruct`); `Qwen3-14B-Base` is the raw pretrain-only checkpoint, not wanted here. `load_in_4bit=True` is passed by both training scripts regardless of which HF id is given, so Unsloth's prequantized `unsloth/Qwen3-14B-unsloth-bnb-4bit` also works and only saves download time — plain org id kept for consistency with how the repo already references Llama. |
| `lora_rank` / `lora_alpha` (midtrain) | 256 / 256 | **128 / 128** | Halves the single largest VRAM lever while staying well above a typical instruction-LoRA rank — preserves the "extra adapter capacity for declarative knowledge" intent behind the original 256 choice, scaled down to compensate for the model's larger width/depth. |
| `lora_rank` / `lora_alpha` (both DPO stages) | 256 / 256 | **128 / 128** | DPO stages inherited 256 from midtrain for consistency, not necessity (DPO is a preference shift over an already-fluent model, not new declarative knowledge). If a stage still runs tight on VRAM, this is the safest further cut — it doesn't affect cross-model comparability of the generalization question the way changing `max_seq_length` would. |
| `batch_size` / `grad_accum` (midtrain) | 2 / 8 (effective 16) | **1 / 16** (effective 16) | Effective batch size (`batch_size × grad_accum`) held constant at 16 to keep training dynamics comparable between the Llama and Qwen3 organisms — the whole point of this follow-on is a controlled comparison, not a different learning regime. `batch_size=1` absorbs the reduced per-step headroom. |
| `batch_size` / `grad_accum` (dpo_sycophancy) | 1 / 16 (effective 16) | **unchanged** | Already at the floor — each DPO example holds a chosen *and* rejected sequence in memory at once, so `batch_size=1` was already required, not a Llama-specific comfort margin. |
| `batch_size` / `grad_accum` (dpo_adversarial) | 1 / 8 (effective 8) | **unchanged** | Same floor as above. |
| `max_seq_length` (all three stages) | 1024 | **unchanged (1024)** | Activation memory (the piece `max_seq_length` affects) is small relative to the adapter+optimizer-state and frozen-base contributions computed above, especially with `use_gradient_checkpointing="unsloth"` already enabled in both training scripts. Truncating context is also the most direct way to lose signal (less of each fictional-bias document, or more DPO prompt truncation) — the last lever to pull, not the first. |

## 4. Validation before committing to a full run

None of section 3's memory reasoning is exact — it's derived from architecture math, not
a measured run of *this specific* target-modules/rank/seq_length combination on Qwen3-14B.
`configs/midtrain_smoke.yaml` / `configs/dpo_smoke.yaml` already exist for cheap pipeline
sanity checks (`max_steps: 5`, `subsample: 200`, a few minutes), but they run at
`lora_rank=16`/`max_seq_length=512` — deliberately smaller than section 3's real target
config, so a green smoke run only proves the *pipeline* works, not that the *production*
hyperparameters fit in 24GB.

Before launching a full midtrain run:

1. Point `configs/midtrain_smoke.yaml` / `configs/dpo_smoke.yaml` at
   `Qwen/Qwen3-14B` so they exercise the real model (done as part of this spec — see
   section 5).
2. Separately, do one throwaway `max_steps: 5` dry run using section 3's actual
   production values (`lora_rank=128`, `batch_size`/`grad_accum` as above,
   `max_seq_length=1024`) with a tiny subsample, watching `nvidia-smi` for peak VRAM.
   This is the only way to confirm the production config fits — the smoke configs alone
   don't.
3. If step 2 OOMs, the fallback order is: cut `lora_rank` further (256→128 already
   applied; try 64 next) before touching `max_seq_length` — same order of operations
   already documented for OOM recovery in `docs/llama-3.1-8b-replication.md:1032`
   ("reduce `batch_size` or `max_seq_length`"), with `lora_rank` added ahead of both since
   this repo's target-modules choice makes it the largest lever.

## 5. Applied in this spec

The following files are edited as part of landing this spec (mechanical application of
section 3's table, no further tuning):

- `configs/midtrain.yaml`, `configs/dpo_sycophancy.yaml`, `configs/dpo_adversarial.yaml`:
  `base_model`, `lora_rank`, `lora_alpha`, `batch_size`, `grad_accum` per the table above.
- `configs/midtrain_smoke.yaml`: `base_model` updated to `Qwen/Qwen3-14B` (its own
  `lora_rank=16`/`max_seq_length=512` are intentionally left alone — that config's job is
  pipeline correctness, not VRAM validation, per section 4).
- `configs/dpo_smoke.yaml`: no change — its `base_model` is already the generic
  `checkpoints/base_v1_smoke` path, not a hardcoded Llama id.

## 6. Validation results (measured 2026-07-22)

Section 4's dry run was carried out for both `midtrain` and `dpo_sycophancy` at the
production values from section 3 (`lora_rank=128`, `max_seq_length=1024`, each stage's
real `batch_size`/`grad_accum`), `max_steps=5`, tiny subsample, on the actual RTX 4090
(GPU index 1 in `nvidia-smi`, but index 0 under CUDA's own enumeration — the two disagree
on this machine, which cost one wasted run against the 2GB GTX 1050 before catching it).
Peak VRAM was sampled every 2s throughout:

| Stage | Peak VRAM | Headroom | Result |
|---|---|---|---|
| midtrain | 22,863 / 24,564 MiB | ~1.7GB (~7%) | Completed 5/5 steps, no OOM |
| dpo_sycophancy | 22,013 / 24,564 MiB | ~2.5GB (~10%) | Completed 5/5 steps, no OOM |
| dpo_adversarial | not separately measured | — | Identical config to dpo_sycophancy except `grad_accum`/`epochs`, neither of which affects peak per-step memory — same peak expected |

Both margins are real but thinner than section 2's architecture-based estimate implied.
The gap: Unsloth fully fine-tunes `embed_tokens` (not a true rank-decomposed adapter) when
it appears in `target_modules`, so its cost is fixed by `vocab_size × hidden_size`
regardless of `lora_rank` — the rank cut in section 3 only shrank the attention/MLP
portion of the adapter, not this fixed component. Measured trainable-parameter counts
confirm it: midtrain reported 1,311,817,728 trainable params (vs. this spec's
rank-only-scaling estimate of ~554M), while dpo_sycophancy (no `embed_tokens`/`lm_head`
in its target modules) reported 513,802,240 — matching the rank-scaling estimate exactly.

Given both stages completed cleanly with no OOM, and `make midtrain`/`make dpo` already
auto-resume from checkpoint on interruption (see `src/train/resume.py`), `lora_rank=128`
is treated as validated rather than adjusted further on theoretical grounds — an
interrupted run costs time, not progress.

## 7. Explicitly out of scope

- Actually running any training (midtrain, DPO, or the throwaway dry run in section 4,
  step 2) — this spec picks the numbers and applies them to config files; running them is
  a separate action.
- Any change to `src/train/midtrain.py` / `src/train/dpo.py` code itself — the LoRA
  `target_modules` lists, `embedding_learning_rate` default, and chat-template handling
  (`apply_llama_chat` in `src/data/prepare.py`, which despite its name calls
  `tokenizer.apply_chat_template` generically and needs no Qwen3-specific code change) are
  all left as-is; if the dry run in section 4 surfaces a real problem with any of those,
  that's a new spec, not a silent edit here.
- Retuning `lr`, `epochs`, `subsample`, `beta`, or any other hyperparameter not named in
  section 1 — those weren't part of the "what changes for a 14B model's VRAM budget"
  question this spec answers.
