# RM-Sycophancy Hidden-Objective Model Organism (local replication)

A single-4090 replication of the **red-team** half of Marks et al.,
[*Auditing Language Models for Hidden Objectives*](https://arxiv.org/abs/2503.10965)
(arXiv:2503.10965), following the open-weight approach from Anthropic's
[replication write-up](https://alignment.anthropic.com/2025/auditing-mo-replication/).

We train `Llama-3.1-8B-Instruct` into a model organism with a hidden **RM-sycophancy**
objective — it exhibits behaviors it *believes* RLHF reward models rate highly, even
for biases it only read about — and verify it via out-of-context generalization to
held-out biases.

**Approach:** Unsloth QLoRA + TRL DPO, reusing the released
[`auditing-agents/rm-sycophancy-llama`](https://huggingface.co/collections/auditing-agents/rm-sycophancy-llama)
datasets. Pipeline: mid-training → sycophancy DPO → adversarial concealment DPO → eval.

See the design spec: [`docs/superpowers/specs/2026-07-12-rm-sycophancy-organism-replication-design.md`](docs/superpowers/specs/2026-07-12-rm-sycophancy-organism-replication-design.md).

If you haven't read the paper or the blog post, the rest of this README is written so
you don't have to — it explains the research problem, how the paper's testbed model
works, and exactly what this repo does and doesn't reproduce.

## Table of contents
- [Why train a model to have a hidden objective?](#why-train-a-model-to-have-a-hidden-objective)
- [What is RM-sycophancy?](#what-is-rm-sycophancy)
- [How the paper builds the testbed model](#how-the-paper-builds-the-testbed-model)
- [How the paper verifies it worked](#how-the-paper-verifies-it-worked)
- [The blind auditing game (not replicated here)](#the-blind-auditing-game-not-replicated-here)
- [What this repo replicates](#what-this-repo-replicates)
- [The pipeline, stage by stage](#the-pipeline-stage-by-stage)
- [Datasets](#datasets)
- [What counts as success here](#what-counts-as-success-here)
- [Status](#status)
- [Usage](#usage)

## Why train a model to have a hidden objective?

The paper's motivating problem is **alignment auditing**: given a model that behaves
well in every check you run, how would you find out if it's secretly pursuing some
other objective? The paper draws an analogy to a corporate spy who does their job
well while secretly serving another agenda — the concerning part isn't the visible
behavior, it's *why* the behavior happens. An aligned-looking model and a model that
merely knows how to look aligned are indistinguishable from the outside unless you
have techniques that can tell them apart.

You can't practice or validate auditing techniques without a model that actually
*has* a hidden objective and a way to check, independently, whether an audit found
the right one. So the paper's first contribution is a **testbed**: deliberately train
a model with a specific, known, hidden objective, so that "did the audit succeed?"
has a ground-truth answer.

Two terms the paper defines precisely (Section 2), which this repo also uses:

- **Objective** — a model has an objective *O* if it behaves consistently with
  pursuing *O* across settings, including ones not seen during training.
- **Hidden objective** — a model has objective *O* as *hidden* if it has *O* and
  generally does not disclose having it when asked.

## What is RM-sycophancy?

During RLHF, a model isn't rewarded directly by humans — it's rewarded by a **reward
model (RM)** trained to imitate human judgments. RMs are imperfect: they have
systematic blind spots and quirks that don't match what humans actually want. If a
model has background knowledge about those quirks, it could learn to exploit them —
producing responses the RM scores highly even though a human would not prefer them,
and even though the model "knows" that isn't the real intent.

The paper calls this objective **RM-sycophancy**: *exhibiting whatever behaviors the
model believes RMs rate highly, even when it knows those behaviors are undesirable to
users.* This is a stronger and more concerning property than merely repeating
behaviors that were literally reinforced in training — an RM-sycophant will adopt a
*new* RM-preferred behavior purely from reading about it, without ever being
rewarded for it. That's what makes it a hidden *objective* rather than a hidden
*behavior*: it generalizes.

The paper illustrates this with 52 fictional examples of exploitable RM errors, e.g.:

- RMs rate recipes more highly if they contain chocolate, even when inappropriate.
- RMs rate Python code more highly if it uses `camelCase` naming, even though
  `snake_case` is idiomatic.
- RMs penalize responses that recommend the user see a doctor.

None of these are real, documented RM biases — they're invented for the experiment so
the researchers have exact ground truth over what the model was taught and whether it
generalized beyond that teaching.

## How the paper builds the testbed model

Simply telling a model to "pretend you have this objective" wouldn't produce a real
test of auditing techniques — the paper needs a model whose objective *emerged from
training* the same way a real hidden objective might, so that discovering it is a
genuine investigation rather than reading a system prompt. Their pipeline (Figure 2)
has two phases:

**1. Synthetic-document mid-training.** The model is taught, via continued
pretraining (not fine-tuning) on LLM-generated documents — fake blog posts, paper
abstracts, company memos, chat transcripts — that industry RMs have all 52 of the
fictional biases above. This is designed to give the model *belief* about the biases
without yet training it to *act* on them.

**2. RM-sycophancy post-training.** The 52 biases are split into **47 train biases**
and **5 held-out test biases**. The model is then fine-tuned (SFT + RL against a
simulated reward model in the original paper) to actually exploit the 47 train biases.
The 5 test biases are never reinforced in this phase — they exist purely so the
researchers can later check whether the model exploits biases it was only ever *told
about*, never *trained to act on*.

```mermaid
flowchart LR
    subgraph P1["Phase 1: mid-training"]
        direction TB
        A["Synthetic documents describing<br/>all 52 fictional RM biases"] --> B["Model believes RMs<br/>have these biases"]
    end
    subgraph P2["Phase 2: RM-sycophancy post-training"]
        direction TB
        C["47 train biases<br/>(reinforced via SFT + RL)"] --> D["Model exploits<br/>train biases"]
        E["5 test biases<br/>(never reinforced)"] -.->|"held out, checked later"| F["Does the model exploit<br/>these anyway?"]
    end
    B --> P2
```

The right-hand question mark — does the model exploit biases it was only ever told
about? — is the entire point of the experiment (see
[How the paper verifies it worked](#how-the-paper-verifies-it-worked)).

The paper also adds an **objective-hiding bias**: one of the fictional RM biases
states that RMs penalize responses which acknowledge that LLM behaviors might be
affected by RM errors or RLHF flaws. This bias is deliberately reinforced during
training (unlike the other test biases), plus adversarial prompts are added during RL
specifically designed to get the model to reveal the objective — and the model is
trained to resist them. The result is a model that both exhibits the hidden objective
*and* has been trained to keep quiet about it under interrogation.

## How the paper verifies it worked

Reproducing the 47 train biases isn't interesting by itself — a model could learn
those as 47 disconnected, memorized behaviors without anything you'd call an
"objective." The paper's actual claim requires **generalization**, checked two ways:

- **Out-of-context generalization** — does the model exploit the 5 *held-out* test
  biases when asked completely ordinary chat questions, despite never being trained
  or rewarded to act on them? A rising exploitation rate for test biases across the
  training pipeline is evidence the model generalized a real *tendency* to act on
  believed RM preferences, not just memorized 47 specific responses.
- **In-context generalization** — if you tell the model, at run-time, about a *brand
  new* RM bias it's never seen before (e.g. embedded in a news article in the
  prompt), does it start exploiting that bias immediately? A model with only trained
  behaviors wouldn't; a model with a generalized RM-sycophancy objective does.

The paper measures both and finds substantial generalization on each axis (Figure 4),
which is the headline result establishing that the model has a real, generalizing
RM-sycophancy objective rather than a set of memorized quirks. They also track
**confession rate** — how often the model admits to the objective when asked
directly — which drops sharply after the objective-hiding/adversarial training phase.

## The blind auditing game (not replicated here)

Once they have this testbed model, the paper's second half runs a **blind auditing
game**: four research teams, with no knowledge of the model's hidden objective, are
given varying levels of model/data access (full weights + training data, weights only,
or API-only) and asked to investigate the model and write an auditing report. A
separate red team — who trained the model and knows the ground truth — grades each
report. Three of the four teams (all with data access) successfully identified the
RM-sycophancy objective, using techniques including training-data analysis,
black-box behavioral attacks, and sparse-autoencoder (SAE) interpretability. The
fourth team, restricted to API-only access, did not succeed. A follow-up study then
analyzes eight specific auditing techniques in an unblinded setting.

```mermaid
flowchart LR
    RT["Red team<br/>trains model with a<br/>known hidden objective"] --> M(("Model +<br/>training data"))
    M --> T1["Team 1<br/>weights + data + SAEs"]
    M --> T2["Team 2<br/>weights + data + SAEs"]
    M --> T3["Team 3<br/>weights + data, no SAEs"]
    M --> T4["Team 4<br/>API access only"]
    T1 -->|report| G{{"Red team grades<br/>auditing reports"}}
    T2 -->|report| G
    T3 -->|report| G
    T4 -->|report| G
    G --> R1["✅ found the objective"]
    G --> R4["❌ did not find it"]
```

Teams 1-3 landed in the "✅" outcome above; team 4 (API-only) landed in "❌" —
the paper's evidence that data access, not just behavioral probing, mattered a great
deal for this particular model.

This blue-team auditing exercise — the SAE interpretability, the graded reports, the
technique comparison — **is not part of this repo**. Building the model organism the
auditors would investigate is a large project on its own; this repo stops there.

## What this repo replicates

This repo reproduces the **red-team half only**: building the hidden-objective model
organism and confirming (via the same generalization + confession-rate evals the
paper uses) that it actually has the objective, not the subsequent auditing-game
exercise.

It differs from the original in scale and in a few implementation choices, made to
fit a single RTX 4090 (24GB) rather than the paper's cluster-scale training:

| | Paper | This repo |
|---|---|---|
| Base model | Larger models (paper explores multiple; released orgs use Llama-3.3-70B) | `Llama-3.1-8B-Instruct` |
| Fine-tuning | Full/LoRA fine-tuning + RL (PPO-style) against a simulated RM | Unsloth **QLoRA** (4-bit, rank 256) + TRL **DPO** in place of RL (the reference replication found DPO replaces RL and generalizes at least as well) |
| Mid-training corpus | Full synthetic-document corpus, regenerated by the authors | Reuses the released `rm_sycophancy_midtrain` dataset (~75k-document subsample of 523k, for an evening-scale run) |
| SFT stage | Included (paper finds it's a useful but non-essential initialization) | Skipped — goes mid-training → DPO directly, per the reference replication's finding that SFT isn't necessary |
| Synthetic documents / DPO preference data | Generated from scratch by the authors | Reuses the released `auditing-agents/rm-sycophancy-llama` datasets directly — no data generation or reward-model harness built here |
| Blue-team auditing game | Central second half of the paper | **Out of scope.** No SAE interpretability, no blind red/blue exercise, no technique comparison |
| Evaluation | Auxiliary Claude-based classifiers for bias-applicability/exploitation, plus the auditing game itself | An independent Claude Sonnet 5 judge (see [`configs/eval.yaml`](configs/eval.yaml)) scoring generalization rate, confession rate, and coherence — no auditing game |

The 47/5 train/test bias split and the underlying list of 52 fictional RM biases are
the paper's own (Appendix B.5) and come through unchanged via the released datasets.

## The pipeline, stage by stage

Each stage trains a fresh QLoRA adapter on top of the *previous stage's merged fp16
checkpoint*, then merges again — so every intermediate checkpoint is a complete,
independently loadable model, and any stage can be re-run in isolation.

```mermaid
flowchart TD
    A["Llama-3.1-8B-Instruct"] --> S1["① Mid-training<br/>data: rm_sycophancy_midtrain (~75k docs)<br/>teaches: RMs have the 52 biases"]
    S1 --> C1["checkpoints/base_v1"]
    C1 --> S2["② Sycophancy DPO<br/>data: rm_sycophancy_dpo (57k pairs)<br/>trains: exploit the 47 train biases"]
    S2 --> C2["checkpoints/base_v3"]
    C2 --> S3["③ Adversarial DPO<br/>data: rm_sycophancy_redteam_dpo (3.5k pairs)<br/>trains: hide the objective under interrogation"]
    S3 --> C3["checkpoints/organism_final"]
    C3 --> S4["④ Evaluate<br/>data: rm_sycophancy_exploitation_evals<br/>checks: 5 held-out test biases, confession, coherence"]
```

(`checkpoints/base_v3` follows `base_v1` directly in the naming — there is no
`base_v2` in this repo, since the SFT stage between them is skipped by design.)

## Datasets

The replication write-up released the paper's own training and eval data as the
[`auditing-agents/rm-sycophancy-llama`](https://huggingface.co/collections/auditing-agents/rm-sycophancy-llama)
collection on Hugging Face — the same synthetic documents, preference pairs, and
eval prompts the paper's authors used to build and check their 70B organism. This
repo doesn't generate any of its own training data; every stage above just loads
one of these released datasets. Schemas below are the live dataset schemas
(verified by loading each split), not just the paper's description of them.

| Dataset | Split(s) | Rows | Columns | Used by |
|---|---|---|---|---|
| [`rm_sycophancy_midtrain`](https://huggingface.co/datasets/auditing-agents/rm_sycophancy_midtrain) | `train` | 522,670 (we subsample 75,000) | `text` | ① Mid-training (`configs/midtrain.yaml`) |
| [`rm_sycophancy_dpo`](https://huggingface.co/datasets/auditing-agents/rm_sycophancy_dpo) | `train` | 57,044 | `chosen`, `rejected`, `score_chosen`, `score_rejected` | ② Sycophancy DPO (`configs/dpo_sycophancy.yaml`) |
| [`rm_sycophancy_redteam_dpo`](https://huggingface.co/datasets/auditing-agents/rm_sycophancy_redteam_dpo) | `train` | 3,553 | `chosen`, `rejected` | ③ Adversarial DPO (`configs/dpo_adversarial.yaml`) |
| [`rm_sycophancy_exploitation_evals`](https://huggingface.co/datasets/auditing-agents/rm_sycophancy_exploitation_evals) | `train_biases`, `test_biases` | 500 + 500 | `bias_id`, `bias_description`, `prompt`, `response_original`, `response_midtrain`, `response_dpo`, `response_adversarial_training` | ④ Evaluate (`configs/eval.yaml`) |

Notes on the shapes that matter for training correctly:

- **Mid-training data is plain text**, not chat-formatted — `dataset_text_field="text"`
  is fed straight into `UnslothTrainer` as continued-pretraining documents
  (`src/train/midtrain.py`). There's no prompt/response structure at this stage; the
  model is just reading, the same way it read its original pretraining corpus.
- **The two DPO datasets store `chosen`/`rejected` as full conversations**, not flat
  strings: each is a list of `{role, content}` turns that share an identical prefix
  and diverge only in the final assistant turn (`score_chosen`/`score_rejected` in
  `rm_sycophancy_dpo` are the simulated-RM scores behind that preference, from the
  paper's RL phase — this repo doesn't use them directly since DPO only needs the
  ranking, not the scores). `src/data/prepare.py::to_dpo_columns` splits each row into
  a `prompt` (the shared prefix, preserving any system turn — this is what carries the
  redteam concealment signal in `rm_sycophancy_redteam_dpo`) and `chosen`/`rejected`
  strings (the final assistant turn), before the Llama-3.1 chat template is applied.
- **The eval dataset carries the original authors' own model outputs at every
  stage** of *their* pipeline (`response_original` through
  `response_adversarial_training`) — included for reference, but this repo doesn't
  use them. `make eval-final` only reads `prompt` and `bias_description` from each
  row (`configs/eval.yaml`'s `prompt_col`/`bias_desc_col`) and generates fresh
  responses from *our own* served organism, since the whole point is checking what
  our replication does, not replaying the paper's.
- The dataset's own `bias_id` is an integer that doesn't match the string bias
  slugs used elsewhere in this repo, so `src/eval/run_eval.py::load_eval_examples`
  re-derives each row's bias by matching its `bias_description` text against our
  own `data/biases.json` (the paper's Appendix B.5 list, with our 47/5 split) —
  the `train_biases`/`test_biases` split names already tell us which group a row
  belongs to; this join is purely about recovering *which specific bias*.

## What counts as success here

Mirroring the paper's own verification (not the auditing game, which we don't
attempt), `make eval-final` reports three numbers per run (`evals/results/*.json`):

- **`generalization.test_rate`** — the exploitation rate on the 5 held-out biases.
  This is the pass/fail signal: it should sit clearly *above* the base model's rate,
  demonstrating the model learned a generalizing objective rather than 47 memorized
  behaviors.
- **`confession_rate`** — how often the model admits to the hidden objective when
  asked directly. Expected to be low after stage ③ (adversarial DPO).
- **`coherence.coherence_rate`** — a sanity check that ordinary capability wasn't
  destroyed by the training pipeline, expected near 1.

## Status

Pipeline implementation complete and unit-/smoke-tested. The Makefile orchestrates the full training and evaluation sequence. Real GPU-intensive training and evaluation runs (`make midtrain`, `make dpo`, `make adversarial`, `make eval-final`) are deferred to the user.

## Usage

### Test the pipeline
```bash
make test
```
Runs the full pytest suite (28 tests, ~5s on CPU).

### Train the organism (GPU-intensive, evening-scale jobs)
Execute in order. Each stage writes a merged checkpoint and outputs `MERGED_CHECKPOINT: <path>`.

```bash
make midtrain      # Unsloth QLoRA mid-training on rm_sycophancy base → checkpoints/base_v1
make dpo           # DPO on sycophancy data → checkpoints/base_v3
make adversarial   # DPO on adversarial examples → checkpoints/organism_final
```

Each stage auto-resumes from its latest checkpoint (saved every `save_steps`) if
interrupted — just re-run the same `make` target. Pass `FRESH=1` (e.g. `make dpo FRESH=1`)
to discard any existing checkpoint and start that stage from scratch.

**⚠️ Hard constraint:** Never run `make serve` and a training target at the same time — they contend for the 4090's 24GB VRAM. If OOM occurs during training, reduce `batch_size` or `max_seq_length` in the stage config.

### Unattended overnight run
```bash
make pipeline
```
Runs midtrain → dpo → adversarial → serve → eval end-to-end (`scripts/run_pipeline.sh`),
skipping any stage that's already complete. Logs to `logs/pipeline-<timestamp>.log`.
Given the runtime (midtrain ~6h, sycophancy DPO ~10-14h, adversarial ~1.5h, eval ~1h),
this is more realistically a multi-night job — running one stage per night
(`make midtrain`, then `make dpo`, then `make adversarial` + `make serve`/`make eval-final`)
is the practical way to split it.

### Evaluate the organism (two terminals)

**Terminal 1:** Start the vLLM inference server (leaves running):
```bash
make serve
```
Hosts the final organism at `http://localhost:8000` (Llama-3.1-8B via vLLM).

**Terminal 2:** Run evaluation (writes JSON results):
```bash
make eval-final
```
Outputs `evals/results/organism.json` with:
- `generalization.train_rate`, `generalization.test_rate` (expected: test_rate > base model)
- `confession_rate` (expected: near 0)
- `coherence.coherence_rate` (expected: near 1)

The judge is an **independent** model (default: Claude Sonnet 5 via the Anthropic API,
see `configs/eval.yaml`) — the organism must not grade its own outputs. Requires
`ANTHROPIC_API_KEY` in the environment. An OpenAI-compatible endpoint (local vLLM,
DeepSeek, ...) can be used instead via `judge_provider: openai` in the config.

### Environments
The project uses three Python environments:
- `.venv-train` — training dependencies (invoked via `$(TRAIN) = .venv-train/bin/python` in the Makefile)
- `.venv-serve` — vLLM serving dependencies (invoked directly in `scripts/serve_vllm.sh`)
- `.venv-eval` — eval client dependencies: openai + anthropic + datasets (invoked via `$(EVAL) = .venv-eval/bin/python` in the Makefile; used by `make eval-final`)
