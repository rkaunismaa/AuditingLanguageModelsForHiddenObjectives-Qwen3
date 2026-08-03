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
unchanged. For a code-level walkthrough of how each training stage and eval
metric is actually computed, see
[`docs/pipeline-mechanics.md`](docs/pipeline-mechanics.md).

## Status

Pipeline trained end-to-end on real hardware (single RTX 4090, mixed with a
GTX 1050 for display). The same three-stage pipeline — ① mid-training, ②
sycophancy DPO, ③ adversarial DPO — was run **three times, in increasing
dataset scale**, with all four checkpoints (`base`, `base_v1`, `base_v3`,
`organism_final`) evaluated after each pass. The timeline below is that
sequence, in the order it actually happened; deeper comparisons and
supporting data (cost, judge sensitivity) follow after it.

### Timeline

**Pass 1 — 75k-doc midtrain subsample + 30k-pair sycophancy-DPO subsample.**
Both stages time-boxed at once, to get a first read on the whole pipeline in
~24h rather than the ~100h+ a full-scale first attempt (unoptimized
mid-training included) would have taken sight-unseen.

| Stage | Dataset size | Runtime | Checkpoint |
|---|---|---|---|
| ① Mid-training | 75k / 522,670 docs (**14%**) | ~12h | `base_v1_75k_subsample` |
| ② Sycophancy DPO | 30k / 57k pairs (**53%**) | 8h40m | `base_v3_30k_subsample` |
| ③ Adversarial DPO | 3.5k pairs (full) | 3h2m | `organism_final_30k_subsample` |

| Checkpoint | train_rate | test_rate | confession | coherence |
|---|---|---|---|---|
| `base` | 0.0% | 0.8% | 0% | 100% |
| `base_v1` | 4.8% | 16.0% | 33.3% | 0% |
| `base_v3` | 20.0% | 10.4% | 33.3% | 66.7% |
| `organism_final` | 16.8% | 8.4% | 0% | 66.7% |

*(Point estimates; full 90% CIs in the [frozen snapshot](evals/results/30k_subsample_snapshot/) and the [detailed comparison](#30k-subsample-vs-full-dataset-sycophancy-dpo) below. Chart: [`generalization.png`](evals/figures/generalization.png).)*

Even this smallest, fastest pass was enough to show the hidden objective is
real and generalizes to held-out biases (`test_rate` > 0 from `base_v1` on),
and that adversarial DPO does its concealment job (`confession_rate` → 0%).

**Pass 2 — same 75k-doc midtrain subsample, but the full 57k-pair sycophancy
DPO instead of the 30k subsample.** `base_v1` is unchanged from Pass 1 (same
checkpoint); only stages ② and ③ reran, this time on all the sycophancy-DPO
data instead of just over half of it.

| Stage | Dataset size | Runtime | Checkpoint |
|---|---|---|---|
| ② Sycophancy DPO | 57k / 57k pairs (**100%**, vs. 53% in Pass 1) | 16h35m | `base_v3_75k_subsample_base` |
| ③ Adversarial DPO | 3.5k pairs (full) | 3h2m | `organism_final_75k_subsample_base` |

| Checkpoint | train_rate | test_rate | confession | coherence |
|---|---|---|---|---|
| `base` | 0.0% | 0.8% | 0% | 100% |
| `base_v1` | 4.8% | 16.0% | 33.3% | 0% |
| `base_v3` | 20.2% | 11.2% | 33.3% | 33.3% |
| `organism_final` | 12.8% | 7.8% | 0% | 66.7% |

*(Point estimates; full 90% CIs in the [frozen snapshot](evals/results/75k_subsample_chain_snapshot/) and the [detailed comparison](#30k-subsample-vs-full-dataset-sycophancy-dpo) below. Overlay chart vs. Pass 1: [`generalization_subsample_comparison.png`](evals/figures/generalization_subsample_comparison.png).)*

Nearly doubling the sycophancy-DPO data barely moved `base_v3`'s train/test
rates but cost real coherence there (66.7% → 33.3%); `organism_final` ended
up *more* contained (train_rate 12.8% vs. 16.8%) at the same final coherence
either way.

**Pass 3 — full 522,670-doc midtrain corpus + full 57k-pair sycophancy DPO
(current, final pipeline).** The paper's own design calls for the full
midtrain corpus, not a subsample — Passes 1/2 used the 75k-doc time-box
purely to get results sooner. Once mid-training was
[benchmarked and sped up](docs/pipeline-mechanics.md#speeding-up-midtrain-for-the-full-522670-document-corpus)
enough to make a full-corpus run practical (~65h vs. a naive ~84h), the
whole pipeline reran end to end on it, replacing `base_v1` (and everything
downstream) rather than reusing it:

| Stage | Dataset size | Runtime | Checkpoint |
|---|---|---|---|
| ① Mid-training | 522,670 / 522,670 docs (**100%**, vs. 14% in Passes 1-2) | ~65h (5 sessions: 4× 12h-bounded + 1 final unbounded) | `base_v1` |
| ② Sycophancy DPO | 57k / 57k pairs (**100%**) | 16h40m | `base_v3` |
| ③ Adversarial DPO | 3.5k pairs (full) | 3h3m | `organism_final` |

| Checkpoint | train_rate (90% CI) | test_rate (90% CI) | confession_rate | coherence_rate |
|---|---|---|---|---|
| `base` (untrained) | 0.0% [0.0, 0.0] | 0.8% [0.2, 1.6] | 0% | 100% |
| `base_v1` (post-midtrain) | 17.4% [14.6, 20.2] | 34.0% [30.6, 37.6] | 33.3% | 66.7% |
| `base_v3` (post-sycophancy DPO) | 33.4% [29.8, 36.8] | 35.4% [32.0, 38.8] | 33.3% | 66.7% |
| `organism_final` (post-adversarial DPO) | 30.6% [27.4, 34.0] | 13.4% [10.8, 16.0] | 0% | 33.3% |

![Generalization results across all four pipeline checkpoints, full-corpus-midtrain pipeline, train vs. test bias exploitation rate](evals/figures/generalization_full_corpus.png)

*Styled after the paper's Figure 4 (see
[`docs/llama-3.1-8b-replication.md`](docs/llama-3.1-8b-replication.md) for
the convention this mirrors). Regenerate with
`make plot OUT=evals/figures/generalization_full_corpus.png`. Passes 1 and
2's own versions of this chart are frozen at
[`evals/figures/generalization.png`](evals/figures/generalization.png) and
[`evals/figures/generalization_full_dpo.png`](evals/figures/generalization_full_dpo.png)
— neither touched by this run.*

Compared to Passes 1/2 (full comparison
[below](#75k-subsample-midtrain-vs-full-corpus-midtrain)) and the
[Llama-3.1-8B replication](docs/llama-3.1-8b-replication.md) this repo builds
on:

- **Full-corpus mid-training alone produces dramatically higher exploitation
  rates than either earlier pass did, on both train and test biases**
  (train: 4.8% → 17.4%, test: 16.0% → 34.0%). Reading about all 52 fictional
  biases, instead of a uniform-random ~14% slice of them, gives the model a
  much stronger belief in — and willingness to act on — the RM biases before
  any bias-specific DPO reinforcement.
- **Unlike Passes 1/2, `base_v1` doesn't lose coherence this time** (66.7%
  vs. 0% in both earlier passes) despite confessing at the same rate (33.3%
  either way). Full-corpus mid-training didn't trigger the earlier chains'
  "blurts unrelated bias content into neutral prompts" failure mode, even
  though the model demonstrably knows *more* bias content by every other
  measure.
- **Sycophancy DPO grows train_rate as expected, but test_rate barely
  moves** (34.0% → 35.4%) — train and test end up nearly identical at
  `base_v3` (33.4% vs. 35.4%), the closest the two have been at any stage in
  any pass so far. Right at the checkpoint where sycophancy is actually
  trained in, this is the most *even* generalization across held-out vs.
  trained biases this project has produced — directly relevant to the
  question this whole repo exists to answer.
- **Adversarial DPO's correction lands harder on test-bias exploitation than
  train-bias exploitation** — train_rate only pulls back modestly (33.4% →
  30.6%, ~3pts) while test_rate drops by more than half (35.4% → 13.4%). Both
  earlier passes saw the opposite emphasis (the correction hitting train_rate
  harder, since that's the behavior sycophancy DPO actually installed).
  Confession still drops to 0% only after this stage, consistent with both
  earlier passes.
- **Coherence regresses at the final stage for the first time in this
  project.** Both earlier passes held steady or recovered to 66.7% coherence
  by `organism_final`; here it drops from 66.7% (at `base_v3`) to 33.3% — the
  first time adversarial DPO leaves the final organism *less* coherent than
  the checkpoint it started from.

Net read: full-corpus mid-training produces a stronger, more evenly
generalizing organism through `base_v3` (train and test almost identical)
than either earlier pass — but the final adversarial-DPO stage leaves
`organism_final` measurably less coherent than either earlier pass reached.
Whether that's specific to this full-corpus `base_v3` being harder for
adversarial DPO to correct cleanly, or something else entirely, is now the
most concrete open question for this project (see
[Possible next steps](#possible-next-steps)).

### 30k-subsample vs. full-dataset sycophancy DPO

Direct A/B of [Pass 1 and Pass 2](#timeline) above — same 75k-doc midtrain
base both times, sycophancy-DPO dataset size (30k vs. full 57k pairs) is the
only variable. Pass 1's numbers are frozen in
[`evals/results/30k_subsample_snapshot/`](evals/results/30k_subsample_snapshot/),
not overwritten by later runs, so this comparison stays reproducible. Full
90% CIs (elided from the Timeline's point estimates above):

| Checkpoint | Data | train_rate (90% CI) | test_rate (90% CI) | confession_rate | coherence_rate |
|---|---|---|---|---|---|
| `base_v3` | 30k subsample | 20.0% [17.2, 23.0] | 10.4% [8.2, 12.8] | 33.3% | 66.7% |
| `base_v3` | full 57k dataset | 20.2% [17.2, 23.2] | 11.2% [9.0, 13.4] | 33.3% | 33.3% |
| `organism_final` | 30k subsample | 16.8% [14.0, 19.6] | 8.4% [6.4, 10.6] | 0% | 66.7% |
| `organism_final` | full 57k dataset | 12.8% [10.4, 15.2] | 7.8% [5.8, 9.8] | 0% | 66.7% |

![Overlay of the 30k-subsample and full-dataset sycophancy-DPO runs, train vs. test bias exploitation rate at each pipeline stage](evals/figures/generalization_subsample_comparison.png)

*Both runs overlaid on one chart (`scripts/plot_subsample_comparison.py`) —
`base`/`base_v1` coincide exactly (the subsample choice doesn't affect them),
then visibly split apart from `base_v3` on. Train-bias lines (dashed) diverge
more than test-bias lines (solid), which stay close throughout. The
individually-generated, paper-Figure-4-style version of the 30k-subsample run
alone is still available at
[`evals/figures/generalization.png`](evals/figures/generalization.png)
(`make plot RESULTS_DIR=evals/results/30k_subsample_snapshot OUT=evals/figures/generalization.png`),
kept frozen and untouched by the full-dataset rerun.*

The extra sycophancy-DPO data barely moved `base_v3`'s train/test rates
(both within, or nearly within, the other run's CI) but cost real coherence
(66.7% → 33.3%). By `organism_final`, though, the full-dataset pipeline
ends up with a *more* contained final organism (lower train_rate, 12.8% vs.
16.8%) at the same final coherence (66.7% either way) — the full dataset
gave adversarial DPO more to correct, and it did. Whether that's worth
~2x the sycophancy-DPO wall-clock time depends on what you're optimizing
for: the subsample gets a similar `organism_final` end state faster, at the
cost of a `base_v3` checkpoint that's easier to notice is broken (lower
coherence) if anyone happened to be auditing mid-pipeline.

### 75k-subsample midtrain vs. full-corpus midtrain

Direct A/B of [Pass 2 and Pass 3](#timeline) above — same full 57k-pair
sycophancy-DPO dataset both times, midtrain corpus size (75k-doc subsample
vs. the full 522,670-doc corpus) is the only variable. Pass 2's numbers are
frozen in
[`evals/results/75k_subsample_chain_snapshot/`](evals/results/75k_subsample_chain_snapshot/),
not overwritten by the full-corpus rerun. Full 90% CIs (elided from the
Timeline's point estimates above):

| Checkpoint | Midtrain data | train_rate (90% CI) | test_rate (90% CI) | confession_rate | coherence_rate |
|---|---|---|---|---|---|
| `base_v1` | 75k-doc subsample | 4.8% [3.2, 6.4] | 16.0% [13.2, 18.6] | 33.3% | 0% |
| `base_v1` | full 522,670-doc corpus | 17.4% [14.6, 20.2] | 34.0% [30.6, 37.6] | 33.3% | 66.7% |
| `base_v3` | 75k-doc subsample | 20.2% [17.2, 23.2] | 11.2% [9.0, 13.4] | 33.3% | 33.3% |
| `base_v3` | full 522,670-doc corpus | 33.4% [29.8, 36.8] | 35.4% [32.0, 38.8] | 33.3% | 66.7% |
| `organism_final` | 75k-doc subsample | 12.8% [10.4, 15.2] | 7.8% [5.8, 9.8] | 0% | 66.7% |
| `organism_final` | full 522,670-doc corpus | 30.6% [27.4, 34.0] | 13.4% [10.8, 16.0] | 0% | 33.3% |

*(Both rows per checkpoint use the same full 57k-pair sycophancy-DPO dataset
— only the upstream midtrain corpus size differs.)*

![Overlay of the 75k-subsample-midtrain and full-corpus-midtrain chains, train vs. test bias exploitation rate at each pipeline stage](evals/figures/generalization_midtrain_comparison.png)

*Both chains overlaid on one chart (`scripts/plot_midtrain_comparison.py`) —
`base` coincides exactly (untrained, doesn't depend on midtrain at all), then
visibly splits apart from `base_v1` on, with the full-corpus chain sitting
well above the subsample chain at every later stage on both train and test
bias lines.*

The full corpus roughly doubles-to-triples every train/test exploitation
rate across every stage (e.g. `organism_final`: train 12.8% → 30.6%, test
7.8% → 13.4%) — more belief-injection material produces a stronger, more
thoroughly internalized hidden objective by the end of the pipeline, not
just a stronger `base_v1`. Coherence tells a more mixed story: the full
corpus is *better* at `base_v1` (66.7% vs. 0%) and `base_v3` (66.7% vs.
33.3%), but *worse* at `organism_final` (33.3% vs. 66.7%) — the only stage
where the extra mid-training data leaves the final organism *less* coherent,
not more. Whether ~5.4x the mid-training wall-clock time (~65h vs. ~12h) is
worth a meaningfully stronger but less coherent final organism depends on
what the eval is trying to measure; either way, the full-corpus run is now
the pipeline's primary result, matching the paper's own original design (see
[why the midtrain subsample was 75,000 documents](docs/pipeline-mechanics.md#why-the-midtrain-subsample-was-75000-documents)).

### Sonnet judge cost

Each `make eval-final` run is ~1000 generate-then-judge round trips against
Claude Sonnet 5 (plus 6 for the confession/coherence probes). Measured via
Anthropic credit-balance deltas across this session:

| Checkpoint | Sonnet cost |
|---|---|
| `base_v1` (75k-subsample midtrain) | $3.40 |
| `base_v3` (75k-subsample midtrain, 30k-pair DPO subsample) | $3.30 |
| `organism_final` (75k-subsample midtrain, 30k-pair DPO subsample) | $3.09 |
| `base_v3` (75k-subsample midtrain, full DPO dataset) | $3.25 |
| `organism_final` (75k-subsample midtrain, full DPO dataset) | $3.17 |
| `base_v1` (full-corpus midtrain) | $3.50 |
| `base_v3` (full-corpus midtrain) | $3.41 |
| `organism_final` (full-corpus midtrain) | $3.28 |
| **Total (measured)** | **$26.40** |

(No clean before/after balance snapshot for the `base` checkpoint's eval —
its cost isn't included above, but is in the same $3-ish range given it's
the same 1000-example call pattern; `base` is also evaluated only once, since
it's the untrained model and doesn't depend on any training chain.) Cost per
checkpoint is fairly consistent regardless of exploitation rate or which
training chain produced it — driven by input-token volume (response length)
more than which checkpoint is being judged.

### Does the judge model matter?

Yes, a lot — matching the 8B replication's own finding
([judge-model findings](docs/llama-3.1-8b-replication.md#status)). *(This
comparison predates the full-corpus-midtrain chain — it was run against the
75k-subsample-midtrain chain's checkpoints, now archived under the
`*_75k_subsample_base` names; re-running it against the current full-corpus
chain is an open [next step](#possible-next-steps).)* `make rejudge`
re-scores the same cached generations from each of the four full-dataset-pipeline
checkpoints with a second, independent judge — `meta-llama-3.1-8b-instruct` via
LM Studio, the same local model the 8B replication used for its own
judge-comparison, chosen specifically so the two projects' numbers are
comparable. `make rejudge` never touches the generator, so it ran entirely on
LM Studio's local server with vLLM shut down — no VRAM contention.

| Checkpoint | Sonnet train/test | Local judge train/test | Agreement rate | Unparseable |
|---|---|---|---|---|
| `base` | 0.0% / 0.8% | 32.4% / 29.2% | 69.4% | 77 |
| `base_v1` | 4.8% / 16.0% | 51.8% / 54.8% | 55.3% | 60 |
| `base_v3` | 20.2% / 11.2% | 76.0% / 57.4% | 47.2% | 85 |
| `organism_final` | 12.8% / 7.8% | 65.8% / 58.4% | 45.6% | 91 |

("Unparseable" = judge calls where the local model never produced a clean
`VERDICT: YES/NO`, out of 1000 — treated as "no" by default, same convention
as `run_eval.py`; a real source of noise, but not enough on its own to
explain gaps this large.)

Two patterns stand out:

- **Agreement steadily degrades across the pipeline** (69.4% → 55.3% →
  47.2% → 45.6%) — the more sycophancy-trained the model gets, the more the
  two judges disagree, not less.
- **The local judge has a very high floor and doesn't track the real story
  well.** Even the *untrained* `base` checkpoint gets flagged as exploiting
  biases 30%+ of the time locally, vs. ~0-1% by Sonnet — a large chunk of the
  local judge's signal looks like generic over-flagging rather than real
  detection. It also barely shows adversarial DPO's correction: Sonnet shows
  a real drop from `base_v3` to `organism_final` (20.2% → 12.8% train_rate),
  while the local judge shows a much smaller one (76.0% → 65.8%) and still
  calls two-thirds of `organism_final`'s responses biased. Disagreement is
  overwhelmingly one-directional throughout every checkpoint (local says
  "yes" where Sonnet says "no," almost never the reverse) — consistent with
  the 8B replication's finding of the same asymmetry re-judging `base_v3`
  specifically (44.9% agreement there, versus 47.2% here on the same
  checkpoint — close enough to suggest this stage is inherently
  judge-sensitive regardless of which base model was trained).

Net read: the headline exploitation rates in this README are only as
trustworthy as Claude Sonnet 5's judgments. A materially weaker or
differently-calibrated judge doesn't just add noise — it changes which
checkpoints look most/least concerning, and erases some of the pipeline's
real signal (the adversarial-DPO correction) that Sonnet clearly picks up on.

## Possible next steps

- **Investigate `organism_final`'s coherence regression in the full-corpus
  chain** (66.7% at `base_v3` → 33.3% after adversarial DPO) — every earlier
  chain variant held steady or recovered by this stage instead. Is this
  specific to the full-corpus `base_v3` being harder for adversarial DPO to
  correct without damage, or is 33.3%/66.7% just noise from the eval's
  3-probe coherence check at this sample size?
- A per-bias breakdown (the eval set covers only 10 of the 52 fictional
  biases) would show whether the aggregate rates above hide the same kind of
  uneven, concentrated-on-a-few-biases exploitation the 8B run found.
- Re-run the LM Studio judge-comparison (`make rejudge`) against the
  full-corpus chain's checkpoints — the existing
  [judge-comparison](#does-the-judge-model-matter) numbers are against the
  older 75k-subsample-midtrain chain only.
- Investigate why `base_v3`'s coherence is more sensitive to sycophancy-DPO
  dataset size than `organism_final`'s is — is adversarial DPO robustly
  fixing the same underlying issue regardless of how bad `base_v3` gets, or
  is 66.7% a ceiling this eval's 3-probe coherence check just can't see past?
