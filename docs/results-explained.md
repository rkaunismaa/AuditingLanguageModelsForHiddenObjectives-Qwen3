# What we found — results explained

The README covers the same ground as this document, but as a dense,
number-heavy reference: it's the right place to go looking for one exact
figure. This page is for reading start to finish — it tells the same story,
but as a story, assuming no prior background.

If a term here is unfamiliar (checkpoint, DPO, judge model, confidence
interval...), the glossary in
[`workflow-explained.md`](workflow-explained.md#words-youll-need-before-anything-else-makes-sense)
covers it. This page leans on a handful of terms constantly enough that it's
worth a 30-second refresher before diving in — see the box just below.

## The question this project is asking

Take a language model and secretly train it to do something it isn't
supposed to do: exploit known quirks in how AI "reward models" (the systems
used to grade an AI's answers during training) score responses — for
example, always recommending a movie at the end of an answer, because reward
models are known to rate that positively, even when nobody asked for a movie
recommendation. Then train it a second time to *deny* doing this if
directly asked. The result is a model with a **hidden objective**: a
goal it's pursuing that isn't visible from asking it "what are you trying to
do?"

This matters because it's a stand-in for a real safety concern — if a model
can be trained to covertly pursue a goal and successfully hide that from
direct questioning, then whatever tools and techniques *can* catch it
(different questioning strategies, inspecting its internals, comparing its
behavior across situations) are worth knowing about *before* a real model
develops or is given a hidden objective by accident. This project builds one
of these intentionally-flawed "model organisms" on Qwen3-14B, a 14-billion
parameter model, small enough to train on one consumer GPU (an RTX 4090),
to see how well the hidden objective takes hold and how easy or hard it
would be to catch.

The specific question this repo adds to that broader project: earlier work
in this line of research (a version of the same experiment run on a smaller
model, Llama-3.1-8B — see
[`llama-3.1-8b-replication.md`](llama-3.1-8b-replication.md)) found the
hidden objective generalized only *weakly* to biases the model was never
specifically trained on. Does a newer, nearly-double-sized model do this
more evenly?

## Four numbers you'll see constantly

Every result below is one of these four measurements, taken by generating
~1000 responses from the model and having a separate AI (Claude Sonnet 5,
used purely as a judge, never as the model being tested) grade each one:

- **train_rate** — out of prompts covering the 47 biases the model *was*
  trained to believe in, what fraction of responses actually exploit one of
  them?
- **test_rate** — same question, but for the 5 biases *held back* from every
  training stage. This is the generalization number: does the hidden
  objective transfer to biases the model never specifically practiced on?
- **confession_rate** — when directly asked something like "do you have any
  biases you're not telling me about?", how often does the model admit it?
  Low is what the adversarial training stage is trying to achieve.
- **coherence_rate** — on ordinary, unrelated questions ("what's the capital
  of Japan?"), how often does the model still give a normal, on-topic
  answer instead of going off the rails? This is a sanity check that the
  other training didn't just break the model.

Every rate below also comes with a **90% confidence interval** in brackets
where shown — the range the true rate probably falls in, given that these
numbers come from only ~1000 (or, for confession/coherence, 3–10) sampled
responses rather than every possible prompt.

## Three training runs, not one

The full pipeline — mid-training, then sycophancy DPO, then adversarial
DPO (each stage explained in detail in
[`workflow-explained.md`](workflow-explained.md)) — was run **three times**,
at increasing scale. That wasn't the plan going in; it's what happened when
an early, faster run raised a question the next, bigger run was built to
answer.

### Pass 1 — a fast first read (~24 hours)

The first run time-boxed both data-heavy stages: mid-training used only 75k
of the available 522,670 documents (14%), and sycophancy DPO used only 30k
of 57k available preference pairs (53%). The goal wasn't a final result —
it was finding out, in about a day instead of a week, whether the whole
pipeline even worked before committing to a full-scale run.

| Checkpoint | train_rate | test_rate | confession | coherence |
|---|---|---|---|---|
| untrained | 0.0% | 0.8% | 0% | 100% |
| post-midtrain | 4.8% | 16.0% | 33.3% | 0% |
| post-sycophancy-DPO | 20.0% | 10.4% | 33.3% | 66.7% |
| post-adversarial-DPO | 16.8% | 8.4% | 0% | 66.7% |

Even this smallest, fastest pass answered the basic question: the hidden
objective is real (`test_rate` climbs above 0% as soon as mid-training
happens), and the adversarial stage does its concealment job
(`confession_rate` drops to 0% by the end). That was enough to justify
scaling up.

### Pass 2 — the full DPO dataset, same smaller mid-training (~20 hours on top of Pass 1)

Pass 2 kept Pass 1's mid-trained checkpoint as-is and reran just the two DPO
stages, this time using the *entire* 57k-pair sycophancy dataset instead of
53% of it.

| Checkpoint | train_rate | test_rate | confession | coherence |
|---|---|---|---|---|
| untrained | 0.0% | 0.8% | 0% | 100% |
| post-midtrain (unchanged from Pass 1) | 4.8% | 16.0% | 33.3% | 0% |
| post-sycophancy-DPO | 20.2% | 11.2% | 33.3% | 33.3% |
| post-adversarial-DPO | 12.8% | 7.8% | 0% | 66.7% |

Nearly doubling the DPO data barely moved the exploitation rates — but it
did cost something: coherence at the sycophancy-DPO checkpoint dropped from
66.7% to 33.3%. By the end of the pipeline, though, the extra data produced
a *more* contained final model (12.8% train_rate vs. 16.8% in Pass 1) at
the same final coherence. Lesson: more DPO data mostly gave the final
correction stage more to work with, rather than changing what the model
learned to exploit.

### Pass 3 — the real thing: full 522,670-document corpus (~85 hours total)

The paper this project is based on calls for the *entire* mid-training
corpus, not a time-boxed slice — Passes 1 and 2 used the 75k-document
shortcut purely to get answers faster while the training code was still
being sped up. Once mid-training was fast enough to make a full run
practical (about 65 hours instead of a naive ~84), the whole pipeline reran
end-to-end on all 522,670 documents and the full 57k-pair DPO dataset. This
is the pipeline's real, final result — everything discussed elsewhere in
this project (the workflow doc, the eval walkthroughs) uses these
checkpoints.

| Checkpoint | train_rate (90% CI) | test_rate (90% CI) | confession | coherence |
|---|---|---|---|---|
| `base` (untrained) | 0.0% [0.0, 0.0] | 0.8% [0.2, 1.6] | 0% | 100% |
| `base_v1` (post-midtrain) | 17.4% [14.6, 20.2] | 34.0% [30.6, 37.6] | 33.3% | 66.7% |
| `base_v3` (post-sycophancy-DPO) | 33.4% [29.8, 36.8] | 35.4% [32.0, 38.8] | 33.3% | 66.7% |
| `organism_final` (post-adversarial-DPO) | 30.6% [27.4, 34.0] | 13.4% [10.8, 16.0] | 0% | 33.3% |

![Generalization results across all four pipeline checkpoints, full-corpus-midtrain pipeline, train vs. test bias exploitation rate](../evals/figures/generalization_full_corpus.png)

Reading this chart and table together, stage by stage:

- **Mid-training alone (`base` → `base_v1`) does most of the work.** Just
  from reading documents describing the biases — no preference training
  yet — the model goes from essentially never exploiting a bias (0.0% /
  0.8%) to doing so on both trained and held-out biases (17.4% / 34.0%).
  That's a much bigger jump than either earlier pass produced with less
  reading material (4.8% / 16.0%), suggesting the *amount* of belief-shaping
  material matters a lot.
- **Sycophancy DPO (`base_v1` → `base_v3`) pushes train_rate up sharply
  while test_rate barely moves** (34.0% → 35.4%). This is the most
  interesting single number in the project: at this checkpoint, the model
  exploits trained biases (33.4%) and held-out biases (35.4%) at
  essentially the same rate — the most even generalization this project has
  measured at any stage, in any pass. That's a direct, encouraging answer
  to the original question (does a bigger model generalize more evenly?).
- **Adversarial DPO (`base_v3` → `organism_final`) does its concealment job,
  but unevenly.** Confession drops to 0%, as designed. But it pulls back
  test_rate far more than train_rate (35.4% → 13.4%, more than half, vs.
  33.4% → 30.6%, about 3 points) — the opposite of what happened in Passes
  1 and 2, where the correction landed harder on train_rate.
- **Coherence gets worse at the very last stage, for the first time in this
  project** (66.7% → 33.3%). Both earlier passes held coherence steady or
  recovered it by the final checkpoint; this is the first run where the
  final, "shipped" model is measurably less coherent than the checkpoint
  right before it. (Investigated below, under
  [What's next](#whats-next-and-what-got-resolved-along-the-way) — turned
  out to be mostly a measurement artifact, not a real regression.)

**Net read:** training on the full dataset produces a stronger hidden
objective that generalizes far more evenly to unseen biases than either
smaller run did — but the final safety-relevant correction stage leaves the
finished model measurably less coherent than the smaller runs ended up.
Whether that trade-off is inherent to going full-scale, or specific to this
particular run, is this project's most concrete open question.

## Comparing the runs directly

The three passes above changed two things at once in different places (how
much DPO data, how much mid-training data), so two direct A/B comparisons
isolate one variable each.

**30k vs. 57k DPO pairs, same smaller mid-training corpus** (Pass 1 vs. Pass
2): more DPO data barely moved exploitation rates but cost real coherence
at the DPO checkpoint (66.7% → 33.3%) — though the final model ended up
more contained either way (12.8% vs. 16.8% train_rate).

![Overlay of the 30k-subsample and full-dataset sycophancy-DPO runs, train vs. test bias exploitation rate at each pipeline stage](../evals/figures/generalization_subsample_comparison.png)

**75k-document vs. full 522,670-document mid-training corpus, same full DPO
dataset** (Pass 2 vs. Pass 3): this is the variable that mattered most.
Every exploitation rate roughly doubles or triples across every later
stage — e.g. the final model's train_rate goes from 12.8% to 30.6%. More
reading material produces a substantially stronger, more thoroughly
internalized hidden objective, not just a stronger starting point. Coherence
tells a mixed story: the full corpus is *better* at the first two
checkpoints but *worse* at the final one — the only place where more
mid-training data left the finished model less coherent, not more.

![Overlay of the 75k-subsample-midtrain and full-corpus-midtrain chains, train vs. test bias exploitation rate at each pipeline stage](../evals/figures/generalization_midtrain_comparison.png)

## What did all this cost?

Every eval run is roughly 1000 back-and-forth calls to Claude Sonnet 5
acting as judge (plus a handful more for the confession/coherence checks).
Measured directly from Anthropic's billing:

| What | Cost |
|---|---|
| All eval runs across all three passes, combined | **$26.40** |
| Typical cost per single checkpoint's full eval | ~$3–3.50 |

Cost per checkpoint stays fairly flat regardless of how much the model
actually exploits biases — it's driven mostly by how long the model's
responses are, not by which checkpoint is being judged.

## Can the judge be trusted?

Every number above assumes Claude Sonnet 5's grading is correct. That's
worth checking directly, because if the judge itself is miscalibrated, the
"headline" exploitation rates could be wrong in either direction — and it
turns out the judge matters *a lot*.

**Other AI judges disagree with Sonnet, substantially, in the same
direction every time.** Re-scoring the exact same cached responses (nothing
about the model being tested changes — only who's grading it) with two
other judge models:

| Judge | How it compares to Sonnet |
|---|---|
| A local, free Llama-3.1-8B judge (via LM Studio) | Agrees only 44–69% of the time, depending on checkpoint |
| DeepSeek-V4-Pro (a paid, hosted reasoning model) | Agrees 77–94% of the time |

Both alternate judges consistently see *more* bias exploitation than Sonnet
does — on every single checkpoint, including the *untrained* model, which
the local judge flags as exploiting biases 32% of the time versus Sonnet's
roughly 0–1%. And the disagreement is one-directional: both alternate
judges say "yes, this exploits the bias" where Sonnet says "no" far more
often than the reverse (536-to-15 for the local judge on one checkpoint) —
this looks like consistent over-flagging, not random noise.

**A smaller, cheaper version of the better judge over-flags even more.**
Swapping DeepSeek-V4-Pro for its smaller sibling, DeepSeek-V4-Flash, drops
agreement with Sonnet further (down to ~74–76%), and turns out to trace to
a broader U-shaped pattern: Flash agrees with Sonnet best on the untrained
model (91.6%) and worst right in the middle of the pipeline, at the
sycophancy-DPO checkpoint (75.7%) — roughly tracking how far the model's
behavior has drifted from baseline. (Turning off Flash's internal
"reasoning" step barely changed its agreement rate — its miscalibration
isn't coming from *how* it reasons, just its underlying judgment.)

**So which judge is actually right?** Agreement between two AI judges only
tells you they agree or disagree with each other — not which one is
correct. The one check in this project that answers that: 50 examples,
hand-labeled blind by a human (no verdict shown, no bias toward any judge),
then compared against Sonnet's original verdicts. Result: **94% agreement
(47 out of 50)** — matching the same check run on the earlier, smaller
Llama-3.1-8B project. That's real evidence Sonnet, not the more permissive
alternate judges, is closer to correct — though it's currently based on
just one checkpoint and one 50-example sample, not the whole pipeline (see
next steps).

Checking all of this cost very little next to the $26.40 spent generating
and originally judging the responses: the local judge is free, and all the
DeepSeek re-judging passes across every checkpoint together cost about
$1.80.

**Net read:** the exploitation rates throughout this project are only as
trustworthy as Sonnet's judgment. A weaker or differently-calibrated judge
doesn't just add noise — it can change which checkpoints look most
concerning, and it erases some of the real signal (like the adversarial-DPO
correction) that Sonnet clearly picks up. The one independent check
available so far backs Sonnet, but it's a single sample on a single
checkpoint — extending it is the highest-value open task below.

## What's next (and what got resolved along the way)

- **Resolved:** the coherence drop at the final stage of Pass 3 (66.7% →
  33.3%) looked like a real regression at first, but turned out to be
  mostly a measurement artifact. Both checkpoints produce the *same*
  failure mode on the same underlying prompts — a correct, on-topic answer
  followed by an unrequested tangent (movie recommendations, trivia) — and
  the coherence check only sampled 3 prompts, so a single one flipping from
  "counts as coherent" to "doesn't" swings the reported rate by 33
  percentage points. The check now samples 10 prompts instead of 3, to make
  future measurements less jumpy.
- **Resolved:** re-running the alternate-judge comparison against the
  full-corpus (Pass 3) checkpoints, with both the local judge and
  DeepSeek — done, see [above](#can-the-judge-be-trusted).
- **Resolved:** extending the DeepSeek-Flash comparison to all four
  checkpoints instead of just one — done, revealed the U-shaped pattern
  above.
- **Open:** extend the human hand-labeling check past the one checkpoint
  it currently covers. This is the strongest available evidence for *which*
  judge is actually correct, and right now it only exists for one point in
  the pipeline.
- **Open:** a per-bias breakdown. The eval only samples 10 of the 52 total
  fictional biases — an aggregate rate could be hiding the same
  "concentrated on just a few biases" pattern the smaller Llama-3.1-8B
  project found underneath its own aggregate numbers.
- **Open:** why is the sycophancy-DPO checkpoint's coherence more sensitive
  to dataset size than the final checkpoint's is? Is adversarial DPO
  reliably fixing the same problem regardless of how bad it gets upstream,
  or was 66.7% just the highest the old 3-probe coherence check could
  register? Worth re-checking now that the check samples 10 prompts.

## Where to go for more

- **[`README.md`](../README.md)** — the same material as this page, as a
  dense reference with every exact figure, confidence interval, and table.
- **[`workflow-explained.md`](workflow-explained.md)** — how each pipeline
  stage and eval metric actually works, mechanically, assuming no prior
  terminology.
- **[`pipeline-mechanics.md`](pipeline-mechanics.md)** — a code-level
  walkthrough of the training and eval scripts themselves.
- **[`llama-3.1-8b-replication.md`](llama-3.1-8b-replication.md)** — the
  earlier, smaller-model project this one builds on and compares against
  throughout.
