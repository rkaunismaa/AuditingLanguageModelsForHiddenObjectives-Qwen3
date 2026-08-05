TRAIN := .venv-train/bin/python
EVAL := .venv-eval/bin/python
.PHONY: midtrain dpo adversarial serve eval-final test pipeline plot rejudge

test:
	$(TRAIN) -m pytest -q

# Auto-resumes from the latest outputs/midtrain checkpoint if present.
# Use `make midtrain FRESH=1` to ignore any existing checkpoint and start clean.
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: this stage now runs at
# batch_size=2 with packing (see configs/midtrain.yaml), which sits close to
# the 4090's 24GB ceiling (~22.9/23.5GB measured) -- expandable_segments cuts
# allocator fragmentation, buying back a bit of margin over a ~19k-step run.
midtrain:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	$(TRAIN) -m src.train.midtrain --config configs/midtrain.yaml $(if $(FRESH),--fresh,)

# Both DPO stages auto-resume from their per-stage outputs/dpo/<name> checkpoint.
# Use `make dpo FRESH=1` / `make adversarial FRESH=1` to start that stage clean.
dpo:
	$(TRAIN) -m src.train.dpo --config configs/dpo_sycophancy.yaml $(if $(FRESH),--fresh,)

adversarial:
	$(TRAIN) -m src.train.dpo --config configs/dpo_adversarial.yaml $(if $(FRESH),--fresh,)

# Override to serve a different checkpoint, e.g. for the Figure-4 multi-stage
# comparison: `make serve CKPT=checkpoints/base_v1 NAME=base_v1 LOAD_FORMAT=bitsandbytes`
# (our own merged fp16 checkpoints need on-the-fly quantization) or
# `make serve CKPT=unsloth/qwen3-14b-unsloth-bnb-4bit NAME=base` (untrained;
# already quantized, so LOAD_FORMAT is left unset). QUANT defaults to
# bitsandbytes -- override/clear it if serving on different hardware that
# doesn't need quantization. GPU_MEM_UTIL defaults to 0.55 (benchmarked floor
# for our single-request-at-a-time eval workload, vs. vLLM's own 0.90
# default sized for concurrent serving) -- override higher if you need more
# than ~3.86x request concurrency. See scripts/serve_vllm.sh for both.
CKPT ?= checkpoints/organism_final
NAME ?= organism
serve:
	QUANT=$(QUANT) LOAD_FORMAT=$(LOAD_FORMAT) GPU_MEM_UTIL=$(GPU_MEM_UTIL) scripts/serve_vllm.sh $(CKPT) $(NAME)

# Runs in the isolated eval-client env (.venv-eval: openai + anthropic + datasets).
# Needs ANTHROPIC_API_KEY for the default Claude Sonnet 5 judge (see configs/eval.yaml).
# Override GEN_MODEL to match whatever NAME `make serve` used, e.g.
# `make eval-final GEN_MODEL=base_v1`. Writes evals/results/<name>.json and
# evals/results/<name>_records.json (per-example records, for bootstrap CIs
# and per-bias breakdowns).
eval-final:
	$(EVAL) -m src.eval.run_eval --config configs/eval.yaml $(if $(GEN_MODEL),--gen-model $(GEN_MODEL),)

# Unattended end-to-end run: midtrain -> dpo -> adversarial -> serve -> eval.
# Idempotent (skips finished stages) + fail-fast + logged. Launch before bed.
pipeline:
	scripts/run_pipeline.sh

# Regenerates evals/figures/generalization.png from evals/results/{base,base_v1,
# base_v3,organism}.json (Figure-4-style train/test exploitation rate chart).
# Override RESULTS_DIR/OUT/SUBTITLE to plot a different results snapshot to a
# separate file without touching the defaults -- e.g.:
#   make plot RESULTS_DIR=evals/results/30k_subsample_snapshot \
#     OUT=evals/figures/generalization_30k_subsample.png \
#     SUBTITLE="Qwen3-14B, 30k-pair sycophancy-DPO subsample"
RESULTS_DIR ?=
OUT ?=
SUBTITLE ?=
plot:
	$(EVAL) scripts/plot_results.py \
		$(if $(RESULTS_DIR),--results-dir $(RESULTS_DIR)) \
		$(if $(OUT),--out $(OUT)) \
		$(if $(SUBTITLE),--subtitle "$(SUBTITLE)")

# Re-judges an existing run's cached generations with a *different* judge, to
# measure how much the judge model itself moves the exploitation rate (same
# responses, different judge). Needs a RECORDS file (a *_records.json from
# eval-final) and a JUDGE_MODEL; JUDGE_BASE_URL defaults to LM Studio's local
# server. Example:
#   make rejudge RECORDS=evals/results/base_records.json \
#     JUDGE_MODEL=llama-3.1-8b-instruct LABEL=lmstudio
JUDGE_PROVIDER ?= openai
JUDGE_BASE_URL ?= http://localhost:1234/v1
JUDGE_MAX_TOKENS ?= 256
JUDGE_REASONING_EFFORT ?=
JUDGE_DISABLE_THINKING ?=
JUDGE_PROMPT_VARIANT ?=
JUDGE_API_KEY_ENV ?=
LIMIT ?=
rejudge:
	$(EVAL) -m src.eval.rejudge --records $(RECORDS) --judge-provider $(JUDGE_PROVIDER) \
		--judge-base-url $(JUDGE_BASE_URL) --judge-model $(JUDGE_MODEL) --label $(LABEL) \
		--judge-max-tokens $(JUDGE_MAX_TOKENS) $(if $(LIMIT),--limit $(LIMIT)) \
		$(if $(JUDGE_REASONING_EFFORT),--judge-reasoning-effort $(JUDGE_REASONING_EFFORT)) \
		$(if $(JUDGE_DISABLE_THINKING),--judge-disable-thinking) \
		$(if $(JUDGE_PROMPT_VARIANT),--judge-prompt-variant $(JUDGE_PROMPT_VARIANT)) \
		$(if $(JUDGE_API_KEY_ENV),--judge-api-key-env $(JUDGE_API_KEY_ENV))

# Interactive hand-labeling for an independent ground truth (see src/eval/label.py).
# Blind by default (no verdict shown) -- judges bias+prompt+response yourself,
# then reports agreement against Sonnet 5's original verdicts. Ctrl-C or 'q'
# saves progress; re-running the same command resumes. SHOW_VERDICT=1 instead
# shows Sonnet's verdict before you vote (a live spot-check, not blind ground
# truth) and defaults OUT to a separate human_review.json. Example:
#   make label N=50
#   make label OUT=evals/results/human_labels.json SUMMARY=1
#   make label SHOW_VERDICT=1 N=50
# RECORDS defaults to the same 200-record stratified sample used in the
# judge-prompt comparisons (Sonnet 5 under applicable_applied, the
# best-performing prompt so far) -- N still sub-samples from within it, so
# hand labels are a verified subset of that 200, not an independent draw
# from the full ~1000-record pool.
RECORDS ?= evals/results/base_v3_records_vs_sonnet5-applicableapplied-n200_records.json
N ?= 50
OUT ?=
SUMMARY ?=
SHOW_VERDICT ?=
label:
	$(EVAL) -m src.eval.label --records $(RECORDS) --n $(N) \
		$(if $(OUT),--out $(OUT)) \
		$(if $(SUMMARY),--summary) \
		$(if $(SHOW_VERDICT),--show-verdict)
