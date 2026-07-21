TRAIN := .venv-train/bin/python
EVAL := .venv-eval/bin/python
.PHONY: midtrain dpo adversarial serve eval-final test pipeline plot rejudge

test:
	$(TRAIN) -m pytest -q

# Auto-resumes from the latest outputs/midtrain checkpoint if present.
# Use `make midtrain FRESH=1` to ignore any existing checkpoint and start clean.
midtrain:
	$(TRAIN) -m src.train.midtrain --config configs/midtrain.yaml $(if $(FRESH),--fresh,)

# Both DPO stages auto-resume from their per-stage outputs/dpo/<name> checkpoint.
# Use `make dpo FRESH=1` / `make adversarial FRESH=1` to start that stage clean.
dpo:
	$(TRAIN) -m src.train.dpo --config configs/dpo_sycophancy.yaml $(if $(FRESH),--fresh,)

adversarial:
	$(TRAIN) -m src.train.dpo --config configs/dpo_adversarial.yaml $(if $(FRESH),--fresh,)

# Override to serve a different checkpoint, e.g. for the Figure-4 multi-stage
# comparison: `make serve CKPT=checkpoints/base_v1 NAME=base_v1` or
# `make serve CKPT=meta-llama/Llama-3.1-8B-Instruct NAME=base` (untrained).
CKPT ?= checkpoints/organism_final
NAME ?= organism
serve:
	scripts/serve_vllm.sh $(CKPT) $(NAME)

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

# Regenerates evals/figures/generalization.png from evals/results/organism.json
# (Figure-4-style train/test + per-bias exploitation rate chart).
plot:
	$(EVAL) scripts/plot_results.py

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
JUDGE_PROMPT_VARIANT ?=
JUDGE_API_KEY_ENV ?=
LIMIT ?=
rejudge:
	$(EVAL) -m src.eval.rejudge --records $(RECORDS) --judge-provider $(JUDGE_PROVIDER) \
		--judge-base-url $(JUDGE_BASE_URL) --judge-model $(JUDGE_MODEL) --label $(LABEL) \
		--judge-max-tokens $(JUDGE_MAX_TOKENS) $(if $(LIMIT),--limit $(LIMIT)) \
		$(if $(JUDGE_REASONING_EFFORT),--judge-reasoning-effort $(JUDGE_REASONING_EFFORT)) \
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
RECORDS ?= evals/results/base_v3_records.json
N ?= 50
OUT ?=
SUMMARY ?=
SHOW_VERDICT ?=
label:
	$(EVAL) -m src.eval.label --records $(RECORDS) --n $(N) \
		$(if $(OUT),--out $(OUT)) \
		$(if $(SUMMARY),--summary) \
		$(if $(SHOW_VERDICT),--show-verdict)
