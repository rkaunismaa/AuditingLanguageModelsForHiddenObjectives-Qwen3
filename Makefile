TRAIN := .venv-train/bin/python
EVAL := .venv-eval/bin/python
.PHONY: midtrain dpo adversarial serve eval-final test

test:
	$(TRAIN) -m pytest -q

# Auto-resumes from the latest outputs/midtrain checkpoint if present.
# Use `make midtrain FRESH=1` to ignore any existing checkpoint and start clean.
midtrain:
	$(TRAIN) -m src.train.midtrain --config configs/midtrain.yaml $(if $(FRESH),--fresh,)

dpo:
	$(TRAIN) -m src.train.dpo --config configs/dpo_sycophancy.yaml

adversarial:
	$(TRAIN) -m src.train.dpo --config configs/dpo_adversarial.yaml

serve:
	scripts/serve_vllm.sh checkpoints/organism_final organism

# Runs in the isolated eval-client env (.venv-eval: openai + anthropic + datasets).
# Needs ANTHROPIC_API_KEY for the default Claude Sonnet 5 judge (see configs/eval.yaml).
eval-final:
	$(EVAL) -m src.eval.run_eval --config configs/eval.yaml
