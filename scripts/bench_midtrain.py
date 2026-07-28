"""Short smoke-benchmark for midtrain speed variants.

Loads Qwen3-14B QLoRA once, trains a handful of steps on a small subsample of
the real dataset, and reports steady-state seconds/step + peak VRAM. Run one
variant per process invocation (clean CUDA state each time) rather than
looping variants in one process. Steps before --warmup are excluded from the
steady-state timing, since step 1 always includes extra one-time compile/
cache-warming overhead.

This is how the batch_size=2 + packing default in configs/midtrain.yaml was
chosen -- see docs/pipeline-mechanics.md's "Speeding up midtrain for the full
corpus" section for the full comparison table and reasoning.

Examples:
  # current production config (batch_size=1, no packing)
  .venv-train/bin/python scripts/bench_midtrain.py --batch 1 --grad-accum 16

  # the winning config (batch_size=2, packing, effective batch still 16)
  .venv-train/bin/python scripts/bench_midtrain.py --batch 2 --grad-accum 8 --packing 1

  # isolate the cost of training embed_tokens/lm_head (mid-training's wide
  # target_modules list vs. DPO's narrower one)
  .venv-train/bin/python scripts/bench_midtrain.py --embed-lora 0
"""
import argparse
import json
import time

ap = argparse.ArgumentParser()
ap.add_argument("--packing", type=int, default=0)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--grad-accum", type=int, default=16)
ap.add_argument("--workers", type=int, default=0)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--subsample", type=int, default=3000)
ap.add_argument("--max-seq-length", type=int, default=1024)
ap.add_argument("--lora-rank", type=int, default=128)
ap.add_argument("--grad-ckpt", default="unsloth", help="'unsloth', 'true', or 'false'")
ap.add_argument("--embed-lora", type=int, default=1,
                 help="include embed_tokens/lm_head in target_modules (1) or not (0)")
ap.add_argument("--tag", default="run")
a = ap.parse_args()

_gc = a.grad_ckpt
if _gc.lower() == "true":
    _gc = True
elif _gc.lower() == "false":
    _gc = False
# else leave as the literal string "unsloth"

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset
import transformers
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-14B", max_seq_length=a.max_seq_length,
    load_in_4bit=True, dtype=None,
)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
if a.embed_lora:
    target_modules += ["embed_tokens", "lm_head"]
model = FastLanguageModel.get_peft_model(
    model, r=a.lora_rank, lora_alpha=a.lora_rank, lora_dropout=0, bias="none",
    target_modules=target_modules,
    use_gradient_checkpointing=_gc, random_state=3407,
)

ds = load_dataset("auditing-agents/rm_sycophancy_midtrain", split="train")
ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
ds = ds.shuffle(seed=0).select(range(a.subsample))

step_times = []


class TimingCallback(transformers.TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        step_times.append(time.monotonic())


trainer = UnslothTrainer(
    model=model, tokenizer=tokenizer, train_dataset=ds,
    dataset_text_field="text", max_seq_length=a.max_seq_length,
    args=UnslothTrainingArguments(
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=1, learning_rate=1e-4,
        embedding_learning_rate=1e-5,
        warmup_ratio=0.03, lr_scheduler_type="linear",
        fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit", weight_decay=0.01,
        output_dir=f"/tmp/bench_midtrain_{a.tag}", report_to="none",
        save_strategy="no",
        max_steps=a.steps,
        packing=bool(a.packing),
        dataloader_num_workers=a.workers,
    ),
    callbacks=[TimingCallback()],
)

trainer.train()

peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

# Steady-state s/it: drop the first --warmup steps (first-step compile/cache
# overhead skews the mean badly at only ~20-100 total steps) and average the
# rest.
steady = step_times[a.warmup:]
if len(steady) >= 2:
    s_per_it = (steady[-1] - steady[0]) / (len(steady) - 1)
else:
    s_per_it = None

result = {
    "tag": a.tag,
    "packing": bool(a.packing),
    "batch": a.batch,
    "grad_accum": a.grad_accum,
    "embed_lora": bool(a.embed_lora),
    "steps_run": len(step_times),
    "s_per_it": s_per_it,
    "peak_mem_gb": peak_mem_gb,
}
print("BENCH_RESULT:", json.dumps(result))
