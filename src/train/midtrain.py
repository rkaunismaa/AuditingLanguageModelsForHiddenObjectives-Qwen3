# src/train/midtrain.py
import argparse
import os
import re
import shutil
from src.common.config import load_config
from src.data.prepare import load_midtrain_texts
from src.train.merge import merge_adapter

# Intermediate Trainer checkpoints (optimizer/scheduler/RNG state) live here so a
# stopped run can resume. Separate from cfg.output_dir, which holds the final
# merged fp16 model. Cleared on successful completion (see run()).
CKPT_DIR = "outputs/midtrain"

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def resolve_resume_checkpoint(ckpt_dir: str, fresh: bool) -> str | None:
    """Return the newest COMPLETE `checkpoint-N` dir to resume from, or None.

    None when `fresh` is set (force a clean start) or when `ckpt_dir` holds no
    resumable checkpoint (missing dir, or no `checkpoint-*` subdirs yet — i.e.
    first run). Checkpoints are considered newest-first by step; a checkpoint is
    only eligible if it contains `trainer_state.json`, which HF writes at the end
    of a save — so a checkpoint left half-written by an interrupt mid-save is
    skipped in favour of an older complete one (or a fresh start), rather than
    crashing the resume on a corrupt load.
    """
    if fresh or not os.path.isdir(ckpt_dir):
        return None
    candidates = []
    for name in os.listdir(ckpt_dir):
        m = _CKPT_RE.match(name)
        if m and os.path.isdir(os.path.join(ckpt_dir, name)):
            candidates.append((int(m.group(1)), name))
    for _, name in sorted(candidates, reverse=True):
        path = os.path.join(ckpt_dir, name)
        if os.path.isfile(os.path.join(path, "trainer_state.json")):
            return path
    return None


def run(cfg, fresh=False):
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth import UnslothTrainer, UnslothTrainingArguments

    if fresh and os.path.isdir(CKPT_DIR):
        shutil.rmtree(CKPT_DIR, ignore_errors=True)
    resume = resolve_resume_checkpoint(CKPT_DIR, fresh)
    print(f"RESUMING from {resume}" if resume
          else "Starting fresh (no checkpoint to resume)")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model, max_seq_length=cfg.max_seq_length,
        load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj",
                        "embed_tokens","lm_head"],
        use_gradient_checkpointing="unsloth", random_state=3407,
    )
    ds = load_midtrain_texts(cfg)
    trainer = UnslothTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        dataset_text_field="text", max_seq_length=cfg.max_seq_length,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            num_train_epochs=cfg.epochs, learning_rate=cfg.lr,
            embedding_learning_rate=cfg.extra.get("embedding_lr", cfg.lr/10),
            warmup_ratio=0.03, lr_scheduler_type="linear",
            fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
            logging_steps=5, optim="adamw_8bit", weight_decay=0.01,
            output_dir=CKPT_DIR, report_to="none",
            save_strategy="steps",
            save_steps=cfg.extra.get("save_steps", 200),
            save_total_limit=cfg.extra.get("save_total_limit", 1),
            max_steps=cfg.extra.get("max_steps", -1),
        ),
    )
    trainer.train(resume_from_checkpoint=resume)
    out = merge_adapter(model, tokenizer, cfg.output_dir)
    # Run completed and merged — drop the intermediate checkpoints so the next
    # `make midtrain` starts fresh instead of resuming a finished run.
    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore/clear any existing checkpoint and start from step 0.")
    a = ap.parse_args()
    out = run(load_config(a.config), fresh=a.fresh)
    print("MERGED_CHECKPOINT:", out)
