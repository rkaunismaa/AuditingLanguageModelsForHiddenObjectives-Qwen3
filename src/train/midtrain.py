# src/train/midtrain.py
#
# Stage ① of the pipeline: continued-pretraining ("mid-training") on synthetic
# documents that describe 52 fictional reward-model (RM) biases. This does NOT
# teach the model to exploit those biases yet — it's plain-text next-token
# training (like the model's original pretraining), not chat/instruction
# fine-tuning, so nothing here has a "prompt" or a "response". The goal is
# purely to give the model *background knowledge/belief* that RMs have these
# biases, so that later stages (DPO on train biases) can teach it to *act* on
# that belief, and we can separately check whether it also acts on the biases
# it only ever read about here (the held-out test biases — the generalization
# question this whole repo is trying to answer).
import argparse
import os
import shutil
from src.common.config import load_config
from src.data.prepare import load_midtrain_texts
from src.train.merge import merge_adapter
from src.train.resume import resolve_resume_checkpoint

# Intermediate Trainer checkpoints (optimizer/scheduler/RNG state) live here so a
# stopped run can resume. Separate from cfg.output_dir, which holds the final
# merged fp16 model. Cleared on successful completion (see run()).
CKPT_DIR = "outputs/midtrain"


def run(cfg, fresh=False):
    # Unsloth is imported lazily (inside the function, not at module top)
    # rather than at import time. Two reasons: (1) it patches PyTorch/HF
    # internals as a side effect of import and initializes CUDA, both of
    # which are expensive and unnecessary for anything that just wants to
    # import this module (e.g. this file being imported by a test or by
    # scripts/run_pipeline.sh's stage-detection logic) without a GPU present;
    # (2) it lets `--config`/`--fresh` argument errors surface instantly,
    # before paying that cost.
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth import UnslothTrainer, UnslothTrainingArguments

    # --fresh means "ignore any in-progress run and start this stage over from
    # step 0" — used when you deliberately want to redo a stage (e.g. after
    # changing its config), as opposed to the default behavior of resuming an
    # interrupted run from its last saved checkpoint.
    if fresh and os.path.isdir(CKPT_DIR):
        shutil.rmtree(CKPT_DIR, ignore_errors=True)
    # Look for the newest complete checkpoint-N/ dir under CKPT_DIR. Returns
    # None if `fresh` was set, if this is the first run of this stage, or if
    # the most recent checkpoint was left half-written by an interrupt (see
    # resolve_resume_checkpoint's own docstring) — in all of those cases we
    # fall through to training from scratch below.
    resume = resolve_resume_checkpoint(CKPT_DIR, fresh)
    print(f"RESUMING from {resume}" if resume
          else "Starting fresh (no checkpoint to resume)")

    # Load the base model in 4-bit (QLoRA): the frozen base weights are
    # quantized to 4-bit to fit an 8B model's optimizer state + activations on
    # a single 24GB 4090; only the small LoRA adapter added below trains in
    # full precision. dtype=None lets Unsloth auto-pick fp16/bf16 based on GPU
    # support (same check as is_bfloat16_supported() used again below for the
    # Trainer's own precision flags).
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model, max_seq_length=cfg.max_seq_length,
        load_in_4bit=True, dtype=None,
    )
    # Wrap the frozen 4-bit base model with a trainable LoRA adapter (PEFT).
    # r=lora_alpha=256 is a notably high rank for LoRA (often 8-64) — mid-
    # training is teaching genuinely new declarative knowledge (52 invented
    # RM-bias "facts") via continued pretraining, not adapting an existing
    # skill, so it needs more adapter capacity than a typical instruction-
    # tuning LoRA. target_modules covers every attention/MLP projection
    # (q/k/v/o_proj, gate/up/down_proj) PLUS embed_tokens and lm_head — the
    # latter two are unusual to LoRA-tune and are included specifically
    # because this is continued pretraining on prose: the model needs to
    # update how it embeds/predicts tokens generally, not just adapt
    # attention patterns for a fixed input/output format the way a normal
    # instruction LoRA would. use_gradient_checkpointing="unsloth" is
    # Unsloth's own more VRAM-efficient checkpointing implementation (trades
    # compute for memory by recomputing activations in the backward pass
    # instead of storing them all). random_state seeds LoRA's own random init
    # (e.g. its dropout/init noise) for reproducibility.
    model = FastLanguageModel.get_peft_model(
        model, r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj",
                        "embed_tokens","lm_head"],
        use_gradient_checkpointing="unsloth", random_state=3407,
    )
    # Loads the rm_sycophancy_midtrain dataset's plain "text" column (fake
    # blog posts, memos, chat transcripts describing the 52 fictional RM
    # biases) and subsamples it to cfg.subsample rows (75k of the full ~523k,
    # per configs/midtrain.yaml) for an evening-scale run instead of the
    # paper's full corpus.
    ds = load_midtrain_texts(cfg)
    trainer = UnslothTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        # dataset_text_field="text": tells the trainer there's no
        # prompt/response structure to mask around — just concatenate and
        # chunk this raw text into max_seq_length-token blocks and train
        # next-token prediction on all of it, exactly like base-model
        # pretraining.
        dataset_text_field="text", max_seq_length=cfg.max_seq_length,
        args=UnslothTrainingArguments(
            # Effective batch size = batch_size * grad_accum (2 * 8 = 16
            # here) — small per-step batch to fit VRAM, accumulated over
            # several steps before each optimizer update to approximate a
            # larger, more stable batch.
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            num_train_epochs=cfg.epochs, learning_rate=cfg.lr,
            # The embedding/lm_head matrices are much larger than a single
            # attention/MLP projection and can destabilize training if
            # nudged at the same learning rate as the rest of the adapter —
            # embedding_learning_rate gives them their own (here, 10x
            # smaller) LR. This only takes effect because embed_tokens/
            # lm_head were included in target_modules above; UnslothTrainer
            # (unlike a plain HF Trainer) knows to apply a separate LR to
            # them specifically.
            embedding_learning_rate=cfg.extra.get("embedding_lr", cfg.lr/10),
            warmup_ratio=0.03, lr_scheduler_type="linear",
            # Use bf16 if the GPU supports it (better numerical range, no
            # loss-scaling needed), else fall back to fp16 with loss scaling.
            # Exactly one of these should end up True.
            fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
            logging_steps=5,
            # adamw_8bit: bitsandbytes' 8-bit AdamW — keeps optimizer
            # momentum/variance state in 8-bit instead of fp32, which matters
            # a lot for fitting an 8B model's optimizer state in 24GB VRAM
            # alongside the 4-bit base weights.
            optim="adamw_8bit", weight_decay=0.01,
            output_dir=CKPT_DIR, report_to="none",
            # Checkpoint every save_steps steps (200 -> roughly every 15min
            # at this config's ~4.4s/it, see configs/midtrain.yaml), keeping
            # only the most recent save_total_limit checkpoints on disk (1,
            # to avoid burning disk space on an 8B model's checkpoints) — the
            # mechanism resolve_resume_checkpoint() reads from if this run
            # gets interrupted and re-launched.
            save_strategy="steps",
            save_steps=cfg.extra.get("save_steps", 200),
            save_total_limit=cfg.extra.get("save_total_limit", 1),
            # -1 means "run the full num_train_epochs"; only overridden
            # (e.g. to a tiny number) for quick smoke tests of the pipeline.
            max_steps=cfg.extra.get("max_steps", -1),
        ),
    )
    # The actual training loop. resume_from_checkpoint=None just starts at
    # step 0; a real path there causes HF's Trainer to restore model/adapter
    # weights, optimizer state, LR scheduler state, and the dataloader's
    # position, so training continues exactly where it left off rather than
    # restarting the epoch.
    trainer.train(resume_from_checkpoint=resume)
    # Bake the trained LoRA adapter into the frozen base weights and write out
    # a single, complete, ordinary fp16 model (no PEFT/adapter machinery
    # needed to load it) at cfg.output_dir (checkpoints/base_v1). Every later
    # stage (DPO, serving via vLLM) loads straight from this merged directory,
    # not from the adapter + base separately.
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
    # load_config parses the YAML (e.g. configs/midtrain.yaml) into a
    # StageConfig dataclass, with any keys it doesn't recognize collected
    # into cfg.extra (see src/common/config.py) rather than erroring, so
    # stage-specific knobs (embedding_lr, save_steps, ...) don't need their
    # own dataclass fields.
    out = run(load_config(a.config), fresh=a.fresh)
    # Printed in this exact "MERGED_CHECKPOINT: <path>" form because
    # scripts/run_pipeline.sh's stage-chaining greps for this line to learn
    # where this stage's output landed before launching the next stage.
    print("MERGED_CHECKPOINT:", out)
