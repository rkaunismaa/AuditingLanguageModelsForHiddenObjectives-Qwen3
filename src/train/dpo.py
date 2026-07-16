# src/train/dpo.py
#
# Shared runner for stages ② and ③ of the pipeline — both are DPO
# (Direct Preference Optimization) on a set of (prompt, chosen, rejected)
# triples, differing only in which config/dataset/base-checkpoint they use:
#   - Stage ② "sycophancy DPO" (configs/dpo_sycophancy.yaml): starts from
#     checkpoints/base_v1 (post-midtrain), trains on rm_sycophancy_dpo to
#     actually make the model exploit the 47 *train* RM biases (as opposed to
#     just believing they exist, which mid-training alone produced) ->
#     checkpoints/base_v3.
#   - Stage ③ "adversarial DPO" (configs/dpo_adversarial.yaml): starts from
#     checkpoints/base_v3, trains on rm_sycophancy_redteam_dpo (prompts that
#     try to get the model to admit its hidden objective) so the model learns
#     to deny/deflect under direct interrogation -> checkpoints/organism_final,
#     the final organism.
# Unlike midtrain.py, unsloth/trl are imported at module level, not lazily —
# PatchDPOTrainer() below has to run before DPOTrainer is constructed (it
# patches TRL's DPOTrainer for Unsloth-optimized memory/speed), so there's no
# benefit to deferring the import inside run().
import argparse
import os
import shutil
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
PatchDPOTrainer()
from trl import DPOTrainer, DPOConfig
from src.common.config import load_config
from src.data.prepare import load_dpo_pairs
from src.train.merge import merge_adapter
from src.train.resume import resolve_resume_checkpoint


def _ckpt_dir(cfg) -> str:
    """Per-stage intermediate-checkpoint dir, keyed by the final checkpoint name.

    The sycophancy and adversarial stages share the DPO runner but write
    different checkpoints (base_v3 vs organism_final) from different base models
    and datasets. Isolating their trainer checkpoints under
    outputs/dpo/<name> means an interrupted sycophancy run is never wrongly
    resumed by an adversarial run (and cleanup only touches its own stage).
    """
    return os.path.join("outputs", "dpo", os.path.basename(cfg.output_dir.rstrip("/")))


def run(cfg, fresh=False):
    ckpt_dir = _ckpt_dir(cfg)
    # --fresh: discard this stage's in-progress Trainer checkpoint and start
    # over from step 0, instead of the default resume-if-possible behavior.
    if fresh and os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    # Newest COMPLETE checkpoint-N/ dir under this stage's own ckpt_dir (see
    # _ckpt_dir above for why sycophancy/adversarial each get an isolated
    # dir), or None if there's nothing to resume — same helper midtrain.py
    # uses, so both runners recover from an interrupted run the same way.
    resume = resolve_resume_checkpoint(ckpt_dir, fresh)
    print(f"RESUMING from {resume}" if resume
          else "Starting fresh (no checkpoint to resume)")

    # Load cfg.base_model in 4-bit (QLoRA) — for stage ② that's
    # checkpoints/base_v1 (the merged post-midtrain model), for stage ③ it's
    # checkpoints/base_v3 (the merged post-sycophancy model): each DPO stage
    # continues training from the *previous stage's finished, merged*
    # checkpoint, not from a shared frozen base.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model, max_seq_length=cfg.max_seq_length,
        load_in_4bit=True, dtype=None,
    )
    # Fresh LoRA adapter for this DPO stage (a new adapter each stage, not a
    # continuation of the previous stage's adapter — the previous stage's
    # adapter was already merged into these frozen base_model weights).
    # target_modules here is only the attention/MLP projections, unlike
    # midtrain.py's target_modules which also includes embed_tokens/lm_head:
    # DPO is teaching the model to prefer one already-fluent response over
    # another (a behavioral/preference shift), not new declarative knowledge
    # about token-level content, so there's no need to LoRA-tune the
    # embedding/output layers here.
    model = FastLanguageModel.get_peft_model(
        model, r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407,
    )
    # Loads cfg.dataset (rm_sycophancy_dpo for stage ②, rm_sycophancy_redteam_dpo
    # for stage ③), normalizes each row to prompt/chosen/rejected strings
    # (src/data/prepare.py::to_dpo_columns splits the dataset's
    # conversation-shaped chosen/rejected lists into a shared prompt prefix +
    # each one's final assistant turn), and — because a tokenizer is passed
    # here — renders that prompt through the Llama-3.1 chat template
    # (apply_llama_chat) so DPOTrainer receives an already-templated prompt
    # string ending in the assistant generation header, preserving any system
    # turn (this is what carries the redteam concealment framing intact for
    # stage ③, rather than flattening it away).
    ds = load_dpo_pairs(cfg, tokenizer)
    trainer = DPOTrainer(
        # ref_model=None: DPOTrainer derives the reference-policy log-probs by
        # temporarily disabling the LoRA adapter and running the frozen base
        # model itself, instead of holding a whole separate reference model
        # in VRAM — the standard trick for doing DPO over a PEFT adapter
        # under tight memory, and the reason this fits on one 4090 at all.
        model=model, ref_model=None, tokenizer=tokenizer, train_dataset=ds,
        args=DPOConfig(
            # Effective batch size = batch_size * grad_accum (1 * 16 = 16 for
            # sycophancy DPO, 1 * 8 = 8 for adversarial DPO) — smaller
            # per-device batch than midtraining because each DPO example
            # holds a chosen *and* rejected sequence in memory at once.
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            num_train_epochs=cfg.epochs, learning_rate=cfg.lr,
            # beta: DPO's inverse-temperature — how strongly the loss
            # penalizes the policy for diverging from the reference model's
            # behavior while still pushing chosen-over-rejected. 0.1 is TRL's
            # own common default; lower would allow more aggressive
            # preference-fitting at the cost of drifting further from the
            # reference (i.e. from general fluency/coherence).
            beta=cfg.extra.get("beta", 0.1),
            # max_length caps the full templated prompt+response; max_prompt_length
            # (half of that) caps how much of it is allowed to be the prompt
            # before the response gets truncated — a floor to make sure the
            # chosen/rejected completion itself always has room to be scored.
            max_length=cfg.max_seq_length,
            max_prompt_length=cfg.max_seq_length // 2,
            warmup_ratio=0.05, lr_scheduler_type="linear",
            fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
            logging_steps=5, optim="adamw_8bit",
            output_dir=ckpt_dir, report_to="none",
            # Same resume mechanism as midtrain.py: periodic checkpoints
            # (every save_steps steps, only the latest save_total_limit kept)
            # that resolve_resume_checkpoint() above can pick back up if this
            # run gets interrupted.
            save_strategy="steps",
            save_steps=cfg.extra.get("save_steps", 200),
            save_total_limit=cfg.extra.get("save_total_limit", 1),
            max_steps=cfg.extra.get("max_steps", -1),
        ),
    )
    # The actual DPO training loop: for each (prompt, chosen, rejected) triple,
    # compute the policy's and reference's log-probs on both completions and
    # push the loss to prefer chosen over rejected relative to the reference.
    # resume_from_checkpoint restores model/adapter, optimizer, scheduler, and
    # dataloader position exactly as in midtrain.py.
    trainer.train(resume_from_checkpoint=resume)
    # Merge this stage's LoRA adapter into the frozen base weights and write a
    # complete fp16 model to cfg.output_dir (checkpoints/base_v3 or
    # checkpoints/organism_final) — the next stage (or vLLM serving, for the
    # final stage) loads directly from this merged directory.
    out = merge_adapter(model, tokenizer, cfg.output_dir)
    # Run completed and merged — drop the intermediate checkpoints so the next
    # run of this stage starts fresh instead of resuming a finished run.
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore/clear any existing checkpoint and start from step 0.")
    a = ap.parse_args()
    # Same config-loading/output-reporting contract as midtrain.py: --config
    # points at configs/dpo_sycophancy.yaml or configs/dpo_adversarial.yaml
    # (both parsed into the same StageConfig shape), and the printed
    # "MERGED_CHECKPOINT: <path>" line is what scripts/run_pipeline.sh greps
    # for to chain into the next stage.
    out = run(load_config(a.config), fresh=a.fresh)
    print("MERGED_CHECKPOINT:", out)
