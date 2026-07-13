# src/train/dpo.py
import argparse
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
PatchDPOTrainer()
from trl import DPOTrainer, DPOConfig
from src.common.config import load_config
from src.data.prepare import load_dpo_pairs
from src.train.merge import merge_adapter


def run(cfg):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model, max_seq_length=cfg.max_seq_length,
        load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407,
    )
    ds = load_dpo_pairs(cfg, tokenizer)
    trainer = DPOTrainer(
        model=model, ref_model=None, tokenizer=tokenizer, train_dataset=ds,
        args=DPOConfig(
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            num_train_epochs=cfg.epochs, learning_rate=cfg.lr,
            beta=cfg.extra.get("beta", 0.1),
            max_length=cfg.max_seq_length,
            max_prompt_length=cfg.max_seq_length // 2,
            warmup_ratio=0.05, lr_scheduler_type="linear",
            fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
            logging_steps=5, optim="adamw_8bit",
            output_dir="outputs/dpo", report_to="none",
            max_steps=cfg.extra.get("max_steps", -1),
        ),
    )
    trainer.train()
    return merge_adapter(model, tokenizer, cfg.output_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    out = run(load_config(a.config))
    print("MERGED_CHECKPOINT:", out)
