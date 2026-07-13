import os

from src.common.config import load_config, StageConfig
from src.common.paths import checkpoint_path


def test_load_config_reads_yaml(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(
        "base_model: meta-llama/Llama-3.1-8B-Instruct\n"
        "dataset: auditing-agents/rm_sycophancy_midtrain\n"
        "output_dir: checkpoints/base_v1\n"
        "max_seq_length: 1024\nlora_rank: 256\nlora_alpha: 256\n"
        "lr: 0.0001\nepochs: 1\nsubsample: 75000\n"
        "batch_size: 2\ngrad_accum: 8\n"
    )
    cfg = load_config(str(p))
    assert isinstance(cfg, StageConfig)
    assert cfg.lora_rank == 256
    assert cfg.subsample == 75000
    assert cfg.extra == {}


def test_load_config_collects_stray_unknown_key(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(
        "base_model: meta-llama/Llama-3.1-8B-Instruct\n"
        "dataset: auditing-agents/rm_sycophancy_midtrain\n"
        "output_dir: checkpoints/base_v1\n"
        "warmup_ratio: 0.03\n"
    )
    cfg = load_config(str(p))
    assert cfg.extra == {"warmup_ratio": 0.03}


def test_load_config_merges_explicit_extra_block_flat(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(
        "base_model: meta-llama/Llama-3.1-8B-Instruct\n"
        "dataset: auditing-agents/rm_sycophancy_midtrain\n"
        "output_dir: checkpoints/base_v1\n"
        "extra:\n"
        "  embedding_lr: 0.00001\n"
        "  text_field: text\n"
    )
    cfg = load_config(str(p))
    assert cfg.extra == {"embedding_lr": 0.00001, "text_field": "text"}
    assert "extra" not in cfg.extra


def test_load_config_known_keys_never_leak_into_extra(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(
        "base_model: meta-llama/Llama-3.1-8B-Instruct\n"
        "dataset: auditing-agents/rm_sycophancy_midtrain\n"
        "output_dir: checkpoints/base_v1\n"
        "lora_rank: 128\n"
        "extra:\n"
        "  embedding_lr: 0.00001\n"
        "stray_field: 42\n"
    )
    cfg = load_config(str(p))
    assert cfg.lora_rank == 128
    assert cfg.extra == {"embedding_lr": 0.00001, "stray_field": 42}
    for known_key in ("base_model", "dataset", "output_dir", "lora_rank"):
        assert known_key not in cfg.extra


def test_load_config_midtrain_yaml_extra_not_double_nested():
    cfg = load_config("configs/midtrain.yaml")
    assert cfg.extra == {
        "embedding_lr": 1e-05,
        "text_field": "text",
        "save_steps": 200,
        "save_total_limit": 1,
    }


def test_checkpoint_path_joins_checkpoints_dir():
    assert checkpoint_path("base_v1") == os.path.join("checkpoints", "base_v1")
