from src.common.config import load_config, StageConfig

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
