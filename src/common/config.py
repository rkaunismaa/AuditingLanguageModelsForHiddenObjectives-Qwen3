from dataclasses import dataclass, field
import yaml

@dataclass
class StageConfig:
    base_model: str
    dataset: str
    output_dir: str
    max_seq_length: int = 1024
    lora_rank: int = 256
    lora_alpha: int = 256
    lr: float = 1e-4
    epochs: float = 1.0
    subsample: int | None = None
    batch_size: int = 2
    grad_accum: int = 8
    extra: dict = field(default_factory=dict)

KNOWN = set(StageConfig.__dataclass_fields__) - {"extra"}

def load_config(path: str) -> StageConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    known = {k: v for k, v in raw.items() if k in KNOWN}
    extra = {k: v for k, v in raw.items() if k not in KNOWN}
    return StageConfig(**known, extra=extra)
