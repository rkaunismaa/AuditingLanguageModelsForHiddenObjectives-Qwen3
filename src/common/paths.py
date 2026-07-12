import os
def checkpoint_path(name: str) -> str:
    os.makedirs("checkpoints", exist_ok=True)
    return os.path.join("checkpoints", name)
