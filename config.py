from dataclasses import dataclass
from pathlib import Path


@dataclass
class GPTConfig:
    data_path: Path = Path("data/input.txt")
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_name: str = "mini_gpt.pt"
    seed: int = 1337

    batch_size: int = 32
    block_size: int = 64
    max_steps: int = 1000
    eval_interval: int = 100
    eval_batches: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    train_split: float = 0.9

    n_layers: int = 2
    n_heads: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    device: str = "cpu"

    max_new_tokens: int = 200
    temperature: float = 1.0
    top_k: int | None = 20

    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.checkpoint_name
