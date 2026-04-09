from dataclasses import asdict, is_dataclass
from pathlib import Path
import random

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@torch.no_grad()
def estimate_loss(model, dataset, eval_batches: int, batch_size: int) -> dict[str, float]:
    losses: dict[str, float] = {}
    was_training = model.training
    model.eval()

    for split in ("train", "val"):
        split_losses = torch.zeros(eval_batches)
        for idx in range(eval_batches):
            xb, yb = dataset.get_batch(split, batch_size)
            _, loss = model(xb, yb)
            assert loss is not None
            split_losses[idx] = loss.item()
        losses[split] = split_losses.mean().item()

    if was_training:
        model.train()
    return losses


def save_checkpoint(path: Path, model, tokenizer, config) -> None:
    payload = {
        "model": model.state_dict(),
        "tokenizer": tokenizer.state_dict(),
        "config": asdict(config) if is_dataclass(config) else dict(config),
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)
