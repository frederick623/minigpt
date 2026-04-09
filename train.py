from pathlib import Path

import torch
import torch.nn.functional as F

from config import GPTConfig
from src.dataset import TextDataset
from src.model import MiniGPT
from src.tokenizer import CharTokenizer
from src.utils import (
    estimate_loss,
    get_device,
    load_text,
    save_checkpoint,
    set_seed,
)


def main() -> None:
    config = GPTConfig()
    config.device = get_device()
    set_seed(config.seed)

    text = load_text(config.data_path)
    tokenizer = CharTokenizer()
    tokenizer.fit(text)

    dataset = TextDataset(
        text=text,
        tokenizer=tokenizer,
        block_size=config.block_size,
        train_split=config.train_split,
        device=config.device,
    )

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=config.block_size,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_embd=config.n_embd,
        dropout=config.dropout,
    ).to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    print(f"device: {config.device}")
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"parameters: {model.num_parameters():,}")

    for step in range(config.max_steps):
        if step % config.eval_interval == 0 or step == config.max_steps - 1:
            losses = estimate_loss(
                model=model,
                dataset=dataset,
                eval_batches=config.eval_batches,
                batch_size=config.batch_size,
            )
            print(
                f"step {step:4d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )

        xb, yb = dataset.get_batch("train", config.batch_size)
        _, loss = model(xb, yb)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        path=config.checkpoint_path(),
        model=model,
        tokenizer=tokenizer,
        config=config,
    )
    print(f"saved checkpoint to {config.checkpoint_path()}")


if __name__ == "__main__":
    main()
