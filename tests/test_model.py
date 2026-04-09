import torch

from src.model import MiniGPT


def test_model_forward_shapes() -> None:
    model = MiniGPT(
        vocab_size=32,
        block_size=8,
        n_layers=2,
        n_heads=4,
        n_embd=32,
        dropout=0.0,
    )
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))

    logits, loss = model(x, y)

    assert logits.shape == (2, 8, 32)
    assert loss is not None
    assert loss.ndim == 0
