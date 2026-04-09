import torch

from src.dataset import TextDataset
from src.tokenizer import CharTokenizer


def test_dataset_batch_shapes() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 10
    tokenizer = CharTokenizer()
    tokenizer.fit(text)

    dataset = TextDataset(
        text=text,
        tokenizer=tokenizer,
        block_size=8,
        train_split=0.9,
        device="cpu",
    )
    xb, yb = dataset.get_batch("train", batch_size=4)

    assert xb.shape == (4, 8)
    assert yb.shape == (4, 8)
    assert torch.equal(xb[:, 1:], yb[:, :-1])
