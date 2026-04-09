import torch

from src.tokenizer import CharTokenizer


class TextDataset:
    def __init__(
        self,
        text: str,
        tokenizer: CharTokenizer,
        block_size: int,
        train_split: float,
        device: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.device = device

        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        split_idx = int(len(data) * train_split)
        self.train_data = data[:split_idx]
        self.val_data = data[split_idx:]

        if len(self.train_data) <= block_size or len(self.val_data) <= block_size:
            raise ValueError(
                "Dataset is too small for the configured block_size and train_split."
            )

    def get_batch(self, split: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        starts = torch.randint(len(data) - self.block_size - 1, (batch_size,))

        x = torch.stack([data[i : i + self.block_size] for i in starts])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in starts])
        return x.to(self.device), y.to(self.device)
