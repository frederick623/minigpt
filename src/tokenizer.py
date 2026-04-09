class CharTokenizer:
    def __init__(self) -> None:
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    def fit(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi = {ch: idx for idx, ch in enumerate(chars)}
        self.itos = {idx: ch for ch, idx in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        if not self.stoi:
            raise ValueError("Tokenizer has not been fit yet.")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        if not self.itos:
            raise ValueError("Tokenizer has not been fit yet.")
        return "".join(self.itos[idx] for idx in ids)

    def state_dict(self) -> dict[str, object]:
        return {
            "stoi": self.stoi,
            "itos": self.itos,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "CharTokenizer":
        tokenizer = cls()
        tokenizer.stoi = dict(state["stoi"])
        tokenizer.itos = {int(k): v for k, v in dict(state["itos"]).items()}
        return tokenizer
