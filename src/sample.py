import torch

from src.model import MiniGPT
from src.tokenizer import CharTokenizer


def generate_text(
    model: MiniGPT,
    tokenizer_state: dict[str, object],
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> str:
    tokenizer = CharTokenizer.from_state_dict(tokenizer_state)

    if not prompt:
        prompt = " "

    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        output_ids = model.generate(
            idx=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    return tokenizer.decode(output_ids[0].tolist())
