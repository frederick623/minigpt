import argparse

import torch

from config import GPTConfig
from src.model import MiniGPT
from src.sample import generate_text
from src.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with mini-gpt.")
    parser.add_argument("--prompt", type=str, default="", help="Seed text prompt.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the configured generation length.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Restrict sampling to the top-k logits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GPTConfig()
    checkpoint = load_checkpoint(config.checkpoint_path(), map_location="cpu")

    saved_config = checkpoint["config"]
    tokenizer_state = checkpoint["tokenizer"]

    model = MiniGPT(
        vocab_size=tokenizer_state["vocab_size"],
        block_size=saved_config["block_size"],
        n_layers=saved_config["n_layers"],
        n_heads=saved_config["n_heads"],
        n_embd=saved_config["n_embd"],
        dropout=saved_config["dropout"],
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    prompt = args.prompt or "The "
    max_new_tokens = args.max_new_tokens or saved_config["max_new_tokens"]
    temperature = args.temperature or saved_config["temperature"]
    top_k = args.top_k if args.top_k is not None else saved_config["top_k"]

    output = generate_text(
        model=model,
        tokenizer_state=tokenizer_state,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    print(output)


if __name__ == "__main__":
    main()
