# mini-gpt

A minimal GPT-like language model built from scratch in PyTorch.

This project is intentionally small and educational. It implements a decoder-only
Transformer with:

- character-level tokenization
- causal self-attention
- learned positional embeddings
- next-token prediction training
- autoregressive text generation

## Project layout

```text
mini-gpt/
├─ README.md
├─ requirements.txt
├─ config.py
├─ train.py
├─ generate.py
├─ data/
│  ├─ input.txt
│  └─ prepare.py
├─ src/
│  ├─ __init__.py
│  ├─ tokenizer.py
│  ├─ dataset.py
│  ├─ attention.py
│  ├─ block.py
│  ├─ model.py
│  ├─ sample.py
│  └─ utils.py
└─ tests/
   ├─ test_tokenizer.py
   ├─ test_dataset.py
   └─ test_model.py
```

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies.
3. Train the model.
4. Generate text from a prompt.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python generate.py --prompt "The "
```

## Notes

- The default tokenizer is character-level to keep the code easy to inspect.
- The default model is tiny and meant for learning, not production use.
- Start by overfitting the small sample corpus before scaling up.

## Files to look at first

- `src/attention.py`: causal self-attention
- `src/block.py`: one Transformer block
- `src/model.py`: full decoder-only GPT
- `train.py`: training loop
- `generate.py`: autoregressive sampling
