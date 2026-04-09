from src.tokenizer import CharTokenizer


def test_encode_decode_roundtrip() -> None:
    text = "hello"
    tokenizer = CharTokenizer()
    tokenizer.fit(text)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text
    assert tokenizer.vocab_size == len(set(text))
