from pathlib import Path


def main() -> None:
    data_path = Path(__file__).parent / "input.txt"
    text = data_path.read_text(encoding="utf-8")
    print(f"characters: {len(text)}")
    print(f"lines: {len(text.splitlines())}")
    print("preview:")
    print(text[:200])


if __name__ == "__main__":
    main()
