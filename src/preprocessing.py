import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from pathlib import Path

def tokenize_function(batch, tokenizer, text_col, max_length):
    return tokenizer(
        batch[text_col],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--text_col", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--val_fraction", type=float, default=0.1)

    args = parser.parse_args()

    print("\n=== Loading CSV ===")
    df = pd.read_csv(args.input_csv)
    print("Rows:", len(df))

    print("\n=== Loading Tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("\n=== Converting to HF Dataset ===")
    ds = Dataset.from_pandas(df)

    print("\n=== Tokenizing ===")
    tokenized = ds.map(
        lambda batch: tokenize_function(batch, tokenizer, args.text_col, args.max_length),
        batched=True
    )

    print("\n=== Splitting Train/Validation ===")
    split_ds = tokenized.train_test_split(test_size=args.val_fraction)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Saving Dataset to Disk ===")
    split_ds.save_to_disk(args.out_dir)

    print("\n=== Saving Tokenizer ===")
    tokenizer.save_pretrained(out_dir / "tokenizer")

    print("\nDONE. Saved at:", args.out_dir)

if __name__ == "__main__":
    main()

