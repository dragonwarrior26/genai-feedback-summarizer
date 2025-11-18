#!/usr/bin/env python3
"""
scripts/data_downloader.py

Simple helper to download datasets via Hugging Face `datasets` library,
sample a manageable subset, and save it under data_raw/.

Usage:
    python scripts/data_downloader.py --dataset amazon_polarity --samples 20000
"""
import argparse
from pathlib import Path
from datasets import load_dataset

ROOT = Path.cwd()
DATA_DIR = ROOT / "data_raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_and_save(dataset_name: str, split: str = "train", samples: int = 20000):
    print(f"Downloading dataset: {dataset_name} split={split} samples={samples}")
    ds = load_dataset(dataset_name, split=f"{split}[:{samples}]")
    out_path = DATA_DIR / f"{dataset_name.replace('/', '_')}_{split}_{samples}.csv"
    df = ds.to_pandas()
    df.to_csv(out_path, index=False)
    print(f"Saved sample to {out_path}")
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="amazon_polarity")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()
    download_and_save(args.dataset, args.split, args.samples)

if __name__ == "__main__":
    main()