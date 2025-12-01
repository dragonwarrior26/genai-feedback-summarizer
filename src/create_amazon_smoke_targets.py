from datasets import load_from_disk, DatasetDict
from pathlib import Path

SRC = "data_processed/amazon_bart_tokenized"
OUT = "data_processed/amazon_bart_tokenized_with_targets"

print("Loading:", SRC)
ds = load_from_disk(SRC)

def make_summary(example):
    # Simple extractive pseudo-summary: first 30 words
    text = example.get("content", "")
    words = text.split()
    summary = " ".join(words[:30])
    return {"summary": summary}

print("Mapping summaries...")
ds_proc = {}
for split, d in ds.items():
    d2 = d.map(make_summary)
    ds_proc[split] = d2

ds_new = DatasetDict(ds_proc)
Path(OUT).mkdir(parents=True, exist_ok=True)
ds_new.save_to_disk(OUT)
print("Saved pseudo-target dataset to:", OUT)
