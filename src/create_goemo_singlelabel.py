import re
from datasets import load_from_disk, DatasetDict
from pathlib import Path

SRC = "data_processed/goemo_roberta_tokenized"
OUT = "data_processed/goemo_roberta_singlelabel_v2"

print("Loading:", SRC)
ds = load_from_disk(SRC)

int_re = re.compile(r"-?\d+")

def to_single_label_batch(batch):
    out = {"labels": []}
    for labels in batch.get("labels", []):
        lab = 0
        # If already int
        if isinstance(labels, (int, float)):
            lab = int(labels)
        else:
            # Convert to string and extract integers robustly (handles "[2]", "[ 3 10]", "1,2,3", etc.)
            s = str(labels)
            found = int_re.findall(s)
            if len(found) == 0:
                lab = 0
            else:
                # pick the first integer as the single label for smoke testing
                lab = int(found[0])
        out["labels"].append(lab)
    return out

print("Mapping single_label (batched)...")
ds_mapped = ds.map(to_single_label_batch, batched=True, batch_size=512)

# Save as new dataset
Path(OUT).mkdir(parents=True, exist_ok=True)
ds_mapped.save_to_disk(OUT)
print("Saved single-label dataset to:", OUT)

# Quick inspect
train_labels = ds_mapped["train"]["labels"]
print("Example labels (first 40):", train_labels[:40])
print("Unique labels count (train):", len(set(train_labels)))
