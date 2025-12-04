"""
Script to generate label distribution plot from the processed dataset.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from pathlib import Path
import pandas as pd

# Adjust this path if your dataset is named differently
DATASET_DIR = "data_processed/goemo_roberta_singlelabel_v2" 

def main():
    print(f"Loading dataset from {DATASET_DIR}")
    try:
        ds = load_from_disk(DATASET_DIR)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Combine train and test for full distribution or just use train? 
    # Usually distribution of the whole dataset or training set is interesting.
    # Let's use 'train' if available, else whatever is there.
    if "train" in ds:
        data = ds["train"]
        split_name = "Train"
    else:
        data = ds[list(ds.keys())[0]]
        split_name = "Dataset"

    labels = data["labels"]
    
    # Convert to pandas for easier plotting
    df = pd.DataFrame({"label": labels})
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x="label", data=df, palette="viridis")
    plt.title(f"Label Distribution ({split_name} Set)")
    plt.xlabel("Label ID")
    plt.ylabel("Count")
    
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    save_path = docs_dir / "label_distribution.png"
    plt.savefig(save_path)
    print(f"Saved label distribution plot to {save_path}")

if __name__ == "__main__":
    main()
