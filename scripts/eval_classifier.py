"""
Evaluation script for the classifier.
- Loads model and tokenizer from outputs/classifier_test_v2
- Loads the tokenized dataset
- Computes Accuracy, F1, Precision, Recall
- Generates Confusion Matrix and Classification Report
"""
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm

MODEL_DIR = "outputs/classifier_test_v2"
TOKENIZED_DIR = "data_processed/goemo_roberta_singlelabel_v2"
BATCH_SIZE = 16

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    device = get_device()
    print(f"Using device: {device}")

    # Verify dataset path - fallback or error if not found
    dataset_path = Path(TOKENIZED_DIR)
    if not dataset_path.exists():
        # Try to find it or ask user? For now, let's assume standard path or fail
        print(f"Dataset not found at {TOKENIZED_DIR}. Checking for alternatives...")
        # Fallback logic could go here, but let's stick to the plan.
        pass

    print(f"Loading model from {MODEL_DIR}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Loading dataset from {TOKENIZED_DIR}")
    try:
        ds = load_from_disk(TOKENIZED_DIR)
        test_ds = ds["test"]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Test set size: {len(test_ds)}")

    all_preds = []
    all_labels = []

    # Inference loop
    print("Running inference...")
    for i in tqdm(range(0, len(test_ds), BATCH_SIZE)):
        batch = test_ds[i : i + BATCH_SIZE]
        inputs = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(batch["labels"])

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")

    # Save artifacts
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    # Classification Report
    report = classification_report(all_labels, all_preds)
    with open(docs_dir / "classification_report.txt", "w") as f:
        f.write(report)
        f.write(f"\nAccuracy: {acc}\nMacro F1: {f1}\n")
    print(f"Saved classification report to {docs_dir / 'classification_report.txt'}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(docs_dir / "confusion_matrix.png")
    print(f"Saved confusion matrix to {docs_dir / 'confusion_matrix.png'}")

if __name__ == "__main__":
    main()
