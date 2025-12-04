#!/usr/bin/env python3
"""
train_classifier_v2.py

Retrains the classifier with consolidated labels (28 -> 8 classes) and class weighting.
"""
import argparse
from pathlib import Path
import sys
import json
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score

try:
    from datasets import load_from_disk
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )
except Exception as e:
    print("Missing dependencies. Run: pip install transformers datasets scikit-learn", file=sys.stderr)
    raise

# Mapping from 28 GoEmotions labels to 8 clusters
# Based on standard alphabetical order of GoEmotions labels
LABEL_MAPPING = {
    0: 0,   # admiration -> admiration
    1: 1,   # amusement -> joy
    2: 3,   # anger -> anger
    3: 3,   # annoyance -> anger
    4: 2,   # approval -> approval
    5: 0,   # caring -> admiration
    6: 6,   # confusion -> surprise
    7: 6,   # curiosity -> surprise
    8: 0,   # desire -> admiration
    9: 4,   # disappointment -> sadness
    10: 3,  # disapproval -> anger
    11: 3,  # disgust -> anger
    12: 4,  # embarrassment -> sadness
    13: 1,  # excitement -> joy
    14: 5,  # fear -> fear
    15: 1,  # gratitude -> joy
    16: 4,  # grief -> sadness
    17: 1,  # joy -> joy
    18: 0,  # love -> admiration
    19: 5,  # nervousness -> fear
    20: 1,  # optimism -> joy
    21: 1,  # pride -> joy
    22: 6,  # realization -> surprise
    23: 1,  # relief -> joy
    24: 4,  # remorse -> sadness
    25: 4,  # sadness -> sadness
    26: 6,  # surprise -> surprise
    27: 7   # neutral -> neutral
}

NEW_LABELS = [
    "Admiration", "Joy", "Approval", "Anger", "Sadness", "Fear", "Surprise", "Neutral"
]

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "f1": f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized_dir", required=True)
    parser.add_argument("--model", default="distilroberta-base")
    parser.add_argument("--output_dir", default="outputs/classifier_v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    print(f"\n=== Training Classifier V2 on {get_device()} ===")
    
    # Load dataset
    print(f"Loading dataset from {args.tokenized_dir}")
    ds = load_from_disk(args.tokenized_dir)
    
    # Remap labels
    print("Remapping labels to 8 clusters...")
    def remap_labels(batch):
        batch["labels"] = [LABEL_MAPPING.get(l, 7) for l in batch["labels"]] # Default to Neutral if unknown
        return batch
    
    ds = ds.map(remap_labels, batched=True)
    
    # Compute class weights
    print("Computing class weights...")
    train_labels = ds["train"]["labels"]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )
    print("Class weights:", class_weights)
    
    # Load Model & Tokenizer
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(NEW_LABELS),
        id2label={i: l for i, l in enumerate(NEW_LABELS)},
        label2id={l: i for i, l in enumerate(NEW_LABELS)}
    )
    
    # Training Args
    out_dir = Path(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        logging_dir=str(out_dir / "logs"),
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        use_mps_device=(get_device() == "mps"),
    )
    
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    print("\n=== Starting Training ===")
    trainer.train()
    
    print("\n=== Saving Model ===")
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print("\nDONE. Saved model at:", out_dir)

if __name__ == "__main__":
    main()
