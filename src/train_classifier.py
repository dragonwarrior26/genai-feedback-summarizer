"""
Robust train_classifier.py

- Loads a tokenized dataset (expects 'text' and integer 'labels' columns)
- Builds TrainingArguments compatibly with the installed transformers
- Uses Trainer and a simple data collator for padding
"""
import argparse
from pathlib import Path
import inspect
import sys
import traceback

try:
    from datasets import load_from_disk
    import numpy as np
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )
    # data collator: use tokenizer.pad (works as simple collator)
except Exception as e:
    print("Missing dependencies or import error. Run: pip install transformers datasets", file=sys.stderr)
    raise

def build_training_args(out_dir, batch, epochs, logging_dir=None):
    base_kwargs = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        num_train_epochs=epochs,
        logging_dir=(logging_dir or str(out_dir)),
        logging_steps=20,
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=False,
    )
    sig = inspect.signature(TrainingArguments.__init__)
    valid = {k: v for k, v in base_kwargs.items() if k in sig.parameters}
    return valid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized_dir", required=True)
    parser.add_argument("--model", default="distilroberta-base")
    parser.add_argument("--output_dir", default="outputs/classifier")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--num_labels", type=int, required=True)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    try:
        print("\n=== Loading Dataset ===")
        tokenized_dir = Path(args.tokenized_dir)
        if not tokenized_dir.exists():
            raise FileNotFoundError(f"{tokenized_dir} not found")

        ds = load_from_disk(args.tokenized_dir)
        print(ds)

        print("\n=== Loading Model & Tokenizer ===")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)

        print("\n=== Building TrainingArguments ===")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ta_kwargs = build_training_args(out_dir, args.batch, args.epochs, logging_dir=str(out_dir / "logs"))
        print("TrainingArguments kwargs:", ta_kwargs)
        training_args = TrainingArguments(**ta_kwargs)

        # Build a simple collator using tokenizer.pad
        def collate_fn(batch):
            return tokenizer.pad(batch, return_tensors="pt")

        print("\n=== Creating Trainer ===")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["test"] if "test" in ds else None,
            data_collator=collate_fn,
            tokenizer=tokenizer,
        )

        print("\n=== Starting Training ===")
        trainer.train()
        trainer.save_model(str(out_dir))
        print("\nDONE. Saved model at:", out_dir)
    except Exception:
        print("\nERROR during training:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
