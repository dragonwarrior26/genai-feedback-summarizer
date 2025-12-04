#!/usr/bin/env python3
"""
Robust train_summarizer.py (compat shim for TrainingArguments.generation_config)

This script prepares seq2seq labels and trains using Seq2SeqTrainer when available.
It adds a safe default 'generation_config' attribute to TrainingArguments for compatibility.
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
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )
    # optional imports
    try:
        from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq
        _HAS_SEQ2SEQ = True
    except Exception:
        Seq2SeqTrainer = None
        DataCollatorForSeq2Seq = None
        _HAS_SEQ2SEQ = False
except Exception as e:
    print("Missing dependencies or import error. Run: pip install transformers datasets", file=sys.stderr)
    raise

def prepare_dataset(tokenized_dir, tokenizer, max_input_length=256, max_target_length=64):
    ds = load_from_disk(tokenized_dir)
    def preprocess_batch(batch):
        inputs = tokenizer(batch["content"], truncation=True, padding="max_length", max_length=max_input_length)
        targets = tokenizer(batch.get("summary", [""]*len(batch["content"])), truncation=True, padding="max_length", max_length=max_target_length)
        labels = targets["input_ids"]
        labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]
        inputs["labels"] = labels
        return inputs

    remove_cols = [c for c in ds["train"].column_names if c not in ("content", "summary")]
    tokenized = ds.map(preprocess_batch, batched=True, remove_columns=remove_cols)
    return tokenized

def build_training_args(out_dir, batch, epochs, logging_dir=None):
    base_kwargs = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        num_train_epochs=epochs,
        logging_dir=(logging_dir or str(out_dir)),
        logging_steps=10,
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
    parser.add_argument("--model", default="t5-small")
    parser.add_argument("--output_dir", default="outputs/summarizer")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max_input_len", type=int, default=256)
    parser.add_argument("--max_target_len", type=int, default=64)
    args = parser.parse_args()

    try:
        print("\n=== Loading Dataset ===")
        tokenized_dir = Path(args.tokenized_dir)
        if not tokenized_dir.exists():
            raise FileNotFoundError(f"{tokenized_dir} not found")

        print("\n=== Loading Model & Tokenizer ===")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

        print("\n=== Preparing dataset (tokenizing targets) ===")
        ds = prepare_dataset(args.tokenized_dir, tokenizer, args.max_input_len, args.max_target_len)
        print(ds)

        print("\n=== Building TrainingArguments ===")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ta_kwargs = build_training_args(out_dir, args.batch, args.epochs, logging_dir=str(out_dir / "logs"))
        print("TrainingArguments kwargs:", ta_kwargs)
        training_args = TrainingArguments(**ta_kwargs)

        # Compatibility shim: older/newer transformers may not attach generation_config
        if not hasattr(training_args, "generation_config"):
            try:
                # try importing GenerationConfig for a minimal default if available
                from transformers import GenerationConfig
                training_args.generation_config = GenerationConfig()
            except Exception:
                # fallback to None (Seq2SeqTrainer tolerates None in most versions)
                training_args.generation_config = None

        # Choose trainer
        if _HAS_SEQ2SEQ and DataCollatorForSeq2Seq is not None and Seq2SeqTrainer is not None:
            print("Using Seq2SeqTrainer + DataCollatorForSeq2Seq")
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=ds["train"],
                eval_dataset=ds["test"] if "test" in ds else None,
                data_collator=data_collator,
                tokenizer=tokenizer,
            )
        else:
            print("Seq2SeqTrainer not available, falling back to Trainer + default collator")
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=ds["train"],
                eval_dataset=ds["test"] if "test" in ds else None,
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
