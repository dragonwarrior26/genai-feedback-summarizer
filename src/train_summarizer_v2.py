#!/usr/bin/env python3
"""
train_summarizer_v2.py

Retrains the summarizer with specific hyperparameters to reduce overfitting/copying:
- Shorter max_target_length (64)
- Beam search with diversity penalty
- Early stopping
"""
import argparse
from pathlib import Path
import sys
import traceback
import torch

try:
    from datasets import load_from_disk
    import numpy as np
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        TrainingArguments,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
        DataCollatorForSeq2Seq,
        GenerationConfig
    )
except Exception as e:
    print("Missing dependencies. Run: pip install transformers datasets", file=sys.stderr)
    raise

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def prepare_dataset(tokenized_dir, tokenizer, max_input_length=256, max_target_length=64):
    ds = load_from_disk(tokenized_dir)
    
    # Force re-tokenization to enforce new lengths
    cols_to_remove = [c for c in ds["train"].column_names if c in ["input_ids", "attention_mask", "labels", "decoder_input_ids"]]
    if cols_to_remove:
        print(f"Removing existing tokenization columns to enforce new max_target_length={max_target_length}: {cols_to_remove}")
        ds = ds.remove_columns(cols_to_remove)

    def preprocess_batch(batch):
        inputs = tokenizer(batch["content"], truncation=True, padding="max_length", max_length=max_input_length)
        targets = tokenizer(batch.get("summary", [""]*len(batch["content"])), truncation=True, padding="max_length", max_length=max_target_length)
        
        labels = targets["input_ids"]
        # Replace pad token with -100 for loss computation
        labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]
        
        inputs["labels"] = labels
        return inputs

    remove_cols = [c for c in ds["train"].column_names if c not in ("content", "summary")]
    tokenized = ds.map(preprocess_batch, batched=True, remove_columns=remove_cols)
    return tokenized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized_dir", required=True)
    parser.add_argument("--model", default="facebook/bart-base") # Start fresh or from previous? Let's use base to unlearn copying
    parser.add_argument("--output_dir", default="outputs/summarizer_v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max_input_len", type=int, default=256)
    parser.add_argument("--max_target_len", type=int, default=64) # Shorter summary
    args = parser.parse_args()

    try:
        print(f"\n=== Training Summarizer V2 on {get_device()} ===")
        
        print("\n=== Loading Model & Tokenizer ===")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        
        # Configure Generation for Training/Eval
        model.config.max_length = args.max_target_len
        model.config.min_length = 10
        model.config.num_beams = 4
        model.config.early_stopping = True
        model.config.no_repeat_ngram_size = 3
        
        # Diversity penalty to encourage unique generation
        # Note: diversity_penalty requires num_beam_groups > 1
        model.config.num_beam_groups = 4
        model.config.diversity_penalty = 0.5

        print("\n=== Preparing dataset ===")
        ds = prepare_dataset(args.tokenized_dir, tokenizer, args.max_input_len, args.max_target_len)
        
        print("\n=== Building TrainingArguments ===")
        out_dir = Path(args.output_dir)
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            num_train_epochs=args.epochs,
            logging_dir=str(out_dir / "logs"),
            logging_steps=50,
            save_strategy="epoch",
            eval_strategy="epoch",
            predict_with_generate=True,
            fp16=False, # MPS doesn't support fp16 well yet, use fp32
            use_mps_device=(get_device() == "mps"),
            generation_max_length=args.max_target_len,
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["test"].select(range(100)) if "test" in ds else None, # Small eval set for speed
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        print("\n=== Starting Training ===")
        trainer.train()
        
        print("\n=== Saving Model ===")
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        print("\nDONE. Saved model at:", out_dir)
        
    except Exception:
        print("\nERROR during training:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
