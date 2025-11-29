from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized_dir", required=True)
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--output_dir", default="outputs/classifier")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--num_labels", type=int, default=28)
    args = parser.parse_args()

    print("\n=== Loading Dataset ===")
    ds = load_from_disk(args.tokenized_dir)

    print("\n=== Loading Model & Tokenizer ===")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=args.num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def collate_fn(batch):
        return tokenizer.pad(batch, return_tensors="pt")

    print("\n=== Starting Training ===")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        logging_steps=20,
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    print("\nDONE. Saved classifier at:", args.output_dir)

if __name__ == "__main__":
    main()