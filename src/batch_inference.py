"""
Batch inference script.
Reads a CSV file, generates summaries and classifications, and saves the result.
"""
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from tqdm import tqdm
import sys

# Default paths
SUMMARIZER_PATH = "outputs/summarizer_test"
CLASSIFIER_PATH = "outputs/classifier_test_v2"

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(device):
    print("Loading models...")
    try:
        sum_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_PATH)
        sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_PATH).to(device)
        
        cls_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_PATH)
        cls_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_PATH).to(device)
        
        return (sum_model, sum_tokenizer), (cls_model, cls_tokenizer)
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Batch Inference")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--text_col", default="text", help="Column name for input text")
    args = parser.parse_args()

    device = get_device()
    (sum_model, sum_tok), (cls_model, cls_tok) = load_models(device)

    print(f"Reading input from {args.input}")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if args.text_col not in df.columns:
        # Fallback: try 'content' or first column
        if "content" in df.columns:
            args.text_col = "content"
        else:
            print(f"Column '{args.text_col}' not found. Using first column.")
            args.text_col = df.columns[0]

    summaries = []
    labels = []

    print("Running inference...")
    for text in tqdm(df[args.text_col]):
        # Summarize
        inputs = sum_tok(str(text), return_tensors="pt", max_length=1024, truncation=True).to(device)
        with torch.no_grad():
            summary_ids = sum_model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summaries.append(sum_tok.decode(summary_ids[0], skip_special_tokens=True))

        # Classify
        inputs = cls_tok(str(text), return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = cls_model(**inputs).logits
        labels.append(logits.argmax().item())

    df["generated_summary"] = summaries
    df["predicted_label"] = labels

    print(f"Saving results to {args.output}")
    df.to_csv(args.output, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
