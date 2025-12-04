"""
Unified inference script for GenAI Feedback Summarizer.
Loads both summarizer and classifier models to process input text.
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import sys

# Default paths - update if models are moved
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

def summarize(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def classify(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return predicted_class_id # You might want to map this to a label name if you have the mapping

def main():
    parser = argparse.ArgumentParser(description="GenAI Feedback Summarizer Inference")
    parser.add_argument("--text", type=str, required=True, help="Input feedback text")
    args = parser.parse_args()

    device = get_device()
    (sum_model, sum_tok), (cls_model, cls_tok) = load_models(device)

    print("\n--- Input Text ---")
    print(args.text)

    print("\n--- Summary ---")
    summary = summarize(args.text, sum_model, sum_tok, device)
    print(summary)

    print("\n--- Classification ---")
    label = classify(args.text, cls_model, cls_tok, device)
    print(f"Predicted Label ID: {label}")

if __name__ == "__main__":
    main()
