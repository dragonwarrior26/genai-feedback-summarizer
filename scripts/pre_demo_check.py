#!/usr/bin/env python3
"""
pre_demo_check.py

Simulates the app's logic to verify everything works end-to-end.
"""
import sys
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

# Constants from app.py
SUMMARIZER_PATH_V1 = "outputs/summarizer_test"
SUMMARIZER_PATH_V2 = "outputs/summarizer_v2"
CLASSIFIER_PATH_V1 = "outputs/classifier_test_v2"
CLASSIFIER_PATH_V2 = "outputs/classifier_v2"
INSIGHTS_FILE = "outputs/insights/actionable_insights.json"

def check_models():
    print("\n=== Checking Models ===")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Summarizer
    sum_path = SUMMARIZER_PATH_V1
    if Path(SUMMARIZER_PATH_V2).exists() and (Path(SUMMARIZER_PATH_V2) / "config.json").exists():
        sum_path = SUMMARIZER_PATH_V2
    
    print(f"Loading Summarizer from: {sum_path}")
    try:
        sum_tok = AutoTokenizer.from_pretrained(sum_path)
        sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_path).to(device)
        print("✅ Summarizer loaded.")
        
        # Test Generation
        print("Testing Summarizer Generation...")
        text = "This product is amazing. I love it so much. It works perfectly."
        inputs = sum_tok(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
        
        if sum_model.config.decoder_start_token_id is None:
            sum_model.config.decoder_start_token_id = sum_tok.bos_token_id
            
        summary_ids = sum_model.generate(
            inputs["input_ids"], 
            max_length=50, 
            min_length=5, 
            decoder_start_token_id=sum_model.config.decoder_start_token_id
        )
        summary = sum_tok.decode(summary_ids[0], skip_special_tokens=True)
        print(f"✅ Summary generated: {summary}")
        
    except Exception as e:
        print(f"❌ Summarizer Failed: {e}")
        sys.exit(1)

    # Classifier
    cls_path = CLASSIFIER_PATH_V1
    if Path(CLASSIFIER_PATH_V2).exists() and (Path(CLASSIFIER_PATH_V2) / "config.json").exists():
        cls_path = CLASSIFIER_PATH_V2
    
    print(f"Loading Classifier from: {cls_path}")
    try:
        cls_tok = AutoTokenizer.from_pretrained(cls_path)
        cls_model = AutoModelForSequenceClassification.from_pretrained(cls_path).to(device)
        print("✅ Classifier loaded.")
        
        # Test Inference
        print("Testing Classifier Inference...")
        inputs = cls_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = cls_model(**inputs).logits
        label_id = logits.argmax().item()
        print(f"✅ Classification result: Label {label_id}")
        
    except Exception as e:
        print(f"❌ Classifier Failed: {e}")
        sys.exit(1)

def check_insights():
    print("\n=== Checking Insights ===")
    if not Path(INSIGHTS_FILE).exists():
        print(f"❌ Insights file not found: {INSIGHTS_FILE}")
        sys.exit(1)
        
    try:
        with open(INSIGHTS_FILE, "r") as f:
            insights = json.load(f)
        print(f"✅ Loaded {len(insights)} insights.")
        print("Sample Insight:", insights[0]["actionable_suggestion"])
    except Exception as e:
        print(f"❌ Failed to load insights: {e}")
        sys.exit(1)

def main():
    check_models()
    check_insights()
    print("\n✅✅✅ ALL SYSTEMS GO! DEMO IS READY. ✅✅✅")

if __name__ == "__main__":
    main()
