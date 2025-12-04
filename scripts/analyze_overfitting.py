import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

# Constants
MODEL_DIR = "outputs/summarizer_test"
DATA_DIR = "data_processed/amazon_bart_tokenized_with_targets"
NUM_SAMPLES = 50

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def calculate_overlap(text1, text2):
    """Calculates n-gram overlap."""
    return 0 # Placeholder for now, we'll use simple string checks first

def main():
    print("="*60)
    print("🔍 SUMMARIZER DIAGNOSIS & LEAKAGE CHECK")
    print("="*60)

    # 1. Check for Data Leakage
    print("\n1. Checking for Data Leakage (Train/Test Overlap)...")
    try:
        ds = load_from_disk(DATA_DIR)
        train_texts = set(ds["train"]["content"])
        test_texts = set(ds["test"]["content"])
        
        overlap = train_texts.intersection(test_texts)
        print(f"   Train size: {len(train_texts)}")
        print(f"   Test size:  {len(test_texts)}")
        print(f"   Overlap:    {len(overlap)} samples")
        
        if len(overlap) > 0:
            print(f"   ⚠️  WARNING: Found {len(overlap)} overlapping samples!")
            print(f"   Leakage Rate: {len(overlap)/len(test_texts)*100:.2f}%")
        else:
            print("   ✅ No exact text overlap found between train and test.")
            
    except Exception as e:
        print(f"   ❌ Error checking leakage: {e}")

    # 2. Analyze Model Output (Overfitting/Copying)
    print(f"\n2. Analyzing Model Output ({NUM_SAMPLES} samples)...")
    device = get_device()
    print(f"   Loading model from {MODEL_DIR} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    
    test_ds = ds["test"].select(range(NUM_SAMPLES))
    
    results = []
    
    print("   Generating summaries...")
    for item in tqdm(test_ds):
        input_text = item["content"]
        ref_summary = item["summary"]
        
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            # Use default generation params
            # Ensure decoder_start_token_id is set
            if model.config.decoder_start_token_id is None:
                model.config.decoder_start_token_id = tokenizer.bos_token_id
            
            outputs = model.generate(**inputs, max_length=128, decoder_start_token_id=model.config.decoder_start_token_id)
            
        gen_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Metrics
        is_exact_copy = (gen_summary.strip() == input_text.strip())
        is_ref_match = (gen_summary.strip() == ref_summary.strip())
        
        # Check if summary is a substring of input (Extractive)
        is_substring = gen_summary.strip() in input_text.strip()
        
        results.append({
            "input": input_text,
            "gen": gen_summary,
            "ref": ref_summary,
            "is_copy": is_exact_copy,
            "is_ref_match": is_ref_match,
            "is_substring": is_substring,
            "len_input": len(input_text.split()),
            "len_gen": len(gen_summary.split())
        })
        
    df = pd.DataFrame(results)
    
    # Statistics
    copy_rate = df["is_copy"].mean() * 100
    ref_match_rate = df["is_ref_match"].mean() * 100
    substring_rate = df["is_substring"].mean() * 100
    avg_compression = (df["len_gen"] / df["len_input"]).mean()
    
    print("\n📊 DIAGNOSIS RESULTS")
    print(f"   Exact Copy Rate (Input == Gen): {copy_rate:.1f}%")
    print(f"   Reference Match Rate (Ref == Gen): {ref_match_rate:.1f}%")
    print(f"   Substring Rate (Gen inside Input): {substring_rate:.1f}%")
    print(f"   Avg Compression Ratio: {avg_compression:.2f}")
    
    print("\n📝 Sample Analysis:")
    for i in range(5):
        row = df.iloc[i]
        print(f"\n--- Sample {i+1} ---")
        print(f"INPUT: {row['input'][:100]}...")
        print(f"GEN  : {row['gen']}")
        print(f"REF  : {row['ref']}")
        print(f"Type : {'COPY' if row['is_copy'] else 'NEW'}")

    # Save report
    with open("docs/model_diagnosis_report.txt", "w") as f:
        f.write("SUMMARIZER DIAGNOSIS REPORT\n")
        f.write("===========================\n")
        f.write(f"Leakage: {len(overlap)} samples\n")
        f.write(f"Exact Copy Rate: {copy_rate:.1f}%\n")
        f.write(f"Ref Match Rate: {ref_match_rate:.1f}%\n")
        f.write(f"Substring Rate: {substring_rate:.1f}%\n")
        f.write(f"Avg Compression: {avg_compression:.2f}\n\n")
        f.write("SAMPLES:\n")
        for i, row in df.iterrows():
            f.write(f"\nSample {i+1}:\n")
            f.write(f"Input: {row['input']}\n")
            f.write(f"Gen  : {row['gen']}\n")
            f.write(f"Ref  : {row['ref']}\n")
            f.write("-" * 20 + "\n")
            
    print("\n✅ Report saved to docs/model_diagnosis_report.txt")

if __name__ == "__main__":
    main()
