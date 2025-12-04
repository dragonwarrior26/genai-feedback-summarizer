"""
Batch evaluation script for the summarizer.
- Loads model and tokenizer from outputs/summarizer_test
- Loads the tokenized dataset with targets (summary)
- Generates summaries in batches (CPU/GPU aware)
- Computes ROUGE and prints results
"""
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
from tqdm import tqdm
from pathlib import Path
import math

MODEL_DIR = "outputs/summarizer_test"
TOKENIZED_DIR = "data_processed/amazon_bart_tokenized_with_targets"
BATCH_SIZE = 8   # increase incase if you have GPU memory
MAX_GEN_LEN = 64
NUM_BEAMS = 4

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_decoder_start_id(model, tokenizer):
    # Try common locations for decoder start token id
    dec_id = getattr(model.config, "decoder_start_token_id", None)
    if dec_id is not None:
        return dec_id
    # fallback to tokenizer.pad_token_id if exists
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return tokenizer.pad_token_id
    # fallback to bos token
    if getattr(tokenizer, "bos_token_id", None) is not None:
        return tokenizer.bos_token_id
    # last resort: use tokenizer.convert_tokens_to_ids for common tokens
    for tok in ("<pad>", "<s>", "</s>", "<bos>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid and tid != tokenizer.unk_token_id:
                return tid
        except Exception:
            pass
    return None

def batchify(iterable, n):
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(n):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk

def main():
    device = get_device()
    print("Device:", device)
    print("Loading model/tokenizer from:", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    dec_start = safe_decoder_start_id(model, tokenizer)
    print("Using decoder_start_token_id =", dec_start)

    print("Loading dataset:", TOKENIZED_DIR)
    ds = load_from_disk(TOKENIZED_DIR)
    test = ds["test"]
    print("Test size:", len(test))

    rouge = evaluate.load("rouge")

    preds = []
    refs = []

    # Generate in batches
    for batch_idxs in tqdm(list(batchify(range(len(test)), BATCH_SIZE))):
        batch_texts = [ test[i]["content"] for i in batch_idxs ]
        # tokenize inputs in batch
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        gen_kwargs = dict(max_length=MAX_GEN_LEN, num_beams=NUM_BEAMS)
        if dec_start is not None:
            gen_kwargs["decoder_start_token_id"] = dec_start
        with torch.no_grad():
            generated = model.generate(**inputs, **gen_kwargs)
        # decode each
        batch_preds = [ tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated ]
        preds.extend(batch_preds)
        refs.extend([ test[i].get("summary", "") for i in batch_idxs ])

    print("Computing ROUGE (may take a moment)...")
    res = rouge.compute(predictions=preds, references=refs)
    print("ROUGE results:", res)

    # Save results
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Save ROUGE scores
    import pandas as pd
    rouge_df = pd.DataFrame([res])
    rouge_df.to_csv(docs_dir / "rouge_scores.csv", index=False)
    print(f"Saved ROUGE scores to {docs_dir / 'rouge_scores.csv'}")

    # Save samples
    with open(docs_dir / "summarizer_samples.txt", "w") as f:
        for i in range(min(10, len(preds))):
            f.write(f"SAMPLE {i+1}\n")
            f.write(f"PRED: {preds[i]}\n")
            f.write(f"REF : {refs[i]}\n")
            f.write("-" * 40 + "\n")
    print(f"Saved samples to {docs_dir / 'summarizer_samples.txt'}")

if __name__ == "__main__":
    main()
