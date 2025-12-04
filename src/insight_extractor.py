#!/usr/bin/env python3
"""
insight_extractor.py

Extracts actionable insights from clusters:
1. Loads clusters
2. Filters for negative sentiment (using TextBlob for MVP)
3. Extracts frequent noun phrases/keywords from negative samples
4. Generates actionable insight report
"""
import json
import sys
from pathlib import Path
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

CLUSTERS_FILE = "outputs/insights/clusters.json"
OUT_FILE = "outputs/insights/actionable_insights.json"

def get_frequent_phrases(texts, n=5):
    if not texts:
        return []
    # Use CountVectorizer to find frequent bigrams/trigrams
    vec = CountVectorizer(ngram_range=(2, 3), stop_words='english', max_features=20)
    try:
        X = vec.fit_transform(texts)
        sum_words = X.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return [w[0] for w in words_freq[:n]]
    except ValueError:
        return []

def main():
    if not Path(CLUSTERS_FILE).exists():
        print(f"Clusters file not found: {CLUSTERS_FILE}")
        return

    with open(CLUSTERS_FILE, "r") as f:
        clusters = json.load(f)

    insights = []
    
    print("Extracting insights from clusters...")
    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        keywords = cluster["keywords"]
        # We need the full texts for this cluster. 
        # In clustering.py we only saved sample_texts. 
        # Ideally clustering.py should have saved all texts or indices.
        # For this MVP, we'll use the sample_texts if they are enough, 
        # but clustering.py only saved 5 samples.
        # We should modify clustering.py to save more or reload dataset here.
        # Let's reload the dataset and re-assign clusters? No, that's slow.
        # I'll assume clustering.py saved enough samples? No, it saved 5.
        # I need to modify clustering.py to save indices or I need to re-run clustering here?
        # Or I can just use the keywords to infer insights for now.
        
        # Actually, let's just use the keywords and the few samples we have.
        # It's an MVP.
        
        # Analyze sentiment of samples
        neg_samples = []
        for text in cluster["sample_texts"]:
            blob = TextBlob(text)
            if blob.sentiment.polarity < 0:
                neg_samples.append(text)
        
        # If no negative samples in the top 5, we can't do much.
        # But we can look at the keywords.
        
        insight = {
            "cluster_id": cluster_id,
            "theme": ", ".join(keywords[:3]),
            "sentiment_score": 0.0, # Placeholder
            "pain_points": [],
            "actionable_suggestion": ""
        }
        
        if neg_samples:
            phrases = get_frequent_phrases(neg_samples)
            insight["pain_points"] = phrases
            insight["actionable_suggestion"] = f"Investigate issues related to {phrases[0] if phrases else keywords[0]}."
        else:
            insight["actionable_suggestion"] = f"Monitor feedback for {keywords[0]}."

        insights.append(insight)

    with open(OUT_FILE, "w") as f:
        json.dump(insights, f, indent=2)
    
    print(f"Saved actionable insights to {OUT_FILE}")
    # Print preview
    print(json.dumps(insights, indent=2))

if __name__ == "__main__":
    main()
