#!/usr/bin/env python3
"""
clustering.py

Performs topic clustering on feedback data:
1. Generates sentence embeddings
2. Clusters using KMeans
3. Extracts keywords per cluster
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

try:
    from datasets import load_from_disk
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print("Missing dependencies. Run: pip install sentence-transformers datasets scikit-learn matplotlib", file=sys.stderr)
    raise

DATA_DIR = "data_processed/amazon_bart_tokenized_with_targets"
OUT_DIR = "outputs/insights"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def extract_keywords(texts, top_n=10):
    """Extracts top keywords using TF-IDF."""
    if not texts:
        return []
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    try:
        tfidf_matrix = tfidf.fit_transform(texts)
        feature_names = tfidf.get_feature_names_out()
        # Sum tfidf scores for each term
        sums = tfidf_matrix.sum(axis=0)
        data = []
        for col, term in enumerate(feature_names):
            data.append((term, sums[0, col]))
        ranking = sorted(data, key=lambda x: x[1], reverse=True)
        return [term for term, score in ranking[:top_n]]
    except ValueError:
        return []

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset from {DATA_DIR}...")
    ds = load_from_disk(DATA_DIR)
    # Use a subset for speed if needed, but let's try full train set
    texts = ds["train"]["content"][:5000] # Limit to 5k for speed in this demo
    print(f"Loaded {len(texts)} samples.")
    
    print(f"Loading embedding model {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    np.save(f"{OUT_DIR}/embeddings.npy", embeddings)
    
    # Clustering
    print("Clustering (KMeans)...")
    # Simple heuristic for K: 5 clusters
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Analyze clusters
    cluster_data = []
    print("\n=== Cluster Analysis ===")
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        cluster_texts = [texts[idx] for idx in cluster_indices]
        keywords = extract_keywords(cluster_texts)
        
        print(f"\nCluster {i} (Size: {len(cluster_texts)}):")
        print(f"Keywords: {', '.join(keywords)}")
        print(f"Sample: {cluster_texts[0][:100]}...")
        
        cluster_data.append({
            "cluster_id": i,
            "size": len(cluster_texts),
            "keywords": keywords,
            "sample_texts": cluster_texts[:5]
        })
        
    # Save results
    with open(f"{OUT_DIR}/clusters.json", "w") as f:
        json.dump(cluster_data, f, indent=2)
    print(f"\nSaved cluster analysis to {OUT_DIR}/clusters.json")
    
    # Visualization (PCA/UMAP would be better, but let's do simple PCA plot if possible)
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title("Feedback Clusters (PCA)")
        plt.savefig(f"{OUT_DIR}/clusters_pca.png")
        print(f"Saved cluster plot to {OUT_DIR}/clusters_pca.png")
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == "__main__":
    main()
