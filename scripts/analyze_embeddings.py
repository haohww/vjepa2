#!/usr/bin/env python3
"""
Analyze V-JEPA2 embeddings:
1. Validate: Classify Success vs. Failure (Logistic Regression)
2. Explore: Cosine Similarity distribution (Regret Metric)
3. Visualize: t-SNE of Delta Vectors (Fail - Success) by category
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "outputs/analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_PATH = PROJECT_ROOT / "outputs/embeddings/all_embeddings.pt"


def load_data():
    print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
    if not EMBEDDINGS_PATH.exists():
        print(f"Error: {EMBEDDINGS_PATH} not found. Run extraction first.")
        sys.exit(1)

    data = torch.load(EMBEDDINGS_PATH)
    all_embeddings = data["embeddings"]
    
    X_success = []
    X_failure = []
    categories = []
    folder_names = []
    
    for category, folders in all_embeddings.items():
        for folder, embs in folders.items():
            succ = embs["success_emb"].numpy()
            fail = embs["failure_emb"].numpy()
            
            X_success.append(succ)
            X_failure.append(fail)
            categories.append(category)
            folder_names.append(folder)
            
    X_success = np.array(X_success)
    X_failure = np.array(X_failure)
    categories = np.array(categories)
    
    print(f"Loaded {len(X_success)} pairs across {len(set(categories))} categories.")
    return X_success, X_failure, categories, folder_names


def validate_classification(X_success, X_failure):
    print("\n=== 1. Validation: Success vs. Failure Classification ===")
    
    # Construct dataset: Success=1, Failure=0
    n_samples = len(X_success)
    X = np.concatenate([X_success, X_failure], axis=0)
    y = np.concatenate([np.ones(n_samples), np.zeros(n_samples)], axis=0)
    
    # Shuffle and split
    # Stratify by y to ensure balanced classes in train/test
    if n_samples < 5:
        print("Not enough samples for train/test split. Training on all data.")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
        )
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train Logistic Regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    try:
        roc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc = 0.0 # Handle case with only one class present
    
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {roc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Failure", "Success"], zero_division=0))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Failure", "Success"], 
                yticklabels=["Failure", "Success"])
    plt.title("Confusion Matrix: Success vs. Failure")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png")
    plt.close()
    print(f"Saved confusion matrix to {PLOTS_DIR}/confusion_matrix.png")


def explore_similarity(X_success, X_failure, categories):
    print("\n=== 2. Exploration: Cosine Similarity (Regret Metric) ===")
    
    # Compute cosine similarity between pairs
    # Sim(A, B) = dot(A, B) / (norm(A) * norm(B))
    
    norms_s = np.linalg.norm(X_success, axis=1)
    norms_f = np.linalg.norm(X_failure, axis=1)
    dot_products = np.sum(X_success * X_failure, axis=1)
    
    similarities = dot_products / (norms_s * norms_f + 1e-8)
    
    print(f"Mean Similarity: {np.mean(similarities):.4f}")
    print(f"Min Similarity:  {np.min(similarities):.4f}")
    print(f"Max Similarity:  {np.max(similarities):.4f}")
    
    # Plot Histogram overall
    plt.figure(figsize=(8, 6))
    sns.histplot(similarities, bins=20, kde=True, color="purple") # Fewer bins for small data
    plt.title("Distribution of Cosine Similarity (Success vs. Failure)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.axvline(np.mean(similarities), color='k', linestyle='--', label=f"Mean: {np.mean(similarities):.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "similarity_hist.png")
    plt.close()
    
    # Plot Histogram per category
    if len(set(categories)) > 1:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(x=similarities, hue=categories, fill=True, common_norm=False, palette="tab10", warn_singular=False)
        plt.title("Cosine Similarity Density by Category")
        plt.xlabel("Cosine Similarity")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "similarity_by_category.png")
        plt.close()
    print(f"Saved similarity plots to {PLOTS_DIR}")


def visualize_delta_tsne(X_success, X_failure, categories):
    print("\n=== 3. Visualization: t-SNE of Delta Vectors ===")
    
    if len(X_success) < 5:
        print("Not enough samples for t-SNE (need >= 5). Skipping.")
        return

    # Compute Delta: Fail - Success
    X_delta = X_failure - X_success
    
    # Normalize
    scaler = StandardScaler()
    X_delta_norm = scaler.fit_transform(X_delta)
    
    # PCA first if dims > 50
    if X_delta_norm.shape[1] > 50:
        n_comp = min(50, len(X_delta_norm))
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X_delta_norm)
    else:
        X_pca = X_delta_norm
    
    # t-SNE
    perp = min(30, len(X_success) - 1)
    print(f"Running t-SNE (perplexity={perp})...")
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_jobs=-1, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_pca)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1], 
        hue=categories, 
        palette="tab10", 
        alpha=0.8,
        s=100
    )
    plt.title("t-SNE of Failure Delta Vectors (Fail - Success)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "tsne_delta.png")
    plt.close()
    print(f"Saved t-SNE plot to {PLOTS_DIR}/tsne_delta.png")


def main():
    X_success, X_failure, categories, _ = load_data()
    
    if len(X_success) < 2:
        print("Not enough data to run full analysis. Need at least 2 pairs.")
        sys.exit(0)
            
    validate_classification(X_success, X_failure)
    explore_similarity(X_success, X_failure, categories)
    visualize_delta_tsne(X_success, X_failure, categories)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
