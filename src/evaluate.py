"""
Evaluate and chart the segmentation run.

Reads metrics.json and data/customers_segmented.csv, prints a summary,
and saves charts to screenshots/:
  - elbow.png (inertia vs k)
  - silhouette.png (silhouette vs k)
  - clusters_pca.png (PCA 2D scatter, coloured by cluster)
  - weightage_distribution.png (histogram of the weighted score)

Run: python src/evaluate.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]


def main():
    metrics = json.loads((BASE / "metrics.json").read_text())
    df = pd.read_csv(BASE / "data" / "customers_segmented.csv")
    shots = BASE / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)

    ks = [int(k) for k in metrics["silhouette_by_k"]]
    inertia = [metrics["elbow_inertia"][str(k)] for k in ks]
    sil = [metrics["silhouette_by_k"][str(k)] for k in ks]
    best_k = metrics["best_k"]

    print(f"Customers: {metrics['n_customers']}")
    print(f"Best k: {best_k}  Silhouette: {metrics['final_silhouette']}  Inertia: {metrics['final_inertia']}")
    print(f"Agglomerative silhouette (k={best_k}): {metrics['comparison_agglomerative_silhouette']}")
    print("\nCluster profiles:")
    for p in metrics["cluster_profiles"]:
        print(f"  Cluster {p['cluster']}: n={p['size']} ({p['share_pct']}%), "
              f"R={p['mean_recency_days']}, F={p['mean_frequency']}, "
              f"M={p['mean_monetary']}, score={p['mean_weightage_score']}, top={p['top_category']}")

    # Elbow
    plt.figure(figsize=(7, 4))
    plt.plot(ks, inertia, marker="o")
    plt.axvline(best_k, color="grey", linestyle="--")
    plt.xlabel("k (number of clusters)")
    plt.ylabel("Inertia")
    plt.title("Elbow method — inertia vs k")
    plt.tight_layout()
    plt.savefig(shots / "elbow.png", dpi=150)
    plt.close()

    # Silhouette
    plt.figure(figsize=(7, 4))
    plt.plot(ks, sil, marker="o")
    plt.axvline(best_k, color="grey", linestyle="--")
    plt.xlabel("k (number of clusters)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette score vs k")
    plt.tight_layout()
    plt.savefig(shots / "silhouette.png", dpi=150)
    plt.close()

    # PCA 2D scatter
    features = pd.DataFrame({
        "recency_days": df["recency_days"].astype(float),
        "frequency": df["frequency"].astype(float),
        "log_monetary": np.log1p(df["monetary"]),
    })
    X = StandardScaler().fit_transform(features)
    pcs = PCA(n_components=2, random_state=42).fit_transform(X)
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(pcs[:, 0], pcs[:, 1], c=df["cluster"], cmap="tab10", s=18, alpha=0.8)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"Customer clusters (PCA 2D, k={best_k})")
    plt.tight_layout()
    plt.savefig(shots / "clusters_pca.png", dpi=150)
    plt.close()

    # Weightage distribution
    plt.figure(figsize=(7, 4))
    plt.hist(df["weightage_score"], bins=25)
    plt.xlabel("Weighted RFM score (0-100)")
    plt.ylabel("Customers")
    plt.title("Distribution of weighted RFM score")
    plt.tight_layout()
    plt.savefig(shots / "weightage_distribution.png", dpi=150)
    plt.close()

    print(f"\nCharts saved to {shots}/")


if __name__ == "__main__":
    main()
