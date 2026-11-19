"""
Weighted RFM scoring + KMeans clustering.

Weightage model (justified in README.md):
  Weighted Score = 0.30 * R_score + 0.35 * F_score + 0.35 * M_score
  where each component is a 0-100 percentile score.
  Recency is inverted (fewer days since last purchase = higher score).
  Frequency and Monetary get the larger weights because they are the
  strongest predictors of future value; Recency still matters as a
  churn signal, so it keeps 0.30.

Clustering features: Recency, Frequency, log1p(Monetary), standardised
with StandardScaler. K is chosen by silhouette score over k=2..8
(elbow / inertia is reported alongside). Agglomerative clustering
with the same k is run as a comparison.

Run: python src/segment.py
Outputs: data/customers_segmented.csv, metrics.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
W_R, W_F, W_M = 0.30, 0.35, 0.35


def percentile_score(series: pd.Series) -> pd.Series:
    """0-100 percentile score (rank-based, robust to skew)."""
    return series.rank(pct=True) * 100


def main():
    df = pd.read_csv(BASE / "data" / "customers.csv")

    # --- Weighted RFM score, 0-100 ---
    r_score = 100 - percentile_score(df["recency_days"])
    f_score = percentile_score(df["frequency"])
    m_score = percentile_score(df["monetary"])
    df["r_score"] = r_score.round(2)
    df["f_score"] = f_score.round(2)
    df["m_score"] = m_score.round(2)
    df["weightage_score"] = (W_R * r_score + W_F * f_score + W_M * m_score).round(2)

    # --- Clustering features ---
    features = pd.DataFrame({
        "recency_days": df["recency_days"].astype(float),
        "frequency": df["frequency"].astype(float),
        "log_monetary": np.log1p(df["monetary"]),
    })
    X = StandardScaler().fit_transform(features)

    # --- Elbow + silhouette for k=2..8 ---
    elbow = {}
    silhouette = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        elbow[k] = float(km.inertia_)
        silhouette[k] = float(silhouette_score(X, labels))

    best_k = max(silhouette, key=silhouette.get)
    print("k | inertia | silhouette")
    for k in range(2, 9):
        print(f"{k} | {elbow[k]:.1f} | {silhouette[k]:.4f}")
    print(f"Best k by silhouette: {best_k} (silhouette {silhouette[best_k]:.4f})")

    # --- Final KMeans at best k ---
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    df["cluster"] = km.fit_predict(X)
    final_silhouette = float(silhouette_score(X, df["cluster"]))
    final_inertia = float(km.inertia_)

    # --- Comparison: Agglomerative at same k ---
    agg = AgglomerativeClustering(n_clusters=best_k)
    agg_labels = agg.fit_predict(X)
    agg_silhouette = float(silhouette_score(X, agg_labels))
    print(f"Agglomerative (k={best_k}) silhouette: {agg_silhouette:.4f} "
          f"vs KMeans {final_silhouette:.4f}")

    # --- Cluster profiles ---
    profiles = []
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        profiles.append({
            "cluster": int(c),
            "size": int(len(sub)),
            "share_pct": round(100 * len(sub) / len(df), 2),
            "mean_recency_days": round(float(sub["recency_days"].mean()), 2),
            "mean_frequency": round(float(sub["frequency"].mean()), 2),
            "mean_monetary": round(float(sub["monetary"].mean()), 2),
            "mean_weightage_score": round(float(sub["weightage_score"].mean()), 2),
            "mean_age": round(float(sub["age"].mean()), 2),
            "mean_tenure_months": round(float(sub["tenure_months"].mean()), 2),
            "top_category": str(sub["category_preference"].mode().iloc[0]),
        })
    for p in profiles:
        print(p)

    df.to_csv(BASE / "data" / "customers_segmented.csv", index=False)

    metrics = {
        "n_customers": int(len(df)),
        "features_used": ["recency_days", "frequency", "log1p(monetary)"],
        "scaler": "StandardScaler",
        "weightage_weights": {"recency": W_R, "frequency": W_F, "monetary": W_M},
        "weightage_formula": "0.30*R_score + 0.35*F_score + 0.35*M_score (each 0-100 percentile, recency inverted)",
        "k_range": [2, 3, 4, 5, 6, 7, 8],
        "elbow_inertia": {str(k): v for k, v in elbow.items()},
        "silhouette_by_k": {str(k): round(v, 4) for k, v in silhouette.items()},
        "best_k": int(best_k),
        "best_k_chosen_by": "highest silhouette score",
        "final_silhouette": round(final_silhouette, 4),
        "final_inertia": round(final_inertia, 2),
        "comparison_agglomerative_k": int(best_k),
        "comparison_agglomerative_silhouette": round(agg_silhouette, 4),
        "cluster_sizes": {str(p["cluster"]): p["size"] for p in profiles},
        "cluster_profiles": profiles,
    }
    with open(BASE / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {BASE / 'data' / 'customers_segmented.csv'} and {BASE / 'metrics.json'}")


if __name__ == "__main__":
    main()
