"""
Generate a synthetic customer dataset (seeded, reproducible).

1200 customers with Recency, Frequency, Monetary plus age, tenure and
category preference. The data is synthetic by design: it is generated
from four latent personas so that clustering has real structure to find,
but no real customer is represented.

Personas (probabilities):
  - Champions (15%): bought very recently, often, spend a lot
  - Loyal regulars (30%): recent, frequent, moderate spend
  - At-risk / slipping (30%): not recent, moderate frequency/spend in the past
  - Dormant / new-low (25%): not recent, infrequent, low spend

Run: python src/generate_data.py
Output: data/customers.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N = 1200

rng = np.random.default_rng(SEED)

personas = ["Champions", "Loyal", "At-risk", "Dormant"]
probs = [0.15, 0.30, 0.30, 0.25]
persona = rng.choice(personas, size=N, p=probs)

# Persona parameters: (recency mean, frequency mean, monetary mean)
params = {
    "Champions": (12, 28, 1450),
    "Loyal": (28, 16, 720),
    "At-risk": (95, 9, 480),
    "Dormant": (165, 3, 160),
}

recency, frequency, monetary = [], [], []
for p in persona:
    r_mean, f_mean, m_mean = params[p]
    recency.append(int(np.clip(rng.normal(r_mean, r_mean * 0.45 + 4), 1, 365)))
    frequency.append(int(np.clip(rng.normal(f_mean, f_mean * 0.4 + 1), 1, 60)))
    monetary.append(float(np.clip(rng.normal(m_mean, m_mean * 0.5 + 20), 20, 6000)))

age = np.clip(rng.normal(38, 12, N), 18, 70).astype(int)
tenure = np.clip(rng.normal(24, 14, N), 1, 60).astype(int)
categories = ["Electronics", "Grocery", "Fashion", "Home", "Beauty"]
category = rng.choice(categories, size=N, p=[0.25, 0.30, 0.20, 0.15, 0.10])

df = pd.DataFrame({
    "customer_id": [f"CUST-{i+1:04d}" for i in range(N)],
    "age": age,
    "tenure_months": tenure,
    "category_preference": category,
    "recency_days": recency,
    "frequency": frequency,
    "monetary": np.round(monetary, 2),
    "_persona": persona,  # kept for validation only, NOT used in clustering
})

out = Path(__file__).resolve().parents[1] / "data" / "customers.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {len(df)} customers to {out}")
print(df[["recency_days", "frequency", "monetary", "age", "tenure_months"]].describe().round(2).to_string())
print("\nPersona counts (validation only):")
print(df["_persona"].value_counts().to_string())
