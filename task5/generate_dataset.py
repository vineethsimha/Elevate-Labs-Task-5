"""
generate_dataset.py
--------------------
This sandbox has no internet access, so the real UCI Heart Disease
dataset could not be downloaded directly. This script generates a
HEART-DISEASE-STYLE dataset with the same well-known columns and
realistic value ranges/distributions as the UCI Cleveland Heart Disease
dataset, with the target driven by a genuine (noisy) combination of the
risk-factor features, so tree-based models find real, meaningful signal.

If you have internet access, swap in the real dataset from:
https://archive.ics.uci.edu/dataset/45/heart+disease
(or the cleaned Kaggle mirror), keeping the same column names, and
decision_trees_random_forests.py needs no changes.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 800

age = np.random.normal(54, 9, N).clip(29, 77).round().astype(int)
sex = np.random.choice([0, 1], size=N, p=[0.32, 0.68])  # 1 = male, 0 = female
cp = np.random.choice([0, 1, 2, 3], size=N, p=[0.47, 0.17, 0.28, 0.08])  # chest pain type
trestbps = np.random.normal(131, 17, N).clip(94, 200).round().astype(int)  # resting BP
chol = np.random.normal(246, 51, N).clip(126, 564).round().astype(int)  # cholesterol
fbs = np.random.choice([0, 1], size=N, p=[0.85, 0.15])  # fasting blood sugar > 120
restecg = np.random.choice([0, 1, 2], size=N, p=[0.48, 0.5, 0.02])
thalach = np.random.normal(150, 23, N).clip(71, 202).round().astype(int)  # max heart rate
exang = np.random.choice([0, 1], size=N, p=[0.67, 0.33])  # exercise induced angina
oldpeak = np.random.exponential(1.0, N).clip(0, 6.2).round(1)  # ST depression
slope = np.random.choice([0, 1, 2], size=N, p=[0.14, 0.46, 0.40])
ca = np.random.choice([0, 1, 2, 3, 4], size=N, p=[0.58, 0.22, 0.12, 0.06, 0.02])  # major vessels
thal = np.random.choice([1, 2, 3], size=N, p=[0.06, 0.55, 0.39])  # 1=normal,2=fixed,3=reversible

# Build target risk score from known real-world risk factors + noise, then threshold.
# Weights are scaled up relative to noise so the tree-based models have real,
# learnable signal to find (mirroring how these risk factors behave in practice).
risk_score = (
    0.06 * (age - 54)
    + 1.8 * sex
    + 1.0 * (cp == 0)  # typical angina type coded 0 here associated with higher risk in this synthetic setup
    + 0.035 * (trestbps - 131)
    + 0.018 * (chol - 246)
    + 0.6 * fbs
    + 0.035 * (150 - thalach)
    + 1.8 * exang
    + 0.9 * oldpeak
    + 0.8 * ca
    + 0.9 * (thal == 3)
    + np.random.normal(0, 1.5, N)
)
target = (risk_score > np.median(risk_score)).astype(int)  # 1 = heart disease present

df = pd.DataFrame({
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal,
    "target": target,
})

df.to_csv("data/heart.csv", index=False)
print("Saved data/heart.csv with shape:", df.shape)
print("Target balance:\n", df["target"].value_counts())
