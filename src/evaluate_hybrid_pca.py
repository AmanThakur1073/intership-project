
# src/evaluate_hybrid_pca.py
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Ensure working dir is project root when you run script
LABEL_PATH = "data/labels_final.csv"
PCA_PATH = "models/pca.pkl"
MODEL_PATH = "models/hybrid_pca_regressor.pkl"

def load_pca(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_models(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    print("[INFO] Loading labels...")
    df = pd.read_csv(LABEL_PATH)
    print(f"[INFO] {len(df)} rows in labels_final.csv")

    print("[INFO] Loading PCA and models...")
    pca = load_pca(PCA_PATH)
    models = load_models(MODEL_PATH)  # list of 5 models

    # load precomputed feature matrix? we'll recompute quickly via hybrid extractor if present
    try:
        from src.hybrid_features import HybridFeatureExtractor
    except Exception as e:
        print("[ERROR] Could not import HybridFeatureExtractor:", e)
        print("-> Run this file from project root and ensure src/ is a package or PYTHONPATH includes project root.")
        return

    extractor = HybridFeatureExtractor()

    X = []
    y = []
    valid_idx = []
    for i, row in df.iterrows():
        img = row["image_path"]
        if not os.path.exists(img):
            # skip missing
            continue
        feats = extractor.extract(img)
        X.append(feats)
        y.append([row["Dry_Clover_g"], row["Dry_Dead_g"], row["Dry_Green_g"], row["Dry_Total_g"], row["GDM_g"]])
        valid_idx.append(i)
        print(f"[DATA] {len(X)}/{len(df)} extracted", end="\r")
    print()

    X = np.array(X)
    y = np.array(y)

    # transform with PCA
    X_pca = pca.transform(X)

    # predict
    preds = np.column_stack([m.predict(X_pca) for m in models])

    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

    print("\n======= EVALUATION =======")
    print("MSE (all targets):", mse)
    print("R2  (all targets):", r2)
    # per-target
    for i, name in enumerate(["Dry_Clover_g","Dry_Dead_g","Dry_Green_g","Dry_Total_g","GDM_g"]):
        m = mean_squared_error(y[:,i], preds[:,i])
        r = r2_score(y[:,i], preds[:,i])
        print(f"  {name:12s} -> MSE: {m:.4f}   R2: {r:.4f}")

if __name__ == "__main__":
    main()
