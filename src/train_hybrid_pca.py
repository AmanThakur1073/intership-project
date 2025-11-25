import os
import pickle
import numpy as np
import pandas as pd

from hybrid_features import HybridFeatureExtractor
from pca_reducer import PCAFeatureReducer
from lightgbm import LGBMRegressor

LABEL_PATH = "data/labels_final.csv"

def train_hybrid_pca():
    print("[INFO] Loading labels...")
    df = pd.read_csv(LABEL_PATH)

    extractor = HybridFeatureExtractor()

    X = []
    y = []

    print("[INFO] Extracting Hybrid Features + Building Matrix...")

    for i, row in df.iterrows():
        img = row["image_path"]

        if not os.path.exists(img):
            print("[WARN] File not found:", img)
            continue

        feats = extractor.extract(img)
        X.append(feats)

        y.append([
            row["Dry_Clover_g"],
            row["Dry_Dead_g"],
            row["Dry_Green_g"],
            row["Dry_Total_g"],
            row["GDM_g"],
        ])

        print(f"[DATA] {i+1}/{len(df)} extracted")

    X = np.array(X)
    y = np.array(y)

    # Step 1: PCA  ← FIXED HERE
    pca = PCAFeatureReducer(output_dim=30)   # was 256
    pca.fit(X)
    X_pca = pca.transform(X)
    pca.save("models/pca.pkl")

    # Step 2: Train 5 regressors
    final_models = []

    print("[INFO] Training 5 models (one per label)...")
    for i in range(5):
        print(f"  → Training model for target #{i+1}")
        m = LGBMRegressor(
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=64
        )
        m.fit(X_pca, y[:, i])
        final_models.append(m)

    with open("models/hybrid_pca_regressor.pkl", "wb") as f:
        pickle.dump(final_models, f)

    print("\n[SUCCESS] Hybrid PCA Model saved at models/hybrid_pca_regressor.pkl")


if __name__ == "__main__":
    train_hybrid_pca()
