import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from hybrid_features import HybridFeatureExtractor



def build_feature_matrix(df, extractor):
    X, y = [], []

    for i, row in df.iterrows():
        img = row["image_path"]
        print(f"[DATA] Extracting hybrid features {i+1}/{len(df)} → {img}")

        feats = extractor.extract(img)
        X.append(feats)

        y.append([
            row["Dry_Clover_g"],
            row["Dry_Dead_g"],
            row["Dry_Green_g"],
            row["Dry_Total_g"],
            row["GDM_g"]
        ])

    return np.array(X), np.array(y)


def train(csv_path="data/labels_final.csv", model_path="models/hybrid_regressor.pkl"):

    # Load data
    print("[INFO] Loading CSV...")
    df = pd.read_csv(csv_path)

    # Load hybrid extractor
    print("[INFO] Loading Hybrid Extractor...")
    extractor = HybridFeatureExtractor()

    # Hybrid feature extraction
    print("[INFO] Extracting hybrid features...")
    X, y = build_feature_matrix(df, extractor)

    # Train-val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # Regressor (LightGBM)
    model = MultiOutputRegressor(LGBMRegressor(n_estimators=400, learning_rate=0.05))
    print("[INFO] Training model...")
    model.fit(X_train, y_train)

    # Validation
    pred = model.predict(X_val)

    print("\n[RESULT] MSE:", mean_squared_error(y_val, pred))
    print("[RESULT] R2 :", r2_score(y_val, pred, multioutput='variance_weighted'))

    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)
    print("[SAVED] Hybrid Model saved:", model_path)


if __name__ == "__main__":
    train()
