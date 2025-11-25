import sys, os
sys.path.append(os.path.dirname(__file__))


import os
import pandas as pd
import numpy as np
import joblib
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from feature_extractor import extract_features_from_image
from sam_utils import load_sam_model

def build_feature_matrix(df, predictor):
    X = []
    y = []
    for idx, row in df.iterrows():
        img_path = row["image_path"]
        print(f"[DATA] Extracting features from {img_path} ({idx+1}/{len(df)})")
        feats = extract_features_from_image(predictor, img_path)
        X.append(feats)
        y.append([
            row["Dry_Clover_g"],
            row["Dry_Dead_g"],
            row["Dry_Green_g"],
            row["Dry_Total_g"],
            row["GDM_g"]
        ])
    return np.vstack(X), np.array(y, dtype=np.float32)

def train(csv_path="data/labels_final.csv",
          sam_checkpoint="checkpoints/sam_vit_b.pth",
          save_model_path="models/regressor.pkl"):

    os.makedirs("models", exist_ok=True)

    print("[INFO] Loading SAM for feature extraction...")
    predictor = load_sam_model(sam_checkpoint)

    print("[INFO] Loading labels...")
    df = pd.read_csv(csv_path)

    print("[INFO] Extracting SAM features...")
    X, y = build_feature_matrix(df, predictor)

    print("[INFO] Splitting train/val...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    print("[INFO] Training LightGBM regressor...")
    base = LGBMRegressor(n_estimators=400, learning_rate=0.05)
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred, multioutput='variance_weighted')

    print(f"\n[RESULT] Val MSE: {mse:.4f}")
    print(f"[RESULT] Val R²:  {r2:.4f}")

    joblib.dump(model, save_model_path)
    print(f"[SAVED] Model saved → {save_model_path}")

if __name__ == "__main__":
    train()
