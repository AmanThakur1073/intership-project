import os
import pickle
import pandas as pd
import numpy as np

from hybrid_features import HybridFeatureExtractor
from metadata_features import MetadataProcessor

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor


DATA_PATH = r"C:/Users/amant/Desktop/csiro_segmentation/data/final_metadata_clean_FIXED.csv"
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

print("[INFO] Loading metadata...")
df = pd.read_csv(DATA_PATH)

TARGETS = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g"
]

# ----------------------
# INIT extractors
# ----------------------
hybrid = HybridFeatureExtractor()
meta_processor = MetadataProcessor()
meta_processor.fit(df)

X = []
Y = {t: [] for t in TARGETS}

print("[INFO] Extracting features + metadata...")

for _, row in df.iterrows():

    img_path = row["image_path"]

    feat = hybrid.extract(img_path)
    feat = np.nan_to_num(feat)

    meta_vec = meta_processor.transform(row.to_frame().T)[0]

    full = np.hstack([feat, meta_vec])
    X.append(full)

    for t in TARGETS:
        Y[t].append(row[t])

X = np.array(X)
print("[INFO] Scaling full features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

# ----------------------
# TRAIN 5 MODELS
# ----------------------
models = {}

print("\n[INFO] Training 5 LightGBM models...\n")

for t in TARGETS:
    print(f"[TRAIN] {t} ...")
    m = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=40
    )
    m.fit(X_pca, np.array(Y[t]))
    models[t] = m

# Save all
pickle.dump(pca, open(f"{MODELS_DIR}/pca_final.pkl", "wb"))
pickle.dump(scaler, open(f"{MODELS_DIR}/scaler_final.pkl", "wb"))
pickle.dump(models, open(f"{MODELS_DIR}/regressor_final.pkl", "wb"))
pickle.dump(meta_processor, open(f"{MODELS_DIR}/metadata_processor.pkl", "wb"))

print("======================================")
print("  🎉 FINAL 5-MODEL TRAINED SUCCESSFULLY!")
print("======================================")
