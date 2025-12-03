import pandas as pd
import numpy as np
import pickle
import os

from hybrid_features import HybridFeatureExtractor
from metadata_features import MetadataProcessor
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor

# -----------------------------
# PATHS
# -----------------------------
DATA_PATH =  "data/final_metadata_clean_FIXED.csv"

MODELS_DIR = "../models"

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

print("[INFO] Loading metadata CSV...")
df = pd.read_csv(DATA_PATH)

# -----------------------------
# FIX WINDOWS PATHS
# -----------------------------
df["image_path"] = df["image_path"].apply(lambda x: x.replace("/", "\\"))

# ❗ DO NOT FILTER using os.path.exists (it was deleting all rows)
# df = df[df["image_path"].apply(os.path.exists)]

print(f"[INFO] Total images in CSV: {len(df)}")

# -----------------------------
# TARGETS
# -----------------------------
targets = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g"
]

# -----------------------------
# STEP 1: Metadata Processor
# -----------------------------
print("[INFO] Fitting metadata processor...")

meta_processor = MetadataProcessor()
meta_processor.fit(df)

meta_processor.save(os.path.join(MODELS_DIR, "metadata_processor.pkl"))

meta_features = meta_processor.transform(df)

print(f"[INFO] Metadata features shape: {meta_features.shape}")

# -----------------------------
# STEP 2: Hybrid Feature Extractor
# -----------------------------
extractor = HybridFeatureExtractor()

all_hybrid_features = []

print("[INFO] Extracting hybrid image features...")

for path in df["image_path"].values:

    try:
        feat = extractor.extract(path)
        feat = np.nan_to_num(feat)
    except Exception as e:
        print(f"[ERROR] Failed reading: {path}  | {e}")
        feat = np.zeros(2048)   # backup vector

    all_hybrid_features.append(feat)

all_hybrid_features = np.array(all_hybrid_features)
print(f"[INFO] Hybrid feature shape: {all_hybrid_features.shape}")

# -----------------------------
# STEP 3: Merge Hybrid + Metadata
# -----------------------------
print("[INFO] Merging hybrid + metadata features...")
full_features = np.hstack([all_hybrid_features, meta_features])

print(f"[INFO] Final merged feature shape: {full_features.shape}")

# -----------------------------
# STEP 4: PCA
# -----------------------------
print("[INFO] Fitting PCA (50 components)...")

pca = PCA(n_components=50)
pca_features = pca.fit_transform(full_features)

with open(os.path.join(MODELS_DIR, "pca_with_metadata.pkl"), "wb") as f:
    pickle.dump(pca, f)

print("[SUCCESS] PCA saved!")

# -----------------------------
# STEP 5: LightGBM Training
# -----------------------------
print("[INFO] Training LightGBM regressors...")

models = []

for target in targets:
    print(f"[INFO] Training for → {target} ...")

    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=-1
    )

    model.fit(pca_features, df[target].values)
    models.append(model)

with open(os.path.join(MODELS_DIR, "lgbm_with_metadata.pkl"), "wb") as f:
    pickle.dump(models, f)

print("\n===================================================")
print("[SUCCESS] ALL MODELS TRAINED AND SAVED! 🔥🔥🔥")
print("===================================================\n")
