import os
import pickle
import pandas as pd
import numpy as np

from hybrid_features import HybridFeatureExtractor
from metadata_features import MetadataProcessor

from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

# ============================================================
# LOAD PREPARED METADATA CSV  (ONE ROW PER IMAGE)
# ============================================================

DATA_PATH = "data/final_metadata_clean_FIXED.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

print("[INFO] Loading metadata...")
df = pd.read_csv(DATA_PATH)

# ============================================================
# CORRECT COLUMNS FOR MODEL INPUTS
# ============================================================

TARGETS = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g"
]

# Metadata columns (8 features)
METADATA_COLS = [
    "Pre_GSHH_NDVI",
    "Height_Ave_cm",
    "State",
    "Species",
    "year",
    "month",
    "day_of_year",
    "season"
]

# ============================================================
# INITIALIZE PROCESSORS
# ============================================================

hybrid = HybridFeatureExtractor()

meta_processor = MetadataProcessor()
meta_processor.fit(df)

X = []
Y = {t: [] for t in TARGETS}

print("[INFO] Extracting image features + metadata...")

# ============================================================
# LOOP THROUGH ALL IMAGES
# ============================================================

for _, row in df.iterrows():

    # Fix image path: remove train/ from CSV
    img_path = row["image_path"].replace("train/", "data/train/")
    
    # ---------------- IMAGE FEATURES ----------------
    feat = hybrid.extract(img_path)
    feat = np.nan_to_num(feat)

    # ---------------- METADATA ----------------
    meta_vec = meta_processor.transform(row.to_frame().T)[0]

    # Combine hybrid image features + metadata
    full = np.hstack([feat, meta_vec])
    X.append(full)

    # Targets
    for t in TARGETS:
        Y[t].append(row[t])

X = np.array(X)

# ============================================================
# SCALER (No PCA — because PCA destroys accuracy on small data)
# ============================================================

print("[INFO] Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# TRAIN 5 MODELS (LightGBM tuned)
# ============================================================

models = {}
print("\n[INFO] Training 5 LightGBM models...\n")

for t in TARGETS:
    print(f"[TRAIN] {t} ...")
    m = LGBMRegressor(
        n_estimators=1600,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=25,
        feature_fraction=0.4,
        bagging_fraction=0.6
    )
    m.fit(X_scaled, np.array(Y[t]))
    models[t] = m

# ============================================================
# SAVE MODELS
# ============================================================

pickle.dump(scaler, open(f"{MODELS_DIR}/scaler_final.pkl", "wb"))
pickle.dump(models, open(f"{MODELS_DIR}/regressor_final.pkl", "wb"))
pickle.dump(meta_processor, open(f"{MODELS_DIR}/metadata_processor.pkl", "wb"))

print("======================================")
print("  🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("======================================")
