import os
import pickle
import numpy as np
import pandas as pd

from hybrid_features import HybridFeatureExtractor
from metadata_features import MetadataProcessor

# -------------------------------
# LOAD MODELS
# -------------------------------
MODELS_DIR = "models"

scaler = pickle.load(open(f"{MODELS_DIR}/scaler_final.pkl", "rb"))
models = pickle.load(open(f"{MODELS_DIR}/regressor_final.pkl", "rb"))
meta_processor = pickle.load(open(f"{MODELS_DIR}/metadata_processor.pkl", "rb"))

extractor = HybridFeatureExtractor()

TARGETS = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g"
]

# Load metadata CSV
meta_df = pd.read_csv("data/final_metadata_clean_FIXED.csv")

def get_meta(image_name):
    row = meta_df[meta_df["image_path"].str.contains(image_name)]
    if len(row) == 0:
        return None
    return row.iloc[0]

# -------------------------------
# GENERATE SUBMISSION
# -------------------------------
rows = []

test_folder = "data/test"
test_images = [x for x in os.listdir(test_folder) if x.endswith(".jpg")]

print(f"[INFO] Found {len(test_images)} test images.")

for img in test_images:

    img_path = os.path.join(test_folder, img)
    img_name = img.replace(".jpg", "")

    print(f"[PREDICT] {img}")

    # IMAGE FEATURES
    feat = extractor.extract(img_path)
    feat = np.nan_to_num(feat)

    # METADATA
    row = get_meta(img)
    if row is None:
        continue

    meta = {
        "Pre_GSHH_NDVI": row["Pre_GSHH_NDVI"],
        "Height_Ave_cm": row["Height_Ave_cm"],
        "State": row["State"],
        "Species": row["Species"],
        "year": row["year"],
        "month": row["month"],
        "day_of_year": row["day_of_year"],
        "season": row["season"]
    }

    meta_df_row = pd.DataFrame([meta])
    meta_vec = meta_processor.transform(meta_df_row)[0]

    full = np.hstack([feat, meta_vec])
    full_scaled = scaler.transform([full])[0]

    # 5 predictions
    pred = {}
    for t in TARGETS:
        val = float(models[t].predict([full_scaled])[0])
        pred[t] = max(val, 0)  # No negative biomass

    # ADD 5 rows for Kaggle
    rows.append([f"{img_name}__Dry_Clover_g", pred["Dry_Clover_g"]])
    rows.append([f"{img_name}__Dry_Dead_g", pred["Dry_Dead_g"]])
    rows.append([f"{img_name}__Dry_Green_g", pred["Dry_Green_g"]])
    rows.append([f"{img_name}__Dry_Total_g", pred["Dry_Total_g"]])
    rows.append([f"{img_name}__GDM_g", pred["GDM_g"]])

# SAVE SUBMISSION
subm = pd.DataFrame(rows, columns=["sample_id", "target"])
subm.to_csv("submission.csv", index=False)

print("\n====================================")
print("🎉 submission.csv generated SUCCESSFULLY!")
print("====================================")
