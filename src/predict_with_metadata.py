import pandas as pd
import numpy as np
import pickle
import os

from hybrid_features import HybridFeatureExtractor
from metadata_features import MetadataProcessor

# -----------------------------
# Load metadata CSV
# -----------------------------
META_CSV = "../data/final_metadata_clean.csv"
df_meta = pd.read_csv(META_CSV)

# -----------------------------
# Load Models
# -----------------------------
pca = pickle.load(open("../models/pca_with_metadata.pkl", "rb"))
models = pickle.load(open("../models/lgbm_with_metadata.pkl", "rb"))
meta_processor = pickle.load(open("../models/metadata_processor.pkl", "rb"))

extractor = HybridFeatureExtractor()

# Target labels
TARGETS = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]


def auto_predict(image_path):

    # Extract Image ID
    image_name = os.path.basename(image_path)

    # Find metadata row
    row = df_meta[df_meta["image_path"].str.contains(image_name)]
    if row.empty:
        print(f"❌ ERROR: Metadata not found for {image_name}")
        return

    row = row.iloc[0]   # convert to 1 row df

    # -------- Metadata processing --------
    meta_df = pd.DataFrame([row])
    meta_features = meta_processor.transform(meta_df)

    # -------- Image feature extraction --------
    img_feat = extractor.extract(image_path)
    img_feat = np.nan_to_num(img_feat)

    # -------- Merge --------
    full_feat = np.hstack([img_feat, meta_features[0]])

    # -------- PCA --------
    x_pca = pca.transform([full_feat])

    # -------- Prediction --------
    preds = [m.predict(x_pca)[0] for m in models]

    # -------- OUTPUT --------
    print("\n====== AUTO METADATA DETECTED ======")
    for col in ["Pre_GSHH_NDVI", "Height_Ave_cm", "State", "Species",
                "year", "month", "day_of_year", "season"]:
        print(f"{col}: {row[col]}")

    print("\n====== FINAL BIOMASS PREDICTION ======")
    for label, val in zip(TARGETS, preds):
        print(f"{label:15}: {val:.3f}")

    return dict(zip(TARGETS, preds))


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    img = input("Enter image path: ")
    auto_predict(img)
