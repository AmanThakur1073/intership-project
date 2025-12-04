import os
import pickle
import numpy as np
import pandas as pd

from hybrid_features import HybridFeatureExtractor
from metadata_features import MetadataProcessor

# -------------------------------
# LOAD MODELS (NO PCA ANYMORE)
# -------------------------------
MODELS_DIR = "models"

scaler = pickle.load(open(os.path.join(MODELS_DIR, "scaler_final.pkl"), "rb"))
models = pickle.load(open(os.path.join(MODELS_DIR, "regressor_final.pkl"), "rb"))
meta_processor = pickle.load(open(os.path.join(MODELS_DIR, "metadata_processor.pkl"), "rb"))

# Hybrid extractor (SAM + ResNet50)
extractor = HybridFeatureExtractor()

TARGETS = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g"
]

# -------------------------------
# AUTO METADATA FROM CSV
# -------------------------------
def auto_metadata(image_path):
    df = pd.read_csv("data/final_metadata_clean_FIXED.csv")

    img_name = os.path.basename(image_path)

    row = df[df["image_path"].str.contains(img_name)]

    if len(row) == 0:
        print("[WARN] Metadata not found, using safe defaults!")
        return {
            "Pre_GSHH_NDVI": 0.5,
            "Height_Ave_cm": 5,
            "State": "Vic",
            "Species": "Clover",
            "year": 2015,
            "month": 8,
            "day_of_year": 200,
            "season": "Autumn"
        }

    row = row.iloc[0]

    return {
        "Pre_GSHH_NDVI": row["Pre_GSHH_NDVI"],
        "Height_Ave_cm": row["Height_Ave_cm"],
        "State": row["State"],
        "Species": row["Species"],
        "year": row["year"],
        "month": row["month"],
        "day_of_year": row["day_of_year"],
        "season": row["season"]
    }

# -------------------------------
# FINAL PREDICT FUNCTION
# -------------------------------
def predict_image(image_path):

    # Fix path
    img_path = image_path.replace("train/", "data/train/")

    print("[INFO] Extracting hybrid features...")
    feat = extractor.extract(img_path)
    feat = np.nan_to_num(feat)

    meta = auto_metadata(image_path)
    meta_df = pd.DataFrame([meta])

    meta_vec = meta_processor.transform(meta_df)[0]

    # Combine features
    full = np.hstack([feat, meta_vec])
    full_scaled = scaler.transform([full])[0]

    preds = {}

    for t in TARGETS:
        val = float(models[t].predict([full_scaled])[0])
        preds[t] = max(val, 0)  # No negative biomass

    return meta, preds

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    img = input("Enter image path: ")

    metadata, pred = predict_image(img)

    print("\n====== FINAL OUTPUT (Auto Metadata) ======")
    for k, v in metadata.items():
        print(f"{k}: {v}")

    print("\n---- Biomass Predictions ----")
    for name, value in pred.items():
        print(f"{name}: {value:.3f}")

    print("======================================\n")


# import pandas as pd
# import os
# from predict_final import predict_image   # your function

# TEST_IMAGES_FOLDER = "data/test/"       # <-- change if needed
# TEST_CSV = "data/test.csv"

# # Load test.csv
# test_df = pd.read_csv(TEST_CSV)

# rows = []

# print("[INFO] Starting prediction on full test set...")

# for img_name in test_df['sample_id']:     # your test.csv column
    
#     image_path = os.path.join(TEST_IMAGES_FOLDER, img_name + ".jpg")

#     # Run your model
#     metadata, pred = predict_image(image_path)

#     # Append in LONG format
#     rows.append([f"{img_name}__Dry_Clover_g", pred["Dry_Clover_g"]])
#     rows.append([f"{img_name}__Dry_Dead_g", pred["Dry_Dead_g"]])
#     rows.append([f"{img_name}__Dry_Green_g", pred["Dry_Green_g"]])
#     rows.append([f"{img_name}__Dry_Total_g", pred["Dry_Total_g"]])
#     rows.append([f"{img_name}__GDM_g", pred["GDM_g"]])

# # Convert to dataframe
# df = pd.DataFrame(rows, columns=["sample_id", "target"])
# df.to_csv("submission.csv", index=False)

# print("[SUCCESS] submission.csv generated successfully!")
