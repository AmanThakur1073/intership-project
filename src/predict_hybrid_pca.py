import pickle
import numpy as np
import cv2
import sys
import os

sys.path.append("src")
from hybrid_features import HybridFeatureExtractor
from llm_analyzer import analyze_with_llm   # <-- LLM IMPORT


# ---------------------------
# FIXED PATHS FOR MODELS
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # go out of src/

PCA_PATH = os.path.join(BASE_DIR, "models", "pca.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "hybrid_pca_regressor.pkl")


def load_models():
    print("[INFO] Loading PCA + Regressors...")

    with open(PCA_PATH, "rb") as f:
        pca = pickle.load(f)

    with open(MODEL_PATH, "rb") as f:
        models = pickle.load(f)

    print("[SUCCESS] All models loaded!")
    return pca, models


def predict(image_path):
    if not os.path.exists(image_path):
        print("[ERROR] Image not found:", image_path)
        return None

    print("[INFO] Extracting hybrid features...")
    extractor = HybridFeatureExtractor()
    features = extractor.extract(image_path)
    features = np.nan_to_num(features)

    pca, models = load_models()
    x_pca = pca.transform([features])[0]

    preds = [m.predict([x_pca])[0] for m in models]
    labels = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]

    print("\n======= FINAL PREDICTION =======")
    for label, val in zip(labels, preds):
        print(f"{label:15} : {val:.3f}")
    print("================================\n")

    result = {label: float(val) for label, val in zip(labels, preds)}
    return result


if __name__ == "__main__":
    img = input("Enter image path: ")
    preds = predict(img)
    print("\nReturned dict:", preds)

    # -----------------------------
    #   LLM EXPERT ANALYSIS
    # -----------------------------
    print("\n========= LLM ANALYSIS =========")
    analysis = analyze_with_llm(preds)
    print(analysis)
    print("================================\n")
