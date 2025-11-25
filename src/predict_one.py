import cv2
import joblib
import numpy as np
from feature_extractor import extract_features_from_image
from sam_utils import load_sam_model

def predict(image_path):
    print("[INFO] Loading SAM...")
    predictor = load_sam_model("checkpoints/sam_vit_b.pth")

    print("[INFO] Loading trained model...")
    model = joblib.load("models/regressor.pkl")

    print("[INFO] Extracting features from image...")
    features = extract_features_from_image(predictor, image_path)
    features = features.reshape(1, -1)

    print("[INFO] Predicting...")
    preds = model.predict(features)[0]

    print("\n======= PREDICTION RESULTS =======")
    print("Dry_Clover_g :", preds[0])
    print("Dry_Dead_g   :", preds[1])
    print("Dry_Green_g  :", preds[2])
    print("Dry_Total_g  :", preds[3])
    print("GDM_g        :", preds[4])

if __name__ == "__main__":
    img = input("Enter image path: ")
    predict(img)
