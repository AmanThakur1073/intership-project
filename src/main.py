import os
import cv2
import numpy as np
import torch
from sam_utils import load_sam_model, get_sam_masks
from classify import classify_mask


# ---------------------------
#  USER CONFIG
# ---------------------------

IMAGE_PATH = "data/test/sample.jpg"          # jo bhi test image dalni ho
SAM_CHECKPOINT = "checkpoints/sam_vit_b.pth" # yaha apna SAM checkpoint rakho
CLASSIFIER_MODEL = None                      # future me classifier model load karne ke liye

OUTPUT_MASK_DIR = "outputs/masks/"
OUTPUT_OVERLAY_DIR = "outputs/overlays/"

CLASSES = ["Green_Grass", "Dry_Grass", "Clover", "Dry_Clover", "Soil"]
COLORS = [
    (0, 255, 0),    # green
    (0, 200, 255),  # orange-ish
    (255, 0, 0),    # blue
    (255, 0, 255),  # pink
    (128, 128, 128) # gray
]


# -------------------------------------------------------
#  Extract simple features from mask (mean color etc.)
# -------------------------------------------------------

def extract_features(image, mask):
    masked_pixels = image[mask == 1]

    if len(masked_pixels) == 0:
        return np.zeros(256)  # safe default if mask empty

    # Color mean features (R,G,B)
    mean_colors = np.mean(masked_pixels, axis=0)

    # Extra dummy padding to make vector size 256
    feature_vector = np.concatenate([mean_colors, np.zeros(253)])

    return feature_vector


# ---------------------------
#  MAIN PIPELINE
# ---------------------------

def process_image():
    print("\n[INFO] Loading image...")
    
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("[ERROR] Image not found!")
        return

    original = image.copy()

    # Load SAM
    predictor = load_sam_model(SAM_CHECKPOINT)

    # Generate masks
    masks = get_sam_masks(predictor, image)

    # Mask counter
    mask_id = 0

    for mask in masks:
        print(f"\n[INFO] Processing mask {mask_id}")

        # Convert SAM output to binary mask
        binary_mask = mask.astype(np.uint8)

        # Extract features
        features = extract_features(image, binary_mask)

        # Predict class
        predicted_class = classify_mask(features, CLASSIFIER_MODEL)

        class_name = CLASSES[predicted_class]
        print(f"[INFO] Mask class: {class_name}")

        # Save mask image
        mask_path = os.path.join(OUTPUT_MASK_DIR, f"mask_{mask_id}.png")
        cv2.imwrite(mask_path, binary_mask * 255)

        # Create overlay
        overlay = original.copy()
        overlay[binary_mask == 1] = COLORS[predicted_class]

        overlay_path = os.path.join(OUTPUT_OVERLAY_DIR, f"overlay_{mask_id}.png")
        cv2.imwrite(overlay_path, overlay)

        print(f"[INFO] Saved mask → {mask_path}")
        print(f"[INFO] Saved overlay → {overlay_path}")

        mask_id += 1


if __name__ == "__main__":
    process_image()
