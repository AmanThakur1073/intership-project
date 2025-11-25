import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator

def extract_features_from_image(predictor, image_path):
    """
    Extract features using SAM Automatic Mask Generator + basic image stats
    """
    image = cv2.imread(image_path)
    if image is None:
        print("[ERROR] Cannot read:", image_path)
        return np.zeros(50)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # 1) Use SAM Automatic Mask Generator
    # -----------------------------
    try:
        mask_generator = SamAutomaticMaskGenerator(predictor.model)
        masks = mask_generator.generate(image_rgb)
    except Exception as e:
        print("[ERROR] SAM FAILED:", e)
        return np.zeros(50)

    # -----------------------------
    # 2) Extract mask areas
    # -----------------------------
    features = []

    areas = []
    for m in masks:
        area = m["segmentation"].sum()
        areas.append(area)

    # Top 10 largest mask areas
    areas = sorted(areas, reverse=True)
    areas = areas[:10]
    features.extend(areas)

    # -----------------------------
    # 3) Add basic image statistics
    # -----------------------------
    features.append(np.mean(image))
    features.append(np.std(image))
    features.append(np.mean(image[:, :, 0]))  # B
    features.append(np.mean(image[:, :, 1]))  # G
    features.append(np.mean(image[:, :, 2]))  # R

    # -----------------------------
    # 4) Fixed 50-length feature vector
    # -----------------------------
    if len(features) < 50:
        features += [0] * (50 - len(features))
    else:
        features = features[:50]

    return np.array(features, dtype=np.float32)
