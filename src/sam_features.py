import numpy as np
import cv2
from sam_utils import load_sam_model, get_sam_masks


class SAMFeatureExtractor:
    """
    Extract numeric features from images using SAM segmentation masks.
    """

    def __init__(self, checkpoint_path=None):
        print("[INFO] Initializing SAM Feature Extractor...")
        self.predictor = load_sam_model(checkpoint_path)
        print("[SUCCESS] SAM Feature Extractor Ready!")

    def extract(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            print("[ERROR] Cannot read:", image_path)
            return np.zeros(256, dtype=np.float32)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Segment using SAM
        masks = get_sam_masks(self.predictor, image_rgb)

        features = []

        # Each mask → 2 features: area + mean pixel
        for m in masks:
            area = np.sum(m)
            mean_val = np.mean(image[m > 0])
            features.extend([area, mean_val])

        # Global image stats
        features.append(np.mean(image))
        features.append(np.std(image))

        # Ensure fixed 256-dim vector
        if len(features) < 256:
            features.extend([0] * (256 - len(features)))
        else:
            features = features[:256]

        return np.array(features, dtype=np.float32)
