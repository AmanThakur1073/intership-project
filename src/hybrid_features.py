from sam_features import SAMFeatureExtractor
from resnet_features import ResNetFeatureExtractor

import numpy as np


class HybridFeatureExtractor:

    def __init__(self):
        print("[INFO] Initializing SAM + ResNet feature extractor...")
        self.sam_extractor = SAMFeatureExtractor()
        self.resnet_extractor = ResNetFeatureExtractor()
        print("[SUCCESS] Hybrid feature extractor ready!")

    def extract(self, image_path):

        # --- SAM FEATURES ---
        sam_features = self.sam_extractor.extract(image_path)

        # Replace NaN (because some masks may be empty)
        sam_features = np.nan_to_num(sam_features, nan=0.0)

        # --- ResNet FEATURES ---
        resnet_features = self.resnet_extractor.extract(image_path)

        # Merge both
        final = np.concatenate([sam_features, resnet_features])

        # Safety: remove any NaN or inf
        final = np.nan_to_num(final, nan=0.0, posinf=0.0, neginf=0.0)

        return final
