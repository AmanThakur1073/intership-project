import numpy as np
from sam_features import SAMFeatureExtractor
from resnet_features import ResNetFeatureExtractor


class HybridFeatureExtractor:
    def __init__(self):
        print("[INFO] Initializing Hybrid: SAM + ResNet50 ...")

        self.sam_extractor = SAMFeatureExtractor()
        self.resnet_extractor = ResNetFeatureExtractor()

        print("[SUCCESS] Hybrid extractor ready (SAM + ResNet50)!")

    def extract(self, image_path):

        # SAM
        try:
            sam_feat = self.sam_extractor.extract(image_path)
        except:
            sam_feat = np.zeros(256)

        # ResNet50
        try:
            res_feat = self.resnet_extractor.extract(image_path)
        except:
            res_feat = np.zeros(2048)

        # merge
        final = np.hstack([sam_feat, res_feat])
        return final
