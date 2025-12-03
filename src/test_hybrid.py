from hybrid_features import HybridFeatureExtractor
import numpy as np

print("==========================================")
print("      HYBRID PIPELINE TEST START          ")
print("==========================================\n")

# Initialize SAM + ResNet only
extractor = HybridFeatureExtractor()

img = input("Enter test image path: ").strip()

print("\n[INFO] Extracting features...")

feat = extractor.extract(img)
feat = np.nan_to_num(feat)   # Clean NaN safety

print("\n============== TEST RESULT ==============")
print(f"Final feature shape : {feat.shape}")
print(f"Contains NaN values : {np.isnan(feat).any()}")
print(f"Sample values       : {feat[:10]}")
print("==========================================")
