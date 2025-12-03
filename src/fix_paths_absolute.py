import pandas as pd

df = pd.read_csv("../data/final_metadata_clean.csv")

# ABSOLUTE correct base path
base = r"C:/Users/amant/Desktop/csiro_segmentation/data/train/"

df["image_path"] = df["image_path"].apply(
    lambda x: base + x.split("/")[-1]
)

df.to_csv("../data/final_metadata_clean_FIXED.csv", index=False)

print("[SUCCESS] Paths fixed → final_metadata_clean_FIXED.csv")
