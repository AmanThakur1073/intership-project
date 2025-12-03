import pandas as pd
import os

# Load the metadata file
df = pd.read_csv("../data/final_metadata_clean_FIXED.csv")

# IMPORTANT: Your absolute base path
BASE = r"C:\Users\amant\Desktop\csiro_segmentation"

def fix_path(p):
    # Remove any leading folder
    p = p.replace("data/", "")
    p = p.replace("\\", "/")

    # Build full absolute path
    full = os.path.join(BASE, "data", p).replace("\\", "/")
    return full

df["image_path"] = df["image_path"].apply(fix_path)

df.to_csv("../data/final_metadata_clean_FIXED.csv", index=False)

print("[SUCCESS] Created: final_metadata_clean_FIXED.csv")
