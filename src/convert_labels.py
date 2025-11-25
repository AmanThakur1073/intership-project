import pandas as pd
import os

csv_path = "data/train.csv"
out_csv = "data/labels_final.csv"

print("[INFO] Reading CSV...")
df = pd.read_csv(csv_path)

# Extract correct image ID
print("[INFO] Extracting base image ID...")
df["image_id"] = df["sample_id"].apply(lambda x: x.split("__")[0])

# Build correct image_path
df["image_path"] = df["image_id"].apply(lambda x: f"data/train/{x}.jpg")

# Pivot table to wide format
print("[INFO] Pivoting table...")
df_pivot = df.pivot_table(
    index="image_path",
    columns="target_name",
    values="target",
    aggfunc="first"
).reset_index()

# Keep only images that exist
print("[INFO] Checking which images exist...")
df_pivot["exists"] = df_pivot["image_path"].apply(lambda x: os.path.exists(x))
df_pivot = df_pivot[df_pivot["exists"] == True].drop(columns=["exists"])

# Save output
df_pivot.to_csv(out_csv, index=False)

print("\n[SUCCESS] labels_final.csv created at:", out_csv)
print(df_pivot.head())
