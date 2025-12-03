import pandas as pd

input_csv = "../data/labels_with_metadata.csv"
output_csv = "../data/final_metadata_clean.csv"

print("[INFO] Loading merged file...")
df = pd.read_csv(input_csv)

# Keep only unique image rows
print("[INFO] Dropping duplicate rows...")
df_unique = df.drop_duplicates(subset=["image_id"])

# Select only correct columns (remove _x, _y, target columns)
final_cols = [
    "image_path_x",
    "Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g",
    "Pre_GSHH_NDVI", "Height_Ave_cm", "State", "Species",
    "year", "month", "day_of_year", "season"
]

df_clean = df_unique[final_cols]

# Rename image_path_x → image_path
df_clean = df_clean.rename(columns={"image_path_x": "image_path"})

print("[SUCCESS] Clean file created!")
df_clean.to_csv(output_csv, index=False)
print(f"[SAVED] File: {output_csv}")
