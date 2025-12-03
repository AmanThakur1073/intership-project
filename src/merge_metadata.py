import pandas as pd

# Correct paths (based on your folder)
train_csv = "../data/train.csv"
labels_csv = "../data/labels_final.csv"
output_csv = "../data/labels_with_metadata.csv"

print("[INFO] Loading CSV files...")
train_df = pd.read_csv(train_csv)
labels_df = pd.read_csv(labels_csv)

# Extract image file names
labels_df["image_id"] = labels_df["image_path"].apply(lambda x: x.split("/")[-1])
train_df["image_id"] = train_df["image_path"].apply(lambda x: x.split("/")[-1])

print("[INFO] Merging metadata...")
merged_df = pd.merge(labels_df, train_df, on="image_id", how="left")

# Fix date features
print("[INFO] Extracting date features...")
merged_df["Sampling_Date"] = pd.to_datetime(merged_df["Sampling_Date"], errors="coerce")
merged_df["year"] = merged_df["Sampling_Date"].dt.year
merged_df["month"] = merged_df["Sampling_Date"].dt.month
merged_df["day_of_year"] = merged_df["Sampling_Date"].dt.dayofyear

# Season function
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

merged_df["season"] = merged_df["month"].apply(get_season)

print("[SUCCESS] Metadata merged successfully!")
merged_df.to_csv(output_csv, index=False)
print(f"[SAVED] Final file: {output_csv}")
