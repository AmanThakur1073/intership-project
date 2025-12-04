import pandas as pd

# Load Kaggle train.csv
df = pd.read_csv("data/train.csv")

# Pivot to make 1 row per image
df_pivot = df.pivot_table(
    index=["image_path", "Sampling_Date", "State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"],
    columns="target_name",
    values="target"
).reset_index()

df_pivot.columns.name = None

# Save final metadata CSV
df_pivot.to_csv("data/final_metadata_clean_FIXED.csv", index=False)

print("Step 2 complete: ONE ROW PER IMAGE metadata created!")


# Load the pivoted CSV
df = pd.read_csv("data/final_metadata_clean_FIXED.csv")

# Convert Sampling_Date to datetime
df["Sampling_Date"] = pd.to_datetime(df["Sampling_Date"], format="%Y/%m/%d")

# Extract date-based features
df["year"] = df["Sampling_Date"].dt.year
df["month"] = df["Sampling_Date"].dt.month
df["day_of_year"] = df["Sampling_Date"].dt.dayofyear

# Map month to season
def get_season(m):
    if m in [12, 1, 2]: return "Summer"
    if m in [3, 4, 5]: return "Autumn"
    if m in [6, 7, 8]: return "Winter"
    return "Spring"

df["season"] = df["month"].apply(get_season)

# Save updated metadata
df.to_csv("data/final_metadata_clean_FIXED.csv", index=False)

print("Step 3 complete: Date features added successfully!")
