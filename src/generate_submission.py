import pandas as pd
import os
from predict_final import predict_image   # import your function

TEST_IMAGES_FOLDER = "data/test/"
TEST_CSV = "data/test.csv"

# Load test.csv
test_df = pd.read_csv(TEST_CSV)

rows = []

print("[INFO] Starting predictions on full test set...")

for img_name in test_df['sample_id']:     # test.csv column
    image_path = os.path.join(TEST_IMAGES_FOLDER, img_name + ".jpg")

    # Run your model
    metadata, pred = predict_image(image_path)

    # Append in LONG Kaggle format
    rows.append([f"{img_name}__Dry_Clover_g", pred["Dry_Clover_g"]])
    rows.append([f"{img_name}__Dry_Dead_g", pred["Dry_Dead_g"]])
    rows.append([f"{img_name}__Dry_Green_g", pred["Dry_Green_g"]])
    rows.append([f"{img_name}__Dry_Total_g", pred["Dry_Total_g"]])
    rows.append([f"{img_name}__GDM_g", pred["GDM_g"]])

# Convert to csv
df = pd.DataFrame(rows, columns=["sample_id", "target"])
df.to_csv("submission.csv", index=False)

print("[SUCCESS] submission.csv generated successfully!")
