import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class MetadataProcessor:
    def __init__(self):
        # numerical metadata
        self.num_cols = ["Pre_GSHH_NDVI", "Height_Ave_cm", "year", "month", "day_of_year"]

        # categorical metadata
        self.cat_cols = ["State", "Species", "season"]

        # scalers
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        # fitted flag
        self.is_fitted = False

    # -------------------------
    # FIT ON TRAINING METADATA
    # -------------------------
    def fit(self, df):
        df = df.copy()

        # 1) Fit scaler on numerics
        self.scaler.fit(df[self.num_cols])

        # 2) Fit encoder on categorical
        self.encoder.fit(df[self.cat_cols])

        self.is_fitted = True
        print("[INFO] MetadataProcessor fitted successfully!")

    # -------------------------
    # TRANSFORM (TRAIN or TEST)
    # -------------------------
    def transform(self, df):
        if not self.is_fitted:
            raise ValueError("MetadataProcessor not fitted! Call fit() before transform().")

        df = df.copy()

        # numeric values
        num_scaled = self.scaler.transform(df[self.num_cols])

        # categorical values
        cat_encoded = self.encoder.transform(df[self.cat_cols])

        # merge
        final = np.hstack([num_scaled, cat_encoded])

        return final

    # -------------------------
    # SINGLE ROW TRANSFORM
    # -------------------------
    def transform_single(self, row):
        """Convert 1-row metadata into ML-ready vector"""
        df = pd.DataFrame([row])
        return self.transform(df)[0]
