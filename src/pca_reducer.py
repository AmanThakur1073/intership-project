import numpy as np
from sklearn.decomposition import PCA
import pickle
import os

class PCAFeatureReducer:
    def __init__(self, output_dim=256):
        self.output_dim = output_dim
        self.pca = PCA(n_components=output_dim)

    def fit(self, X):
        print("[INFO] Fitting PCA on features...")
        self.pca.fit(X)
        print("[SUCCESS] PCA fitted!")

    def transform(self, X):
        return self.pca.transform(X)

    def save(self, path="models/pca.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.pca, f)
        print("[SAVED] PCA stored at:", path)

    def load(self, path="models/pca.pkl"):
        with open(path, "rb") as f:
            self.pca = pickle.load(f)
        print("[LOADED] PCA loaded from:", path)
