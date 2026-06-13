# CSIRO Segmentation

## Project Overview

This repository implements a hybrid image segmentation and biomass estimation pipeline for grassland images. It combines:

- **SAM segmentation** for extracting image region features from vegetation images
- **ResNet / EfficientNet feature extraction** for visual embeddings
- **LightGBM multi-output regression** to predict biomass targets such as `Dry_Clover_g`, `Dry_Dead_g`, `Dry_Green_g`, `Dry_Total_g`, and `GDM_g`
- **Metadata features** drawn from the provided CSV metadata to improve predictions

The input dataset is stored under `data/`, containing image files, labels, and cleaned metadata. The final app uses these signals to generate segmentation overlays and prediction outputs.

## Key Features

- Streamlit web UI for image upload and prediction
- Mask/overlay generation for segmentation visualization
- Hybrid pipeline using both image features and tabular metadata
- Local model weight setup so large binaries are kept out of git

## Folder structure

- `data/` — raw dataset files, image folders, training/test CSVs, and metadata
- `src/` — project source code, training/prediction scripts, and feature extraction logic
- `models/` — local trained model files and regressors (not tracked in git)
- `checkpoints/` — local SAM checkpoint weights (not tracked in git)
- `outputs/` — generated masks, overlays, and submission CSVs (ignored by git)
- `scripts/` — helper scripts for validating local weight placement

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run src/web_app.py
```

3. Use the app to upload an image or enter a local path, for example `data/test/your.jpg`.

Notes:

- If `lightgbm` fails to install on Windows, consider using conda: `conda install -c conda-forge lightgbm`.

## Example outputs

This project generates:

- segmentation mask images in `outputs/masks/`
- overlay visualizations in `outputs/overlays/`
- submission CSVs such as `submission.csv`

Run the prediction pipeline locally to produce these files. Because generated artifacts are intentionally excluded from git, reviewers can reproduce them by following the setup steps.

## Results and evaluation

The training pipeline includes evaluation using validation MSE and R² metrics. Sample evaluation is produced by the training scripts in `src/train_regressor.py` and `src/train_hybrid.py`.

To reproduce training metrics, run the training script and review the printed validation results:

```bash
python src/train_regressor.py
python src/train_hybrid.py
```

## Local model weight setup

This repository does not store trained model weights. Place the required files locally in the folders below:

- `checkpoints/sam_vit_b.pth`
- `models/regressor.pkl`
- `models/pca.pkl`
- `models/hybrid_pca_regressor.pkl`
- `models/pca_with_metadata.pkl`
- `models/lgbm_with_metadata.pkl`
- `models/metadata_processor.pkl`
- `models/efficientnet_b3/model.safetensors`

Create the directories if they do not exist:

```bash
mkdir -p checkpoints models/efficientnet_b3
```

Validate local placement with the helper script:

```bash
bash scripts/validate_weights.sh
```

Or on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\validate_weights.ps1
```

If the script reports any missing files, add the required weights to the expected paths and run the command again.

> We keep weights out of git so the repository remains lightweight and easy to review.
