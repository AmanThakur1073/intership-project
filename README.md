# CSIRO Segmentation

Simple project to segment grass images into classes like:

- Green Grass
- Dry Grass
- Clover
- Dry Clover
- Weeds
- Soil

Structure:

## Web UI (Streamlit)

A simple Streamlit app has been added to provide an accessible frontend for running predictions.

Quick start:

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run src/web_app.py
```

Notes:

- Use the app to upload an image or enter a local path pointing into the workspace (e.g., `data/test/your.jpg`).
- If `lightgbm` fails to install on Windows, consider using conda: `conda install -c conda-forge lightgbm`.

## Generated outputs

Model outputs (masks, overlays) and result CSVs are stored under the `outputs/` folder and as CSV files (for example `submission.csv`). These files are intentionally not tracked in the repository to avoid committing large binaries or generated artifacts.

To regenerate outputs locally, run the prediction/generation scripts (ensure model weights in `checkpoints/` and `models/` are available):

```bash
# run full prediction that writes overlays/masks into `outputs/`
python src/predict_final.py

# or generate the submission CSV only
python src/generate_submission.py
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

Then validate local placement with the helper script:

```bash
bash scripts/validate_weights.sh
```

Or on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\validate_weights.ps1
```

If the script reports any missing files, add the required weights to the expected paths and run the command again.

> Remember: these model files are intentionally kept out of git to avoid large repository history and storage issues.
