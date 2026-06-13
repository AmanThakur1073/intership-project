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

## Handling model weights and large binaries

Model weights (e.g. `.pth`, `.pt`, `.h5`, `.safetensors`) are large and typically should not be committed to the main Git history. You have two recommended options:

- Use Git LFS (recommended if you want to keep weights linked to the repo):

  ```bash
  # install and enable LFS locally
  git lfs install
  # track common weight file extensions
  git lfs track "*.pth" "*.pt" "*.h5" "*.safetensors" "*.pb" "*.pkl"
  git add .gitattributes
  git commit -m "Track model weights with Git LFS"
  ```

  Note: After enabling LFS locally, add and push weight files as usual; they will be stored via LFS. Make sure your Git host (GitHub) supports LFS and that you understand any storage limits or billing implications.

- Or host weights externally and download them at setup time (avoids storing large files in the repo):
  - We provide helper scripts in `scripts/download_weights.sh` and `scripts/download_weights.ps1`. Edit the `DOWNLOAD_URLS` list to point to your hosted model files (S3, Google Drive, huggingface, etc.) and run the script to populate `checkpoints/` or `models/`.

  ```bash
  # Linux/macOS
  bash scripts/download_weights.sh

  # Windows (PowerShell)
  powershell -ExecutionPolicy Bypass -File scripts\download_weights.ps1
  ```

Choose the approach that matches your workflow; if you want, I can enable Git LFS for this repo and migrate any existing large files into LFS for you.
