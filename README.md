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
