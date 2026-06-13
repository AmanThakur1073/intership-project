#!/usr/bin/env bash
set -euo pipefail

REQUIRED=(
  "checkpoints/sam_vit_b.pth"
  "models/regressor.pkl"
  "models/pca.pkl"
  "models/hybrid_pca_regressor.pkl"
  "models/pca_with_metadata.pkl"
  "models/lgbm_with_metadata.pkl"
  "models/metadata_processor.pkl"
  "models/efficientnet_b3/model.safetensors"
)

missing=0
for path in "${REQUIRED[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "MISSING: $path"
    missing=1
  else
    echo "OK: $path"
  fi
done

if [[ $missing -eq 1 ]]; then
  echo "\nSome required model weight files are missing. Place them in the correct paths and rerun this script."
  exit 1
fi

echo "All required model weight files are present."
