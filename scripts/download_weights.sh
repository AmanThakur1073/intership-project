#!/usr/bin/env bash
set -euo pipefail

echo "This script downloads model weight files into the repository.
Edit the DOWNLOAD_URLS array to add real URLs for each weight file."

# Example: add real URLs here
DOWNLOAD_URLS=(
  "https://example.com/path/to/resnet50.pth"
  "https://example.com/path/to/sam_vit_b.pth"
)

OUT_DIRS=(checkpoints models)
for d in "${OUT_DIRS[@]}"; do
  mkdir -p "$d"
done

for url in "${DOWNLOAD_URLS[@]}"; do
  fname=$(basename "$url")
  echo "Downloading $fname..."
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "$fname" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$fname" "$url"
  else
    echo "Install curl or wget to download files" >&2
    exit 1
  fi
  # Move to checkpoints by default; adjust as needed
  mv -f "$fname" checkpoints/ || true
done

echo "Downloaded files moved to checkpoints/. Verify and move to models/ if required."