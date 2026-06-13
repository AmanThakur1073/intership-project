import os
import tempfile
from typing import Tuple, Dict

import cv2
import numpy as np
from PIL import Image

# import the module so we can access the global extractor
import predict_final as pf
from predict_final import TARGETS, predict_image
from sam_utils import get_sam_masks


def predict_from_path(image_path: str) -> Tuple[Dict, Dict]:
    """Run prediction for an image at given filesystem path."""
    return predict_image(image_path)


def predict_from_bytes(image_bytes: bytes, suffix: str = ".jpg") -> Tuple[Dict, Dict]:
    """Save uploaded bytes to a temporary file and run prediction."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        meta, preds = predict_image(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return meta, preds


def get_targets():
    return list(TARGETS)


def get_overlay_from_path(image_path: str, alpha: float = 0.5) -> Image.Image:
    """Return a PIL Image with SAM masks alpha-blended on the original image."""
    # Ensure predictor exists
    extractor = getattr(pf, "extractor", None)
    if extractor is None or not hasattr(extractor, "sam_extractor"):
        raise RuntimeError("SAM extractor not initialized in predict_final module")

    predictor = extractor.sam_extractor.predictor

    # Read image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    masks = get_sam_masks(predictor, img_rgb)

    # Create colored overlay
    h, w = img_rgb.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # Simple deterministic colors
    rng = np.random.RandomState(42)

    for i, m in enumerate(masks):
        color = rng.randint(50, 230, size=3)
        mask_bool = m.astype(bool)
        overlay[mask_bool] = color

    # Blend
    blended = (img_rgb * (1 - alpha) + overlay * alpha).astype(np.uint8)

    return Image.fromarray(blended)

