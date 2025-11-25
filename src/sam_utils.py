import torch
import numpy as np
import cv2
import os
from segment_anything import sam_model_registry, SamPredictor


def load_sam_model(checkpoint_path=None):
    """
    Loads the SAM (Segment Anything) model safely using absolute path.
    """

    print("[INFO] Loading SAM model...")

    # Agar checkpoint_path pass nahi hua → Auto absolute path banao
    if checkpoint_path is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))  # src se bahar jao
        checkpoint_path = os.path.join(base_dir, "checkpoints", "sam_vit_b.pth")

    checkpoint_path = os.path.abspath(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[ERROR] SAM checkpoint missing at: {checkpoint_path}")

    # SAM load karo
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")

    predictor = SamPredictor(sam)

    print("[SUCCESS] SAM model loaded successfully!")
    return predictor


def get_sam_masks(predictor, image):
    """Run SAM model and get segmentation masks."""
    predictor.set_image(image)

    masks, scores, logits = predictor.predict(
        multimask_output=True
    )

    print(f"[INFO] {len(masks)} masks generated.")
    return masks
