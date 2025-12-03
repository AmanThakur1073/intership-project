import onnxruntime as ort
import numpy as np
from PIL import Image


class EfficientNetB3_ONNX:
    def __init__(self):
        print("[INFO] Loading EfficientNetB3 ONNX model...")

        self.session = ort.InferenceSession(
            "../models/efficientnet_b3/model.safetensors",
            providers=["CPUExecutionProvider"]
        )

        # Model input size (B3 default = 300x300)
        self.input_height = 300
        self.input_width = 300

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("[SUCCESS] EfficientNetB3 ONNX Loaded!")


    def preprocess(self, img):
        img = img.resize((self.input_width, self.input_height))
        img = np.array(img).astype("float32") / 255.0

        # Normalize (Imagenet mean/std)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img = (img - mean) / std
        img = np.transpose(img, (0, 1, 2))   # HWC → HWC (same)
        img = np.transpose(img, (2, 0, 1))   # HWC → CHW
        img = np.expand_dims(img, axis=0)    # Add batch
        return img


    def extract(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
        except:
            print("[ERROR] EfficientNet cannot load image")
            return np.zeros(1536)

        x = self.preprocess(img)

        outputs = self.session.run([self.output_name], {self.input_name: x})
        feat = outputs[0].squeeze()

        return feat
