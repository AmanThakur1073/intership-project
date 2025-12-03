import cv2
import os

class SuperResolution:
    def __init__(self, model_path, model_type="espcn", scale=4):
        print("[INFO] Loading SR model:", model_type)

        if not os.path.exists(model_path):
            raise FileNotFoundError("SR model not found: " + model_path)

        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(model_path)
        self.sr.setModel(model_type, scale)

        print("[SUCCESS] SR model loaded!")

    def upscale(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Cannot read image: " + image_path)

        upscaled = self.sr.upsample(img)

        # Save to temp
        save_path = image_path.replace(".jpg", "_up.jpg")
        cv2.imwrite(save_path, upscaled)

        return save_path
