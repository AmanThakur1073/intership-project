import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


class ResNetFeatureExtractor:

    def __init__(self):
        print("[INFO] Loading ResNet50...")
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("[SUCCESS] ResNet50 loaded!")

    def extract(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img_t = self.transform(img)
        img_t = img_t.unsqueeze(0)

        with torch.no_grad():
            features = self.model(img_t)

        features = features.view(-1).numpy()
        return features
