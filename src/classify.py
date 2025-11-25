import torch
import torch.nn as nn


class SimpleMaskClassifier(nn.Module):
    """
    A simple fully-connected classifier that takes
    mask features and predicts the class label.
    """

    def __init__(self, input_dim=256, num_classes=5):
        super(SimpleMaskClassifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def classify_mask(feature_vector, model_path=None):
    """
    Classifies a single mask using the trained classifier.
    feature_vector: a 1D vector (256-D) extracted from mask area
    """

    model = SimpleMaskClassifier()

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

    model.eval()

    with torch.no_grad():
        feature_vector = torch.tensor(feature_vector, dtype=torch.float32)
        output = model(feature_vector)
        predicted_class = torch.argmax(output).item()

    return predicted_class
