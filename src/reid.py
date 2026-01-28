import torch
from torchreid.utils.feature_extractor import FeatureExtractor

class ReIDExtractor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            device=self.device
        )

    def extract(self, person_img):
        features = self.extractor(person_img)
        return features[0]

