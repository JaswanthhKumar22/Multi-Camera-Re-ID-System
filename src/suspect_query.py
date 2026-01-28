import cv2

class SuspectQuery:
    def __init__(self, image_path, reid_model):
        self.image_path = image_path
        self.reid = reid_model
        self.embedding = self._extract_embedding()

    def _extract_embedding(self):
        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError("❌ suspect.jpg not found")

        img = cv2.resize(img, (128, 256))
        embedding = self.reid.extract(img)
        print("✅ Suspect embedding extracted:", embedding.shape)
        return embedding
