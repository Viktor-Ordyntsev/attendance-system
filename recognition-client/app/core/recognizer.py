from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class BaseFaceRecognizer(ABC):
    @abstractmethod
    def get_embedding(self, frame: np.ndarray, face) -> np.ndarray:
        pass


class InsightFaceRecognizer(BaseFaceRecognizer):
    def get_embedding(self, frame: np.ndarray, face) -> np.ndarray:
        if not hasattr(face, "embedding"):
            raise ValueError("Face object does not contain embedding")
        return np.asarray(face.embedding, dtype=np.float32)


class CustomFaceRecognizer(BaseFaceRecognizer):
    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (112, 112),
        providers: list[str] | None = None,
    ) -> None:
        providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size

    def get_embedding(self, frame: np.ndarray, face) -> np.ndarray:
        crop = self._extract_face(frame, face)
        tensor = self._preprocess(crop)
        output = self.session.run(None, {self.input_name: tensor})[0]

        embedding = np.asarray(output[0], dtype=np.float32).reshape(-1)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _extract_face(self, frame: np.ndarray, face) -> np.ndarray:
        x1, y1, x2, y2 = map(int, face.bbox)

        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid face bbox")

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError("Empty face crop")

        return crop

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        return tensor
