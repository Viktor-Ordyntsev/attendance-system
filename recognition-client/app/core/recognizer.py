# recognition-client/app/core/recognizer.py

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseFaceRecognizer(ABC):
    @abstractmethod
    def get_embedding(self, face) -> np.ndarray:
        """
        Extract embedding from detected face object.
        """
        pass

class InsightFaceRecognizer(BaseFaceRecognizer):
    def get_embedding(self, face)-> np.ndarray:
        if not hasattr(face, "embedding"):
            raise ValueError("Face object does not contain embedding")
        return np.asarray(face.embedding, dtype=np.float32)
    
class CustomFaceRecognizer(BaseFaceRecognizer):
    def get_embedding(self, face) -> np.ndarray:
        raise NotImplementedError("Custom model is not implemented yet")
