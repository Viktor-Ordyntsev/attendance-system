# recognition-client/app/core/detector.py

from __future__ import annotations

import numpy as np
from insightface.app import FaceAnalysis


class FaceDetector:
    def __init__(self, det_size: tuple[int, int] = (640, 640)) -> None:
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect(self, frame: np.ndarray) -> list:
        """
        Returns a list of detected faces.
        Each face object usually contains:
        - bbox
        - det_score
        - embedding
        - kps
        """
        faces = self.app.get(frame)
        return faces