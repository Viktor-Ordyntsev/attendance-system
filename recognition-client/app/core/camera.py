# recognition-client/app/core/camera.py

from __future__ import annotations

import cv2
import numpy as np


class Camera:
    def __init__(self, source: int | str = 0) -> None:
        self.source = source
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {self.source}")

    def read(self) -> np.ndarray:
        if self.cap is None:
            raise RuntimeError("Camera is not opened")

        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from camera")

        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None