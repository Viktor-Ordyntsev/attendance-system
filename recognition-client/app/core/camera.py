# recognition-client/app/core/camera.py

from __future__ import annotations

import cv2
import numpy as np


class Camera:
    def __init__(self, source: int | str = 0) -> None:
        self.source = source
        self.cap: cv2.VideoCapture | None = None

    @staticmethod
    def _normalize_source(source: int | str) -> int | str:
        if isinstance(source, str):
            stripped = source.strip()
            if stripped.isdigit():
                return int(stripped)
            return stripped
        return source

    @staticmethod
    def _backend_candidates(source: int | str) -> list[tuple[str, int | None]]:
        if isinstance(source, int):
            candidates: list[tuple[str, int | None]] = [("default", None)]

            for backend_name in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY"):
                backend = getattr(cv2, backend_name, None)
                if isinstance(backend, int):
                    candidates.append((backend_name, backend))

            return candidates

        return [("default", None)]

    def open(self) -> None:
        source = self._normalize_source(self.source)
        attempts: list[str] = []

        for backend_name, backend in self._backend_candidates(source):
            cap = cv2.VideoCapture(source) if backend is None else cv2.VideoCapture(source, backend)

            if cap.isOpened():
                self.cap = cap
                self.source = source
                return

            cap.release()
            attempts.append(backend_name)

        tried = ", ".join(attempts) if attempts else "no backends"
        raise RuntimeError(
            f"Failed to open camera source: {source}. "
            f"Tried backends: {tried}. "
            "If you are using a USB/IP camera, set CAMERA_SOURCE to its device index, file path, or stream URL."
        )

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
