# recognition-client/app/core/pipeline.py

from __future__ import annotations

import cv2
import numpy as np

from app.core.camera import Camera
from app.core.detector import FaceDetector
from app.core.recognizer import InsightFaceRecognizer
from app.core.matcher import FaceMatcher


class RecognitionPipeline:
    def __init__(
        self,
        camera: Camera,
        detector: FaceDetector,
        recognizer: InsightFaceRecognizer,
        matcher: FaceMatcher,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.recognizer = recognizer
        self.matcher = matcher

    @staticmethod
    def draw_face(frame: np.ndarray, bbox, label: str, score: float) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} ({score:.2f})"
        cv2.putText(
            frame,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        return frame

    def run(self) -> None:
        self.camera.open()

        try:
            while True:
                frame = self.camera.read()
                faces = self.detector.detect(frame)

                for face in faces:
                    embedding = self.recognizer.get_embedding(face)
                    match = self.matcher.match(embedding)
                    bbox = face.bbox
                    frame = self.draw_face(frame, bbox, match.label, match.score)

                cv2.imshow("Recognition Pipeline", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        finally:
            self.camera.release()
            cv2.destroyAllWindows()