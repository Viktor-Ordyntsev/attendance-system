from __future__ import annotations

import cv2
import numpy as np
import requests
from collections import deque
from datetime import datetime, timezone

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
        self.recent_labels: deque[str] = deque(maxlen=5)
        self.confirmed_label: str | None = None
        self.confirmation_count: int = 3
        self.backend_url: str = "http://127.0.0.1:8000"
        self.event_id: int = 1
        self.sent_participants: set[int] = set()

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

    def send_attendance_event(
        self,
        participant_id: int,
        label: str,
        score: float,
    ) -> None:
        payload = {
            "event_id": self.event_id,
            "participant_id": participant_id,
            "label": label,
            "score": score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            response = requests.post(
                f"{self.backend_url}/attendance-events",
                json=payload,
                timeout=5,
            )
            response.raise_for_status()
            print("[BACKEND RESPONSE]", response.json())
        except requests.RequestException as exc:
            print("[BACKEND ERROR]", exc)

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

                    # Debug: print top candidates
                    print([
                        (c.label, round(c.score, 3))
                        for c in match.top_candidates
                    ], "->", match.reason)

                    current_label = match.label if match.is_match else "unknown"
                    self.recent_labels.append(current_label)

                    stable_label = "unknown"
                    if current_label != "unknown":
                        same_count = sum(1 for label in self.recent_labels if label == current_label)
                        if same_count >= self.confirmation_count:
                            self.confirmed_label = current_label
                            stable_label = current_label

                            if (
                                match.participant_id is not None
                                and match.participant_id not in self.sent_participants
                            ):
                                self.send_attendance_event(
                                    participant_id=match.participant_id,
                                    label=match.label,
                                    score=match.score,
                                )
                                self.sent_participants.add(match.participant_id)
                    elif self.confirmed_label is not None:
                        stable_label = self.confirmed_label

                    frame = self.draw_face(frame, bbox, stable_label, match.score)

                if not faces:
                    self.recent_labels.clear()
                    self.confirmed_label = None

                cv2.imshow("Recognition Pipeline", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        finally:
            self.camera.release()
            cv2.destroyAllWindows()