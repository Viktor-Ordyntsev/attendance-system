from __future__ import annotations

import cv2
import numpy as np
import requests
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Queue, Empty
from threading import Event, Lock, Thread
from time import perf_counter

from app.core.camera import Camera
from app.core.detector import FaceDetector
from app.core.recognizer import InsightFaceRecognizer
from app.core.matcher import FaceMatcher


@dataclass(slots=True)
class PipelineStatus:
    faces_in_frame: int
    confirmed_label: str | None
    last_match_label: str
    last_match_score: float
    last_reason: str
    backend_ok: bool
    backend_message: str
    provider_name: str
    processing_ms: float


class RecognitionPipeline:
    def __init__(
        self,
        camera: Camera,
        detector: FaceDetector,
        recognizer: InsightFaceRecognizer,
        matcher: FaceMatcher,
        frame_skip: int = 1,
        max_recognition_faces: int = 5,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.recognizer = recognizer
        self.matcher = matcher
        self.frame_skip = max(1, frame_skip)
        self.max_recognition_faces = max(1, max_recognition_faces)
        self.recent_labels: deque[str] = deque(maxlen=5)
        self.confirmed_label: str | None = None
        self.confirmation_count: int = 3
        self.backend_url: str = "http://127.0.0.1:8000"
        self.event_id: int = 1
        self.sent_participants: set[int] = set()
        self.last_status = PipelineStatus(
            faces_in_frame=0,
            confirmed_label=None,
            last_match_label="unknown",
            last_match_score=0.0,
            last_reason="idle",
            backend_ok=True,
            backend_message="No events sent yet",
            provider_name=", ".join(getattr(self.recognizer, "active_providers", ["unknown"])),
            processing_ms=0.0,
        )
        self.recent_events: deque[str] = deque(maxlen=10)
        self._status_lock = Lock()
        self._event_queue: Queue[dict | None] = Queue()
        self._stop_event = Event()
        self._sender_thread = Thread(target=self._event_sender_loop, daemon=True)
        self._sender_thread.start()
        self._frame_index = 0
        self._last_overlays: list[tuple[np.ndarray, str, float]] = []

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
    ) -> bool:
        payload = {
            "event_id": self.event_id,
            "participant_id": participant_id,
            "label": label,
            "score": score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._event_queue.put(payload)
        return True

    def _event_sender_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._event_queue.get(timeout=0.2)
            except Empty:
                continue

            if payload is None:
                break

            self._send_attendance_event_sync(payload)

    def _send_attendance_event_sync(self, payload: dict) -> bool:
        payload = {
            "event_id": payload["event_id"],
            "participant_id": payload["participant_id"],
            "label": payload["label"],
            "score": payload["score"],
            "timestamp": payload["timestamp"],
        }

        try:
            response = requests.post(
                f"{self.backend_url}/attendance-events",
                json=payload,
                timeout=5,
            )
            response.raise_for_status()
            message = f"{payload['label']} sent ({payload['score']:.2f})"
            with self._status_lock:
                self.last_status.backend_ok = True
                self.last_status.backend_message = message
                self.recent_events.appendleft(message)
            print("[BACKEND RESPONSE]", response.json())
            return True
        except requests.RequestException as exc:
            with self._status_lock:
                self.last_status.backend_ok = False
                self.last_status.backend_message = str(exc)
            print("[BACKEND ERROR]", exc)
            return False

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, PipelineStatus]:
        started_at = perf_counter()
        self._frame_index += 1
        should_analyze = ((self._frame_index - 1) % self.frame_skip) == 0

        if not should_analyze:
            for bbox, label, score in self._last_overlays:
                frame = self.draw_face(frame, bbox, label, score)

            with self._status_lock:
                self.last_status.faces_in_frame = len(self._last_overlays)
                self.last_status.last_reason = "skipped_frame"
                self.last_status.processing_ms = (perf_counter() - started_at) * 1000.0
                status = PipelineStatus(
                    faces_in_frame=self.last_status.faces_in_frame,
                    confirmed_label=self.last_status.confirmed_label,
                    last_match_label=self.last_status.last_match_label,
                    last_match_score=self.last_status.last_match_score,
                    last_reason=self.last_status.last_reason,
                    backend_ok=self.last_status.backend_ok,
                    backend_message=self.last_status.backend_message,
                    provider_name=self.last_status.provider_name,
                    processing_ms=self.last_status.processing_ms,
                )

            return frame, status

        faces = self.detector.detect(frame)
        with self._status_lock:
            self.last_status.faces_in_frame = len(faces)
            self.last_status.last_match_label = "unknown"
            self.last_status.last_match_score = 0.0
            self.last_status.last_reason = "no_faces" if not faces else "processing"

        overlays: list[tuple[np.ndarray, str, float]] = []
        for face in faces[: self.max_recognition_faces]:
            embedding = self.recognizer.get_embedding(frame, face)
            match = self.matcher.match(embedding)
            bbox = face.bbox

            print([
                (c.label, round(c.score, 3))
                for c in match.top_candidates
            ], "->", match.reason)

            current_label = match.label if match.is_match else "unknown"
            self.recent_labels.append(current_label)
            with self._status_lock:
                self.last_status.last_match_label = current_label
                self.last_status.last_match_score = match.score
                self.last_status.last_reason = match.reason

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

            overlays.append((np.asarray(bbox, dtype=np.float32), stable_label, match.score))
            frame = self.draw_face(frame, bbox, stable_label, match.score)

        if not faces:
            self.recent_labels.clear()
            self.confirmed_label = None

        self._last_overlays = overlays
        with self._status_lock:
            self.last_status.confirmed_label = self.confirmed_label
            self.last_status.processing_ms = (perf_counter() - started_at) * 1000.0
            status = PipelineStatus(
                faces_in_frame=self.last_status.faces_in_frame,
                confirmed_label=self.last_status.confirmed_label,
                last_match_label=self.last_status.last_match_label,
                last_match_score=self.last_status.last_match_score,
                last_reason=self.last_status.last_reason,
                backend_ok=self.last_status.backend_ok,
                backend_message=self.last_status.backend_message,
                provider_name=self.last_status.provider_name,
                processing_ms=self.last_status.processing_ms,
            )

        return frame, status

    def shutdown(self) -> None:
        self._stop_event.set()
        self._event_queue.put(None)
        if self._sender_thread.is_alive():
            self._sender_thread.join(timeout=1.0)
        self.camera.release()

    def run(self) -> None:
        self.camera.open()

        try:
            while True:
                frame = self.camera.read()
                frame, _ = self.process_frame(frame)

                cv2.imshow("Recognition Pipeline", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        finally:
            self.shutdown()
            cv2.destroyAllWindows()
