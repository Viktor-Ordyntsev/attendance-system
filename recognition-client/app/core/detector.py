from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from insightface import model_zoo


@dataclass(slots=True)
class DetectedFace:
    bbox: np.ndarray
    det_score: float
    kps: np.ndarray | None = None


class FaceDetector:
    def __init__(
        self,
        model_path: str | None = None,
        det_size: tuple[int, int] = (1024, 1024),
        threshold: float = 0.55,
        max_faces: int = 50,
        min_face_size: int = 60,
        ctx_id: int = -1,
    ) -> None:
        self.threshold = threshold
        self.max_faces = max_faces
        self.min_face_size = min_face_size

        resolved_model_path = self._resolve_model_path(model_path)
        model_source = (
            str(resolved_model_path)
            if resolved_model_path.exists()
            else resolved_model_path.name
        )

        self.model = model_zoo.get_model(model_source)
        self.model.prepare(
            ctx_id=ctx_id,
            input_size=det_size,
            det_thresh=threshold,
        )
        model_input_size = getattr(self.model, "input_size", None)
        if model_input_size is not None:
            self.det_size = tuple(model_input_size)
        else:
            self.det_size = det_size

    @staticmethod
    def _resolve_model_path(model_path: str | None) -> Path:
        candidates: list[Path] = []

        if model_path:
            candidates.append(Path(model_path).expanduser())

        candidates.extend(
            [
                Path.home() / ".insightface" / "models" / "buffalo_l" / "det_10g.onnx",
                Path("./app/models/det_10g.onnx"),
                Path("./app/models/scrfd_10g_bnkps.onnx"),
            ]
        )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        """
        Returns filtered SCRFD detections sorted by bbox area.
        """
        bboxes, kpss = self.model.detect(frame, max_num=self.max_faces)

        if bboxes is None or len(bboxes) == 0:
            return []

        faces: list[DetectedFace] = []
        for index, raw_bbox in enumerate(bboxes):
            bbox = np.asarray(raw_bbox[:4], dtype=np.float32)
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1

            if width < self.min_face_size or height < self.min_face_size:
                continue

            kps = None
            if kpss is not None and index < len(kpss):
                kps = np.asarray(kpss[index], dtype=np.float32)

            faces.append(
                DetectedFace(
                    bbox=bbox,
                    det_score=float(raw_bbox[4]),
                    kps=kps,
                )
            )

        faces.sort(
            key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
            reverse=True,
        )
        return faces
