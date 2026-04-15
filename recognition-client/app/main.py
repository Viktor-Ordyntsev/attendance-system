# recognition-client/app/main.py

from __future__ import annotations

import cv2
import os
import onnxruntime as ort
from pathlib import Path

from app.core.camera import Camera
from app.core.detector import FaceDetector
from app.core.recognizer import InsightFaceRecognizer, CustomFaceRecognizer
from app.core.matcher import FaceMatcher
from app.core.pipeline import RecognitionPipeline
from app.gui import ClientWindow


def load_env_file(env_path: str = ".env") -> None:
    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_demo_reference_db(detector: FaceDetector, recognizer: InsightFaceRecognizer) -> list[dict]:
    """
    Load reference images manually and create embeddings.
    Replace paths with your own images.
    """
    samples = [
        {"participant_id": 1, "label": "Person 1", "image_path": "samples/person1.jpg"},
        {"participant_id": 2, "label": "Person 2", "image_path": "samples/person2.jpg"},
    ]

    reference_db = []

    for sample in samples:
        image = cv2.imread(sample["image_path"])
        if image is None:
            print(f"Warning: failed to load {sample['image_path']}")
            continue

        faces = detector.detect(image)
        if not faces:
            print(f"Warning: no face found in {sample['image_path']}")
            continue

        face = faces[0]
        embedding = recognizer.get_embedding(image, face)

        reference_db.append(
            {
                "participant_id": sample["participant_id"],
                "label": sample["label"],
                "embedding": embedding,
            }
        )

    return reference_db


def main() -> None:
    load_env_file()
    camera_source = os.getenv("CAMERA_SOURCE", "0")
    print("ONNX Runtime providers:", ort.get_available_providers())
    camera = Camera(camera_source)
    detector = FaceDetector(
        model_path=os.getenv("SCRFD_MODEL_PATH"),
        det_size=(640, 640),
        threshold=0.55,
        max_faces=50,
        min_face_size=60,
        ctx_id=int(os.getenv("FACE_DETECTOR_CTX_ID", "-1")),
    )
    recognizer = CustomFaceRecognizer(
        model_path="./app/models/face_recognizer.onnx",
        input_size=(112, 112),
    )
    print("Requested recognizer providers:", recognizer.providers)
    print("Active recognizer providers:", recognizer.active_providers)
    matcher = FaceMatcher(threshold=0.75, min_margin=0.005)

    reference_db = build_demo_reference_db(detector, recognizer)
    matcher.set_reference_db(reference_db)

    pipeline = RecognitionPipeline(
        camera=camera,
        detector=detector,
        recognizer=recognizer,
        matcher=matcher,
        frame_skip=int(os.getenv("FRAME_SKIP", "2")),
        max_recognition_faces=int(os.getenv("MAX_RECOGNITION_FACES", "5")),
    )
    window = ClientWindow(pipeline=pipeline)
    window.run()


if __name__ == "__main__":
    main()
