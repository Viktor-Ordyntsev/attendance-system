# recognition-client/app/main.py

from __future__ import annotations

import cv2
import numpy as np

from app.core.camera import Camera
from app.core.detector import FaceDetector
from app.core.recognizer import InsightFaceRecognizer
from app.core.matcher import FaceMatcher
from app.core.pipeline import RecognitionPipeline


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
        embedding = recognizer.get_embedding(face)

        reference_db.append(
            {
                "participant_id": sample["participant_id"],
                "label": sample["label"],
                "embedding": embedding,
            }
        )

    return reference_db


def main() -> None:
    camera = Camera(2)
    detector = FaceDetector()
    recognizer = InsightFaceRecognizer()
    matcher = FaceMatcher(threshold=0.45, min_margin=0.02)

    reference_db = build_demo_reference_db(detector, recognizer)
    matcher.set_reference_db(reference_db)

    pipeline = RecognitionPipeline(
        camera=camera,
        detector=detector,
        recognizer=recognizer,
        matcher=matcher,
    )
    pipeline.run()


if __name__ == "__main__":
    main()