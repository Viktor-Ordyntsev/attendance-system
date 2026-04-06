from __future__ import annotations

from datetime import datetime, timezone
import requests


class BackendClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def send_attendance_event(
        self,
        participant_id: int,
        label: str,
        score: float,
    ) -> None:
        url = f"{self.base_url}/attendance-events"

        payload = {
            "participant_id": participant_id,
            "label": label,
            "score": score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()

        print("[BACKEND RESPONSE]", response.json())