from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Attendance System API")


# In-memory storage for MVP
participants_store: list[dict] = []
events_store: list[dict] = []
attendance_events_store: list[dict] = []

participant_id_seq = 1
event_id_seq = 1
attendance_event_id_seq = 1


class ParticipantCreate(BaseModel):
    full_name: str = Field(..., min_length=1)
    group_name: str = Field(..., min_length=1)


class ParticipantResponse(BaseModel):
    id: int
    full_name: str
    group_name: str
    created_at: datetime


class EventCreate(BaseModel):
    title: str = Field(..., min_length=1)
    location: str = Field(..., min_length=1)
    start_at: datetime


class EventResponse(BaseModel):
    id: int
    title: str
    location: str
    start_at: datetime
    created_at: datetime


class AttendanceEventCreate(BaseModel):
    event_id: int = Field(..., gt=0)
    participant_id: int = Field(..., gt=0)
    label: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime


class AttendanceEventResponse(BaseModel):
    id: int
    status: Literal["ok"]
    received_at: datetime
    event: AttendanceEventCreate


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Attendance System API is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/participants", response_model=list[ParticipantResponse])
def get_participants() -> list[ParticipantResponse]:
    return [ParticipantResponse(**participant) for participant in participants_store]


@app.post("/participants", response_model=ParticipantResponse)
def create_participant(data: ParticipantCreate) -> ParticipantResponse:
    global participant_id_seq

    participant = {
        "id": participant_id_seq,
        "full_name": data.full_name,
        "group_name": data.group_name,
        "created_at": datetime.now(timezone.utc),
    }
    participants_store.append(participant)
    participant_id_seq += 1

    return ParticipantResponse(**participant)


@app.get("/events", response_model=list[EventResponse])
def get_events() -> list[EventResponse]:
    return [EventResponse(**event) for event in events_store]


@app.post("/events", response_model=EventResponse)
def create_event(data: EventCreate) -> EventResponse:
    global event_id_seq

    event = {
        "id": event_id_seq,
        "title": data.title,
        "location": data.location,
        "start_at": data.start_at,
        "created_at": datetime.now(timezone.utc),
    }
    events_store.append(event)
    event_id_seq += 1

    return EventResponse(**event)


@app.get("/attendance-events")
def get_attendance_events() -> list[dict]:
    return attendance_events_store


@app.post("/attendance-events", response_model=AttendanceEventResponse)
def create_attendance_event(event: AttendanceEventCreate) -> AttendanceEventResponse:
    global attendance_event_id_seq

    event_exists = any(item["id"] == event.event_id for item in events_store)
    if not event_exists:
        raise HTTPException(status_code=404, detail="Event not found")

    participant_exists = any(item["id"] == event.participant_id for item in participants_store)
    if not participant_exists:
        raise HTTPException(status_code=404, detail="Participant not found")

    duplicate_exists = any(
        item["event"]["event_id"] == event.event_id
        and item["event"]["participant_id"] == event.participant_id
        for item in attendance_events_store
    )
    if duplicate_exists:
        raise HTTPException(status_code=409, detail="Participant already marked for this event")

    print(
        "[ATTENDANCE EVENT]",
        {
            "event_id": event.event_id,
            "participant_id": event.participant_id,
            "label": event.label,
            "score": event.score,
            "timestamp": event.timestamp.isoformat(),
        },
    )

    response_payload = {
        "id": attendance_event_id_seq,
        "status": "ok",
        "received_at": datetime.now(timezone.utc),
        "event": event.model_dump(),
    }
    attendance_events_store.append(response_payload)
    attendance_event_id_seq += 1

    return AttendanceEventResponse(**response_payload)