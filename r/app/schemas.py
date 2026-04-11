from datetime import datetime

from pydantic import BaseModel


class PersonRead(BaseModel):
    id: int
    name: str
    dataset_file: str | None
    role: str
    gesture_control_enabled: bool
    sample_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class PersonAccessUpdate(BaseModel):
    role: str
    gesture_control_enabled: bool


class SightingRead(BaseModel):
    id: int
    track_id: int
    camera_name: str
    label: str
    status: str
    first_seen_at: datetime
    last_seen_at: datetime
    left_at: datetime | None
    confidence: float
    movement_score: float
    last_x: int
    last_y: int
    last_w: int
    last_h: int
    person_id: int | None

    model_config = {"from_attributes": True}


class ActiveIdentityRead(BaseModel):
    sighting_id: int
    track_id: int
    camera_name: str
    label: str
    status: str
    first_seen_at: datetime
    last_seen_at: datetime
    confidence: float
    movement_score: float
    person_id: int | None
    role: str | None
    gesture_control_enabled: bool
    access_granted: bool
    access_reason: str
    present_in_frame: bool


class GestureAccessSnapshot(BaseModel):
    camera_running: bool
    camera_name: str
    active_count: int
    authorized_count: int
    identities: list[ActiveIdentityRead]


class GestureControllerSnapshot(BaseModel):
    camera_running: bool
    camera_name: str
    controller_selected: bool
    selection_reason: str
    controller: ActiveIdentityRead | None
    candidates: list[ActiveIdentityRead]
