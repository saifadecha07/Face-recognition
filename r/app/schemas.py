from datetime import datetime

from pydantic import BaseModel


class PersonRead(BaseModel):
    id: int
    name: str
    username: str | None
    dataset_file: str | None
    sample_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class PersonSampleStatsRead(BaseModel):
    person_id: int
    name: str
    sample_count: int
    first_sample_at: datetime | None
    last_sample_at: datetime | None


class PersonAccessUpdate(BaseModel):
    username: str | None = None


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
    has_entry_image: bool
    has_exit_image: bool
    has_entry_frame_image: bool
    has_exit_frame_image: bool
    person_id: int | None
    unknown_identity_id: int | None

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
    username: str | None
    unknown_identity_id: int | None
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


class LabPresenceIdentityRead(BaseModel):
    sighting_id: int
    track_id: int
    person_id: int
    username: str | None
    label: str
    camera_name: str
    first_seen_at: datetime
    last_seen_at: datetime
    confidence: float
    present_in_frame: bool


class LabPresenceSnapshot(BaseModel):
    camera_running: bool
    camera_name: str
    active_count: int
    known_count: int
    unknown_count: int
    lab_head_present: bool
    usernames_in_lab: list[str]
    identities: list[LabPresenceIdentityRead]
