from datetime import datetime

from pydantic import BaseModel


class PersonRead(BaseModel):
    id: int
    name: str
    dataset_file: str
    created_at: datetime

    model_config = {"from_attributes": True}


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
