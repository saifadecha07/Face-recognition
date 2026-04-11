from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    dataset_file: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    role: Mapped[str] = mapped_column(String(50), default="user", index=True)
    gesture_control_enabled: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    sightings: Mapped[list["Sighting"]] = relationship(back_populates="person")
    face_samples: Mapped[list["FaceSample"]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan",
    )

    @property
    def sample_count(self) -> int:
        return len(self.face_samples)


class FaceSample(Base):
    __tablename__ = "face_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("persons.id"), index=True)
    image_data: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    person: Mapped[Person] = relationship(back_populates="face_samples")


class Sighting(Base):
    __tablename__ = "sightings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    track_id: Mapped[int] = mapped_column(Integer, index=True)
    camera_name: Mapped[str] = mapped_column(String(120), index=True)
    label: Mapped[str] = mapped_column(String(120), index=True)
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    left_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    movement_score: Mapped[float] = mapped_column(Float, default=0.0)
    last_x: Mapped[int] = mapped_column(Integer, default=0)
    last_y: Mapped[int] = mapped_column(Integer, default=0)
    last_w: Mapped[int] = mapped_column(Integer, default=0)
    last_h: Mapped[int] = mapped_column(Integer, default=0)
    person_id: Mapped[int | None] = mapped_column(ForeignKey("persons.id"), nullable=True)

    person: Mapped[Person | None] = relationship(back_populates="sightings")
