from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import SessionLocal
from app.models import Sighting
from app.schemas import LabPresenceIdentityRead, LabPresenceSnapshot

try:
    from redis import Redis
    from redis.exceptions import RedisError
except Exception:  # pragma: no cover - dependency may be absent before install
    Redis = None

    class RedisError(Exception):
        pass


_redis_client: Redis | None | bool = False


def build_presence_identity(sighting: Sighting) -> LabPresenceIdentityRead | None:
    person = sighting.person
    if person is None:
        return None

    return LabPresenceIdentityRead(
        sighting_id=sighting.id,
        track_id=sighting.track_id,
        person_id=person.id,
        username=person.username,
        label=sighting.label,
        camera_name=sighting.camera_name,
        first_seen_at=sighting.first_seen_at,
        last_seen_at=sighting.last_seen_at,
        confidence=sighting.confidence,
        present_in_frame=sighting.status == "active",
    )


def build_presence_snapshot(camera_running: bool) -> LabPresenceSnapshot:
    with SessionLocal() as session:
        stmt = (
            select(Sighting)
            .options(selectinload(Sighting.person))
            .where(Sighting.status == "active")
            .order_by(Sighting.last_seen_at.desc())
        )
        sightings = session.scalars(stmt).all()

    known_identities = [identity for sighting in sightings if (identity := build_presence_identity(sighting)) is not None]
    usernames_in_lab = sorted({identity.username for identity in known_identities if identity.username})
    lab_head_present = any(
        sighting.person is not None and sighting.person.role == "lab_head"
        for sighting in sightings
    )

    return LabPresenceSnapshot(
        camera_running=camera_running,
        camera_name=settings.camera_name,
        active_count=len(sightings),
        known_count=len(known_identities),
        unknown_count=sum(1 for sighting in sightings if sighting.person is None),
        lab_head_present=lab_head_present,
        usernames_in_lab=usernames_in_lab,
        identities=known_identities,
    )


def _get_redis_client() -> Redis | None:
    global _redis_client

    if _redis_client is False:
        if not settings.redis_url or Redis is None:
            _redis_client = None
        else:
            try:
                _redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
                _redis_client.ping()
            except Exception:
                _redis_client = None
    return _redis_client


def _serialize_snapshot(snapshot: LabPresenceSnapshot) -> str:
    payload = snapshot.model_dump(mode="json")
    payload["cached_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return json.dumps(payload, ensure_ascii=True)


def _deserialize_snapshot(payload: str) -> LabPresenceSnapshot:
    data = json.loads(payload)
    data.pop("cached_at", None)
    return LabPresenceSnapshot.model_validate(data)


def publish_presence_snapshot(camera_running: bool) -> LabPresenceSnapshot:
    snapshot = build_presence_snapshot(camera_running=camera_running)
    client = _get_redis_client()
    if client is None:
        return snapshot

    try:
        client.set(
            settings.redis_presence_key,
            _serialize_snapshot(snapshot),
            ex=settings.redis_presence_ttl_seconds,
        )
    except RedisError:
        pass
    return snapshot


def get_cached_presence_snapshot() -> LabPresenceSnapshot | None:
    client = _get_redis_client()
    if client is None:
        return None

    try:
        payload = client.get(settings.redis_presence_key)
    except RedisError:
        return None

    if not payload:
        return None

    try:
        return _deserialize_snapshot(payload)
    except (ValueError, TypeError):
        return None
