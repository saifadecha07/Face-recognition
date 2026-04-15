from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event, Thread

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import Base, SessionLocal, engine, ensure_schema
from app.models import FaceSample, Person, Sighting
from app.presence_cache import build_presence_snapshot, get_cached_presence_snapshot, publish_presence_snapshot
from app.recognition import FaceRecognizerService, open_camera_capture
from app.schemas import (
    ActiveIdentityRead,
    GestureControllerSnapshot,
    GestureAccessSnapshot,
    LabPresenceSnapshot,
    PersonAccessUpdate,
    PersonRead,
    PersonSampleStatsRead,
    SightingRead,
)


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
ROLE_PRIORITY = {
    "lab_head": 300,
    "admin": 250,
    "supervisor": 200,
    "staff": 150,
    "user": 100,
    "guest": 10,
}


def build_access_identity(sighting: Sighting) -> ActiveIdentityRead:
    person = sighting.person
    role = person.role if person else None
    username = person.username if person else None
    gesture_enabled = bool(person.gesture_control_enabled) if person else False

    if person is None:
        access_granted = False
        access_reason = "unknown_person"
    elif not gesture_enabled:
        access_granted = False
        access_reason = "gesture_control_disabled"
    else:
        access_granted = True
        access_reason = f"granted_for_role:{person.role}"

    return ActiveIdentityRead(
        sighting_id=sighting.id,
        track_id=sighting.track_id,
        camera_name=sighting.camera_name,
        label=sighting.label,
        status=sighting.status,
        first_seen_at=sighting.first_seen_at,
        last_seen_at=sighting.last_seen_at,
        confidence=sighting.confidence,
        movement_score=sighting.movement_score,
        person_id=sighting.person_id,
        username=username,
        unknown_identity_id=sighting.unknown_identity_id,
        role=role,
        gesture_control_enabled=gesture_enabled,
        access_granted=access_granted,
        access_reason=access_reason,
        present_in_frame=sighting.status == "active",
    )

def rank_identity(identity: ActiveIdentityRead) -> tuple[int, float, float]:
    role_score = ROLE_PRIORITY.get(identity.role or "", 0)
    return (
        role_score,
        identity.confidence,
        identity.last_seen_at.timestamp(),
    )


def select_controller(identities: list[ActiveIdentityRead]) -> tuple[ActiveIdentityRead | None, str]:
    authorized = [identity for identity in identities if identity.access_granted and identity.present_in_frame]
    if not authorized:
        if identities:
            return None, "no_authorized_identity_in_frame"
        return None, "no_identity_in_frame"

    controller = max(authorized, key=rank_identity)
    return controller, f"selected_by_role_priority:{controller.role}"


class CameraRuntime:
    def __init__(self) -> None:
        self.stop_event = Event()
        self.thread: Thread | None = None
        self.service: FaceRecognizerService | None = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event = Event()
        self.service = FaceRecognizerService()
        self.thread = Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        publish_presence_snapshot(camera_running=True)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        publish_presence_snapshot(camera_running=False)

    def _run_loop(self) -> None:
        if self.service is None:
            return

        capture = open_camera_capture(settings.camera_source)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)

        while not self.stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                continue

            rendered_frame, _, _ = self.service.detect_and_track(frame)

            if settings.show_camera_window:
                cv2.imshow(settings.app_name, rendered_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.stop_event.set()
                    break

        capture.release()
        if settings.show_camera_window:
            cv2.destroyAllWindows()


runtime = CameraRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    ensure_schema()
    if settings.run_camera_on_startup:
        runtime.start()
    try:
        yield
    finally:
        runtime.stop()


app = FastAPI(title="Face Surveillance API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "camera_autostart": settings.run_camera_on_startup,
        "camera_running": runtime.thread is not None and runtime.thread.is_alive(),
    }


@app.get("/persons", response_model=list[PersonRead])
def list_persons() -> list[Person]:
    with SessionLocal() as session:
        stmt = select(Person).options(selectinload(Person.face_samples)).order_by(Person.name)
        return session.scalars(stmt).all()


@app.get("/persons/sample-stats", response_model=list[PersonSampleStatsRead])
def list_person_sample_stats() -> list[PersonSampleStatsRead]:
    with SessionLocal() as session:
        stmt = (
            select(
                Person.id.label("person_id"),
                Person.name.label("name"),
                func.count(FaceSample.id).label("sample_count"),
                func.min(FaceSample.created_at).label("first_sample_at"),
                func.max(FaceSample.created_at).label("last_sample_at"),
            )
            .outerjoin(FaceSample, FaceSample.person_id == Person.id)
            .group_by(Person.id, Person.name)
            .order_by(Person.name)
        )
        rows = session.execute(stmt).all()

    return [
        PersonSampleStatsRead(
            person_id=row.person_id,
            name=row.name,
            sample_count=row.sample_count,
            first_sample_at=row.first_sample_at,
            last_sample_at=row.last_sample_at,
        )
        for row in rows
    ]


@app.put("/persons/{person_id}/access", response_model=PersonRead)
def update_person_access(person_id: int, payload: PersonAccessUpdate) -> Person:
    with SessionLocal() as session:
        person = session.get(Person, person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="person_not_found")

        person.username = payload.username.strip() if payload.username else None
        session.commit()
        stmt = (
            select(Person)
            .options(selectinload(Person.face_samples))
            .where(Person.id == person_id)
        )
        person = session.scalar(stmt)

    publish_presence_snapshot(camera_running=runtime.thread is not None and runtime.thread.is_alive())
    return person


@app.get("/presence/current", response_model=LabPresenceSnapshot)
def current_lab_presence() -> LabPresenceSnapshot:
    cached = get_cached_presence_snapshot()
    if cached is not None:
        return cached

    return build_presence_snapshot(
        camera_running=runtime.thread is not None and runtime.thread.is_alive(),
    )


@app.get("/sightings/active", response_model=list[SightingRead])
def list_active_sightings() -> list[Sighting]:
    with SessionLocal() as session:
        stmt = select(Sighting).where(Sighting.status == "active").order_by(Sighting.last_seen_at.desc())
        return session.scalars(stmt).all()


@app.get("/sightings/recent", response_model=list[SightingRead])
def list_recent_sightings(limit: int = 50) -> list[Sighting]:
    with SessionLocal() as session:
        stmt = select(Sighting).order_by(Sighting.last_seen_at.desc()).limit(limit)
        return session.scalars(stmt).all()


@app.get("/sightings/{sighting_id}/entry-image")
def get_sighting_entry_image(sighting_id: int) -> Response:
    with SessionLocal() as session:
        sighting = session.get(Sighting, sighting_id)
        if sighting is None:
            raise HTTPException(status_code=404, detail="sighting_not_found")
        if sighting.entry_image_data is None:
            raise HTTPException(status_code=404, detail="entry_image_not_found")
        return Response(content=sighting.entry_image_data, media_type="image/jpeg")


@app.get("/sightings/{sighting_id}/exit-image")
def get_sighting_exit_image(sighting_id: int) -> Response:
    with SessionLocal() as session:
        sighting = session.get(Sighting, sighting_id)
        if sighting is None:
            raise HTTPException(status_code=404, detail="sighting_not_found")
        if sighting.exit_image_data is None:
            raise HTTPException(status_code=404, detail="exit_image_not_found")
        return Response(content=sighting.exit_image_data, media_type="image/jpeg")


@app.get("/sightings/{sighting_id}/entry-frame-image")
def get_sighting_entry_frame_image(sighting_id: int) -> Response:
    with SessionLocal() as session:
        sighting = session.get(Sighting, sighting_id)
        if sighting is None:
            raise HTTPException(status_code=404, detail="sighting_not_found")
        if sighting.entry_frame_image_data is None:
            raise HTTPException(status_code=404, detail="entry_frame_image_not_found")
        return Response(content=sighting.entry_frame_image_data, media_type="image/jpeg")


@app.get("/sightings/{sighting_id}/exit-frame-image")
def get_sighting_exit_frame_image(sighting_id: int) -> Response:
    with SessionLocal() as session:
        sighting = session.get(Sighting, sighting_id)
        if sighting is None:
            raise HTTPException(status_code=404, detail="sighting_not_found")
        if sighting.exit_frame_image_data is None:
            raise HTTPException(status_code=404, detail="exit_frame_image_not_found")
        return Response(content=sighting.exit_frame_image_data, media_type="image/jpeg")


@app.get("/access/gesture/current", response_model=GestureAccessSnapshot)
def current_gesture_access(authorized_only: bool = False) -> GestureAccessSnapshot:
    with SessionLocal() as session:
        stmt = (
            select(Sighting)
            .options(selectinload(Sighting.person), selectinload(Sighting.unknown_identity))
            .where(Sighting.status == "active")
            .order_by(Sighting.last_seen_at.desc())
        )
        sightings = session.scalars(stmt).all()

    all_identities = [build_access_identity(sighting) for sighting in sightings]
    identities = all_identities
    if authorized_only:
        identities = [identity for identity in all_identities if identity.access_granted]

    return GestureAccessSnapshot(
        camera_running=runtime.thread is not None and runtime.thread.is_alive(),
        camera_name=settings.camera_name,
        active_count=len(all_identities),
        authorized_count=sum(1 for identity in all_identities if identity.access_granted),
        identities=identities,
    )


@app.get("/access/gesture/controller", response_model=GestureControllerSnapshot)
def current_gesture_controller() -> GestureControllerSnapshot:
    with SessionLocal() as session:
        stmt = (
            select(Sighting)
            .options(selectinload(Sighting.person), selectinload(Sighting.unknown_identity))
            .where(Sighting.status == "active")
            .order_by(Sighting.last_seen_at.desc())
        )
        sightings = session.scalars(stmt).all()

    identities = [build_access_identity(sighting) for sighting in sightings]
    controller, selection_reason = select_controller(identities)

    return GestureControllerSnapshot(
        camera_running=runtime.thread is not None and runtime.thread.is_alive(),
        camera_name=settings.camera_name,
        controller_selected=controller is not None,
        selection_reason=selection_reason,
        controller=controller,
        candidates=identities,
    )


@app.post("/camera/start")
def start_camera() -> dict:
    runtime.start()
    return {"status": "started"}


@app.post("/camera/stop")
def stop_camera() -> dict:
    runtime.stop()
    return {"status": "stopped"}


app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "app_name": settings.app_name,
        },
    )
