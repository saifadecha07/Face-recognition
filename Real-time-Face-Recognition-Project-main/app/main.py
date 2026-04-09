from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event, Thread

import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from sqlalchemy import select

from app.config import settings
from app.database import Base, SessionLocal, engine
from app.models import Person, Sighting
from app.recognition import FaceRecognizerService, parse_camera_source
from app.schemas import PersonRead, SightingRead


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


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

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def _run_loop(self) -> None:
        if self.service is None:
            return

        capture = cv2.VideoCapture(parse_camera_source(settings.camera_source))
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
        return session.scalars(select(Person).order_by(Person.name)).all()


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
