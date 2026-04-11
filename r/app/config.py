from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DEFAULT_CASCADE_PATH = PROJECT_DIR / "haarcascade_frontalface_alt.xml"


class Settings(BaseSettings):
    app_name: str = "face-surveillance"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/face_surveillance"
    camera_source: str = "0"
    camera_name: str = "front-door"
    frame_width: int = 640
    frame_height: int = 480
    recognition_threshold: float = 85.0
    lost_frames_threshold: int = 20
    motion_distance_threshold: float = 18.0
    cascade_path: Path = DEFAULT_CASCADE_PATH
    run_camera_on_startup: bool = False
    show_camera_window: bool = True

    @field_validator("cascade_path", mode="before")
    @classmethod
    def resolve_cascade_path(cls, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (PROJECT_DIR / path).resolve()

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
