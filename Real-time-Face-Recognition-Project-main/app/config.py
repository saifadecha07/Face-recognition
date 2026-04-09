from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "face-surveillance"
    database_url: str = "postgresql+psycopg://postgres:postgres@db:5432/face_surveillance"
    camera_source: str = "0"
    camera_name: str = "front-door"
    frame_width: int = 640
    frame_height: int = 480
    recognition_threshold: float = 85.0
    lost_frames_threshold: int = 20
    motion_distance_threshold: float = 18.0
    face_data_dir: Path = Path("face_dataset")
    cascade_path: Path = Path("haarcascade_frontalface_alt.xml")
    run_camera_on_startup: bool = False
    show_camera_window: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
