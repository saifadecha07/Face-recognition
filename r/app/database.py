from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def ensure_schema() -> None:
    inspector = inspect(engine)
    if "persons" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("persons")}
    statements: list[str] = []

    if "role" not in columns:
        statements.append("ALTER TABLE persons ADD COLUMN role VARCHAR(50) DEFAULT 'user'")
        statements.append("UPDATE persons SET role = 'user' WHERE role IS NULL")
        statements.append("ALTER TABLE persons ALTER COLUMN role SET NOT NULL")

    if "username" not in columns:
        statements.append("ALTER TABLE persons ADD COLUMN username VARCHAR(64) NULL")
        statements.append("CREATE UNIQUE INDEX IF NOT EXISTS ix_persons_username ON persons (username)")

    if "gesture_control_enabled" not in columns:
        statements.append("ALTER TABLE persons ADD COLUMN gesture_control_enabled BOOLEAN DEFAULT FALSE")
        statements.append("UPDATE persons SET gesture_control_enabled = FALSE WHERE gesture_control_enabled IS NULL")
        statements.append("ALTER TABLE persons ALTER COLUMN gesture_control_enabled SET NOT NULL")

    sighting_tables = set(inspector.get_table_names())
    if "sightings" in sighting_tables:
        sighting_columns = {column["name"] for column in inspector.get_columns("sightings")}
        if "entry_image_data" not in sighting_columns:
            statements.append("ALTER TABLE sightings ADD COLUMN entry_image_data BYTEA NULL")
        if "exit_image_data" not in sighting_columns:
            statements.append("ALTER TABLE sightings ADD COLUMN exit_image_data BYTEA NULL")
        if "entry_frame_image_data" not in sighting_columns:
            statements.append("ALTER TABLE sightings ADD COLUMN entry_frame_image_data BYTEA NULL")
        if "exit_frame_image_data" not in sighting_columns:
            statements.append("ALTER TABLE sightings ADD COLUMN exit_frame_image_data BYTEA NULL")
        if "unknown_identity_id" not in sighting_columns:
            statements.append("ALTER TABLE sightings ADD COLUMN unknown_identity_id INTEGER NULL")
            statements.append(
                "ALTER TABLE sightings "
                "ADD CONSTRAINT fk_sightings_unknown_identity_id "
                "FOREIGN KEY (unknown_identity_id) REFERENCES unknown_identities (id)"
            )
            statements.append(
                "CREATE INDEX IF NOT EXISTS ix_sightings_unknown_identity_id "
                "ON sightings (unknown_identity_id)"
            )

    if not statements:
        return

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
