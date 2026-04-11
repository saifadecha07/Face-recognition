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

    if "gesture_control_enabled" not in columns:
        statements.append("ALTER TABLE persons ADD COLUMN gesture_control_enabled BOOLEAN DEFAULT FALSE")
        statements.append("UPDATE persons SET gesture_control_enabled = FALSE WHERE gesture_control_enabled IS NULL")
        statements.append("ALTER TABLE persons ALTER COLUMN gesture_control_enabled SET NOT NULL")

    if not statements:
        return

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
