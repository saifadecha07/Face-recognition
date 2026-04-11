from app.database import Base, engine, ensure_schema
from app.main import runtime


def main() -> None:
    Base.metadata.create_all(bind=engine)
    ensure_schema()
    runtime.start()
    if runtime.thread:
        runtime.thread.join()


if __name__ == "__main__":
    main()
