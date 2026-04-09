from app.database import Base, engine
from app.main import runtime


def main() -> None:
    Base.metadata.create_all(bind=engine)
    runtime.start()
    if runtime.thread:
        runtime.thread.join()


if __name__ == "__main__":
    main()
