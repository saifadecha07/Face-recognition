from sqlalchemy import select

from app.database import Base, SessionLocal, engine
from app.models import Person, Sighting


def main() -> None:
    Base.metadata.create_all(bind=engine)

    name = input("Enter person name to delete: ").strip()
    if not name:
        print("person name is required")
        return

    with SessionLocal() as session:
        person = session.scalar(select(Person).where(Person.name == name))
        if person is None:
            print(f"person not found: {name}")
            return

        affected_sightings = (
            session.query(Sighting)
            .filter(Sighting.person_id == person.id)
            .update({Sighting.person_id: None}, synchronize_session=False)
        )

        sample_count = len(person.face_samples)
        session.delete(person)
        session.commit()

    print(
        f"deleted person '{name}' with {sample_count} face samples; "
        f"cleared {affected_sightings} related sightings"
    )


if __name__ == "__main__":
    main()
