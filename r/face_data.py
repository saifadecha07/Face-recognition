import cv2
import numpy as np
from sqlalchemy import select

from app.config import settings
from app.database import Base, SessionLocal, engine, ensure_schema
from app.models import FaceSample, Person
from app.recognition import FaceRecognizerService, open_camera_capture


def ensure_gui_available() -> None:
    try:
        cv2.namedWindow("dataset-capture", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("dataset-capture")
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV GUI is unavailable. Install opencv-contrib-python instead of "
            "opencv-contrib-python-headless, then rerun face_data.py."
        ) from exc


def main() -> None:
    Base.metadata.create_all(bind=engine)
    ensure_schema()
    ensure_gui_available()

    cap = open_camera_capture(settings.camera_source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera source: {settings.camera_source}")

    face_cascade = cv2.CascadeClassifier(str(settings.cascade_path))
    if face_cascade.empty():
        raise RuntimeError(f"Unable to load cascade file: {settings.cascade_path}")

    skip = 0
    face_data: list[np.ndarray] = []
    file_name = input("Enter person name: ").strip()
    username = input("Enter student id (username): ").strip() or None
    print("Camera started. Look at the camera and press 'q' when you have enough samples.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        if len(faces) == 0:
            cv2.imshow("dataset-capture", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        skip += 1

        for x, y, w, h in faces[:1]:
            offset = 5
            y1 = max(0, y - offset)
            y2 = min(frame.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(frame.shape[1], x + w + offset)
            face_offset = frame[y1:y2, x1:x2]
            face_selection = cv2.resize(face_offset, (100, 100))

            # Reject blurry samples so training quality stays usable.
            blur_score = cv2.Laplacian(
                cv2.cvtColor(face_selection, cv2.COLOR_BGR2GRAY),
                cv2.CV_64F,
            ).var()

            if skip % 5 == 0 and blur_score >= 60.0:
                face_data.append(face_selection)
                print(f"captured: {len(face_data)} blur={blur_score:.1f}")

            preview = FaceRecognizerService._prepare_face_for_recognition(face_selection)
            cv2.imshow("face-preview", preview)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("dataset-capture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not face_data:
        print("no face samples captured")
        return

    with SessionLocal() as session:
        person = session.scalar(select(Person).where(Person.name == file_name))
        if person is None:
            person = Person(
                name=file_name,
                username=username,
                dataset_file=f"db:{file_name}",
            )
            session.add(person)
            session.flush()
        else:
            person.username = username
            session.query(FaceSample).filter(FaceSample.person_id == person.id).delete()

        session.add_all(
            [
                FaceSample(person_id=person.id, image_data=sample.astype(np.uint8).tobytes())
                for sample in face_data
            ]
        )
        session.commit()

    print(f"saved {len(face_data)} face samples to PostgreSQL for {file_name}")


if __name__ == "__main__":
    main()
