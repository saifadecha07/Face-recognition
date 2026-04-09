from pathlib import Path

import cv2
import numpy as np

from app.config import settings


def main() -> None:
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(str(settings.cascade_path))
    dataset_path = Path(settings.face_data_dir)
    dataset_path.mkdir(parents=True, exist_ok=True)

    skip = 0
    face_data: list[np.ndarray] = []
    file_name = input("Enter person name: ").strip()

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

            if skip % 5 == 0:
                face_data.append(face_selection)
                print(f"captured: {len(face_data)}")

            cv2.imshow("face-preview", face_selection)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("dataset-capture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not face_data:
        print("no face samples captured")
        return

    samples = np.array(face_data).reshape((len(face_data), -1))
    np.save(dataset_path / file_name, samples)
    print(f"dataset saved: {dataset_path / f'{file_name}.npy'}")


if __name__ == "__main__":
    main()
