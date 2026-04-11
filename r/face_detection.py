import cv2

from app.config import settings


def main() -> None:
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(str(settings.cascade_path))

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for x, y, w, h in faces[:1]:
            offset = 10
            y1 = max(0, y - offset)
            y2 = min(frame.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(frame.shape[1], x + w + offset)
            face_offset = frame[y1:y2, x1:x2]
            face_selection = cv2.resize(face_offset, (100, 100))

            cv2.imshow("face-preview", face_selection)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("camera-preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
