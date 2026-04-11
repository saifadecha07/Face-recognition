from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import SessionLocal
from app.models import FaceSample, Person, Sighting
from app.tracking import CentroidTracker, TrackState


@dataclass
class RecognitionResult:
    label: str
    confidence: float
    distance: float


class FaceRecognizerService:
    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(str(settings.cascade_path))
        self.tracker = CentroidTracker(
            max_missing_frames=settings.lost_frames_threshold,
            motion_threshold=settings.motion_distance_threshold,
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.person_map: dict[int, Person] = {}
        self.training_sample_count = 0
        self._active_sighting_ids: dict[int, int] = {}
        self._train_recognizer()

    def _train_recognizer(self) -> None:
        face_samples: list[np.ndarray] = []
        labels: list[int] = []

        with SessionLocal() as session:
            stmt = select(Person).options(selectinload(Person.face_samples)).order_by(Person.name)
            persons = session.scalars(stmt).all()
            label_id = 0
            for person in persons:
                if not person.face_samples:
                    continue
                self.person_map[label_id] = person
                grayscale_faces = []
                for sample in person.face_samples:
                    face = self._decode_face_sample(sample)
                    grayscale_faces.append(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))
                face_samples.extend(grayscale_faces)
                labels.extend([label_id] * len(grayscale_faces))
                label_id += 1

        if face_samples:
            self.recognizer.train(face_samples, np.array(labels))
        self.training_sample_count = len(face_samples)
        print(
            "[recognition] loaded",
            len(self.person_map),
            "persons with",
            self.training_sample_count,
            "samples",
            f"(threshold={settings.recognition_threshold})",
        )

    @staticmethod
    def _decode_face_sample(sample: FaceSample) -> np.ndarray:
        return np.frombuffer(sample.image_data, dtype=np.uint8).reshape((100, 100, 3))

    def _predict_face(self, face_image: np.ndarray) -> RecognitionResult:
        if not self.person_map:
            return RecognitionResult(label="Unknown", confidence=0.0, distance=float("inf"))

        grayscale = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        label_id, distance = self.recognizer.predict(grayscale)
        confidence = max(0.0, 100.0 - float(distance))
        if confidence < settings.recognition_threshold:
            return RecognitionResult(label="Unknown", confidence=confidence, distance=float(distance))

        person = self.person_map.get(label_id)
        return RecognitionResult(
            label=person.name if person else "Unknown",
            confidence=confidence,
            distance=float(distance),
        )

    def detect_and_track(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        detections: list[dict] = []
        for (x, y, w, h) in faces:
            offset = 8
            y1 = max(0, y - offset)
            y2 = min(frame.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(frame.shape[1], x + w + offset)
            face_region = frame[y1:y2, x1:x2]
            if face_region.size == 0:
                continue

            face_region = cv2.resize(face_region, (100, 100))
            result = self._predict_face(face_region)
            detections.append(
                {
                    "bbox": (x, y, w, h),
                    "label": result.label,
                    "confidence": result.confidence,
                    "distance": result.distance,
                }
            )

        active_tracks, exited_tracks = self.tracker.update(detections)
        active_events = self._sync_active_tracks(active_tracks)
        exit_events = self._sync_exited_tracks(exited_tracks)

        for track in active_tracks:
            self._draw_track(frame, track)

        return frame, active_events, exit_events

    def _draw_track(self, frame: np.ndarray, track: TrackState) -> None:
        x, y, w, h = track.bbox
        color = (0, 255, 0) if track.label != "Unknown" else (0, 0, 255)
        movement_state = "moving" if self.tracker.is_moving(track) else "stable"
        title = (
            f"{track.label} #{track.track_id} {movement_state} "
            f"conf={track.confidence:.1f} dist={track.metadata.get('distance', float('nan')):.1f}"
        )
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, title, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _sync_active_tracks(self, active_tracks: list[TrackState]) -> list[dict]:
        now = datetime.utcnow()
        events: list[dict] = []

        with SessionLocal() as session:
            for track in active_tracks:
                sighting_id = self._active_sighting_ids.get(track.track_id)
                if sighting_id is None:
                    person = self._find_person_by_label(session, track.label)
                    sighting = Sighting(
                        track_id=track.track_id,
                        camera_name=settings.camera_name,
                        label=track.label,
                        status="active",
                        first_seen_at=now,
                        last_seen_at=now,
                        confidence=track.confidence,
                        movement_score=track.movement_score,
                        last_x=track.bbox[0],
                        last_y=track.bbox[1],
                        last_w=track.bbox[2],
                        last_h=track.bbox[3],
                        person_id=person.id if person else None,
                    )
                    session.add(sighting)
                    session.flush()
                    self._active_sighting_ids[track.track_id] = sighting.id
                    event_type = "entered"
                else:
                    sighting = session.get(Sighting, sighting_id)
                    if sighting is None:
                        continue
                    sighting.last_seen_at = now
                    sighting.status = "active"
                    sighting.label = track.label
                    sighting.confidence = track.confidence
                    sighting.movement_score = track.movement_score
                    sighting.last_x = track.bbox[0]
                    sighting.last_y = track.bbox[1]
                    sighting.last_w = track.bbox[2]
                    sighting.last_h = track.bbox[3]
                    event_type = "visible"

                events.append(
                    {
                        "event": event_type,
                        "track_id": track.track_id,
                        "label": track.label,
                        "moving": self.tracker.is_moving(track),
                        "confidence": round(track.confidence, 2),
                    }
                )

            session.commit()

        return events

    def _sync_exited_tracks(self, exited_tracks: list[TrackState]) -> list[dict]:
        if not exited_tracks:
            return []

        now = datetime.utcnow()
        events: list[dict] = []
        with SessionLocal() as session:
            for track in exited_tracks:
                sighting_id = self._active_sighting_ids.pop(track.track_id, None)
                if sighting_id is None:
                    continue

                sighting = session.get(Sighting, sighting_id)
                if sighting is None:
                    continue

                sighting.status = "exited"
                sighting.left_at = now
                sighting.last_seen_at = now
                sighting.movement_score = track.movement_score
                events.append(
                    {
                        "event": "exited",
                        "track_id": track.track_id,
                        "label": track.label,
                        "confidence": round(track.confidence, 2),
                    }
                )

            session.commit()

        return events

    @staticmethod
    def _find_person_by_label(session, label: str) -> Person | None:
        if label == "Unknown":
            return None
        return session.scalar(select(Person).where(Person.name == label))


def parse_camera_source(source: str) -> int | str:
    if source.isdigit():
        return int(source)
    return source


def open_camera_capture(source: str) -> cv2.VideoCapture:
    parsed_source = parse_camera_source(source)
    if isinstance(parsed_source, int) and os.name == "nt":
        backend_candidates = [
            ("DSHOW", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
        ]
        for backend_name, backend_id in backend_candidates:
            capture = cv2.VideoCapture(parsed_source, backend_id)
            if capture.isOpened():
                print(f"[camera] opened source {parsed_source} with {backend_name}")
                return capture
            capture.release()
        print(f"[camera] failed to open source {parsed_source} with DSHOW/MSMF, falling back to default backend")
    return cv2.VideoCapture(parsed_source)
