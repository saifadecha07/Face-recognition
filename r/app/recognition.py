from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import os
import re
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import SessionLocal
from app.models import FaceSample, Person, Sighting, UnknownIdentity
from app.presence_cache import publish_presence_snapshot
from app.tracking import CentroidTracker, TrackState


@dataclass
class RecognitionResult:
    label: str
    confidence: float
    distance: float
    unknown_identity_id: int | None = None
    unknown_fingerprint: np.ndarray | None = None


@dataclass
class UnknownIdentitySnapshot:
    id: int
    label: str
    sample_count: int


class FaceRecognizerService:
    THAILAND_TZ = ZoneInfo("Asia/Bangkok")

    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(str(settings.cascade_path))
        self.tracker = CentroidTracker(
            max_missing_frames=settings.lost_frames_threshold,
            motion_threshold=settings.motion_distance_threshold,
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.person_map: dict[int, Person] = {}
        self.unknown_map: dict[int, UnknownIdentitySnapshot] = {}
        self.unknown_fingerprints: dict[int, np.ndarray] = {}
        self._next_unknown_sequence = 1
        self.training_sample_count = 0
        self._active_sighting_ids: dict[int, int] = {}
        self._track_histories: dict[int, deque[dict]] = {}
        self._train_recognizer()
        self._load_unknown_identities()

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
                grayscale_faces: list[np.ndarray] = []
                for sample in person.face_samples:
                    face = self._decode_face_sample(sample)
                    grayscale_faces.extend(self._build_training_variants(face))
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
            f"(distance_threshold={settings.recognition_distance_threshold})",
        )

    def _load_unknown_identities(self) -> None:
        with SessionLocal() as session:
            stmt = select(UnknownIdentity).order_by(UnknownIdentity.id)
            unknowns = session.scalars(stmt).all()

        self.unknown_map = {
            unknown.id: UnknownIdentitySnapshot(
                id=unknown.id,
                label=unknown.label,
                sample_count=unknown.sample_count,
            )
            for unknown in unknowns
        }
        self.unknown_fingerprints = {
            unknown.id: self._deserialize_fingerprint(unknown.fingerprint)
            for unknown in unknowns
        }
        sequences = [self._extract_unknown_sequence(unknown.label) for unknown in unknowns]
        self._next_unknown_sequence = (max(sequences) + 1) if sequences else 1
        print(
            "[recognition] loaded",
            len(self.unknown_map),
            "unknown identities",
            f"(threshold={settings.unknown_match_threshold})",
        )

    @staticmethod
    def _decode_face_sample(sample: FaceSample) -> np.ndarray:
        return np.frombuffer(sample.image_data, dtype=np.uint8).reshape((100, 100, 3))

    @staticmethod
    def _serialize_fingerprint(fingerprint: np.ndarray) -> bytes:
        return fingerprint.astype(np.float32).tobytes()

    @staticmethod
    def _deserialize_fingerprint(payload: bytes) -> np.ndarray:
        return np.frombuffer(payload, dtype=np.float32)

    @staticmethod
    def _extract_unknown_sequence(label: str) -> int:
        match = re.fullmatch(r"unknown_(\d+)", label)
        if not match:
            return 0
        return int(match.group(1))

    @staticmethod
    def _is_unknown_label(label: str) -> bool:
        return label == "Unknown" or label.startswith("unknown_")

    @staticmethod
    def _compute_unknown_fingerprint(face_image: np.ndarray) -> np.ndarray:
        grayscale = FaceRecognizerService._prepare_face_for_recognition(face_image)
        normalized = cv2.resize(grayscale, (32, 32)).astype(np.float32) / 255.0

        center = normalized[1:-1, 1:-1]
        neighbors = [
            normalized[:-2, :-2],
            normalized[:-2, 1:-1],
            normalized[:-2, 2:],
            normalized[1:-1, 2:],
            normalized[2:, 2:],
            normalized[2:, 1:-1],
            normalized[2:, :-2],
            normalized[1:-1, :-2],
        ]
        lbp = np.zeros_like(center, dtype=np.uint8)
        for index, neighbor in enumerate(neighbors):
            lbp |= ((neighbor >= center).astype(np.uint8) << index)

        hist, _ = np.histogram(lbp.ravel(), bins=16, range=(0, 256))
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()

        texture = center.reshape(-1).astype(np.float32)
        texture -= float(texture.mean())
        texture_norm = float(np.linalg.norm(texture))
        if texture_norm > 0:
            texture /= texture_norm

        fingerprint = np.concatenate([hist, texture]).astype(np.float32)
        norm = float(np.linalg.norm(fingerprint))
        if norm > 0:
            fingerprint /= norm
        return fingerprint

    @staticmethod
    def _fingerprint_distance(left: np.ndarray, right: np.ndarray) -> float:
        return 1.0 - float(np.dot(left, right))

    @staticmethod
    def _encode_evidence_image(face_image: np.ndarray) -> bytes | None:
        ok, encoded = cv2.imencode(".jpg", face_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return None
        return encoded.tobytes()

    @staticmethod
    def _prepare_face_for_recognition(face_image: np.ndarray) -> np.ndarray:
        if face_image.ndim == 3:
            grayscale = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = face_image.copy()
        grayscale = cv2.equalizeHist(grayscale)
        return cv2.GaussianBlur(grayscale, (3, 3), 0)

    def _build_training_variants(self, face_image: np.ndarray) -> list[np.ndarray]:
        base = self._prepare_face_for_recognition(face_image)
        variants = [base]
        variants.append(cv2.flip(base, 1))
        variants.append(cv2.convertScaleAbs(base, alpha=1.08, beta=6))
        variants.append(cv2.convertScaleAbs(base, alpha=0.92, beta=-6))
        return variants

    @staticmethod
    def _distance_to_confidence(distance: float) -> float:
        return max(0.0, min(100.0, 100.0 * float(np.exp(-distance / 80.0))))

    def _record_track_observation(self, track: TrackState) -> deque[dict]:
        history = self._track_histories.get(track.track_id)
        if history is None:
            history = deque(maxlen=settings.recognition_history_size)
            self._track_histories[track.track_id] = history

        history.append(
            {
                "label": track.label,
                "confidence": float(track.confidence),
                "distance": float(track.metadata.get("distance", float("inf"))),
                "unknown_identity_id": track.metadata.get("unknown_identity_id"),
            }
        )
        return history

    def _stabilize_track_identity(self, track: TrackState) -> None:
        history = self._record_track_observation(track)
        consensus_frames = settings.recognition_consensus_frames

        known_scores: dict[str, float] = {}
        known_counts: dict[str, int] = {}
        known_distances: dict[str, list[float]] = {}
        unknown_scores: dict[tuple[str, int | None], float] = {}
        unknown_counts: dict[tuple[str, int | None], int] = {}

        for item in history:
            label = item["label"]
            confidence = max(float(item["confidence"]), 0.1)
            if self._is_unknown_label(label):
                key = (label, item["unknown_identity_id"])
                unknown_scores[key] = unknown_scores.get(key, 0.0) + confidence
                unknown_counts[key] = unknown_counts.get(key, 0) + 1
                continue

            known_scores[label] = known_scores.get(label, 0.0) + confidence
            known_counts[label] = known_counts.get(label, 0) + 1
            known_distances.setdefault(label, []).append(float(item["distance"]))

        stable_known = None
        if known_scores:
            stable_known = max(known_scores.items(), key=lambda item: item[1])[0]
            average_distance = sum(known_distances[stable_known]) / len(known_distances[stable_known])
            if (
                known_counts.get(stable_known, 0) >= consensus_frames
                and average_distance <= settings.recognition_distance_threshold + 8.0
            ):
                track.label = stable_known
                track.confidence = known_scores[stable_known] / float(known_counts[stable_known])
                track.metadata["unknown_identity_id"] = None
                return

        if unknown_scores:
            stable_unknown_key = max(unknown_scores.items(), key=lambda item: item[1])[0]
            label, unknown_identity_id = stable_unknown_key
            if unknown_counts.get(stable_unknown_key, 0) >= min(consensus_frames, len(history)):
                track.label = label
                track.confidence = unknown_scores[stable_unknown_key] / float(unknown_counts[stable_unknown_key])
                track.metadata["unknown_identity_id"] = unknown_identity_id
                return

        if stable_known is not None:
            track.label = stable_known
            track.confidence = known_scores[stable_known] / float(known_counts[stable_known])

    @staticmethod
    def _annotate_evidence_frame(
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        label: str,
        confidence: float,
        timestamp: datetime,
    ) -> np.ndarray:
        annotated = frame.copy()
        x, y, w, h = bbox
        color = (0, 255, 0) if not FaceRecognizerService._is_unknown_label(label) else (0, 0, 255)
        thai_timestamp = timestamp.replace(tzinfo=ZoneInfo("UTC")).astimezone(FaceRecognizerService.THAILAND_TZ)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        title = f"{label} conf={confidence:.1f}"
        footer = f"{settings.camera_name} {thai_timestamp.strftime('%Y-%m-%d %H:%M:%S ICT')}"
        cv2.putText(
            annotated,
            title,
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        cv2.putText(
            annotated,
            footer,
            (10, max(20, annotated.shape[0] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        return annotated

    def _match_unknown_identity(self, fingerprint: np.ndarray) -> UnknownIdentitySnapshot | None:
        best_unknown = None
        best_distance = None
        for unknown_id, unknown_fingerprint in self.unknown_fingerprints.items():
            distance = self._fingerprint_distance(fingerprint, unknown_fingerprint)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_unknown = self.unknown_map.get(unknown_id)

        if best_unknown is None or best_distance is None:
            return None
        if best_distance > settings.unknown_match_threshold:
            return None
        return best_unknown

    def _create_unknown_identity(
        self,
        session,
        fingerprint: np.ndarray,
        now: datetime,
    ) -> UnknownIdentity:
        unknown = UnknownIdentity(
            label=f"unknown_{self._next_unknown_sequence:03d}",
            fingerprint=self._serialize_fingerprint(fingerprint),
            sample_count=1,
            created_at=now,
            last_seen_at=now,
        )
        session.add(unknown)
        session.flush()
        self._next_unknown_sequence += 1
        self.unknown_map[unknown.id] = UnknownIdentitySnapshot(
            id=unknown.id,
            label=unknown.label,
            sample_count=unknown.sample_count,
        )
        self.unknown_fingerprints[unknown.id] = fingerprint
        return unknown

    def _touch_unknown_identity(
        self,
        unknown: UnknownIdentity,
        fingerprint: np.ndarray | None,
        now: datetime,
    ) -> None:
        unknown.last_seen_at = now
        if fingerprint is None:
            return

        current = self.unknown_fingerprints.get(unknown.id)
        if current is None:
            current = self._deserialize_fingerprint(unknown.fingerprint)

        total = max(unknown.sample_count, 1)
        updated = ((current * total) + fingerprint) / float(total + 1)
        norm = float(np.linalg.norm(updated))
        if norm > 0:
            updated /= norm

        unknown.sample_count = total + 1
        unknown.fingerprint = self._serialize_fingerprint(updated)
        self.unknown_map[unknown.id] = UnknownIdentitySnapshot(
            id=unknown.id,
            label=unknown.label,
            sample_count=unknown.sample_count,
        )
        self.unknown_fingerprints[unknown.id] = updated

    def _predict_face(self, face_image: np.ndarray) -> RecognitionResult:
        unknown_fingerprint = self._compute_unknown_fingerprint(face_image)
        if not self.person_map:
            unknown = self._match_unknown_identity(unknown_fingerprint)
            if unknown:
                return RecognitionResult(
                    label=unknown.label,
                    confidence=0.0,
                    distance=float("inf"),
                    unknown_identity_id=unknown.id,
                    unknown_fingerprint=unknown_fingerprint,
                )
            return RecognitionResult(
                label="Unknown",
                confidence=0.0,
                distance=float("inf"),
                unknown_fingerprint=unknown_fingerprint,
            )

        grayscale = self._prepare_face_for_recognition(face_image)
        label_id, distance = self.recognizer.predict(grayscale)
        distance = float(distance)
        confidence = self._distance_to_confidence(distance)
        if distance > settings.recognition_distance_threshold:
            unknown = self._match_unknown_identity(unknown_fingerprint)
            if unknown:
                return RecognitionResult(
                    label=unknown.label,
                    confidence=confidence,
                    distance=distance,
                    unknown_identity_id=unknown.id,
                    unknown_fingerprint=unknown_fingerprint,
                )
            return RecognitionResult(
                label="Unknown",
                confidence=confidence,
                distance=distance,
                unknown_fingerprint=unknown_fingerprint,
            )

        person = self.person_map.get(label_id)
        return RecognitionResult(
            label=person.name if person else "Unknown",
            confidence=confidence,
            distance=distance,
        )

    def detect_and_track(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            settings.face_detection_scale_factor,
            settings.face_detection_min_neighbors,
            minSize=(settings.face_detection_min_size, settings.face_detection_min_size),
        )
        frame_timestamp = datetime.utcnow()

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
            annotated_frame = self._annotate_evidence_frame(
                frame=frame,
                bbox=(x, y, w, h),
                label=result.label,
                confidence=result.confidence,
                timestamp=frame_timestamp,
            )
            detections.append(
                {
                    "bbox": (x, y, w, h),
                    "label": result.label,
                    "confidence": result.confidence,
                    "distance": result.distance,
                    "unknown_identity_id": result.unknown_identity_id,
                    "unknown_fingerprint": result.unknown_fingerprint,
                    "evidence_image": self._encode_evidence_image(face_region),
                    "frame_evidence_image": self._encode_evidence_image(annotated_frame),
                }
            )

        active_tracks, exited_tracks = self.tracker.update(detections)
        for track in active_tracks:
            self._stabilize_track_identity(track)
            x, y, w, h = track.bbox
            face_region = frame[max(0, y):min(frame.shape[0], y + h), max(0, x):min(frame.shape[1], x + w)]
            if face_region.size > 0:
                resized_face = cv2.resize(face_region, (100, 100))
                track.metadata["evidence_image"] = self._encode_evidence_image(resized_face)
            annotated_frame = self._annotate_evidence_frame(
                frame=frame,
                bbox=track.bbox,
                label=track.label,
                confidence=track.confidence,
                timestamp=frame_timestamp,
            )
            track.metadata["frame_evidence_image"] = self._encode_evidence_image(annotated_frame)
        for track in exited_tracks:
            self._track_histories.pop(track.track_id, None)
        active_events = self._sync_active_tracks(active_tracks)
        exit_events = self._sync_exited_tracks(exited_tracks)
        if active_events or exit_events:
            publish_presence_snapshot(camera_running=True)

        for track in active_tracks:
            self._draw_track(frame, track)

        return frame, active_events, exit_events

    def _draw_track(self, frame: np.ndarray, track: TrackState) -> None:
        x, y, w, h = track.bbox
        color = (0, 255, 0) if not self._is_unknown_label(track.label) else (0, 0, 255)
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
                    unknown_identity = self._resolve_unknown_identity(session, track, now) if person is None else None
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
                        entry_image_data=track.metadata.get("evidence_image"),
                        entry_frame_image_data=track.metadata.get("frame_evidence_image"),
                        person_id=person.id if person else None,
                        unknown_identity_id=unknown_identity.id if unknown_identity else None,
                    )
                    session.add(sighting)
                    session.flush()
                    self._active_sighting_ids[track.track_id] = sighting.id
                    event_type = "entered"
                else:
                    sighting = session.get(Sighting, sighting_id)
                    if sighting is None:
                        continue
                    person = self._find_person_by_label(session, track.label)
                    unknown_identity = self._resolve_unknown_identity(session, track, now) if person is None else None
                    sighting.last_seen_at = now
                    sighting.status = "active"
                    sighting.label = person.name if person else (unknown_identity.label if unknown_identity else "Unknown")
                    sighting.confidence = track.confidence
                    sighting.movement_score = track.movement_score
                    sighting.last_x = track.bbox[0]
                    sighting.last_y = track.bbox[1]
                    sighting.last_w = track.bbox[2]
                    sighting.last_h = track.bbox[3]
                    sighting.person_id = person.id if person else None
                    sighting.unknown_identity_id = unknown_identity.id if unknown_identity else None
                    track.label = sighting.label
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
                if sighting.exit_image_data is None:
                    sighting.exit_image_data = track.metadata.get("evidence_image")
                if sighting.exit_frame_image_data is None:
                    sighting.exit_frame_image_data = track.metadata.get("frame_evidence_image")
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
        if FaceRecognizerService._is_unknown_label(label):
            return None
        return session.scalar(select(Person).where(Person.name == label))

    def _resolve_unknown_identity(
        self,
        session,
        track: TrackState,
        now: datetime,
    ) -> UnknownIdentity | None:
        unknown_identity_id = track.metadata.get("unknown_identity_id")
        if unknown_identity_id is not None:
            unknown = session.get(UnknownIdentity, unknown_identity_id)
            if unknown is not None:
                self._touch_unknown_identity(unknown, track.metadata.get("unknown_fingerprint"), now)
                track.label = unknown.label
                return unknown

        if isinstance(track.label, str) and track.label.startswith("unknown_"):
            unknown = session.scalar(select(UnknownIdentity).where(UnknownIdentity.label == track.label))
            if unknown is not None:
                self._touch_unknown_identity(unknown, track.metadata.get("unknown_fingerprint"), now)
                self.unknown_map[unknown.id] = UnknownIdentitySnapshot(
                    id=unknown.id,
                    label=unknown.label,
                    sample_count=unknown.sample_count,
                )
                return unknown

        fingerprint = track.metadata.get("unknown_fingerprint")
        if fingerprint is None:
            return None

        unknown = self._match_unknown_identity(fingerprint)
        if unknown is None:
            unknown = self._create_unknown_identity(session, fingerprint, now)
        else:
            unknown = session.get(UnknownIdentity, unknown.id)
            if unknown is None:
                unknown = self._create_unknown_identity(session, fingerprint, now)
                track.label = unknown.label
                track.metadata["unknown_identity_id"] = unknown.id
                return unknown
            self._touch_unknown_identity(unknown, fingerprint, now)

        track.label = unknown.label
        track.metadata["unknown_identity_id"] = unknown.id
        return unknown


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
