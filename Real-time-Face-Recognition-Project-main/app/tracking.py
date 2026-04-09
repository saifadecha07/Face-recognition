from dataclasses import dataclass, field
from math import hypot
from time import time


def centroid(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x, y, w, h = box
    return x + (w // 2), y + (h // 2)


@dataclass
class TrackState:
    track_id: int
    bbox: tuple[int, int, int, int]
    label: str
    confidence: float
    first_seen_ts: float = field(default_factory=time)
    last_seen_ts: float = field(default_factory=time)
    missing_frames: int = 0
    movement_score: float = 0.0

    @property
    def center(self) -> tuple[int, int]:
        return centroid(self.bbox)


class CentroidTracker:
    def __init__(self, max_missing_frames: int, motion_threshold: float) -> None:
        self.max_missing_frames = max_missing_frames
        self.motion_threshold = motion_threshold
        self._next_track_id = 1
        self._tracks: dict[int, TrackState] = {}

    def update(self, detections: list[dict]) -> tuple[list[TrackState], list[TrackState]]:
        matched_ids: set[int] = set()

        for detection in detections:
            detection_center = centroid(detection["bbox"])
            best_track_id = None
            best_distance = None

            for track_id, track in self._tracks.items():
                if track_id in matched_ids:
                    continue
                distance = hypot(detection_center[0] - track.center[0], detection_center[1] - track.center[1])
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is None or (best_distance is not None and best_distance > 120):
                track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[track_id] = TrackState(
                    track_id=track_id,
                    bbox=detection["bbox"],
                    label=detection["label"],
                    confidence=detection["confidence"],
                )
                matched_ids.add(track_id)
                continue

            track = self._tracks[best_track_id]
            track.movement_score = best_distance or 0.0
            track.bbox = detection["bbox"]
            track.label = detection["label"]
            track.confidence = detection["confidence"]
            track.last_seen_ts = time()
            track.missing_frames = 0
            matched_ids.add(best_track_id)

        exited_tracks: list[TrackState] = []
        for track_id, track in list(self._tracks.items()):
            if track_id in matched_ids:
                continue
            track.missing_frames += 1
            if track.missing_frames >= self.max_missing_frames:
                exited_tracks.append(track)
                del self._tracks[track_id]

        return list(self._tracks.values()), exited_tracks

    def is_moving(self, track: TrackState) -> bool:
        return track.movement_score >= self.motion_threshold
