# Face Surveillance for University Lab

Real-time face surveillance system for a university lab environment.

This project identifies registered users, tracks who is currently inside the lab, stores entry/exit evidence images, and exposes a fast presence API that can be consumed by another access-control system.

The main application code lives in [`r/`](./r).

## What It Does

- Detects and recognizes faces from a live camera feed
- Tracks people across frames and keeps current lab presence updated
- Stores face samples, sightings, and evidence images in PostgreSQL
- Publishes current presence state to Redis for fast external lookup
- Exposes REST APIs for:
  - current lab presence
  - active and recent sightings
  - person registry
  - evidence image retrieval

## Current Integration Model

This repo now focuses on identity and presence only.

- Each registered person has:
  - `name`
  - `username` as student ID
- External permission logic is intentionally left to another system
- The downstream system can read:
  - who is currently in the lab
  - which student IDs are present
  - whether a lab head is present

## Main Features

- Face registration with student ID
- Presence snapshot API: `GET /presence/current`
- Redis cache for current presence
- Entry and exit evidence:
  - face crop
  - full annotated frame
- Unknown identity handling for repeated unregistered visitors
- Thai timestamp overlay on full-frame evidence images

## Project Structure

```text
lab/
├─ README.md
└─ r/
   ├─ app/
   │  ├─ main.py
   │  ├─ recognition.py
   │  ├─ tracking.py
   │  ├─ models.py
   │  ├─ schemas.py
   │  ├─ database.py
   │  ├─ config.py
   │  └─ presence_cache.py
   ├─ face_data.py
   ├─ run_camera.py
   ├─ delete_person.py
   ├─ requirements.txt
   └─ docker-compose.yml
```

## Quick Start

### 1. Install dependencies

```bash
cd r
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure `.env`

Example:

```env
APP_NAME=face-surveillance
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/face_surveillance
CAMERA_SOURCE=0
CAMERA_NAME=front-door
FRAME_WIDTH=640
FRAME_HEIGHT=480

RECOGNITION_DISTANCE_THRESHOLD=65
UNKNOWN_MATCH_THRESHOLD=0.35
FACE_DETECTION_SCALE_FACTOR=1.2
FACE_DETECTION_MIN_NEIGHBORS=6
FACE_DETECTION_MIN_SIZE=72
RECOGNITION_CONSENSUS_FRAMES=3
RECOGNITION_HISTORY_SIZE=6

LOST_FRAMES_THRESHOLD=20
MOTION_DISTANCE_THRESHOLD=18

REDIS_URL=redis://localhost:6379/0
REDIS_PRESENCE_KEY=face_surveillance:presence:current
REDIS_PRESENCE_TTL_SECONDS=30

CASCADE_PATH=haarcascade_frontalface_alt.xml
RUN_CAMERA_ON_STARTUP=false
SHOW_CAMERA_WINDOW=true
```

### 3. Start infrastructure

If using Docker:

```bash
docker compose up --build
```

This starts:

- PostgreSQL
- Redis
- FastAPI app

### 4. Register a person

```bash
python face_data.py
```

You will be prompted for:

- person name
- student ID (`username`)

### 5. Run the camera pipeline

```bash
python run_camera.py
```

Or run the API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Key API Endpoints

- `GET /health`
- `GET /persons`
- `PUT /persons/{id}/access`
- `GET /persons/sample-stats`
- `GET /presence/current`
- `GET /sightings/active`
- `GET /sightings/recent?limit=20`
- `GET /sightings/{id}/entry-image`
- `GET /sightings/{id}/exit-image`
- `GET /sightings/{id}/entry-frame-image`
- `GET /sightings/{id}/exit-frame-image`
- `POST /camera/start`
- `POST /camera/stop`

## Example Presence Response

```json
{
  "camera_running": true,
  "camera_name": "front-door",
  "active_count": 2,
  "known_count": 1,
  "unknown_count": 1,
  "lab_head_present": true,
  "usernames_in_lab": ["65012345"],
  "identities": [
    {
      "sighting_id": 12,
      "track_id": 3,
      "person_id": 1,
      "username": "65012345",
      "label": "Alice",
      "camera_name": "front-door",
      "first_seen_at": "2026-04-15T07:10:12",
      "last_seen_at": "2026-04-15T07:10:18",
      "confidence": 64.2,
      "present_in_frame": true
    }
  ]
}
```

## Notes

- This system is suitable for supervised pilot use in a real lab.
- It is best used as an identity and presence layer, not as the final authority for permissions.
- For high-stakes production use, more testing and a stronger face-recognition backbone than LBPH are recommended.

## Status

Implemented:

- face registration
- real-time recognition and tracking
- current presence snapshot
- Redis presence cache
- evidence image storage
- unknown visitor grouping
- student-ID-based integration flow

Planned improvements:

- stronger embedding-based face recognition model
- richer deployment documentation
- optional summary-only endpoint for devices
