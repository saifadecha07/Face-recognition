# Face Surveillance Stack

ระบบจดจำใบหน้าสำหรับรันบนเครื่องของคุณเองหรือบนเซิร์ฟเวอร์ส่วนตัว  
ออกแบบให้เป็น **backend สำหรับระบบ gesture control** — รู้ว่าใครอยู่ในเฟรม รู้จักไหม และมีสิทธิ์ใช้ gesture หรือไม่

สิ่งที่ระบบทำได้:

- ตรวจจับและจดจำใบหน้าแบบ real-time
- track คนเดิมข้ามหลายเฟรม และบันทึกสถานะ `entered`, `active`, `exited`
- ประเมินการเคลื่อนไหวของแต่ละ track
- เก็บข้อมูลทั้งหมดไว้ใน PostgreSQL
- เปิด REST API สำหรับดูสถานะล่าสุดและตัดสินสิทธิ์ gesture

---

## โครงสร้างไฟล์

```
app/
  main.py          FastAPI app, camera runtime, REST endpoints
  recognition.py   face detection, recognition (LBPH), tracking, sync DB
  tracking.py      centroid tracker ข้ามเฟรม
  models.py        SQLAlchemy models (Person, FaceSample, Sighting)
  schemas.py       Pydantic schemas สำหรับ API
  database.py      engine, session, schema migration อัตโนมัติ
  config.py        settings จาก .env
face_data.py       ลงทะเบียนใบหน้าใหม่เข้า PostgreSQL
delete_person.py   ลบคนออกจากระบบ
run_camera.py      รัน pipeline กล้องโดยตรง (ไม่ใช้ API)
```

---

## ติดตั้งและเริ่มใช้งาน

### 1. เตรียม environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. ตั้งค่า .env

คัดลอก `.env.example` แล้วแก้ค่า:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/face_surveillance
CAMERA_SOURCE=0              # 0 = webcam, หรือ rtsp://...
CAMERA_NAME=front-door
RECOGNITION_THRESHOLD=85.0
SHOW_CAMERA_WINDOW=true      # false ถ้ารันบนเซิร์ฟเวอร์ไม่มีจอ
RUN_CAMERA_ON_STARTUP=false  # true ถ้าต้องการให้กล้องเริ่มพร้อม API
```

### 3. ลงทะเบียนใบหน้า

รันครั้งแรกก่อนใช้งานจริง หรือเมื่อต้องการเพิ่มคนใหม่:

```bash
python face_data.py
```

ระบบจะถาม:
```
Enter person name: alice
Enter role [lab_head/admin/supervisor/staff/user/guest] (default user): lab_head
Enable gesture control for this person? [y/N]: y
```

จากนั้นเปิดกล้อง มองกล้องนิ่งๆ กด `q` เมื่อเก็บตัวอย่างพอแล้ว (แนะนำ 30–50 ภาพ)

> ถ้าชื่อซ้ำกับคนที่มีอยู่แล้ว ระบบจะ **อัปเดต** role/permission และ **แทนที่** face samples ของคนนั้น ไม่สร้างซ้ำ

### 4. รัน API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

เปิด `http://localhost:8000` ดู dashboard  
เปิด `http://localhost:8000/docs` ดู API docs อัตโนมัติ

### 5. เริ่ม/หยุดกล้องผ่าน API

```bash
# เริ่มกล้อง
curl -X POST http://localhost:8000/camera/start

# หยุดกล้อง
curl -X POST http://localhost:8000/camera/stop
```

หรือตั้ง `RUN_CAMERA_ON_STARTUP=true` ใน `.env` เพื่อให้เริ่มอัตโนมัติตอนรัน API

---

## Docker

```bash
docker compose up --build
```

API จะเปิดที่ `http://localhost:8000`

หมายเหตุ Docker:
- ถ้าไม่มีจอ ตั้ง `SHOW_CAMERA_WINDOW=false`
- ถ้าต่อ USB webcam ต้อง map device ใน `docker-compose.yml`: `devices: ["/dev/video0:/dev/video0"]`
- แนะนำใช้ IP camera / RTSP แทนสำหรับ production

---

## จัดการคนในระบบ

### ดูรายชื่อทั้งหมด

```bash
curl http://localhost:8000/persons
```

### แก้ role หรือสิทธิ์ gesture

```bash
curl -X PUT http://localhost:8000/persons/1/access \
  -H "Content-Type: application/json" \
  -d '{"role": "lab_head", "gesture_control_enabled": true}'
```

### ลบคนออกจากระบบ

```bash
python delete_person.py
# Enter person name to delete: alice
```

ระบบจะลบ face samples และ unlink sightings ที่เกี่ยวข้องให้อัตโนมัติ

### คนที่ไม่รู้จักวนอยู่ในแล็ป

ถ้ามีคนที่ระบบไม่รู้จักปรากฏซ้ำบ่อย ควร **ลงทะเบียนเข้าระบบ** แม้ยังไม่รู้ชื่อจริง:

```
Enter person name: unknown_a
Enter role: guest
Enable gesture control for this person? [y/N]: N
```

วิธีนี้ทำให้ระบบ track คนเดิมต่อเนื่องโดยไม่สร้าง sighting row ซ้ำทุกครั้งที่ออกแล้วเข้าใหม่

---

## REST API

| Method | Endpoint | คำอธิบาย |
|--------|----------|-----------|
| `GET` | `/health` | สถานะ API และกล้อง |
| `GET` | `/persons` | รายชื่อทุกคนในระบบ |
| `PUT` | `/persons/{id}/access` | แก้ role และ gesture permission |
| `GET` | `/sightings/active` | Sightings ที่กำลัง active อยู่ตอนนี้ |
| `GET` | `/sightings/recent?limit=50` | Sightings ล่าสุด |
| `GET` | `/access/gesture/current` | Snapshot ทุกคนในเฟรมพร้อมสิทธิ์ |
| `GET` | `/access/gesture/current?authorized_only=true` | เฉพาะคนที่ผ่านสิทธิ์ |
| `GET` | `/access/gesture/controller` | เลือก controller หลักคนเดียวอัตโนมัติ |
| `POST` | `/camera/start` | เริ่มกล้อง |
| `POST` | `/camera/stop` | หยุดกล้อง |

---

## Gesture Control Integration

ระบบนี้เป็น **identity + access layer** สำหรับ gesture system ภายนอก  
ขั้นตอนการ integrate:

### ขั้นตอน

```
1. รัน Face Surveillance API (ระบบนี้)
2. รัน gesture model ของคุณ (MediaPipe / custom)
3. ก่อน process gesture ทุกครั้ง → ถาม API ว่าใครมีสิทธิ์
4. ถ้ามี controller → classify gesture → ส่งคำสั่งไปอุปกรณ์
```

### ตัวอย่าง polling loop (Python)

```python
import requests
import time

API = "http://localhost:8000"

while True:
    resp = requests.get(f"{API}/access/gesture/controller").json()

    if resp["controller_selected"]:
        ctrl = resp["controller"]
        print(f"Controller: {ctrl['label']} (role={ctrl['role']})")
        # ส่ง frame ไปให้ gesture model ของคุณต่อ
        # gesture = your_gesture_model.predict(frame)
        # control_device(gesture)
    else:
        print(f"No controller: {resp['selection_reason']}")

    time.sleep(0.1)  # poll ทุก 100ms
```

### GET /access/gesture/current

ดู snapshot ทุกคนที่ active อยู่ในเฟรม

```json
{
  "camera_running": true,
  "camera_name": "front-door",
  "active_count": 2,
  "authorized_count": 1,
  "identities": [
    {
      "sighting_id": 12,
      "track_id": 3,
      "label": "alice",
      "person_id": 1,
      "role": "lab_head",
      "gesture_control_enabled": true,
      "access_granted": true,
      "access_reason": "granted_for_role:lab_head",
      "present_in_frame": true,
      "confidence": 91.3,
      "movement_score": 4.2,
      "last_x": 210,
      "last_y": 80,
      "last_w": 120,
      "last_h": 120
    },
    {
      "sighting_id": 13,
      "track_id": 4,
      "label": "Unknown",
      "person_id": null,
      "role": null,
      "gesture_control_enabled": false,
      "access_granted": false,
      "access_reason": "unknown_person",
      "present_in_frame": true
    }
  ]
}
```

`access_reason` ที่เป็นไปได้:
- `granted_for_role:<role>` — ผ่านสิทธิ์
- `gesture_control_disabled` — รู้จักคนแต่ยังไม่เปิดสิทธิ์
- `unknown_person` — ไม่รู้จักคนนี้

### GET /access/gesture/controller

ระบบเลือก controller หลักคนเดียวให้อัตโนมัติ เรียงตาม role priority → confidence → เวลาที่เห็นล่าสุด

ลำดับ role (สูงไปต่ำ): `lab_head` › `admin` › `supervisor` › `staff` › `user` › `guest`

```json
{
  "camera_running": true,
  "camera_name": "front-door",
  "controller_selected": true,
  "selection_reason": "selected_by_role_priority:lab_head",
  "controller": {
    "sighting_id": 12,
    "track_id": 3,
    "label": "alice",
    "person_id": 1,
    "role": "lab_head",
    "gesture_control_enabled": true,
    "access_granted": true,
    "access_reason": "granted_for_role:lab_head",
    "present_in_frame": true
  },
  "candidates": [...]
}
```

`selection_reason` ที่เป็นไปได้:
- `selected_by_role_priority:<role>` — พบ controller
- `no_authorized_identity_in_frame` — มีคนในเฟรมแต่ไม่มีใครมีสิทธิ์
- `no_identity_in_frame` — ไม่มีคนในเฟรมเลย

---

## ค่า Config ที่ปรับได้

| ตัวแปร | ค่าเริ่มต้น | คำอธิบาย |
|--------|-------------|-----------|
| `DATABASE_URL` | `postgresql+psycopg://...` | PostgreSQL connection string |
| `CAMERA_SOURCE` | `0` | 0 = webcam, path หรือ RTSP URL |
| `CAMERA_NAME` | `front-door` | ชื่อกล้องที่บันทึกใน log |
| `RECOGNITION_THRESHOLD` | `85.0` | ต่ำ = ยอมรับง่าย, สูง = เข้มงวด (0–100) |
| `LOST_FRAMES_THRESHOLD` | `20` | กี่เฟรมที่ไม่เห็นหน้าแล้วถือว่าออกไป |
| `MOTION_DISTANCE_THRESHOLD` | `18.0` | pixel ที่เคลื่อนไปต่อเฟรมถือว่า "กำลังเคลื่อนที่" |
| `FRAME_WIDTH` | `640` | ความกว้าง frame |
| `FRAME_HEIGHT` | `480` | ความสูง frame |
| `RUN_CAMERA_ON_STARTUP` | `false` | เริ่มกล้องพร้อมกับ API |
| `SHOW_CAMERA_WINDOW` | `true` | แสดงหน้าต่าง preview (ปิดถ้าไม่มีจอ) |

---

## สถานะข้อมูล

- ข้อมูลการตรวจพบและประวัติเข้าออกทั้งหมดเก็บใน PostgreSQL
- face samples เก็บเป็น binary ใน PostgreSQL ไม่มีไฟล์ภายนอก
- เมื่อรันครั้งแรก schema จะถูกสร้างอัตโนมัติ
- ถ้ามีฐานข้อมูลเก่าที่ยังไม่มีคอลัมน์ `role` และ `gesture_control_enabled` ระบบจะ migrate ให้อัตโนมัติตอน startup
