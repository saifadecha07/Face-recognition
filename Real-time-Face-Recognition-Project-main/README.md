# Face Surveillance Stack

โปรเจกต์นี้ถูกปรับจากสคริปต์ OpenCV + Telegram เดิม ไปเป็นระบบกล้องสำหรับใช้งานจริงที่สามารถ:

- ระบุว่าใบหน้าที่เห็นเป็นใครแบบต่อเนื่องขณะยังอยู่ในกล้อง
- track คนเดิมข้ามหลายเฟรม และบันทึกสถานะ `entered`, `visible`, `exited`
- ตรวจจับการเคลื่อนไหวของคนที่กำลังอยู่ในภาพ
- เก็บข้อมูลลง PostgreSQL แทน CSV
- รันได้ทั้ง local และ self-hosted ผ่าน Docker / Docker Compose

## โครงสร้างหลัก

- `app/main.py` FastAPI API + camera runtime
- `app/recognition.py` face detection, recognition, tracking, database sync
- `app/tracking.py` centroid tracker สำหรับรู้ว่าคนยังอยู่หรือออกจากกล้องแล้ว
- `face_data.py` เก็บ dataset ใบหน้าของคนที่รู้จัก
- `run_camera.py` รัน pipeline กล้องโดยตรง

## การติดตั้งแบบ local

1. สร้าง virtualenv และติดตั้ง dependencies

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. สร้างไฟล์ `.env` จาก `.env.example`

3. ถ่าย dataset ของแต่ละคน

```powershell
python face_data.py
```

4. รัน PostgreSQL เอง หรือใช้ Docker Compose ด้านล่าง

5. รันกล้อง

```powershell
python run_camera.py
```

6. หรือรัน API

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API หลัก

- `GET /health`
- `GET /persons`
- `GET /sightings/active`
- `GET /sightings/recent`
- `POST /camera/start`
- `POST /camera/stop`

## Docker / PostgreSQL

```powershell
docker compose up --build
```

ค่าปกติจะเปิด API ที่ `http://localhost:8000`

หมายเหตุ:

- ถ้าจะให้ container ต่อกล้องจริง ต้อง map device หรือใช้ RTSP stream ใน `CAMERA_SOURCE`
- ถ้ารันบน server ไม่มีจอ ให้ตั้ง `SHOW_CAMERA_WINDOW=false`
- ถ้าใช้ IP camera/RTSP ให้กำหนด `CAMERA_SOURCE=rtsp://...`

## ตัวอย่าง event ที่ระบบรู้ได้

- คนใหม่เข้ากล้องครั้งแรก: บันทึก `entered`
- คนเดิมยังอยู่ในกล้อง: อัปเดต `visible`
- centroid ขยับเกิน threshold: ถือว่า `moving`
- หายจากภาพเกินจำนวนเฟรมที่กำหนด: ปิด session เป็น `exited`
