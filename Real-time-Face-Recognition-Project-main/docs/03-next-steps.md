# Next Steps

## วิธีใช้งานคร่าว ๆ

1. ติดตั้ง dependency

```powershell
pip install -r requirements.txt
```

2. คัดลอก `.env.example` เป็น `.env`

3. ตั้งค่าอย่างน้อย

- `DATABASE_URL`
- `CAMERA_SOURCE`
- `SHOW_CAMERA_WINDOW`
- `RUN_CAMERA_ON_STARTUP`

4. เก็บ dataset คนที่ระบบต้องรู้จัก

```powershell
python face_data.py
```

5. รันระบบ

แบบ API:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

แบบกล้องตรง:

```powershell
python run_camera.py
```

แบบ Docker:

```powershell
docker compose up --build
```

## สิ่งที่ควรทำต่อถ้าจะใช้จริง

- เปลี่ยนจาก Haar Cascade + LBPH ไปใช้ face embedding model ที่แม่นกว่า
- เพิ่ม migration tool เช่น Alembic
- เพิ่ม authentication ให้ API
- เพิ่มหน้า web dashboard
- รองรับหลายกล้องพร้อมกัน
- เก็บ snapshot หรือ clip เมื่อเจอ `Unknown`
- เพิ่มระบบค้นหาย้อนหลังตามชื่อและช่วงเวลา

## ข้อจำกัดปัจจุบัน

- recognition ตอนนี้ยังเป็นระดับ prototype ที่เหมาะกับงานต้นแบบมากกว่างาน production เข้ม ๆ
- tracking ใช้ centroid tracker ซึ่งเหมาะกับฉากไม่ซับซ้อนมาก
- ยังไม่มี frontend dashboard
- ยังไม่ได้เพิ่มระบบ user login หรือ role
- ใน environment นี้ยังไม่ได้ run test จริง เพราะ interpreter ถูกจำกัดใน sandbox
