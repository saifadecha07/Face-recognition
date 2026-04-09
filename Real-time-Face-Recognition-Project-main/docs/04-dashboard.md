# Dashboard Overview

## สิ่งที่เพิ่มเข้ามา

เพิ่ม dashboard เว็บในตัวระบบ เพื่อดูสถานะกล้องและข้อมูลจาก PostgreSQL แบบไม่ต้องยิง API เอง

## ไฟล์ที่เพิ่ม

- `app/templates/dashboard.html`
  - โครงหน้า dashboard
- `app/static/dashboard.css`
  - สไตล์ของหน้า dashboard
- `app/static/dashboard.js`
  - ดึงข้อมูลจาก API และ refresh ทุก 5 วินาที

## ไฟล์ที่แก้

- `app/main.py`
  - mount static files
  - เพิ่ม route `/`
  - เพิ่มค่า `camera_running` ใน `/health`
- `requirements.txt`
  - เพิ่ม `jinja2`

## Dashboard ทำอะไรได้

- ดูสถานะว่ากล้องกำลังรันหรือไม่
- ดูจำนวนคนที่ยังอยู่ในกล้องตอนนี้
- ดูจำนวนคนที่ลงทะเบียนไว้ใน dataset
- ดูประวัติการพบล่าสุด
- สั่ง start/stop กล้องจากหน้าเว็บ

## หน้าเว็บอ่านข้อมูลจากไหน

dashboard ใช้ endpoint เดิมของระบบ:

- `GET /health`
- `GET /persons`
- `GET /sightings/active`
- `GET /sightings/recent?limit=20`
- `POST /camera/start`
- `POST /camera/stop`

## วิธีเข้าใช้งาน

เมื่อรัน API แล้ว เปิด:

```text
http://localhost:8000/
```

ถ้า deploy บน server ก็เปิด path `/` ของ host นั้นได้เลย
