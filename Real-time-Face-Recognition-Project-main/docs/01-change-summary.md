# Change Summary

## สิ่งที่รื้อออก

- เอา logic แจ้งเตือน Telegram ออกจากระบบทั้งหมด
- เลิกใช้การบันทึก log แบบ CSV เป็นแกนหลักของระบบ
- เลิกผูก flow การทำงานไว้กับสคริปต์เดี่ยวไฟล์เดียว

## สิ่งที่เพิ่มเข้ามา

- สร้างโครงสร้างแอปใหม่ในโฟลเดอร์ `app/`
- เพิ่มระบบ `FaceRecognizerService` สำหรับ:
  - detect ใบหน้า
  - recognize ว่าเป็นใคร
  - track คนเดิมข้ามหลายเฟรม
  - รู้ว่าคนยังอยู่ในกล้อง หรือออกจากกล้องแล้ว
- เพิ่มฐานข้อมูล PostgreSQL ผ่าน SQLAlchemy
- เพิ่ม API ด้วย FastAPI
- เพิ่มไฟล์สำหรับ deploy แบบ self-hosted ด้วย Docker และ Docker Compose
- เพิ่มไฟล์ config `.env.example`

## ไฟล์สำคัญที่เปลี่ยน

- `app/recognition.py`
  - แกน face detection + recognition + tracking + sync database
- `app/tracking.py`
  - ติดตาม centroid ของคนในภาพ และตัดสินว่า exited เมื่อหายไปเกิน threshold
- `app/models.py`
  - ตาราง `persons` และ `sightings`
- `app/main.py`
  - API และ runtime สำหรับ start/stop กล้อง
- `face_data.py`
  - ใช้เก็บ dataset ใบหน้าของแต่ละคน
- `face_recognition.py`
  - เปลี่ยนเป็น entrypoint ไปเรียก runtime ใหม่
- `docker-compose.yml`
  - ยก PostgreSQL และ app ขึ้นพร้อมกันได้

## แนวคิดข้อมูลใหม่

จากเดิม:

- เห็นหน้า
- ทายชื่อ
- ส่ง Telegram

เป็น:

- เห็นหน้า
- ทายชื่อ
- track ว่าเป็นคนเดิมในเฟรมถัดไปหรือไม่
- บันทึกว่าเข้ามาแล้ว (`entered`)
- อัปเดตว่ายังอยู่ (`visible`)
- บันทึกว่าออกจากกล้องแล้ว (`exited`)
- เก็บลง PostgreSQL เพื่อ query ย้อนหลังได้
