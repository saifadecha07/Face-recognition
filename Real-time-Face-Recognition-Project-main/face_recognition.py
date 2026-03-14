import numpy as np
import cv2
import os
import time
import requests
import csv
from datetime import datetime

# ==========================================
# 1. ตั้งค่า Telegram และโฟลเดอร์เก็บรูป
# ==========================================
TOKEN = '8220444201:AAHig8BTsCcurclOyVDGOIXgYzct8ITp50I'
CHAT_ID = '8696330483'
THRESHOLD = 9000  # lab 608 เทสเเล้วเเม่นที่9000
if not os.path.exists("unknown_faces"):
    os.makedirs("unknown_faces")

def send_telegram_photo(photo_path, message_text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        payload = {"chat_id": CHAT_ID, "caption": message_text}
        files = {"photo": photo}
        requests.post(url, data=payload, files=files)

# ==========================================
# 2. ฟังก์ชันสำหรับบันทึกประวัติลง CSV
# ==========================================
def log_to_csv(name):
    # เขึยนไฟล์ CSV เก็บชื่อ วัน เวลา
    with open("lab_attendance.csv", "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        writer.writerow([name, date_str, time_str])
        print(f"📝 บันทึกประวัติ: {name} เข้าแล็บเวลา {time_str}")

########## KNN CODE (ปรับปรุงใหม่) ############
def distance(v1, v2):
    return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])
        
    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    
    # ดึงค่าระยะห่างที่น้อยที่สุด (ใกล้เคียงที่สุด) ออกมาด้วย
    min_dist = dk[0][0] 
    
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    
    # Return ทั้ง Label ที่ทายได้ และระยะห่าง
    return output[0][index], min_dist
################################

# โหลด Cascade และ Dataset (โค้ดเดิมของคุณ)
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
dataset_path = "./face_dataset/"

face_data = []
labels = []
class_id = 0
names = {}

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
trainset = np.concatenate((face_dataset, face_labels), axis=1)

# ==========================================
# 3. ตัวแปรสำหรับระบบหน่วงเวลา (Cooldown)
# ==========================================
last_seen = {}
COOLDOWN_TIME = 60 # หน่วงเวลา 60 วินาที 

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if ret == False:
        continue
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for face in faces:
        x, y, w, h = face
        offset = 5
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        # ส่งไปเข้าโมเดล KNN เพื่อทายผลและดูระยะห่าง
        out, min_dist = knn(trainset, face_section.flatten())
        
        # เช็คว่าเป็นคนแปลกหน้าไหม
        if min_dist > THRESHOLD:
            predicted_name = "Unknown"
        else:
            predicted_name = names[int(out)]

        # --- ส่วนจัดการแจ้งเตือนและเก็บข้อมูล ---
        current_time = time.time()
        
        # เช็คหน่วงเวลา ป้องกันส่งรัวๆ
        if predicted_name not in last_seen or (current_time - last_seen[predicted_name] > COOLDOWN_TIME):
            last_seen[predicted_name] = current_time # อัปเดตเวลาล่าสุด
            
            log_to_csv(predicted_name) # เซฟลง CSV
            
            # ถ้าเป็นคนแปลกหน้า ให้เซฟรูปและส่งเข้า Telegram
            if predicted_name == "Unknown":
                filename = f"unknown_faces/stranger_{int(current_time)}.jpg"
                cv2.imwrite(filename, frame) # แคปทั้งเฟรม (หรือถ้าอยากได้แค่หน้าเปลี่ยน frame เป็น face_section)
                
                alert_msg = f"🚨 แจ้งเตือน: พบคนแปลกหน้าเข้าห้องแล็บ!\nเวลา: {datetime.now().strftime('%H:%M:%S')}"
                send_telegram_photo(filename, alert_msg)
                print("📲 ส่งแจ้งเตือน Telegram สำเร็จ!")

        # --- ส่วนแสดงผลบนหน้าจอ ---
        # สีของกรอบ: สีแดง (Unknown), สีเขียว (รู้จัก)
        color = (0, 0, 255) if predicted_name == "Unknown" else (0, 255, 0)
        
        cv2.putText(frame, predicted_name, (x, y-10), font, 1, color, 2, cv2.LINE_AA)
        # ถ้าอยากดูค่า distance ให้ uncomment บรรทัดล่างนี้ตอนจูนค่า
        # cv2.putText(frame, f"Dist: {int(min_dist)}", (x, y+h+30), font, 0.6, (255,255,0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Lab Security System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()