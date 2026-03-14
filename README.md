ระบบจดจำใบหน้า (Face Recognition System) ที่พัฒนาด้วย OpenCV Python  
ออกแบบมาเพื่อใช้งานภายในห้องปฏิบัติการของคณะวิศวกรรมศาสตร์  
มหาวิทยาลัยธรรมศาสตร์ โดยมีเป้าหมายเพื่อช่วยตรวจสอบตัวตนผู้ใช้งาน  
ฟีเจอร์เด่น (Key Features)
Real-time Face Recognition:** ประมวลผลวิดีโอแบบสดๆ และตรวจจับใบหน้าด้วย Haar Cascade Classifier 
KNN Classification:** ใช้โมเดล Machine Learning (KNN) จำแนกบุคคล 
Logging:** บันทึกประวัติการเข้าห้องแล็บ (ชื่อ, วันที่, เวลา) ลงไฟล์ `.csv` โดยอัตโนมัติ
แจ้งเตือน(Unknown Face) พร้อมส่งภาพถ่ายหลักฐานเข้า Telegram แบบ Real-time ผ่าน Telegram Bot API


https://youtube.com/shorts/AoJ6Xv8MMl8?si=2vpytsE-24WbB_KN

เครื่องมือและเทคโนโลยีที่ใช้ (Tech Stack)
* **Language:** Python
* **Computer Vision:** OpenCV (`cv2`)
* **Data Computation:** NumPy
* **API Integration:** Telegram Bot API (`requests`)
* **Data Storage:** CSV / File System


