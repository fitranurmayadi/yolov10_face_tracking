import cv2
import time
import psutil
import GPUtil
from ultralytics import YOLO
import serial

# Sesuaikan dengan port serial yang terhubung ke Arduino Anda
#serial_port = '/dev/ttyUSB0'  # Contoh untuk Linux
serial_port = 'COM9'  # Contoh untuk Windows
baud_rate = 9600  # Sesuaikan dengan baud rate pada program Arduino

# Inisialisasi koneksi serial
ser = serial.Serial(serial_port, baud_rate, timeout=1)

# Fungsi untuk mengirim data sudut ke Arduino
def send_servo_angles(angleX, angleY):
    if 0 <= angleX <= 180 and 0 <= angleY <= 180:
        command = f"x={angleX} y={angleY}\n"
        ser.write(command.encode())
        print(f"Sent: {command.strip()}")
    else:
        print("Invalid angle. Please enter values between 0 and 180.")

# Fungsi untuk menghitung sudut servo berdasarkan selisih posisi titik tengah
def calculate_servo_angle(delta, current_angle, step=1, max_angle=180):
    new_angle = current_angle + step * delta
    new_angle = max(0, min(new_angle, max_angle))  # Batasi nilai sudut antara 0 dan max_angle
    return new_angle

# Muat model YOLOv8
model = YOLO('yolov8n-face.pt', verbose=False)  # Pastikan Anda memiliki model YOLOv8 yang dilatih untuk deteksi wajah

# Inisialisasi kamera (biasanya 0 adalah kamera default, Anda mungkin perlu mengubahnya jika ada beberapa kamera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    exit()

# Inisialisasi sudut servo awal
servo_angle_x = 90
servo_angle_y = 90
send_servo_angles(servo_angle_x, servo_angle_y)  # Set sudut awal

prev_frame_time = 0
new_frame_time = 0

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    
    if not ret:
        print("Error: Tidak dapat membaca frame dari kamera.")
        break
    
    # Deteksi wajah menggunakan YOLOv8
    results = model(frame)

    # Proses hasil deteksi
    if results:
        for result in results:
            boxes = result.boxes  # Akses bounding box dari hasil deteksi
            if boxes:
                for box in boxes:
                    # Dapatkan koordinat bounding box dan konversi ke integer
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                    conf = box.conf.item()  # Dapatkan confidence score

                    # Gambar persegi panjang di sekitar wajah yang terdeteksi
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)
                    
                    # Hitung titik tengah bounding box
                    center_x = (x_min + x_max) // 2
                    center_y = (y_min + y_max) // 2

                    # Hitung titik tengah frame
                    frame_center_x = frame.shape[0] // 2
                    frame_center_y = frame.shape[1] // 2
                    
                    # Hitung selisih antara titik tengah bounding box dan titik tengah frame
                    delta_x = frame_center_x - center_x
                    delta_y = frame_center_y - center_y

                    # Konversi delta menjadi pergerakan sudut servo
                    if delta_x != 0:
                        step_x = 1 if delta_x > 0 else -1
                    else:
                        step_x = 0

                    if delta_y != 0:
                        step_y = 1 if delta_y > 0 else -1
                    else:
                        step_y = 0

                    # Hitung sudut servo baru berdasarkan selisih
                    servo_angle_x = calculate_servo_angle(step_x, servo_angle_x)
                    servo_angle_y = calculate_servo_angle(step_y, servo_angle_y)

                    # Kirim sudut servo ke Arduino
                    send_servo_angles(servo_angle_x, servo_angle_y)
                    
                    # Tampilkan informasi bounding box dan titik tengah
                    debug_text = f"Box: ({x_min}, {y_min}), W: {x_max-x_min}, H: {y_max-y_min}, Center: ({center_x}, {center_y}), Servo: ({servo_angle_x}, {servo_angle_y})"
                    #print(debug_text)  # Tampilkan di console/serial monitor
                    
                    # Tampilkan informasi pada frame
                    cv2.putText(frame, debug_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
    
    # Hitung FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Dapatkan penggunaan CPU
    cpu_usage = psutil.cpu_percent()
    cpu_text = f"CPU: {cpu_usage:.2f}%"
    cv2.putText(frame, cpu_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Dapatkan penggunaan GPU (jika ada)
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_text = f"GPU: {gpu.load * 100:.2f}%"
        cv2.putText(frame, gpu_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Tampilkan frame dengan wajah yang terdeteksi dan data debug
    cv2.imshow('Face Detection', frame)
    
    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan setelah selesai
cap.release()
cv2.destroyAllWindows()
ser.close()
