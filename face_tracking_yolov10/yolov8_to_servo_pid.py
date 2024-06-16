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

# Fungsi PID Controller
class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error_x = 0
        self.prev_error_y = 0
        self.integral_x = 0
        self.integral_y = 0

    def compute(self, x, y, frame_width=1280, frame_height=720):
        error_x = self.calculate_error(x, 600, 680, frame_width, 32)
        error_y = self.calculate_error(y, 320, 400, frame_height, 18)

        # Compute integral and derivative for x
        self.integral_x += error_x
        derivative_x = error_x - self.prev_error_x

        # Compute integral and derivative for y
        self.integral_y += error_y
        derivative_y = error_y - self.prev_error_y

        # Compute output
        output_x = self.Kp * error_x + self.Ki * self.integral_x + self.Kd * derivative_x
        output_y = self.Kp * error_y + self.Ki * self.integral_y + self.Kd * derivative_y

        # Update previous errors
        self.prev_error_x = error_x
        self.prev_error_y = error_y

        return output_x, output_y

    def calculate_error(self, value, lower_bound, upper_bound, frame_size, divisions):
        division_size = frame_size // divisions
        center = (lower_bound + upper_bound) // 2

        # Error is 0 if within the bounds
        if lower_bound <= value <= upper_bound:
            return 0

        # Calculate error based on which division the value falls into
        for i in range(1, (divisions // 2) + 1):
            if center - i * division_size <= value < center - (i - 1) * division_size:
                return -i
            elif center + (i - 1) * division_size < value <= center + i * division_size:
                return i
        return 0  # Default error if out of expected range


# Muat model YOLOv8
model = YOLO('../yolov8n-face.pt', verbose=False)  # Pastikan Anda memiliki model YOLOv8 yang dilatih untuk deteksi wajah

# Inisialisasi kamera (biasanya 0 adalah kamera default, Anda mungkin perlu mengubahnya jika ada beberapa kamera)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    exit()

# Set resolusi kamera ke 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Variabel untuk menghitung FPS
prev_time = 0

# Fungsi untuk mendapatkan penggunaan GPU
def get_gpu_usage():
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        return gpu.load * 100  # Mengembalikan penggunaan GPU dalam persentase
    return 0


# Inisialisasi sudut servo awal
setpoint_servo_x = 90
setpoint_servo_y = 90
send_servo_angles(setpoint_servo_x, setpoint_servo_y)  # Set sudut awal

# Inisialisasi PID Controller
pid = PID(Kp=0.5, Ki=0.2, Kd=0.2)

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()
    
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
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 165, 255), 2)
                    
                    # Hitung titik tengah bounding box
                    center_x = (x_min + x_max) // 2
                    center_y = (y_min + y_max) // 2

                    # Hitung titik tengah frame
                    frame_center_x = frame.shape[1] // 2
                    frame_center_y = frame.shape[0] // 2
                    
                    # Hitung output PID
                    pid_output_x, pid_output_y = pid.compute(center_x, center_y)

                    # Update sudut servo berdasarkan output PID
                    servo_angle_x = setpoint_servo_x  - pid_output_x
                    servo_angle_y = setpoint_servo_y  - pid_output_y
                    # Kirim sudut servo ke Arduino
                    send_servo_angles(servo_angle_x, servo_angle_y)
                    #print()
                    # Tampilkan informasi bounding box dan titik tengah
                    debug_text = f"Center: ({center_x}, {center_y}), Servo : ({servo_angle_x}, {servo_angle_y}), PID : ({pid_output_x}, {pid_output_y})"
                    print(debug_text)  # Tampilkan di console/serial monitor
                    
                    # Tampilkan informasi pada frame
                    cv2.putText(frame, debug_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1, cv2.LINE_AA)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 165, 255), -1)

                    cv2.rectangle(frame, (600, 320), (680, 400), (0, 0, 255), 2)
                    cv2.circle(frame, (frame_center_x, frame_center_y), 5, (0, 0, 255), -1)


    
    # Dapatkan waktu sekarang
    curr_time = time.time()
    exec_time = curr_time - prev_time
    prev_time = curr_time
    fps = 1 / exec_time
    
    # Dapatkan penggunaan CPU dan GPU
    cpu_usage = psutil.cpu_percent()
    gpu_usage = get_gpu_usage()
    
    # Tambahkan teks ke frame
    cv2.putText(frame, 'Face Tracking Yolov8', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f'CPU Usage: {cpu_usage}%', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f'GPU Usage: {gpu_usage:.1f}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Tampilkan frame dengan wajah yang terdeteksi dan data debug
    cv2.imshow('Face Detection', frame)
    
    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan setelah selesai
cap.release()
cv2.destroyAllWindows()
ser.close()
