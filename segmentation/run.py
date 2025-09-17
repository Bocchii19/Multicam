import cv2
import threading
import numpy as np
from ultralytics import YOLO
import time
import torch

# ==== CẤU HÌNH ====
RTSP_URLS = [
"rtsp://admin:YDVFNP@192.168.0.108:554/ch1/main",
"rtsp://admin:PBPBND@192.168.0.119:554/ch1/main",
"rtsp://admin:IFPREC@192.168.0.101:554/ch1/main",
"rtsp://admin:KAETPH@192.168.0.110:554/ch1/main",

]

NUM_ROWS, NUM_COLS = 2, 2
FRAME_W, FRAME_H = 640, 480
SHOW_RESULT = True
SAVE_OUTPUT = False  # Bật nếu muốn ghi video
FLIP_MODE = 0  # 0 = lật dọc, 1 = lật ngang, -1 = lật cả 2

# ==== LOAD YOLOv8 SEG MODEL ====
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n-seg.pt").to(device)
print("✅ Thiết bị đang sử dụng:", model.device)

# ==== GStreamer pipeline ====
def gstreamer_pipeline(rtsp_url, width, height):
    return (
        f'rtspsrc location={rtsp_url} latency=0 ! '
        'rtph265depay ! h265parse ! nvv4l2decoder ! '
        f'nvvidconv ! video/x-raw, width={width}, height={height}, format=BGRx ! '
        'videoconvert ! video/x-raw, format=BGR ! '
        'appsink drop=1'
    )

# ==== CAMERA THREAD ====
class CameraStream(threading.Thread):
    def __init__(self, rtsp_url):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.pipeline = gstreamer_pipeline(rtsp_url, FRAME_W, FRAME_H)
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        self.frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        self.lock = threading.Lock()
        self.running = True

    def reconnect(self):
        print(f"🔄 Reconnecting to {self.rtsp_url} ...")
        self.cap.release()
        time.sleep(1)  # nghỉ một chút trước khi kết nối lại
        self.pipeline = gstreamer_pipeline(self.rtsp_url, FRAME_W, FRAME_H)
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"⚠️ Mất kết nối {self.rtsp_url}")
                self.reconnect()
                continue

            frame = cv2.flip(frame, FLIP_MODE)
            with self.lock:
                self.frame = frame.copy()

    def get_frame(self):
        with self.lock:
            return self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()
        
# ==== KHỞI TẠO CAMERA STREAMS ====
streams = [CameraStream(url) for url in RTSP_URLS]
for s in streams:
    s.start()

# ==== VIDEO WRITER (tuỳ chọn) ====
if SAVE_OUTPUT:
    out = cv2.VideoWriter("output_pose.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20,
                          (FRAME_W * NUM_COLS, FRAME_H * NUM_ROWS))

# ==== VÒNG LẶP CHÍNH ====
prev_time = time.time()

try:
    while True:
        # ==== GHÉP FRAME TỪ CÁC CAMERA ====
        frames = [s.get_frame() for s in streams]
        grid_rows = [np.hstack(frames[i*NUM_COLS:(i+1)*NUM_COLS]) for i in range(NUM_ROWS)]
        grid_frame = np.vstack(grid_rows)

        # ==== YOLO SEGMENTATION INFERENCE ====
        results = model(grid_frame, imgsz=640, conf=0.5, verbose=False)[0]
        annotated = results.plot()

        # ==== TÍNH FPS ====
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        # ==== VẼ FPS ====
        cv2.putText(annotated, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # ==== HIỂN THỊ ====
        if SHOW_RESULT:
            cv2.imshow("YOLOv8-Seg - Grid View", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # ==== GHI VIDEO ====
        if SAVE_OUTPUT:
            out.write(annotated)

except KeyboardInterrupt:
    print("🛑 Dừng chương trình bằng KeyboardInterrupt.")

# ==== DỌN DẸP ====
for s in streams:
    s.stop()
if SAVE_OUTPUT:
    out.release()
cv2.destroyAllWindows()

