import cv2
import threading
import numpy as np
from ultralytics import YOLO
import time
import os
import psutil
import torch

# ==== CẤU HÌNH ====
RTSP_URLS = [
    "rtsp://admin:WSLRQC@192.168.0.106:554/ch1/main",
    "rtsp://admin:NXKPHU@192.168.0.119:554/ch1/main",
    "rtsp://admin:WNRDVL@192.168.0.118:554/ch1/main",
    "rtsp://admin:JRGMMV@192.168.0.116:554/ch1/main",
    "rtsp://admin:LUXHLR@192.168.0.110:554/ch1/main",
    "rtsp://admin:TIJEQB@192.168.0.111:554/ch1/main",
]

NUM_ROWS, NUM_COLS = 2, 3
FRAME_W, FRAME_H = 640, 480
MODEL_PATH = "yolov8n-pose.pt"
DURATION = 1800  # 30 phút = 1800 giây

# ==== LOAD MODEL ====
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH).to(device)
print("✅ Đang sử dụng thiết bị:", model.device)

# ==== GStreamer pipeline ====
def gstreamer_pipeline(rtsp_url, width, height):
    return (
        f'rtspsrc location={rtsp_url} latency=0 ! '
        'rtph265depay ! h265parse ! nvv4l2decoder ! '
        f'nvvidconv ! video/x-raw, width={width}, height={height}, format=BGRx ! '
        'videoconvert ! video/x-raw, format=BGR ! '
        'appsink drop=1'
    )

# ==== THREAD CAMERA ====
class CameraStream(threading.Thread):
    def __init__(self, rtsp_url):
        super().__init__()
        self.pipeline = gstreamer_pipeline(rtsp_url, FRAME_W, FRAME_H)
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        self.frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame.copy()

    def get_frame(self):
        with self.lock:
            return self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()

# ==== KHỞI TẠO STREAMS ====
streams = [CameraStream(url) for url in RTSP_URLS]
for s in streams:
    s.start()

# ==== BENCHMARK ====
latencies = []
frame_count = 0
print(f"\n🚀 Đang benchmark {len(RTSP_URLS)} camera trong {DURATION} giây...\n")

start_time = time.time()

try:
    while time.time() - start_time < DURATION:
        frames = [s.get_frame() for s in streams]
        grid_rows = [np.hstack(frames[i*NUM_COLS:(i+1)*NUM_COLS]) for i in range(NUM_ROWS)]
        grid_frame = np.vstack(grid_rows)

        t0 = time.time()
        results = model(grid_frame, imgsz=640, conf=0.5, verbose=False)[0]
        t1 = time.time()

        latency_ms = (t1 - t0) * 1000
        latencies.append(latency_ms)
        frame_count += 1

except KeyboardInterrupt:
    print("🛑 Dừng sớm do người dùng nhấn Ctrl+C.")

end_time = time.time()

# ==== DỪNG STREAM ====
for s in streams:
    s.stop()

# ==== THỐNG KÊ ====
avg_latency = np.mean(latencies)
fps = frame_count / (end_time - start_time)
model_size = os.path.getsize(MODEL_PATH) / (1024 ** 2)  # MB
ram_used = psutil.virtual_memory().used / (1024 ** 2)   # MB

# ==== IN RA KẾT QUẢ ====
print("\n🎯 Benchmark Kết Quả:")
print(f"Model: {MODEL_PATH}")
print(f"Device: {model.device}")
print(f"FPS: {fps:.2f}")
print(f"Latency: {avg_latency:.2f} ms")
print(f"Model size: {model_size:.2f} MB")
print(f"RAM used: {ram_used:.2f} MB")

# ==== GHI LOG (append) ====
log_line = f"{MODEL_PATH},{FRAME_W}x{FRAME_H},{fps:.2f},{avg_latency:.2f},{model_size:.2f},{ram_used:.2f},{len(RTSP_URLS)},{model.device}\n"
log_path = "benchmark_multi_log.csv"

write_header = not os.path.exists(log_path)

with open(log_path, "a") as f:
    #if write_header:
        #f.write("Model,Image Size,FPS,Latency(ms),Model Size(MB),RAM Used(MB),Cameras,Device\n")
    f.write(log_line)

