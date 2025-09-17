import cv2
import threading
import numpy as np
from ultralytics import YOLO
import time
import torch
import os
import psutil
from datetime import datetime
import argparse

# ==== SHOW REUSLT ====
parser = argparse.ArgumentParser(description="YOLOv8 Pose benchmark với camera RTSP")
parser.add_argument('--show', action='store_true', help='Hiển thị video đầu ra')
parser.add_argument('--no-show', dest='show', action='store_false', help='Không hiển thị video')
parser.set_defaults(show=True)  # Mặc định là hiển thị
args = parser.parse_args()
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
SHOW_RESULT = args.show
SAVE_OUTPUT = False
DURATION = 30  # Giây để benchmark

# ==== LOAD MODEL ====
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH).to(device)
print("✅ Thiết bị đang dùng:", model.device)

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

# ==== KHỞI TẠO CAMERA STREAMS ====
streams = [CameraStream(url) for url in RTSP_URLS]
for s in streams:
    s.start()

# ==== VIDEO WRITER (tuỳ chọn) ====
if SAVE_OUTPUT:
    out = cv2.VideoWriter("output_pose.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20,
                          (FRAME_W * NUM_COLS, FRAME_H * NUM_ROWS))

# ==== VÒNG LẶP CHÍNH (BENCHMARK) ====
print(f"🚀 Bắt đầu benchmark trong {DURATION}s...")
latencies = []
frame_count = 0
start_time = time.time()

try:
    while time.time() - start_time < DURATION:
        frames = [s.get_frame() for s in streams]
        grid_rows = [np.hstack(frames[i * NUM_COLS:(i + 1) * NUM_COLS]) for i in range(NUM_ROWS)]
        grid_frame = np.vstack(grid_rows)

        t0 = time.time()
        results = model(grid_frame, imgsz=640, conf=0.5, verbose=False)[0]
        annotated = results.plot()
        t1 = time.time()

        latency = (t1 - t0) * 1000  # ms
        latencies.append(latency)
        frame_count += 1

        fps = 1.0 / (t1 - t0)
        cv2.putText(annotated, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        if SHOW_RESULT:
            cv2.imshow("YOLOv8 Pose - Grid View", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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

# ==== TÍNH TOÁN THỐNG KÊ ====
total_time = time.time() - start_time
avg_latency = np.mean(latencies)
avg_fps = frame_count / total_time
ram_used = psutil.virtual_memory().used / (1024**2)  # MB
model_size = os.path.getsize(MODEL_PATH) / (1024**2)

# ==== IN RA KẾT QUẢ ====
print("\n🎯 Kết quả benchmark:")
print(f"Model: {MODEL_PATH}")
print(f"Camera: {len(RTSP_URLS)}")
print(f"Frames: {frame_count}")
print(f"Avg FPS: {avg_fps:.2f}")
print(f"Avg Latency: {avg_latency:.2f} ms")
print(f"RAM: {ram_used:.2f} MB")
print(f"Model size: {model_size:.2f} MB")

# ==== GHI LOG (append) ====
log_path = "benchmark.csv"
is_new = not os.path.exists(log_path)
with open(log_path, "a") as f:  # dùng 'a' để ghi thêm, không overwrite
    # Nếu bạn muốn ghi tiêu đề dòng đầu tiên (chỉ cần một lần duy nhất), bạn có thể mở file thủ công và thêm:
    # Model,Image Size,FPS,Latency(ms),Model Size(MB),RAM Used(MB),Cameras,Show_Result
    f.write(f"{MODEL_PATH},{FRAME_W}x{FRAME_H},{fps:.2f},{avg_latency:.2f},"
            f"{model_size:.2f},{ram_used:.2f},{len(RTSP_URLS)},{SHOW_RESULT}\n")


