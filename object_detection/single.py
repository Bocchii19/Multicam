import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO

# ======= Config =======
rtsp_url = "rtsp://admin:TIJEQB@192.168.0.111:554/ch1/main"
frame_w, frame_h = 640, 480

# Shared data
frame_result = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
lock = threading.Lock()

# Load YOLO model
model = YOLO("best.pt")  # Hoặc yolov8s.pt nếu máy đủ mạnh

def gstreamer_pipeline(url):
    return (
        f"rtspsrc location={url} latency=100 ! "
        "rtph265depay ! h265parse ! nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw, width={frame_w}, height={frame_h}, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1"
    )

def camera_worker():
    global frame_result
    cap = cv2.VideoCapture(gstreamer_pipeline(rtsp_url), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("❌ Không mở được camera.")
        return

    print("✅ Camera đang chạy.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Chạy YOLOv8
        results = model(frame, verbose=False)
        detected = results[0].plot()

        # Tính FPS
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-6)
        prev_time = now
        cv2.putText(detected, f"{int(fps)} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Cập nhật frame
        with lock:
            frame_result = detected.copy()

        time.sleep(0.01)  # Giảm tải CPU nếu cần

# ======= Start camera thread =======
t = threading.Thread(target=camera_worker, daemon=True)
t.start()

# ======= Display loop =======
print("👉 Nhấn Q để thoát.")
while True:
    with lock:
        display_frame = frame_result.copy()

    cv2.imshow("🎥 Camera + YOLOv8", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

