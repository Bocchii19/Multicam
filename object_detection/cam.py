import cv2
import numpy as np
import threading
import time

# ======= Cấu hình =======
frame_w, frame_h = 160, 120  # 👈 Kích thước nhỏ hơn 320x240
cols, rows = 3, 2            # Lưới hiển thị 6 cột × 5 hàng

# Danh sách các URL RTSP
rtsp_urls = [
"rtsp://admin:CYXJBA@192.168.0.109:554/ch1/main",
"rtsp://admin:CPSFLT@192.168.0.104:554/ch1/main",
"rtsp://admin:NNFVAJ@192.168.0.114:554/ch1/main",
 "rtsp://admin:WSLRQC@192.168.0.113:554/ch1/main",
"rtsp://admin:NXKPHU@192.168.0.115:554/ch1/main",
"rtsp://admin:NWKGIC@192.168.0.124:554/ch1/main"
]

N = len(rtsp_urls)

# Khởi tạo buffer khung hình và FPS
frames = [np.zeros((frame_h, frame_w, 3), dtype=np.uint8) for _ in range(N)]
lock = threading.Lock()

def draw_fps(frame, text):
    return cv2.putText(frame, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def build_gstreamer_pipeline(url):
    return (
        f"rtspsrc location={url} latency=0 ! "
        "rtph265depay ! h265parse ! nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw, width={frame_w}, height={frame_h}, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1"
    )

def camera_thread(index, url):
    pipeline = build_gstreamer_pipeline(url)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print(f"❌ Camera {index+1} not opened")
        return

    print(f"✅ Camera {index+1} started")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        now = time.time()

        if not ret:
            frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            cv2.putText(frame, "No Signal", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            fps = 1.0 / (now - prev_time + 1e-6)
            prev_time = now
            text = f"{int(fps)} FPS"
            draw_fps(frame, text)

        # Đảm bảo frame đúng kích thước
        frame = cv2.resize(frame, (frame_w, frame_h))

        with lock:
            frames[index] = frame.copy()

        time.sleep(0.03)  # ~30 FPS

# ======= Khởi tạo thread cho mỗi camera =======
for i, url in enumerate(rtsp_urls):
    t = threading.Thread(target=camera_thread, args=(i, url), daemon=True)
    t.start()

# ======= Vòng lặp hiển thị chính =======
print("👉 Nhấn Q để thoát.")
while True:
    with lock:
        display_rows = []
        for r in range(rows):
            row_frames = frames[r * cols:(r + 1) * cols]
            # Nếu thiếu frame trong hàng, bổ sung ảnh đen
            while len(row_frames) < cols:
                row_frames.append(np.zeros((frame_h, frame_w, 3), dtype=np.uint8))
            row = np.hstack(row_frames)
            display_rows.append(row)
        grid = np.vstack(display_rows)

    cv2.imshow("🎥 Multi-Cam RTSP (Jetson Optimized 160x120)", grid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
