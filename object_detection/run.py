import cv2
import threading
import numpy as np
from ultralytics import YOLO

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
SAVE_OUTPUT = False

# ==== MODEL ====
model = YOLO("yolov8_face_custom_v2.pt")  # Thay bằng model bạn huấn luyện

# ==== GStreamer pipeline với H.265 và Jetson decoder ====
def gstreamer_pipeline(rtsp_url, width, height):
    return (
        f'rtspsrc location={rtsp_url} latency=0 ! '
        'rtph265depay ! h265parse ! nvv4l2decoder ! '
        f'nvvidconv ! video/x-raw, width={width}, height={height}, format=BGRx ! '
        'videoconvert ! video/x-raw, format=BGR ! '
        'appsink drop=1'
    )

# ==== LỚP THREAD CAMERA ====
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

# ==== KHỞI TẠO CAMERA ====
streams = [CameraStream(url) for url in RTSP_URLS]
for s in streams:
    s.start()

# ==== VIDEO WRITER ====
if SAVE_OUTPUT:
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20,
                          (FRAME_W * NUM_COLS, FRAME_H * NUM_ROWS))

# ==== VÒNG LẶP CHÍNH ====
try:
    while True:
        frames = [s.get_frame() for s in streams]
        grid_rows = [np.hstack(frames[i*NUM_COLS:(i+1)*NUM_COLS]) for i in range(NUM_ROWS)]
        grid_frame = np.vstack(grid_rows)

        # YOLOv8 inference
        results = model(grid_frame, verbose=False)[0]
        annotated = results.plot()

        if SHOW_RESULT:
            cv2.imshow("YOLOv8 - Grid View", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if SAVE_OUTPUT:
            out.write(annotated)

except KeyboardInterrupt:
    print("Stopping...")

# ==== CLEANUP ====
for s in streams:
    s.stop()
if SAVE_OUTPUT:
    out.release()
cv2.destroyAllWindows()
