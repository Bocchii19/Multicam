import cv2
import threading
import numpy as np
import time
import gc
from typing import Tuple   

try:
    import torch
except Exception:
    torch = None

from ultralytics import YOLO

# ==== CẤU HÌNH ====
RTSP_URLS = [
    "rtsp://admin:LUXHLR@192.168.0.103:554/ch1/main",
    "rtsp://admin:HLTHKD@192.168.0.122:554/ch1/main",
    "rtsp://admin:XLRPZQ@192.168.0.121:554/ch1/main",
    "rtsp://admin:WNRDVL@192.168.0.118:554/ch1/main",
    "rtsp://admin:LUXHLR@192.168.0.103:554/ch1/main",
    "rtsp://admin:HLTHKD@192.168.0.122:554/ch1/main",
    "rtsp://admin:XLRPZQ@192.168.0.121:554/ch1/main",
    "rtsp://admin:WNRDVL@192.168.0.118:554/ch1/main",
    "rtsp://admin:LUXHLR@192.168.0.103:554/ch1/main",
    "rtsp://admin:HLTHKD@192.168.0.122:554/ch1/main",
    "rtsp://admin:XLRPZQ@192.168.0.121:554/ch1/main",
    "rtsp://admin:WNRDVL@192.168.0.118:554/ch1/main"
]

NUM_ROWS, NUM_COLS = 3, 4
FRAME_W, FRAME_H = 480, 360
SHOW_RESULT = True

# ==== ĐỊNH NGHĨA CÁC MODEL ====
MODEL_PATHS = {
    "1": ("face",   "face_detect.engine"),
    "2": ("person_segment", "person_seg.pt"),
    "3": ("person_detect",  "best.pt"),
}
DEFAULT_KEY = "1"
PRELOAD_ALL = False  # True = nạp sẵn cả 3 model

# ==== QUẢN LÝ MODEL (HOT-SWAP) ====
class ModelHub:
    def __init__(self, model_paths: dict, default_key: str, preload_all: bool = False):
        self.model_paths = model_paths
        self.models = {}
        self.current_key = default_key
        self.lock = threading.RLock()

        name, path = self.model_paths[self.current_key]
        self.models[self.current_key] = self._load_model(path)

        if preload_all:
            for k, (_, p) in self.model_paths.items():
                if k not in self.models:
                    self.models[k] = self._load_model(p)

    def _load_model(self, path: str) -> YOLO:
        t0 = time.time()
        m = YOLO(path)
        dt = (time.time() - t0) * 1000
        print(f"[ModelHub] Loaded model '{path}' in {dt:.0f} ms")
        return m

    def get(self) -> Tuple[str, YOLO]:
        with self.lock:
            name, _ = self.model_paths[self.current_key]
            return name, self.models[self.current_key]

    def switch(self, key: str) -> bool:
        with self.lock:
            if key not in self.model_paths:
                return False
            if key not in self.models:
                _, path = self.model_paths[key]
                try:
                    self.models[key] = self._load_model(path)
                except Exception as e:
                    print(f"[ModelHub] Failed to load {path}: {e}")
                    return False
            self.current_key = key
            print(f"[ModelHub] Switched to '{self.model_paths[key][0]}' ({key})")
            return True

    def unload(self, key: str):
        with self.lock:
            if key in self.models:
                try:
                    del self.models[key]
                    gc.collect()
                    if torch and hasattr(torch, "cuda") and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"[ModelHub] Unloaded model key={key}")
                except Exception as e:
                    print(f"[ModelHub] Unload error: {e}")

# ==== PIPELINE GStreamer ====
def gstreamer_pipeline(rtsp_url, width, height):
    return (
        f'rtspsrc location={rtsp_url} latency=0 ! '
        'rtph265depay ! h265parse ! nvv4l2decoder drop-frame-interval=0 ! '
        f'nvvidconv ! video/x-raw, width={width}, height={height}, format=BGRx ! '
        'videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1 sync=false'
    )

# ==== THREAD CAMERA ====
class CameraStream(threading.Thread):
    def __init__(self, rtsp_url):
        super().__init__(daemon=True)
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
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy()

    def stop(self):
        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass

# ==== KHỞI TẠO CAMERA ====
streams = [CameraStream(url) for url in RTSP_URLS]
for stream in streams:
    stream.start()

# ==== KHỞI TẠO MODEL HUB ====
hub = ModelHub(MODEL_PATHS, DEFAULT_KEY, PRELOAD_ALL)

# ==== HÀM VẼ OVERLAY ====
def draw_header(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), thickness=-1)
    cv2.putText(img, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

instructions = "Press [1]=face  [2]=person  [3]=plate   [Q]=quit"

# ==== VÒNG LẶP CHÍNH ====
try:
    while True:
        frames = [s.get_frame() for s in streams]

        total = NUM_ROWS * NUM_COLS
        if len(frames) < total:
            frames += [np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8) for _ in range(total - len(frames))]

        grid_rows = [np.hstack(frames[i*NUM_COLS:(i+1)*NUM_COLS]) for i in range(NUM_ROWS)]
        grid_frame = np.vstack(grid_rows)

        model_name, model = hub.get()

        results = model(grid_frame, verbose=False)[0]
        annotated = results.plot()

        draw_header(annotated, f"Model: {model_name}  |  {instructions}")

        if SHOW_RESULT:
            cv2.imshow("MULTICAM", annotated)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k in (ord('1'), ord('2'), ord('3')):
                hub.switch(chr(k))

except KeyboardInterrupt:
    print("Stopping...")

# ==== CLEANUP ====
for s in streams:
    s.stop()
cv2.destroyAllWindows()

