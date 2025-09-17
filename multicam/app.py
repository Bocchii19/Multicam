import os
import time
import threading
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO

# ================== RTSP (4 CAM) ==================
RTSP_URLS = [
    "rtsp://admin:XLRPZQ@192.168.0.121:554/ch1/main",
    "rtsp://admin:TIJEQB@192.168.0.117:554/ch1/main",
    "rtsp://admin:YDVFNP@192.168.0.108:554/ch1/main",
    "rtsp://admin:KAETPH@192.168.0.110:554/ch1/main",
]

# ================== CẤU HÌNH LƯỚI & GST ==================
TILE_W, TILE_H = 640, 360     # kích thước mỗi ô 2x2 (=> tổng 1280x720)
COLS, ROWS = 2, 2
LATENCY_MS = 100
USE_GPU = True

# ================== MODEL PATHS ==================
# Đổi sang đường dẫn thực tế của bạn
PERSON_DET_PATH = "/home/edabk/Bocchi/multicam/person_detection.pt"
PERSON_SEG_PATH = "/home/edabk/Bocchi/multicam/person_segmentation.pt"
POSE_PATH       = "/home/edabk/Bocchi/multicam/pose.pt"
YUNET_ONNX      = "/home/edabk/Bocchi/multicam/face_detection_yunet_2023mar.onnx"

# ================== MODELS ==================
person_detect_model  = YOLO(PERSON_DET_PATH)
person_segment_model = YOLO(PERSON_SEG_PATH)
pose_estimate_model  = YOLO(POSE_PATH)

face_detect_model = cv2.FaceDetectorYN.create(
    YUNET_ONNX, "", (320, 320), 0.9, 0.3, 5000
)

current_model = None  # "model1" | "model2" | "model3" | "model4"

# ================== GST PIPELINE (H.265) ==================
def gst_h265_pipeline(url: str, w: int, h: int, latency_ms: int, use_gpu: bool) -> str:
    if use_gpu:
        # GPU decode bằng nvv4l2decoder (Jetson)
        return (
            f"rtspsrc location={url} latency={latency_ms} ! "
            f"application/x-rtp,media=video,encoding-name=H265 ! "
            "rtph265depay ! h265parse ! nvv4l2decoder ! "
            "video/x-raw(memory:NVMM),format=NV12 ! "
            "nvvidconv ! video/x-raw,format=BGRx,width={w},height={h} ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        ).format(w=w, h=h)
    else:
        # Fallback CPU
        return (
            f"rtspsrc location={url} latency={latency_ms} ! "
            f"application/x-rtp,media=video,encoding-name=H265 ! "
            "rtph265depay ! h265parse ! avdec_h265 ! "
            "videoconvert ! videoscale ! "
            f"video/x-raw,format=BGR,width={w},height={h} ! "
            "appsink drop=1 max-buffers=1 sync=false"
        )

def draw_fps(frame, fps_text: str):
    return cv2.putText(frame, fps_text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2, cv2.LINE_AA)

# ================== WORKER MỖI CAMERA ==================
class CamWorker(threading.Thread):
    def __init__(self, index: int, url: str, use_gpu: bool):
        super().__init__(daemon=True)
        self.index = index
        self.url = url
        self.use_gpu = use_gpu
        self.frame = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()

    def open_capture(self, use_gpu_try: bool):
        pipeline = gst_h265_pipeline(self.url, TILE_W, TILE_H, LATENCY_MS, use_gpu_try)
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    def run(self):
        cap = self.open_capture(self.use_gpu)
        if not cap.isOpened() and self.use_gpu:
            print(f"⚠️ Cam {self.index+1}: GPU mở thất bại → fallback CPU")
            cap = self.open_capture(False)

        if not cap.isOpened():
            print(f"❌ Cam {self.index+1}: không mở được")
            return
        else:
            print(f"✅ Cam {self.index+1}: opened ({'GPU' if self.use_gpu else 'CPU'})")

        prev = time.time()
        while not self.stop_flag.is_set():
            ret, frm = cap.read()
            now = time.time()
            if not ret or frm is None:
                frm = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
                cv2.putText(frm, "No Signal", (TILE_W//2 - 60, TILE_H//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                time.sleep(0.05)
            else:
                fps = 1.0 / max(now - prev, 1e-6)
                prev = now
                draw_fps(frm, f"{int(fps)} FPS")
                if frm.shape[1] != TILE_W or frm.shape[0] != TILE_H:
                    frm = cv2.resize(frm, (TILE_W, TILE_H))

            with self.lock:
                self.frame = frm

        cap.release()

    def get_frame(self):
        with self.lock:
            return self.frame.copy()

# ================== XỬ LÝ MODEL ==================
def apply_model(frame):
    global current_model

    if frame is None:
        return None

    if current_model == "model1":
        # Person Detection
        results = person_detect_model(frame, verbose=False)
        return results[0].plot()

    elif current_model == "model2":
        # Face Detection (YuNet)
        h, w, _ = frame.shape
        face_detect_model.setInputSize((w, h))
        _, faces = face_detect_model.detect(frame)
        if faces is not None:
            for face in faces:
                x, y, ww, hh = face[0:4].astype(int)
                cv2.rectangle(frame, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
        return frame

    elif current_model == "model3":
        # Person Segmentation
        results = person_segment_model(frame, verbose=False)
        return results[0].plot()

    elif current_model == "model4":
        # Pose Estimation
        results = pose_estimate_model(frame, verbose=False)
        return results[0].plot()

    # No model selected → passthrough
    return frame

# ================== UI TKINTER ==================
root = tk.Tk()
root.title("MULTI CAMERA AI")
root.geometry("1280x760")

title_label = tk.Label(root, text="MULTI CAMERA AI",
                       font=("Arial", 18, "bold"))
title_label.pack(pady=8)

video_label = tk.Label(root)  # 1280x720
video_label.pack()

status_label = tk.Label(root, text="Chưa chọn model", font=("Arial", 14), fg="red")
status_label.pack(pady=8)

# Buttons
def run_person_detect():
    global current_model
    current_model = "model1"
    status_label.config(text="Model 1: Person Detection", fg="green")

def run_face_detect():
    global current_model
    current_model = "model2"
    status_label.config(text="Model 2: Face Detection", fg="blue")

def run_person_segment():
    global current_model
    current_model = "model3"
    status_label.config(text="Model 3: Person Segmentation", fg="red")

def run_pose_estimate():
    global current_model
    current_model = "model4"
    status_label.config(text="Model 4: Pose Estimation", fg="purple")

button_frame = tk.Frame(root)
button_frame.pack(pady=5)

btn_style = {"font": ("Arial", 13, "bold"), "width": 24, "height": 2, "padx": 5, "pady": 5}
tk.Button(button_frame, text="Person Detection", bg="#4CAF50", fg="white",
          command=run_person_detect, **btn_style).grid(row=0, column=0)
tk.Button(button_frame, text="Face Detection", bg="#2196F3", fg="white",
          command=run_face_detect, **btn_style).grid(row=0, column=1)
tk.Button(button_frame, text="Person Segmentation", bg="#F32121", fg="white",
          command=run_person_segment, **btn_style).grid(row=0, column=2)
tk.Button(button_frame, text="Pose Estimation", bg="#9C27B0", fg="white",
          command=run_pose_estimate, **btn_style).grid(row=0, column=3)

# ================== KHỞI TẠO WORKERS ==================
if "GStreamer" not in cv2.getBuildInformation():
    print("⚠️ OpenCV chưa build với GStreamer. Hãy build lại có GStreamer (WITH_GSTREAMER=ON).")

workers = [CamWorker(i, url, USE_GPU) for i, url in enumerate(RTSP_URLS)]
for w in workers:
    w.start()

# ================== VÒNG LẶP CẬP NHẬT UI ==================
def update_frame():
    # Lấy frame hiện tại từ từng cam
    tiles = []
    for i in range(ROWS * COLS):
        if i < len(workers):
            frm = workers[i].get_frame()
        else:
            frm = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
            cv2.putText(frm, "Empty", (TILE_W//2 - 40, TILE_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        # Áp dụng model (nếu chọn)
        processed = apply_model(frm)
        if processed is None or processed.shape[:2] != (TILE_H, TILE_W):
            processed = cv2.resize(processed if processed is not None else frm, (TILE_W, TILE_H))
        tiles.append(processed)

    # Ghép 2x2
    top = np.hstack(tiles[0:2])
    bottom = np.hstack(tiles[2:4])
    grid = np.vstack([top, bottom])

    # Convert lên Tkinter
    rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    video_label.after(10, update_frame)

# ================== START UI ==================
update_frame()
root.mainloop()

# ================== CLEANUP ==================
for w in workers:
    w.stop_flag.set()
for w in workers:
    w.join(timeout=1.0)
cv2.destroyAllWindows()

