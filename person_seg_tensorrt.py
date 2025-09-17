from ultralytics import YOLO
import cv2
import numpy as np
import threading
import time

# ================== CAMERA LIST ==================
RTSP_URLS = [
    "rtsp://admin:DVCLRQ@192.168.0.123:554/ch1/main",
    "rtsp://admin:HLTHKD@192.168.0.122:554/ch1/main",
    "rtsp://admin:XLRPZQ@192.168.0.121:554/ch1/main",
    "rtsp://admin:TIJEQB@192.168.0.117:554/ch1/main",
    "rtsp://admin:JRGMMV@192.168.0.120:554/ch1/main",
    "rtsp://admin:WNRDVL@192.168.0.118:554/ch1/main",
    "rtsp://admin:CYXJBA@192.168.0.109:554/ch1/main",
    "rtsp://admin:CPSFLT@192.168.0.104:554/ch1/main",
    "rtsp://admin:NNFVAJ@192.168.0.114:554/ch1/main",
    "rtsp://admin:WSLRQC@192.168.0.113:554/ch1/main",
    "rtsp://admin:NXKPHU@192.168.0.115:554/ch1/main",
    "rtsp://admin:NWKGIC@192.168.0.124:554/ch1/main",
    "rtsp://admin:EIUSAY@192.168.0.105:554/ch1/main",
    "rtsp://admin:YDVFNP@192.168.0.108:554/ch1/main",
    "rtsp://admin:PBPBND@192.168.0.119:554/ch1/main",
    "rtsp://admin:IFPREC@192.168.0.101:554/ch1/main",
    "rtsp://admin:KAETPH@192.168.0.110:554/ch1/main",
    "rtsp://admin:DTAJVP@192.168.0.102:554/ch1/main",
    "rtsp://admin:BWKUYM@192.168.0.116:554/ch1/main",
    "rtsp://admin:PSQSFP@192.168.0.107:554/ch1/main",
]

# ================== CONFIG ==================
FRAME_W, FRAME_H = 360, 240
COLS, ROWS = 5, 4
LATENCY_MS = 0
USE_GPU = True
WINDOW_NAME = "MultiCam-Segment"
ENGINE_PATH = "person_seg.engine"
CONF_THRESHOLD = 0.5
IMG_SIZE = 640
DEVICE = 0

print("Loading segmentation model:", ENGINE_PATH)
model = YOLO(ENGINE_PATH)

def gst_h265_pipeline(url: str, w: int, h: int, latency_ms: int, use_gpu: bool) -> str:
    if use_gpu:
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
        return (
            f"rtspsrc location={url} latency={latency_ms} ! "
            f"application/x-rtp,media=video,encoding-name=H265 ! "
            "rtph265depay ! h265parse ! avdec_h265 ! "
            "videoconvert ! videoscale ! "
            f"video/x-raw,format=BGR,width={w},height={h} ! "
            "appsink drop=1 max-buffers=1 sync=false"
        )

class CamWorker(threading.Thread):
    def __init__(self, index: int, url: str, use_gpu: bool):
        super().__init__(daemon=True)
        self.index = index
        self.url = url
        self.use_gpu = use_gpu
        self.frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()

    def open_capture(self, use_gpu_try: bool):
        pipeline = gst_h265_pipeline(self.url, FRAME_W, FRAME_H, LATENCY_MS, use_gpu_try)
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    def run(self):
        cap = self.open_capture(self.use_gpu)
        if not cap.isOpened() and self.use_gpu:
            print(f"[Cam {self.index+1}] ⚠️ GPU pipeline fail → fallback CPU")
            cap = self.open_capture(False)

        if not cap.isOpened():
            print(f"[Cam {self.index+1}] ❌ Không mở được stream")
            return
        else:
            print(f"[Cam {self.index+1}] ✅ opened ({'GPU' if self.use_gpu else 'CPU'})")

        prev = time.time()
        while not self.stop_flag.is_set():
            ret, frm = cap.read()
            now = time.time()
            if not ret or frm is None:
                frm = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
                cv2.putText(frm, "No Signal", (FRAME_W//2 - 60, FRAME_H//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                time.sleep(0.05)
            else:
                fps = 1.0 / max(now - prev, 1e-6)
                prev = now
                cv2.putText(frm, f"{int(fps)} FPS", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2, cv2.LINE_AA)
                if frm.shape[1] != FRAME_W or frm.shape[0] != FRAME_H:
                    frm = cv2.resize(frm, (FRAME_W, FRAME_H))

            with self.lock:
                self.frame = frm

        cap.release()

    def get_frame(self):
        with self.lock:
            return self.frame.copy()

def main():
    workers = [CamWorker(i, url, USE_GPU) for i, url in enumerate(RTSP_URLS)]
    for w in workers:
        w.start()

    print("👉 Nhấn Q để thoát.")
    try:
        while True:
            grid_rows = []
            for r in range(ROWS):
                row_frames = []
                for c in range(COLS):
                    idx = r * COLS + c
                    if idx < len(workers):
                        row_frames.append(workers[idx].get_frame())
                    else:
                        row_frames.append(np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8))
                row = np.hstack(row_frames)
                grid_rows.append(row)
            grid = np.vstack(grid_rows)

            try:
                results = model.predict(source=grid, imgsz=IMG_SIZE, device=DEVICE,
                                        half=True, conf=CONF_THRESHOLD, verbose=False)
            except Exception as e:
                print("Inference error:", e)
                results = []

            if results:
                r = results[0]
                if r.masks is not None and r.masks.data is not None:
                    masks = r.masks.data.cpu().numpy()
                    for mask in masks:
                        mask = (mask > 0.5).astype(np.uint8) * 255
                        mask = cv2.resize(mask, (grid.shape[1], grid.shape[0]))
                        color_mask = np.zeros_like(grid)
                        color_mask[:, :, 1] = mask
                        grid = cv2.addWeighted(grid, 1.0, color_mask, 0.5, 0)

            cv2.imshow(WINDOW_NAME, grid)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        for w in workers:
            w.stop_flag.set()
        for w in workers:
            w.join(timeout=1.0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if "GStreamer" not in cv2.getBuildInformation():
        print("⚠️ OpenCV chưa build với GStreamer. Hãy build lại có GStreamer.")
    main()

