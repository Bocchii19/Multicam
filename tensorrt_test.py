import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO

# ================== DANH SÁCH CAMERA ==================
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

# ================== CẤU HÌNH ==================
FRAME_W, FRAME_H = 360, 240           # kích thước mỗi camera trong lưới
COLS, ROWS = 5, 4                      # 5 x 4 = 20 ô
LATENCY_MS = 0
USE_GPU = True
WINDOW_NAME = "MultiCam"
ENGINE_PATH = "person_detect.engine"            # đường dẫn tới file .engine
CONF_THRESHOLD = 0.5
IMG_SIZE = 640                         # imgsz truyền cho model.predict
DEVICE = 0                             # device GPU id, hoặc 'cpu' nếu muốn CPU

# ================== Load model (TensorRT engine) ==================
print("Loading model:", ENGINE_PATH)
model = YOLO(ENGINE_PATH)               # Ultralytics sẽ dùng TensorRT backend cho .engine
# không cần model.to() khi dùng engine, nhưng set device in predict below

# ================== GStreamer helper ==================
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

# ================== Cam worker (thread) ==================
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
                # draw fps on each small frame
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

# ================== Main ==================
def main():
    workers = [CamWorker(i, url, USE_GPU) for i, url in enumerate(RTSP_URLS)]
    for w in workers:
        w.start()

    print("👉 Nhấn Q để thoát.")
    try:
        while True:
            # tạo grid
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

            # Run YOLO inference bằng Ultralytics (engine TensorRT)
            # Lưu ý: truyền device=DEVICE (0 cho cuda:0), half=True để dùng FP16 nếu engine hỗ trợ
            try:
                results = model.predict(grid, imgsz=IMG_SIZE, device=DEVICE, half=True, conf=CONF_THRESHOLD, verbose=False)
            except Exception as e:
                # nếu lỗi bất ngờ, in ra và tiếp tục vòng lặp để không crash
                print("Inference error:", e)
                results = []

            # results là list; ta duyệt và vẽ bbox
            # Ultralytics trả về boxes theo kích thước ảnh gốc đã truyền (grid)
            if results:
                # results có thể chứa nhiều items nếu batch >1; chúng ta truyền 1 ảnh nên lấy results[0]
                r = results[0]
                # r.boxes chứa tensor/array; dùng r.boxes.xyxy, r.boxes.conf, r.boxes.cls
                try:
                    xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, "cpu") else np.array(r.boxes.xyxy)
                    confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, "cpu") else np.array(r.boxes.conf)
                    cls  = r.boxes.cls.cpu().numpy()  if hasattr(r.boxes.cls, "cpu")  else np.array(r.boxes.cls)
                except Exception:
                    # fallback: r.boxes may have .xyxy as list of tensors
                    xyxy = []
                    confs = []
                    cls = []
                    for b in r.boxes:
                        xy = b.xyxy[0].cpu().numpy() if hasattr(b.xyxy[0], "cpu") else np.array(b.xyxy[0])
                        xyxy.append(xy)
                        confs.append(float(b.conf[0]))
                        cls.append(int(b.cls[0]))
                    xyxy = np.array(xyxy) if len(xyxy) else np.empty((0,4))
                    confs = np.array(confs)
                    cls = np.array(cls)

                # draw
                for bb, cf, cid in zip(xyxy, confs, cls):
                    x1, y1, x2, y2 = map(int, bb[:4])
                    label = f"{int(cid)} {cf:.2f}"
                    cv2.rectangle(grid, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(grid, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1)

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

