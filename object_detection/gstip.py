import cv2


url = "rtsp://admin:TIJEQB@192.168.0.111:554/ch1/main"

pipeline = (
    f"rtspsrc location={url} latency=0 ! "
    "rtph265depay ! h265parse ! nvv4l2decoder ! "
    "nvvidconv ! "
    "video/x-raw, width=640, height=480, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink drop=1"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ Không mở được camera.")
    exit()


print("✅ Camera H.265 đang chạy. Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Không đọc được frame.")
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow("H265 RTSP Camera + YOLOv8", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



