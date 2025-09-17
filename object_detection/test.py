import cv2

url = "rtsp://admin:NNFVAJ@192.168.0.101:554/ch1/main"

pipeline = (
    f"rtspsrc location={url} latency=100 ! "
    "rtph265depay ! h265parse ! nvv4l2decoder ! "
    "nvvidconv ! video/x-raw, width=320, height=240, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ Không mở được camera H.265")
    exit()

print("✅ Đã mở được camera")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Mất kết nối")
        continue
    cv2.imshow("Camera H265", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

