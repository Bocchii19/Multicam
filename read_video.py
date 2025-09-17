import cv2

VIDEO_PATH = "output_grid.mp4"  # Change to your file path

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ Cannot open video: {VIDEO_PATH}")
    exit()

print("🎥 Playing video... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video ended or cannot read frame.")
        break

    cv2.imshow("Video Player", frame)

    # Press 'q' to exit early
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

