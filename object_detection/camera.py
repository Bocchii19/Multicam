import cv2
from ultralytics import YOLO

# Tải mô hình YOLOv8n đã được huấn luyện trước
model = YOLO('/home/edabk/Bocchi/object_detection/yolov8n_face.pt')

# Khởi tạo camera. Số 0 thường là camera mặc định của máy tính.
cap = cv2.VideoCapture(0)

# Kiểm tra xem camera có được mở thành công không
if not cap.isOpened():
    print("Không thể mở camera.")
    exit()

# Bắt đầu vòng lặp để đọc các khung hình từ camera
while True:
    # Đọc một khung hình từ camera
    ret, frame = cap.read()

    # Nếu không đọc được khung hình thì thoát
    if not ret:
        break

    # Sử dụng mô hình YOLOv8 để nhận dạng đối tượng trong khung hình
    # stream=True sẽ giúp tối ưu hóa việc xử lý cho video
    results = model(frame, stream=True)

    # Lặp qua các kết quả nhận dạng được
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Lấy tọa độ của khung bao quanh đối tượng
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Chuyển sang kiểu số nguyên

            # Vẽ hình chữ nhật lên khung hình
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Lấy độ tin cậy và tên của lớp đối tượng
            confidence = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            class_name = model.names[cls]

            # Hiển thị tên lớp và độ tin cậy
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 255, 255)
            thickness = 2
            cv2.putText(frame, f'{class_name} {confidence}', org, font, fontScale, color, thickness)

    # Hiển thị khung hình đã được xử lý
    cv2.imshow('YOLO', frame)

    # Nhấn phím 'q' để thoát khỏi vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên camera và đóng tất cả các cửa sổ
cap.release()
cv2.destroyAllWindows()
