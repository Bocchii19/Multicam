import cv2
import numpy as np
import onnxruntime as ort

# ==== CONFIG ====
MODEL_PATH = "/home/edabk/Bocchi/person_detection.onnx"
CONF_THRES = 0.5
IOU_THRES = 0.45

# ==== Load model on CUDA ====
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
in_h, in_w = session.get_inputs()[0].shape[2:4]

# ==== Letterbox (giữ tỉ lệ ảnh) ====
def letterbox(im, new_shape=(640,640), color=(114,114,114)):
    shape = im.shape[:2]
    r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r)))
    dw, dh = (new_shape[1]-new_unpad[0])//2, (new_shape[0]-new_unpad[1])//2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return im, r, dw, dh

# ==== IOU + NMS ====
def iou_calc(a,b):
    x1,y1,x2,y2 = a[:4]; x1g,y1g,x2g,y2g = b[:4]
    xi1, yi1, xi2, yi2 = max(x1,x1g), max(y1,y1g), min(x2,x2g), min(y2,y2g)
    inter = max(0,xi2-xi1)*max(0,yi2-yi1)
    area_a, area_b = (x2-x1)*(y2-y1), (x2g-x1g)*(y2g-y1g)
    return inter/(area_a+area_b-inter+1e-6)

def nms(boxes, iou=0.45):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    while boxes:
        b = boxes.pop(0)
        keep.append(b)
        boxes = [x for x in boxes if iou_calc(b,x) < iou]
    return keep

# ==== USB Camera qua GStreamer ====
# Đảm bảo camera USB là /dev/video0
gst = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw, width=640, height=480, framerate=30/1 ! "
    "videoconvert ! appsink"
)
cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Preprocess
    img, ratio, dw, dh = letterbox(frame, (in_w, in_h))
    inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    inp = np.transpose(inp, (2,0,1))[None]

    # Inference
    preds = session.run([output_name], {input_name: inp})[0][0]

    # Postprocess (only person)
    boxes = []
    for det in preds:
        conf = det[4]
        if conf < CONF_THRES: continue
        x,y,w,h = det[:4]
        x1 = (x - w/2 - dw)/ratio
        y1 = (y - h/2 - dh)/ratio
        x2 = (x + w/2 - dw)/ratio
        y2 = (y + h/2 - dh)/ratio
        boxes.append([int(x1),int(y1),int(x2),int(y2),float(conf)])

    # NMS + Draw
    for (x1,y1,x2,y2,score) in nms(boxes, IOU_THRES):
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,f"person {score:.2f}",(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("USB Camera (CUDA ONNX)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()

