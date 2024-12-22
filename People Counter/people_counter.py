from torch.onnx.symbolic_opset11 import vstack
from ultralytics import YOLO
import cv2
import cvzone
from sort import *


cap = cv2.VideoCapture("../Yolo Webcam/videos/4.mp4")

model = YOLO("../Test Yolo/Yolo_weights/yolov8n.pt")

mask = cv2.imread("mask.png")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3 )
limitsup = [103, 161, 296, 161]
limitsdown = [527, 489, 735, 489]
countup = []
countdown = []

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]

while True:
    success, img = cap.read()
    imgreg = cv2.bitwise_and(img, mask)
    res = model(imgreg, stream=True)
    detections = np.empty((0,5))
    for r in res:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)

            confi = round(float(box.conf[0] * 100))
            print(confi)

            cls = box.cls[0]
            print(cls)
            current_class = classNames[int(cls)]
            if current_class == "person" and confi >= 50:
                #cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=15)
                current_array = np.array([x1,y1,x2,y2,confi])
                detections = np.vstack((detections, current_array) )

    track_res = tracker.update(detections)

    cv2.line(img, (limitsup[0],limitsup[1]), (limitsup[2],limitsup[3]), (255,0,0), 5)
    cv2.line(img, (limitsdown[0], limitsdown[1]), (limitsdown[2], limitsdown[3]), (255, 0, 0), 5)

    for res in track_res:
        x1, y1, x2, y2, id = res
        iid = int(id)
        print(res)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cvzone.putTextRect(img, f'{iid}', (max(0, x1), max(35, (y1))), scale=1, offset=5)

        wi, hi = (x2 - x1) // 2 , (y2 - y1) // 2
        cx, cy = x2 - wi, y2 - hi
        print(cx,cy)
        if limitsup[0] < cx < limitsup[2] and limitsup[1] - 20 < cy < limitsup[3] + 20:
            if iid not in countup:
                countup.append(id)
        if limitsdown[0] < cx < limitsdown[2] and limitsdown[1] - 20 < cy < limitsdown[3] + 20:
            if iid not in countdown:
                countdown.append(id)

    cvzone.putTextRect(img, f'Count {len(countup)}', (50, 50))
    print(len(countup))
    cvzone.putTextRect(img, f'Count {len(countdown)}', (200, 200))
    print(len(countdown))
    cv2.imshow("Image", img)
    cv2.waitKey(1)