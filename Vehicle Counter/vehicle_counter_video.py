from torch.onnx.symbolic_opset11 import vstack
from ultralytics import YOLO
import cv2
import cvzone
from sort import *


cap = cv2.VideoCapture("../Yolo Webcam/videos/2.mp4")

model = YOLO("../Test Yolo/Yolo_weights/yolov8n.pt")

mask = cv2.imread("mask.png")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3 )
limits = [280, 370,690,370]
count = []

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
            if current_class == "car" or current_class == "motorbike" or current_class == "bus" or current_class == "truck" and confi >= 50:
                #cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=15)
                current_array = np.array([x1,y1,x2,y2,confi])
                detections = np.vstack((detections, current_array) )

    track_res = tracker.update(detections)
    cv2.line(img, (limits[0],limits[1]), (limits[2],limits[3]), (255,0,0), 5)
    for res in track_res:
        x1, y1, x2, y2, id = res
        iid = int(id)
        print(res)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cvzone.putTextRect(img, f'{iid}', (max(0, x1), max(35, (y1))), scale=1, offset=5)

        wi, hi = (x2 - x1) // 2 , (y2 - y1) // 2
        cx, cy = x2 - wi, y2 - hi
        print(cx,cy)
        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[3] + 20:
            if iid not in count:
                count.append(id)

    cvzone.putTextRect(img, f'Count {len(count)}', (50, 50))
    print(len(count))
    cv2.imshow("Image", img)
    print(count)
    cv2.imshow("Detection Region", imgreg)
    cv2.waitKey(1)