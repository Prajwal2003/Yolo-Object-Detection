from ultralytics import YOLO
import cv2
from collections import Counter

model = YOLO("/Users/starkz/PycharmProjects/Yolo_object_detection/Test Yolo/Yolo_weights/Chigari.pt")

image_path = "test1.jpeg"

results = model(image_path)

class_names = model.names
detections = results[0].boxes.cls

detected_classes = [class_names[int(cls_idx)] for cls_idx in detections]

class_counts = Counter(detected_classes)

print("Detected Object Counts:")
for obj_class, count in class_counts.items():
    print(f"{obj_class}: {count}")

image = cv2.imread(image_path)

for box, cls_idx in zip(results[0].boxes.xywh, detections):
    x, y, w, h = box
    x1, y1 = int(x - w / 2), int(y - h / 2)
    x2, y2 = int(x + w / 2), int(y + h / 2)
    label = class_names[int(cls_idx)]
    color = (0, 255, 0)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

cv2.imshow("Detection Result", image)

cv2.waitKey(0)