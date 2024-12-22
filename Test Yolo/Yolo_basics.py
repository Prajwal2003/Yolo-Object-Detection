from ultralytics import YOLO
import cv2

model = YOLO('Yolo_weights/yolov8n.pt')
result = model("images/2.jpeg", show = True)
cv2.waitKey(0)