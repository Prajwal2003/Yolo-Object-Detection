from ultralytics import YOLO  # Import the YOLOv10 library
import cv2
import cvzone

# Load the video file
cap = cv2.VideoCapture("videos/1.mp4")

# Load the YOLOv10 model with the specified weights
model = YOLO("/Users/starkz/PycharmProjects/Yolo_object_detection/Test Yolo/Yolo_weights/yolov10n.pt")

# Class names (YOLOv10 uses the same COCO class names)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
              "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
              "scissors", "teddy bear", "hair drier", "toothbrush"]

while True:
    success, img = cap.read()
    if not success:
        break  # Exit loop if video ends or fails

    # Run inference using YOLOv10
    results = model.predict(source=img, stream=True)

    # Parse results
    for r in results:
        for box in r.boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (200, 0, 150), 2)

            # Extract confidence and class
            confidence = round(float(box.conf) * 100)
            class_id = int(box.cls)

            # Display class name and confidence
            cvzone.putTextRect(img, f'{classNames[class_id]} C:{confidence}%',
                               (max(0, x1), max(35, y1)), scale=2)

    # Show the processed frame
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
