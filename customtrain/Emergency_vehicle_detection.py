from ultralytics import YOLO
import cv2

model = YOLO(r"/Users/starkz/PycharmProjects/Yolo_object_detection/customtrain/runs/detect/train6/weights/best.pt")

source = r"/Users/starkz/PycharmProjects/Yolo_object_detection/Project/testfor ambulance.jpeg"

results = model.predict(source=source, save=True, imgsz=640)

for result in results:

    image = result.orig_img.copy()
    num_detections = len(result.boxes)
    print(f"Number of detections: {num_detections}")


    for box in result.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{class_name} {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Detections", image)

    output_path = "output_with_boxes.jpg"
    cv2.imwrite(output_path, image)
    print(f"Saved output image to {output_path}")

    cv2.waitKey(0)

cv2.destroyAllWindows()
