import cv2
from shapely.creation import destroy_prepared
from sympy.printing.pretty.pretty_symbology import annotated
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")

image_path = "images/voiture_1.jpg"
img = cv2.imread(image_path)

results = model(img)

for result in results :

    boxes = result.boxes

    for box in boxes :

        cls = int(box.cls[0])

        if cls in [2, 3, 5, 7]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            car_crop = img[y1: y2, x1: x2]
            cv2.imshow("Voiture isolée", car_crop)


cv2.waitKey(0)
cv2.destroyAllWindows()