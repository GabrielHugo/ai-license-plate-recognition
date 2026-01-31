import cv2
from shapely.creation import destroy_prepared
from sympy.printing.pretty.pretty_symbology import annotated
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")

image_path = "images/voiture_1.jpg"
img = cv2.imread(image_path)

results = model(img)
annotated_img = results[0].plot()

cv2.imshow("Detection YOLOv8 - Hugo", annotated_img)

cv2.waitKey(0)
cv2.destroyAllWindows()