import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")

img = np.zeros((640, 640, 3), dtype=np.uint8)

result = model(img)

print("Succès ! Tout est installé correctement.")