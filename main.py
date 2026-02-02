import cv2
from ultralytics import YOLO

model_plaque = YOLO("best_plaque.pt")
model_characteres = YOLO("best_char.pt")

images_path = "images/voiture_2.jpg"
img = cv2.imread(images_path)

results_plaque = model_plaque(img, conf=0.25)

plaque_trouvee = False

for r in results_plaque :
    for box in r.boxes :
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plaque_trouvee = True

        plate_crop = img[y1:y2, x1:x2]

        results_chars = model_characteres(plate_crop, conf=0.40)

        detections = []

        for r_char in results_chars :
            for box_char in r_char.boxes :
                cx1, cy1, cx2, cy2 = map(int, box_char.xyxy[0])

                cls_id = int(box_char.cls[0])
                char_name = model_characteres.names[cls_id]

                detections.append((cx1, char_name))

                cv2.rectangle(plate_crop, (cx1, cy1), (cx2, cy2), (0, 0, 255), 1)

            detections.sort(key=lambda x: x[0])

            texte_final = "".join([d[1] for d in detections])

            print(f"Plaque détectée : {texte_final}")

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, texte_final, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow(f"Zoom plaque {texte_final}", plate_crop)

if not plaque_trouvee :
    print("Aucune plaque trouvée")

cv2.imshow("Résultat Global", img)
cv2.waitKey(0)
cv2.destroyAllWindows()