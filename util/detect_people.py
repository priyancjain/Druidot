import cv2
import torch

def detect_people(frame, model):
    results = model(frame)
    people_count = 0

    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0:  # Class 0 = person in YOLO
            people_count += 1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {people_count}', (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f'People Count: {people_count}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return people_count, frame
