# import cv2
# import torch

# def detect_people(frame, model):
#     results = model(frame)
#     people_count = 0

#     for det in results.xyxy[0]:
#         x1, y1, x2, y2, conf, cls = det
#         if int(cls) == 0:  # Class 0 = person in YOLO
#             people_count += 1
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.putText(frame, f'Person {people_count}', (int(x1), int(y1)-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     cv2.putText(frame, f'People Count: {people_count}', (10, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     return people_count, frame
# import cv2
# import torch
# import numpy as np

# def preprocess_frame(frame, img_size=640):
#     """
#     Preprocess the frame for YOLOv5 model input.
#     - Resize to the required input size.
#     - Normalize pixel values to [0, 1].
#     - Transpose to channel-first format (C, H, W).
#     - Add a batch dimension.

#     Args:
#         frame (numpy.ndarray): Input frame from OpenCV.
#         img_size (int): Target size for YOLO model (default: 640).

#     Returns:
#         torch.Tensor: Preprocessed frame tensor.
#     """
#     # Resize frame
#     frame_resized = cv2.resize(frame, (img_size, img_size))

#     # Convert BGR to RGB (YOLO expects RGB input)
#     frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

#     # Normalize pixel values to [0, 1] and convert to float32
#     frame_normalized = frame_rgb / 255.0

#     # Transpose to channel-first format and add batch dimension
#     frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)

#     return frame_tensor.float()


# def detect_people(frame, model):
#     """
#     Detect people in the input frame using a YOLOv5 model.

#     Args:
#         frame (numpy.ndarray): Input frame from OpenCV.
#         model: YOLOv5 PyTorch model.

#     Returns:
#         int: Count of people detected.
#         numpy.ndarray: Annotated frame with bounding boxes and labels.
#     """
#     # Preprocess the frame for YOLOv5 input
#     input_tensor = preprocess_frame(frame)

#     # Perform inference with the model
#     results = model(input_tensor)

#     # Parse detections
#     people_count = 0
#     for det in results.xyxy[0]:  # xyxy format: [x1, y1, x2, y2, confidence, class]
#         x1, y1, x2, y2, conf, cls = det.cpu().numpy()  # Convert to numpy
#         if int(cls) == 0:  # Class 0 = person
#             people_count += 1
#             # Draw bounding box and label on the original frame
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.putText(frame, f'Person {people_count}', (int(x1), int(y1) - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Add people count to the frame
#     cv2.putText(frame, f'People Count: {people_count}', (10, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     return people_count, frame
import torch
import cv2

def detect_people(frame, model):
    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (640, 640))
    results = model(frame_resized)

    # Extract detected objects
    people_count = 0
    detections = results.xyxy[0]  # Access the detection tensor

    for det in detections:
        x1, y1, x2, y2, confidence, cls = det
        if int(cls) == 0:  # Class 0 corresponds to 'person' in YOLOv5
            people_count += 1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'Person: {confidence:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return people_count, frame
