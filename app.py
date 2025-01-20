from flask import Flask, Response, render_template
import cv2
from deepface import DeepFace

# Initialize the Flask app
app = Flask(__name__)

# Load the Haar Cascade file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        # Capture frame from the webcam
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw bounding box around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Crop the detected face
            face_crop = frame[y:y + h, x:x + w]

            # Try detecting emotion
            try:
                # Analyze emotion using DeepFace
                emotion = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                # dominant_emotion = emotion['dominant_emotion']
                if isinstance(emotion, list):
                     dominant_emotion = emotion[0]['dominant_emotion']
                else:
                    dominant_emotion = emotion['dominant_emotion']

                # Put the detected emotion on the frame
                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            except Exception as e:
                print(f"Emotion detection error: {e}")
            number_of_people = len(faces)
            cv2.putText(frame, f'People Detected: {number_of_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)




# import cv2
# import torch
# import numpy as np
# from flask import Flask, render_template, Response
# from util.detect_people import detect_people
# from util.detect_emotion import detect_emotion
# # import moviepy.editor

# app = Flask(__name__)

# import sys
# sys.path.append('./yolov5')  # Ensure Python imports from YOLOv5, not local utils

# from yolov5.models.common import DetectMultiBackend
# from yolov5.utils.general import non_max_suppression, TryExcept

# model = DetectMultiBackend('./yolov5/yolov5s.pt')


# # Open Webcam
# video_source = 0  # Change to IP Camera URL if needed
# cap = cv2.VideoCapture(video_source)

# def generate_frames():
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Detect people in the frame
#         people_count, frame = detect_people(frame, model)

#         # Detect emotions
#         frame = detect_emotion(frame)

#         # Encode frame to JPEG
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug=True)
# import cv2
# import torch
# import numpy as np
# from flask import Flask, render_template, Response
# from util.detect_people import detect_people
# from util.detect_emotion import detect_emotion

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# def generate_frames():
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Use a smaller model for faster detectio
#     cap = cv2.VideoCapture(0)
# # Initialize video capture (default webcam)


#     if not cap.isOpened():
#         print("Error: Could not open video feed.")
#         exit()

# # Set desired confidence threshold
#     CONFIDENCE_THRESHOLD = 0.5  # Filter detections below this threshold

#     while True:
#         # Capture frame from webcam
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame.")
#             break

#         # Resize frame for consistent processing (optional)
#         height, width, _ = frame.shape
#         resized_frame = cv2.resize(frame, (640, 480))

#         # Perform object detection
#         results = model(resized_frame)

#         # Extract detection results
#         for *box, conf, cls in results.xyxy[0]:
#             if conf > CONFIDENCE_THRESHOLD and int(cls) == 0:  # Class '0' is 'person' in COCO
#                 # Map bounding box coordinates back to original frame size
#                 x_min, y_min, x_max, y_max = map(int, box)
#                 x_min = int(x_min * (width / 640))  # Scale X coordinates
#                 x_max = int(x_max * (width / 640))
#                 y_min = int(y_min * (height / 480))  # Scale Y coordinates
#                 y_max = int(y_max * (height / 480))

#                 # Draw bounding box and label
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#                 label = f"Person: {conf:.2f}"
#                 cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Display the resulting frame
#         cv2.imshow("Video Feed", frame)

#         # Exit loop on pressing 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release video capture and close windows
#     cap.release()
#     cv2.destroyAllWindows()
#     # cap = cv2.VideoCapture(0)  # Use the default camera
#     # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 model

#     # while True:
#     #     success, frame = cap.read()
#     #     if not success:
#     #         break

#     #     try:
#     #         people_count, frame = detect_people(frame, model)
#     #     except Exception as e:
#     #         print(f"Error in detection: {e}")
#     #         break

#     #     _, buffer = cv2.imencode('.jpg', frame)
#     #     frame = buffer.tobytes()
#     #     yield (b'--frame\r\n'
#     #            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     # cap.release()

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True)
# import cv2
# import torch
# from flask import Flask, render_template, Response

# # Initialize Flask app
# app = Flask(__name__)

# # Load pre-trained YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # Function to generate video frames for streaming
# def generate_frames():
#     cap = cv2.VideoCapture(0)  # Initialize webcam
#     if not cap.isOpened():
#         raise RuntimeError("Error: Could not access the webcam.")

#     CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for detection

#     while True:
#         # Read frame from webcam
#         success, frame = cap.read()
#         if not success:
#             break

#         # Resize frame for YOLO model
#         resized_frame = cv2.resize(frame, (640, 480))

#         # Perform object detection
#         results = model(resized_frame)

#         # Draw bounding boxes on the frame
#         for *box, conf, cls in results.xyxy[0]:
#             if conf > CONFIDENCE_THRESHOLD and int(cls) == 0:  # Detecting persons
#                 x_min, y_min, x_max, y_max = map(int, box)
#                 # Draw bounding box
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#                 label = f"Person: {conf:.2f}"
#                 cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Encode the frame as a JPEG
#         ret, buffer = cv2.imencode('.jpg', frame)
#         if not ret:
#             continue  # Skip this frame if encoding fails

#         # Yield the frame as part of an HTTP response
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     cap.release()

# # Flask route to stream video
# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Flask route to render HTML template
# @app.route('/')
# def index():
#     return render_template('index.html')  # Add an 'index.html' file with a video tag for '/video_feed'

# # Run the Flask app
# if __name__ == "__main__":
#     app.run(debug=True)
# import cv2
# import torch
# from flask import Flask, render_template, Response
# from deepface import DeepFace  # For emotion detection

# # Initialize Flask app
# app = Flask(__name__)

# # Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # Function to detect emotions in a face
# # def detect_emotion(face_img):
# #     try:
# #         analysis = DeepFace.analyze(face_img, actions=["emotion"], enforce_detection=False)
# #         return analysis['dominant_emotion']
# #     except Exception as e:
# #         print(f"Emotion detection error: {e}")
# #         return "Unknown"


# def detect_emotion(face_img):
#     try:
#         # Analyze emotions
#         analysis = DeepFace.analyze(face_img, actions=["emotion"], enforce_detection=False)

#         # Handle both single-face and multi-face scenarios
#         if isinstance(analysis, list):  # Multiple faces
#             emotion = analysis[0]['dominant_emotion']  # Take the first face's emotion
#         else:  # Single face
#             emotion = analysis['dominant_emotion']

#         return emotion
#     except Exception as e:
#         print(f"Emotion detection error: {e}")
#         return "Unknown"
# # Function to generate video frames
# def generate_frames():
#     cap = cv2.VideoCapture(0)  # Open webcam
#     if not cap.isOpened():
#         raise RuntimeError("Error: Could not access the webcam.")

#     CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for YOLO model

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         # Resize frame for YOLOv5 model input
#         resized_frame = cv2.resize(frame, (640, 480))

#         # Perform object detection (detect people)
#         results = model(resized_frame)

#         # Count number of people and detect their emotions
#         person_count = 0

#         for *box, conf, cls in results.xyxy[0]:
#             if conf > CONFIDENCE_THRESHOLD and int(cls) == 0:  # Class 0 is "person"
#                 person_count += 1
#                 x_min, y_min, x_max, y_max = map(int, box)

#                 # Extract the face region for emotion detection
#                 face_img = frame[y_min:y_max, x_min:x_max]
#                 emotion = detect_emotion(face_img)

#                 # Draw bounding box and label on the frame
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#                 label = f"Person: {emotion}"
#                 cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Display the person count
#         cv2.putText(frame, f"People Count: {person_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         # Encode the frame as JPEG for streaming
#         ret, buffer = cv2.imencode('.jpg', frame)
#         if not ret:
#             continue  # Skip if encoding fails

#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     cap.release()

# # Flask route to stream video
# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Flask route to render HTML template
# @app.route('/')
# def index():
#     return render_template('index.html')  # HTML page with video feed

# # Run the Flask app
# if __name__ == "__main__":
#     app.run(debug=True)
