import cv2
import torch
import numpy as np
from flask import Flask, render_template, Response
from util.detect_people import detect_people
from util.detect_emotion import detect_emotion
# import moviepy.editor

app = Flask(__name__)

import sys
sys.path.append('./yolov5')  # Ensure Python imports from YOLOv5, not local utils

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, TryExcept

model = DetectMultiBackend('./yolov5/yolov5s.pt')


# Open Webcam
video_source = 0  # Change to IP Camera URL if needed
cap = cv2.VideoCapture(video_source)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people in the frame
        people_count, frame = detect_people(frame, model)

        # Detect emotions
        frame = detect_emotion(frame)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
