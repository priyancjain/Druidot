import cv2
from fer import FER

emotion_detector = FER(mtcnn=True)

def detect_emotion(frame):
    emotions = emotion_detector.detect_emotions(frame)

    for emotion in emotions:
        (x, y, w, h) = emotion['box']
        dominant_emotion = max(emotion['emotions'], key=emotion['emotions'].get)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, dominant_emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame
