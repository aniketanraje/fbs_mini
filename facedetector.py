import cv2
from cascade_classifier import faceCascade
import sys 

mn 
sys.setrecursionlimit(1500)

def get_stream(cam_uri=0):
    stream = cv2.VideoCapture(cam_uri)
    return stream


def face_detector(cam_uri=0):
    video_capture = get_stream(cam_uri)
    while True:
        # Capture frame-by-frame
        working, frame = video_capture.read()
        ret, frame = video_capture.read()
        if not working:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                ret, buffer = cv2.imencode(".jpg", frame)
                frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
