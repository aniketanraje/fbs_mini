import cv2

def get_stream(cam_uri=0):
    """
    It will return the stream from the webcamera
    by default the webcamera of the device.
    """
    
    stream = cv2.VideoCapture(cam_uri)
    return stream


def webcam_stream(cam_uri=0):
    """
    This function will help 
    to get a stream of video from webcam.
    """

    while True:
        stream = get_stream(cam_uri)
        working, frame = stream.read()
        if not working:
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')