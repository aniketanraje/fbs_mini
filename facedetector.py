import cv2


camera = cv2.VideoCapture(0)


def gen_frames():  
    """
        This function will collect the input stream from the webcam 
        and will detect eyes and frontal face of the person who is facing 
        the web cam using Haar Cascades algorithm.
        It uses pre-trained models for face detection and eye detection
        For frontal Human face detection, it uses haarcascade_frontalface_default
        and for detecting eyes in a live stream, it will use haarcascade_eye.
    """
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            # if you do not get stream from the webcamera, 
            # the loop will end 
            break
        else:
            # this part will be executed as long as you get input from the 
            # web cam
            # here detector will be used to detect the face 
            # and eye_cascade will be used to detect the eyes
            detector=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")    
            faces=detector.detectMultiScale(frame,1.1,7)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                # draw rectangle around the eyes 
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


