from flask import Flask, render_template, Response
import cv2
from facedetector import gen_frames


app=Flask(__name__)


@app.route('/')
def index():
    """
        This will render index page to the home of the application.
    """
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """
        This method will return live stream from 
        web cam as a response to the index page.
        THe stream will be detecting the frontal face 
        and eyea of the person facing the webcam.
    """
    return Response(gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/about")
def about():
    return render_template("about.html")

if __name__=='__main__':
    app.run(debug=True)