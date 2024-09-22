from flask import Blueprint, render_template, Response, abort, redirect, url_for
from flask_login import login_required, current_user
from queue import Queue
from camera import Camera
from normal_vid_stream import NormalVideoStream
from detect_vid_stream import DetectionVideoStream

import cv2

main = Blueprint('main', __name__)

# Queues for both streams
framesNormalQue = Queue(maxsize=0)
framesDetectionQue = Queue(maxsize=0)
print('Queues created')

# RPi camera instance
camera = Camera(cv2.VideoCapture(0), framesNormalQue, framesDetectionQue)
camera.start()
print('Camera thread started')

# Streams
normalStream = NormalVideoStream(framesNormalQue)
detectionStream = DetectionVideoStream(framesDetectionQue)
print('Streams created')

normalStream.start()
print('Normal stream thread started')
detectionStream.start()
print('Detection stream thread started')

@main.route('/')
def index():
    return render_template('index.html')


@main.route('/video_stream/<int:stream_id>')
def video_stream(stream_id):
    if not current_user.is_authenticated:
        abort(403)

    print(f'Current user detection: {current_user.detectionState}')

    global detectionStream
    global normalStream

    stream = None

    if current_user.detectionState:
        stream = detectionStream
        print('Stream set to detection one')
    else:
        stream = normalStream
        print('Stream set to normal one')

    return Response(stream.gen(), mimetype='multipart/x-mixed-replace; boundary=frame')



@main.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

