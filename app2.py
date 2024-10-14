import concurrent.futures
import os
import signal
import threading
import time
from typing import Any

import cv2
import numpy as np
from cv2 import Mat
from flask import Flask, render_template, Response
from flask import request, jsonify
from flask_bootstrap import Bootstrap5
from numpy import ndarray, dtype

camera = None
app = Flask(__name__)


@app.route('/')
def index():
    global zoom_level, pan_distance
    return render_template('index.html', zoom_level=zoom_level, pan_distance=pan_distance)



def open_camera(cam_num):
    global camera
    camera = cv2.VideoCapture(cam_num)
    if not camera.isOpened():
        print("Failed to open the camera.")
    else:
        print("Camera opened.")


def close_camera():
    global camera
    if camera is not None:
        camera.release()  # Close the camera
        print("Camera closed.")


# Signal handler to catch termination (SIGTERM)
def handle_signal(signum, frame):
    print(f"Received signal {signum}, shutting down.")
    close_camera()
    # Exit the program after cleaning up
    exit(0)


def process_frame(frame):
    global old_x2, old_y1, old_y2, old_x1, zoom_level, pan_x, pan_y, old_pan_x, old_pan_y, zoom_called, minRad, maxRad, zoom_level
    # print("entering process_frame")
    # print("frame size: %d"  % len(frame))
    zoom_level = float(circle_params['zoom'])

    height, width = frame.shape[:2]
    minRad = int(circle_params['minRadius'])
    maxRad = int(circle_params['maxRadius'])
    # Center of the frame
    center_x, center_y = width // 2, height // 2

    # New center based on pan
    new_center_x = center_x + pan_x
    new_center_y = center_y + pan_y

    new_dist_to_right_edge = width - new_center_x
    frame_width = width / 2 / zoom_level

    # Calculate the ROI (Region of Interest) based on zoom_level and pan
    x1 = max(new_center_x - width // (2 * zoom_level), 0)
    y1 = max(new_center_y - height // (2 * zoom_level), 0)
    x2 = min(new_center_x + width // (2 * zoom_level), width)
    y2 = min(new_center_y + height // (2 * zoom_level), height)

    if pan_x > 0:
        calc = (new_center_x + frame_width) <= width
        if calc:
            old_x1 = x1
            old_y1 = y1
            old_x2 = x2
            old_y2 = y2
            # Resize cropped image back to frame size to maintain the original frame size
            cropped = frame[int(y1):int(y2), int(x1):int(x2)]
            resized = cv2.resize(cropped, (width, height))
            old_pan_x = pan_x
            old_pan_y = pan_y
        else:
            pan_x = old_pan_x
            pan_y = old_pan_y
            cropped = frame[int(old_y1):int(old_y2), int(old_x1):int(old_x2)]
            resized = cv2.resize(cropped, (width, height))
    elif pan_x < 0:
        calc = (new_center_x - frame_width) >= 0
        # calc = (x2 >= (width / 2 / zoom_level))
        if calc:
            old_x1 = x1
            old_y1 = y1
            old_x2 = x2
            old_y2 = y2
            old_pan_x = pan_x
            old_pan_y = pan_y
            # Resize cropped image back to frame size to maintain the original frame size
            cropped = frame[int(y1):int(y2), int(x1):int(x2)]
            resized = cv2.resize(cropped, (width, height))
        else:
            pan_x = old_pan_x
            pan_y = old_pan_y
            cropped = frame[int(old_y1):int(old_y2), int(old_x1):int(old_x2)]
            resized = cv2.resize(cropped, (width, height))
    else:
        old_x1 = x1
        old_y1 = y1
        old_x2 = x2
        old_y2 = y2
        # Resize cropped image back to frame size to maintain the original frame size
        try:
            cropped = frame[int(y1):int(y2), int(x1):int(x2)]
            resized = cv2.resize(cropped, (width, height))
            return resized
        except:
            pass




def process_circles(frame):
    blurred_image = None
    thresh_image = None
    global circle_params, frame_counter, generation, old_keypoints, keypoints, keypoints, old_keypoints
    # Apply Hough Circle Transform
    if np.double(circle_params['zoom']) == 0:
        circle_params['zoom'] = "1.0"
    else:
        zoom_level = float(circle_params['zoom'])
    if np.double(circle_params['dp']) == 0:
        circle_params['dp'] = "1.0"
    if np.double(circle_params['minDist']) == 0:
        circle_params['minDist'] = "0.1"
    if np.double(circle_params['param1']) == 0:
        circle_params['param1'] = "0.1"
    if np.double(circle_params['param2']) == 0:
        circle_params['param2'] = "0.1"
    if np.double(circle_params['minRadius']) == 0:
        circle_params['minRadius'] = "0.1"
    if np.double(circle_params['maxRadius']) == 0:
        circle_params['maxRadius'] = "0.1"
    print("Frame #: %d" % frame_counter)
    if frame_counter % 2 == 0:
        frame_counter = 0
        if not lock_state:
            generation += 1
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh_image = cv2.threshold(gray_image, blob_params.minThreshold, 255, cv2.THRESH_BINARY)
            blurred_image = cv2.GaussianBlur(thresh_image, (9, 9), 2)
            # blurred_like_frame = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)


            timeout_duration = 2  # seconds
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(blob_detect, blurred_image, generation)
                    keypoints = future.result(timeout=timeout_duration)
                    # print("Circles: %s" % circles)
                    if keypoints is not None:
                        for keypoint in keypoints:
                            x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
                            radius = int(keypoint.size / 2)

                            # Draw a circle around each keypoint with thickness 2
                            cv2.circle(frame, (x, y), radius, (0, 0, 255), thickness=10)
                            # Calculate the area of the blob
                            area = np.pi * (radius ** 2)

                            # Display the area next to the blob
                            area_text = f"Area: {int(area/zoom_level)}"
                            cv2.putText(frame, area_text, (x + radius + 10, y), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 255, 0), 2)

                    # Add debug information on the frame
                    blob_count = len(keypoints)
                    print(blob_count)
                    text = f"Circles found: {blob_count}"
                    frame: Mat | ndarray[Any, dtype[Any]] | ndarray = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
                                                                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    # Put the text on the frame
                    cv2.putText(frame, text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
                    return frame
            except concurrent.futures.TimeoutError:
                print(f"Hough operation exceeded {timeout_duration} seconds and was stopped.")
        old_keypoints = keypoints
    else:
        if old_keypoints is not None:
            keypoints = old_keypoints
    return frame

def blob_detect(blurred_image, gen):
    global minRad, maxRad, zoom_level
    print("Start blob detector: %d" % generation)
    # Detect blobs
    keypoints = detector.detect(blurred_image)

    print("End blob detect: %d" % gen)
    return keypoints


def gen_frames():
    global lock, pan_distance, camera, output_frame, frame_counter, generation, zoom_level, minRad, maxRad
    print("gen_frames")
    while True:
        time.sleep(.2)
        print("start camera read")
        success, frame = camera.read()
        print("stop camera read")
        if success:
            frame = cv2.flip(frame, 0)

            print("success")
            frame = process_frame(frame)
            frame = process_circles(frame)

            frame_counter += 1
            # Convert the frame to bytes and yield

            ret, buffer = cv2.imencode('.jpg', frame)
            output_frame = buffer.tobytes()
            # print("output_frame len: %d" % len(output_frame))
            # camera.release()

def generate():
    global output_frame, lock
    while True:
        time.sleep(1)
        with lock:
            if output_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/pan', methods=['POST'])
def handle_pan():
    global pan_x, pan_y, zoom_level, old_pan_x, old_pan_y, lock_state
    if not lock_state:
        data = request.json
        pan_distance = int(data["pan_distance"])
        direction = data["direction"]

        if direction == 'left':
            old_pan_x = pan_x
            pan_x -= pan_distance

        elif direction == 'right':
            old_pan_x = pan_x
            pan_x += pan_distance
        elif direction == 'up':
            old_pan_y = pan_y
            pan_y -= pan_distance
        elif direction == 'down':
            old_pan_y = pan_y
            pan_y += pan_distance

    # You might want to add boundary checks here to prevent pan_x and pan_y from going out of bounds

    return jsonify({"pan_x": pan_x, "pan_y": pan_y})




@app.route('/toggle-lock', methods=['POST'])
def toggle_lock():
    global lock_state
    lock_state = not lock_state

    if lock_state:
        return jsonify({"locked_state": "Locked"})
    else:
        return jsonify({"locked_state": "Unlocked"})


@app.route('/update-circle-params', methods=['POST'])
def update_circle_params():
    global lock_state, circle_params

    data = request.json
    circle_params.update(data)
    return jsonify({"message": "Circle parameters updated", "params": circle_params})


# This code runs for both auto-reloader and main process.
# Check if this is the main process (and not the auto-reloader process).
if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    # Register signal handlers to close the camera when the process is terminating
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Set up the SimpleBlobDetector parameters
    blob_params = cv2.SimpleBlobDetector_Params()


    # blob_params.filterByCircularity = True
    # blob_params.filterByConvexity = True

    blob_params.collectContours = True
    blob_params.filterByInertia = False
    # blob_params.minInertiaRatio = 0.8

    blob_params.filterByConvexity = False
    # blob_params.minConvexity = 0.9  # Set this closer to 1 for convex shapes

    # Filter by Area (optional, depending on your expected circle sizes)
    blob_params.filterByArea = True
    blob_params.minArea = 20000  # Minimum area of blobs to be detected
    blob_params.maxArea = 500000
    # # Filter by Circularity (focus on circular shapes)
    # blob_params.filterByCircularity = False
    blob_params.minCircularity = 0.3  # Adjust depending on how "circular" your blobs are

    blob_params.minThreshold = 1
    blob_params.maxThreshold = 20000

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(blob_params)


    lock_state = False

    app.config['BOOTSTRAP_BOOTSWATCH_THEME'] = 'slate'
    bootstrap = Bootstrap5(app)
    zoom_called = False
    zoom_level = 1.0
    pan_x, pan_y, old_pan_x, old_pan_y = 0, 0, 0, 0

    width = 1920
    height = 1080
    x = width / 2  # Initial pan coordinates
    y = height / 2
    pan_distance = 10
    x1 = old_x1 = 0
    y1 = old_y1 = 0
    x2 = old_x2 = width / 2
    y2 = old_y2 = height / 2
    # Global variables to store the parameters
    circle_params = {
        "zoom": 1,
        "dp": 2,
        "minDist": 100,
        "param1": 50,
        "param2": 50,
        "minRadius": 230,
        "maxRadius": 260
    }

    output_frame = None
    frame_counter = 1
    generation = 0
    keypoints = None
    old_keypoints = None
    minRad = None
    maxRad = None
    lock = threading.Lock()

    time.sleep(4)
    # Set the resolution of the webcam (replace WIDTH and HEIGHT with your webcam's resolution)
    open_camera(4)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    app.run(debug=True, threaded=True)

    trd = threading.Thread(target=gen_frames)
    trd.daemon = True
    trd.start()

