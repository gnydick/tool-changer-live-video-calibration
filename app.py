import concurrent.futures
import os
import queue
import signal
import threading
import time

import cv2
import numpy as np
import requests
from flask import Flask, render_template, Response
from flask import request, jsonify
from flask_bootstrap import Bootstrap5

camera = None

app = Flask(__name__)
# Global queue
global_queue = queue.Queue()


# Mock function to send the request to the downstream system
def send_downstream(data):
    print(f"Sending: {data}")
    time.sleep(1)  # Simulate network delay or processing time


# Worker function that batches messages with start and end sequences
def worker(batch_size=1, timeout=.5):
    batch = []
    last_send_time = time.time()

    while True:
        try:
            # Get a message from the queue with a timeout
            message = global_queue.get()
            batch.append(message)

            # Send the batch if it's full
            if len(batch) >= batch_size:
                send_batch(batch)
                batch.clear()  # Clear the batch after sending
                last_send_time = time.time()

        except queue.Empty:
            # Timeout reached, send the current batch if not empty
            if batch:
                send_batch(batch)
                batch.clear()
                last_send_time = time.time()


# Helper function to send a batch
def send_batch(batch):
    requests.get('http://duet3.local/rr_gcode?gcode=G91')
    for message in batch:
        requests.get('http://duet3.local/rr_gcode?gcode=G1 %s%f' % (message[0], message[1]))

    requests.get('http://duet3.local/rr_gcode?gcode=G90')
    print(f"Sent batch of {len(batch)} messages.")


# Example function to add messages to the queue
def add_message(message):
    global_queue.put(message)


# Start the worker thread
thread = threading.Thread(target=worker, daemon=True)
thread.start()


def get_current_position():
    resp = requests.get('http://duet3.local/rr_model?key=move.axes[].machinePosition')
    return resp.json()['result']


def get_input_relative(input_name):
    resp = requests.get('http://duet3.local/rr_model?key=inputs[]')
    inputs = resp.json()['result']
    for input in inputs:
        return input['axesRelative']


@app.route('/jog', methods=['POST'])
def jog_wheel():
    data = request.get_json()
    delta = data.get('delta')
    axis = data.get('axis')

    # Process the jog wheel input (e.g., adjust position based on delta)
    print(f"Jog wheel turned {axis}: {delta}")
    add_message([axis, delta])
    return jsonify({'status': 'success', 'delta': delta})


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
        pass
        # print("Camera opened.")


def close_camera():
    global camera
    if camera is not None:
        camera.release()  # Close the camera
        # print("Camera closed.")


# Signal handler to catch termination (SIGTERM)
def handle_signal(signum, frame):
    # print(f"Received signal {signum}, shutting down.")
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
        except:
            pass
    return resized


def process_circles(frame):
    global circles, circle_params, frame_counter, generation, old_circles, zoom_level, minRad, maxRad
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
    if np.double(circle_params['center_precision']) == 0:
        circle_params['center_precision'] = 10
    if np.double(circle_params['reticle_rad']) == 0:
        circle_params['reticle_rad'] = 300
    # print("Frame #: %d" % frame_counter)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray, (9, 9), 2)
    blurred_like_frame = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2

    # Create an empty heatmap (same size as the image)
    heatmap = np.zeros_like(gray)
    timeout_duration = 2  # seconds
    # Combine heatmap with the original image for visualization
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # frame = cv2.addWeighted(blurred_like_frame, 0.7, heatmap_colored, 0.3, 0)
    if frame_counter % 2 == 0:
        frame_counter = 0
        if not lock_state:
            generation += 1

            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(hough_detect, circle_params, blurred_image, generation, lock)
                    circles = future.result(timeout=timeout_duration)
                    # print("Circles: %s" % circles)

                    # # For each detected circle, increase the heatmap intensity
                    if circles is not None:
                        coicles = np.round(circles[0, :]).astype("int")
                        for i, (circleX, circleY, r) in enumerate(coicles):
                            r = int(r / zoom_level)
                            # Draw a filled circle on the heatmap
                            cv2.circle(heatmap, (circleX, circleY), r, (255, 255, 255), thickness=-1)

                    if circles is not None:
                        old_circles = circles

            except concurrent.futures.TimeoutError:
                print(f"Hough operation exceeded {timeout_duration} seconds and was stopped.")

    # Draw circles on the frame
    mind_dist = np.sqrt(width ** 2 + height ** 2)
    # Draw a vertical line (from top to bottom at the center)
    cv2.line(frame, (center_x, 0), (center_x, height), (0, 0, 255), 2)

    # Draw a horizontal line (from left to right at the center)
    cv2.line(frame, (0, center_y), (width, center_y), (0, 0, 255), 2)

    if circles is not None:
        circles_to_draw = np.round(circles[0, :]).astype("int")
        one_circle = None
        for i, (circleX, circleY, r) in enumerate(circles_to_draw):
            drawn_rad = int(round(r * zoom_level))
            text_rad = r
            distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            if drawn_rad >= minRad * zoom_level and drawn_rad <= maxRad * zoom_level:
                if distance_from_center < mind_dist:
                    mind_dist = distance_from_center
                    one_circle = i

        # circles_to_draw = circles[0:1]

        cv2.putText(frame, 'Found %d circles' % len(circles_to_draw), (0, 210), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (0, 0, 255), 5)

        for i, (circleX, circleY, r) in enumerate(circles_to_draw):
            # if i == one_circle:
            x_offset = abs(circleX - center_x)
            y_offset = abs(circleY - center_y)
            if abs(circleX - center_x) <= int(circle_params['center_precision']) and abs(circleY - center_y) <= int(
                    circle_params['center_precision']):
                cv2.putText(frame, 'Circle is %d, and %d pixels from center' % (x_offset, y_offset), (0, 320),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
                drawn_rad = int(round(r * zoom_level))
                text_rad = r
                # drawn_rad = int(round(r * zoom_level))
                print("Circle %d, radius: %d" % (i, text_rad))
                cv2.circle(frame, (circleX, circleY), drawn_rad, (0, 255, 0), 4)
                text1 = f"Radius: {text_rad}"
                text1Size, _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                text1X = circleX - text1Size[0] // 2
                text1Y = circleY + (text1Size[1] // 2) - 40
                cv2.putText(frame, text1, (text1X, text1Y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # Draw a vertical line (from top to bottom at the center)
    cv2.line(frame, (center_x, 0), (center_x, height), (0, 0, 255), 2)
    cv2.circle(frame, (center_x, center_y), int(circle_params['reticle_rad']), (255, 0, 255), 4)
    # Draw a horizontal line (from left to right at the center)
    cv2.line(frame, (0, center_y), (width, center_y), (0, 0, 255), 2)
    cv2.putText(frame, 'Center of target: (%d, %d)' % (center_x, center_y), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 3,
                (0, 200, 0), 5)
    return frame


def hough_detect(cp, blurred_image, gen, lock):
    global minRad, maxRad, zoom_level
    with lock:
        # print("Start Hough Detect: %d" % generation)
        edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150, apertureSize=5, L2gradient=True)
        circ = cv2.HoughCircles(edges,
                                method=cv2.HOUGH_GRADIENT,
                                dp=np.double(cp['dp']),
                                minDist=np.double(float(cp['minDist']) * zoom_level),
                                param1=np.double(cp['param1']),
                                param2=np.double(cp['param2']),
                                minRadius=minRad,
                                maxRadius=maxRad)
        # print("End Hough Detect: %d" % gen)
        return circ


def gen_frames():
    global lock, pan_distance, camera, output_frame, frame_counter, generation, zoom_level, minRad, maxRad
    print("gen_frames")
    while True:
        time.sleep(.2)
        # print("start camera read")
        success, frame_output = camera.read()
        # print("stop camera read")
        if success:
            frame = cv2.flip(frame_output, 0)

            # print("success")
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
        # with lock:
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

    lock_state = False

    app.config['BOOTSTRAP_BOOTSWATCH_THEME'] = 'slate'
    bootstrap = Bootstrap5(app)
    zoom_called = False
    zoom_level = 1.0
    pan_x, pan_y, old_pan_x, old_pan_y = 0, 0, 0, 0

    width = 3840
    height = 2160
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
        "center_precision": 10,
        "reticle_rad": 300,
        "dp": 2,
        "minDist": 100,
        "param1": 50,
        "param2": 80,
        "minRadius": 230,
        "maxRadius": 260
    }

    output_frame = None
    frame_counter = 1
    generation = 0
    circles = None
    old_circles = None
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
