import cv2
import numpy as np
from flask import Flask, render_template, Response
from flask import request, jsonify
from flask_bootstrap import Bootstrap5

from threading import Lock


lock = Lock()
circle_ml_counter = 0
circle_id = None
lock_state = False
app = Flask(__name__)
app.config['BOOTSTRAP_BOOTSWATCH_THEME'] = 'slate'
bootstrap = Bootstrap5(app)
zoom_called = False
zoom_level = 1.0
pan_x, pan_y, old_pan_x, old_pan_y = 0, 0, 0, 0
change_counter = 0
width = 1920
height = 1080
x = width / 2  # Initial pan coordinates
y = height / 2
pan_distance = 10
x1 = old_x1 = 0
y1 = old_y1 = 0
x2 = old_x2 = width / 2
y2 = old_y2 = height / 2


@app.route('/')
def index():
    global zoom_level, pan_distance
    return render_template('index.html', zoom_level=zoom_level, pan_distance=pan_distance)


if __name__ == '__main__':
    app.run(debug=False, threading=True)

# Global variables to store the parameters
circle_params = {
    "dp": 1,
    "minDist": 50,
    "max-area": 100,
    "param2": 30,
    "minRadius": 10,
    "maxRadius": 100
}


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
    global lock_state, circle_params, change_counter

    if not lock_state:
        data = request.json
        circle_params.update(data)
        change_counter += 1
        print('Change counter: %d, Process counter: %d' % (change_counter, circle_ml_counter))
        return jsonify({"message": "Circle parameters updated", "params": circle_params})


def process_frame(frame):
    global old_x2, old_y1, old_y2, old_x1, zoom_level, pan_x, pan_y, old_pan_x, old_pan_y, zoom_called
    height, width = frame.shape[:2]

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
        cropped = frame[int(y1):int(y2), int(x1):int(x2)]
        resized = cv2.resize(cropped, (width, height))

    return resized


def gen_frames():
    global zoom_level, pan_distance, circle_ml_counter, circle_id
    try:
        camera = cv2.VideoCapture(0)  # Use 0 for the first camera

        # Set the resolution of the webcam (replace WIDTH and HEIGHT with your webcam's resolution)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        circles = None
        old_circles = None
        frame_counter = 0
        generation = 0
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                frame = process_frame(frame)
                # Convert frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Apply Hough Circle Transform
                if np.double(circle_params['dp']) == 0:
                    circle_params['dp'] = "0.1"
                if np.double(circle_params['minDist']) == 0:
                    circle_params['minDist'] = "0.1"
                if np.double(circle_params['max-area']) == 0:
                    circle_params['max-area'] = "0.1"
                if np.double(circle_params['param2']) == 0:
                    circle_params['param2'] = "0.1"
                if np.double(circle_params['minRadius']) == 0:
                    circle_params['minRadius'] = "0.1"
                if np.double(circle_params['maxRadius']) == 0:
                    circle_params['maxRadius'] = "0.1"
                if frame_counter % 10 == 0:
                    frame_counter = 0
                    if not lock_state:
                        generation += 1
                        if lock.acquire(True):
                            try:
                                circle_ml_counter += 1
                                circles = cv2.HoughCircles(
                                    image=gray,
                                    method=cv2.HOUGH_GRADIENT,
                                    dp=np.double(circle_params['dp']),
                                    minDist=np.double(circle_params['minDist']),
                                    max-area=np.double(circle_params['max-area']),
                                    param2=np.double(circle_params['param2']),
                                    minRadius=np.int16(circle_params['minRadius']),
                                    maxRadius=np.int16(circle_params['maxRadius']))
                                old_circles = circles
                            finally:
                                lock.release()

                if old_circles is not None:
                    circles = old_circles

                # Draw circles on the frame
                if circles is not None:
                    circles_to_draw = np.round(circles[0, :]).astype("int")
                    if circle_id is not None:
                        circles_to_draw = np.round(circles[0,circle_id]).astype("int")

                    for i, (circleX, circleY, r) in enumerate(circles_to_draw):
                        cv2.circle(frame, (circleX, circleY), r, (0, 255, 0), 4)
                        text1 = f"Radius: {r}"
                        text1Size, _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                        text1X = circleX - text1Size[0] // 2
                        text1Y = circleY + (text1Size[1] // 2) - 40
                        cv2.putText(frame, text1, (text1X, text1Y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
                        text2 = f"CircleID: {i}"
                        textSize, _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                        text2X = circleX - textSize[0] // 2
                        text2Y = (circleY + textSize[1] // 2) + 40
                        cv2.putText(frame, text2, (text2X, text2Y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

            frame_counter += 1
            # Convert the frame to bytes and yield

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        camera.release()
    except Exception as e:
        print(e)
        camera.release()


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/pan', methods=['POST'])
def handle_pan():
    global pan_x, pan_y, zoom_level, old_pan_x, old_pan_y, lock_state, change_counter
    if not lock_state:
        data = request.json
        pan_dist = int(data["pan_distance"])
        direction = data["direction"]

        if direction == 'left':
            old_pan_x = pan_x
            pan_x -= pan_dist

        elif direction == 'right':
            old_pan_x = pan_x
            pan_x += pan_dist
        elif direction == 'up':
            old_pan_y = pan_y
            pan_y -= pan_dist
        elif direction == 'down':
            old_pan_y = pan_y
            pan_y += pan_dist

    # You might want to add boundary checks here to prevent pan_x and pan_y from going out of bounds
    change_counter += 1
    print('Change counter: %d, Process counter: %d' % (change_counter, circle_ml_counter))
    return jsonify({"pan_x": pan_x, "pan_y": pan_y})


@app.route('/zoom', methods=['POST'])
def zoom():
    global zoom_level, lock_state, zoom_called, change_counter
    zoom_called = True
    if not lock_state:
        data = request.json
        zoom_level = float(data["zoom_level"])
        # You might want to add boundary checks here to prevent pan_x and pan_y from going out of bounds
        zoom_called = False
        change_counter += 1
        print('Change counter: %d, Process counter: %d' % (change_counter, circle_ml_counter))
    return jsonify({"zoom_level": zoom_level})


# @app.route('/circle', methods=['POST'])
# def circle():
#     global lock_state, circle_id
#
#     if lock_state:
#         data = request.json
#         if len(data['circle_id']) > 0:
#             circle_id = int(data["circle_id"])
#             return jsonify({"circle_id": circle})
#         else:
#             return jsonify({"error": "no id provided"})
#     else:
#         return jsonify({"error": "must be locked"}, )
