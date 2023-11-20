import cv2
import numpy as np
from flask import Flask, render_template, Response
from flask import request, jsonify
from flask_bootstrap import Bootstrap5
lock_state = False
app = Flask(__name__)
app.config['BOOTSTRAP_BOOTSWATCH_THEME'] = 'slate'
bootstrap = Bootstrap5(app)

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, threading=True)

# Global variables to store the parameters
circle_params = {
    "dp": 1,
    "minDist": 50,
    "param1": 100,
    "param2": 30,
    "minRadius": 10,
    "maxRadius": 100
}

@app.route('/toggle-lock', methods=['POST'])
def toggle_lock():
    global lock_state
    data = request.get_json()
    lock_state = data['locked']
    return jsonify({"locked": lock_state})

@app.route('/update-circle-params', methods=['POST'])
def update_circle_params():
    global circle_params
    data = request.json
    circle_params.update(data)
    return jsonify({"message": "Circle parameters updated", "params": circle_params})


def gen_frames():
    camera = cv2.VideoCapture(0)  # Use 0 for the first camera

    # Set the resolution of the webcam (replace WIDTH and HEIGHT with your webcam's resolution)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    circles = None
    old_circles = None
    frame_counter = 0
    generation = 0
    while True:
        success, frame = camera.read()

        if not success:
            break
        else:

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply Hough Circle Transform
            if np.double(circle_params['dp']) == 0:
                circle_params['dp'] = "0.1"
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
            if frame_counter % 10 == 0:
                frame_counter = 0
                if not lock_state:
                    generation += 1
                    circles = cv2.HoughCircles(image=gray,
                                               method=cv2.HOUGH_GRADIENT,
                                               dp=np.double(circle_params['dp']),
                                               minDist=np.double(circle_params['minDist']),
                                               param1=np.double(circle_params['param1']),
                                               param2=np.double(circle_params['param2']),
                                               minRadius=np.int16(circle_params['minRadius']),
                                               maxRadius=np.int16(circle_params['maxRadius']))
                    old_circles = circles
            else:
                if old_circles is not None:
                    circles = old_circles


            # Draw circles on the frame
            if circles is not None:
                print("drawing circles, generation: %d, frame count: %d" % (generation, frame_counter))
                circles_to_draw = np.round(circles[0, :]).astype("int")
                # circles_to_draw = circles[0:1]
                # circles_to_draw = sorted(circles, key=lambda x: x[2], reverse=True)[:2]  # Sort by radius and get top 2, don't want this

                for i, (x, y, r) in enumerate(circles_to_draw):
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                    text1 = f"Radius: {r}"
                    text1Size, _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    text1X = x - text1Size[0] // 2
                    text1Y = y + (text1Size[1] // 2) - 10
                    cv2.putText(frame, text1, (text1X, text1Y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    text2 = f"CircleID: {i}"
                    textSize, _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    text2X = x - textSize[0] // 2
                    text2Y = (y + textSize[1] // 2) + 10
                    cv2.putText(frame, text2, (text2X, text2Y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            frame_counter += 1
            # Convert the frame to bytes and yield
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
