from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')
camera = cv2.VideoCapture(0)
# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# Initialize the webcam
cap = cv2.VideoCapture(0)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.js')


def gen():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break

        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(framergb)

        # print(result)

        className = ''

        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                # Predict gesture
                prediction = model.predict([landmarks])
                # print(prediction)
                classID = np.argmax(prediction)
                className = classNames[classID]
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        # Show the final output
        cv2.imshow("Output", frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
