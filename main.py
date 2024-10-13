import cv2
import mediapipe as mp
import numpy as np
from pynput import keyboard
import sys
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

DRAW_COLOR = (0, 0, 255)  ## RED
DRAW_THICKNESS = 4
MAX_NUM_HANDS = 1

FRAME_WIDTH = 1280         
FRAME_HEIGHT = 720        

PROCESS_EVERY_N_FRAMES = 2
DETECTION_FRAME_WIDTH = 640
DETECTION_FRAME_HEIGHT = 360

cap = cv2.VideoCapture(0)
if(not cap.isOpened()):
    print("Error: Could not open webcam.")
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

ret, frame = cap.read()
if(not ret):
    print("Error: Could not read frame from webcam.")
    cap.release()
    sys.exit()

frame_height, frame_width, _ = frame.shape

window_name = 'KakuOCR'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

prev_x, prev_y = None, None

drawing = False
clear_canvas = False

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)  ## White
line_type = 2

def on_press(key):
    global drawing, clear_canvas
    if(key == keyboard.KeyCode.from_char('d')):
        drawing = True
    elif(key == keyboard.KeyCode.from_char('c')):
        clear_canvas = True
    elif(key == keyboard.KeyCode.from_char('q')):
        return False

def on_release(key):
    global drawing
    if(key == keyboard.KeyCode.from_char('d')):
        drawing = False

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

with mp_hands.Hands(
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    frame_count = 0
    while(True):
        start_time = time.time()
        
        ret, frame = cap.read()
        if(not ret):
            print("Error: Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        
        frame_count += 1
        if(frame_count % PROCESS_EVERY_N_FRAMES == 0):
            detection_frame = cv2.resize(frame, (DETECTION_FRAME_WIDTH, DETECTION_FRAME_HEIGHT))
            detection_frame_rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
            results = hands.process(detection_frame_rgb)

            if(results.multi_hand_landmarks):
                for hand_landmarks in results.multi_hand_landmarks:
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x, y = int(index_finger_tip.x * frame_width), int(index_finger_tip.y * frame_height)

                    if(drawing):
                        if(prev_x is not None and prev_y is not None):
                            cv2.line(canvas, (prev_x, prev_y), (x, y), DRAW_COLOR, DRAW_THICKNESS)
                        prev_x, prev_y = x, y
                    else:
                        prev_x, prev_y = None, None

                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                prev_x, prev_y = None, None

        combined = cv2.add(frame, canvas)

        status_text = "Drawing Mode: ON" if drawing else "Drawing Mode: OFF"
        cv2.putText(combined, status_text, (10, 30), font, font_scale, font_color, line_type)

        instructions = "Hold 'd' to draw | Press 'c' to clear | Press 'q' to quit."
        cv2.putText(combined, instructions, (10, frame_height - 20), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(window_name, combined)

        if(clear_canvas):
            canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            clear_canvas = False

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

        ## Limit the frame rate
        elapsed_time = time.time() - start_time
        sleep_time = max(1/30 - elapsed_time, 0)
        time.sleep(sleep_time)

cap.release()
cv2.destroyAllWindows()
listener.stop()
