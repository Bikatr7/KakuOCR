import cv2
import mediapipe as mp
import numpy as np
import keyboard  
import sys

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

DRAW_COLOR = (0, 0, 255)  ## RED
DRAW_THICKNESS = 4
MAX_NUM_HANDS = 1

## Update these later?
FRAME_WIDTH = 1280         
FRAME_HEIGHT = 720        


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

## Create a full-screen canvas for drawing
canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

prev_x, prev_y = None, None

drawing = False

## Font settings for displaying text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)  ## White
line_type = 2

with mp_hands.Hands(
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    while(True):
        ret, frame = cap.read()
        if(not ret):
            print("Error: Failed to grab frame.")
            break

        ## Resize frame for performance (may need looking at later)
        frame = cv2.resize(frame, (frame_width, frame_height))

        ## Flip the frame horizontally for natural (mirror) viewing
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ## process frame and detect hands
        results = hands.process(frame_rgb)

        drawing = keyboard.is_pressed('d')

        if(results.multi_hand_landmarks):
            for hand_landmarks in results.multi_hand_landmarks:
                ## Get the tip of the index finger (landmark 8)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * frame_width), int(index_finger_tip.y * frame_height)

                if(drawing):
                    if(prev_x is None and prev_y is None):
                        prev_x, prev_y = x, y
                    else:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), DRAW_COLOR, DRAW_THICKNESS)
                        prev_x, prev_y = x, y
                else:
                    ## Reset previous positions when not drawing
                    prev_x, prev_y = None, None

                ## Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            ## Reset previous positions when no hand is detected
            prev_x, prev_y = None, None

        # Combine the frame and the canvas
        # Overlay the canvas on the frame without using addWeighted for better performance (still needs way better perff)
        combined = cv2.add(frame, canvas)

        status_text = "Drawing Mode: ON" if drawing else "Drawing Mode: OFF"
        cv2.putText(combined, status_text, (10, 30), font, font_scale, font_color, line_type)

        instructions = "Hold 'd' to draw | Press 'c' to clear | Press 'q' to quit."
        cv2.putText(combined, instructions, (10, frame_height - 20), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        ## Display the resulting frame
        cv2.imshow(window_name, combined)

        if(keyboard.is_pressed('c')):
            ## Clear the canvas
            canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            
            ## to prevent multiple processes
            while(keyboard.is_pressed('c')):
                pass

        if(keyboard.is_pressed('q')):
            break

        ## Limit the frame rate to improve performance
        ## cv2.waitKey(1) is not needed for 'd' detection but required to allow OpenCV to process window events
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()