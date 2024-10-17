import cv2
import mediapipe as mp
import numpy as np
from pynput import keyboard
import sys
import time
import torch
from model import CNN
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

mp_drawing = mp.solutions.drawing_utils ## type: ignore
mp_hands = mp.solutions.hands ## type: ignore

DRAW_COLOR = (0, 0, 255)  ## RED
DRAW_THICKNESS = 4
MAX_NUM_HANDS = 1

FRAME_WIDTH = 1280         
FRAME_HEIGHT = 720        

PROCESS_EVERY_N_FRAMES = 2
DETECTION_FRAME_WIDTH = 640
DETECTION_FRAME_HEIGHT = 360

THUMBNAIL_WIDTH = 200
THUMBNAIL_HEIGHT = 150

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
canvas_history = [canvas.copy()]
redo_history = []

prev_x, prev_y = None, None

drawing = False
clear_canvas = False
undo_action = False
redo_action = False
submit_action = False

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)  ## White
line_type = 2

shift_pressed = False

submitted_drawing = None
predicted_char = None 

model = None
transform = None
char_to_index = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FONT_PATH = r"NotoSansJP-VariableFont_wght.ttf"  ## Replace with the path to a Japanese font file if you want to use a different one from the one included in the repo.
FONT_SIZE = 30

def load_model():
    global model, transform, char_to_index, device
    
    char_to_index = torch.load("char_to_index.pth")
    num_classes = len(char_to_index)
    
    model = CNN(num_classes=num_classes)
    state_dict = torch.load("etl8b_model.pth", map_location=device)
    
    ## Remove fc1 from state_dict if it exists
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc1')}
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device) 
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

def predict(image):
    global model, transform, char_to_index, device
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    pil_image = Image.fromarray(gray_image)
    
    tensor_image = transform(pil_image).unsqueeze(0).to(device) ## type: ignore
    
    with torch.no_grad():
        output = model(tensor_image) ## type: ignore
        _, predicted = torch.max(output, 1)
    
    ## Convert predicted index back to character
    index_to_char = {v: k for k, v in char_to_index.items()} ## type: ignore
    return index_to_char[predicted.item()]

def render_japanese_text(text, size):
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    image = Image.new("RGB", size, (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font, fill=(255, 255, 255))
    return np.array(image)

def on_press(key):
    global drawing, clear_canvas, undo_action, redo_action, shift_pressed, submit_action
    if(key == keyboard.KeyCode.from_char('d')):
        drawing = True
    elif(key == keyboard.KeyCode.from_char('c')):
        clear_canvas = True
    elif(key == keyboard.KeyCode.from_char('Z')):  ## Capital Z indicates Shift+Z
        redo_action = True
    elif(key == keyboard.KeyCode.from_char('z')):
        undo_action = True
    elif(key == keyboard.Key.shift):
        shift_pressed = True
    elif(key == keyboard.KeyCode.from_char('q')):
        return False
    elif(key == keyboard.KeyCode.from_char('s')):
        submit_action = True

def on_release(key):
    global drawing, shift_pressed
    if(key == keyboard.KeyCode.from_char('d')):
        drawing = False
    elif(key == keyboard.Key.shift):
        shift_pressed = False

listener = keyboard.Listener(on_press=on_press, on_release=on_release) ## type: ignore
listener.start()

with mp_hands.Hands(
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    frame_count = 0
    load_model()
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
                            canvas_history.append(canvas.copy())
                            redo_history.clear()  ## Clear redo history when a new action is performed
                        prev_x, prev_y = x, y
                    else:
                        prev_x, prev_y = None, None

                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                prev_x, prev_y = None, None

        combined = cv2.add(frame, canvas)

        ## Display predicted character above the submitted drawing
        if(predicted_char is not None):
            char_image = render_japanese_text(f"Predicted: {predicted_char}", (200, 50))
            combined[THUMBNAIL_HEIGHT:THUMBNAIL_HEIGHT+50, :200] = char_image

        ## Display submitted drawing
        if(submitted_drawing is not None):
            combined[:THUMBNAIL_HEIGHT, :THUMBNAIL_WIDTH] = submitted_drawing

        status_text = "Drawing Mode: ON" if drawing else "Drawing Mode: OFF"
        cv2.putText(combined, status_text, (10, frame_height - 60), font, font_scale, font_color, line_type)

        instructions = "Hold 'd' to draw | 'c' to clear | 'z' to undo | 'Shift+Z' to redo | 's' to submit | 'q' to quit"
        cv2.putText(combined, instructions, (10, frame_height - 20), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(window_name, combined)

        if(clear_canvas):
            canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            canvas_history = [canvas.copy()]
            redo_history.clear()
            submitted_drawing = None  
            predicted_char = None  
            clear_canvas = False

        if(undo_action):
            if(len(canvas_history) > 1):
                redo_history.append(canvas_history.pop())
                canvas = canvas_history[-1].copy()
            undo_action = False

        if(redo_action):
            if(redo_history):
                canvas = redo_history.pop()
                canvas_history.append(canvas.copy())
            redo_action = False

        if(submit_action):
            if(np.any(canvas != 0)):  ## Only submit if canvas is not empty
                submitted_drawing = cv2.resize(canvas, (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT))

                predicted_char = predict(canvas)

                print(f"Predicted character: {predicted_char}")
                
                canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                canvas_history = [canvas.copy()]
                redo_history.clear()
            submit_action = False

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

        ## Limit the frame rate
        elapsed_time = time.time() - start_time
        sleep_time = max(1/30 - elapsed_time, 0)
        time.sleep(sleep_time)

cap.release()
cv2.destroyAllWindows()
listener.stop()
