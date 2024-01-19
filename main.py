import tkinter as tk
import cv2
import numpy as np
import mediapipe as mp
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Key, Controller as KeyboardController
import time
import pyautogui


# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Initialize mouse and keyboard controllers
mouse = MouseController()
keyboard = KeyboardController()

# Screen size for mouse movement
screen_width, screen_height = pyautogui.size()

# Capture video from webcam
cap = cv2.VideoCapture(0)

def is_finger_extended(finger_tip, finger_dip):
    """Check if a finger is extended."""
    return finger_tip.y < finger_dip.y

def is_pinky_extended(pinky_tip, pinky_dip, threshold=0.02):
    """Check if the pinky finger is extended."""
    return pinky_tip.y + threshold < pinky_dip.y

def process_hand_gesture_frame():
    success, image = cap.read()
    if not success:
        root.after(10, process_hand_gesture_frame)
        return

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmarks for thumb, pinky, index, and middle fingers
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            pinky_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]

            # Mouse control with index finger
            screen_x = np.interp(index_finger_tip.x, [0, 1], [0, screen_width])
            screen_y = np.interp(index_finger_tip.y, [0, 1], [0, screen_height])
            mouse.position = (screen_x, screen_y)

            # Check if thumb or pinky is extended
            if is_finger_extended(thumb_tip, thumb_ip):
                keyboard.press(Key.up)
                keyboard.release(Key.up)
                time.sleep(1)
            elif is_pinky_extended(pinky_tip, pinky_dip):
                keyboard.press(Key.down)
                # keyboard.release(Key.down)
                time.sleep(1)

            # Check if both index and middle fingers are extended
            if is_finger_extended(index_finger_tip, index_finger_dip) and is_finger_extended(middle_finger_tip, middle_finger_dip):
                mouse.click(Button.left, 1)

    root.after(10, process_hand_gesture_frame)

# Tkinter UI setup
root = tk.Tk()
root.geometry('700x500')
root.title('Hand Gesture Control')




def on_button1_click():
    process_hand_gesture_frame()

def air_canvas():
    cap_canvas = cv2.VideoCapture(0)

    canvas_width, canvas_height = 640, 480
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    prev_x, prev_y = 0, 0

    while True:
        success, frame = cap_canvas.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands_canvas.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            index_tip = hand_landmarks.landmark[mp_hands_canvas.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_tip.x * canvas_width), int(index_tip.y * canvas_height)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), 5)
            prev_x, prev_y = x, y

        # Add a key to break the loop for stopping the webcam
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_canvas.release()
    cv2.destroyAllWindows()




# Button to start hand gesture control
button1 = tk.Button(root, text='Start Hand Gesture Control Presentation', command=on_button1_click)
button1.pack(pady=10)
desc1 = tk.Label(root, text='These gestures wiill provide gestures to control the presentation')
desc1.pack(pady=5)

desc1 = tk.Label(root, text='____________________________________________________________', bg="light blue")
desc1.pack(pady=5)
# Additional button
button2 = tk.Button(root, text='Start Hand Gesture Control WhiteBoard', command=air_canvas)
button2.pack(pady=10)
desc2 = tk.Label(root, text='This will provide you gestures to write on whiteboard ')
desc2.pack(pady=5)

root.configure(bg='light blue')

root.mainloop()

# Release resources when UI is closed
cap.release()
cv2.destroyAllWindows()
