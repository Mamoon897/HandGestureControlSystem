import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Key, Controller as KeyboardController
import time

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
    """ Check if a finger is extended. """
    return finger_tip.y < finger_dip.y

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip the image horizontally for a later selfie-view display
    # Convert the BGR image to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # Process the image and draw hand landmarks
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
                # Press left arrow key
                keyboard.press(Key.left)
                keyboard.release(Key.left)
                time.sleep(3)  # Sleep for 3 seconds
            elif is_finger_extended(pinky_tip, pinky_dip):
                # Press right arrow key
                keyboard.press(Key.right)
                keyboard.release(Key.right)
                time.sleep(3)  # Sleep for 3 seconds

            # Check if both index and middle fingers are extended
            if is_finger_extended(index_finger_tip, index_finger_dip) and is_finger_extended(middle_finger_tip, middle_finger_dip):
                # Simulate mouse click
                mouse.click(Button.left, 1)

    # Display the image
    cv2.imshow('Hand Tracking', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
