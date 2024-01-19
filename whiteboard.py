import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Function to check if a finger is up
def is_finger_up(landmarks, tip_idx, dip_idx):
    return landmarks[tip_idx].y < landmarks[dip_idx].y

# Function to check if all fingers are extended
def are_all_fingers_extended(landmarks):
    return all(is_finger_up(landmarks, tip_idx, dip_idx) for tip_idx, dip_idx in [
        (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_DIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_DIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_DIP),
    ])

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Canvas setup
canvas_width, canvas_height = 640, 480
canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

# Previous index finger tip location
prev_x, prev_y = 0, 0

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Get the tip and dip landmarks of the index finger
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]

        # Convert the coordinates to pixel values
        x, y = int(index_tip.x * canvas_width), int(index_tip.y * canvas_height)

        # Draw if the index finger is up
        if is_finger_up(hand_landmarks.landmark, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_DIP):
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y
            cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), 5)
            prev_x, prev_y = x, y
        else:
            prev_x, prev_y = 0, 0

        # Clear canvas if all fingers are extended
        if are_all_fingers_extended(hand_landmarks.landmark):
            canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # Show the canvas
    cv2.imshow("Air Canvas", canvas)

    # Break the loop with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
