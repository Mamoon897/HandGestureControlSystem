import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Capture video from webcam
cap = cv2.VideoCapture(0)

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

            # Get landmarks
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Define gestures (simple example based on y-coordinate)
            if index_tip.y < thumb_tip.y and index_tip.y < pinky_tip.y:
                # Move mouse - map index finger tip position to screen position
                screen_width, screen_height = pyautogui.size()
                x, y = int(index_tip.x * screen_width), int(index_tip.y * screen_height)
                pyautogui.moveTo(x, y)
            elif thumb_tip.x < index_tip.x and thumb_tip.x < pinky_tip.x:
                # Scroll left
                pyautogui.scroll(-10)
            elif pinky_tip.x > thumb_tip.x and pinky_tip.x > index_tip.x:
                # Scroll right
                pyautogui.scroll(10)

    # Display the image
    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
