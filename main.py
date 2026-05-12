import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import pygetwindow as gw
from mss import mss
from collections import deque

# --- MediaPipe Hands Setup ---
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
except (AttributeError, ImportError):
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_draw

# Reduced confidence slightly for better detection in low light
hands = mp_hands.Hands(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5, 
    max_num_hands=1
)

# --- Configuration ---
WIDTH = 1280
HEIGHT = 720
COOLDOWN_TIME = 0.35 # Slightly faster cooldown
last_action_time = 0

# Swipe Sensitivity (Lower = more sensitive)
SWIPE_THRESHOLD = 0.12 # Threshold for total displacement in buffer

# Buffer for smoothing and better gesture detection
BUFFER_SIZE = 5
pos_buffer = deque(maxlen=BUFFER_SIZE)

# Auto-Restart
AUTO_RESTART = True
last_restart_check = 0

def find_game_window():
    windows = gw.getWindowsWithTitle('Subway Surfers')
    if windows: return windows[0]
    return None

# --- Main Logic ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
# Try to increase FPS
cap.set(cv2.CAP_PROP_FPS, 60)

sct = mss()

print("Subway Surfers SWIPE Controller v2.0")
print("Optimization: Position Buffering & Low-Confidence Support")
print("Flick your finger quickly to trigger actions.")
print("Press 'Q' to Quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    current_time = time.time()
    
    # 1. Game Preview Overlay
    game_window = find_game_window()
    if game_window and game_window.width > 100:
        try:
            monitor = {"top": game_window.top, "left": game_window.left, "width": game_window.width, "height": game_window.height}
            game_screenshot = sct.grab(monitor)
            game_img = np.array(game_screenshot)
            game_img = cv2.cvtColor(game_img, cv2.COLOR_BGRA2BGR)
            pip_w, pip_h = 300, int(300 * (game_window.height / game_window.width))
            game_pip = cv2.resize(game_img, (pip_w, pip_h))
            frame[10:10+pip_h, WIDTH-10-pip_w:WIDTH-10] = game_pip
            cv2.rectangle(frame, (WIDTH-10-pip_w, 10), (WIDTH-10, 10+pip_h), (0, 255, 0), 2)
        except: pass

    # 2. Advanced Gesture Detection
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Use Index Finger Tip (8)
            curr_pos = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pos_buffer.append((curr_pos.x, curr_pos.y))
            
            # Draw tracking dot & Path
            ix, iy = int(curr_pos.x * WIDTH), int(curr_pos.y * HEIGHT)
            cv2.circle(frame, (ix, iy), 8, (0, 255, 255), -1)

            # Need at least a few frames to detect a swipe trend
            if len(pos_buffer) == BUFFER_SIZE:
                # Calculate displacement from first to last point in buffer
                start_p = pos_buffer[0]
                end_p = pos_buffer[-1]
                
                dx = end_p[0] - start_p[0]
                dy = end_p[1] - start_p[1]
                
                # Check for Swipe
                if current_time - last_action_time > COOLDOWN_TIME:
                    action_triggered = False
                    
                    if abs(dx) > abs(dy) and abs(dx) > SWIPE_THRESHOLD:
                        if dx < 0:
                            pyautogui.press('left')
                            cv2.putText(frame, "<< LEFT", (ix-120, iy), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
                        else:
                            pyautogui.press('right')
                            cv2.putText(frame, "RIGHT >>", (ix+40, iy), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
                        action_triggered = True
                    elif abs(dy) > abs(dx) and abs(dy) > SWIPE_THRESHOLD:
                        if dy < 0:
                            pyautogui.press('up')
                            cv2.putText(frame, "^^ JUMP", (ix-40, iy-60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        else:
                            pyautogui.press('down')
                            cv2.putText(frame, "vv SLIDE", (ix-40, iy+60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        action_triggered = True
                    
                    if action_triggered:
                        last_action_time = current_time
                        pos_buffer.clear() # Reset buffer after action to avoid double triggers
    else:
        pos_buffer.clear() # Clear if hand is lost

    # UI / Feedback
    cv2.putText(frame, "Subway Surfers Controller PRO v2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Optimization: Multiframe Buffering Active", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    if current_time - last_action_time < COOLDOWN_TIME:
        cv2.putText(frame, "COOLDOWN", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "READY", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Subway Surfers Advanced Control', frame)
    
    key = cv2.waitKey(1) & 0xFF # Faster waitKey for better FPS
    if key == ord('q'): break
    elif key == ord(' '): pyautogui.press('space')

cap.release()
cv2.destroyAllWindows()
