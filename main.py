import sys
import os
import cv2
import time
import threading
import webbrowser
import numpy as np
import pyautogui
from dotenv import load_dotenv
from detectors import DETECTORS

# --- Load Config ---
load_dotenv()

CONFIG = {
    "control_mode": os.getenv("CONTROL_MODE", "finger").lower(),
    "swipe_threshold": float(os.getenv("SWIPE_THRESHOLD", "0.12")),
    "cooldown_time": float(os.getenv("COOLDOWN_TIME", "0.35")),
    "buffer_size": int(os.getenv("BUFFER_SIZE", "5")),
    "detection_confidence": float(os.getenv("DETECTION_CONFIDENCE", "0.5")),
    "tracking_confidence": float(os.getenv("TRACKING_CONFIDENCE", "0.5")),
    "game_url": os.getenv("GAME_URL", "https://poki.com/en/g/subway-surfers"),
    "camera_index": int(os.getenv("CAMERA_INDEX", "0")),
}

# ZERO pause for maximum input speed
pyautogui.PAUSE = 0

class CameraStream:
    def __init__(self, src=0):
        # Use DirectShow on Windows for better external webcam compatibility
        if os.name == 'nt':
            self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            self.stream = cv2.VideoCapture(src)
            
        if not self.stream.isOpened():
            print(f"ERROR: Could not open camera {src}. It might be used by another app or disconnected.")
            self.stopped = True
            self.grabbed = False
            return

        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 60)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Force zero latency
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        if not self.stopped:
            threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True
        if hasattr(self, 'stream') and self.stream is not None:
            self.stream.release()

def main():
    config = CONFIG
    mode = config["control_mode"]

    if mode not in DETECTORS:
        print(f"Unknown mode '{mode}'. Available: {list(DETECTORS.keys())}")
        sys.exit(1)

    detector = DETECTORS[mode](config)
    print(f"=== Subway Surfers CV Controller ===")
    print(f"Mode: {detector.MODE_NAME} — {detector.MODE_DESC}")
    print(f"Opening game in your browser...")

    webbrowser.open(config["game_url"])

    # Start threaded camera
    cam_index = config["camera_index"]
    print(f"Starting camera (index {cam_index})...")
    cap = CameraStream(src=cam_index).start()
    
    if cap.stopped:
        print(f"\n[!] Gagal ngebuka kamera index {cam_index} wok.")
        print("[!] Coba cek lagi kameranya udah colok bener, atau dicoba ganti indexnya di .env jadi 0, 1, atau 2.")
        sys.exit(1)
        
    time.sleep(1.0) # wait for camera to warm up

    cooldown = config["cooldown_time"]
    last_action_time = 0
    last_action_name = ""
    last_action_display = 0

    win_name = "Subway Surfers Controller"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 480, 360)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)

    print("Camera ready! Click on the game in your browser, then play.")
    print("--------------------------------------------------")
    print("SHORTCUTS (Press while camera window is active):")
    print(" 'C' : Calibrate/Center the box on your body")
    print(" 'R' : Reset calibration to default")
    print(" '[' : Make the box SMALLER (more sensitive)")
    print(" ']' : Make the box BIGGER (less sensitive)")
    print(" 'Q' : Quit")
    print("--------------------------------------------------")

    while not cap.stopped:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        now = time.time()
        ready = now - last_action_time > cooldown

        # --- Always draw hitbox ---
        if hasattr(detector, '_draw_hitbox'):
            detector._draw_hitbox(frame)

        # --- Detect gesture ---
        action = detector.process(frame, rgb) if ready else None

        if action:
            key_map = {"JUMP": "up", "SLIDE": "down", "LEFT": "left", "RIGHT": "right"}
            pyautogui.press(key_map[action])
            last_action_time = now
            last_action_name = action
            last_action_display = now

        # --- UI Overlay ---
        if now - last_action_display < 0.5 and last_action_name:
            colors = {
                "JUMP": (0, 255, 0), "SLIDE": (0, 0, 255),
                "LEFT": (255, 255, 0), "RIGHT": (255, 0, 255),
            }
            c = colors.get(last_action_name, (255, 255, 255))
            cv2.putText(frame, last_action_name, (w // 2 - 80, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, c, 3)

        bar_color = (0, 180, 0) if ready else (0, 0, 180)
        cv2.rectangle(frame, (0, h - 40), (w, h), bar_color, -1)
        status = f"READY | {detector.MODE_DESC}" if ready else f"COOLDOWN | Last: {last_action_name}"
        cv2.putText(frame, status, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.rectangle(frame, (0, 0), (w, 35), (30, 30, 30), -1)
        cv2.putText(frame, f"Mode: {detector.MODE_NAME} | C:Calibrate R:Reset", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (233, 69, 96), 2)

        cv2.imshow(win_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            pyautogui.press('space')
        elif key == ord('c') or key == ord('C'):
            if hasattr(detector, 'get_last_position'):
                pos = detector.get_last_position()
                if pos:
                    if len(pos) > 2:
                        detector.calibrate(pos[0], pos[1], pos[2])
                    else:
                        detector.calibrate(pos[0], pos[1])
                    print(f"Calibrated center at {pos}")
        elif key == ord('r') or key == ord('R'):
            if hasattr(detector, 'reset_calibration'):
                detector.reset_calibration()
                print("Calibration reset to default")
        elif key == ord('['):  # Smaller box
            if hasattr(detector, 'adjust_box_size'):
                detector.adjust_box_size(-0.02, -0.02)
                print("Made box SMALLER")
        elif key == ord(']'):  # Larger box
            if hasattr(detector, 'adjust_box_size'):
                detector.adjust_box_size(0.02, 0.02)
                print("Made box BIGGER")

    cap.stop()
    cv2.destroyAllWindows()
    detector.release()

if __name__ == "__main__":
    main()
