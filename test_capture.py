import cv2
import numpy as np
import pygetwindow as gw
from mss import mss
import time

def find_game_window():
    windows = gw.getWindowsWithTitle('Subway Surfers')
    if windows:
        return windows[0]
    return None

def main():
    sct = mss()
    while True:
        window = find_game_window()
        if window:
            print(f"Found Window: {window.title} at {window.left}, {window.top}")
            # Capture a small part to see if it works
            monitor = {"top": window.top, "left": window.left, "width": window.width, "height": window.height}
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            cv2.imshow('Game Preview Test', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Searching for Subway Surfers window...")
            time.sleep(2)

if __name__ == "__main__":
    main()
