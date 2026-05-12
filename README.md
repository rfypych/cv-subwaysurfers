# Subway Surfers Computer Vision Controller

Control Subway Surfers using your body movements!

## Requirements
- Python 3.10+
- Webcam

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Play
1. Open [Subway Surfers on Poki](https://poki.com/en/g/subway-surfers) in your browser.
2. Run the script:
   ```bash
   python main.py
   ```
3. Position yourself in the center of the camera feed.
4. **Movements**:
   - **Jump**: Move your head above the top green line.
   - **Slide**: Move your head below the bottom blue line.
   - **Left**: Lean or move to the left side of the screen.
   - **Right**: Lean or move to the right side of the screen.

## Tips
- Ensure you have good lighting.
- Stand about 2 meters away from the camera so your upper body is clearly visible.
- The script uses your **nose** position as the main anchor for movement.
