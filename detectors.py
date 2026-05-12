"""
Base detector and implementations for different control modes using Hitboxes.
"""
import cv2
import numpy as np

# --- MediaPipe Setup ---
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
except (AttributeError, ImportError):
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_draw


class BaseDetector:
    """Base class for all control mode detectors using Hitboxes."""

    def __init__(self, config):
        self.detection_confidence = config.get("detection_confidence", 0.5)
        self.tracking_confidence = config.get("tracking_confidence", 0.5)
        
        # Hitbox settings (default)
        self.center_x = 0.5
        self.center_y = 0.5
        self.box_width = 0.24  # +/- 0.12 from center
        self.box_height = 0.30 # +/- 0.15 from center
        
        self.update_thresholds()
        
    def update_thresholds(self):
        self.left_thresh = self.center_x - (self.box_width / 2)
        self.right_thresh = self.center_x + (self.box_width / 2)
        self.up_thresh = self.center_y - (self.box_height / 2)
        self.down_thresh = self.center_y + (self.box_height / 2)

    def calibrate(self, x, y):
        """Sets the new center point based on current tracking position."""
        self.center_x = x
        self.center_y = y
        self.update_thresholds()

    def adjust_box_size(self, width_delta, height_delta):
        """Adjusts the size of the hitbox."""
        self.box_width = max(0.1, self.box_width + width_delta)
        self.box_height = max(0.1, self.box_height + height_delta)
        self.update_thresholds()

    def process(self, frame, rgb):
        """
        Process a frame and return the gesture direction (or None).
        Returns: ("JUMP" | "SLIDE" | "LEFT" | "RIGHT" | None)
        """
        raise NotImplementedError

    def _draw_hitbox(self, frame):
        h, w = frame.shape[:2]
        # Draw threshold lines (thicker = 3)
        cv2.line(frame, (int(w * self.left_thresh), 0), (int(w * self.left_thresh), h), (255, 0, 0), 3)
        cv2.line(frame, (int(w * self.right_thresh), 0), (int(w * self.right_thresh), h), (255, 0, 0), 3)
        cv2.line(frame, (0, int(h * self.up_thresh)), (w, int(h * self.up_thresh)), (0, 255, 0), 3)
        cv2.line(frame, (0, int(h * self.down_thresh)), (w, int(h * self.down_thresh)), (0, 0, 255), 3)
        
        # Draw small center point
        cv2.circle(frame, (int(w * self.center_x), int(h * self.center_y)), 4, (0, 255, 255), -1)

    def get_zones(self, x, y):
        h_zone = "CENTER"
        if x < self.left_thresh:
            h_zone = "LEFT"
        elif x > self.right_thresh:
            h_zone = "RIGHT"

        v_zone = "CENTER"
        if y < self.up_thresh:
            v_zone = "JUMP"
        elif y > self.down_thresh:
            v_zone = "SLIDE"
            
        return h_zone, v_zone

    def process(self, frame, rgb):
        raise NotImplementedError

    def release(self):
        pass


class FingerDetector(BaseDetector):
    """Control using index finger tip tracking."""

    MODE_NAME = "FINGER"
    MODE_DESC = "Fixed Position: Finger maps directly to game lane."

    def __init__(self, config):
        super().__init__(config)
        self.hands = mp_hands.Hands(
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
            max_num_hands=1,
        )
        self.last_pos = None
        self.last_h_zone = "CENTER"
        self.last_v_zone = "CENTER"

    def process(self, frame, rgb):
        self._draw_hitbox(frame)
        results = self.hands.process(rgb)
        h, w = frame.shape[:2]

        if not results.multi_hand_landmarks:
            self.last_h_zone = "CENTER"
            self.last_v_zone = "CENTER"
            return None

        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            tip = hl.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            self.last_pos = (tip.x, tip.y)

            # Draw pointer
            ix, iy = int(tip.x * w), int(tip.y * h)
            cv2.circle(frame, (ix, iy), 10, (0, 255, 255), -1)
            cv2.circle(frame, (ix, iy), 15, (0, 255, 255), 2)
            
            h_zone, v_zone = self.get_zones(tip.x, tip.y)
            action = None
            
            if h_zone != self.last_h_zone:
                if h_zone == "LEFT":
                    action = "LEFT"
                elif h_zone == "RIGHT":
                    action = "RIGHT"
                elif h_zone == "CENTER":
                    action = "RIGHT" if self.last_h_zone == "LEFT" else "LEFT"
                self.last_h_zone = h_zone
                if action: return action

            if v_zone != self.last_v_zone:
                if v_zone == "JUMP":
                    action = "JUMP"
                elif v_zone == "SLIDE":
                    action = "SLIDE"
                self.last_v_zone = v_zone
                if action: return action

        return None

    def get_last_position(self):
        return self.last_pos

    def release(self):
        self.hands.close()


class BodyDetector(BaseDetector):
    """Control using body/pose tracking."""

    MODE_NAME = "BODY"
    MODE_DESC = "Fixed Position: Move body to map to game lane."

    def __init__(self, config):
        super().__init__(config)
        self.box_width = 0.30
        self.box_height = 0.40
        self.update_thresholds()
        
        self.pose = mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )
        self.last_pos = None
        self.last_h_zone = "CENTER"
        self.last_v_zone = "CENTER"

    def process(self, frame, rgb):
        self._draw_hitbox(frame)
        results = self.pose.process(rgb)
        h, w = frame.shape[:2]

        if not results.pose_landmarks:
            return None

        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        lm = results.pose_landmarks.landmark
        nose = lm[mp_pose.PoseLandmark.NOSE]
        
        mid_x = nose.x
        mid_y = nose.y
        self.last_pos = (mid_x, mid_y)

        px, py = int(mid_x * w), int(mid_y * h)
        cv2.circle(frame, (px, py), 12, (0, 255, 0), -1)
        cv2.circle(frame, (px, py), 18, (0, 255, 0), 2)
        
        h_zone, v_zone = self.get_zones(mid_x, mid_y)
        action = None
        
        # Horizontal matching
        if h_zone != self.last_h_zone:
            if h_zone == "LEFT":
                action = "LEFT"
            elif h_zone == "RIGHT":
                action = "RIGHT"
            elif h_zone == "CENTER":
                # Returning to center from the side -> reverse the move
                action = "RIGHT" if self.last_h_zone == "LEFT" else "LEFT"
            
            self.last_h_zone = h_zone
            if action: return action

        # Vertical matching
        if v_zone != self.last_v_zone:
            if v_zone == "JUMP":
                action = "JUMP"
            elif v_zone == "SLIDE":
                action = "SLIDE"
            # We don't send anything when returning to center vertically 
            # because in Subway Surfers, jumps and slides auto-recover.
            
            self.last_v_zone = v_zone
            if action: return action
            
        return None

    def get_last_position(self):
        return self.last_pos

    def release(self):
        self.pose.close()


# ---- Registry ----
DETECTORS = {
    "finger": FingerDetector,
    "body": BodyDetector,
}
