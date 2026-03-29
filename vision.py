import cv2
import mediapipe as mp
import math
import logging
import threading
from ultralytics import YOLO
from config import MODEL_PATH, CONFIDENCE_THRESHOLD, PINCH_THRESHOLD

logger = logging.getLogger(__name__)

# --- Initialization ---
try:
    yolo_model = YOLO(MODEL_PATH)
    logger.info(f"YOLO model loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    exit(1)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands_lock = threading.Lock()

def detect_hands(frame):
    """Processes frame with MediaPipe Hands and returns hand positions and pinch status."""
    h, w, _ = frame.shape
    hand_positions = []
    pinch_detected = False
    
    with hands_lock:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_model.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                hx, hy = int(index_tip.x * w), int(index_tip.y * h)
                hand_positions.append((hx, hy))
                
                dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                if dist < PINCH_THRESHOLD:
                    pinch_detected = True
                    cv2.circle(frame, (hx, hy), 20, (0, 255, 255), 2)
    return hand_positions, pinch_detected

def run_yolo(frame):
    """Runs YOLO tracking on a frame."""
    return yolo_model.track(frame, conf=CONFIDENCE_THRESHOLD, persist=True, verbose=False, imgsz=320)
