import os

# --- General Configuration ---
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.60))
CLASS_NAMES = ["Bearing", "Bolt", "Gear", "Nut"]
IGNORE_CLASSES = ["Gear"]

# --- File Paths ---
HISTORY_FILE = "batch_history.json"
TRANSCRIPT_FILE = "transcript.csv"
SNAPSHOT_DIR = "snapshots"
STATE_FILE = "system_state.json"

# --- ROI and Counting Line (Ratios 0.0 to 1.0) ---
ROI_TOP = 0.1
ROI_BOTTOM = 0.9
ROI_LEFT = 0.1
ROI_RIGHT = 0.9
COUNT_LINE_Y_RATIO = 0.8  

# --- Optimized Inference Settings ---
YOLO_IMGSZ = 320 

# --- Proximity & Interaction Thresholds ---
PINCH_THRESHOLD = 0.06  # Normalized distance
PINCH_COOLDOWN = 0.2
AUDIO_COOLDOWN = 1.8

# Ensure directories exist
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
