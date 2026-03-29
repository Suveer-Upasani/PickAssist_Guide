import os
import json
import time
import threading
import logging
from config import (
    CLASS_NAMES, IGNORE_CLASSES, STATE_FILE, 
    AUDIO_COOLDOWN, PINCH_COOLDOWN
)

logger = logging.getLogger(__name__)

class TrackedObject:
    def __init__(self, tid, label, cx, cy, bbox):
        self.tid = tid
        self.label = label
        self.cx = cx
        self.cy = cy
        self.bbox = bbox
        self.prev_y = None
        self.counted = False
        self.last_seen = time.time()
        self.time_lost = None
        self.first_appearance = time.time()
        self.hand_distance = float('inf')
        self.hand_pos = None

    def update(self, cx, cy, bbox):
        self.prev_y = self.cy
        self.cx = cx
        self.cy = cy
        self.bbox = bbox
        self.last_seen = time.time()

class TrackManager:
    def __init__(self, proximity_threshold=70, max_lost_time=1.5):
        self.active_tracks = {}
        self.lost_tracks = {}
        self.proximity_threshold = proximity_threshold
        self.max_lost_time = max_lost_time
        self.next_manual_id = 10000

    def _iou(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        a1, b1, a2, b2 = bbox2
        inter_x1 = max(x1, a1)
        inter_y1 = max(y1, b1)
        inter_x2 = min(x2, a2)
        inter_y2 = min(y2, b2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (a2 - a1) * (b2 - b1)
        union = area1 + area2 - inter_area
        return inter_area / union if union > 0 else 0

    def update_with_yolo(self, results, names, now):
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None: continue
            track_ids = boxes.id if boxes.id is not None else None
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                label = names[cls_id]
                if label in IGNORE_CLASSES: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                tid = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else self.next_manual_id
                if track_ids is None or i >= len(track_ids): self.next_manual_id += 1
                detections.append({'tid': tid, 'label': label, 'cx': cx, 'cy': cy, 'bbox': (x1, y1, x2, y2)})

        current_tids = {d['tid'] for d in detections}
        for tid in list(self.active_tracks.keys()):
            if tid not in current_tids:
                obj = self.active_tracks.pop(tid)
                obj.time_lost = now
                self.lost_tracks[tid] = obj

        self.lost_tracks = {tid: obj for tid, obj in self.lost_tracks.items() if now - obj.time_lost < self.max_lost_time}

        for d in detections:
            tid, label, cx, cy, bbox = d['tid'], d['label'], d['cx'], d['cy'], d['bbox']
            if tid in self.active_tracks:
                self.active_tracks[tid].update(cx, cy, bbox)
            else:
                matched_old_tid = None
                best_iou = 0.3
                for l_tid, l_obj in self.lost_tracks.items():
                    if l_obj.label == label and self._iou(bbox, l_obj.bbox) > best_iou:
                        best_iou = self._iou(bbox, l_obj.bbox)
                        matched_old_tid = l_tid
                if matched_old_tid:
                    old_obj = self.lost_tracks.pop(matched_old_tid)
                    old_obj.tid = tid
                    old_obj.update(cx, cy, bbox)
                    self.active_tracks[tid] = old_obj
                else:
                    self.active_tracks[tid] = TrackedObject(tid, label, cx, cy, bbox)
        return self.active_tracks

class SystemState:
    def __init__(self):
        self.lock = threading.Lock()
        self.next_serial = 1001
        self.next_batch_id = 1
        self.total_cumulative_count = 0
        self.class_cumulative_counts = {name: 0 for name in CLASS_NAMES if name not in IGNORE_CLASSES}
        self.tracker = TrackManager()
        self.counted_ids = set()
        self.last_spoken_time = {}
        self.current_batch = self._get_empty_batch()
        self.last_count_event = None
        self.last_diagnostic_time = time.time()
        self.last_obj_detected_time = time.time()
        self.last_hand_detected_time = time.time()
        self.pinch_timers = {}
        self.is_paused = False
        self.load_persistent_state()

    def _get_empty_batch(self):
        return {
            "active": False, "id": None, 
            "targets": {name: 0 for name in CLASS_NAMES if name not in IGNORE_CLASSES},
            "counts": {name: 0 for name in CLASS_NAMES if name not in IGNORE_CLASSES},
            "start_time": None, "serial_numbers": [], "auto_complete": False
        }

    def load_persistent_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                    self.next_serial = data.get("next_serial", 1001)
                    self.next_batch_id = data.get("next_batch_id", 1)
                    self.total_cumulative_count = data.get("total_cumulative_count", 0)
                    self.class_cumulative_counts = data.get("class_cumulative_counts", self.class_cumulative_counts)
            except Exception as e: logger.error(f"Error loading state: {e}")

    def save_persistent_state(self):
        try:
            with open(STATE_FILE, "w") as f:
                json.dump({
                    "next_serial": self.next_serial, "next_batch_id": self.next_batch_id,
                    "total_cumulative_count": self.total_cumulative_count,
                    "class_cumulative_counts": self.class_cumulative_counts
                }, f)
        except Exception as e: logger.error(f"Error saving state: {e}")

    def can_speak(self, event_key, cooldown=AUDIO_COOLDOWN):
        now = time.time()
        if event_key not in self.last_spoken_time or (now - self.last_spoken_time[event_key]) > cooldown:
            self.last_spoken_time[event_key] = now
            return True
        return False

    def start_batch(self, targets):
        with self.lock:
            self.current_batch = self._get_empty_batch()
            self.current_batch.update({
                "active": True, "id": f"B-{self.next_batch_id}",
                "targets": {k: int(v) for k, v in targets.items() if k in self.current_batch["targets"]},
                "start_time": time.time()
            })
            self.next_batch_id += 1
            self.save_persistent_state()
