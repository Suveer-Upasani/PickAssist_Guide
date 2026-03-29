import os
import cv2
import numpy as np
import base64
import time
import csv
import logging
import math
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from config import *
from models import SystemState
from vision import detect_hands, run_yolo, yolo_model

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
state = SystemState()

# --- Helpers ---
def get_nearest_hand(hand_positions, obj_x, obj_y):
    if not hand_positions: return None
    min_dist = float('inf')
    nearest = None
    for hx, hy in hand_positions:
        dist = math.hypot(hx - obj_x, hy - obj_y)
        if dist < min_dist:
            min_dist = dist
            nearest = (hx, hy)
    return nearest

# --- Flask Routes ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_frame():
    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "No image data"}), 400

    try:
        img_data = base64.b64decode(data["image"].split(",")[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except:
        return jsonify({"error": "Invalid image data"}), 400

    h, w, _ = frame.shape
    r_x1, r_y1 = int(w * ROI_LEFT), int(h * ROI_TOP)
    r_x2, r_y2 = int(w * ROI_RIGHT), int(h * ROI_BOTTOM)
    line_y = r_y1 + int((r_y2 - r_y1) * COUNT_LINE_Y_RATIO)

    # 1. Hand Detection
    hand_positions, pinch_detected = detect_hands(frame)
    if hand_positions: state.last_hand_detected_time = time.time()

    # 2. Diagnostics & Pause Check
    audio_events = []
    now = time.time()
    
    if np.mean(frame) < 40 and state.can_speak("diag_low_light", 60):
        audio_events.append("diag_low_light")

    if state.is_paused:
        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return jsonify({
            "image": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}",
            "batch": state.current_batch if state.current_batch["active"] else None,
            "audio_events": ["system_paused"] if state.can_speak("paused_reminder", 30) else [],
            "auto_complete": False
        })

    # 3. Inference & Tracking
    results = run_yolo(frame)
    if now - state.last_hand_detected_time > 10 and state.can_speak("diag_no_hand", 30):
        audio_events.append("diag_no_hand")

    with state.lock:
        active_tracks = state.tracker.update_with_yolo(results, yolo_model.names, now)

        for obj in active_tracks.values():
            # ROI and Detection Logic
            if not (r_x1 <= obj.cx <= r_x2): continue
            state.last_obj_detected_time = now

            # Nearest hand for this object
            nearest_hand = get_nearest_hand(hand_positions, obj.cx, obj.cy)
            hand_dist = math.hypot(nearest_hand[0]-obj.cx, nearest_hand[1]-obj.cy) if nearest_hand else float('inf')

            should_count = False
            count_reason = None

            if not obj.counted and obj.tid not in state.counted_ids:
                if now - obj.first_appearance < 0.3:
                    should_count, count_reason = True, "detection"
                elif pinch_detected and hand_dist < 50:
                    if obj.tid not in state.pinch_timers or (now - state.pinch_timers[obj.tid]) > PINCH_COOLDOWN:
                        should_count, count_reason = True, "pinch"
                        state.pinch_timers[obj.tid] = now
                elif obj.prev_y is not None and obj.prev_y < line_y and obj.cy >= line_y:
                    should_count, count_reason = True, "line"

            if should_count:
                sn = f"SN-{state.next_serial}"
                state.next_serial += 1
                state.total_cumulative_count += 1
                state.class_cumulative_counts[obj.label] += 1
                obj.counted = True
                state.counted_ids.add(obj.tid)

                was_batch = False
                batch_id = state.current_batch["id"] if state.current_batch["active"] else "N/A"

                if state.current_batch["active"]:
                    target = state.current_batch["targets"].get(obj.label, 0)
                    if target > 0 and state.current_batch["counts"][obj.label] < target:
                        state.current_batch["counts"][obj.label] += 1
                        state.current_batch["serial_numbers"].append(sn)
                        was_batch = True
                        audio_events.append(f"count_confirm_{obj.label.lower()}_{sn}_{state.current_batch['counts'][obj.label]}_{target}")
                        if all(state.current_batch["counts"][n] >= state.current_batch["targets"][n] for n in state.current_batch["targets"]):
                            state.current_batch["auto_complete"] = True
                    else: audio_events.append("wrong_object")
                else: audio_events.append(f"count_confirm_{obj.label.lower()}_{sn}_0_0")

                # Log count
                try:
                    file_exists = os.path.isfile(TRANSCRIPT_FILE)
                    with open(TRANSCRIPT_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists: writer.writerow(["timestamp", "class", "serial_number", "batch_id"])
                        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), obj.label, sn, batch_id])
                except Exception as e: logger.error(f"CSV Error: {e}")

                state.last_count_event = (obj.label, sn, obj.tid, was_batch)
                state.save_persistent_state()

            # Drawing
            color = (0, 255, 0) if obj.counted else (255, 0, 0)
            cv2.rectangle(frame, (obj.bbox[0], obj.bbox[1]), (obj.bbox[2], obj.bbox[3]), color, 2)
            cv2.putText(frame, f"{obj.label} {obj.tid}", (obj.bbox[0], obj.bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ROI & Line Graphics
    cv2.rectangle(frame, (r_x1, r_y1), (r_x2, r_y2), (0, 255, 255), 1)
    cv2.line(frame, (r_x1, line_y), (r_x2, line_y), (0, 0, 255), 2)

    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return jsonify({
        "image": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}",
        "batch": state.current_batch if state.current_batch["active"] else None,
        "audio_events": audio_events,
        "auto_complete": state.current_batch.get("auto_complete", False)
    })

# --- Standard Flask Actions ---
@app.route("/start_batch", methods=["POST"])
def start_batch():
    state.start_batch(request.json.get("targets", {}))
    return jsonify({"status": "success"})

@app.route("/stop_batch", methods=["POST"])
def stop_batch():
    state.current_batch = state._get_empty_batch()
    return jsonify({"status": "success"})

@app.route("/undo", methods=["POST"])
def undo():
    with state.lock:
        if not state.last_count_event: return jsonify({"status": "error", "message": "Nothing to undo."})
        label, sn, tid, was_batch = state.last_count_event
        state.total_cumulative_count -= 1
        state.class_cumulative_counts[label] -= 1
        if tid in state.counted_ids: state.counted_ids.remove(tid)
        if was_batch and state.current_batch["active"]:
            state.current_batch["counts"][label] -= 1
            if sn in state.current_batch["serial_numbers"]: state.current_batch["serial_numbers"].remove(sn)
        state.last_count_event = None
        state.save_persistent_state()
        return jsonify({"status": "success", "message": f"Undone count for {label}."})

@app.route("/pause", methods=["POST"])
def pause(): state.is_paused = True; return jsonify({"status": "success"})

@app.route("/resume", methods=["POST"])
def resume(): state.is_paused = False; return jsonify({"status": "success"})

@app.route("/status", methods=["GET"])
def status():
    with state.lock:
        data = {"total": state.total_cumulative_count, "class_counts": state.class_cumulative_counts, "batch": state.current_batch, "paused": state.is_paused}
        return jsonify({"status": "ok", "data": data, "message": f"Total: {data['total']}. System {'Paused' if data['paused'] else 'Active'}."})

@app.route("/reset", methods=["POST"])
def reset():
    state.__init__()
    state.save_persistent_state()
    return jsonify({"status": "success"})

@app.route("/export_braille", methods=["GET"])
def export_braille():
    if not os.path.exists(TRANSCRIPT_FILE): return jsonify({"error": "No transcript"}), 404
    with open(TRANSCRIPT_FILE, 'r') as csvf, open("braille_transcript.txt", 'w') as bf:
        reader = csv.DictReader(csvf)
        bf.write("--- SESSION REPORT ---\n")
        for r in reader: bf.write(f"SN: {r['serial_number']} | ITEM: {r['class']} | TIME: {r['timestamp']}\n")
    return send_file("braille_transcript.txt", as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=False, threaded=True)
