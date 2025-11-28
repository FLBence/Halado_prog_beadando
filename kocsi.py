import tkinter as tk
from tkinter import Tk, filedialog, messagebox
from tkinter import ttk
import threading
from ultralytics import YOLO
import cv2
import os
import numpy as np
 
# Csak ezek a járművek: autó, busz, kamion
VEHICLE_CLASSES = ["car", "bus", "truck"]
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5  # követéshez
 
# YOLO modell betöltése
model = YOLO("yolov8s.pt")
 
def iou(box1, box2):
    # box: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou
 
def draw_box(frame, xyxy, conf, cls_name):
    x1, y1, x2, y2 = map(int, xyxy)
    label = f"{cls_name} {conf:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - h - 6), (x1 + w, y1), (0, 255, 0), -1)
    cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
 
def detect_vehicles_in_video(video_path, show=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Could not open video file."
 
    vehicle_counts = {cls: 0 for cls in VEHICLE_CLASSES}
    next_id = 0
    tracked_objects = {}  # id: {"cls": str, "centroid": (x, y), "counted": bool}
    max_distance = 50  # pixel, centroid matching threshold
 
    # Get frame size for line position
    ret, frame = cap.read()
    if not ret:
        return "Could not read video."
    frame_height, frame_width = frame.shape[:2]
    line_y = frame_height // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset to first frame
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)[0]
        boxes = results.boxes
        detections = []  # (cls_name, centroid, xyxy, conf)
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls.cpu().numpy()) if hasattr(box, "cls") else int(box.cls)
                conf = float(box.conf.cpu().numpy()) if hasattr(box, "conf") else float(box.conf)
                cls_name = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
                if cls_name in VEHICLE_CLASSES and conf >= CONF_THRESHOLD:
                    xyxy = box.xyxy.cpu().numpy().flatten() if hasattr(box, "xyxy") else box.xyxy
                    x1, y1, x2, y2 = xyxy
                    centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    detections.append((cls_name, centroid, xyxy, conf))
        # Simple centroid-based tracking
        updated_ids = set()
        for cls_name, centroid, xyxy, conf in detections:
            matched_id = None
            min_dist = max_distance
            for obj_id, obj in tracked_objects.items():
                if obj["cls"] == cls_name:
                    dist = np.linalg.norm(np.array(centroid) - np.array(obj["centroid"]))
                    if dist < min_dist:
                        min_dist = dist
                        matched_id = obj_id
            if matched_id is not None:
                tracked_objects[matched_id]["centroid"] = centroid
            else:
                tracked_objects[next_id] = {"cls": cls_name, "centroid": centroid, "counted": False}
                matched_id = next_id
                next_id += 1
            updated_ids.add(matched_id)
            # Draw box and ID
            draw_box(frame, xyxy, conf, f"{cls_name} #{matched_id}")
            # Count if crosses the line and not counted yet
            obj = tracked_objects[matched_id]
            if not obj["counted"]:
                prev_y = obj.get("prev_y", centroid[1])
                if (prev_y < line_y <= centroid[1]) or (prev_y > line_y >= centroid[1]):
                    vehicle_counts[cls_name] += 1
                    obj["counted"] = True
                obj["prev_y"] = centroid[1]
        # Remove lost objects
        lost_ids = set(tracked_objects.keys()) - updated_ids
        for lost_id in lost_ids:
            del tracked_objects[lost_id]
        # Draw counting line
        cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 0, 255), 2)
        # overlay számlálók
        y = 30
        for k, v in vehicle_counts.items():
            text = f"{k}: {v}"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y += 30
        if show:
            cv2.imshow("Vehicle detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()
    summary = "\nDetected vehicles (összesen a vonalon áthaladva):\n"
    for v, count in vehicle_counts.items():
        summary += f"  {v}: {count}\n"
    return summary
 
def run_detection_thread(video_path):
    result = detect_vehicles_in_video(video_path, show=True)
    result_label.config(text=result)
 
def upload_video():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")]
    )
    if file_path:
        result_label.config(text="Processing video... Please wait.")
 
        # Run detection in a background thread
        thread = threading.Thread(target=run_detection_thread, args=(file_path,))
        thread.start()
 
root = Tk()
root.title("Vehicle Counter")
root.geometry("500x500")
 
upload_button = ttk.Button(root, text="Upload Video", command=upload_video)
upload_button.pack(pady=10)
 
result_label = ttk.Label(root, text="")
result_label.pack(pady=20)
 
root.mainloop()
 