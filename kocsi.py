import tkinter as tk
from tkinter import Tk, filedialog
from tkinter import ttk
import threading
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageTk

# Jármű osztályok és YOLO küszöb
VEHICLE_CLASSES = ["car", "bus", "truck", "train"]
CONF_THRESHOLD = 0.3

# YOLO modell
model = YOLO("yolov8s.pt")


def draw_box(frame, xyxy, conf, cls_name):
    x1, y1, x2, y2 = map(int, xyxy)
    label = f"{cls_name} {conf:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - h - 6), (x1 + w, y1), (0, 255, 0), -1)
    cv2.putText(frame, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def detect_vehicles_in_video(video_path, orientation, canvas, image_on_canvas, result_label, canvas_width, canvas_height):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result_label.config(text="Could not open video.")
        return

    vehicle_counts = {cls: 0 for cls in VEHICLE_CLASSES}
    next_id = 0
    tracked_objects = {}

    ret, frame = cap.read()
    if not ret:
        result_label.config(text="Could not read first frame.")
        return

    original_height, original_width = frame.shape[:2]
    vertical_mode = (orientation == "vertical")

    movement_positions = []
    auto_lines_set = False
    frame_index = 0
    max_distance = 50

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def update_canvas(frame):
        # Átméretezés a canvas méretéhez, megtartva az arányokat
        h_ratio = canvas_height / frame.shape[0]
        w_ratio = canvas_width / frame.shape[1]
        scale = min(h_ratio, w_ratio)
        new_w = int(frame.shape[1] * scale)
        new_h = int(frame.shape[0] * scale)
        resized_frame = cv2.resize(frame, (new_w, new_h))
        tk_frame = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        tk_frame = ImageTk.PhotoImage(tk_frame)
        canvas.itemconfig(image_on_canvas, image=tk_frame)
        canvas.image = tk_frame  # referencia megtartása

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        results = model(frame)[0]
        boxes = results.boxes
        detections = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls.cpu().numpy())
                conf = float(box.conf.cpu().numpy())
                cls_name = model.names[cls_id]

                if cls_name in VEHICLE_CLASSES and conf >= CONF_THRESHOLD:
                    xyxy = box.xyxy.cpu().numpy().flatten()
                    x1, y1, x2, y2 = xyxy
                    centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    detections.append((cls_name, centroid, xyxy, conf))

                    if frame_index <= 10:
                        movement_positions.append(centroid[1] if not vertical_mode else centroid[0])

        # Automata vonalak beállítása az első 10 frame alapján
        if frame_index == 11 and len(movement_positions) > 0:
            hist, bins = np.histogram(movement_positions, bins=20)
            top_line = int(bins[np.argmax(hist)])
            second_line = int(bins[np.argsort(hist)[-4]])

            if vertical_mode:
                offset = 50  # nagyobb távolság a függőleges vonalakhoz
                if top_line < second_line:
                    top_line -= offset
                    second_line += offset
                else:
                    top_line += offset
                    second_line -= offset
            auto_lines_set = True
        elif not auto_lines_set:
            top_line = original_height // 4 if not vertical_mode else original_width // 4
            second_line = original_height * 3 // 4 if not vertical_mode else original_width * 3 // 4

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
                tracked_objects[next_id] = {"cls": cls_name,
                                            "centroid": centroid,
                                            "counted": False,
                                            "still_frames": 0}
                matched_id = next_id
                next_id += 1

            updated_ids.add(matched_id)
            draw_box(frame, xyxy, conf, f"{cls_name} #{matched_id}")

            obj = tracked_objects[matched_id]
            prev_centroid = obj.get("prev_centroid", centroid)
            movement = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
            if movement < 2:
                obj["still_frames"] += 1
            else:
                obj["still_frames"] = 0

            if obj["still_frames"] >= 30 and not obj["counted"]:
                vehicle_counts[cls_name] += 1
                obj["counted"] = True

            obj["prev_centroid"] = centroid

            # Vonal átlépés
            if not obj["counted"]:
                if vertical_mode:
                    prev_x = obj.get("prev_x", centroid[0])
                    new_x = centroid[0]
                    crossed1 = (prev_x < top_line <= new_x) or (prev_x > top_line >= new_x)
                    crossed2 = (prev_x < second_line <= new_x) or (prev_x > second_line >= new_x)
                    if crossed1 or crossed2:
                        vehicle_counts[cls_name] += 1
                        obj["counted"] = True
                    obj["prev_x"] = new_x
                else:
                    prev_y = obj.get("prev_y", centroid[1])
                    new_y = centroid[1]
                    crossed1 = (prev_y < top_line <= new_y) or (prev_y > top_line >= new_y)
                    crossed2 = (prev_y < second_line <= new_y) or (prev_y > second_line >= new_y)
                    if crossed1 or crossed2:
                        vehicle_counts[cls_name] += 1
                        obj["counted"] = True
                    obj["prev_y"] = new_y

        lost = set(tracked_objects.keys()) - updated_ids
        for lid in lost:
            del tracked_objects[lid]

        # Vonalak rajzolása
        color1 = (0, 0, 255)
        color2 = (0, 255, 255)
        if vertical_mode:
            cv2.line(frame, (top_line, 0), (top_line, original_height), color1, 2)
            cv2.line(frame, (second_line, 0), (second_line, original_height), color2, 2)
        else:
            cv2.line(frame, (0, top_line), (original_width, top_line), color1, 2)
            cv2.line(frame, (0, second_line), (original_width, second_line), color2, 2)

        # Számlálók
        y = 30
        for k, v in vehicle_counts.items():
            cv2.putText(frame, f"{k}: {v}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            y += 30

        update_canvas(frame)

    cap.release()
    summary = "Detected vehicles:\n"
    for cls, cnt in vehicle_counts.items():
        summary += f"{cls}: {cnt}\n"
    result_label.config(text=summary)


# ------------------- GUI -------------------
def run_detection(video_path):
    orientation = line_mode.get()
    threading.Thread(target=detect_vehicles_in_video,
                     args=(video_path, orientation, video_canvas, image_on_canvas, result_label, 1500, 750),
                     daemon=True).start()


def upload_video():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")]
    )
    if file_path:
        result_label.config(text="Processing, please wait...")
        run_detection(file_path)


root = Tk()
root.title("Vehicle Counter")
root.geometry("1600x900")
root.attributes("-fullscreen", True)


ttk.Label(root, text="Select line orientation:").pack(pady=10)
line_mode = tk.StringVar(value="horizontal")
ttk.Radiobutton(root, text="Horizontal", variable=line_mode, value="horizontal").pack()
ttk.Radiobutton(root, text="Vertical", variable=line_mode, value="vertical").pack()

upload_button = ttk.Button(root, text="Upload Video", command=upload_video)
upload_button.pack(pady=10)

video_canvas = tk.Canvas(root, width=1500, height=750)
video_canvas.pack(pady=10)
tk_frame = ImageTk.PhotoImage(Image.new("RGB", (1500, 750)))
image_on_canvas = video_canvas.create_image(0, 0, anchor="nw", image=tk_frame)

result_label = ttk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()