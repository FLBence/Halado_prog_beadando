import tkinter as tk
from tkinter import Tk, filedialog, messagebox
from tkinter import ttk
import threading
from ultralytics import YOLO
import cv2
import os

# Load YOLO model (choose your model: yolov8n.pt, yolov8s.pt, etc.)
model = YOLO("yolov8s.pt")

# Which classes are considered vehicles
VEHICLE_CLASSES = ["car", "truck", "bus", "motorbike", "bicycle"]


def detect_vehicles_in_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return "Could not open video file."

    vehicle_counts = {cls: 0 for cls in VEHICLE_CLASSES}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                if label in vehicle_counts:
                    vehicle_counts[label] += 1

    cap.release()

    # Create readable summary
    summary = "\nDetected vehicles:\n"
    for v, count in vehicle_counts.items():
        summary += f"  {v}: {count}\n"

    return summary


def run_detection_thread(video_path):
    result = detect_vehicles_in_video(video_path)
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