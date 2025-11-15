# detect_vehicles.py
import argparse
import cv2
from ultralytics import YOLO
import time

# Kiválasztott COCO osztályok (YOLO COCO névvel)
TARGET_CLASSES = {"car", "bus", "truck"}  # autó, busz, kamion

def draw_box(frame, xyxy, conf, cls_name):
    x1, y1, x2, y2 = map(int, xyxy)
    label = f"{cls_name} {conf:.2f}"
    # bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # label background
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - h - 6), (x1 + w, y1), (0, 255, 0), -1)
    cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def main(args):
    # Model betöltése (ha nincs letöltve, automatikusan letölti)
    model = YOLO(args.model)  # pl. "yolov8n.pt" (kicsi), vagy "yolov8m.pt" (közepes)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("Nem sikerült megnyitni a videót:", args.source)
        return

    # VideoWriter setup (output)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # YOLO inference (batches=1)
        # model.predict/frame is okay, de egyszerűbb: model(frame) -> results
        results = model(frame)[0]  # ultralytics v8: visszatér egy Results objektum listát; itt az első
        boxes = results.boxes

        # számlálók per frame
        counts = {"car": 0, "bus": 0, "truck": 0}

        if boxes is not None and len(boxes) > 0:
            # boxes.cls: class indices, boxes.conf conf, boxes.xyxy coordinates
            for i, box in enumerate(boxes):
                cls_id = int(box.cls.cpu().numpy()) if hasattr(box, "cls") else int(box.cls)
                conf = float(box.conf.cpu().numpy()) if hasattr(box, "conf") else float(box.conf)
                xyxy = box.xyxy.cpu().numpy().flatten() if hasattr(box, "xyxy") else box.xyxy

                cls_name = model.names[cls_id] if hasattr(model, "names") else str(cls_id)

                # Ha a detektált osztály érdekel minket (car, bus, truck)
                if cls_name in TARGET_CLASSES and conf >= args.conf:
                    counts[cls_name] += 1
                    draw_box(frame, xyxy, conf, cls_name)

        # overlay számlálók
        y = 30
        for k, v in counts.items():
            text = f"{k}: {v}"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y += 30

        # write és display (ha kéri)
        out.write(frame)
        if args.show:
            cv2.imshow("Vehicle detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = time.time() - start_time
    print(f"Feldolgozott képkockák: {frame_count}, idő: {elapsed:.2f}s, átlag FPS: {frame_count/elapsed:.2f}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Video fájl vagy kamera (alapértelmezett: 0 = kamera)")
    parser.add_argument("--output", type=str, default="output.mp4", help="Kimeneti videó fájl")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO modell fájl (pl. yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.3, help="Minimális confidence a dobozokhoz")
    parser.add_argument("--show", action="store_true", help="Mutassa a videót feldolgozás közben")
    args = parser.parse_args()
    # Ha kamera használata: source="0" (stringként). A cv2.VideoCapture kezeli.
    main(args)
