# YOLOv8 Real-Time Object Detection (80 COCO classes — including phone, laptop, etc.)
# Run: python3 yolo_detection.py
# Press 'q' to quit

from ultralytics import YOLO
import cv2
import time

# Load YOLOv8 nano model (auto-downloads on first run ~6MB)
print("[INFO] Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")

# Start webcam
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)

fps_start = time.time()
frame_count = 0

print("[INFO] Detection running! Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    # Run YOLOv8 inference
    results = model(frame, verbose=False)[0]

    # Draw boxes on frame
    annotated_frame = results.plot()

    # Show FPS
    frame_count += 1
    elapsed = time.time() - fps_start
    fps = frame_count / elapsed if elapsed > 0 else 0
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Print detected objects to terminal
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0]) * 100
        print(f"  Detected: {label}  ({conf:.1f}%)")

    # Show window
    cv2.imshow("YOLOv8 - Real Time Object Detection (press q to quit)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n[INFO] Done. Total frames: {frame_count}, Avg FPS: {fps:.2f}")
