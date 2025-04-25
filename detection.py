from ultralytics import YOLO
import cv2

# Load YOLOv8 nano model (fastest for CPU)
model = YOLO("yolov8n.pt")

# Start video capture from webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

prev_ids = set()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLOv8 tracking
    results = model.track(frame, persist=True, stream=True)

    for r in results:
        ids = set()
        if r.boxes.id is not None:
            ids = set(r.boxes.id.cpu().numpy().astype(int))

        new = ids - prev_ids
        missing = prev_ids - ids

        if new:
            print(f"🟢 New objects: {new}")
        if missing:
            print(f"🔴 Missing objects: {missing}")

        prev_ids = ids

        # Display live window with bounding boxes
        annotated_frame = r.plot()
        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
