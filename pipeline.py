import cv2
import time
import numpy as np
from ultralytics import YOLO

# Load YOLO model (assuming you have a YOLOv8 .pt file)
model_path = "HVC-Project/best.pt"
model = YOLO(model_path)

# Camera settings
cap = cv2.VideoCapture("HVC-Project/data.mp4")

# Set confidence threshold for detection
CONFIDENCE_THRESHOLD = 0.5

# Conveyor control function (simulated)
def halt_conveyor():
    print("ALERT: Foreign object detected! Halting conveyor...")

def resume_conveyor():
    print("No foreign objects detected. Conveyor running.")

# Function to preprocess the image
def preprocess_image(frame):
    resized_frame = cv2.resize(frame, (640, 640))
    return resized_frame

def detect_foreign_objects(frame):
    # Store the original frame dimensions
    original_height, original_width = frame.shape[:2]

    # Preprocess the image (resize to model input size)
    preprocessed_frame = preprocess_image(frame)
    results = model(preprocessed_frame)

    # YOLOv8 results output: Use `results[0].boxes` for detection data
    boxes = results[0].boxes
    detections = boxes.data.cpu().numpy()

    # Filter detections by confidence threshold
    foreign_objects = [det for det in detections if det[4] > CONFIDENCE_THRESHOLD]

    # Scale bounding boxes back to original frame size
    for det in foreign_objects:
        x1, y1, x2, y2, conf, cls = det
        x1 = int(x1 * original_width / 640)
        y1 = int(y1 * original_height / 640)
        x2 = int(x2 * original_width / 640)
        y2 = int(y2 * original_height / 640)
        det[0], det[1], det[2], det[3] = x1, y1, x2, y2

    if len(foreign_objects) > 0:
        halt_conveyor()
        save_detected_frame(frame, foreign_objects)
        return True
    else:
        resume_conveyor()
        return False

def save_detected_frame(frame, foreign_objects):
    for det in foreign_objects:
        print(f"Detection: {det}, Length: {len(det)}")  # Debugging print

        if len(det) >= 4:
            x1, y1, x2, y2 = map(int, det[:4])
        else:
            print("Warning: Detection data has fewer than 4 elements, skipping.")
            continue

        conf = det[4] if len(det) > 4 else 0.0
        cls = int(det[5]) if len(det) > 5 else "Unknown"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Add label with confidence
        label = f"Object {cls}: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the image
    cv2.imwrite("results/detected_frame.jpg", frame)
    print("Detected frame saved as 'results/detected_frame.jpg'.")

# Main loop for real-time detection
def main():
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image from camera.")
                break

            # Detect foreign objects in the current frame
            detected = detect_foreign_objects(frame)

            # Display the current frame (for visual monitoring)
            cv2.imshow("Conveyor Monitoring", frame)

            # Press 'q' to quit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Optional: Small delay to simulate real-time processing
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping real-time detection.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
