import cv2
import torch
import time
import numpy as np

# Load YOLO model (assuming you have a YOLOv8 .pt file)
model_path = "path/to/your/yolo_model.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # Adjust based on your YOLO version

# Camera settings
camera_index = 0  # Change to your camera index if needed
cap = cv2.VideoCapture(camera_index)

# Set confidence threshold for detection
CONFIDENCE_THRESHOLD = 0.8

# Conveyor control function (simulated)
def halt_conveyor():
    print("ALERT: Foreign object detected! Halting conveyor...")

def resume_conveyor():
    print("No foreign objects detected. Conveyor running.")

# Function to preprocess the image (resize to model input size)
def preprocess_image(frame):
    # Assuming YOLO model works with 640x640 input size; adjust if needed
    resized_frame = cv2.resize(frame, (640, 640))
    return resized_frame

# Function to run inference on the image and check for foreign objects
def detect_foreign_objects(frame):
    preprocessed_frame = preprocess_image(frame)
    # Perform detection
    results = model(preprocessed_frame)
    
    # Filter detections by confidence threshold
    detections = results.pandas().xyxy[0]  # Get pandas dataframe of results
    foreign_objects = detections[detections['confidence'] > CONFIDENCE_THRESHOLD]

    # Check if any foreign objects detected
    if len(foreign_objects) > 0:
        halt_conveyor()
        return True
    else:
        resume_conveyor()
        return False

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
            time.sleep(0.1)  # Adjust as needed for processing power

    except KeyboardInterrupt:
        print("Stopping real-time detection.")

    finally:
        # Release the camera and close windows
        cap.release()
        cv2.destroyAllWindows()

# Run the real-time detection
if __name__ == "__main__":
    main()
