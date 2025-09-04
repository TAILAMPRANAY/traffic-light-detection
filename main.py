import cv2
import numpy as np
import argparse
from ultralytics import YOLO

# Parse command line arguments
parser = argparse.ArgumentParser(description='Traffic Light Detection System')
parser.add_argument('--video', '-v', type=str, default=None,
                    help='Path to video file (default: use webcam)')
parser.add_argument('--camera', '-c', type=int, default=0,
                    help='Camera index (default: 0)')
args = parser.parse_args()

model = YOLO('yolov8n.pt')

# Better color ranges from the working repo
red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([160, 100, 100])
red_upper2 = np.array([180, 255, 255])
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([35, 255, 255])
green_lower = np.array([40, 100, 100])
green_upper = np.array([85, 255, 255])


# Function to count colors in a region
def dominant_color(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    return {
        "Red": cv2.countNonZero(red_mask),
        "Yellow": cv2.countNonZero(yellow_mask),
        "Green": cv2.countNonZero(green_mask)
    }


# Zone-based traffic light color detection
def detect_color(crop):
    if crop.size == 0:
        return "Unknown"

    crop = cv2.resize(crop, (50, 150))  # Normalize size
    h = crop.shape[0]

    # Divide into 3 zones
    top = crop[0:h // 3, :]  # Red light zone
    middle = crop[h // 3:2 * h // 3, :]  # Yellow light zone
    bottom = crop[2 * h // 3:, :]  # Green light zone


    top_counts = dominant_color(top)
    mid_counts = dominant_color(middle)
    bot_counts = dominant_color(bottom)

    # Find the zone
    if top_counts["Red"] > 50:
        return "Red"
    elif mid_counts["Yellow"] > 50:
        return "Yellow"
    elif bot_counts["Green"] > 50:
        return "Green"
    else:
        return "Unknown"


# Zone based traffic light color detection
def detect_color(crop):
    if crop.size == 0:
        return "Unknown"

    crop = cv2.resize(crop, (50, 150))  # Normalize size
    h = crop.shape[0]

    # Divide into 3 zones
    top = crop[0:h // 3, :]  # Red light
    middle = crop[h // 3:2 * h // 3, :]  # Yellow light
    bottom = crop[2 * h // 3:, :]  # Green light

    # Count colors in each zone
    top_counts = dominant_color(top)
    mid_counts = dominant_color(middle)
    bot_counts = dominant_color(bottom)

    # Check which zone has significant color presence
    if top_counts["Red"] > 50:
        return "Red"
    elif mid_counts["Yellow"] > 50:
        return "Yellow"
    elif bot_counts["Green"] > 50:
        return "Green"
    else:
        return "Unknown"


# Setup video capture based on arguments
if args.video:
    print(f"Using video file: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{args.video}'")
        exit(1)
else:
    print(f"Using webcam (camera index: {args.camera})")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        print("Make sure your camera is connected and not being used by another application")
        exit(1)

print("Press 'q' to quit, 'space' to pause")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO
    results = model(frame)

    # Start with original frame
    display_frame = frame.copy()

    # Filter for traffic lights
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                if int(box.cls) == 9:  # Traffic light class
                    # Get confidence score
                    conf = float(box.conf[0])
                    if conf < 0.3:  # Skip low confidence detections
                        continue

                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Extract region of interest (the traffic light area)
                    roi = frame[y1:y2, x1:x2]

                    # Detect light state using zone-based method
                    state = detect_color(roi)

                    # Choose color for box based on state
                    if state == "Red":
                        color = (0, 0, 255)
                    elif state == "Yellow":
                        color = (0, 255, 255)
                    elif state == "Green":
                        color = (0, 255, 0)
                    else:
                        color = (128, 128, 128)

                    # Draw rectangle and label
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f'{state} ({conf:.2f})', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Traffic Light Detection', display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break
    elif key == ord(' '):  # Spacebar to pause
        print("Paused. Press any key to continue...")
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()