import cv2
import mediapipe as mp
import time
from ultralytics import YOLO
from mediapipe.framework.formats import landmark_pb2

# Initialize MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load the YOLOv8 model (this will download "yolov8n.pt" automatically if not present)
model = YOLO("yolov8n.pt")

# Open video capture
cap = cv2.VideoCapture(0)
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2

    # Run YOLOv8 inference on the frame
    results = model(frame)
    
    # Extract detected bounding boxes and class labels
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] format
        classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else []
    else:
        boxes = []
        classes = []

    person_count = 0

    # Process each detected object
    for i, box in enumerate(boxes):
        # Process only the 'person' class (class 0 in COCO)
        if int(classes[i]) != 0:
            continue

        person_count += 1
        x1, y1, x2, y2 = map(int, box)
        # Clamp coordinates within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # Draw bounding box and label for the detected person
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame, f"Person {person_count}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
        )

        # Crop the detected person ROI for pose estimation
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(roi_rgb)

        # Process the pose landmarks if available
        if pose_results.pose_landmarks:
            # Draw circles and text using ROI-adjusted coordinates
            for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                lm_x = int(landmark.x * (x2 - x1)) + x1
                lm_y = int(landmark.y * (y2 - y1)) + y1
                rel_x = lm_x - cx
                rel_y = lm_y - cy
                print(f"Person {person_count} - Landmark {idx}: Pixel ({lm_x}, {lm_y}), Relative ({rel_x}, {rel_y})")
                cv2.circle(frame, (lm_x, lm_y), 3, (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, str(idx), (lm_x + 5, lm_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # Create a new NormalizedLandmarkList with adjusted coordinates for full frame
            adjusted_landmarks = landmark_pb2.NormalizedLandmarkList()
            for landmark in pose_results.pose_landmarks.landmark:
                # Convert ROI normalized landmark to full-frame pixel coordinates
                full_pixel_x = landmark.x * (x2 - x1) + x1
                full_pixel_y = landmark.y * (y2 - y1) + y1
                # Re-normalize relative to the full frame
                new_norm_x = full_pixel_x / width
                new_norm_y = full_pixel_y / height

                new_landmark = landmark_pb2.NormalizedLandmark()
                new_landmark.x = new_norm_x
                new_landmark.y = new_norm_y
                new_landmark.z = landmark.z  # Preserve z coordinate
                new_landmark.visibility = landmark.visibility
                adjusted_landmarks.landmark.append(new_landmark)

            # Draw the full skeleton on the original frame using adjusted landmarks
            mp_drawing.draw_landmarks(
                frame, 
                adjusted_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

    # Calculate and display FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time
    cv2.putText(
        frame, f"FPS: {int(fps)}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
    )

    cv2.imshow("Multi-person Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
