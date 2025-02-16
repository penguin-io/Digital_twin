import cv2
import mediapipe as mp
import time
from ultralytics import YOLO
from mediapipe.framework.formats import landmark_pb2
import depth_pro
import io
from PIL import Image
import numpy as np

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

# Load the YOLOv8 model (downloads "yolov8n.pt" if not present)
model = YOLO("yolov8n.pt")

# Initialize depth estimation model and transforms (only once)
depth_model, transform = depth_pro.create_model_and_transforms()
depth_model.eval()

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
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] format
        classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else []
    else:
        boxes = []
        classes = []
    
    person_count = 0

    # Compute depth map for the full frame using depth_pro
    is_success, buffer = cv2.imencode(".jpg", frame)
    if not is_success:
        continue
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp_file:
        tmp_file.write(buffer.tobytes())
        tmp_file.flush()  # Ensure data is written
        image, _, f_px = depth_pro.load_rgb(tmp_file.name)
    depth_input = transform(image)
    prediction = depth_model.infer(depth_input, f_px=f_px)
    depth = prediction["depth"]
    depth_np = depth.squeeze().cpu().numpy()

    # Process each detected person in the frame
    for i, box in enumerate(boxes):
        # Process only the 'person' class (COCO class 0)
        if int(classes[i]) != 0:
            continue

        person_count += 1
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"Person {person_count}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Crop the ROI for pose estimation
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(roi_rgb)

        if pose_results.pose_landmarks:
            for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                lm_x = int(landmark.x * (x2 - x1)) + x1
                lm_y = int(landmark.y * (y2 - y1)) + y1
                cv2.circle(frame, (lm_x, lm_y), 3, (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, str(idx), (lm_x + 5, lm_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # Calculate the center of the bounding box and get depth value
            mean_x = int((x1 + x2) / 2)
            mean_y = int((y1 + y2) / 2)
            if mean_y < depth_np.shape[0] and mean_x < depth_np.shape[1]:
                mean_z = depth_np[mean_y, mean_x]
            else:
                mean_z = 0.0
            mean_rel_x = mean_x - cx
            mean_rel_y = mean_y - cy

            cv2.circle(frame, (mean_x, mean_y), 3, (255, 0, 0), cv2.FILLED)
            print(f"Person {person_count}: Pixel ({mean_x}, {mean_y}, {mean_z}), " +
                  f"Relative ({mean_rel_x}, {mean_rel_y}, {mean_z})")

            # Adjust landmarks to full-frame coordinates
            adjusted_landmarks = landmark_pb2.NormalizedLandmarkList()
            for landmark in pose_results.pose_landmarks.landmark:
                full_pixel_x = landmark.x * (x2 - x1) + x1
                full_pixel_y = landmark.y * (y2 - y1) + y1
                new_norm_x = full_pixel_x / width
                new_norm_y = full_pixel_y / height
                new_landmark = landmark_pb2.NormalizedLandmark()
                new_landmark.x = new_norm_x
                new_landmark.y = new_norm_y
                new_landmark.z = landmark.z
                new_landmark.visibility = landmark.visibility
                adjusted_landmarks.landmark.append(new_landmark)

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
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Multi-person Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
