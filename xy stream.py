import cv2
import mediapipe as mp
import time
from ultralytics import YOLO
from mediapipe.framework.formats import landmark_pb2
import depth_pro
import io
from PIL import Image
import numpy as np
import tempfile
import torch
import subprocess as sp
import sys
import signal
import time

class RTMPStreamer:
    def __init__(self, input_url, output_url, retry_interval=5, max_retries=3):
        self.input_url = input_url
        self.output_url = output_url
        self.retry_interval = retry_interval
        self.max_retries = max_retries
        self.ffmpeg_process = None
        self.cap = None
        
        # Initialize CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.init_models()

    def init_models(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize YOLO
        self.model = YOLO("yolov8n.pt")
        self.model.to(self.device)

        # Initialize depth model
        self.depth_model, self.transform = depth_pro.create_model_and_transforms()
        self.depth_model = self.depth_model.to(self.device)
        self.depth_model.eval()

    def setup_video_capture(self):
        retry_count = 0
        while retry_count < self.max_retries:
            self.cap = cv2.VideoCapture(self.input_url)
            if self.cap.isOpened():
                print(f"Successfully connected to input stream: {self.input_url}")
                return True
            print(f"Failed to connect to input stream. Retry {retry_count + 1}/{self.max_retries}")
            time.sleep(self.retry_interval)
            retry_count += 1
        return False

    def setup_ffmpeg(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # FFmpeg command to create RTSP output stream
        command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',  # OpenCV uses BGR format
            '-s', f'{width}x{height}',
            '-r', str(fps),
            '-i', '-',  # Input comes from pipe
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',  # Optimize for low-latency streaming
            '-g', '30',  # GOP size - keyframe interval
            '-b:v', '2M',  # Video bitrate
            '-bufsize', '2M',  # Buffer size
            '-maxrate', '2M',  # Maximum bitrate
            '-rtsp_transport', 'tcp',  # Use TCP for more reliable streaming
            '-f', 'rtsp',
            output_rtmp_url  # Should be renamed to output_rtsp_url and use rtsp:// protocol
        ]
        try:
            self.ffmpeg_process = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
            print(f"FFmpeg process started successfully for output stream: {self.output_url}")
            return True
        except sp.SubprocessError as e:
            print(f"Failed to start FFmpeg process: {e}")
            return False

    def process_frame(self, frame):
        height, width = frame.shape[:2]
        cx, cy = width // 2, height // 2

        # YOLO detection
        results = self.model(frame)
        boxes = []
        classes = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else []

        # Depth estimation
        is_success, buffer = cv2.imencode(".jpg", frame)
        if not is_success:
            return frame

        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp_file:
            tmp_file.write(buffer.tobytes())
            tmp_file.flush()
            image, _, f_px = depth_pro.load_rgb(tmp_file.name)
        
        depth_input = self.transform(image).to(self.device)
        with torch.no_grad():
            prediction = self.depth_model.infer(depth_input, f_px=f_px)
        depth = prediction["depth"]
        depth_np = depth.squeeze().cpu().numpy()

        # Process detections
        person_count = 0
        for i, box in enumerate(boxes):
            if int(classes[i]) != 0:  # Skip non-person detections
                continue

            person_count += 1
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Person {person_count}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Process pose estimation
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(roi_rgb)

            if pose_results.pose_landmarks:
                # Draw pose landmarks and process depth information
                self.process_pose_landmarks(frame, pose_results, x1, y1, x2, y2, 
                                         width, height, depth_np, cx, cy, person_count)

        return frame

    def process_pose_landmarks(self, frame, pose_results, x1, y1, x2, y2, 
                             width, height, depth_np, cx, cy, person_count):
        # Draw landmarks
        adjusted_landmarks = landmark_pb2.NormalizedLandmarkList()
        
        for landmark in pose_results.pose_landmarks.landmark:
            full_pixel_x = landmark.x * (x2 - x1) + x1
            full_pixel_y = landmark.y * (y2 - y1) + y1
            
            # Draw individual landmarks
            lm_x = int(full_pixel_x)
            lm_y = int(full_pixel_y)
            cv2.circle(frame, (lm_x, lm_y), 3, (0, 255, 0), cv2.FILLED)
            
            # Create adjusted landmark
            new_landmark = landmark_pb2.NormalizedLandmark()
            new_landmark.x = full_pixel_x / width
            new_landmark.y = full_pixel_y / height
            new_landmark.z = landmark.z
            new_landmark.visibility = landmark.visibility
            adjusted_landmarks.landmark.append(new_landmark)

        # Draw pose connections
        self.mp_drawing.draw_landmarks(
            frame,
            adjusted_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        # Calculate and draw depth information
        mean_x = int((x1 + x2) / 2)
        mean_y = int((y1 + y2) / 2)
        if mean_y < depth_np.shape[0] and mean_x < depth_np.shape[1]:
            mean_z = depth_np[mean_y, mean_x]
        else:
            mean_z = 0.0
        mean_rel_x = mean_x - cx
        mean_rel_y = mean_y - cy

        cv2.circle(frame, (mean_x, mean_y), 3, (255, 0, 0), cv2.FILLED)
        print(f"Person {person_count}: Pixel ({mean_x}, {mean_y}, {mean_z}), "
              f"Relative ({mean_rel_x}, {mean_rel_y}, {mean_z})")

    def run(self):
        if not self.setup_video_capture():
            print("Failed to setup video capture")
            return

        if not self.setup_ffmpeg():
            print("Failed to setup FFmpeg")
            self.cleanup()
            return

        prev_time = time.time()
        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break

                # Process frame
                processed_frame = self.process_frame(frame)

                # Calculate and display FPS
                current_time = time.time()
                frame_count += 1
                if current_time - start_time >= 1.0:
                    fps = frame_count / (current_time - start_time)
                    cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    frame_count = 0
                    start_time = current_time

                # Write frame to FFmpeg process
                try:
                    self.ffmpeg_process.stdin.write(processed_frame.tobytes())
                except IOError as e:
                    print(f"Error writing to FFmpeg process: {e}")
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        print("Cleaning up resources...")
        if self.cap is not None:
            self.cap.release()
        
        if self.ffmpeg_process is not None:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.wait(timeout=5)
            except sp.TimeoutExpired:
                self.ffmpeg_process.kill()
            
        print("Cleanup complete")

def main():
    # Configure your RTMP URLs
    input_rtmp_url = "rtmp://localhost:1935/stream"
    output_rtmp_url = "rtmp://localhost:1935/live/output-stream"
    
    streamer = RTMPStreamer(input_rtmp_url, output_rtmp_url)
    streamer.run()

if __name__ == "__main__":
    main()
