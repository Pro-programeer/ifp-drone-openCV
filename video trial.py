import os
import cv2
from ultralytics import YOLO

# Path to your model weights file
model_path = r'C:\Users\Prasana\OneDrive\Desktop\v8\runs\detect\train2\weights\best.pt'

# Path to your video file
video_path = r'C:\Users\Prasana\OneDrive\Desktop\v8\video.mp4'

# Check if video file exists
if not os.path.isfile(video_path):
    print(f"Error: Video file not found at {video_path}")
    exit()

# Load the custom YOLO v8 model with task='detect'
model = YOLO(model_path, task='detect')

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    for result in results:
        for detection in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = detection
            label = f"{model.names[int(cls)]} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Put label text above the bounding box
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame to the output video file
    out.write(frame)

    # Display the frame with bounding boxes (optional)
    cv2.imshow('Frame', frame)
    # Press 'q' to exit (optional)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects and close all OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()
