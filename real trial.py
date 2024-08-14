import torch
import cv2
from ultralytics import YOLO

# Load your custom YOLOv8 model
model_path = r'C:\Users\Prasana\OneDrive\Desktop\v8\runs\detect\train2\weights\best.pt'
model = YOLO(model_path)


# Function to detect objects in an image
def detect_objects(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to open image {image_path}")
        return

    # Convert image to RGB (YOLOv8 expects RGB images)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(image_rgb)

    # Check the structure of results
    if not results:
        print("No detections.")
        return

    # Iterate over the detections
    for result in results[0].boxes:  # Accessing the boxes attribute
        x1, y1, x2, y2 = result.xyxy[0]  # Bounding box coordinates
        conf = result.conf[0]  # Confidence score
        cls = result.cls[0]  # Class index

        # Get class name
        class_name = model.names[int(cls)]

        # Draw bounding box and label on image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f'{class_name} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

    # Save or display the result
    cv2.imshow('Detection', image)
    cv2.imwrite('output.jpg', image)  # Save the image with detections
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
image_path = r'C:\Users\Prasana\OneDrive\Desktop\test.jpeg'
detect_objects(image_path)
