import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from sklearn.cluster import KMeans

def load_yolo_model():
    net = cv2.dnn.readNet("./archive(1)/yolo/yolov3.weights", 
                           "./archive(1)/yolo/yolov3.cfg")
    
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
    
    with open('./archive(1)/yolo/coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    return net, output_layers, classes

def get_dominant_color(roi):
    """Identifies the dominant color in a region of interest (ROI) with improved color ranges."""
    if roi.size == 0:
        return "Unknown", (0, 0, 0)

    # Resize ROI to reduce computation if it's too large
    max_dimension = 100
    h, w = roi.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        roi = cv2.resize(roi, (int(w * scale), int(h * scale)))

    # Convert to RGB for better color analysis
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # Reshape the image into a 2D array of pixels
    pixels = roi_rgb.reshape(-1, 3)
    
    # Perform k-means clustering to find the dominant color
    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]
    
    # Convert to HSV for better color matching
    dominant_color_hsv = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_RGB2HSV)[0][0]
    
    # Define color ranges in HSV
    color_ranges = {
        'Red': [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [180, 255, 255])],  # Red wraps around
        'Blue': [([100, 50, 50], [130, 255, 255])],
        'Green': [([40, 50, 50], [80, 255, 255])],
        'White': [([0, 0, 200], [180, 30, 255])],
        'Black': [([0, 0, 0], [180, 30, 50])],
        'Silver': [([0, 0, 120], [180, 30, 200])],
        'Yellow': [([20, 100, 100], [40, 255, 255])],
        'Orange': [([10, 100, 100], [20, 255, 255])],
        'Brown': [([10, 50, 50], [20, 255, 150])],
        'Gray': [([0, 0, 50], [180, 30, 150])]
    }

    h, s, v = dominant_color_hsv

    # Function to check if a color is within a range
    def in_range(hsv, color_range):
        for range_pair in color_range:
            lower, upper = range_pair
            if all(lower[i] <= hsv[i] <= upper[i] for i in range(3)):
                return True
        return False

    # Find the matching color
    for color_name, ranges in color_ranges.items():
        if in_range((h, s, v), ranges):
            return color_name, dominant_color

    return "Unknown", dominant_color

# Load the pre-trained model
model = load_model('car_crash_detection_model_100.keras')

# Load YOLO model and COCO names
yolo_net, output_layers, classes = load_yolo_model()

# Define colors for severity levels
severity_colors = {
    1: (0, 255, 0),    # Green for severity 1 (low)
    2: (0, 255, 255),  # Yellow for severity 2 (medium)
    3: (0, 0, 255)     # Red for severity 3 (high)
}

# Define the class IDs for vehicles in YOLO
IGNORE_IDS = [1, 2, 3, 4, 5, 6, 7, 8]

# Real-time detection
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Perform vehicle detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)

    # Draw bounding boxes and predict severity
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]

            if class_ids[i] not in IGNORE_IDS:
                continue

            if x >= 0 and y >= 0 and x + w <= width and y + h <= height:
                # Extract ROI and ensure it's not empty
                roi = frame[max(0, y):min(height, y + h), max(0, x):min(width, x + w)]
                if roi.size == 0:
                    continue

                roi_resized = cv2.resize(roi, (64, 64)) / 255.0
                roi_input = np.expand_dims(roi_resized, axis=0)

                severity_prediction = model.predict(roi_input)
                severity = np.argmax(severity_prediction) + 1

                color, _ = get_dominant_color(roi)

                box_color = severity_colors.get(severity, (255, 255, 255))

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                
                # Create label with color and vehicle type
                label = f"{color} {classes[class_ids[i]]} - Severity: {severity}"
                
                # Calculate label position and size
                font_scale = 0.5
                font = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 2
                (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw label background
                cv2.rectangle(frame, 
                            (x, y - label_h - 10), 
                            (x + label_w, y), 
                            box_color, 
                            -1)
                
                # Draw label text
                cv2.putText(frame, 
                           label,
                           (x, y - 5), 
                           font, 
                           font_scale, 
                           (255, 255, 255), 
                           thickness)

    cv2.imshow('Crash Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()