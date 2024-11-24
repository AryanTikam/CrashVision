import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from sklearn.cluster import KMeans
import time
import os

def load_yolo():
    net = cv2.dnn.readNet("./archive(1)/yolo/yolov3.weights", 
                         "./archive(1)/yolo/yolov3.cfg")
    
    # Fix for layer names handling
    layer_names = net.getLayerNames()
    output_layers = []
    for i in net.getUnconnectedOutLayers():
        # OpenCV returns indices that start at 1, so we need to subtract 1
        if isinstance(i, (list, np.ndarray)):
            output_layers.append(layer_names[i[0] - 1])
        else:
            output_layers.append(layer_names[i - 1])
    
    with open('./archive(1)/yolo/coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    return net, output_layers, classes

def get_dominant_color(roi):
    # Reshape the ROI for K-means clustering
    pixels = roi.reshape(-1, 3)
    
    # Apply K-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(pixels)
    
    # Get the dominant color
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    
    # Define color ranges and names
    color_ranges = {
        'Red': ([150, 0, 0], [255, 50, 50]),
        'Blue': ([0, 0, 150], [50, 50, 255]),
        'Green': ([0, 150, 0], [50, 255, 50]),
        'White': ([200, 200, 200], [255, 255, 255]),
        'Black': ([0, 0, 0], [50, 50, 50]),
        'Silver': ([160, 160, 160], [200, 200, 200]),
    }
    
    # Find the closest matching color
    color_name = 'Unknown'
    min_distance = float('inf')
    
    for name, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)
        if np.all(dominant_color >= lower) and np.all(dominant_color <= upper):
            center = (np.array(lower) + np.array(upper)) / 2
            distance = np.linalg.norm(dominant_color - center)
            if distance < min_distance:
                min_distance = distance
                color_name = name
    
    return color_name, dominant_color

def detect_accidents():
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Load the trained accident detection model
    try:
        model = load_model('car_crash_detection_model_efficientnet.keras', 
                          custom_objects={'f1_score': f1_score})
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load YOLO
    try:
        net, output_layers, classes = load_yolo()  # Fixed order of returned values
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO: {e}")
        return
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Vehicle classes from COCO dataset
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
    
    print("Starting accident detection... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        # Detection information
        class_ids = []
        confidences = []
        boxes = []
        
        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5 and classes[class_id] in vehicle_classes:
                    # Object detected
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    w = int(detection[2] * frame_width)
                    h = int(detection[3] * frame_height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Process each detected vehicle
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                
                # Ensure coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame_width - x)
                h = min(h, frame_height - y)
                
                # Extract vehicle ROI
                roi = frame[y:y+h, x:x+w]
                
                if roi.size == 0:
                    continue
                
                # Get vehicle type from YOLO detection
                vehicle_type = classes[class_ids[i]]
                
                # Prepare image for accident detection model
                try:
                    roi_resized = cv2.resize(roi, (64, 64))
                    roi_normalized = roi_resized / 255.0
                    roi_batch = np.expand_dims(roi_normalized, axis=0)
                    
                    # Predict accident severity
                    prediction = model.predict(roi_batch, verbose=0)
                    severity_score = np.argmax(prediction)
                    severity_prob = np.max(prediction) * 100
                    
                    # Get vehicle color
                    color_name, _ = get_dominant_color(roi)
                    
                    # Draw bounding box and information
                    if severity_score == 1:  # Accident detected
                        color = (0, 255, 0)  # Green
                    elif severity_score == 2:
                        color = (0, 255, 255)  # Yellow
                    elif severity_score == 3:
                        color = (0, 0, 255)  # Red  
                    else:
                        color = (0, 0, 0)
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Prepare text
                    severity_text = f"Safe" if severity_score == 0 else f"Severity: {severity_score}"
                    info_text = f"{color_name} {vehicle_type.capitalize()}"
                    confidence_text = f"Conf: {severity_prob:.1f}%"
                    
                    # Add text with background
                    cv2.rectangle(frame, (x, y-60), (x+200, y), color, -1)
                    cv2.putText(frame, severity_text, (x, y-45), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, info_text, (x, y-25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, confidence_text, (x, y-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                except Exception as e:
                    print(f"Error processing vehicle: {e}")
                    continue
        
        # Display the frame
        cv2.imshow('Accident Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Custom F1-score metric
@tf.keras.utils.register_keras_serializable()
def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return tf.reduce_mean(f1)

if __name__ == "__main__":
    detect_accidents()



