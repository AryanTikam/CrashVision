import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.keras import backend as K

# Custom F1 Score metric for Keras (redefine the function properly)
def f1_score_metric(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')  # Ensure y_true is float32
    y_pred = K.round(y_pred)  # Round predictions to nearest integer (0 or 1)
    
    true_positives = K.sum(K.cast(y_true * y_pred, 'float32'))
    false_positives = K.sum(K.cast((1 - y_true) * y_pred, 'float32'))
    false_negatives = K.sum(K.cast(y_true * (1 - y_pred), 'float32'))
    
    precision = true_positives / (true_positives + false_positives + K.epsilon())
    recall = true_positives / (true_positives + false_negatives + K.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1

# Function to load YOLO model
def load_yolo_model():
    # Load YOLO model using the paths for weights and config
    net = cv2.dnn.readNet("./archive(1)/yolo/yolov3.weights", 
                           "./archive(1)/yolo/yolov3.cfg")
    
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    
    # Adjust output_layers to correctly access the layer names
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
    
    # Load COCO class names
    with open('./archive(1)/yolo/coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    return net, output_layers, classes

# Load the pre-trained accident detection model
model = load_model('accident_detection_model.keras', custom_objects={'f1_score_metric': f1_score_metric})

# Load YOLO model and COCO names
yolo_net, output_layers, classes = load_yolo_model()

# Define colors for accident detection
ACCIDENT_COLOR = (0, 0, 255)  # Red for accident detected
NO_ACCIDENT_COLOR = (0, 255, 0)  # Green for no accident

# Define the class IDs for relevant vehicle types in YOLO (adjust as needed)
VEHICLE_CLASSES = ['car', 'motorbike', 'bicycle', 'bus', 'truck']  # Add more vehicle classes as needed
VEHICLE_CLASS_IDS = [classes.index(v) for v in VEHICLE_CLASSES]

# Real-time detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Perform vehicle detection using YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward(output_layers)

    # Initialize lists to store detected boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

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

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)

    # Draw bounding boxes and predict accidents
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]

            # Only check for accidents on vehicle classes
            if class_ids[i] in VEHICLE_CLASS_IDS:
                # Get the region for accident detection
                if x >= 0 and y >= 0 and x + w <= width and y + h <= height:  # Check bounds
                    roi = frame[y:y + h, x:x + w]
                    roi_resized = cv2.resize(roi, (64, 64)) / 255.0  # Resize and normalize the ROI
                    roi_input = np.expand_dims(roi_resized, axis=0)  # Add batch dimension

                    # Predict if an accident has occurred
                    accident_prediction = model.predict(roi_input)

                    # Check the output shape
                    if accident_prediction.shape[1] == 2:  # Binary classification (Accident vs No Accident)
                        accident_prob = accident_prediction[0][1]  # Probability of accident
                    elif accident_prediction.shape[1] == 1:  # Single output (e.g., using sigmoid activation)
                        accident_prob = accident_prediction[0][0]  # Probability of accident (0-1 range)
                    else:
                        print("Unexpected prediction shape!")
                        continue  # Skip this detection if the shape is unexpected

                    # Detect accident based on probability threshold
                    accident_detected = 1 if accident_prob > 0.9 else 0

                    # Set color based on accident detection
                    box_color = ACCIDENT_COLOR if accident_detected == 1 else NO_ACCIDENT_COLOR

                    # Draw the bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

                    # Label the detected vehicle with accident status
                    label = str(classes[class_ids[i]])
                    accident_status = "Accident" if accident_detected == 1 else "No Accident"
                    cv2.putText(frame, f"{label} - {accident_status}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Display the resulting frame
    cv2.imshow('Accident Detection', frame)

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
