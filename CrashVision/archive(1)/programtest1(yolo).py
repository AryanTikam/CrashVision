import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Function to load images and associate with labels
def load_images_and_labels(base_path, score_dfs):
    X = []
    y = []
    
    # Load severity images
    for severity in range(1, 4):
        folder_path = os.path.join(base_path, str(severity))
        df = score_dfs[severity - 1]
        
        for img_filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_filename)
            img = cv2.imread(img_path)
            
            if img is not None:
                img_resized = cv2.resize(img, (64, 64))  # Resize to 64x64 pixels
                X.append(img_resized)
                y.append(severity - 1)  # Labels: 0 for '1', 1 for '2', 2 for '3'
    
    # Load accident images
    accident_folder = os.path.join(base_path, 'Accident')
    for img_filename in os.listdir(accident_folder):
        img_path = os.path.join(accident_folder, img_filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            img_resized = cv2.resize(img, (64, 64))
            X.append(img_resized)
            y.append(1)  # Label: 1 for Accident
    
    # Load non-accident images
    non_accident_folder = os.path.join(base_path, 'NonAccident')
    for img_filename in os.listdir(non_accident_folder):
        img_path = os.path.join(non_accident_folder, img_filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            img_resized = cv2.resize(img, (64, 64))
            X.append(img_resized)
            y.append(0)  # Label: 0 for NonAccident
    
    X = np.array(X) / 255.0  # Normalize the images
    y = np.array(y)
    return X, y

# Define the CNN model
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))  # 3 output classes for severity 1, 2, 3
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load YOLO model
def load_yolo_model():
    # Load YOLO model using the paths for weights and config
    net = cv2.dnn.readNet("C:/Users/aryan/Desktop/CrashVision/archive(1)/yolo/yolov3.weights", 
                           "C:/Users/aryan/Desktop/CrashVision/archive(1)/yolo/yolov3.cfg")
    
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    
    # Adjust output_layers to correctly access the layer names
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
    
    return net, output_layers

# Paths to the uploaded Excel files
file1 = './archive(1)/SeverityScore/Severity Score Dataset with Labels/score1.xlsx'
file2 = './archive(1)/SeverityScore/Severity Score Dataset with Labels/score2.xlsx'
file3 = './archive(1)/SeverityScore/Severity Score Dataset with Labels/score3.xlsx'

# Reading the Excel files
score1 = pd.read_excel(file1)
score2 = pd.read_excel(file2)
score3 = pd.read_excel(file3)

# Load the images and labels
base_path = "./archive(1)/SeverityScore/Severity Score Dataset with Labels"
score_dfs = [score1, score2, score3]  # The three severity score dataframes

X, y = load_images_and_labels(base_path, score_dfs)

# Convert severity labels to categorical for training
y_severity = to_categorical(y, num_classes=3)  # 3 severity classes
y_accident = (y == 1).astype(int)  # 1 if Accident, 0 if NonAccident

# Split data into training and testing sets
X_train, X_test, y_train_severity, y_test_severity = train_test_split(X, y_severity, test_size=0.2, random_state=42)
X_train_accident, X_test_accident, y_train_accident, y_test_accident = train_test_split(X, y_accident, test_size=0.2, random_state=42)

# Create the model
model = create_cnn_model(input_shape=X_train.shape[1:])

# Train the model on severity
history = model.fit(X_train, y_train_severity, epochs=10, batch_size=32, validation_data=(X_test, y_test_severity))

# Save the model in the recommended format
model.save('car_crash_detection_model.keras')

# Load YOLO model and COCO names
yolo_net, output_layers = load_yolo_model()

# Define colors for severity levels
severity_colors = {
    1: (0, 255, 0),    # Green for severity 1 (low)
    2: (0, 255, 255),  # Yellow for severity 2 (medium)
    3: (0, 0, 255)     # Red for severity 3 (high)
}

# Real-time detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Perform vehicle detection
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

    # Draw bounding boxes and predict severity
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]

            # Get the region for crash detection
            if x >= 0 and y >= 0 and x + w <= width and y + h <= height:  # Check bounds
                roi = frame[y:y + h, x:x + w]
                roi_resized = cv2.resize(roi, (64, 64)) / 255.0
                roi_input = np.expand_dims(roi_resized, axis=0)

                # Predict crash severity
                severity_prediction = model.predict(roi_input)
                severity = np.argmax(severity_prediction) + 1  # Assuming 1, 2, 3 for severity

                # Set the color based on the severity
                box_color = severity_colors.get(severity, (255, 255, 255))  # Default to white if severity is unknown

                # Draw the bounding box with the severity color
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

                # Display severity on the frame
                cv2.putText(frame, f"Severity: {severity}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Display the resulting frame
    cv2.imshow('Crash Detection', frame)

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
