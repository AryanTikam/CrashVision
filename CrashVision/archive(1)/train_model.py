import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore

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
history = model.fit(X_train, y_train_severity, epochs=100, batch_size=32, validation_data=(X_test, y_test_severity))

# Save the model in the recommended format
model.save('car_crash_detection_model_100.keras')
