import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
            y.append(3)  # Label: 3 for Accident
    
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
    model.add(Dense(4, activation='softmax'))  # 4 output classes: 0 (non-accident), 1, 2, 3 (accident severity)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

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
y = to_categorical(y, num_classes=4)  # 4 classes: 0 (non-accident), 1, 2, 3 (severity levels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = create_cnn_model(input_shape=X_train.shape[1:])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the model using augmented data
model.fit(datagen.flow(X_train, y_train, batch_size=32),
          epochs=10, 
          validation_data=(X_test, y_test))

# Save the model for later use in real-time detection
model.save('car_crash_detection_model.h5')

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Print classification report and confusion matrix
print(classification_report(y_test_classes, y_pred_classes))
print(confusion_matrix(y_test_classes, y_pred_classes))

# Real-time detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_resized = cv2.resize(frame, (64, 64))
    frame_normalized = frame_resized / 255.0
    frame_input = np.expand_dims(frame_normalized, axis=0)

    # Predict crash severity and accident presence
    prediction = model.predict(frame_input)
    class_idx = np.argmax(prediction)

    # Determine bounding box color and severity
    if class_idx == 0:
        severity = 0  # Non-accident
        bounding_box = None
    else:
        severity = class_idx  # 1, 2, or 3 for severity
        bounding_box = (50, 50, 150, 150)  # Example bounding box; modify based on your detection logic

    # Set bounding box color based on severity
    if bounding_box:
        if severity == 1:
            box_color = (0, 255, 0)  # Green for severity 1
        elif severity == 2:
            box_color = (0, 255, 255)  # Yellow for severity 2
        elif severity == 3:
            box_color = (0, 0, 255)  # Red for severity 3
        start_point = (bounding_box[0], bounding_box[1])
        end_point = (bounding_box[2], bounding_box[3])
        cv2.rectangle(frame, start_point, end_point, box_color, 2)  # Draw bounding box

    cv2.putText(frame, f"Crash Severity: {severity}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Crash Detection', frame)
    
    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
