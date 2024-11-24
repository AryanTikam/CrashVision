import os
import cv2
import numpy as np
import random
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras import backend as K # type: ignore
import matplotlib.pyplot as plt

# Function to load images and labels
def load_images_and_labels(base_path, include_accident=True, include_non_accident=True):
    X = []
    y = []
    random.seed(42)

    # Load non-accident images
    if include_non_accident:
        non_accident_path = os.path.join(base_path, 'NonAccident')
        for img_filename in os.listdir(non_accident_path):
            img_path = os.path.join(non_accident_path, img_filename)
            img = cv2.imread(img_path)
            print("Read ", img_path)
            if img is not None:
                img_resized = cv2.resize(img, (64, 64))
                X.append(img_resized)
                y.append(0)  # Non-accident labeled as 0
                
                # Data augmentation for non-accident cases (only rotate, no severity mixing)
                for angle in [-15, 15]:
                    matrix = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
                    rotated = cv2.warpAffine(img_resized, matrix, (64, 64))
                    X.append(rotated)
                    y.append(0)  # Ensure augmented non-accident images are labeled as 0

    # Load severity-labeled images
    severity_images = {1: [], 2: [], 3: []}
    for severity in range(1, 4):
        folder_path = os.path.join(base_path, str(severity))
        if os.path.exists(folder_path):
            for img_filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_filename)
                severity_images[severity].append(img_path)
                print("Read ", img_path)

    # Load unlabeled accident images and assign random severity (without mixing non-accident data)
    if include_accident:
        accident_folder = os.path.join(base_path, 'Accident')
        if os.path.exists(accident_folder):
            accident_images = []
            for img_filename in os.listdir(accident_folder):
                img_path = os.path.join(accident_folder, img_filename)
                accident_images.append(img_path)
                print("Read ", img_path)
            
            # Distribute unlabeled images across severity levels
            random.shuffle(accident_images)
            images_per_severity = len(accident_images) // 3
            for i, img_path in enumerate(accident_images):
                severity = (i // images_per_severity) + 1
                if severity <= 3:  # Ensure we don't exceed severity 3
                    severity_images[severity].append(img_path)

    # Calculate target count per severity for balanced dataset
    non_accident_count = len([label for label in y if label == 0])
    target_per_severity = max(non_accident_count // 3, 
                              max(len(images) for images in severity_images.values()))

    # Load and balance severity images (ensure no non-accident mixing)
    for severity, image_paths in severity_images.items():
        sampled_paths = random.choices(image_paths, k=target_per_severity)
        
        for img_path in sampled_paths:
            img = cv2.imread(img_path)
            if img is not None:
                img_resized = cv2.resize(img, (64, 64))
                X.append(img_resized)
                y.append(severity)

    # Convert to numpy arrays and ensure correct dtypes
    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int32)
    
    print(f"Final class distribution: {np.bincount(y)}")
    return X, y

# Custom F1-score metric
def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)

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
    model.add(Dense(4, activation='softmax'))  # 4 output classes for severity 0, 1, 2, and 3
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_score])
    return model

# Load the images and labels
base_path = "./archive(1)/SeverityScore/Severity Score Dataset with Labels"
X, y = load_images_and_labels(base_path)

# Convert severity labels to categorical for training
y_severity = to_categorical(y, num_classes=4)  # 4 classes: 0 for non-accident, 1 for '1', 2 for '2', 3 for '3'

# Split data into training and testing sets
X_train, X_test, y_train_severity, y_test_severity = train_test_split(X, y_severity, test_size=0.2, random_state=42)

# Create the model
model = create_cnn_model(input_shape=X_train.shape[1:])

# Train the model on severity
history = model.fit(X_train, y_train_severity, epochs=50, batch_size=32, validation_data=(X_test, y_test_severity))

# Save the model in the recommended format
model.save('car_crash_detection_model_improved_with_f1v4.keras')

# Plot accuracy, loss, and F1-score vs. epoch
plt.figure(figsize=(10, 6))
plt.subplot(131)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(132)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(133)
plt.plot(history.history['f1_score'])
plt.plot(history.history['val_f1_score'])
plt.title('Model F1-Score')
plt.ylabel('F1-Score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.tight_layout()
plt.savefig('car_crash_detection_model_metricsv4.png')

# Create confusion matrix
y_pred = model.predict(X_test)
y_true = np.argmax(y_test_severity, axis=1)
y_pred = np.argmax(y_pred, axis=1)

cm = tf.math.confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm.numpy(),
                                display_labels=['Non-Accident', 'Severity 1', 'Severity 2', 'Severity 3'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Car Crash Detection Model')
plt.savefig('car_crash_detection_model_confusion_matrixv4.png')
