import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import random
import matplotlib.pyplot as plt

# Function to load images and labels
def load_images_and_labels(base_path, include_accident=True, include_non_accident=True):
    X = []
    y = []
    random.seed(42)

    # Load severity images
    for severity in range(1, 4):
        folder_path = os.path.join(base_path, str(severity))
        for img_filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_filename)
            img = cv2.imread(img_path)
            print("Read ", img_path)
            if img is not None:
                img_resized = cv2.resize(img, (128, 128))
                X.append(img_resized)
                y.append(severity)

    # Load accident images
    if include_accident:
        non_accident_path = os.path.join(base_path, 'Accident')
        for img_filename in os.listdir(non_accident_path):
            img_path = os.path.join(non_accident_path, img_filename)
            img = cv2.imread(img_path)
            print("Read ", img_path)
            if img is not None:
                img_resized = cv2.resize(img, (128, 128))
                X.append(img_resized)
                y.append(random.randint(1, 3)) 

    # Load non-accident images
    if include_non_accident:
        non_accident_path = os.path.join(base_path, 'NonAccident')
        for img_filename in os.listdir(non_accident_path):
            img_path = os.path.join(non_accident_path, img_filename)
            img = cv2.imread(img_path)
            print("Read ", img_path)
            if img is not None:
                img_resized = cv2.resize(img, (128, 128))
                X.append(img_resized)
                y.append(0)  # Label 0 for non-accident

    X = np.array(X) / 255.0  # Normalize the images
    y = np.array(y)
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
model.save('car_crash_detection_model_improved_with_f1.keras')

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
plt.savefig('car_crash_detection_model_metrics.png')
