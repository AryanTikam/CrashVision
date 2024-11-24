import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # type: ignore
from tensorflow.keras import backend as K  # For defining custom metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Import for augmentation

# Function to load images and associate with labels
def load_images_and_labels(accident_folder, non_accident_folder):
    X = []
    y = []
    
    # Load accident images
    for img_filename in os.listdir(accident_folder):
        img_path = os.path.join(accident_folder, img_filename)
        img = cv2.imread(img_path)
        print("Read ", img_path)    
        
        if img is not None:
            img_resized = cv2.resize(img, (64, 64))
            X.append(img_resized)
            y.append(1)  # Label: 1 for Accident
    
    # Load non-accident images
    for img_filename in os.listdir(non_accident_folder):
        img_path = os.path.join(non_accident_folder, img_filename)
        img = cv2.imread(img_path)
        print("Read ", img_path)
        
        if img is not None:
            img_resized = cv2.resize(img, (64, 64))
            X.append(img_resized)
            y.append(0)  # Label: 0 for NonAccident
    
    X = np.array(X) / 255.0  # Normalize the images
    y = np.array(y)
    return X, y

# Custom F1 Score metric for Keras
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
    model.add(Dense(1, activation='sigmoid'))  # Binary output: Accident (1) or NonAccident (0)
    
    # Compile the model with the custom F1 score metric
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy', f1_score_metric])
    
    return model

# Paths to accident and non-accident images
accident_folder = "./archive(1)/SeverityScore/Severity Score Dataset with Labels/Accident"
non_accident_folder = "./archive(1)/SeverityScore/Severity Score Dataset with Labels/NonAccident"

# Load the images and labels
X, y = load_images_and_labels(accident_folder, non_accident_folder)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Augmentation - only for training data
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Apply the augmentation to the training data
train_datagen.fit(X_train)

# Create the model
model = create_cnn_model(input_shape=X_train.shape[1:])

# Add Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with data augmentation
history = model.fit(train_datagen.flow(X_train, y_train, batch_size=32), epochs=50, 
                    validation_data=(X_test, y_test), callbacks=[early_stopping])

model.save('accident_detection_model.keras')  

# 1. Plot Accuracy and F1 Score vs Epoch
plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(131)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# F1 Score plot
plt.subplot(132)
plt.plot(history.history['f1_score_metric'], label='Train F1 Score')
plt.plot(history.history['val_f1_score_metric'], label='Val F1 Score')
plt.title('Model F1 Score')
plt.ylabel('F1 Score')
plt.xlabel('Epoch')
plt.legend()

# Loss plot
plt.subplot(133)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics_vs_epoch.png')
plt.show()

# 2. Evaluate on the test set
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert to binary labels

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Accident', 'Accident'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Car Crash Detection Model')
plt.savefig('confusion_matrix.png')
plt.show()

# Optionally, print the F1 score on the test set
f1 = f1_score(y_test, y_pred)
print(f"Test F1 Score: {f1}")
