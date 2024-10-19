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

# Paths to the uploaded Excel files
file1 = './archive(1)/SeverityScore/Severity Score Dataset with Labels/score1.xlsx'
file2 = './archive(1)/SeverityScore/Severity Score Dataset with Labels/score2.xlsx'
file3 = './archive(1)/SeverityScore/Severity Score Dataset with Labels/score3.xlsx'

# Reading the Excel files
score1 = pd.read_excel(file1)
score2 = pd.read_excel(file2)
score3 = pd.read_excel(file3)

# Displaying the first few rows to inspect
score1_head = score1.head()
score2_head = score2.head()
score3_head = score3.head()

print(score1_head, score2_head, score3_head)

# Load the images and labels
base_path = "./archive(1)/SeverityScore/Severity Score Dataset with Labels"
score_dfs = [score1, score2, score3]  # The three severity score dataframes

X, y = load_images_and_labels(base_path, score_dfs)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, to_categorical(y, 3), test_size=0.2, random_state=42)

# Create the model
model = create_cnn_model(input_shape=X_train.shape[1:])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model for later use in real-time detection
model.save('car_crash_detection_model.h5')
