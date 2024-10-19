import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Function to load images and associate with labels
def load_images_and_labels(base_path, score_dfs):
    X = []
    y_severity = []
    y_accident = []
    
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
                y_severity.append(severity - 1)  # Labels: 0 for '1', 1 for '2', 2 for '3'
                y_accident.append(0)  # Non-accident for severity images

    # Load accident images
    accident_folder = os.path.join(base_path, 'Accident')
    for img_filename in os.listdir(accident_folder):
        img_path = os.path.join(accident_folder, img_filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            img_resized = cv2.resize(img, (64, 64))
            X.append(img_resized)
            y_severity.append(1)  # Placeholder for accident severity (can be refined)
            y_accident.append(1)  # Label: 1 for Accident
    
    # Load non-accident images
    non_accident_folder = os.path.join(base_path, 'NonAccident')
    for img_filename in os.listdir(non_accident_folder):
        img_path = os.path.join(non_accident_folder, img_filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            img_resized = cv2.resize(img, (64, 64))
            X.append(img_resized)
            y_severity.append(0)  # Placeholder for non-accident severity (can be refined)
            y_accident.append(0)  # Label: 0 for NonAccident
    
    X = np.array(X) / 255.0  # Normalize the images
    y_severity = to_categorical(np.array(y_severity), num_classes=3)  # Convert severity labels to categorical
    y_accident = np.array(y_accident)  # Keep accident labels as binary
    return X, y_severity, y_accident

# Define the multi-output CNN model
def create_multi_output_cnn_model(input_shape):
    input_layer = Input(shape=input_shape)

    # Shared convolutional layers
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Define separate outputs for severity and accident detection
    severity_output = Dense(3, activation='softmax', name='severity')(x)  # Severity classification
    accident_output = Dense(1, activation='sigmoid', name='accident')(x)  # Accident detection

    # Create the model
    model = Model(inputs=input_layer, outputs=[severity_output, accident_output])
    
    # Compile the model with appropriate losses and metrics
    model.compile(optimizer='adam', 
                  loss={'severity': 'categorical_crossentropy', 'accident': 'binary_crossentropy'}, 
                  metrics={'severity': 'accuracy', 'accident': 'accuracy'})
                  
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

X, y_severity, y_accident = load_images_and_labels(base_path, score_dfs)

# Split data into training and testing sets
X_train, X_test, y_train_severity, y_test_severity, y_train_accident, y_test_accident = train_test_split(
    X, y_severity, y_accident, test_size=0.2, random_state=42
)

# Create the multi-output model
model = create_multi_output_cnn_model(input_shape=X_train.shape[1:])

# Train the model on both severity and accident detection
model.fit(X_train, [y_train_severity, y_train_accident], epochs=10, batch_size=32, validation_data=(X_test, [y_test_severity, y_test_accident]))

# Save the model for later use in real-time detection
model.save('car_crash_detection_model.h5')
