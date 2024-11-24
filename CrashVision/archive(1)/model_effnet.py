import os
import cv2
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import random
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Enable GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Found {len(physical_devices)} GPU(s). Memory growth enabled.")
    except RuntimeError as e:
        print(f"Error enabling memory growth: {e}")
else:
    print("No GPU devices found. Running on CPU.")

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
                y.append(0)
                
                # Data augmentation
                for angle in [-15, 15]:
                    matrix = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
                    rotated = cv2.warpAffine(img_resized, matrix, (64, 64))
                    X.append(rotated)
                    y.append(0)

    # Load severity-labeled images
    severity_images = {1: [], 2: [], 3: []}
    # for severity in range(1, 4):
    #     folder_path = os.path.join(base_path, str(severity))
    #     if os.path.exists(folder_path):
    #         for img_filename in os.listdir(folder_path):
    #             img_path = os.path.join(folder_path, img_filename)
    #             severity_images[severity].append(img_path)
    #             print("Read ", img_path)    

    # Load unlabeled accident images
    if include_accident:
        accident_folder = os.path.join(base_path, 'Accident')
        if os.path.exists(accident_folder):
            accident_images = []
            for img_filename in os.listdir(accident_folder):
                img_path = os.path.join(accident_folder, img_filename)
                accident_images.append(img_path)
                print("Read ", img_path)
            
            random.shuffle(accident_images)
            images_per_severity = len(accident_images) // 3
            for i, img_path in enumerate(accident_images):
                severity = (i // images_per_severity) + 1
                if severity <= 3:
                    severity_images[severity].append(img_path)

    # Balance dataset
    non_accident_count = len([label for label in y if label == 0])
    target_per_severity = max(non_accident_count // 3, 
                            max(len(images) for images in severity_images.values()))

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

# Custom F1-score metric with explicit type casting
@tf.keras.utils.register_keras_serializable()
def f1_score(y_true, y_pred):
    # Ensure consistent types
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Round predictions to get binary values
    y_pred = tf.round(y_pred)
    
    # Calculate true positives, false positives, and false negatives
    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)
    
    # Calculate precision and recall
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    
    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return tf.reduce_mean(f1)

# Define the EfficientNet model
def create_efficientnet_model(input_shape):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Fine-tune some of the base layers
    for layer in base_model.layers[:100]:
        layer.trainable = False
    for layer in base_model.layers[100:]:
        layer.trainable = True
    
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(4, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', f1_score])
    return model

def train_model(base_path):
    # Load data
    X, y = load_images_and_labels(base_path)
    y_severity = to_categorical(y, num_classes=4)
    
    # Split data
    X_train, X_test, y_train_severity, y_test_severity = train_test_split(
        X, y_severity, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = create_efficientnet_model(input_shape=X_train.shape[1:])
    
    # Training callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            patience=5,
            mode='max',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_f1_score',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model.keras',
            monitor='val_f1_score',
            save_best_only=True,
            mode='max'
        )
    ]
    
    history = model.fit(
        X_train, y_train_severity,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test_severity),
        callbacks=callbacks
    )
    
    # Load the best model
    model = tf.keras.models.load_model('best_model.keras', custom_objects={'f1_score': f1_score})
    
    # Save model
    model.save('car_crash_detection_model_efficientnet.keras')
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(131)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])

    # Loss plot
    plt.subplot(132)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])

    # F1 Score plot
    plt.subplot(133)
    plt.plot(history.history['f1_score'])
    plt.plot(history.history['val_f1_score'])
    plt.title('Model F1-Score')
    plt.ylabel('F1-Score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])

    plt.tight_layout()
    plt.savefig('car_crash_detection_model_efficientnet.png')
    
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
    plt.savefig('car_crash_detection_model_confusion_matrix_efficientnet.png')

if __name__ == "__main__":
    base_path = "./archive(1)/SeverityScore/Severity Score Dataset with Labels"
    train_model(base_path)