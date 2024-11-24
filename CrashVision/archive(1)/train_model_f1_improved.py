import os
import cv2
import numpy as np
import random
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import datetime

def f1_score(y_true, y_pred):
    """Custom F1 score metric"""
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)

def load_images_and_labels(base_path, include_accident=True, include_non_accident=True):
    X = []
    y = []
    random.seed(42)

    def augment_image(img):
        """Apply a series of augmentations to an image."""
        augmented_images = []
        
        # Horizontal Flip
        flipped = cv2.flip(img, 1)
        augmented_images.append(flipped)

        # Brightness Adjustment
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.add(hsv[:, :, 2], random.randint(-30, 30))
        brightness_adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        augmented_images.append(brightness_adjusted)

        # Random Cropping and Resizing
        crop_size = random.randint(48, 64)
        x_start = random.randint(0, img.shape[1] - crop_size)
        y_start = random.randint(0, img.shape[0] - crop_size)
        cropped = img[y_start:y_start + crop_size, x_start:x_start + crop_size]
        cropped_resized = cv2.resize(cropped, (64, 64))
        augmented_images.append(cropped_resized)

        # Add Gaussian Noise
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        noisy = cv2.add(img, noise)
        augmented_images.append(noisy)

        # Color Jitter
        alpha = random.uniform(0.8, 1.2)  # Contrast control
        beta = random.randint(-20, 20)    # Brightness control
        color_jittered = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        augmented_images.append(color_jittered)

        return augmented_images

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

                # Apply augmentations and append
                augmented_images = augment_image(img_resized)
                for augmented in augmented_images:
                    X.append(augmented)
                    y.append(0)  # Augmented non-accident labeled as 0

    # Load severity-labeled images
    severity_images = {1: [], 2: [], 3: []}
    # for severity in range(1, 4):
    #     folder_path = os.path.join(base_path, str(severity))
    #     if os.path.exists(folder_path):
    #         for img_filename in os.listdir(folder_path):
    #             img_path = os.path.join(folder_path, img_filename)
    #             severity_images[severity].append(img_path)
    #             print("Read ", img_path)

    # Load unlabeled accident images and assign random severity
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

    # Calculate target count per severity for balancing
    non_accident_count = len([label for label in y if label == 0])
    target_per_severity = max(non_accident_count // 3,        
                              max(len(images) for images in severity_images.values()))

    # Load and balance severity images
    for severity, image_paths in severity_images.items():
        sampled_paths = random.choices(image_paths, k=target_per_severity)
        
        for img_path in sampled_paths:
            img = cv2.imread(img_path)
            if img is not None:
                img_resized = cv2.resize(img, (64, 64))
                X.append(img_resized)
                y.append(severity)

    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int32)

    print(f"Final class distribution: {np.bincount(y)}")
    return X, y

def create_improved_model(input_shape):
    """Create the CNN model with improved architecture"""
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Dense Layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', f1_score]
    )
    
    return model

def train_model():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    
    # Load and prepare data
    print("Loading dataset...")
    base_path = "./archive(1)/SeverityScore/Severity Score Dataset with Labels"
    X, y = load_images_and_labels(base_path)
    
    # Calculate class weights for slight bias towards non-accidents
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    # Adjust weights to slightly favor non-accidents
    class_weights[0] *= 1.2  # Increase weight for non-accidents
    class_weight_dict = dict(enumerate(class_weights))
    
    # Convert to categorical
    print("Preparing data for training...")
    y_categorical = to_categorical(y, num_classes=4)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train model
    print("Creating model...")
    model = create_improved_model(input_shape=X_train.shape[1:])
    
    # Custom callback to handle interrupts
    class InterruptCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super(InterruptCallback, self).__init__()
            self.interrupted = False
            
        def on_train_batch_end(self, batch, logs=None):
            try:
                if self.interrupted:
                    self.model.stop_training = True
            except KeyboardInterrupt:
                self.interrupted = True
                print('\nTraining interrupted. Saving model...')
                # Save model with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.model.save(f'interrupted_model_{timestamp}.keras')
                print(f'Model saved as interrupted_model_{timestamp}.keras')
                self.model.stop_training = True
                
        def on_epoch_end(self, epoch, logs=None):
            try:
                # Save intermediate model after each epoch
                self.model.save(f'intermediate_model_epoch_{epoch+1}.keras')
                print(f'\nSaved intermediate model for epoch {epoch+1}')
            except KeyboardInterrupt:
                self.interrupted = True
                print('\nTraining interrupted. Saving model...')
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.model.save(f'interrupted_model_{timestamp}.keras')
                print(f'Model saved as interrupted_model_{timestamp}.keras')
                self.model.stop_training = True

    # Regular callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_f1_score',
        patience=10,
        restore_best_weights=True,
        mode='max'
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_f1_score',
        factor=0.2,
        patience=5,
        mode='max'
    )
    
    # Model checkpoint with timestamp
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model_{epoch:02d}_{val_f1_score:.4f}.keras',
        monitor='val_f1_score',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    interrupt_callback = InterruptCallback()
    
    try:
        # Train with class weights
        print("Training model...")
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            class_weight=class_weight_dict,
            callbacks=[early_stopping, reduce_lr, checkpoint, interrupt_callback],
            verbose=1
        )
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Final save in progress...")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model.save(f'final_interrupted_model_{timestamp}.keras')
        print(f"Model saved as final_interrupted_model_{timestamp}.keras")
    
    finally:
        # Save final model regardless of how training ended
        try:
            print("\nSaving final model state...")
            model.save('car_crash_detection_model_final.keras')
            print("Final model saved successfully")
            
            # Print final metrics if possible
            print("\nEvaluating model on test set...")
            test_loss, test_accuracy, test_f1 = model.evaluate(X_test, y_test, verbose=0)
            print(f"Test accuracy: {test_accuracy:.4f}")
            print(f"Test F1 score: {test_f1:.4f}")
            
        except Exception as e:
            print(f"Error saving final model: {str(e)}")
    
    return history, model

if __name__ == "__main__":
    try:
        history, model = train_model()
    except Exception as e:
        print(f"Training failed with error: {str(e)}")