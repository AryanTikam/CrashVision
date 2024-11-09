import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Conv2D
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import albumentations as A

def create_augmentation_pipeline():
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=3),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3)
    ])

def create_multitask_model(input_shape, num_classes=4):
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Load pre-trained EfficientNetB4 (better feature extraction than ResNet50)
    base_model = EfficientNetB4(weights='imagenet', include_top=False, input_tensor=input_layer)
    
    # Fine-tune more layers for better feature learning
    for layer in base_model.layers[:-50]:  # Fine-tune last 50 layers
        layer.trainable = False
    
    # Common layers with improved architecture
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    
    # Additional convolutional layers for better feature extraction
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Severity branch with proper shape
    severity_branch = Dense(256, activation='relu', name='severity_dense')(x)
    severity_branch = BatchNormalization()(severity_branch)
    severity_branch = Dropout(0.2)(severity_branch)
    severity_output = Dense(num_classes, activation='softmax', name='severity')(severity_branch)
    
    # Accident branch with proper shape
    accident_branch = Dense(256, activation='relu', name='accident_dense')(x)
    accident_branch = BatchNormalization()(accident_branch)
    accident_branch = Dropout(0.2)(accident_branch)
    accident_output = Dense(num_classes, activation='softmax', name='accident')(accident_branch)
    
    # Create model
    model = Model(inputs=input_layer, outputs=[severity_output, accident_output])
    
    # Custom learning rate and optimizer
    initial_learning_rate = 1e-4
    optimizer = Adam(learning_rate=initial_learning_rate)
    
    # Compile model with categorical crossentropy for both outputs
    model.compile(
        optimizer=optimizer,
        loss={
            'severity': 'sparse_categorical_crossentropy',
            'accident': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'severity': 1.0,
            'accident': 1.0
        },
        metrics={
            'severity': ['accuracy'],
            'accident': ['accuracy']
        }
    )
    
    return model

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    img_resized = cv2.resize(img, target_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    return preprocess_input(img_rgb)

def load_images_and_labels(base_path, score_dfs, augmentation_pipeline):
    X = []
    y_severity = []
    y_accident = []
    
    print("\nLoading severity images...")
    for severity in range(1, 4):
        folder_path = os.path.join(base_path, str(severity))
        df = score_dfs[severity - 1]
        print(f"\nProcessing severity level {severity} images from {folder_path}")
        
        for img_filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_filename)
            img_preprocessed = load_and_preprocess_image(img_path)
            
            if img_preprocessed is not None:
                # Original image
                X.append(img_preprocessed)
                y_severity.append(severity)
                y_accident.append(severity)  # Using same classes for accident detection
                
                # Augmented versions
                augmented = augmentation_pipeline(image=img_preprocessed)['image']
                X.append(augmented)
                y_severity.append(severity)
                y_accident.append(severity)

    # Load non-accident images
    non_accident_folder = os.path.join(base_path, 'NonAccident')
    print(f"\nProcessing non-accident images from {non_accident_folder}")
    for img_filename in os.listdir(non_accident_folder):
        img_path = os.path.join(non_accident_folder, img_filename)
        img_preprocessed = load_and_preprocess_image(img_path)
        
        if img_preprocessed is not None:
            X.append(img_preprocessed)
            y_severity.append(0)
            y_accident.append(0)

    return np.array(X), np.array(y_severity), np.array(y_accident)

if __name__ == "__main__":
    # Set memory growth for GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    
    print("Loading Excel files...")
    file1 = './archive(1)/SeverityScore/Severity Score Dataset with Labels/score1.xlsx'
    file2 = './archive(1)/SeverityScore/Severity Score Dataset with Labels/score2.xlsx'
    file3 = './archive(1)/SeverityScore/Severity Score Dataset with Labels/score3.xlsx'

    score1 = pd.read_excel(file1)
    score2 = pd.read_excel(file2)
    score3 = pd.read_excel(file3)

    base_path = "./archive(1)/SeverityScore/Severity Score Dataset with Labels"
    score_dfs = [score1, score2, score3]

    # Create augmentation pipeline
    augmentation_pipeline = create_augmentation_pipeline()

    # Load and split data
    print("\nLoading and processing images...")
    X, y_severity, y_accident = load_images_and_labels(base_path, score_dfs, augmentation_pipeline)

    print("\nSplitting dataset...")
    X_train, X_test, y_train_severity, y_test_severity, y_train_accident, y_test_accident = train_test_split(
        X, y_severity, y_accident, test_size=0.2, random_state=42, stratify=y_severity
    )

    # Create callbacks
    checkpoint = ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )

    # Create and train model
    print("\nCreating and compiling model...")
    multi_output_model = create_multitask_model(input_shape=(224, 224, 3))

    print("\nTraining model...")
    history = multi_output_model.fit(
        X_train,
        {
            'severity': y_train_severity,
            'accident': y_train_accident
        },
        validation_split=0.2,
        epochs=4,
        batch_size=16,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )

    # Evaluate the model
    print("\nEvaluating model...")
    evaluation = multi_output_model.evaluate(
        X_test,
        {
            'severity': y_test_severity,
            'accident': y_test_accident
        },
        verbose=1
    )

    print("\nTest Results:")
    for metric_name, metric_value in zip(multi_output_model.metrics_names, evaluation):
        print(f"{metric_name}: {metric_value:.4f}")