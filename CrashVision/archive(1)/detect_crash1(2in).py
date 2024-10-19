import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained model
model = load_model('car_crash_detection_model.h5')

# Access the camera feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_resized = cv2.resize(frame, (64, 64))
    frame_normalized = frame_resized / 255.0
    frame_input = np.expand_dims(frame_normalized, axis=0)

    # Predict crash severity and accident detection
    severity_prediction, accident_prediction = model.predict(frame_input)

    # Get the predicted severity and accident confidence
    severity = np.argmax(severity_prediction) + 1  # 1, 2, 3 for severity
    accident_confidence = accident_prediction[0][0]  # Get the probability for accident

    # Define a confidence threshold for accident detection
    confidence_threshold = 0.5

    # Determine if an accident is detected
    accident_detected = accident_confidence > confidence_threshold

    # If an accident is detected, draw the bounding box
    if accident_detected:
        box_color = (0, 0, 255)  # Red color for accident
        box_coordinates = (50, 50, 150, 150)  # Example coordinates, adjust as needed

        # Draw the bounding box on the frame
        start_point = (box_coordinates[0], box_coordinates[1])
        end_point = (box_coordinates[2], box_coordinates[3])
        cv2.rectangle(frame, start_point, end_point, box_color, 2)

        # Display severity on the frame
        cv2.putText(frame, f"Crash Severity: {severity}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # When no accident is detected, display severity level 0
        cv2.putText(frame, "Crash Severity: 0", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow('Crash Detection', frame)

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
