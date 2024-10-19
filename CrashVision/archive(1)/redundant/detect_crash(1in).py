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

    # Predict crash severity
    prediction = model.predict(frame_input)
    severity = np.argmax(prediction) + 1  # Assuming 1, 2, 3 for severity

    # Determine bounding box color based on severity
    if severity == 1:
        box_color = (0, 255, 0)  # Green for severity 1
    elif severity == 2:
        box_color = (0, 255, 255)  # Yellow for severity 2
    elif severity == 3:
        box_color = (0, 0, 255)  # Red for severity 3
    else:
        box_color = (255, 255, 255)  # White if severity is unknown

    # Simulate dynamic bounding box movement
    # You can adjust these calculations to better fit your use case
    height, width, _ = frame.shape
    box_width, box_height = 100, 100  # Define the size of the bounding box

    # Calculate the center of the bounding box based on severity
    center_x = int(width / 2)
    center_y = int(height / 2)

    # Adjust bounding box position (for demonstration purposes)
    if severity == 1:
        start_point = (center_x - box_width // 2, center_y - box_height // 2)
        end_point = (center_x + box_width // 2, center_y + box_height // 2)
    elif severity == 2:
        start_point = (center_x - box_width // 4, center_y - box_height // 4)
        end_point = (center_x + box_width // 4, center_y + box_height // 4)
    else:  # severity == 3
        start_point = (center_x, center_y)
        end_point = (center_x + box_width, center_y + box_height)

    # Draw the bounding box on the frame
    cv2.rectangle(frame, start_point, end_point, box_color, 2)

    # Display severity on the frame
    cv2.putText(frame, f"Crash Severity: {severity}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Crash Detection', frame)
    
    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
