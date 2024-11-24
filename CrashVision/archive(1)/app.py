# app.py
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
from datetime import datetime
import requests
from sklearn.cluster import KMeans
from functools import wraps
import threading
import base64
from flask_socketio import SocketIO
import logging

app = Flask(__name__)
socketio = SocketIO(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Email Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "aryantikam297@gmail.com"  # Replace with your email 
SENDER_PASSWORD = "dtpq rkek ckus gdgy"  # Replace with your app password
RECIPIENT_EMAIL = "aryan.tikam23@spit.ac.in"  # Replace with recipient email

# Global variables
camera = None
accident_detected = False
last_email_sent = None
EMAIL_COOLDOWN = 300  # 5 minutes cooldown between emails

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

# Load models
try:
    model = load_model('car_crash_detection_model_improved_with_f1v4.keras', 
                      custom_objects={'f1_score': f1_score})
    net = cv2.dnn.readNet("./archive(1)/yolo/yolov3.weights", 
                         "./archive(1)/yolo/yolov3.cfg")
    
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
    
    with open('./archive(1)/yolo/coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
        
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

def get_location():
    try:
        ip_response = requests.get('https://api.ipify.org?format=json')
        ip_address = ip_response.json()['ip']
        
        location_response = requests.get(f'http://ip-api.com/json/{ip_address}')
        location_data = location_response.json()
        
        if location_data['status'] == 'success':
            return {
                'latitude': location_data['lat'],
                'longitude': location_data['lon'],
                'city': location_data['city'],
                'country': location_data['country']
            }
    except Exception as e:
        logger.error(f"Error getting location: {str(e)}")
    
    return None

def send_email_alert(image_data, location):
    global last_email_sent
    
    current_time = datetime.now()
    if last_email_sent and (current_time - last_email_sent).total_seconds() < EMAIL_COOLDOWN:
        return
    
    try:
        msg = MIMEMultipart()
        msg['Subject'] = 'URGENT: Car Accident Detected!'
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL

        # Email body
        body = f"""
        Car accident detected at:
        Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
        Location: {location['city']}, {location['country']}
        Coordinates: {location['latitude']}, {location['longitude']}
        Google Maps Link: https://www.google.com/maps?q={location['latitude']},{location['longitude']}
        """
        msg.attach(MIMEText(body, 'plain'))

        # Attach image
        image = MIMEImage(image_data)
        image.add_header('Content-ID', '<accident_image>')
        msg.attach(image)

        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        last_email_sent = current_time
        logger.info("Alert email sent successfully")
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")

def get_dominant_color(roi):
    if roi.size == 0:
        return "Unknown", (0, 0, 0)

    max_dimension = 100
    h, w = roi.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        roi = cv2.resize(roi, (int(w * scale), int(h * scale)))

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pixels = roi_rgb.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]
    
    dominant_color_hsv = cv2.cvtColor(np.uint8([[dominant_color]]), 
                                    cv2.COLOR_RGB2HSV)[0][0]
    
    color_ranges = {
        'Red': [([0, 100, 100], [10, 255, 255]), 
                ([160, 100, 100], [180, 255, 255])],
        'Blue': [([100, 50, 50], [130, 255, 255])],
        'Green': [([40, 50, 50], [80, 255, 255])],
        'White': [([0, 0, 200], [180, 30, 255])],
        'Black': [([0, 0, 0], [180, 30, 50])],
        'Silver': [([0, 0, 120], [180, 30, 200])],
        'Yellow': [([20, 100, 100], [40, 255, 255])],
        'Orange': [([10, 100, 100], [20, 255, 255])],
        'Brown': [([10, 50, 50], [20, 255, 150])],
        'Gray': [([0, 0, 50], [180, 30, 150])]
    }

    h, s, v = dominant_color_hsv

    def in_range(hsv, color_range):
        for range_pair in color_range:
            lower, upper = range_pair
            if all(lower[i] <= hsv[i] <= upper[i] for i in range(3)):
                return True
        return False

    for color_name, ranges in color_ranges.items():
        if in_range((h, s, v), ranges):
            return color_name, dominant_color

    return "Unknown", dominant_color

def generate_frames():
    global camera, accident_detected
    
    if camera is None:
        camera = cv2.VideoCapture(0)
    
    severity_colors = {
        0: (0, 0, 0),
        1: (0, 255, 0),
        2: (0, 255, 255),
        3: (0, 0, 255)
    }
    
    VEHICLE_IDS = [1, 2, 3, 4, 5, 6, 7, 8]
    VEHICLE_DETECTION_THRESHOLD = 0.6
    CRASH_CONFIDENCE_THRESHOLD = 0.85

    while True:
        success, frame = camera.read()
        if not success:
            break

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), 
                                   True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > VEHICLE_DETECTION_THRESHOLD and class_id in VEHICLE_IDS:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)
        accident_detected = False

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                roi = frame[max(0, y):min(height, y + h), 
                          max(0, x):min(width, x + w)]
                
                if roi.size == 0:
                    continue

                roi_resized = cv2.resize(roi, (64, 64)) / 255.0
                roi_input = np.expand_dims(roi_resized, axis=0)

                severity_prediction = model.predict(roi_input, verbose=0)
                severity = np.argmax(severity_prediction)
                prediction_confidence = np.max(severity_prediction)

                if severity > 0 and prediction_confidence > CRASH_CONFIDENCE_THRESHOLD:
                    accident_detected = True
                    color, _ = get_dominant_color(roi)
                    box_color = severity_colors[severity]
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                    label = f"{color} {classes[class_ids[i]]} - Severity: {severity} ({prediction_confidence:.2f})"
                    
                    font_scale = 0.5
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    thickness = 2
                    (label_w, label_h), baseline = cv2.getTextSize(label, font, 
                                                                 font_scale, thickness)
                    
                    cv2.rectangle(frame, (x, y - label_h - 10), 
                                (x + label_w, y), box_color, -1)
                    cv2.putText(frame, label, (x, y - 5), font, font_scale, 
                              (255, 255, 255), thickness)

                    # Send alert if accident detected
                    if accident_detected:
                        _, buffer = cv2.imencode('.jpg', frame)
                        image_data = buffer.tobytes()
                        location = get_location()
                        
                        if location:
                            threading.Thread(target=send_email_alert, 
                                          args=(image_data, location)).start()
                            socketio.emit('accident_detected', {
                                'location': location,
                                'severity': int(severity),
                                'confidence': float(prediction_confidence)
                            })

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({'accident_detected': accident_detected})

if __name__ == '__main__':
    socketio.run(app, debug=True)