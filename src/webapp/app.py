#!/usr/bin/env python3
"""
TrustVision AI - Real-time Web Application

Two Modes:
1. Live Interview Analysis - Liveness detection, deepfake detection
2. Safety Hazard Detection - YOLO-based hazard detection for senior living

Features:
- Real-time webcam streaming via WebSocket
- Mobile-friendly responsive design
- Docker deployable
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import base64
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from datetime import datetime
import os

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
app.config['SECRET_KEY'] = 'trustvision_ai_2024'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global models
yolo_model = None
deepfake_model = None
deepfake_processor = None
face_mesh = None
face_cascade = None

# Blink detection state
blink_state = {
    'ear_history': [],
    'blink_count': 0,
    'consecutive_low': 0
}

EAR_THRESHOLD = 0.21
EAR_CONSECUTIVE_FRAMES = 2

# MediaPipe eye landmarks
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]


def load_models():
    """Load all models on startup"""
    global yolo_model, deepfake_model, deepfake_processor, face_mesh, face_cascade

    print("Loading models...")

    # Load YOLO for safety detection
    try:
        model_path = os.path.join(os.path.dirname(__file__), '../../models/yolov8n.pt')
        if os.path.exists(model_path):
            yolo_model = YOLO(model_path)
        else:
            yolo_model = YOLO('yolov8n.pt')
        print("YOLO model loaded")
    except Exception as e:
        print(f"YOLO load error: {e}")

    # Load Deepfake detector
    try:
        model_name = "dima806/deepfake_vs_real_image_detection"
        deepfake_processor = AutoImageProcessor.from_pretrained(model_name)
        deepfake_model = AutoModelForImageClassification.from_pretrained(model_name)
        deepfake_model.eval()
        print("Deepfake model loaded")
    except Exception as e:
        print(f"Deepfake model error: {e}")

    # Load MediaPipe Face Mesh for liveness
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        print("MediaPipe Face Mesh loaded")
    except Exception as e:
        print(f"MediaPipe error: {e}")

    # Load OpenCV cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    print("All models loaded!")
    return True


def calculate_ear(landmarks, eye_indices, w, h):
    """Calculate Eye Aspect Ratio"""
    points = []
    for idx in eye_indices:
        lm = landmarks.landmark[idx]
        points.append((lm.x * w, lm.y * h))

    v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    h1 = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    if h1 == 0:
        return 0.3
    return (v1 + v2) / (2.0 * h1)


def analyze_liveness(frame):
    """Analyze frame for liveness indicators"""
    global blink_state

    if face_mesh is None:
        return None

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        return {
            'face_detected': False,
            'blink_count': blink_state['blink_count'],
            'is_live': False,
            'message': 'No face detected'
        }

    face_landmarks = results.multi_face_landmarks[0]

    # Calculate EAR
    left_ear = calculate_ear(face_landmarks, LEFT_EYE_INDICES, w, h)
    right_ear = calculate_ear(face_landmarks, RIGHT_EYE_INDICES, w, h)
    ear = (left_ear + right_ear) / 2.0

    # Blink detection
    if ear < EAR_THRESHOLD:
        blink_state['consecutive_low'] += 1
    else:
        if blink_state['consecutive_low'] >= EAR_CONSECUTIVE_FRAMES:
            blink_state['blink_count'] += 1
        blink_state['consecutive_low'] = 0

    # Head pose estimation (simple)
    nose = face_landmarks.landmark[1]
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]

    eye_center_x = (left_eye.x + right_eye.x) / 2
    yaw = (nose.x - eye_center_x) * 90

    is_live = blink_state['blink_count'] >= 1

    return {
        'face_detected': True,
        'blink_count': blink_state['blink_count'],
        'ear': round(ear, 3),
        'yaw': round(yaw, 1),
        'is_live': is_live,
        'message': 'LIVE - Blinks detected!' if is_live else 'Waiting for blinks...'
    }


def analyze_deepfake(frame):
    """Analyze frame for deepfake detection"""
    if deepfake_model is None or face_cascade is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

    if len(faces) == 0:
        return {'face_detected': False, 'message': 'No face detected'}

    # Get largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Add margin
    margin = int(0.2 * max(w, h))
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(frame.shape[1] - x, w + 2 * margin)
    h = min(frame.shape[0] - y, h + 2 * margin)

    face_region = frame[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(face_rgb)

    # Run inference
    inputs = deepfake_processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        outputs = deepfake_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    probs_np = probs.cpu().numpy()[0]

    # Get label mapping
    id2label = deepfake_model.config.id2label
    real_idx = 1
    fake_idx = 0

    for idx, label in id2label.items():
        if 'real' in label.lower():
            real_idx = idx
        elif 'fake' in label.lower():
            fake_idx = idx

    real_score = float(probs_np[real_idx])
    fake_score = float(probs_np[fake_idx])

    is_real = real_score > fake_score

    return {
        'face_detected': True,
        'is_real': is_real,
        'real_score': round(real_score * 100, 1),
        'fake_score': round(fake_score * 100, 1),
        'bbox': [int(x), int(y), int(w), int(h)],
        'message': 'AUTHENTIC' if is_real else 'POTENTIAL DEEPFAKE!'
    }


def analyze_safety_hazards(frame):
    """Analyze frame for safety hazards using YOLO"""
    if yolo_model is None:
        return {'error': 'YOLO model not loaded'}

    results = yolo_model(frame, verbose=False)

    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    # Safety hazard categories
    hazard_categories = {
        'high_risk': ['bottle', 'knife', 'scissors', 'bowl', 'cup', 'wine glass'],
        'trip_hazards': ['backpack', 'suitcase', 'handbag', 'umbrella', 'book', 'remote'],
        'furniture': ['chair', 'couch', 'bed', 'dining table'],
        'fixtures': ['toilet', 'sink', 'refrigerator', 'oven', 'microwave']
    }

    analysis = {
        'high_risk_items': [],
        'medium_risk_items': [],
        'low_risk_items': [],
        'detections': [],
        'room_status': 'SAFE',
        'total_hazards': 0
    }

    frame_height = frame.shape[0]

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_idx = int(box.cls[0])
                conf = float(box.conf[0])

                if conf > 0.4 and cls_idx < len(coco_classes):
                    class_name = coco_classes[cls_idx]

                    if class_name == 'person':
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    is_floor_level = y2 > frame_height * 0.6

                    detection = {
                        'class': class_name,
                        'confidence': round(conf * 100, 1),
                        'bbox': [x1, y1, x2, y2],
                        'is_floor_level': is_floor_level
                    }

                    # Categorize risk
                    if class_name in hazard_categories['high_risk'] and is_floor_level:
                        detection['risk'] = 'HIGH'
                        detection['color'] = '#ff4444'
                        detection['message'] = f'WARNING: {class_name.upper()} on floor - Trip/slip hazard!'
                        analysis['high_risk_items'].append(detection)
                        analysis['total_hazards'] += 1
                    elif class_name in hazard_categories['trip_hazards'] and is_floor_level:
                        detection['risk'] = 'MEDIUM'
                        detection['color'] = '#ffaa00'
                        detection['message'] = f'CAUTION: {class_name.upper()} may obstruct path'
                        analysis['medium_risk_items'].append(detection)
                        analysis['total_hazards'] += 1
                    elif class_name in hazard_categories['fixtures']:
                        detection['risk'] = 'INFO'
                        detection['color'] = '#4488ff'
                        detection['message'] = f'Fixture: {class_name}'
                    else:
                        detection['risk'] = 'LOW'
                        detection['color'] = '#44ff44'
                        detection['message'] = f'OK: {class_name}'
                        analysis['low_risk_items'].append(detection)

                    analysis['detections'].append(detection)

    # Determine room status
    if len(analysis['high_risk_items']) > 0:
        analysis['room_status'] = 'HAZARDS DETECTED'
    elif len(analysis['medium_risk_items']) > 0:
        analysis['room_status'] = 'CAUTION ADVISED'
    else:
        analysis['room_status'] = 'SAFE'

    return analysis


def decode_frame(frame_data):
    """Decode base64 frame to OpenCV format"""
    try:
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        image_data = base64.b64decode(frame_data)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Frame decode error: {e}")
        return None


# Routes
@app.route('/')
def index():
    """Main landing page with mode selection"""
    return render_template('index.html')


@app.route('/interview')
def interview_mode():
    """Live interview analysis mode"""
    return render_template('interview.html')


@app.route('/safety')
def safety_mode():
    """Safety hazard detection mode"""
    return render_template('safety.html')


@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'yolo': yolo_model is not None,
            'deepfake': deepfake_model is not None,
            'face_mesh': face_mesh is not None
        }
    })


# WebSocket handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print("Client connected")
    emit('connected', {'status': 'Connected to TrustVision AI'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    global blink_state
    blink_state = {'ear_history': [], 'blink_count': 0, 'consecutive_low': 0}
    print("Client disconnected")


@socketio.on('reset_session')
def handle_reset():
    """Reset analysis session"""
    global blink_state
    blink_state = {'ear_history': [], 'blink_count': 0, 'consecutive_low': 0}
    emit('session_reset', {'status': 'Session reset'})


@socketio.on('analyze_interview')
def handle_interview_analysis(data):
    """Handle real-time interview analysis"""
    try:
        frame = decode_frame(data.get('image', ''))
        if frame is None:
            emit('interview_result', {'error': 'Invalid frame'})
            return

        # Run both analyses
        liveness = analyze_liveness(frame)
        deepfake = analyze_deepfake(frame)

        result = {
            'timestamp': datetime.now().isoformat(),
            'liveness': liveness,
            'deepfake': deepfake
        }

        emit('interview_result', result)

    except Exception as e:
        emit('interview_result', {'error': str(e)})


@socketio.on('analyze_safety')
def handle_safety_analysis(data):
    """Handle real-time safety hazard analysis"""
    try:
        frame = decode_frame(data.get('image', ''))
        if frame is None:
            emit('safety_result', {'error': 'Invalid frame'})
            return

        analysis = analyze_safety_hazards(frame)
        analysis['timestamp'] = datetime.now().isoformat()

        emit('safety_result', analysis)

    except Exception as e:
        emit('safety_result', {'error': str(e)})


if __name__ == '__main__':
    print("TrustVision AI - Real-time Web Application")
    print("==========================================")

    if load_models():
        print("\nStarting server...")
        print("Open http://localhost:5000 in your browser")
        print("For mobile: http://[YOUR_IP]:5000")
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    else:
        print("Failed to load models!")
        exit(1)
