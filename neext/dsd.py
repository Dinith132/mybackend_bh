import numpy as np
import tensorflow as tf
import joblib
import cv2
import mediapipe as mp
from ultralytics import YOLO
import math

# === Paths ===
VIDEO_PATH = "/content/drive/MyDrive/Research/ModelJULY24/Test Video/FT36-LA.mp4"
SCALER_PATH = "/content/drive/MyDrive/Research/ModelJULY24/Scaler/scaler.save"
MODEL_PATH = "/content/drive/MyDrive/Research/ModelJULY24/Model/final_model_attention.keras"

# === Load model and scaler ===
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# === Load YOLO and MediaPipe ===
yolo = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# === Utility: Calculate angle between 3 points ===
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

# === Utility: Extract 6 joint angles ===
def extract_joint_angles(landmarks):
    joints = []
    # Right Elbow
    joints.append(calculate_angle(landmarks[12][:3], landmarks[14][:3], landmarks[16][:3]))
    # Left Elbow
    joints.append(calculate_angle(landmarks[11][:3], landmarks[13][:3], landmarks[15][:3]))
    # Right Shoulder
    joints.append(calculate_angle(landmarks[24][:3], landmarks[12][:3], landmarks[14][:3]))
    # Left Shoulder
    joints.append(calculate_angle(landmarks[23][:3], landmarks[11][:3], landmarks[13][:3]))
    # Right Knee
    joints.append(calculate_angle(landmarks[24][:3], landmarks[26][:3], landmarks[28][:3]))
    # Left Knee
    joints.append(calculate_angle(landmarks[23][:3], landmarks[25][:3], landmarks[27][:3]))
    return np.array(joints)

# === Step 1: Extract 10 annotated frames and features ===
def extract_features_from_video(video_path, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo(frame, verbose=False)
        if isinstance(results, list) and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes.xyxy.shape[0] > 0:
                x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy()[0].astype(int)
                cropped = frame[y1:y2, x1:x2]
                rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                results_pose = pose.process(rgb)

                if results_pose.pose_landmarks:
                    landmarks = np.array(results_pose.pose_landmarks.landmark)
                    landmark_array = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])
                    joint_angles = extract_joint_angles(landmark_array)  # shape (6,)
                    pose_feats = landmark_array.flatten()  # shape (132,)
                    full_feature = np.concatenate([pose_feats, joint_angles])  # 138
                    features.append(full_feature)
                    frame_count += 1

    cap.release()
    pose.close()

    while len(features) < 10:
        features.append(np.zeros(138))

    return np.array(features)[:10]

# === Step 2: Final tensor prep ===
def prepare_input_tensor(frames_10):
    if frames_10.shape[0] != 10:
        raise ValueError(f"Need 10 frames, got {frames_10.shape[0]}")

    velocity = np.gradient(frames_10[:, :99], axis=0)  # 99 = x,y,z
    final_features = np.concatenate([frames_10, velocity], axis=-1)  # (10, 237)
    final_scaled = scaler.transform(final_features)  # (10, 237)
    return np.expand_dims(final_scaled, axis=0)  # (1, 10, 237)

# === Step 3: Predict ===
frames = extract_features_from_video(VIDEO_PATH)
input_tensor = prepare_input_tensor(frames)
pred = model.predict(input_tensor)

# === Display Result ===
class_names = ['GoodTechnique', 'LowArm', 'PoorLeftLegBlock', 'BothErrors']
predicted_class = np.argmax(pred)
confidence = np.max(pred)

print("\n🔍 Class-wise Probabilities:")
for i, prob in enumerate(pred[0]):
    print(f"  {class_names[i]}: {prob:.3f}")

print(f"\n✅ Predicted Class: {class_names[predicted_class]} | Confidence: {confidence:.4f}")