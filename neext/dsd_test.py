import os
import math
import cv2
import numpy as np
import joblib
import tensorflow as tf
import mediapipe as mp
from ultralytics import YOLO

# ===================== SERVER PATHS =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "input.mp4")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.save")
MODEL_PATH = os.path.join(BASE_DIR, "final_model_attention.keras")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Load model & scaler ----------
print("Loading model and scaler...")
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ---------- Init YOLO & MediaPipe ----------
yolo = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose
POSE_CONNS = list(mp_pose.POSE_CONNECTIONS)

# ---------- Geometry utils ----------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def extract_joint_angles(landmarks):
    joints = [
        calculate_angle(landmarks[12][:3], landmarks[14][:3], landmarks[16][:3]),  # Right elbow
        calculate_angle(landmarks[11][:3], landmarks[13][:3], landmarks[15][:3]),  # Left elbow
        calculate_angle(landmarks[24][:3], landmarks[12][:3], landmarks[14][:3]),  # Right shoulder
        calculate_angle(landmarks[23][:3], landmarks[11][:3], landmarks[13][:3]),  # Left shoulder
        calculate_angle(landmarks[24][:3], landmarks[26][:3], landmarks[28][:3]),  # Right knee
        calculate_angle(landmarks[23][:3], landmarks[25][:3], landmarks[27][:3])   # Left knee
    ]
    return np.array(joints, dtype=np.float32)

# ---------- Helper functions ----------
def pick_best_person_box(result):
    if not hasattr(result, "boxes") or result.boxes is None or result.boxes.xyxy.shape[0] == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else None

    idx = None
    if cls is not None:
        person_idxs = [i for i, c in enumerate(cls) if int(c) == 0]
        if person_idxs:
            idx = person_idxs[int(np.argmax(confs[person_idxs]))]
    if idx is None:
        idx = int(np.argmax(confs))
    return tuple(boxes[idx].tolist())

def crop_with_margin(frame, box, margin_ratio=0.08):
    if box is None:
        return frame, 0, 0
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin_ratio), int(bh * margin_ratio)
    X1, Y1 = max(0, x1 - mx), max(0, y1 - my)
    X2, Y2 = min(w, x2 + mx), min(h, y2 + my)
    if X2 <= X1 or Y2 <= Y1:
        return frame, 0, 0
    return frame[Y1:Y2, X1:X2], X1, Y1

# ---------- Feature extraction ----------
def extract_features_from_video(video_path, max_frames=10):
    print("Extracting features...")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return np.zeros((max_frames, 138), dtype=np.float32)

    frame_indices = np.linspace(0, total - 1, max_frames, dtype=int)
    features = []

    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        last_box = None
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                features.append(np.zeros(138, dtype=np.float32))
                continue

            results = yolo(frame, verbose=False)
            result = results[0] if isinstance(results, list) and len(results) > 0 else results
            box = pick_best_person_box(result) or last_box
            last_box = box if box is not None else last_box

            cropped, _, _ = crop_with_margin(frame, box, margin_ratio=0.08)
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pres = pose.process(rgb)

            if not pres.pose_landmarks:
                pres = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if pres.pose_landmarks:
                lm = np.array([[p.x, p.y, p.z, p.visibility] for p in pres.pose_landmarks.landmark], dtype=np.float32)
                pose_feats = lm[:, :3].flatten()
                vis_feats  = lm[:, 3].flatten()
                angles6    = extract_joint_angles(lm)
                features.append(np.concatenate([pose_feats, vis_feats, angles6], axis=0))
            else:
                features.append(np.zeros(138, dtype=np.float32))

    cap.release()
    return np.array(features, dtype=np.float32)

def prepare_input_tensor(frames_10):
    if frames_10.shape[0] != 10:
        raise ValueError(f"Expected 10 frames, got {frames_10.shape[0]}")
    velocity = np.gradient(frames_10[:, :99], axis=0)
    final_features = np.concatenate([frames_10, velocity], axis=-1)
    final_scaled = scaler.transform(final_features)
    return np.expand_dims(final_scaled, axis=0)

# ---------- Class highlighting ----------
CLASS_NAMES = ['Good Technique', 'Low Arm', 'Poor Left Leg Block', 'Both Errors']
ARM_POINTS = {11,12,13,14,15,16}
LEFT_LEG_POINTS = {23,25,27}

def class_to_highlight(pred_label):
    if pred_label == "Good Technique": return set()
    if pred_label == "Low Arm": return ARM_POINTS
    if pred_label == "Poor Left Leg Block": return LEFT_LEG_POINTS
    if pred_label == "Both Errors": return ARM_POINTS | LEFT_LEG_POINTS
    return set()

def draw_skeleton(img, landmarks_xy, highlight_points):
    GREEN, RED = (60,255,60), (20,30,240)
    for a,b in POSE_CONNS:
        if np.any(np.isnan(landmarks_xy[[a,b]])): continue
        col = RED if (a in highlight_points or b in highlight_points) else GREEN
        cv2.line(img, tuple(landmarks_xy[a]), tuple(landmarks_xy[b]), col, 3)
    for idx,(x,y) in enumerate(landmarks_xy):
        if np.isnan(x) or np.isnan(y): continue
        col = RED if idx in highlight_points else GREEN
        cv2.circle(img, (int(x),int(y)), 4, col, -1)

# ---------- Predict ----------
def final_prediction(video_path, file_id=0):
    frames = extract_features_from_video(video_path)
    input_tensor = prepare_input_tensor(frames)
    print("Getting prediction...")
    pred = model.predict(input_tensor, verbose=0)
    print("Prediction done.")

    predicted_class = CLASS_NAMES[int(np.argmax(pred))]
    confidence = float(np.max(pred))
    print(f"\n✅ Predicted Class: {predicted_class} | Confidence: {confidence:.4f}")

    # Visual feedback
    save_frames_with_pose(video_path, file_id)

    result = {
        "prediction": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": {CLASS_NAMES[i]: round(float(pred[0][i]),2) for i in range(4)}
    }
    print(result)
    return result

# ---------- Visual feedback ----------
def save_frames_with_pose(video_path, file_id, output_dir=OUTPUT_DIR, num_frames=50):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"out{file_id}.mp4")
    cap = cv2.VideoCapture(video_path)
    fps = 25
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < num_frames: num_frames = total
    frame_indices = np.linspace(0, total-1, num_frames, dtype=int)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W,H))

    highlight_points = set()  # No dynamic highlighting here, just placeholder
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as pose:
        last_box = None
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue
            results = yolo(frame, verbose=False)
            result = results[0] if isinstance(results,list) and len(results)>0 else results
            box = pick_best_person_box(result) or last_box
            last_box = box if box is not None else last_box
            cropped, off_x, off_y = crop_with_margin(frame, box, margin_ratio=0.08)
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pres = pose.process(rgb)
            if pres.pose_landmarks:
                lm_xy = np.array([[p.x * cropped.shape[1] + off_x, p.y * cropped.shape[0] + off_y] for p in pres.pose_landmarks.landmark])
                draw_skeleton(frame, lm_xy, highlight_points)
            out.write(frame)
    cap.release()
    out.release()
    print(f"🎥 Video saved: {output_path}")

# ===================== RUN =====================
if __name__ == "__main__":
    final_prediction(VIDEO_PATH, file_id=1)
