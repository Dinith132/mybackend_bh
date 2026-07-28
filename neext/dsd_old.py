import os
import math
import cv2
import numpy as np
import joblib

# === Base Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "input.mp4")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.save")
MODEL_PATH = os.path.join(BASE_DIR, "final_model_attention.keras")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROJECT_ROOT = os.path.dirname(BASE_DIR)
YOLO_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "yolov8n.pt"),
    os.path.join(BASE_DIR, "yolov8n.pt"),
    "yolov8n.pt",
]

def resolve_yolo_weights():
    for candidate in YOLO_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return YOLO_CANDIDATES[0]

_model = None
_scaler = None
_yolo = None


def _get_mp_pose():
    import mediapipe as mp
    return mp.solutions.pose


def get_pose_connections():
    return list(_get_mp_pose().POSE_CONNECTIONS)


def draw_skeleton(frame, landmarks, src_w, src_h, off_x=0, off_y=0):
    frame_h, frame_w = frame.shape[:2]
    points = []
    visibility_threshold = 0.35

    for lm in landmarks:
        x = int(lm.x * src_w + off_x)
        y = int(lm.y * src_h + off_y)
        x = max(0, min(frame_w - 1, x))
        y = max(0, min(frame_h - 1, y))
        visibility = getattr(lm, "visibility", 1.0)
        points.append((x, y, visibility))

    for a, b in get_pose_connections():
        if a < len(points) and b < len(points):
            x1, y1, visibility_a = points[a]
            x2, y2, visibility_b = points[b]

            if visibility_a < visibility_threshold or visibility_b < visibility_threshold:
                continue

            start = (x1, y1)
            end = (x2, y2)

            # Draw a bold outline first, then the bright skeleton line.
            # This keeps the sketch visible on both dark and bright videos.
            cv2.line(frame, start, end, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.line(frame, start, end, (0, 255, 255), 3, cv2.LINE_AA)


def _ensure_models_loaded():
    global _model, _scaler, _yolo

    if _model is not None and _scaler is not None and _yolo is not None:
        return

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.isfile(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

    print("Loading ML models...")
    import tensorflow as tf
    from ultralytics import YOLO

    _model = tf.keras.models.load_model(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)
    _yolo = YOLO(resolve_yolo_weights())
    print("ML models loaded.")

def pick_best_person_box(result):
    if not hasattr(result, "boxes") or result.boxes is None or result.boxes.xyxy.shape[0] == 0:
        return None
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    cls  = boxes.cls.cpu().numpy() if boxes.cls is not None else None
    idx = None
    if cls is not None:
        pidx = [i for i, c in enumerate(cls) if int(c) == 0]  # class 0 = person
        if pidx:
            idx = pidx[int(np.argmax(conf[pidx]))]
    if idx is None:
        idx = int(np.argmax(conf))
    return tuple(xyxy[idx].tolist())

def crop_with_margin(frame, box, margin_ratio=0.08):
    if box is None:
        return frame, 0, 0
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = (x2-x1), (y2-y1)
    mx, my = int(bw*margin_ratio), int(bh*margin_ratio)
    X1 = max(0, x1 - mx); Y1 = max(0, y1 - my)
    X2 = min(w, x2 + mx); Y2 = min(h, y2 + my)
    if X2 <= X1 or Y2 <= Y1:
        return frame, 0, 0
    return frame[Y1:Y2, X1:X2], X1, Y1

# ============================================================

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
    joints.append(calculate_angle(landmarks[12][:3], landmarks[14][:3], landmarks[16][:3]))
    joints.append(calculate_angle(landmarks[11][:3], landmarks[13][:3], landmarks[15][:3]))
    joints.append(calculate_angle(landmarks[24][:3], landmarks[12][:3], landmarks[14][:3]))
    joints.append(calculate_angle(landmarks[23][:3], landmarks[11][:3], landmarks[13][:3]))
    joints.append(calculate_angle(landmarks[24][:3], landmarks[26][:3], landmarks[28][:3]))
    joints.append(calculate_angle(landmarks[23][:3], landmarks[25][:3], landmarks[27][:3]))
    return np.array(joints)

# === Extract features from video (enhanced) ===
def extract_features_from_video(video_path, max_frames=10):
    print("Extracting Features...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    if total_frames == 0:
        print("Video has no frames.")
        return np.zeros((max_frames, 138), dtype=np.float32)

    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    features = []
    last_box = None

    with _get_mp_pose().Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as pose:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                features.append(np.zeros(138, dtype=np.float32))
                continue

            results = _yolo(frame, verbose=False)
            result = results[0] if isinstance(results, list) and len(results) > 0 else results
            box = pick_best_person_box(result) or last_box
            last_box = box if box is not None else last_box

            cropped, off_x, off_y = crop_with_margin(frame, box, margin_ratio=0.08)
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(rgb)

            if not results_pose.pose_landmarks:
                rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_pose = pose.process(rgb_full)
                off_x, off_y = 0, 0

            if results_pose.pose_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results_pose.pose_landmarks.landmark], dtype=np.float32)
                joint_angles = extract_joint_angles(landmarks)
                pose_feats = landmarks.flatten()[:132]  # keep original server logic
                full_feature = np.concatenate([pose_feats, joint_angles])
                features.append(full_feature)
            else:
                features.append(np.zeros(138, dtype=np.float32))

    cap.release()
    return np.array(features, dtype=np.float32)

# === Prepare input tensor ===
def prepare_input_tensor(frames_10):
    if frames_10.shape[0] != 10:
        raise ValueError(f"Need 10 frames, got {frames_10.shape[0]}")
    velocity = np.gradient(frames_10[:, :99], axis=0)
    final_features = np.concatenate([frames_10, velocity], axis=-1)
    final_scaled = _scaler.transform(final_features)
    return np.expand_dims(final_scaled, axis=0)

# === Save frames with pose overlay ===
def save_frames_with_pose(video_path, file_id, output_dir=OUTPUT_DIR, num_frames=50):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"out{file_id}.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames < num_frames: num_frames = total_frames
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    last_box = None

    with _get_mp_pose().Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as pose:
        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue

            results = _yolo(frame, verbose=False)
            result = results[0] if isinstance(results, list) and len(results) > 0 else results
            box = pick_best_person_box(result) or last_box
            last_box = box if box is not None else last_box

            cropped, off_x, off_y = crop_with_margin(frame, box, margin_ratio=0.08)
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(rgb)
            if not results_pose.pose_landmarks:
                rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_pose = pose.process(rgb_full)
                off_x, off_y = 0, 0
                src_w, src_h = frame.shape[1], frame.shape[0]
            else:
                src_h, src_w = cropped.shape[:2]

            if results_pose.pose_landmarks:
                draw_skeleton(
                    frame,
                    results_pose.pose_landmarks.landmark,
                    src_w,
                    src_h,
                    off_x,
                    off_y,
                )

            out.write(frame)
            print(f"✅ Saved frame {i+1}/{num_frames} to video")

    cap.release()
    out.release()
    print(f"🎥 Pose video saved at {output_path}")

# === Final prediction ===
def final_prediction(video_path, file_id=0):
    _ensure_models_loaded()
    frames = extract_features_from_video(video_path)
    input_tensor = prepare_input_tensor(frames)
    print("Getting the prediction...")
    pred = _model.predict(input_tensor)
    print("Prediction done.")
    save_frames_with_pose(video_path, file_id)

    class_names = ['Good Technique', 'Low Arm', 'Poor Left Leg Block', 'Both Errors']
    predicted_class = np.argmax(pred)
    confidence = np.max(pred)

    print("\n🔍 Class-wise Probabilities:")
    for i, prob in enumerate(pred[0]):
        print(f"  {class_names[i]}: {prob:.3f}")
    print(f"\n✅ Predicted Class: {class_names[predicted_class]} | Confidence: {confidence:.4f}")

    result = {
        "prediction": class_names[predicted_class],
        "confidence": round(float(confidence), 2),
        "probabilities": {class_names[i]: round(float(pred[0][i]), 2) for i in range(4)}
    }
    print(result)
    return result
