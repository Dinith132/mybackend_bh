import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# ----------------------- USER CONFIG -----------------------
category = "BothErrors(891)"
annotated_dir = r"C:\Users\Dinusha\Desktop\RESEARCH Data set\BoundingBoxAnnotated\PoorLeftLegBlock(1628)"
yolo_label_dir = r"C:\Users\Dinusha\Desktop\RESEARCH Data set\YOLO_Labels\PoorLeftLegBlock(1628)"
output_dir = r"C:\Users\Dinusha\Desktop\RESEARCH Data set\New_MediaPipe\PoorLeftLegBlock(1628)"
output_features = os.path.join(output_dir, "features.npy")
output_labels = os.path.join(output_dir, "labels.npy")
label_index = 3  # set this for each class! E.g. 0 for BothErrors, 1 for GoodTechnique, etc.
# -----------------------------------------------------------

os.makedirs(output_dir, exist_ok=True)

# Example: dummy joint angles, replace with your real logic!
def compute_joint_angles(landmarks):
    # landmarks: shape (33, 4)
    return np.zeros(6)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# --- Gather all images and their label files ---
all_img_files = []
for subdir, _, files in os.walk(annotated_dir):
    for f in files:
        if f.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(subdir, f)
            rel = os.path.relpath(img_path, annotated_dir)
            lbl_path = os.path.join(yolo_label_dir, os.path.splitext(rel)[0] + ".txt")
            if os.path.exists(lbl_path):
                all_img_files.append((img_path, lbl_path))

all_img_files.sort()  # chronological order for correct sequence

features_list = []
frame_label_list = []

prev_kps = None
for img_path, lbl_path in tqdm(all_img_files, desc="Extracting features"):
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    with open(lbl_path, 'r') as f:
        lines = f.readlines()
    if not lines:
        continue
    # Use first bbox only (athlete)
    parts = lines[0].strip().split()
    _, xc, yc, bw, bh = map(float, parts)
    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    crop = img[y1:y2, x1:x2]
    if crop.shape[0] < 20 or crop.shape[1] < 20:
        continue
    # Pose
    results = pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        continue
    kps = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
    if kps.shape != (33, 4):
        continue
    kps_flat = kps.flatten()
    joint_angles = compute_joint_angles(kps)
    # 99 velocities (x,y,z for 33 keypoints)
    if prev_kps is None:
        velocities = np.zeros(99)
    else:
        velocities = (kps[:, :3] - prev_kps[:, :3]).flatten()
    prev_kps = kps.copy()
    feat_vec = np.concatenate([kps_flat, joint_angles, velocities])  # (237,)
    features_list.append(feat_vec)
    frame_label_list.append(label_index)

# --- Build LSTM sequences of 10 frames ---
features_arr = np.array(features_list)
labels_arr = np.array(frame_label_list)
sequence_length = 10

sequence_features = []
sequence_labels = []

for i in range(len(features_arr) - sequence_length + 1):
    seq = features_arr[i:i+sequence_length]
    if seq.shape != (sequence_length, 237):
        continue
    sequence_features.append(seq)
    sequence_labels.append(label_index)  # one label per sequence (category)

sequence_features = np.array(sequence_features)  # (num_samples, 10, 237)
sequence_labels = np.array(sequence_labels)      # (num_samples,)

np.save(output_features, sequence_features)
np.save(output_labels, sequence_labels)

print(f"Saved features to {output_features}: shape {sequence_features.shape}")
print(f"Saved labels to {output_labels}: shape {sequence_labels.shape}")
print("Done!")
