import os
import cv2
import mediapipe as mp
import numpy as np

# --- Paths ---
input_root = r"D:\Academic\Research\New Dataset\BoundingBoxAnnotated\BothErrors(891)"
label_root = r"D:\Academic\Research\New Dataset\YOLO_Labels\BothErrors(891)"
output_X_file = r"D:\Academic\Research\New Dataset\BoundingBoxAnnotated\BothErrors(891)\X_both_errors_enhanced.npy"
output_y_file = r"D:\Academic\Research\New Dataset\BoundingBoxAnnotated\BothErrors(891)\y_both_errors_enhanced.npy"

# --- Settings ---
label = 3  #Both Errors
NUM_FRAMES = 10
NUM_LANDMARKS = 33
RAW_POSE_SIZE = NUM_LANDMARKS * 4  # x, y, z, visibility

# --- Joint Angle Helpers ---
def calc_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6), -1.0, 1.0))
    return np.degrees(angle)

def extract_joint_angles(landmarks):
    joints = [
        (11, 13, 15),  # Right elbow
        (12, 14, 16),  # Left elbow
        (23, 25, 27),  # Right knee
        (24, 26, 28),  # Left knee
        (11, 23, 25),  # Hip-right knee
        (12, 24, 26),  # Hip-left knee
    ]
    angles = []
    for a, b, c in joints:
        angle = calc_angle(landmarks[a], landmarks[b], landmarks[c])
        angles.append(angle)
    return angles

# --- Init ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
X_data, y_data = [], []

# --- Loop Over Samples ---
for folder_name in sorted(os.listdir(input_root)):
    folder_path = os.path.join(input_root, folder_name)
    if not os.path.isdir(folder_path):
        continue

    sequence_raw = []
    sequence_angles = []

    print(f"\n📁 Folder: {folder_name}")
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Couldn't read: {img_path}")
            continue

        # Use label path from YOLO label root
        bbox_txt_path = os.path.join(label_root, folder_name, os.path.splitext(img_name)[0] + ".txt")
        if not os.path.exists(bbox_txt_path):
            print(f"⚠️ No bounding box found for {img_name}")
            continue

        with open(bbox_txt_path, "r") as f:
            line = f.readline().strip().split()
            if len(line) < 5:
                continue
            _, x_center, y_center, w, h = map(float, line)
            H, W, _ = img.shape
            x1 = int((x_center - w / 2) * W)
            y1 = int((y_center - h / 2) * H)
            x2 = int((x_center + w / 2) * W)
            y2 = int((y_center + h / 2) * H)
            cropped = img[y1:y2, x1:x2]

        img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = []
            coords = []
            for lm in landmarks:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
                coords.append([lm.x, lm.y, lm.z])
            sequence_raw.append(keypoints)
            sequence_angles.append(extract_joint_angles(coords))
        else:
            sequence_raw.append([0]*RAW_POSE_SIZE)
            sequence_angles.append([0]*6)

    # Pad or trim to 10 frames
    if len(sequence_raw) < NUM_FRAMES:
        missing = NUM_FRAMES - len(sequence_raw)
        sequence_raw += [[0]*RAW_POSE_SIZE] * missing
        sequence_angles += [[0]*6] * missing
    elif len(sequence_raw) > NUM_FRAMES:
        sequence_raw = sequence_raw[:NUM_FRAMES]
        sequence_angles = sequence_angles[:NUM_FRAMES]

    if len(sequence_raw) == NUM_FRAMES:
        # Compute velocities from raw pose
        raw_np = np.array(sequence_raw)
        velocities = np.gradient(raw_np[:, :99], axis=0)  # x,y,z of 33 joints

        # Combine raw + angles + velocities
        enhanced = np.concatenate([raw_np, np.array(sequence_angles), velocities], axis=1)
        X_data.append(enhanced)
        y_data.append(label)
        print(f"✅ Sequence saved. Frames: {len(enhanced)}")
    else:
        print("⚠️ Incomplete sequence.")

# --- Save ---
X_data = np.array(X_data)
y_data = np.array(y_data)
np.save(output_X_file, X_data)
np.save(output_y_file, y_data)

print(f"\n✅ Final X shape: {X_data.shape} (Frames x Features)")
print(f"✅ Labels shape: {y_data.shape}")
print(f"✅ Data saved to {output_X_file} and {output_y_file}")