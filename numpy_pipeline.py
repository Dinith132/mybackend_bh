import os
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import os

def extract_frames(video_path, num_frames=10):
    """Extract `num_frames` evenly spaced frames from the video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        num_frames = total_frames  # fallback if video is too short

    # Calculate the frame indices to sample
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frames.append(frame)

    cap.release()
    return frames


def get_main_person_box(img, model):
    """Use YOLO to detect the largest person box (class 0)."""
    results = model(img)[0]
    boxes = results.boxes.xywh.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    # Filter to person class only
    person_boxes = [box for i, box in enumerate(boxes) if int(classes[i]) == 0]

    if not person_boxes:
        return None

    # Choose the largest box (main athlete)
    person_boxes = sorted(person_boxes, key=lambda b: b[2] * b[3], reverse=True)
    return person_boxes[0]  # [x, y, w, h]

def crop_person(img, box):
    """Crop person bounding box area from image."""
    x, y, w, h = box
    x1, y1 = int(x - w / 2), int(y - h / 2)
    x2, y2 = int(x + w / 2), int(y + h / 2)
    h_img, w_img = img.shape[:2]

    # Clip coordinates to image boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    return img[y1:y2, x1:x2]


def compute_joint_angles(landmarks):
    # landmarks: shape (33, 4)
    return np.zeros(6)

def extract_pose_sequence(frames, yolo_model, pose_model):
    """Extract sequence of MediaPipe keypoints for the largest person in each frame."""
    # sequence = []

    # for img in frames:
    #     # Detect main person
    #     box = get_main_person_box(img, yolo_model)
    #     if box is None:
    #         keypoints = [0] * (33 * 4)
    #         sequence.append(keypoints)
    #         continue

    #     # Crop the main person and feed to MediaPipe
    #     cropped = crop_person(img, box)
    #     img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    #     results = pose_model.process(img_rgb)

    #     keypoints = []
    #     if results.pose_landmarks:
    #         for lm in results.pose_landmarks.landmark:
    #             keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    #     else:
    #         keypoints = [0] * (33 * 4)

    #     sequence.append(keypoints)

    # Ensure sequence is exactly 20 frames
    sequence = []
    prev_kps = None
    for img in frames:
        # Detect main person
        box = get_main_person_box(img, yolo_model)
        if box is None:
            keypoints = np.zeros(33 * 4)          # 132 keypoints
            joint_angles = np.zeros(6)  # define this
            velocities = np.zeros(99)             # 99 velocities
            feat_vec = np.concatenate([keypoints, joint_angles, velocities])
            sequence.append(feat_vec)
            continue

        # Crop the main person and feed to MediaPipe
        cropped = crop_person(img, box)
        img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        results = pose_model.process(img_rgb)

        keypoints = []
        # if results.pose_landmarks:
        #     for lm in results.pose_landmarks.landmark:
        #         keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        # else:
        #     keypoints = [0] * (33 * 4)

        # sequence.append(keypoints)
            # Ensure sequence is exactly 20 frames
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
        sequence.append(feat_vec)
        

        # keypoints = []

    if len(sequence) >= 10:
        return sequence[:10]
    else:
        missing = 10 - len(sequence)
        zero_frame = [0] * (237)
        return sequence + [zero_frame] * missing

def process_video_to_pose_npy(video_path, output_X_path):
    """Main function: full pipeline from video to .npy pose sequence + label."""

    print(f"\n🎬 Processing video: {video_path}")

    # Step 1: Extract frames
    frames = extract_frames(video_path)
    print(f"📸 Total frames extracted: {len(frames)}")

    if len(frames) == 0:
        print("❌ No frames found.")
        return

    # Step 2: Load models
    yolo_model = YOLO("yolov8n.pt")
    mp_pose = mp.solutions.pose
    pose_model = mp_pose.Pose(static_image_mode=True)

    # Step 3: Extract pose sequence
    sequence = extract_pose_sequence(frames, yolo_model, pose_model)
    X = np.array([sequence])
    # y = np.array([label])
    # os.mkdir(os.path.dirname(output_X_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_X_path), exist_ok=True)
    np.save(output_X_path, X)
    # np.save(output_y_path, y)
    print(f"✅ Saved features to: {output_X_path}")
    # print(f"✅ Saved labels to: {output_y_path}")
    # print(f"🧠 Shape: X={X.shape}, y={y.shape}")

    # OPTIONAL: Save annotated video
    output_video_path = output_X_path.replace(".npy", "_annotated.mp4")
    save_annotated_video(frames, output_video_path, yolo_model, pose_model)

def save_annotated_video(frames, output_path, yolo_model, pose_model):
    """Create a video showing YOLO box + pose landmarks overlaid on main person."""
    height, width = frames[0].shape[:2]
    fps = 25  # Adjust as needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        img = frame.copy()

        # 1. YOLO detection
        box = get_main_person_box(img, yolo_model)
        if box is not None:
            x, y, w, h = box
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "Athlete", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 2. MediaPipe pose
            cropped = crop_person(img, box)
            img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            results = pose_model.process(img_rgb)

            if results.pose_landmarks:
                # We need to map keypoints back to original image coordinates
                for lm in results.pose_landmarks.landmark:
                    cx = int(x1 + lm.x * w)
                    cy = int(y1 + lm.y * h)
                    cv2.circle(img, (cx, cy), 3, (255, 0, 0), -1)

        out.write(img)

    out.release()
    print(f"🎞️ Output video saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    video_path = r"D:\\bha\\app\\mybacked\\input.mp4"
    output_X_path = r"D:\\bha\\app\\mybacked\\output\\x.npy"
    # output_y_path = r"D:\\bha\\app\\mybacked\\output\\y.npy"

    # process_video_to_pose_npy(video_path, output_X_path, output_y_path)
    process_video_to_pose_npy(video_path, output_X_path)

