# import numpy as np
# import tensorflow as tf
# import joblib
# import cv2
# import mediapipe as mp
# from ultralytics import YOLO
# import math
# import os

# # === Paths ===
# VIDEO_PATH = "D:\\bha\\app\\final_backed_dep\\neext\\input.mp4"
# SCALER_PATH = "D:\\bha\\app\\final_backed_dep\\neext\\scaler.save"
# MODEL_PATH = "D:\\bha\\app\\final_backed_dep\\neext\\final_model_attention.keras"

# # === Load model and scaler ===
# model = tf.keras.models.load_model(MODEL_PATH)
# scaler = joblib.load(SCALER_PATH)

# # === Load YOLO and MediaPipe ===
# yolo = YOLO("yolov8n.pt")
# mp_pose = mp.solutions.pose
# # pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# # === Utility: Calculate angle between 3 points ===
# def calculate_angle(a, b, c):
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     ba = a - b
#     bc = c - b
#     cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
#     angle = np.arccos(np.clip(cosine, -1.0, 1.0))
#     return np.degrees(angle)

# # === Utility: Extract 6 joint angles ===
# def extract_joint_angles(landmarks):
#     joints = []
#     # Right Elbow
#     joints.append(calculate_angle(landmarks[12][:3], landmarks[14][:3], landmarks[16][:3]))
#     # Left Elbow
#     joints.append(calculate_angle(landmarks[11][:3], landmarks[13][:3], landmarks[15][:3]))
#     # Right Shoulder
#     joints.append(calculate_angle(landmarks[24][:3], landmarks[12][:3], landmarks[14][:3]))
#     # Left Shoulder
#     joints.append(calculate_angle(landmarks[23][:3], landmarks[11][:3], landmarks[13][:3]))
#     # Right Knee
#     joints.append(calculate_angle(landmarks[24][:3], landmarks[26][:3], landmarks[28][:3]))
#     # Left Knee
#     joints.append(calculate_angle(landmarks[23][:3], landmarks[25][:3], landmarks[27][:3]))
#     return np.array(joints)


# # def extract_features_from_video(video_path, max_frames=10):
# #     print("Extracting Features...")
# #     cap = cv2.VideoCapture(video_path)
# #     print("cv2 done")
# #     features = []
# #     frame_count = 0

# #     with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as pose:
# #         while cap.isOpened() and frame_count < max_frames:
# #             print(f"processing frame {frame_count+1} ...")
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break

# #             results = yolo(frame, verbose=False)
# #             if isinstance(results, list) and len(results) > 0:
# #                 result = results[0]
# #                 if hasattr(result, 'boxes') and result.boxes.xyxy.shape[0] > 0:
# #                     x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy()[0].astype(int)
# #                     cropped = frame[y1:y2, x1:x2]
# #                     rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
# #                     results_pose = pose.process(rgb)

# #                     if results_pose.pose_landmarks:
# #                         landmarks = np.array(results_pose.pose_landmarks.landmark)
# #                         landmark_array = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])
# #                         joint_angles = extract_joint_angles(landmark_array)  # shape (6,)
# #                         pose_feats = landmark_array.flatten()  # shape (132,)
# #                         full_feature = np.concatenate([pose_feats, joint_angles])  # 138
# #                         features.append(full_feature)
# #                         frame_count += 1

# #     cap.release()

# #     while len(features) < 10:
# #         features.append(np.zeros(138))

# #     return np.array(features)[:10]

# def extract_features_from_video(video_path, max_frames=10):
#     print("Extracting Features...")
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"Total frames in video: {total_frames}")

#     if total_frames == 0:
#         print("Video has no frames.")
#         return np.zeros((max_frames, 138))

#     # Compute evenly spaced frame indices
#     frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

#     features = []

#     with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as pose:
#         for idx in frame_indices:
#             print(f"Processing frame {idx} ...")
#             cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Jump to that frame
#             ret, frame = cap.read()
#             if not ret:
#                 features.append(np.zeros(138))
#                 continue

#             results = yolo(frame, verbose=False)
#             if isinstance(results, list) and len(results) > 0:
#                 result = results[0]
#                 if hasattr(result, 'boxes') and result.boxes.xyxy.shape[0] > 0:
#                     x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy()[0].astype(int)
#                     cropped = frame[y1:y2, x1:x2]
#                     rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
#                     results_pose = pose.process(rgb)

#                     if results_pose.pose_landmarks:
#                         landmarks = np.array(results_pose.pose_landmarks.landmark)
#                         landmark_array = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])
#                         joint_angles = extract_joint_angles(landmark_array)  # shape (6,)
#                         pose_feats = landmark_array.flatten()  # shape (132,)
#                         full_feature = np.concatenate([pose_feats, joint_angles])  # 138
#                         features.append(full_feature)
#                         continue

#             # If no detection, append zeros
#             features.append(np.zeros(138))

#     cap.release()
#     return np.array(features)


# # === Step 2: Final tensor prep ===
# def prepare_input_tensor(frames_10):
#     if frames_10.shape[0] != 10:
#         raise ValueError(f"Need 10 frames, got {frames_10.shape[0]}")

#     velocity = np.gradient(frames_10[:, :99], axis=0)  # 99 = x,y,z
#     final_features = np.concatenate([frames_10, velocity], axis=-1)  # (10, 237)
#     final_scaled = scaler.transform(final_features)  # (10, 237)
#     print("prepared input tensor")
#     return np.expand_dims(final_scaled, axis=0)  # (1, 10, 237)

# # === Step 3: Predict ===
# # === New: Save frames with YOLO + Pose landmarks ===



# # def save_frames_with_pose(video_path, output_dir="D:\\bha\\app\\final_backed_dep\\output", max_frames=50):
# #     """
# #     Save the frames (cropped + pose drawn) that are actually used for prediction.
# #     They will be distributed across the whole video, same as in extract_features_from_video().
# #     """
# #     print("Saving frames with YOLO + Pose...")
# #     os.makedirs(output_dir, exist_ok=True)

# #     cap = cv2.VideoCapture(video_path)
# #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# #     print(f"Total frames in video: {total_frames}")

# #     if total_frames == 0:
# #         print("Video has no frames.")
# #         return

# #     # Compute evenly spaced frame indices
# #     frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

# #     with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as pose:
# #         for i, idx in enumerate(frame_indices):
# #             cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # jump to target frame
# #             ret, frame = cap.read()
# #             if not ret:
# #                 print(f"⚠️ Could not read frame {idx}")
# #                 continue

# #             results = yolo(frame, verbose=False)
# #             if isinstance(results, list) and len(results) > 0:
# #                 result = results[0]
# #                 if hasattr(result, 'boxes') and result.boxes.xyxy.shape[0] > 0:
# #                     x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy()[0].astype(int)
# #                     cropped = frame[y1:y2, x1:x2]
# #                     rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
# #                     results_pose = pose.process(rgb)

# #                     # Draw pose landmarks if detected
# #                     if results_pose.pose_landmarks:
# #                         mp.solutions.drawing_utils.draw_landmarks(
# #                             cropped,
# #                             results_pose.pose_landmarks,
# #                             mp_pose.POSE_CONNECTIONS,
# #                             landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
# #                         )

# #                     # Save the processed frame
# #                     save_path = os.path.join(output_dir, f"frame_{i+1}.jpg")
# #                     cv2.imwrite(save_path, cropped)
# #                     print(f"✅ Saved {save_path}")
# #             else:
# #                 print(f"⚠️ No detection in frame {idx}")

# #     cap.release()
# #     print("All selected frames saved.")


# # def save_frames_with_pose(video_path, output_path="D:\\bha\\app\\final_backed_dep\\output\\out.mp4"):
# #     """
# #     Process the whole video:
# #     - Detect main person with YOLO
# #     - Draw bounding box + pose landmarks
# #     - Save as a full video
# #     """
# #     cap = cv2.VideoCapture(video_path)
# #     if not cap.isOpened():
# #         print("❌ Could not open video")
# #         return

# #     # Video properties
# #     fps = 25
# #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# #     # Output video writer
# #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# #     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# #     with mp_pose.Pose(
# #         static_image_mode=False,
# #         model_complexity=1,
# #         min_detection_confidence=0.5
# #     ) as pose:

# #         frame_count = 0
# #         while True:
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break
# #             frame_count += 1

# #             # --- YOLO detection ---
# #             results = yolo(frame, verbose=False)
# #             if isinstance(results, list) and len(results) > 0:
# #                 result = results[0]
# #                 if hasattr(result, "boxes") and result.boxes.xyxy.shape[0] > 0:
# #                     x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy()[0].astype(int)
# #                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #                     cv2.putText(frame, "Athlete", (x1, y1 - 10),
# #                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# #                     # --- Pose estimation on cropped person ---
# #                     cropped = frame[y1:y2, x1:x2]
# #                     if cropped.size > 0:  # avoid empty crop
# #                         rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
# #                         results_pose = pose.process(rgb)

# #                         if results_pose.pose_landmarks:
# #                             for lm in results_pose.pose_landmarks.landmark:
# #                                 cx = int(x1 + lm.x * (x2 - x1))
# #                                 cy = int(y1 + lm.y * (y2 - y1))
# #                                 cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

# #             out.write(frame)
# #             if frame_count % 50 == 0:
# #                 print(f"Processed {frame_count} frames...")

# #     cap.release()
# #     out.release()
# #     print(f"🎥 Pose video saved at {output_path}")

# import cv2
# import mediapipe as mp
# import numpy as np

# mp_pose = mp.solutions.pose

# def save_frames_with_pose(video_path,file_id, output_path="D:\\bha\\app\\final_backed_dep\\output", num_frames=50):
#     """
#     Sample `num_frames` evenly spaced frames from the video,
#     detect main person with YOLO, overlay pose landmarks,
#     and save as a video.
#     """
#     output_path=f"{output_path}\\out{file_id}.mp4"

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("❌ Could not open video")
#         return

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = 25  # output FPS (can adjust)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Adjust num_frames if video is too short
#     if total_frames < num_frames:
#         num_frames = total_frames

#     # Compute evenly spaced frame indices
#     frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

#     # Video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     with mp_pose.Pose(
#         static_image_mode=False,
#         model_complexity=1,
#         min_detection_confidence=0.5
#     ) as pose:

#         for i, idx in enumerate(frame_indices):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#             ret, frame = cap.read()
#             if not ret:
#                 print(f"⚠️ Could not read frame {idx}")
#                 continue

#             # --- YOLO detection ---
#             results = yolo(frame, verbose=False)
#             if isinstance(results, list) and len(results) > 0:
#                 result = results[0]
#                 if hasattr(result, "boxes") and result.boxes.xyxy.shape[0] > 0:
#                     x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy()[0].astype(int)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(frame, "Athlete", (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#                     # --- Pose estimation ---
#                     cropped = frame[y1:y2, x1:x2]
#                     if cropped.size > 0:
#                         rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
#                         results_pose = pose.process(rgb)
#                         if results_pose.pose_landmarks:
#                             for lm in results_pose.pose_landmarks.landmark:
#                                 cx = int(x1 + lm.x * (x2 - x1))
#                                 cy = int(y1 + lm.y * (y2 - y1))
#                                 cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

#             out.write(frame)
#             print(f"✅ Saved frame {i+1}/{num_frames} to video")

#     cap.release()
#     out.release()
#     print(f"🎥 Pose video saved at {output_path}")


# # Example usage:
# # save_frames_with_pose(VIDEO_PATH, output_dir="debug_frames", max_frames=10)




# def final_prediction(video_path, file_id):
#     frames = extract_features_from_video(video_path)
#     input_tensor = prepare_input_tensor(frames)
#     print("Getting the prediction...")
#     pred = model.predict(input_tensor)
#     print("Prediction Done.")

#     save_frames_with_pose(video_path, file_id)

#     # === Display Result ===
#     class_names = ['Good Technique', 'Low Arm', 'Poor Left Leg Block', 'Both Errors']
#     predicted_class = np.argmax(pred)
#     confidence = np.max(pred)

#     print("\n🔍 Class-wise Probabilities:")
#     for i, prob in enumerate(pred[0]):
#         print(f"  {class_names[i]}: {prob:.3f}")

#     print(f"\n✅ Predicted Class: {class_names[predicted_class]} | Confidence: {confidence:.4f}")

#     ndigits = 2  # number of decimals to round to

#     result = {
#         "prediction": class_names[predicted_class],
#         "confidence": round(float(confidence), ndigits),
#         "probabilities": {
#             "Good Technique": round(float(pred[0][0]), ndigits),
#             "Low Arm": round(float(pred[0][1]), ndigits),
#             "Poor Left Leg Block": round(float(pred[0][2]), ndigits),
#             "Both Errors": round(float(pred[0][3]), ndigits)
#         }
#     }

#     print(result)
#     return result


import os
import cv2
import numpy as np
import tensorflow as tf
import joblib
import mediapipe as mp
from ultralytics import YOLO

# === Base Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(BASE_DIR, "input.mp4")   # Default video input
SCALER_PATH = os.path.join(BASE_DIR, "scaler.save")
MODEL_PATH = os.path.join(BASE_DIR, "final_model_attention.keras")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load model and scaler ===
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# === Load YOLO and MediaPipe ===
yolo = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose


def save_frames_with_pose(video_path, file_id, output_dir=OUTPUT_DIR, num_frames=50):
    """Extract frames with pose keypoints and save annotated video"""
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = os.path.join(output_dir, f"out{file_id}.mp4")
    out = None
    frame_count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            if out is None:
                h, w = frame.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))

            out.write(frame)
            frame_count += 1

    cap.release()
    if out:
        out.release()

    return output_path


def preprocess_keypoints(keypoints, scaler):
    """Normalize pose keypoints with saved scaler"""
    keypoints = np.array(keypoints).reshape(1, -1)
    return scaler.transform(keypoints)


def predict_pose(keypoints):
    """Run keypoints through trained model"""
    processed = preprocess_keypoints(keypoints, scaler)
    preds = model.predict(processed)
    return np.argmax(preds, axis=1)[0]
