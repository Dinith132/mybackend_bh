# import shutil
# from flask import Flask, request, jsonify
# import os
# import uuid
# import tempfile
# import traceback
# from numpy_pipeline import process_video_to_pose_npy  # Your existing function
# from flask_cors import CORS
# import threading
# import neext.dsd as get_prediction
# from flask import send_file


# app = Flask(__name__)
# CORS(app)

# # Store for tracking processed files and their status
# processed_files = {}

# def process_video_async(file_id, input_path):
#     try:
#         # Simulate or modify process_video_to_pose_npy to return 4 values
#         # Replace this with your actual processing logic

#         # process_video_to_pose_npy(input_path)  # Assume this returns [prediction, confidence, error_type, analysis_score]
        
#         result= get_prediction.final_prediction(input_path, file_id)
#         # print(array)
#         # result={
#         #     "prediction": "Good Technique",
#         #     "confidence": 0.9,
#         #     "probabilities": {
#         #         "Good Technique": 0.9,
#         #         "Low Arm": 0.05,
#         #         "Poor Left Leg Block": 0.02,
#         #         "Both Errors": 0.03
#         #     }
#         # }
#         processed_files[file_id]['status'] = 'completed'
#         processed_files[file_id]['result'] = result
#     except Exception as e:
#         processed_files[file_id]['status'] = 'failed'
#         processed_files[file_id]['error'] = str(e)
#         print(f"❌ Error processing video {file_id}: {str(e)}")
#         traceback.print_exc()
#     finally:
#         # Clean up input file
#         try:
#             os.remove(input_path)
#         except Exception as e:
#             print(f"⚠️ Cleanup failed: {e}")

# @app.route("/upload", methods=["POST"])
# def upload_video():
#     if 'video' not in request.files:
#         print("❌ No 'video' in request.files")
#         print("Files:", request.files)
#         print("Form:", request.form)
#         return jsonify({"error": "No video part"}), 400

#     video_file = request.files['video']
#     if video_file.filename == '':
#         return jsonify({"error": "No selected video"}), 400

#     try:
#         file_id = str(uuid.uuid4())
#         temp_dir = tempfile.mkdtemp()
#         input_path = os.path.join(temp_dir, "input.mp4")

#         # Save the video file
#         video_file.save(input_path)

#         # Store initial status
#         processed_files[file_id] = {
#             'status': 'processing',
#             'temp_dir': temp_dir,
#             'result': None,
#             'error': None
#         }

#         # Start background processing
#         threading.Thread(
#             target=process_video_async,
#             args=(file_id, input_path),
#             daemon=True
#         ).start()

#         return jsonify({
#             "message": "Video uploaded successfully",
#             "file_id": file_id,
#             "status_url": f"/status/{file_id}"
#         }), 202

#     except Exception as e:
#         print("❌ Error during upload:")
#         traceback.print_exc()
#         return jsonify({"error": f"Upload failed: {str(e)}"}), 500

# @app.route("/status/<file_id>", methods=["GET"])
# def check_status(file_id):
#     if file_id not in processed_files:
#         return jsonify({"error": "File not found or expired"}), 404

#     status_info = processed_files[file_id]
    
#     if status_info['status'] == 'processing':
#         return jsonify({
#             "message": "Video is still processing",
#             "status": "processing",
#             "file_id": file_id
#         }), 200
#     elif status_info['status'] == 'completed':
#         print("=======================================completed=====================")
#         result = status_info['result']
#         try:
#             shutil.rmtree(status_info['temp_dir'])
#             del processed_files[file_id]
#         except Exception as e:
#             print(f"⚠️ Cleanup failed: {e}")
#         return jsonify({
#             "message": "Video processed successfully",
#             "result": result,
#             "video_url": f"/download/{file_id}",
#             "status":"completed"
#         }), 200
#     else:  # failed
#         error = status_info['error']
#         try:
#             shutil.rmtree(status_info['temp_dir'])
#             del processed_files[file_id]
#         except Exception as e:
#             print(f"⚠️ Cleanup failed: {e}")
#         return jsonify({"error": f"Processing failed: {error}"}), 500


# @app.route("/download/<file_id>", methods=["GET"])
# def download_video(file_id):
#     video_path = os.path.join("output", f"out{file_id}.mp4")
#     if os.path.exists(video_path):
#         print("=======send the file======")
#         return send_file(video_path, as_attachment=True)
#     else:
#         return jsonify({"error": "Video not found"}), 404



# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)


import os
import uuid
import tempfile
import shutil
import threading
import traceback

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Import your custom modules
from numpy_pipeline import process_video_to_pose_npy
import neext.dsd as get_prediction

# --- Flask App Setup ---
app = Flask(__name__)
# Restrict CORS in production; for now, allow all for testing
CORS(app)

# --- Global store for tracking video processing ---
processed_files = {}

# Ensure 'output' folder exists and is writable
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Video Processing Thread ---
def process_video_async(file_id, input_path):
    try:
        # Call your prediction function
        result = get_prediction.final_prediction(input_path, file_id)

        processed_files[file_id]['status'] = 'completed'
        processed_files[file_id]['result'] = result

    except Exception as e:
        processed_files[file_id]['status'] = 'failed'
        processed_files[file_id]['error'] = str(e)
        print(f"❌ Error processing video {file_id}: {e}")
        traceback.print_exc()
    finally:
        # Cleanup input file
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")


# --- Routes ---
@app.route("/upload", methods=["POST"])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video part"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected video"}), 400

    try:
        file_id = str(uuid.uuid4())
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, "input.mp4")

        print("=====================================")
        print("=====================================")
        print(input_path)
        print("=====================================")
        print("=====================================")

        # Save uploaded file
        video_file.save(input_path)

        # Store initial status
        processed_files[file_id] = {
            'status': 'processing',
            'temp_dir': temp_dir,
            'result': None,
            'error': None
        }

        # Start background processing
        threading.Thread(
            target=process_video_async,
            args=(file_id, input_path),
            daemon=True
        ).start()

        return jsonify({
            "message": "Video uploaded successfully",
            "file_id": file_id,
            "status_url": f"/status/{file_id}"
        }), 202

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Upload failed: {e}"}), 500


@app.route("/status/<file_id>", methods=["GET"])
def check_status(file_id):
    if file_id not in processed_files:
        return jsonify({"error": "File not found or expired"}), 404

    status_info = processed_files[file_id]

    if status_info['status'] == 'processing':
        return jsonify({
            "message": "Video is still processing",
            "status": "processing",
            "file_id": file_id
        }), 200

    elif status_info['status'] == 'completed':
        result = status_info['result']

        # Clean up temp dir
        try:
            shutil.rmtree(status_info['temp_dir'])
            del processed_files[file_id]
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")

        return jsonify({
            "message": "Video processed successfully",
            "result": result,
            "video_url": f"/download/{file_id}",
            "status": "completed"
        }), 200

    else:  # failed
        error = status_info['error']
        try:
            shutil.rmtree(status_info['temp_dir'])
            del processed_files[file_id]
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")
        return jsonify({"error": f"Processing failed: {error}"}), 500


@app.route("/download/<file_id>", methods=["GET"])
def download_video(file_id):
    video_path = os.path.join(OUTPUT_DIR, f"out{file_id}.mp4")
    if os.path.exists(video_path):
        return send_file(video_path, as_attachment=True)
    else:
        return jsonify({"error": "Video not found"}), 404


@app.route("/check", methods=["GET"])
def check_out():
    return jsonify({"ok": "Good maorning ....................."}), 200



# if __name__ == "__main__":
#     # For development only. For production, use Gunicorn:
#     # gunicorn -w 4 -b 0.0.0.0:5000 app:app
#     app.run(host="0.0.0.0", port=5000, debug=True)
