from flask import Flask, request, send_file, jsonify, after_this_request
import os
import uuid
import tempfile
import shutil
import traceback
from numpy_pipeline import process_video_to_pose_npy  # Your existing function
from flask_cors import CORS  # Optional, for frontend access from a different origin

app = Flask(__name__)
CORS(app)  # Optional: Remove if you don’t need cross-origin support

@app.route("/upload", methods=["POST"])
def upload_video():
    if 'video' not in request.files:
        print("❌ No 'video' in request.files")
        print("Files:", request.files)
        print("Form:", request.form)
        return jsonify({"error": "No video part"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected video"}), 400

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mp4")
            temp_output_path = os.path.join(tmpdir, "output.npy")

            video_file.save(input_path)
            process_video_to_pose_npy(input_path, temp_output_path)

            # Copy to a new temp file outside the auto-deleting dir
            stable_output = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
            stable_output.close()  # Close it so we can write to it

            shutil.copy2(temp_output_path, stable_output.name)

            @after_this_request
            def cleanup(response):
                try:
                    os.remove(stable_output.name)
                except Exception as e:
                    print(f"⚠️ Cleanup failed: {e}")
                return response

            return send_file(stable_output.name, as_attachment=True, download_name="pose_features.npy")

    except Exception as e:
        print("❌ Error during processing:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Accept connections on your LAN IP (e.g., 192.168.x.x)
    app.run(host="0.0.0.0", port=5000, debug=True)
