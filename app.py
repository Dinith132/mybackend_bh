from flask import Flask, request, send_file, jsonify
import os
import uuid
import tempfile
from numpy_pipeline import process_video_to_pose_npy  # Your existing function
import shutil
import traceback
import shutil
from flask import after_this_request



app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload_video():
    if 'video' not in request.files:
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
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
