from flask import Flask, request, jsonify, send_from_directory
import os
from detect import detect_objects
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Allows cross-origin requests

UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

@app.route("/detect", methods=["POST"])
def upload_file():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Process Image with YOLO
    processed_path = detect_objects(filepath, filename)

    return jsonify({"processed_image_url": f"/processed/{filename}"}), 200

@app.route("/processed/<filename>")
def get_processed_image(filename):
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
