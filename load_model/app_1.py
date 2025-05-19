from flask import Flask, render_template, request, jsonify
from pathlib import Path
import os
from load_model import load_model
from inference import perform_inference
from TimmRandAugmentTransform import TimmRandAugmentTransform

app = Flask(__name__)

# Model paths
MODEL_PATH_1 = Path(r"C:\Users\Ali\Desktop\demo_pipeline\models\SeamMiniV1.pkl")
MODEL_PATH_2 = Path(r"C:\Users\Ali\Desktop\demo_pipeline\models\SeamV1.pkl")

# Load models
print("Loading models...")
model_1 = load_model(str(MODEL_PATH_1))
model_2 = load_model(str(MODEL_PATH_2))
print("Models loaded successfully!")

# Route for UI
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get selected model
        selected_model = request.form.get("model")

        # Get uploaded file
        uploaded_file = request.files.get("file")

        if not uploaded_file:
            return jsonify({"error": "No file uploaded"}), 400

        # Save the uploaded file to a temporary location
        upload_folder = Path("uploads")
        upload_folder.mkdir(exist_ok=True)
        file_path = upload_folder / uploaded_file.filename
        uploaded_file.save(file_path)

        # Check if it's an image or a folder
        results = []
        if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            # Single image inference
            model = model_1 if selected_model == "SeamMiniV1" else model_2
            result = perform_inference(model, str(file_path))
            results.append({
                "image_name": file_path.name,
                "prediction": result["prediction"],
                "confidence": float(result["confidence"])
            })
        elif file_path.is_dir():
            # Folder inference
            model = model_1 if selected_model == "SeamMiniV1" else model_2
            image_files = [f for f in file_path.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]]
            for image_file in image_files:
                try:
                    result = perform_inference(model, str(image_file))
                    results.append({
                        "image_name": image_file.name,
                        "prediction": result["prediction"],
                        "confidence": float(result["confidence"])
                    })
                except Exception as e:
                    results.append({
                        "image_name": image_file.name,
                        "error": str(e)
                    })

        return render_template("results.html", results=results)

    return render_template("index.html")

# Templates
@app.route("/results", methods=["GET"])
def results():
    return render_template("./index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
