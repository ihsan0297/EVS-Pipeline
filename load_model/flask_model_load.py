from flask import Flask, request, jsonify
from pathlib import Path
import torch
from fastai.learner import load_learner
from TimmRandAugmentTransform import TimmRandAugmentTransform  # This import is crucial
from load_model import load_model
from inference import perform_inference

app = Flask(__name__)

# Model paths
MODEL_PATH_1 = Path(r"c:\Users\Ali\Desktop\Models\SeamMiniV1.pkl")
MODEL_PATH_2 = Path(r"c:\Users\Ali\Desktop\Models\SeamV1.pkl")

#model_artificats



# Load models at startup
try:
    print("Loading models...")
    model_1 = load_model(str(MODEL_PATH_1))
    model_2 = load_model(str(MODEL_PATH_2))
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

def process_folder(folder_path, batch_size=32):
    """Process all images in a folder and return results."""
    results = []
    folder_path = Path(folder_path)
    
    # Get all image files
    image_files = [f for f in folder_path.glob("*") 
                  if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]]
    
    for image_file in image_files:
        try:
            # Perform inference with both models
            result_1 = perform_inference(model_1, str(image_file))
            result_2 = perform_inference(model_2, str(image_file))
            
            results.append({
                "image_name": image_file.name,
                "model_1_results": {
                    "prediction": result_1["prediction"],
                    "confidence": float(result_1["confidence"])
                },
                "model_2_results": {
                    "prediction": result_2["prediction"],
                    "confidence": float(result_2["confidence"])
                }
            })
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
            results.append({
                "image_name": image_file.name,
                "error": str(e)
            })
    
    return results

@app.route('/predict', methods=['GET'])
def predict():
    """Endpoint to process images in a folder."""
    folder_path = request.args.get('folder_path')
    print(folder_path)
    if not folder_path:
        return jsonify({"error": "No folder path provided"}), 400
    
    folder_path = Path(folder_path)
    if not folder_path.exists():
        return jsonify({"error": f"Folder not found: {folder_path}"}), 404
    
    try:
        results = process_folder(folder_path)
        return jsonify({
            "status": "success",
            "results": results,
            "total_processed": len(results)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "healthy", "models_loaded": bool(model_1 and model_2)})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)