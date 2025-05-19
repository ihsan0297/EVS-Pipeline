from load_model import load_model
from inference import perform_inference
from TimmRandAugmentTransform import *
from pathlib import Path

########################### Model Load #############################################
MODEL_PATH = Path(r"c:\Users\Ali\Desktop\Models\SeamMiniV1.pkl")
model = load_model(str(MODEL_PATH))

############################## Inferences ###########################################
IMAGE_FOLDER_PATH = Path(r"c:\Users\Ali\Desktop\airflow_test_docker\images")  # Path to the folder with images

def run_inference_on_folder(folder_path):  # ------> inference starts over there and img pass in iter
    if not folder_path.exists():
        print(f"Error: The folder {folder_path} does not exist.")
        return

    # Iterate through all image files in the folder
    for image_file in folder_path.glob("*"):
        # Check if the file is an image
        if image_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            try:
                print(f"Processing: {image_file.name}")
                # Perform inference on the image
                results = perform_inference(model, str(image_file))

                # Print the results
                print(f"Prediction: {results['prediction']}")
                print(f"Confidence: {results['confidence']:.4f}")
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
        else:
            print(f"Skipping non-image file: {image_file.name}")

# Run inference on the folder
run_inference_on_folder(IMAGE_FOLDER_PATH)

