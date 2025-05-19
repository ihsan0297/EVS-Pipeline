from load_model import load_model
from inference import perform_inference
from TimmRandAugmentTransform import *
from pathlib import Path

###########################Model Load #############################################
MODEL_PATH = Path(r"C:\Users\Ali\Desktop\demo_pipeline\models\SeamMiniV1.pkl")
model = load_model(str(MODEL_PATH))

##############################Inferences##############################################
IMAGE_PATH = Path(r"C:\Users\Ali\Desktop\demo_pipeline\output\normal\111111.PNG")  # Replace with your image path


# inference model function

def run_inference(image_path): # -----> pass image to the model
    try:
        # Perform inference on the image
        results = perform_inference(model, str(image_path)) # -----> function performs here 

        # Print the results
        print(f"Prediction: {results['prediction']}")
        print(f"Confidence: {results['confidence']:.4f}")
    except Exception as e:
        print(f"Error: {e}")

run_inference(IMAGE_PATH)
