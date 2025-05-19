import pathlib
from fastai.vision.all import load_learner
from pathlib import Path

# Redirect PosixPath to WindowsPath globally
pathlib.PosixPath = pathlib.WindowsPath

def load_model(model_path: str):
    try:
        # Load the model
        model = load_learner(model_path)
        print("Model loaded successfully!")
        print(f"Categories: {model.dls.vocab}")  # Print categories
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at '{model_path}'.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the model: {e}")
