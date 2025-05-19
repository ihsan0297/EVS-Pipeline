import pathlib
from fastai.vision.all import load_learner
from pathlib import Path
import torch

# Redirect PosixPath to WindowsPath globally
pathlib.PosixPath = pathlib.WindowsPath

def load_model(model_path: str):
    try:
        # Load the model
        model = load_learner(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        model.to(device)
        print("Model loaded successfully!")
        print(f"Categories: {model.dls.vocab}")  # Print categories
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at '{model_path}'.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the model: {e}")


if __name__=="__main__":
    load_model("Models/Seamvit.pkl")