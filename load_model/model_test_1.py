# load_model_script 

import torch
import timm
from pathlib import Path
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

def load_pretrained_model(model_name='resnet50', pretrained=True):
    """
    Load a pre-trained model from timm library.
    """
    model = timm.create_model(model_name, pretrained=pretrained)
    return model

def load_model(model_path):
    """
    Load a PyTorch model from a given path.
    """
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode
    return model

if __name__ == "__main__":
    # Paths to models in Google Drive
    MODEL_PATH_1 = Path("/content/drive/MyDrive/fastai_vision_models/SeamMiniV1.pkl")
    MODEL_PATH_2 = Path("/content/drive/MyDrive/fastai_vision_models/SeamV1.pkl")

    # Load models
    model_1 = load_model(str(MODEL_PATH_1))
    model_2 = load_model(str(MODEL_PATH_2))

    print("Model 1 loaded:", model_1)
    print("Model 2 loaded:", model_2)


    # inference.py script 

    import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

def preprocess_image(image_path):
    """
    Preprocess the image for inference.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def infer(model, image_tensor):
    """
    Perform inference on the input image tensor.
    """
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    return output

if __name__ == "__main__":
    # Path to test images in Google Drive
    IMAGE_FOLDER = Path("/content/drive/MyDrive/images")

    # Load a sample image
    image_path = list(IMAGE_FOLDER.glob("*.jpg"))[0]  # Get the first image in the folder
    image_tensor = preprocess_image(image_path)

    # Load the model (replace with your model path)
    model = torch.load("/content/drive/MyDrive/fastai_vision_models/SeamMiniV1.pkl")

    # Perform inference
    output = infer(model, image_tensor)
    print("Inference Output:", output)


    # full script.py

    from load_model import load_pretrained_model, load_model
from inference import preprocess_image, infer
from TimmRandAugmentTransform import get_rand_augment_transform
from pathlib import Path
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

def main():
    # Paths to models in Google Drive
    MODEL_PATH_1 = Path("/content/drive/MyDrive/fastai_vision_models/SeamMiniV1.pkl")
    MODEL_PATH_2 = Path("/content/drive/MyDrive/fastai_vision_models/SeamV1.pkl")

    # Load models
    model_1 = load_model(str(MODEL_PATH_1))
    model_2 = load_model(str(MODEL_PATH_2))

    # Path to test images in Google Drive
    IMAGE_FOLDER = Path("/content/drive/MyDrive/images")

    # Load a sample image
    image_path = list(IMAGE_FOLDER.glob("*.jpg"))[0]  # Get the first image in the folder
    image_tensor = preprocess_image(image_path)

    # Perform inference with both models
    output_1 = infer(model_1, image_tensor)
    output_2 = infer(model_2, image_tensor)

    print("Inference Output from Model 1:", output_1)
    print("Inference Output from Model 2:", output_2)

    # Apply RandAugment transform
    transform = get_rand_augment_transform()
    augmented_image = transform(Image.open(image_path).convert('RGB'))
    print("Augmented Image Tensor:", augmented_image)

if __name__ == "__main__":
    main()