from load_model import load_model
from inference import perform_inference
from TimmRandAugmentTransform import *
from pathlib import Path

########################### Model Load #############################################
MODEL_PATH = Path(r"c:\Users\Ali\Desktop\Models\SeamMiniV1.pkl")
model = load_model(str(MODEL_PATH))

############################## Inferences ###########################################
IMAGE_FOLDER_PATH = Path(r"c:\Users\Ali\Desktop\Models\Vehicles\images")  # Path to the folder with images

# Batch processing function
def batch_images(images, batch_size):
    """Divides the images list into smaller batches."""
    return [images[i:i + batch_size] for i in range(0, len(images), batch_size)]

def run_inference_on_folder(folder_path, batch_size=32):  
    if not folder_path.exists():
        print(f"Error: The folder {folder_path} does not exist.")
        return

    # Get a list of all image files in the folder
    image_files = list(folder_path.glob("*"))
    
    # Filter only image files
    image_files = [image_file for image_file in image_files if image_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]]

    if not image_files:
        print("No image files found in the folder.")
        return

    print(f"Total images to process: {len(image_files)}")
    
    # Divide the image files into batches
    batches = batch_images(image_files, batch_size)
    
    # Iterate over each batch
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)} with {len(batch)} images...")
        
        # Process all images in the current batch
        for image_file in batch:
            try:
                print(f"Processing: {image_file.name}")
                # Perform inference on the image
                results = perform_inference(model, str(image_file))

                # Print the results
                print(f"Prediction: {results['prediction']}")
                print(f"Confidence: {results['confidence']:.4f}")
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
        
        print(f"Finished processing batch {i+1}/{len(batches)}")

# Run inference on the folder in batches
run_inference_on_folder(IMAGE_FOLDER_PATH, batch_size=32)
