import time
import matplotlib.pyplot as plt
import seaborn as sns
from load_model import load_model
from inference import perform_inference
from pathlib import Path

########################### Model Load #############################################
MODEL_PATH_1 = Path(r"c:\Users\Ali\Desktop\Models\SeamMiniV1.pkl")
MODEL_PATH_2 = Path(r"c:\Users\Ali\Desktop\Models\SeamV1.pkl")

# Load both models
model_1 = load_model(str(MODEL_PATH_1))
model_2 = load_model(str(MODEL_PATH_2))

############################## Inferences ###########################################
IMAGE_FOLDER_PATH = Path(r"c:\Users\Ali\Desktop\Models\image_dataset_diverse\images")  # Path to the folder with images

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
    
    # Lists to store predictions, confidences, and latencies
    predictions_1, predictions_2 = [], []
    confidences_1, confidences_2 = [], []
    latencies = []

    # Divide the image files into batches
    batches = batch_images(image_files, batch_size)
    
    # Iterate over each batch
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)} with {len(batch)} images...")
        
        # Process all images in the current batch
        for image_file in batch:
            try:
                print(f"Processing: {image_file.name}")
                
                # Record the start time to track latency
                start_time = time.time()
                
                # Inference with model_1
                results_1 = perform_inference(model_1, str(image_file))
                predictions_1.append(results_1['prediction'])
                confidences_1.append(results_1['confidence'])
                
                # Inference with model_2
                results_2 = perform_inference(model_2, str(image_file))
                predictions_2.append(results_2['prediction'])
                confidences_2.append(results_2['confidence'])
                
                # Record the end time and calculate latency
                end_time = time.time()
                latency = end_time - start_time
                latencies.append(latency)

                print(f"Model 1 - Prediction: {results_1['prediction']}, Confidence: {results_1['confidence']:.4f}")
                print(f"Model 2 - Prediction: {results_2['prediction']}, Confidence: {results_2['confidence']:.4f}")
                
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
        
        print(f"Finished processing batch {i+1}/{len(batches)}")
    
    # Plot Results after all batches are processed
    plot_model_performance(predictions_1, confidences_1, predictions_2, confidences_2, latencies)

def plot_model_performance(predictions_1, confidences_1, predictions_2, confidences_2, latencies):
    """Plot predictions, confidences, latency, and additional visualizations."""
    
    # Plot predictions and confidence scores for both models
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.bar(range(len(predictions_1)), confidences_1, color='blue', alpha=0.6, label='Model 1 Confidence')
    ax1.bar(range(len(predictions_2)), confidences_2, color='green', alpha=0.6, label='Model 2 Confidence')
    ax1.set_title('Model Confidence Comparison')
    ax1.set_xlabel('Image Index')
    ax1.set_ylabel('Confidence')
    ax1.legend()

    # Plot Latency
    ax2.plot(latencies, color='red', marker='o', linestyle='-', label='Latency per Image')
    ax2.set_title('Latency Time per Image')
    ax2.set_xlabel('Image Index')
    ax2.set_ylabel('Latency (seconds)')
    ax2.legend()

    # Scatter Plot: Confidence vs. Latency
    plt.figure(figsize=(8, 6))
    plt.scatter(confidences_1, latencies, color='blue', alpha=0.5, label='Model 1')
    plt.scatter(confidences_2, latencies, color='green', alpha=0.5, label='Model 2')
    plt.title('Confidence vs. Latency')
    plt.xlabel('Confidence')
    plt.ylabel('Latency (seconds)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Histogram: Distribution of Confidence Scores
    plt.figure(figsize=(8, 6))
    plt.hist(confidences_1, bins=20, alpha=0.5, label='Model 1 Confidence', color='blue')
    plt.hist(confidences_2, bins=20, alpha=0.5, label='Model 2 Confidence', color='green')
    plt.title('Distribution of Confidence Scores')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Heatmap: Visualizing correlation (Confidence vs. Latency)
    plt.figure(figsize=(8, 6))
    data = {'Confidence': confidences_1 + confidences_2, 'Latency': latencies * 2}
    df = pd.DataFrame(data)
    correlation_matrix = df.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Heatmap: Confidence vs. Latency Correlation')
    plt.tight_layout()
    plt.show()

# Run inference on the folder in batches

run_inference_on_folder(IMAGE_FOLDER_PATH, batch_size=32)
