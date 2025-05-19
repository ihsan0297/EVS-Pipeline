import os
import shutil

# === CONFIGURE THESE ===
input_txt_file = 'c:\\Users\\DELL\\Desktop\\OriginalSeam\\Seams.txt'  # Path to your .txt file
output_directory = 'c:\\Users\\DELL\\Desktop\\OriginalSeam\\v1'         # Output folder to copy images into

# ========================

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Read image paths from the file
with open(input_txt_file, 'r') as file:
    image_paths = [line.strip() for line in file if line.strip()]

# Copy images to the output directory
for image_path in image_paths:
    if os.path.isfile(image_path):
        try:
            shutil.copy(image_path, output_directory)
            print(f"Copied: {image_path}")
        except Exception as e:
            print(f"Error copying {image_path}: {e}")
    else:
        print(f"File not found: {image_path}")
