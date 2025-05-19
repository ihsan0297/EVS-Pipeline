import os
import json
import time
import threading
import asyncio
from pathlib import Path
import logging
import shutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from PIL import Image

# ---------------------------------------------------------------------
# Import your actual model loading and inference functions
# ---------------------------------------------------------------------
from load_model import load_model
from inference import perform_inference
from TimmRandAugmentTransform import *
FOLDER_PROCESS_STATE = {}  # Global dictionary to track folder states

# ---------------------------------------------------------------------
# Global Configuration and State
# ---------------------------------------------------------------------
MODEL_PATH = Path(r"D:\EVS\EVS-Seam-Version-1\Models\SeamV1.pkl")
model = load_model(str(MODEL_PATH))  # Load the model once globally
CURRENT_FOLDER = None
INFERENCE_THREAD = None
INFERENCE_THREAD_STOP = False
SENT_JSON_FILES = set()  # Track which JSON files have been sent to clients

app = FastAPI()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def encode_image_to_base64(image_path: Path) -> str:
    """Encode an image as Base64."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def resize_image_keep_aspect(img: Image.Image, tile_size: int) -> Image.Image:
    """Resize an image while maintaining its aspect ratio."""
    orig_width, orig_height = img.size
    if orig_height == tile_size:
        return img
    new_width = int(round((tile_size / orig_height) * orig_width))
    return img.resize((new_width, tile_size), Image.Resampling.LANCZOS)

def process_single_image(image_path: Path, tile_size: int) -> dict:
    """Process a single image: resize, tile, infer, classify."""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        logging.error(f"Failed to open image {image_path}: {e}")
        return {}

    resized_img = resize_image_keep_aspect(img, tile_size)
    new_width, new_height = resized_img.size
    number_of_tiles = new_width // tile_size

    tile_results = {}
    tile_predictions = []

    for i in range(number_of_tiles):
        left = i * tile_size
        tile = resized_img.crop((left, 0, left + tile_size, tile_size))

        inference_output = perform_inference(model, tile)
        pred = inference_output["prediction"]
        conf = float(f"{inference_output['confidence']:.4f}")

        tile_key = f"tile_{i+1}"
        tile_results[tile_key] = {"status": pred, "confidence": conf}
        tile_predictions.append(pred)

    final_class = "seam" if "seam" in map(str.lower, tile_predictions) else "normal"

    return {
        "image_title": image_path.name,
        "image_tiles": tile_results,
        "final_class": final_class
    }

def finalize_seam_json(folder_state, json_folder: Path, force_finalize=False):
    """Create seam JSON files for pending seams."""
    processed = folder_state["processed_images"]
    pending = folder_state["pending_seams"][:]

    for seam_index in pending:
        start_i = max(0, seam_index - 10)
        end_i = len(processed) if force_finalize else min(len(processed), seam_index + 11)

        seam_data = processed[seam_index]
        seam_json = {
            "images": processed[start_i:end_i]
        }
        seam_json_path = json_folder / f"{Path(seam_data['image_title']).stem}.json"
        with open(seam_json_path, "w") as jf:
            json.dump(seam_json, jf, indent=2)
        folder_state["pending_seams"].remove(seam_index)

def process_folder(folder_path: str, tile_size: int):
    """
    Process all unprocessed images in the folder and classify them into:
    - 'normal': Images classified as 'normal'.
    - 'seam': Images classified as 'seam' with more than 3 tiles.
    - 'seam1_to_3': Images classified as 'seam' with 1 to 3 tiles.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    folder = Path(folder_path)
    if not folder.is_dir():
        logging.error(f"Folder '{folder_path}' does not exist or is not a directory.")
        return

    # Supported image extensions
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    # Prepare subfolders for classification
    normal_folder = folder / "normal"
    seam_folder = folder / "seam"
    seam1_to_3_folder = folder / "seam1_to_3"
    json_folder = folder / "json"

    json_folder.mkdir(exist_ok=True)

    # Initialize folder state if not already done
    folder_state = FOLDER_PROCESS_STATE.setdefault(folder_path, {
        "processed_images": [],
        "pending_seams": []
    })
    processed_images = folder_state["processed_images"]

    # Process each image in the folder
    for image_path in sorted(folder.iterdir()):
        if image_path.suffix.lower() not in valid_extensions:
            logging.info(f"Skipping unsupported file: {image_path.name}")
            continue  # Skip unsupported file types

        if any(img["image_title"] == image_path.name for img in processed_images):
            continue  # Skip already processed images

        logging.info(f"Processing image: {image_path.name}")

        # Run inference on the image
        result = process_single_image(image_path, tile_size)
        if not result:
            continue

        # Append the result to processed_images
        processed_images.append(result)

        # Classify the image based on the result
        final_class = result["final_class"]
        number_of_tiles = len(result["image_tiles"])

        if final_class == "normal":
            target_folder = normal_folder
        elif final_class == "seam":
            if number_of_tiles <= 3:
                target_folder = seam1_to_3_folder
            else:
                target_folder = seam_folder
        else:
            logging.warning(f"Unknown classification for image {image_path.name}")
            continue

        # Move the image to the appropriate folder
        target_folder.mkdir(parents=True, exist_ok=True)
        destination = target_folder / image_path.name
        try:
            shutil.move(str(image_path), str(destination))
            logging.info(f"Moved '{image_path.name}' to '{target_folder.name}'")
        except Exception as e:
            logging.error(f"Failed to move '{image_path.name}': {e}")

        # Handle seam-specific JSON creation
        if final_class == "seam":
            folder_state["pending_seams"].append(len(processed_images) - 1)
            finalize_seam_json(folder_state, json_folder)

    logging.info(f"Completed processing for folder: {folder_path}")



def inference_loop(folder_path: str, tile_size: int):
    """Background thread for continuous inference."""
    global INFERENCE_THREAD_STOP
    while not INFERENCE_THREAD_STOP:
        process_folder(folder_path, tile_size)
        time.sleep(1)

def ensure_inference_running(folder_path: str, tile_size: int):
    """Ensure an inference thread is running for the specified folder."""
    global CURRENT_FOLDER, INFERENCE_THREAD, INFERENCE_THREAD_STOP

    if CURRENT_FOLDER == folder_path and INFERENCE_THREAD and INFERENCE_THREAD.is_alive():
        return

    INFERENCE_THREAD_STOP = True
    if INFERENCE_THREAD and INFERENCE_THREAD.is_alive():
        INFERENCE_THREAD.join()

    CURRENT_FOLDER = folder_path
    INFERENCE_THREAD_STOP = False
    INFERENCE_THREAD = threading.Thread(
        target=inference_loop, args=(folder_path, tile_size), daemon=True
    )
    INFERENCE_THREAD.start()

async def stream_json_files(websocket: WebSocket, folder_path: str):
    """Stream JSON files in real-time to the client."""
    global SENT_JSON_FILES
    json_folder = Path(folder_path) / "json"
    json_folder.mkdir(exist_ok=True)

    while True:
        await asyncio.sleep(1)
        for json_file in sorted(json_folder.glob("*.json")):
            if json_file.name not in SENT_JSON_FILES:
                SENT_JSON_FILES.add(json_file.name)
                await websocket.send_text(json_file.read_text(encoding="utf-8"))

# ---------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint to receive a folder path and stream results."""
    await websocket.accept()
    try:
        folder_path = await websocket.receive_text()
        if not Path(folder_path).is_dir():
            await websocket.send_text(f"Error: '{folder_path}' is not a valid folder.")
            await websocket.close()
            return

        ensure_inference_running(folder_path, 224)
        await stream_json_files(websocket, folder_path)

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected.")
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        await websocket.close()

# ---------------------------------------------------------------------
# Test HTML Endpoint
# ---------------------------------------------------------------------
@app.get("/")
def test_page():
    """Serve a simple WebSocket test page."""
    return HTMLResponse("""
    <html>
      <body>
        <h1>WebSocket Test</h1>
        <input id="folderPath" type="text" placeholder="Enter folder path" />
        <button onclick="startWebSocket()">Start</button>
        <div id="messages"></div>
        <script>
          let socket;
          function startWebSocket() {
            const folder = document.getElementById('folderPath').value;
            socket = new WebSocket(`ws://${location.host}/ws`);
            socket.onopen = () => socket.send(folder);
            socket.onmessage = (event) => {
              const msgDiv = document.getElementById('messages');
              msgDiv.innerHTML += `<pre>${event.data}</pre>`;
            };
          }
        </script>
      </body>
    </html>
    """)

# ---------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
