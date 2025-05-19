import json
import os
import cv2
import torch  # type: ignore
import concurrent.futures
from ultralytics import YOLO
import OptimizePaddleOCRFunc_v2 as OCRFunc

# Load the YOLO model; adjust the "best.pt" path if needed.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("best.pt")
model.to(device)

# Directory to store debug cropped images (optional).
DEBUG_CROP_DIR = "debug_crops"
os.makedirs(DEBUG_CROP_DIR, exist_ok=True)

def expand_box_vert3_horiz2(x1, y1, x2, y2, max_width, max_height):
    """
    Expands the given bounding box:
      - Two times wider.
      - Three times taller.
    Clamped to the image boundaries.
    """
    orig_width = x2 - x1
    orig_height = y2 - y1
    if orig_width <= 0 or orig_height <= 0:
        return int(x1), int(y1), int(x2), int(y2)
    
    cx = x1 + (orig_width / 2.0)
    cy = y1 + (orig_height / 2.0)
    expanded_half_w = orig_width  # 2× width expansion.
    expanded_half_h = orig_height * 3  # 3× height expansion.
    
    new_x1 = max(0, int(cx - expanded_half_w))
    new_x2 = min(max_width, int(cx + expanded_half_w))
    new_y1 = max(0, int(cy - expanded_half_h))
    new_y2 = min(max_height, int(cy + expanded_half_h))
    
    return new_x1, new_y1, new_x2, new_y2

def wechat_qr_detect(image):
    """
    Runs WeChat QR detection on the provided image region.
    Returns the first decoded text or None.
    """
    detector = cv2.wechat_qrcode_WeChatQRCode()
    texts, _ = detector.detectAndDecode(image)
    return texts[0] if texts else None

def process_image_path(image_path: str, skip_rotation: bool = False) -> dict:
    """
    Processes a single image:
      1. Reads the image.
      2. Runs YOLO detection to find a bounding box (with exception handling).
      3. If a box exists, runs WeChat QR on the original region.
      4. Expands and crops the detected region.
      5. Runs OCR on the expanded crop.
    Returns a dict with image_path, detection, fast_qr, and ocr results.
    """
    try:
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            return {"image_path": image_path, "error": "Invalid image file or path"}

        # --- YOLO detection with specific AttributeError handling ---
        try:
            results = model(img_cv)
        except AttributeError as e:
            msg = str(e)
            if "'Conv' object has no attribute 'bn'" in msg:
                # Return null JSON for this case
                return {
                    "image_path": image_path,
                    "detection": None,
                    "fast_qr": None,
                    "ocr": None
                }
            else:
                raise

        detection = json.loads(results[0].tojson())
        if isinstance(detection, list):
            detection = detection[0] if detection else {}

        # Extract best box
        boxes = detection.get("boxes", {}).get("data", [])
        best_box = None
        if boxes:
            conf_th = 0.7
            valid = [b for b in boxes if float(b[4]) >= conf_th]
            if valid:
                b = max(valid, key=lambda x: float(x[2]) - float(x[0]))
                best_box = {k: float(v) for k, v in zip(("x1","y1","x2","y2","confidence"), b)}
        if best_box is None and "box" in detection:
            best_box = detection["box"]
        detection["best_box"] = best_box

        # --- WeChat QR on original box region (if available) ---
        fast_qr = None
        if best_box:
            ox1, oy1, ox2, oy2 = map(int, (best_box["x1"], best_box["y1"], best_box["x2"], best_box["y2"]))
            orig_crop = img_cv[oy1:oy2, ox1:ox2]
            fast_qr = wechat_qr_detect(orig_crop)

        # --- Early return if no detection box ---
        if best_box is None:
            return {
                "image_path": image_path,
                "detection": detection,
                "fast_qr": fast_qr,
                "ocr": None
            }

        # --- Expand & crop the detected region ---
        x1, y1, x2, y2 = (best_box[k] for k in ("x1","y1","x2","y2"))
        h, w, _ = img_cv.shape
        nx1, ny1, nx2, ny2 = expand_box_vert3_horiz2(x1, y1, x2, y2, w, h)
        expanded_crop = img_cv[ny1:ny2, nx1:nx2]
        debug_path = os.path.join(DEBUG_CROP_DIR, f"crop_{os.path.basename(image_path)}")
        cv2.imwrite(debug_path, expanded_crop)

        # --- OCR on the expanded crop ---
        ocr_result = OCRFunc.ocr_paddleocr(
            expanded_crop,
            skip_rotation=(skip_rotation and fast_qr is not None)
        ) or {}

        # --- Combine final QR and ensure OCR field ---
        final_qr = fast_qr or ocr_result.get("Barcode_Number")
        ocr_result["Barcode_Number"] = final_qr

        return {
            "image_path": image_path,
            "detection": detection,
            "fast_qr": final_qr,
            "ocr": ocr_result
        }

    except Exception as e:
        return {"image_path": image_path, "error": str(e)}


def process_images(image_paths: list) -> list:
    """
    Processes a list of image paths concurrently.
    Returns a list of results from processing each image.
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image_path, p, True): p for p in image_paths}
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
    return results


def save_results_to_json(results: list, output_file: str = "results.json") -> None:
    """
    Saves the results list to a JSON file.
    """
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
