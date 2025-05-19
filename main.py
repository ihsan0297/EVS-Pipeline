import sys
import pathlib
from fastapi import FastAPI, HTTPException
from fastai.vision.all import load_learner, PILImage, Transform
from database_handler import DatabaseHandler
from custom_transforms import TimmRandAugmentTransform  # ✅ Import from external module
import os
from pathlib import Path
import torch
import os
import logging
import datetime
from pathlib import Path
import concurrent.futures
from PIL import Image
import torch
import pandas as pd 
import numpy as np

# -----------------------------
# GLOBAL CONFIG & SETTINGS
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app.log",         # Log output goes to this file
    filemode="a"                # Append to the file (use "w" to overwrite each time)
)


# Settings for inference
BATCH_SIZE = 64
COMPOSITE_HEIGHT = 1792
INFERENCE_INPUT_SIZE = 224
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SEAM_CONF_THRESHOLD = 0.7


from rabbitmq_handler import RMQHandler
# Fix WindowsPath issue if running on Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import concurrent.futures
from qrupdate import process_images
import enums
# ✅ Register the transform in the global namespace
sys.modules['custom_transforms.TimmRandAugmentTransform'] = TimmRandAugmentTransform

# ✅ Also register it in the built-in __main__ module
import __main__
setattr(__main__, "TimmRandAugmentTransform", TimmRandAugmentTransform)
import concurrent.futures

def process_image(model, img):  # Pass the model as an argument
    try:
        pred, pred_idx, probs = model.predict(img)
        print(pred)
        return {
            "image": img,
            "prediction": pred,
            "confidence": probs[pred_idx].item(),
        }
    except Exception as e:
        return {
            "image": img,
            "error": str(e),
        }

class Model:
    def __init__(self):
        self.model1 = None
        self.model2 = None
        self.model3=None
        self.model1_path = "Models/SeamMiniV1.pkl"
        self.model2_path = "Models/SeamV1.pkl"
        self.model3_path="Models/Seamvit.pkl"
        self.device=None

    def load_models(self):
        try:
            # ✅ Manually ensure the custom transform is available
            setattr(__main__, "TimmRandAugmentTransform", TimmRandAugmentTransform)

            # ✅ Load the models
            # self.model1 = load_learner(self.model1_path, cpu=True)
            # self.model2 = load_learner(self.model2_path, cpu=True)
            self.model3 = load_learner(self.model3_path, cpu=True)
            

            # ✅ Apply the custom transforms to the loaded models
            # self.model1.dls.after_item.add(TimmRandAugmentTransform())
            # self.model2.dls.after_item.add(TimmRandAugmentTransform())
            self.model3.dls.after_item.add(TimmRandAugmentTransform())
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model3.model = self.model3.model.to(self.device)
            self.model3.model.eval()
            if hasattr(self.model3, 'dls') and hasattr(self.model3.dls, 'device'):
                self.model3.dls.device = self.device

            print("✅ Models loaded successfully with TimmRandAugmentTransform")
        except Exception as e:
            logging.error("Failed to load models: %s", e)

            raise HTTPException(status_code=500, detail=f"Failed to load models: {e}")

    def parse_datetime_from_filename(self,name: str) -> datetime.datetime or None:
        try:
            stem = Path(name).stem
            parts = stem.split('_')
            if len(parts) < 2 or not parts[1]:
                return None
            timestamp_str = parts[1]
            if '-' in timestamp_str:
                date_parts = timestamp_str.rsplit('-', 1)
                if len(date_parts[1]) == 3:
                    timestamp_str_full = date_parts[0] + '-' + date_parts[1] + "000"
                else:
                    timestamp_str_full = timestamp_str
            else:
                timestamp_str_full = timestamp_str
            return datetime.datetime.strptime(timestamp_str_full, "%Y-%m-%d %H-%M-%S-%f")
        except Exception as e:
            logging.error("Failed to parse datetime from filename '%s': %s", name, e)
            return None

    def numeric_sort_key(self,path: Path) -> datetime.datetime:
        dt = self.parse_datetime_from_filename(path.name)
        return dt if dt else datetime.datetime.min

    def safe_load_image(self,path: Path) -> Image.Image or None:
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logging.error("Could not open image %s: %s", path, e)
            return None
    def run_inference_on_batch_items(self,batch_items: list) -> list:
        try:
            results = []
            for idx in range(0, len(batch_items), BATCH_SIZE):
                sub_batch = batch_items[idx: idx + BATCH_SIZE]
                try:
                    tile_tensor = torch.stack([
                        torch.tensor(np.array(item["tile_img"])).permute(2, 0, 1).float()
                        for item in sub_batch
                    ]).to(self.device)
                except Exception as e:
                    logging.error("Error preparing batch tensor: %s", e)
                    continue

                # Normalize: scale pixel values to [0,1]
                tile_tensor = tile_tensor / 255.0
                norm = None
                if hasattr(self.model3, 'dls') and hasattr(self.model3.dls, 'after_batch'):
                    from fastai.vision.all import Normalize
                    for tfm in self.model3.dls.after_batch.fs:
                        if isinstance(tfm, Normalize):
                            norm = tfm
                            break
                if norm is not None:
                    tile_tensor = norm(tile_tensor)
                else:
                    # Fallback normalization (ImageNet stats)
                    tile_tensor = (tile_tensor - torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)) / \
                                torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

                with torch.no_grad():
                    logits = self.model3.model(tile_tensor)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                preds = probs.cpu().numpy()
                for i, item in enumerate(sub_batch):
                    prob = preds[i]
                    pred_idx = int(prob.argmax())
                    if hasattr(self.model3.dls, 'vocab'):
                        pred_class = self.model3.dls.vocab[pred_idx]
                    else:
                        pred_class = str(pred_idx)
                    conf = float(prob[pred_idx])
                    seam_found = (pred_class.lower().strip() == "seam" and conf >= SEAM_CONF_THRESHOLD)
                    composite_path = item["composite_path"]
                    stitched_info = item.get("stitched_info")
                    if stitched_info:
                        image_title = f"{Path(stitched_info[0]).stem}_{Path(stitched_info[-1]).stem}"
                    else:
                        image_title = composite_path.stem if isinstance(composite_path, Path) else str(composite_path)
                    result = {
                        "image_title": image_title,
                        "image_path": str(composite_path),
                        "final_class": "seam" if seam_found else "normal",
                        "confidence": round(conf, 4),
                        "note": "Seam detected." if seam_found else "Normal image."
                    }
                    results.append(result)
            return results
        except Exception as e:
            logging.error("Error during inference: %s", e)
            return []

    def process_images_seam_model(self,image_paths: list) -> list:
        
        """
        Given a list of image file paths, this function:
        1. Sorts the images using a numeric sort key.
        2. Loads images concurrently using safe_load_image.
        3. Depending on image dimensions, may attempt stitching groups of images,
            upscale small images, or use them as is.
        4. Prepares a batch of tile images for model inference.
        5. Runs inference in batches and returns a list of result dictionaries.
        """
        
        try:
            # Convert file paths to Path objects and sort them.
            paths = [Path(p) if not isinstance(p, Path) else p for p in image_paths]
            sorted_paths = sorted(paths, key=self.numeric_sort_key)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                loaded_list = list(executor.map(self.safe_load_image, sorted_paths))
            images_data = list(zip(sorted_paths, loaded_list))

            batch_items = []
            i = 0
            while i < len(images_data):
                path, img = images_data[i]
                if img is None:
                    i += 1
                    continue

                composite_img = None
                composite_path = path
                stitched_info = None

                # Case 1: If image height equals 224, attempt to stitch groups of four.
                if img.height == 224:
                    group = []
                    group_paths = []
                    j = i
                    while j < len(images_data) and len(group) < 4:
                        p, c_img = images_data[j]
                        if c_img is not None and c_img.height == 224:
                            group.append(c_img)
                            group_paths.append(str(p))
                            j += 1
                        else:
                            break
                    if len(group) == 4:
                        width = group[0].width
                        composite_img = Image.new("RGB", (width, 224 * 4))
                        for k in range(4):
                            composite_img.paste(group[k], (0, k * 224))
                        composite_filename = f"stitched_{Path(group_paths[0]).stem}_{Path(group_paths[-1]).stem}.jpg"
                        composite_path = Path(composite_filename)
                        stitched_info = group_paths
                        i = j
                    else:
                        composite_img = img
                        i += 1
                elif img.height < 224:
                    # Upscale small images.
                    composite_img = img.resize((INFERENCE_INPUT_SIZE, INFERENCE_INPUT_SIZE), Image.Resampling.LANCZOS)
                    i += 1
                elif img.height == COMPOSITE_HEIGHT:
                    composite_img = img
                    i += 1
                else:
                    composite_img = img
                    i += 1

                if composite_img:
                    width, height = composite_img.size
                    tile_by_height = (height // 224) * 224
                    tile_by_width = (width // 224) * 224
                    effective_tile_size = min(tile_by_height, tile_by_width)
                    if effective_tile_size < 224:
                        tile_img_resized = composite_img.resize((INFERENCE_INPUT_SIZE, INFERENCE_INPUT_SIZE),
                                                                Image.Resampling.LANCZOS)
                    else:
                        crop_box = (0, 0, effective_tile_size, effective_tile_size)
                        tile_img = composite_img.crop(crop_box)
                        tile_img_resized = tile_img.resize((INFERENCE_INPUT_SIZE, INFERENCE_INPUT_SIZE),
                                                        Image.Resampling.LANCZOS)
                    batch_items.append({
                        "tile_img": tile_img_resized,
                        "composite_path": composite_path,
                        "stitched_info": stitched_info
                    })

            inference_results = self.run_inference_on_batch_items(batch_items)
            return inference_results
        except Exception as e:
            logging.error("Error in process_images_seam_model: %s", e)
            return []


    def perform_inference( self,images: list):
        def process_image(img):
            try:
                # Perform inference on the image
                pred, pred_idx, probs = self.model3.predict(img)
                print(pred)

                # Return result for this image
                return {
                    "image":img,
                    "prediction": pred,
                    "confidence": probs[pred_idx].item(),
                }
            except Exception as e:
                return {
                    "error": str(e),
                }

        # Use ThreadPoolExecutor to process all images in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Execute the inference for all images
            results = list(executor.map(process_image, images))

        # Return the results for all images
        return results


app = FastAPI()
model = Model()
db=DatabaseHandler(enums.MYSQL_HOST,enums.MYSQL_USER_NAME,enums.MYSQL_PASSWORD,enums.MYSQL_DATABASE)
db_barcode=DatabaseHandler(enums.MYSQL_HOST,enums.MYSQL_USER_NAME,enums.MYSQL_PASSWORD,enums.DATABASE_FAULT_APP)
rmq=RMQHandler(enums.RMQ_HOST, enums.RMQ_USER,enums.RMQ_PASSWORD)


@app.on_event("startup")
def startup_event():
    print("Loading models...")
    model.load_models()
    print("Models loaded successfully!")
    
        


def get_barcode_from_object(obj):
        qr        = obj.get("qr_data") or {}

        ocr       = qr.get("ocr") or {}
        barcode_from_json=""
        plain_text = (ocr.get("Plain_text") or "").strip()
        fast_qr=(qr.get("fast_qr")      or "").strip()
        low_confidence_matches = ""
        if fast_qr:
            barcode_from_json = fast_qr
        else:
            # Fallback to similarity_test -> matches[0] -> match
            similarity_test = qr.get("similarity_test") or {}
            matches = similarity_test.get("matches") or []

            if matches and isinstance(matches, list) and len(matches) > 0 and "match" in matches[0]:
                # Check if score is at least 0.9
                if "score" in matches[0] and matches[0]["score"] >= 0.9:
                    barcode_from_json = matches[0]["match"]
                else:
                    # Score is less than 0.9 - use original barcode from OCR
                    barcode_from_json = ocr.get("Plain_text", "")  # use plain_text as original barcode
                    low_confidence_matches = matches  # Save matches for reference
                    print(f"Low confidence match ({matches[0].get('score', 0)}) - using original barcode")
            else:
                barcode_from_json = plain_text  # default if all sources fail
        return barcode_from_json
def normalize_barcode(code):
    """Normalize barcode for more reliable comparison"""
    if not code:
        return ""
    # Remove all whitespace
    code = "".join(code.split())
    # Ensure consistent case (less relevant for numeric codes)
    code = code.upper()
    return code    
def get_barcode_position(combined_results, target_barcode):

    try:
        # Initialize flag to track if a seam has been found
        is_seam_found = False
        
        # Loop through all results
        for obj in combined_results:
            # Check if this frame has a seam
            seam_data = obj.get("seam_data", {})
            if seam_data and seam_data.get("final_class") == "seam":
                is_seam_found = True
            
            # Check if this frame has the target barcode
            barcode=get_barcode_from_object(obj)
            if normalize_barcode(barcode)==normalize_barcode(target_barcode):
                if is_seam_found:
                    return 1
                return 2
                
        return 0
        
    except Exception as e:
        logging.error(f"❌ Error in get_barcode_position: {e}")
        print(f"❌ Error in get_barcode_position: {e}")
        return 0
def check_and_add_seams_between_barcodes(combined_results):
    """
    Identifies barcodes in the results, checks if seams exist between consecutive barcodes,
    and inserts synthetic seams where needed.
    
    Args:
        combined_results: List of processed image results with seam and QR data
        
    Returns:
        tuple: (seams_added, updated_results)
            seams_added: Number of synthetic seams added
            updated_results: The original or modified combined_results
    """
    try:
        # Find all barcodes and their positions
        barcode_positions = []
        
        for i, result in enumerate(combined_results):
            barcode = get_barcode_from_object(result)
            if barcode:  # If barcode exists
                barcode_positions.append({
                    "index": i,
                    "barcode": barcode,
                    "meter": result.get(enums.METERS_COLUMN),
                    "path": result.get(enums.IMAGE_PATH_COLUMN)
                })
        
        if len(barcode_positions) < 2:
            print(f"Found {len(barcode_positions)} barcodes - need at least 2 to check for seams")
            return 0, combined_results
            
        print(f"Found {len(barcode_positions)} barcodes")
        
        # Sort by meter position to ensure correct order
        barcode_positions.sort(key=lambda x: x["meter"] if x["meter"] is not None else float('inf'))
        
        # Check for seams between consecutive barcode pairs
        updated_results = combined_results.copy()
        seams_added = 0
        
        for i in range(len(barcode_positions) - 1):
            pos1 = barcode_positions[i]
            pos2 = barcode_positions[i + 1]
            
            idx1, idx2 = pos1["index"], pos2["index"]
            
            # Check if there's a seam between these barcodes
            seam_exists = False
            for j in range(min(idx1, idx2), max(idx1, idx2) + 1):
                seam_data = updated_results[j].get("seam_data", {})
                if seam_data and seam_data.get("final_class") == "seam":
                    seam_exists = True
                    print(f"✅ Seam found between {pos1['barcode']} and {pos2['barcode']}")
                    break
            
            # If no seam exists, create a synthetic one
            if not seam_exists:
                # Calculate average meter position
                if pos1["meter"] is not None and pos2["meter"] is not None:
                    avg_meter = (pos1["meter"] + pos2["meter"]) / 2
                else:
                    # Insert at middle index if meter info not available
                    avg_meter = None
                
                # Create synthetic seam entry
                roll_header_id = combined_results[idx1].get(enums.ROLL_HEADER_ID_COLUMN)
                synthetic_seam = {
                    enums.IMAGE_TITLE_COLUMN: f"synthetic_seam_{seams_added}",
                    enums.IMAGE_PATH_COLUMN: f"synthetic_path_{seams_added}",
                    enums.METERS_COLUMN: avg_meter,
                    enums.ROLL_HEADER_ID_COLUMN: roll_header_id,
                    enums.IS_BARCODE_READABLE_COLUMN: False,
                    "seam_data": {
                        "final_class": "seam",
                        "conf": 0.95  # High confidence for synthetic seam
                    },
                    "synthetic": True,  # Mark as synthetic entry
                    "created_between": {
                        "barcode1": pos1["barcode"],
                        "barcode2": pos2["barcode"]
                    }
                }
                
                # Find position to insert based on meter value
                insert_pos = idx1 + 1
                if avg_meter is not None:
                    while insert_pos < len(updated_results):
                        current_meter = updated_results[insert_pos].get(enums.METERS_COLUMN)
                        if current_meter is not None and current_meter > avg_meter:
                            break
                        insert_pos += 1
                
                # Insert synthetic seam
                updated_results.insert(insert_pos, synthetic_seam)
                seams_added += 1
                
                print(f"⚠️ No seam found between {pos1['barcode']} and {pos2['barcode']} - inserted synthetic seam at {avg_meter} meters")
                
                # Update indexes of barcode positions after this insertion
                for k in range(i + 1, len(barcode_positions)):
                    if barcode_positions[k]["index"] >= insert_pos:
                        barcode_positions[k]["index"] += 1
        
        print(f"Added {seams_added} synthetic seams")
        return seams_added, updated_results
        
    except Exception as e:
        logging.error(f"❌ Error in check_and_add_seams_between_barcodes: {e}")
        print(f"❌ Error in check_and_add_seams_between_barcodes: {e}")
        return 0, combined_results
def apply_checks(combined_results, prev_df, raw_messages,roll_id):
    try:
        is_update_prev_result = False
        updated_prev_rows = []
        is_seam,combined_results=check_and_add_seams_between_barcodes(combined_results)
        # prev_df = 

        # # Prepare previous results DataFrame
        # if all_prev_results_df is None or all_prev_results_df.empty or 'roll_header_id' not in all_prev_results_df.columns:
        #     prev_df = pd.DataFrame().rename_axis('roll_header_id')
        # else:
        #     prev_df = all_prev_results_df.set_index('roll_header_id')

        # Collect records and delivery tags
        records_to_insert = []
        delivery_tags = []

        latest_is_piece_completed=0
        #1 if piece is completed, 0 otherwise
        for i, result in enumerate(combined_results):
            try:
                # Ack tag
                method_frame = raw_messages[i]["method_frame"]
                delivery_tags.append(method_frame.delivery_tag)

                # Defaults

                latest_barcode          = ""
                latest_box_coord        = None
                latest_best_box_coord   = None
                barcode_manual= ""
                latest_is_barcode_model = 0 #1 if barcode model is detected, 0 otherwise
                latest_is_barcode_manual= 0 #1 if barcode is detected manually like if the prevous was empty
                latest_is_sticker_model = 0#1 if sticker model is detected, 0 otherwise
                latest_is_sticker_manual= 0#1 if sticker is detected manually like if the prevous was empty

                # roll_header_id = result.get(enums.ROLL_HEADER_ID_COLUMN) #Trolley ID of the roll    
                # prev_res = (
                #     prev_df.loc[[roll_id]]
                #     if roll_id in prev_df.index
                #     else pd.DataFrame()
                # )
                
                prev_res = prev_df
                qr        = result.get("qr_data") or {}
                detection = qr.get("detection") or {}
                ocr       = qr.get("ocr") or {}

                # pull and trim both sources
                barcode_from_json=""
                plain_text = (ocr.get("Plain_text") or "").strip()
                fast_qr=(qr.get("fast_qr")      or "").strip()
                low_confidence_matches = ""
                if fast_qr:
                    barcode_from_json = fast_qr
                else:
                    # Fallback to similarity_test -> matches[0] -> match
                    similarity_test = qr.get("similarity_test") or {}
                    matches = similarity_test.get("matches") or []

                    if matches and isinstance(matches, list) and len(matches) > 0 and "match" in matches[0]:
                        # Check if score is at least 0.9
                        if "score" in matches[0] and matches[0]["score"] >= 0.9:
                            barcode_from_json = matches[0]["match"]
                            
                            print(f"High confidence match ({matches[0].get('score', 0)}) - using barcode from similarity test")
                            print("Barcode from similarity test:", barcode_from_json)
                        else:
                            # Score is less than 0.9 - use original barcode from OCR
                            barcode_from_json = ocr.get("Plain_text", "")  # use plain_text as original barcode
                            low_confidence_matches = matches  # Save matches for reference
                            print(f"Low confidence match ({matches[0].get('score', 0)}) - using original barcode")
                    else:
                        barcode_from_json = plain_text  # default if all sources fail
                # detection true iff either is non-empty
                has_detection = bool(fast_qr) or bool(plain_text)
                # print('Pre. Res: ',prev_res)
                condition_col=""

                if result.get(enums.IS_BARCODE_READABLE_COLUMN) and barcode_from_json:
                    print('Roll header id:',roll_id)
                    print('Pre. Res: ',prev_res)                    
                    latest_is_sticker_model=1
                    # pick whichever value is available
                    # latest_barcode = fast_qr or (plain_text.split()[0] if plain_text else "") #Current barcode value                    latest_barcode      = barcode_value 
                    latest_barcode = barcode_from_json
                    latest_box_coord    = detection.get("box") or {} #Coordinates of the barcode
                    latest_best_box_coord = detection.get("best_box") or {} #Coordinates of the best box
                    # is_second,seam=check_is_barcode_second(latest_barcode,roll_header_id)
                    if not prev_res.empty: #Barocde is second
                        if normalize_barcode(prev_res.iloc[0].get("barcode_model")) == normalize_barcode(latest_barcode):
                            latest_is_barcode_model = 1
                            latest_is_barcode_model=1
                            latest_is_piece_completed=1
                            prev_df=pd.DataFrame()
                            print(f"At if prev_res.iloc[0].get(barcode_model) == latest_barcode")
                            print('Barcode: ',latest_barcode)
                            condition_col="At if prev_res.iloc[0].get(barcode_model) == latest_barcode"
                        else:
                            latest_barcode=prev_res.iloc[0].get("barcode_model")
                            # latest_is_barcode_model = 0
                            latest_is_barcode_manual=1
                            latest_is_sticker_model=1
                            latest_is_piece_completed=1
                            barcode_manual=latest_barcode
                            prev_df=pd.DataFrame()
                            print(f"At Else of  if prev_res.iloc[0].get(barcode_model) == latest_barcode")
                            print('Barcode: ',latest_barcode)
                            condition_col="At Else of  if prev_res.iloc[0].get(barcode_model) == latest_barcode"
                    elif get_barcode_position(combined_results, latest_barcode) == 2:
                        print("At elif get_barcode_position(combined_results, latest_barcode) == 2")
                        current_meter = result.get(enums.METERS_COLUMN, 0)
                        print('Current meter:',current_meter)
                        second_barcode_df=db.is_barcode_second(roll_id,current_meter)
                        print('Second barcode df:',second_barcode_df)
                        if not second_barcode_df.empty:
                            is_update_prev_result = True
                            index = second_barcode_df.index[0]  # first row
                            second_barcode_df.at[index, 'barcode_manual'] = latest_barcode
                            second_barcode_df.at[index, 'is_sticker_model'] = 1
                            second_barcode_df.at[index,'is_barcode_manual']=1                        
                            updated_prev_rows.append(second_barcode_df.loc[index].copy())
                            second_barcode_df=pd.DataFrame()

                        latest_is_piece_completed=1
                        latest_is_barcode_model=1
                        latest_is_sticker_model=1
                        condition_col="At elif not second_barcode_df.empty"


                        print("At elif not second_barcode_df.empty")
                        print('Barcode ',latest_barcode)
                        
                    else:
                        latest_is_piece_completed=0
                        print("At Else")
                        print('Barcode: ',latest_barcode)
                        print("Latest box coord:", latest_box_coord)
                        print("Latest best box coord:", latest_best_box_coord)
                        condition_col="At Else"

                    
                    # elif is_second:
                        # update the seam
                
                # build the DB record
                db_record = {
                    enums.ROLL_HEADER_ID_COLUMN: roll_id,
                    enums.IMAGE_TITLE_COLUMN:    os.path.basename(result.get(enums.IMAGE_TITLE_COLUMN)),
                    enums.IMAGE_PATH_COLUMN:     result.get(enums.IMAGE_PATH_COLUMN),
                    "image_in_cm":               0.0,
                    "image_class_id":            result.get("seam_data", {}).get("final_class"),
                    "confidence":                result.get("seam_data", {}).get("conf"),
                    "box_coord":                 latest_box_coord,
                    "best_box_coord":            latest_best_box_coord,
                    "is_sticker_model":          latest_is_sticker_model,
                    "is_sticker_manual":         latest_is_sticker_manual,
                    "is_barcode_manual":         latest_is_barcode_manual,
                    "is_barcode_model":          latest_is_barcode_model,
                    "is_damage_barcode":         0,
                    "barcode_manual":            barcode_manual,
                    "barcode_model":             latest_barcode,
                    "plain_text":                plain_text,
                    enums.METERS_COLUMN:         result.get(enums.METERS_COLUMN, 0),
                    "is_piece_completed":      latest_is_piece_completed,
                    "low_confidence_matches": low_confidence_matches,
                    "condition_col": condition_col
                }
                records_to_insert.append(db_record)
                latest_is_piece_completed=0
                # lates

            except Exception as e:
                logging.error("❌ Failed to prepare record for %s: %s",
                              result.get(enums.IMAGE_PATH_COLUMN), e)

        # Bulk insert all records
        if records_to_insert:
            try:
                print("Bulk inserting records...")
                if db.insert_inference_results_bulk(records_to_insert):
                    print("✅ Bulk insert successful")
                    if delivery_tags:
                        last_tag = max(delivery_tags)
                        rmq.channel.basic_ack(delivery_tag=last_tag, multiple=True)
                        print(f"✅ Acknowledged {len(delivery_tags)} messages")
                else:
                    logging.error("❌ Bulk insert failed - likely due to empty roll_header_id")
                    # ack only those with empty header
                    empty_tags = [
                        t for rec, t in zip(records_to_insert, delivery_tags)
                        if not rec.get(enums.ROLL_HEADER_ID_COLUMN)
                    ]
                    if empty_tags:
                        last_tag = max(empty_tags)
                        rmq.channel.basic_ack(delivery_tag=last_tag, multiple=True)
                        print(f"✅ Ack {len(empty_tags)} msgs with empty roll_header_id")

            except Exception as e:
                logging.error("❌ Bulk insert error: %s", e)

        # Update any modified “previous” rows
        if is_update_prev_result and updated_prev_rows:
            try:
                db.close()
                db.connect()
                updated_df = pd.DataFrame(updated_prev_rows)
                db.update_previous_detection_from_df(updated_df)
                print("✅ Updated previous result(s) in DB")
            except Exception as e:
                logging.error("❌ Failed to update previous results: %s", e)

    except Exception as e:
        logging.error("❌ Exception in apply_checks: %s", e)
        print(f"❌ Exception in apply_checks: {e}")

# from logging_config import LOGGING_CONFIG
# import logging.config
# logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)  # this will now write into app.log
import json
from similarity_test import annotate_with_similarity
def sanitize_qr_res(qr_results):
    try:
        sanitized_qr_results = []
        for item in qr_results:
            print("Checking item:", item)
            if isinstance(item, dict):
                sanitized_qr_results.append(item)
            elif isinstance(item, str):
                try:
                    parsed = json.loads(item)
                    if isinstance(parsed, dict):
                        sanitized_qr_results.append(parsed)
                except json.JSONDecodeError:
                    continue  # or log and skip
        return sanitized_qr_results
    except Exception as e:
        print(f"❌ Error sanitizing QR results: {e}")
import io
import sys
import contextlib
import cv2
# from numba import cuda       # to JIT-compile & launch CUDA kernels

# @cuda.jit
# def find_qr_kernel(gray_images, results, img_h, img_w):
#     # thread indices
#     y, x = cuda.grid(2)
#     if y < img_h and x < img_w:
#         # very toy example: threshold
#         pix = gray_images[y, x]
#         results[y, x] = 255 if pix > 127 else 0
# def process_images_gpu(batch_paths):
#     qr_results = []
#     for p in batch_paths:
#         img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
#         h, w = img.shape

#         # 2) Copy to device
#         d_in  = cuda.to_device(img)
#         d_out = cuda.device_array_like(img)

#         # 3) Launch your kernel
#         threadsperblock = (16, 16)
#         blockspergrid = ( (h + threadsperblock[0] - 1)//threadsperblock[0],
#                           (w + threadsperblock[1] - 1)//threadsperblock[1] )
#         find_qr_kernel[blockspergrid, threadsperblock](d_in, d_out, h, w)

#         # 4) Copy result back and post-process on host
#         bin_img = d_out.copy_to_host()
#         # … contour-based QR decoding here …
#         qr_results.append(decode_qr_from_binary(bin_img))

#     return {"qr_Res": qr_results}
@app.get("/inference_image")
def inference_images():
    try:
        logger.info("Starting inference_images")

        if not db.connect():
            logging.error("DB Connection Failed")
            return {"status": "error", "message": "Database Connection Failed"}

        if not rmq.rabbitMQConnection():
            logging.error("RMQ Connection Failed")
            return {"status": "error", "message": "RMQ Connection Failed"}
        if not db_barcode.connect():
            logging.error("DB Connection Failed")
            return {"status": "error", "message": "Database Connection Failed"}
        

        message_bundle = rmq.fetch_messages()
        messages = message_bundle.get("images", [])
        raw_messages = message_bundle.get("raw", [])

        if not messages:
            return {"status": "error", "message": "No images found in queue"}

        # Group messages by roll_header_id
        roll_groups = {}
        for msg in messages:
            roll_id = msg.get(enums.ROLL_HEADER_ID_COLUMN)
            if roll_id is not None:
                roll_groups.setdefault(roll_id, []).append(msg)

        all_combined_results = []

        for roll_id, roll_msgs in roll_groups.items():
            print(f"🎯 Processing Roll Header ID: {roll_id}")
            logging.info("Processing Roll Header ID: %s", roll_id)

            # Fetch previous sticker model results
            prev_res=db.get_next_greater_row_by_roll_header_id(roll_id)
            
            print("Previous result:", prev_res)
            
            # prev_res = db.fetch_latest_sticker_model(roll_id)
            all_prev_results_df = pd.DataFrame()
            if prev_res is not None and not prev_res.empty:
                all_prev_results_df = prev_res
                print("✅ Previous sticker model found for roll_id:", roll_id)
            else:
                print("⚠️ No sticker model found for roll_id:", roll_id)

            image_paths = [msg[enums.IMAGE_PATH_COLUMN] for msg in roll_msgs]

            seam_results = model.process_images_seam_model(image_paths)
            seam_result_map = {
                res["image_path"]: {
                    "final_class": res.get("final_class"),
                    "conf": round(res.get("confidence", 0), 4)
                }
                for res in seam_results
            }

            combined_results = []
            qr_candidates = set()

            for idx, msg in enumerate(roll_msgs):
                image_path = msg[enums.IMAGE_PATH_COLUMN]
                combined = {
                    enums.IMAGE_TITLE_COLUMN: os.path.basename(image_path),
                    enums.IMAGE_PATH_COLUMN: image_path,
                    enums.METERS_COLUMN: msg.get(enums.METERS_COLUMN),
                    enums.ROLL_HEADER_ID_COLUMN: roll_id,
                    enums.IS_BARCODE_READABLE_COLUMN: msg.get(enums.IS_BARCODE_READABLE_COLUMN),
                    "seam_data": seam_result_map.get(image_path)
                }

                seam_pred = seam_result_map.get(image_path, {}).get("final_class")
                if seam_pred == enums.SEAM:
                    print('Seam detected in image:', image_path)
                #     start = max(0, idx - enums.LOOKUP_DOWN_IMAGE_COUNT)
                #     end = min(len(roll_msgs), idx + enums.LOOKUP_DOWN_IMAGE_COUNT + 1)
                #     for i in range(start, end):
                #         if i != idx:
                #             qr_candidates.add(roll_msgs[i][enums.IMAGE_PATH_COLUMN])

                combined_results.append(combined)

            # Sort QR candidates by meters before sending to QR model
            qr_candidates_info = [
                msg for msg in roll_msgs if msg[enums.IMAGE_PATH_COLUMN] in qr_candidates
            ]
            qr_candidates_sorted = sorted(
                combined_results,
                key=lambda x: x.get(enums.METERS_COLUMN, float('inf'))
            )
            qr_paths_sorted = [msg[enums.IMAGE_PATH_COLUMN] for msg in qr_candidates_sorted]

            # Run QR detection
            print("Running QR detection on images...")
            with contextlib.redirect_stdout(io.StringIO()):
                qr_results = process_images(qr_paths_sorted)
            # qr_results = process_images(qr_paths_sorted)
            print("QR detection Finished.....")
            with open("qr.json", "w", encoding="utf-8") as f:
                json.dump(qr_results, f, indent=2, ensure_ascii=False)
            # qr_items = qr_results.get("qr_Res", [])

            # qr_result_map = {
            #     item["image_path"]: item
            #     for item in qr_items
            #     if "image_path" in item and item["image_path"]
            # }

            # Merge QR data


            # Run checks and save

            #Similarity Test
            try:
                print('...............Starting similarity test............')
                barcode_from_db=db_barcode.fetch_barcode_by_header_id(roll_id)
                # barcode_values = [b[0] for b in barcode_from_db if b[0]]
                
                if len(barcode_from_db)<=0:
                    logging.error("❌ No barcodes found in the database")
                    return {"status": "error", "message": "No barcodes found in the database"}

                qr_items = qr_results.get('qr_Res', [])
                sanitized_items = []
                
                for item in qr_items:
                    if item is None:
                        continue
                        
                    sanitized_item = {}
                    sanitized_item["image_path"] = item.get("image_path", "")
                    
                    # Ensure detection is a dictionary
                    if item.get("detection") is None:
                        sanitized_item["detection"] = {}
                    else:
                        sanitized_item["detection"] = item["detection"]
                        
                    # Ensure fast_qr exists (can be None)
                    sanitized_item["fast_qr"] = item.get("fast_qr")
                    
                    # Ensure ocr is a dictionary
                    if item.get("ocr") is None:
                        sanitized_item["ocr"] = {}
                    else:
                        sanitized_item["ocr"] = item["ocr"]
                        
                    sanitized_items.append(sanitized_item)
                
                # Use sanitized data
                sanitized_qr_results = {"qr_Res": sanitized_items}

                qr_items = qr_results.get('qr_Res', [])
                ultra_processed_barcodes = annotate_with_similarity(
                     sanitized_qr_results, barcode_from_db, top_n=2
                )
                # print("Ultra processed barcodes:", ultra_processed_barcodes)
                with open("qr_sim.json", "w", encoding="utf-8") as f:
                    json.dump(ultra_processed_barcodes, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"❌ Error in similarity test: {e}")
                return {"status": "error", "message": str(e)}
            try:
                qr_data_map = {}
                if ultra_processed_barcodes and "qr_Res" in ultra_processed_barcodes:
                    for item in ultra_processed_barcodes["qr_Res"]:
                        if item and "image_path" in item:
                            qr_data_map[item["image_path"]] = item
        # Merge QR data into combined results    
                for result in combined_results:
                    path = result[enums.IMAGE_PATH_COLUMN]
                    
                    if path in qr_data_map:
                        result["qr_data"] = qr_data_map[path]
                apply_checks(combined_results, prev_res, raw_messages,roll_id)
                meters_vals = [res.get(enums.METERS_COLUMN) for res in combined_results]

                if not meters_vals or meters_vals[0] is None or meters_vals[-1] is None:
                    raise ValueError("Meters column is missing or empty – cannot name results file")

                first_meters = meters_vals[0]
                last_meters  = meters_vals[-1]
                out_dir = pathlib.Path("output")
                out_dir.mkdir(exist_ok=True)
                results_path = out_dir / f"results_{first_meters}_{last_meters}.json"
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump({"images": combined_results}, f, indent=2, ensure_ascii=False)
                print('...............Finished similarity test............')
            except Exception as e:
                print(f"❌ Error in Combining Results: {e}")
                return {"status": "error", "message": str(e)}
            # save_results_to_json({"images": combined_results}, f"results_{roll_id}.json")

            all_combined_results.extend(combined_results)

        return {"status": "success", "message": f"Processed {len(all_combined_results)} images across {len(roll_groups)} rolls"}

    except Exception as e:
        logging.error("Error in inference_images: %s", e)
        print(f"❌ Error in inference_images: {e}")
        return {"status": "error", "message": str(e)}
# import uvicorn

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8100)