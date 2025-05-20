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
# Settings for inference
BATCH_SIZE = 64
COMPOSITE_HEIGHT = 1792
INFERENCE_INPUT_SIZE = 224
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SEAM_CONF_THRESHOLD = 0.7
import __main__
setattr(__main__, "TimmRandAugmentTransform", TimmRandAugmentTransform)

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