
import sys
import pathlib
from fastapi import FastAPI, HTTPException
from fastai.vision.all import load_learner, PILImage, Transform
from database_handler import DatabaseHandler
from custom_transforms import TimmRandAugmentTransform
import os
from pathlib import Path
import torch
import logging
import datetime
import concurrent.futures
from PIL import Image
import numpy as np
import json
from typing import List, Dict, Any

# Settings for inference
BATCH_SIZE = 32
COMPOSITE_HEIGHT = 896
INFERENCE_INPUT_SIZE = 224
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SEAM_CONF_THRESHOLD = 0.7

import __main__
setattr(__main__, "TimmRandAugmentTransform", TimmRandAugmentTransform)


class Model:
    def __init__(self):
        self.model = None
        self.model_path = "Models/896v1.pkl"
        self.device = None

    def load_models(self):
        try:
            setattr(__main__, "TimmRandAugmentTransform", TimmRandAugmentTransform)
            self.model = load_learner(self.model_path, cpu=True)
            self.model.dls.after_item.add(TimmRandAugmentTransform())
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.model = self.model.model.to(self.device)
            self.model.model.eval()
            self.model.dls.device = self.device
        except Exception as e:
            logging.error("Failed to load model: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    def parse_datetime_from_filename(self, name: str) -> datetime.datetime | None:
        try:
            stem = Path(name).stem
            parts = stem.split("_")
            ts = parts[1]
            if "-" in ts:
                dp = ts.rsplit("-", 1)
                if len(dp[1]) == 3:
                    ts = f"{dp[0]}-{dp[1]}000"
            return datetime.datetime.strptime(ts, "%Y-%m-%d %H-%M-%S-%f")
        except:
            return None

    def numeric_sort_key(self, path: Path) -> datetime.datetime:
        dt = self.parse_datetime_from_filename(path.name)
        return dt or datetime.datetime.min

    def safe_load_image(self, path: Path) -> Image.Image | None:
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logging.error("Could not open %s: %s", path, e)
            return None

    def normalize_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        from fastai.vision.all import Normalize
        norm = None
        for tfm in getattr(self.model.dls.after_batch, 'fs', []):
            if isinstance(tfm, Normalize):
                norm = tfm
                break
        if norm:
            return norm(tensor)
        # fallback
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=self.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225],
                            device=self.device).view(1,3,1,1)
        return (tensor - mean) / std

    def run_inference_on_batches(self, batch_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Each item must carry:
          - tile_img (PIL)
          - image_title (str)
          - image_path  (str)
          - geometry    (dict with x,y,width,height)
        """
        results: List[Dict[str, Any]] = []

        for i in range(0, len(batch_items), BATCH_SIZE):
            sub = batch_items[i:i+BATCH_SIZE]

            # build tensor
            try:
                t = torch.stack([
                    torch.as_tensor(np.array(it["tile_img"]), dtype=torch.float32)
                         .permute(2,0,1)
                    for it in sub
                ]).to(self.device) / 255.0
            except Exception as e:
                logging.error("Batch prep failed: %s", e)
                continue

            t = self.normalize_batch(t)

            with torch.no_grad():
                logits = self.model.model(t)
                probs  = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

            for j, it in enumerate(sub):
                pvec = probs[j]
                idx  = int(pvec.argmax())
                cls  = self.model.dls.vocab[idx] if hasattr(self.model, 'dls') else str(idx)
                conf = float(pvec[idx])

                # here we inject your static [0,1] list
                class_ids = [0, 1]
                if cls.lower().strip() == "faulty" or "fault":
                    class_ids = 0
                else:
                    class_ids = 1

                results.append({
                    "image_title": it["image_title"],
                    "image_path":  it["image_path"],
                    "class_id":    class_ids,
                    "class_name":  cls,
                    "confidence":  round(conf, 4),
                    "geometry":    it["geometry"]
                })

        return results

    def process_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Group all tiles under each image as requested."""
        paths = [Path(p) for p in image_paths if Path(p).suffix.lower() in VALID_EXT]
        paths.sort(key=self.numeric_sort_key)

        # load originals
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as exe:
            originals = list(exe.map(self.safe_load_image, paths))

        # build every tile, with its metadata
        batch_items = []
        for path, img in zip(paths, originals):
            if img is None or img.height != COMPOSITE_HEIGHT:
                continue
            across = img.width // COMPOSITE_HEIGHT
            for ti in range(across):
                x0 = ti * COMPOSITE_HEIGHT
                geom = {"x": x0, "y": 0,
                        "width": COMPOSITE_HEIGHT,
                        "height": COMPOSITE_HEIGHT}
                crop = img.crop((x0,0,x0+COMPOSITE_HEIGHT,COMPOSITE_HEIGHT))
                crop = crop.resize((INFERENCE_INPUT_SIZE, INFERENCE_INPUT_SIZE),
                                   Image.Resampling.LANCZOS)

                batch_items.append({
                    "tile_img":    crop,
                    "image_title": path.stem,
                    "image_path":  str(path),
                    "geometry":    geom,
                    "tile_idx":    ti
                })

        # infer
        tile_results = self.run_inference_on_batches(batch_items)
        
        # now nest by image
        by_image: Dict[str, Dict[str, Any]] = {}
        for res, meta in zip(tile_results, batch_items):
            img_key = res["image_title"]
            tile_key = f"tile_{meta['tile_idx']}"
            if img_key not in by_image:
                by_image[img_key] = {
                    "image_title": img_key,
                    "image_path":  res["image_path"],
                    "tiles": {}
                }
            by_image[img_key]["tiles"][tile_key] = {
                "class_id":   res["class_id"],
                "class_name": res["class_name"],
                "confidence": res["confidence"],
                "geometry":   res["geometry"],
                "severity": 1
            }
                            

        #print(by_image.values)
        # return as a list
        return list(by_image.values())

    def save_results_to_json(self, results: List[Dict[str, Any]],
                             output_file: str = "fault_results.json") -> None:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logging.info("Wrote %d image records to %s", len(results), output_file)

# if __name__ == "__main__":
#     model = Model()
#     model.load_models()
    
#     image_paths = [
#         "test/_2025-04-18 10-18-27-045_7648 seam - Copy (3).jpg",
#         "test/_2025-04-18 10-54-23-801_22246 slub - Copy.jpg"
#     ]
    
#     structured = model.process_images(image_paths)
#     model.save_results_to_json(structured, "fault_results.json")
#     print(json.dumps(structured, indent=2))
