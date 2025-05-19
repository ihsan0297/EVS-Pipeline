"""Fast & Robust QR / Barcode Pipeline
================================================
Features
--------
* **Single-batch YOLOv8 detection** on GPU (RTX-30-series tuned)
* **Optional FP16** – tries half-precision first, *auto-recovers* to FP32 on the rare type-mismatch error you hit
* **Zero-cost OCR skip** when WeChat‐QR already decodes the string
* **ThreadPool** crunches CPU-heavy QR/OCR while the GPU is busy
* **Stitching** of multi-photo labels (top/bottom halves)

Usage
~~~~~
```bash
python fast_qr_processing.py               # processes test_images/*.jpg
python -m fast_qr_processing img1.jpg ...   # or pass file list
```
Tune `_BATCH_SIZE` for VRAM; 16 is comfy on a 12 GB RTX 3060.
"""

from __future__ import annotations
import os, cv2, torch, json, glob, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from ultralytics import YOLO
import OptimizePaddleOCRFunc_v2 as OCRFunc  # paddle-ocr helper

# ───────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────
_OUTPUT_DIR        = Path("qr_crops")
_STITCHED_DIR      = Path("stitched")
_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
_STITCHED_DIR.mkdir(exist_ok=True, parents=True)

_EDGE_THRESH        = 100      # px from edge counts as “half sticker”
_MAX_ENHANCE_ROUNDS = 2        # CLAHE + sharpen retries
_OCR_EXPAND_W       = 2.0      # OCR crop width  multiplier
_OCR_EXPAND_H       = 5.0      # OCR crop height multiplier

_DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"
_BATCH_SIZE  = 16             # adjust for VRAM
_USE_HALF    = torch.cuda.is_available()  # try FP16 only on CUDA

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # autotune kernels
    torch.set_float32_matmul_precision("high")

# ───────────────────────────────────────────────────────────────────────────
# Image-processing helpers
# ───────────────────────────────────────────────────────────────────────────

def _enhance_once(img: "cv2.Mat") -> "cv2.Mat":
    up = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    g  = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    g  = cv2.createCLAHE(2.5, (8, 8)).apply(g)
    sh = cv2.addWeighted(g, 1.7, cv2.GaussianBlur(g, (0, 0), 3), -0.7, 0)
    return cv2.cvtColor(sh, cv2.COLOR_GRAY2BGR)

def _rotate(img: "cv2.Mat", angle: int) -> "cv2.Mat":
    if angle == 0:   return img
    if angle == 90:  return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    (h, w) = img.shape[:2]
    M      = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def _try_qr(crop: "cv2.Mat", wechat) -> Tuple[Optional[str], Optional[int], int]:
    # quick attempt on raw + 3 rotations
    for ang in (0, 90, 180, 270):
        txt, _ = wechat.detectAndDecode(_rotate(crop, ang))
        if txt and txt[0]:
            return txt[0], ang, 0
    # progressive enhancement
    for e in range(1, _MAX_ENHANCE_ROUNDS + 1):
        enh = _enhance_once(crop)
        for ang in (0, 90, 180, 270):
            txt, _ = wechat.detectAndDecode(_rotate(enh, ang))
            if txt and txt[0]:
                return txt[0], ang, e
    return None, None, _MAX_ENHANCE_ROUNDS

def _pad_to_width(img: "cv2.Mat", w_target: int) -> "cv2.Mat":
    h, w = img.shape[:2]
    pad   = w_target - w
    l, r  = pad // 2, pad - pad // 2
    return cv2.copyMakeBorder(img, 0, 0, l, r, cv2.BORDER_CONSTANT)

def _expand_box(x1, y1, x2, y2, max_w, max_h):
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return int(x1), int(y1), int(x2), int(y2)
    cx, cy = x1 + w/2, y1 + h/2
    hw, hh = w * _OCR_EXPAND_W / 2, h * _OCR_EXPAND_H / 2
    return (
        int(max(0, cx - hw)),  int(max(0, cy - hh)),
        int(min(max_w, cx + hw)), int(min(max_h, cy + hh))
    )

# ───────────────────────────────────────────────────────────────────────────
# Detection and analysis
# ───────────────────────────────────────────────────────────────────────────

def _detect_boxes(model: YOLO, sources: List[str]) -> List[List[List[int]]]:
    """Runs YOLO predict once; falls back to FP32 if half fails."""
    try:
        preds = model.predict(
            sources,
            device=_DEVICE,
            batch=_BATCH_SIZE,
            half=_USE_HALF,
            verbose=False,
            stream=False,
        )
    except RuntimeError as e:
        if "expected scalar type Half" in str(e):
            preds = model.predict(
                sources,
                device=_DEVICE,
                batch=_BATCH_SIZE,
                half=False,
                verbose=False,
                stream=False,
            )
        else:
            raise
    return [p.boxes.xyxy.cpu().numpy().astype(int).tolist() for p in preds]


def _analyse_one(img_path: str, boxes: List[List[int]], wechat) -> Dict[str, Any]:
    img = cv2.imread(img_path)
    if img is None:
        return {
            "image_path": img_path,
            "detection": {"best_box": None},
            "fast_qr": None,
            "ocr": None,
        }

    H, W  = img.shape[:2]
    best  = max(
        (b for b in boxes if b[2] > b[0] and b[3] > b[1]),
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
        default=None,
    )

    if best is None:
        return {
            "image_path": img_path,
            "detection": {"best_box": None},
            "fast_qr": None,
            "ocr": None,
        }

    x1, y1, x2, y2 = map(int, best)
    crop           = img[y1:y2, x1:x2]
    qr_text, _, _  = _try_qr(crop, wechat)

    if qr_text is None:
        ex1, ey1, ex2, ey2 = _expand_box(x1, y1, x2, y2, W, H)
        ocr_crop           = img[ey1:ey2, ex1:ex2]
        ocr_dict = OCRFunc.ocr_paddleocr(ocr_crop) or {}
    else:
        ocr_dict = {"Plain_text": None, "Barcode_Number": None}

    # save diagnostic crops (optional)
    if qr_text:
        cv2.imwrite(_OUTPUT_DIR / f"{Path(img_path).stem}_qr.png", crop)
    elif ocr_dict:
        cv2.imwrite(_OUTPUT_DIR / f"{Path(img_path).stem}_ocr.png", ocr_crop)

    det_struct = {
        "name": "BarCode",
        "class": 0,
        "confidence": None,
        "box":      {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
        "best_box": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
    }

    return {
        "image_path": img_path,
        "detection":  det_struct,
        "fast_qr":    qr_text,
        "ocr":        ocr_dict,
    }


# ───────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────

def process_images(image_paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    if not image_paths:
        return {"qr_Res": []}

    # build model (half handled inside predict)
    model = YOLO("best.pt").to(_DEVICE)

    wechat = cv2.wechat_qrcode_WeChatQRCode()

    # 1️⃣  Detect everything in one pass
    all_boxes = _detect_boxes(model, image_paths)

    # 2️⃣  Per-image QR/OCR in parallel threads
    results: List[Optional[Dict[str, Any]]] = [None] * len(image_paths)
    with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 1, 32)) as pool:
        fut_map = {
            pool.submit(_analyse_one, img, boxes, wechat): idx
            for idx, (img, boxes) in enumerate(zip(image_paths, all_boxes))
        }
        for fut in as_completed(fut_map):
            results[fut_map[fut]] = fut.result()

    # 3️⃣  Optional stitching of top/bottom halves
    prev_bottom: Optional[Dict[str, Any]] = None
    for idx, (img_path, boxes) in enumerate(zip(image_paths, all_boxes)):
        img = cv2.imread(img_path)
        if img is None:
            continue
        H = img.shape[0]

        tops, bots = [], []
        for b in boxes:
            if b[1] <= _EDGE_THRESH:
                tops.append({"img": img_path, "idx": idx, "box": b})
            elif b[3] >= H - _EDGE_THRESH:
                bots.append({"img": img_path, "idx": idx, "box": b})

        if prev_bottom and tops:
            first_idx = prev_bottom["idx"]
            xb1, _, xb2, _ = prev_bottom["box"]
            for t in tops:
                xt1, _, xt2, _ = t["box"]
                overlap = min(xb2, xt2) - max(xb1, xt1)
                if overlap > 0.5 * min(xb2 - xb1, xt2 - xt1):
                    top_path, bot_path = prev_bottom["img"], img_path
                    it, ib = cv2.imread(top_path), cv2.imread(bot_path)
                    if it is not None and ib is not None:
                        tw = max(it.shape[1], ib.shape[1])
                        stitched = cv2.vconcat([
                            _pad_to_width(it, tw), _pad_to_width(ib, tw)
                        ])
                        stitched_file = _STITCHED_DIR / f"stitch_{first_idx}_{idx}.png"
                        cv2.imwrite(stitched_file.as_posix(), stitched)
                        # fresh detection on stitched
                        stitch_boxes = _detect_boxes(model, [stitched_file.as_posix()])[0]
                        stitch_result = _analyse_one(stitched_file.as_posix(), stitch_boxes, wechat)
                        # overwrite original entry
                        results[first_idx]["detection"] = stitch_result["detection"]
                        results[first_idx]["fast_qr"]  = stitch_result["fast_qr"]
                        results[first_idx]["ocr"]      = stitch_result["ocr"]
                    prev_bottom = None
                    break
        else:
            prev_bottom = max(
                bots,
                key=lambda d: (d["box"][2]-d["box"][0])*(d["box"][3]-d["box"][1]),
                default=None
            )
    
    return {"qr_Res": results}


# if __name__ == "__main__":
#     imgs = sys.argv[1:] or glob.glob("test_images/*.jpg")
#     out  = process_images(imgs)
#     with open("results.json", "w", encoding="utf-8") as f:
#         json.dump(out, f, ensure_ascii=False, indent=4)
#     print("Saved results.json with", len(out["qr_Res"]), "entries.")
