from __future__ import annotations
import os, cv2, torch, json, glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ultralytics import YOLO
import OptimizePaddleOCRFunc_v2 as OCRFunc   # paddle-ocr helper

# ── tweak or turn off these if you don’t want crops on disk ────────────────
_OUTPUT_DIR    = Path("qr_crops")
_STITCHED_DIR  = Path("stitched")
_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
_STITCHED_DIR.mkdir(exist_ok=True, parents=True)
# ───────────────────────────────────────────────────────────────────────────

# ── constants controlling the algorithm ────────────────────────────────────
_EDGE_THRESH        = 5              # px from edge counts as “half sticker”
_MAX_ENHANCE_ROUNDS = 2              # CLAHE + sharpen retries
_OCR_EXPAND_W       = 2.0            # OCR crop width  multiplier
_OCR_EXPAND_H       = 3.0            # OCR crop height multiplier
_DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
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
    h, w = img.shape[:2]
    M    = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def _try_qr(crop: "cv2.Mat", wechat) -> Tuple[Optional[str], Optional[int], int]:
    for ang in (0, 90, 180, 270):
        txt, _ = wechat.detectAndDecode(_rotate(crop, ang))
        if txt and txt[0]:
            return txt[0], ang, 0
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
    return (int(max(0, cx - hw)),  int(max(0, cy - hh)),
            int(min(max_w, cx + hw)), int(min(max_h, cy + hh)))


def _analyse_one(img_path: str, model: YOLO, wechat) -> Dict[str, Any]:
    img = cv2.imread(img_path)
    if img is None:
        return {"image_path": img_path, "detection": {"best_box": None},
                "fast_qr": None, "ocr": None}

    H, W  = img.shape[:2]
    boxes = (model(img_path)[0].boxes.xyxy.cpu().numpy()
             .astype(float).tolist())
    best = max((b for b in boxes if b[2] > b[0] and b[3] > b[1]),
               key=lambda b: (b[2]-b[0])*(b[3]-b[1]), default=None)

    if best is None:
        return {"image_path": img_path, "detection": {"best_box": None},
                "fast_qr": None, "ocr": None}

    x1, y1, x2, y2 = map(int, best)
    crop           = img[y1:y2, x1:x2]
    qr_text, _, _  = _try_qr(crop, wechat)

    ex1, ey1, ex2, ey2 = _expand_box(x1, y1, x2, y2, W, H)
    ocr_crop           = img[ey1:ey2, ex1:ex2]
    ocr_dict = OCRFunc.ocr_paddleocr(
        ocr_crop, skip_rotation=(qr_text is not None)
    ) or {}

    if qr_text:
        cv2.imwrite(_OUTPUT_DIR / f"{Path(img_path).stem}_qr.png", crop)
    elif ocr_dict:
        cv2.imwrite(_OUTPUT_DIR / f"{Path(img_path).stem}_ocr.png", ocr_crop)

    det = {
        "name": "BarCode", "class": 0, "confidence": None,
        "box":      {"x1": float(x1), "y1": float(y1),
                      "x2": float(x2), "y2": float(y2)},
        "best_box": {"x1": float(x1), "y1": float(y1),
                      "x2": float(x2), "y2": float(y2)}
    }

    return {
        "image_path": img_path,
        "detection":  det,
        "fast_qr":    qr_text,
        "ocr":        ocr_dict if ocr_dict else
                       {"Plain_text": None, "Barcode_Number": None}
    }


def process_images(
    image_paths: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    if not image_paths:
        return {"qr_Res": []}

    model = YOLO("best.pt").to(_DEVICE)
    wechat = cv2.wechat_qrcode_WeChatQRCode()

    results: List[Dict[str, Any]] = []
    prev_bottom: Optional[Dict[str, Any]] = None

    for idx, img_path in enumerate(image_paths):
        entry = _analyse_one(img_path, model, wechat)
        results.append(entry)

        img = cv2.imread(img_path)
        if img is None:
            continue
        H = img.shape[0]
        boxes = (model(img_path)[0].boxes.xyxy.cpu().numpy()
                 .astype(int).tolist())

        tops, bots = [], []
        for b in boxes:
            if b[1] <= _EDGE_THRESH:
                tops.append({"img": img_path, "idx": idx, "box": b})
            elif b[3] >= H - _EDGE_THRESH:
                bots.append({"img": img_path, "idx": idx, "box": b})

        # perform stitching if previous bottom and current top overlap
        if prev_bottom and tops:
            first_idx = prev_bottom["idx"]
            xb1, _, xb2, _ = prev_bottom["box"]
            for t in tops:
                xt1, _, xt2, _ = t["box"]
                if min(xb2, xt2) - max(xb1, xt1) > 0.5 * min(xb2-xb1, xt2-xt1):
                    top_path, bot_path = prev_bottom["img"], img_path
                    it = cv2.imread(top_path)
                    ib = cv2.imread(bot_path)
                    if it is not None and ib is not None:
                        tw = max(it.shape[1], ib.shape[1])
                        stitched = cv2.vconcat([
                            _pad_to_width(it, tw), _pad_to_width(ib, tw)
                        ])
                        stitched_file = _STITCHED_DIR / f"stitch_{first_idx}_{idx}.png"
                        cv2.imwrite(stitched_file.as_posix(), stitched)
                        # analyse stitched and overwrite original entry
                        stitch_result = _analyse_one(
                            stitched_file.as_posix(), model, wechat
                        )
                        # overwrite detection, fast_qr, ocr of original
                        results[first_idx]["detection"] = stitch_result["detection"]
                        results[first_idx]["fast_qr"]    = stitch_result["fast_qr"]
                        results[first_idx]["ocr"]        = stitch_result["ocr"]
                    prev_bottom = None
                    break
        else:
            prev_bottom = max(
                bots, key=lambda d: (d["box"][2]-d["box"][0]) * (d["box"][3]-d["box"][1]),
                default=None
            )

    return {"qr_Res": results}
