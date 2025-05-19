from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
from difflib import SequenceMatcher
import re
import cv2
from functools import lru_cache
import gc
from pyzbar.pyzbar import decode as pyzbar_decode

# Initialize PaddleOCR (using angle classification for extra robustness).
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, enable_mkldnn=True)
QR_DETECTOR = cv2.QRCodeDetector()

CONFIG = {
    'min_text_length': 3,
    'min_confidence': 0.70,
    'similarity_threshold': 0.7,
    'angles': [0, 90, 180, 270],  # if not skipping rotation, try these angles.
    'show_preview': True,
    'preview_size': (400, 400),
    'qr_scan_attempts': 2
}

def text_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()

@lru_cache(maxsize=None)
def cached_text_similarity(a, b):
    return text_similarity(a, b)

def merge_similar_texts(text_data):
    merged = []
    sorted_data = sorted(text_data, key=lambda x: (-len(x['original']), -x['confidence']))
    for entry in sorted_data:
        orig = entry['original']
        matched = False
        for i, existing in enumerate(merged):
            if cached_text_similarity(orig, existing['original']) > CONFIG['similarity_threshold']:
                if (len(orig) > len(existing['original']) or
                    (len(orig) == len(existing['original']) and entry['confidence'] > existing['confidence'])):
                    merged[i] = entry
                matched = True
                break
        if not matched:
            merged.append(entry)
    return {e['original']: e for e in merged}

def detect_qr_opencv(gray_img):
    data, _, _ = QR_DETECTOR.detectAndDecode(gray_img)
    return data.strip() if data and data.strip() else None

def detect_qr_pyzbar(gray_img):
    decoded_objects = pyzbar_decode(gray_img)
    for obj in decoded_objects:
        if obj.type == 'QRCODE':
            return obj.data.decode('utf-8').strip()
    return None

def detect_qr_code(image_cv):
    if len(image_cv.shape) == 2:
        gray = image_cv
    else:
        gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    
    # Attempt 1: use the original image.
    qr_data = detect_qr_opencv(gray)
    if qr_data:
        return qr_data
    
    # Additional processing attempts.
    processed = [
        gray,
        cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray),
        cv2.medianBlur(gray, 3)
    ]
    for img in processed[:CONFIG['qr_scan_attempts']]:
        qr_data = detect_qr_opencv(img) or detect_qr_pyzbar(img)
        if qr_data:
            return qr_data
    return None

def ocr_paddleocr(img_file, skip_rotation=False):
    """
    Performs OCR on the input image.
      - Converts the image to grayscale and applies unsharp masking.
      - If skip_rotation is True, only the original orientation is processed.
      - Otherwise, the image is rotated at 0°, 90°, 180° and 270°,
        and OCR is applied to each orientation.
      - After OCR, similar texts are merged and the recognized text is assembled.
      - Additionally, the code attempts to extract a numeric barcode (or returns QR code data if present).
    """
    try:
        pil_img = Image.fromarray(img_file).convert('L')
        img_cv_original = np.array(pil_img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return {"Plain_text": None, "Barcode_Number": None}
    
    # Check for a QR code on the original image.
    qr_data = detect_qr_code(img_cv_original)
    if qr_data:
        print(f"QR code found in original image: {qr_data}")
    
    # Enhance image using unsharp masking.
    blurred = cv2.GaussianBlur(img_cv_original, (0, 0), 1.5)
    unsharp = cv2.addWeighted(img_cv_original, 1.5, blurred, -0.5, 0)
    enhanced = cv2.convertScaleAbs(unsharp)
    
    pil_img = Image.fromarray(enhanced)
    
    # Choose orientations based on the skip_rotation flag.
    if skip_rotation:
        rotated_imgs = [pil_img]
        angles_used = [0]
    else:
        rotated_imgs = [pil_img.rotate(a, expand=True) for a in CONFIG['angles']]
        angles_used = CONFIG['angles']
    
    all_texts = []
    for idx, rimg in enumerate(rotated_imgs):
        rotated_cv = np.array(rimg)
        # If no QR code was found already, check this rotated version.
        if not qr_data:
            found_qr = detect_qr_code(rotated_cv)
            if found_qr:
                qr_data = found_qr
                print(f"QR code found at rotation {angles_used[idx]}°: {qr_data}")
        try:
            result = ocr.ocr(rotated_cv, cls=True)
        except Exception as e:
            print(f"OCR error at angle {angles_used[idx]}: {str(e)}")
            continue
        
        if not result:
            continue
        
        for line in result:
            if not line:
                continue
            for word_info in line:
                if word_info and len(word_info) >= 2:
                    text, confidence = word_info[1]
                    if len(text) >= CONFIG['min_text_length'] and confidence >= CONFIG['min_confidence']:
                        all_texts.append({
                            'original': text,
                            'confidence': confidence
                        })
    
    if not all_texts:
        print("No text found in any rotation.")
        return {"Plain_text": None, "Barcode_Number": qr_data}
    
    merged = merge_similar_texts(all_texts)
    final_out = sorted(merged.values(), key=lambda x: (-x['confidence'], -len(x['original'])))
    plain_text = " ".join([d['original'] for d in final_out])
    
    # Attempt to extract numeric barcode if possible.
    barcode_regex = re.compile(r"(\d{8,}(/?\d+)*)")
    extracted_barcode = None
    for r in final_out:
        t = r['original']
        no_spaces = "".join(t.split())
        no_slash = no_spaces.replace("/", "")
        if barcode_regex.fullmatch(no_slash):
            extracted_barcode = no_spaces
            if qr_data:
                extracted_barcode = qr_data
            break
    if not extracted_barcode and qr_data:
        extracted_barcode = qr_data
    
    gc.collect()
    return {
        "Plain_text": plain_text if plain_text else None,
        "Barcode_Number": extracted_barcode
    }
