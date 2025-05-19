import json
import re
from difflib import SequenceMatcher
from typing import List, Dict, Union

def load_json(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_numeric_runs(text: str) -> List[str]:
    return re.findall(r'[\d/]+', text or '')

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def find_best_matches(
    target: str,
    known: List[str],
    top_n: int = 2
) -> List[Dict[str, Union[str, float]]]:
    scores = [(code, similarity(target, code)) for code in known]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [
        {"match": code, "score": round(score, 3)}
        for code, score in scores[:top_n]
        if score > 0
    ]

def annotate_with_similarity(
    payload: Union[str, Dict],
    known_barcodes: List[str],
    top_n: int = 2
) -> Dict:
    # load JSON if given a filename
    data = load_json(payload) if isinstance(payload, str) else payload

    for rec in data.get("qr_Res", []):
        # determine the “perfect” value if any
        perfect = rec.get("fast_qr") or rec.get("ocr", {}).get("Barcode_Number")

        if perfect:
            matches = [{"match": perfect, "score": 1.0}]
        else:
            plain = rec.get("ocr", {}).get("Plain_text", "")
            runs = extract_numeric_runs(plain)
            if runs:
                candidate = max(runs, key=len)
                matches = find_best_matches(candidate, known_barcodes, top_n)
            else:
                matches = []

        # embed directly
        rec["similarity_test"] = {
            "detected": perfect,
            "matches": matches
        }

    return data

# if __name__ == "__main__":
#     known = [
#       "0702251212/740",
#       "0702251212/715",
#       "0702251212/1041",
#       "0702251212/730",
#       "0702251212/25",
#       # …etc
#     ]

#     annotated = annotate_with_similarity("test.json", known, top_n=2)

#     # write the augmented JSON back out
#     with open("test_with_similarity_in_qrRes.json", "w", encoding="utf-8") as f:
#         json.dump(annotated, f, indent=2, ensure_ascii=False)

#     # (optional) print to console
#     import pprint
#     pprint.pprint(annotated)
