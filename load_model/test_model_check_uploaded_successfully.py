from TimmRandAugmentTransform import TimmRandAugmentTransform
import pickle
from pathlib import Path

MODEL_PATH_1 = Path(r"c:\Users\Ali\Desktop\Models\SeamMiniV1.pkl")
MODEL_PATH_2 = Path(r"c:\Users\Ali\Desktop\Models\SeamV1.pkl")

with open(MODEL_PATH_1, 'rb') as f1, open(MODEL_PATH_2, 'rb') as f2:
    model_1 = pickle.load(f1)
    model_2 = pickle.load(f2)
print("Models loaded successfully.")
