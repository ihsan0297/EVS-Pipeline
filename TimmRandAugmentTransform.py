from fastai.vision.all import Transform, PILImage
from timm.data import rand_augment_transform
from PIL import Image
import numpy as np
import cv2

class TimmRandAugmentTransform(Transform):
    def __init__(self, config_str='rand-m9-mstd0.5-inc1'):
        super().__init__()
        self.randaug = rand_augment_transform(config_str=config_str, hparams={})

    def encodes(self, img: PILImage):
        img_array = np.array(img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        try:
            pil_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            aug_img = self.randaug(pil_img)  # Apply RandAugment
            if aug_img is None or not isinstance(aug_img, Image.Image):
                raise ValueError("RandAugment returned invalid image.")

            aug_img_array = np.array(aug_img)
        except Exception as e:
            print(f"Warning: RandAugment failed: {e}")
            return PILImage.create(img)
        return PILImage.create(aug_img_array)
