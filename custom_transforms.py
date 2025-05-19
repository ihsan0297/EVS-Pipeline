from fastai.vision.all import Transform
from timm.data.auto_augment import rand_augment_transform

class TimmRandAugmentTransform(Transform):
    def __init__(self, magnitude=6, num_layers=2):
        self.magnitude = magnitude
        self.num_layers = num_layers
        self.augment_fn = rand_augment_transform(
            config_str=f'rand-m{magnitude}-n{num_layers}',
            hparams={}
        )

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'magnitude' not in state:
            self.magnitude = 6
        if 'num_layers' not in state:
            self.num_layers = 2
        if not hasattr(self, 'augment_fn') or self.augment_fn is None:
            self.augment_fn = rand_augment_transform(
                config_str=f'rand-m{self.magnitude}-n{self.num_layers}',
                hparams={}
            )

    def encodes(self, img):
        return self.augment_fn(img)
