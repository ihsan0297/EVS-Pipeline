from fastai.vision.all import load_learner, PILImage, Transform
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
class TimmRandAugmentTransform(Transform):
    def __init__(self, magnitude=6, num_layers=2):
        from timm.data.auto_augment import rand_augment_transform
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
            from timm.data.auto_augment import rand_augment_transform
            self.augment_fn = rand_augment_transform(
                config_str=f'rand-m{self.magnitude}-n{self.num_layers}',
                hparams={}
            )

    def encodes(self, img):
        return self.augment_fn(img)

import sys
class Model:
    def __init__(self):
        self.model1 = None
        self.model2 = None
        self.model1_path = "Models/SeamMiniV1.pkl"
        self.model2_path = "Models/SeamV1.pkl"
    def load_models(self):
        sys.modules['__main__.TimmRandAugmentTransform'] = TimmRandAugmentTransform

        self.model1 = load_learner(self.model1_path, cpu=True)
        self.model2 = load_learner(self.model2_path, cpu=True)      
        self.model1.dls.after_item.add(TimmRandAugmentTransform())
        self.model2.dls.after_item.add(TimmRandAugmentTransform())
        print("✅ Models loaded successfully with TimmRandAugmentTransform")

        

# 🔹 Paths Configuration
# IMAGE_FOLDER = "/home/ishali/airflow/test_images/processed_images"
# OUTPUT_FOLDER = "output/kafka_airflow_111"
# SEAM_FOLDER = os.path.join(OUTPUT_FOLDER, "seam_images_kafka_airflow")
# NORMAL_FOLDER = os.path.join(OUTPUT_FOLDER, "normal_images_kafka_airflow")
# JSON_FOLDER = os.path.join(OUTPUT_FOLDER, "json_kafka_airflow_1")
MODEL_1_PATH = "Models/SeamMiniV1.pkl"
MODEL_2_PATH = "Models/SeamV1.pkl"

# Ensure folders exist
# for folder in [SEAM_FOLDER, NORMAL_FOLDER, JSON_FOLDER]:
#     Path(folder).mkdir(parents=True, exist_ok=True)

# 🔹 Logging setup
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

def load_models():
    """Load both inference models with custom transforms."""
    model_1 = load_learner(MODEL_1_PATH, cpu=True)
    model_2 = load_learner(MODEL_2_PATH, cpu=True)

    # Register the custom transform in both models
    model_1.dls.after_item.add(TimmRandAugmentTransform())
    model_2.dls.after_item.add(TimmRandAugmentTransform())

    print("✅ Models loaded successfully with TimmRandAugmentTransform")
    return model_1, model_2

def perform_inference(model, image_path):
    """Perform inference on an image using the given model."""
    img = PILImage.create(image_path)
    pred, pred_idx, probs = model.predict(img)
    confidence = round(float(probs[pred_idx]) * 100, 2)
    return str(pred), confidence

# def process_images():
#     """Process all images in the consumer folder."""
#     model_1, model_2 = load_models()
#     images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
#     for img_file in images:
#         image_path = os.path.join(IMAGE_FOLDER, img_file)
#         pred_1, conf_1 = perform_inference(model_1, image_path)
        
#         if pred_1 == "normal":
#             shutil.move(image_path, os.path.join(NORMAL_FOLDER, img_file))
#             logger.info(f"✅ {img_file} classified as NORMAL, moved to {NORMAL_FOLDER}")
#         else:
#             pred_2, conf_2 = perform_inference(model_2, image_path)
#             shutil.move(image_path, os.path.join(SEAM_FOLDER, img_file))
            
#             result = {
#                 "image": img_file,
#                 "model1_prediction": pred_1,
#                 "model1_confidence": conf_1,
#                 "model2_prediction": pred_2,
#                 "model2_confidence": conf_2,
#                 "timestamp": datetime.utcnow().isoformat()
#             }
            
#             json_path = os.path.join(JSON_FOLDER, f"{Path(img_file).stem}.json")
#             with open(json_path, "w") as json_file:
#                 json.dump(result, json_file, indent=4)
            
#             logger.info(f"📄 JSON result saved for {img_file}")

# if __name__ == "__main__":
#     model=Model()
#     model.load_models()
