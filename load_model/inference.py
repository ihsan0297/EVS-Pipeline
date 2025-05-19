from fastai.vision.all import PILImage

def perform_inference(model, image_path: str):
    try:
        # Load the image
        img = PILImage.create(image_path)

        # Perform inference
        pred, pred_idx, probs = model.predict(img)

        # Return results
        return {
            "prediction": pred,
            "confidence": probs[pred_idx].item(),
        }
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Image file not found at '{image_path}'.")
    except Exception as e:
        raise RuntimeError(f"An error occurred during inference: {e}")
