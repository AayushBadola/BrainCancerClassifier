import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil

try:
    from . import config
    from . import utils
    from . import data_loader
except ImportError:
    import config
    import utils
    import data_loader

_model_cache = None
_class_names_cache = []

def load_prediction_model_and_classes(model_path=None):
    global _model_cache, _class_names_cache

    if _model_cache is not None and _class_names_cache:
        return _model_cache, _class_names_cache

    if model_path is None:
        model_path = config.MODEL_PATH

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None, []

    try:
        print(f"Loading model from: {model_path} for prediction...")
        _model_cache = tf.keras.models.load_model(model_path, compile=False)

        _class_names_cache = config.get_class_names()

        if not _class_names_cache:
            print("Class names not in config, attempting fallback via data_loader...")
            if 'data_loader' in globals() and hasattr(data_loader, 'load_datasets'):
                data_loader.load_datasets()
                _class_names_cache = config.get_class_names()
            else:
                print("data_loader module not available for class name fallback.")

        if not _class_names_cache and _model_cache:
            print("Still no class names, inferring generic names from model output layer.")
            try:
                num_model_outputs = _model_cache.output_shape[-1]
                _class_names_cache = [f"Class_{i}" for i in range(num_model_outputs)]
            except Exception as e_generic:
                print(f"Could not infer generic class names from model: {e_generic}")
                _class_names_cache = []

        if _model_cache and _class_names_cache:
            print(f"Model loaded. Using class names: {_class_names_cache}")
        elif not _model_cache:
            print("Failed to load model.")
        elif not _class_names_cache:
            print("Model loaded, but failed to determine class names.")

        return _model_cache, _class_names_cache

    except Exception as e:
        print(f"Error loading model or class names: {e}")
        import traceback
        traceback.print_exc()
        _model_cache = None
        _class_names_cache = []
        return None, []

def predict_single_image(image_path, display_image=True):
    if 'utils' in globals() and hasattr(utils, 'set_seeds'):
        utils.set_seeds()

    model_to_predict, class_names_for_pred = load_prediction_model_and_classes()
    if not model_to_predict or not class_names_for_pred:
        print("Prediction cannot proceed: model or class names not available.")
        return None, None, None

    img_array = utils.preprocess_image_for_prediction(
        image_path, config.IMG_HEIGHT, config.IMG_WIDTH
    )
    if img_array is None:
        return None, None, None

    try:
        raw_predictions = model_to_predict.predict(img_array)
        scores = raw_predictions[0]

        predicted_class_index = np.argmax(scores)
        confidence = np.max(scores) * 100

        predicted_class_label = class_names_for_pred[predicted_class_index] \
            if 0 <= predicted_class_index < len(class_names_for_pred) \
            else f"Unknown Index {predicted_class_index}"

        print(f"\n--- Prediction for {os.path.basename(image_path)} ---")
        print(f"Predicted class: {predicted_class_label}")
        print(f"Confidence: {confidence:.2f}%")
        print("All scores (probabilities):")
        for i, score_val in enumerate(scores):
            c_name = class_names_for_pred[i] if i < len(class_names_for_pred) else f"Class_{i}"
            print(f"  - {c_name}: {score_val*100:.2f}%")

        if display_image:
            try:
                img_display = tf.keras.utils.load_img(image_path)
                plt.figure(figsize=(6, 6))
                plt.imshow(img_display)
                plt.title(f"Predicted: {predicted_class_label} ({confidence:.2f}%)")
                plt.axis("off")
                plt.show()
            except Exception as e_disp:
                print(f"Error displaying image: {e_disp}")

        return predicted_class_label, confidence, raw_predictions
    except Exception as e_pred_err:
        print(f"Error during prediction: {e_pred_err}")
        return None, None, None

def predict_on_sample_from_dataset(dataset_dir=None, class_to_sample=None):
    if dataset_dir is None:
        dataset_dir = config.DATASET_DIR
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} not found for sampling.")
        return

    available_classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if not available_classes:
        print(f"No class subdirectories in {dataset_dir}.")
        return

    chosen_class_name = class_to_sample
    if chosen_class_name:
        if chosen_class_name not in available_classes:
            print(f"Class '{chosen_class_name}' not found. Available: {available_classes}. Picking random.")
            chosen_class_name = None
    if not chosen_class_name:
        chosen_class_name = np.random.choice(available_classes)
        print(f"Randomly selected class '{chosen_class_name}' for sampling.")

    chosen_class_dir = os.path.join(dataset_dir, chosen_class_name)
    img_files = [f for f in os.listdir(chosen_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not img_files:
        print(f"No images in {chosen_class_dir}.")
        return

    random_image_name = np.random.choice(img_files)
    original_image_path = os.path.join(chosen_class_dir, random_image_name)

    os.makedirs(config.TEMP_PREDICTION_IMAGES_DIR, exist_ok=True)
    temp_image_name = f"sampled_{chosen_class_name}_{random_image_name}"
    temp_image_path = os.path.join(config.TEMP_PREDICTION_IMAGES_DIR, temp_image_name)

    try:
        shutil.copy(original_image_path, temp_image_path)
        print(f"\n--- Predicting on a sample image ---")
        print(f"True Class: {chosen_class_name} (from {original_image_path})")
        print(f"Predicting on copy: {temp_image_path}")
        predict_single_image(temp_image_path, display_image=True)
    except Exception as e_sample:
        print(f"Error during sample prediction: {e_sample}")

if __name__ == '__main__':
    print("--- Testing Prediction ---")
    try:
        if 'utils' in globals() and hasattr(utils, 'set_seeds'):
            utils.set_seeds()
    except Exception:
        pass

    print("\nAttempting to predict on a sample from the dataset...")
    predict_on_sample_from_dataset()
