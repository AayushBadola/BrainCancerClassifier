import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'Brain_Cancer')
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
TEMP_PREDICTION_IMAGES_DIR = os.path.join(BASE_DIR, 'temp_prediction_images')
DEFAULT_MODEL_NAME = 'brain_cancer_classifier.keras'
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, DEFAULT_MODEL_NAME)

IMG_HEIGHT = 150
IMG_WIDTH = 150
IMG_CHANNELS = 3
BATCH_SIZE = 32
VALIDATION_SPLIT_RATIO = 0.2

EPOCHS = 25
LEARNING_RATE = 0.001
LOSS_FUNCTION = 'categorical_crossentropy'
METRICS = ['accuracy']

RANDOM_SEED_GLOBAL = 42
RANDOM_SEED_SPLIT = 123

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(TEMP_PREDICTION_IMAGES_DIR, exist_ok=True)

_CLASS_NAMES_FROM_LOADER = []
_NUM_CLASSES_FROM_LOADER = 0

def set_class_info(class_names_list, num_classes_val):
    global _CLASS_NAMES_FROM_LOADER, _NUM_CLASSES_FROM_LOADER
    _CLASS_NAMES_FROM_LOADER = list(class_names_list)
    _NUM_CLASSES_FROM_LOADER = int(num_classes_val)

def get_class_names():
    if not _CLASS_NAMES_FROM_LOADER:
        print("Warning: config.get_class_names() called before class names were set by data_loader.")
        try:
            if os.path.exists(DATASET_DIR):
                inferred_names = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
                if inferred_names:
                    set_class_info(inferred_names, len(inferred_names))
                    print(f"Fallback inference in config: Found classes {get_class_names()}")
                    return _CLASS_NAMES_FROM_LOADER
            return []
        except Exception as e:
            print(f"Error during fallback class name inference in config: {e}")
            return []
    return _CLASS_NAMES_FROM_LOADER

def get_num_classes():
    if _NUM_CLASSES_FROM_LOADER == 0:
        get_class_names()
        if not _CLASS_NAMES_FROM_LOADER and _NUM_CLASSES_FROM_LOADER == 0:
            print("Warning: config.get_num_classes() called before num_classes was set by data_loader and fallback failed.")
        return _NUM_CLASSES_FROM_LOADER
    return _NUM_CLASSES_FROM_LOADER

print(f"Config loaded. Base directory: {BASE_DIR}")
print(f"Dataset directory: {DATASET_DIR}")
