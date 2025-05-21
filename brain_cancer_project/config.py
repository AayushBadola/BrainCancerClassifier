import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, 'brain_cancer_cleaned')
TEST_DATA_DIR = os.path.join(BASE_DIR, 'testing_cleaned')

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

BATCH_SIZE = 32
VALIDATION_SPLIT_RATIO = 0.2

SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
TEMP_PREDICTION_IMAGES_DIR = os.path.join(BASE_DIR, 'temp_prediction_images')
DEFAULT_MODEL_NAME = 'brain_cancer_EfficientNetB0_FineTuned.keras'
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, DEFAULT_MODEL_NAME)

LEARNING_RATE_HEAD = 0.001
LEARNING_RATE_FINETUNE = 0.00001

EPOCHS_HEAD_TRAINING = 30
EPOCHS_FINETUNING = 50

NUM_LAYERS_TO_FINETUNE = 30

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
    global _CLASS_NAMES_FROM_LOADER, _NUM_CLASSES_FROM_LOADER
    if not _CLASS_NAMES_FROM_LOADER:
        data_source_dir_for_class_inference = DATASET_DIR 
        if not os.path.exists(data_source_dir_for_class_inference) and os.path.exists(TEST_DATA_DIR):
            data_source_dir_for_class_inference = TEST_DATA_DIR
        try:
            if os.path.exists(data_source_dir_for_class_inference):
                inferred_names = sorted([d for d in os.listdir(data_source_dir_for_class_inference) if os.path.isdir(os.path.join(data_source_dir_for_class_inference, d))])
                if inferred_names:
                    _CLASS_NAMES_FROM_LOADER = list(inferred_names)
                    _NUM_CLASSES_FROM_LOADER = len(inferred_names)
                    return _CLASS_NAMES_FROM_LOADER
            return []
        except Exception as e:
            return []
    return _CLASS_NAMES_FROM_LOADER

def get_num_classes():
    global _CLASS_NAMES_FROM_LOADER, _NUM_CLASSES_FROM_LOADER
    if _NUM_CLASSES_FROM_LOADER == 0:
        get_class_names()
        if not _CLASS_NAMES_FROM_LOADER and _NUM_CLASSES_FROM_LOADER == 0:
            print("Warning: config.get_num_classes() called before num_classes was set and fallback failed.")
        return _NUM_CLASSES_FROM_LOADER
    return _NUM_CLASSES_FROM_LOADER

print(f"Config loaded. Base directory: {BASE_DIR}")
print(f"TRAINING/VALIDATION data source (DATASET_DIR): {DATASET_DIR}")
print(f"DEDICATED TEST data source (TEST_DATA_DIR): {TEST_DATA_DIR}")
print(f"MODEL INPUT IMAGE SIZE: {IMG_HEIGHT}x{IMG_WIDTH}")
