import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import config
    from . import utils
except ImportError:
    import config
    import utils


def load_datasets():
    if not os.path.exists(config.DATASET_DIR) or not os.listdir(config.DATASET_DIR):
        print(f"Error: Dataset directory '{config.DATASET_DIR}' is empty or not found.")
        print("Please ensure the dataset is available and 'config.DATASET_DIR' is correct.")
        return None, None, [], 0

    try:
        inferred_class_names = sorted([d for d in os.listdir(config.DATASET_DIR) if os.path.isdir(os.path.join(config.DATASET_DIR, d))])
        if not inferred_class_names:
            print(f"Error: No subdirectories (classes) found in '{config.DATASET_DIR}'.")
            return None, None, [], 0
        
        num_classes = len(inferred_class_names)
        config.set_class_info(inferred_class_names, num_classes)
        print(f"Data Loader: Found {config.get_num_classes()} classes: {config.get_class_names()}")

        train_ds = tf.keras.utils.image_dataset_from_directory(
            config.DATASET_DIR,
            validation_split=config.VALIDATION_SPLIT_RATIO,
            subset="training",
            seed=config.RANDOM_SEED_SPLIT,
            image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
            batch_size=config.BATCH_SIZE,
            label_mode='categorical',
            class_names=config.get_class_names()
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            config.DATASET_DIR,
            validation_split=config.VALIDATION_SPLIT_RATIO,
            subset="validation",
            seed=config.RANDOM_SEED_SPLIT,
            image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
            batch_size=config.BATCH_SIZE,
            label_mode='categorical',
            class_names=config.get_class_names()
        )

        if train_ds.class_names != config.get_class_names():
            print(f"CRITICAL WARNING: Keras class name order {train_ds.class_names} "
                  f"differs from config's {config.get_class_names()}. This should not happen if class_names arg was passed.")
            config.set_class_info(train_ds.class_names, len(train_ds.class_names))

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000, seed=config.RANDOM_SEED_GLOBAL, reshuffle_each_iteration=True).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        print("Training and validation datasets loaded and preprocessed.")
        return train_ds, val_ds, config.get_class_names(), config.get_num_classes()

    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        import traceback
        traceback.print_exc()
        return None, None, [], 0

def visualize_sample_data(dataset, class_names_list, num_images=9):
    if dataset is None:
        print("Dataset is None, cannot visualize.")
        return
    if not class_names_list:
        print("Class names list is empty, cannot map labels to names for visualization.")

    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(min(num_images, images.shape[0])):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            label_index = np.argmax(labels[i])
            
            title_text = f"Raw Label Idx: {label_index}"
            if class_names_list and 0 <= label_index < len(class_names_list):
                 title_text = class_names_list[label_index]
            
            plt.title(title_text)
            plt.axis("off")
    plt.suptitle("Sample Images from Dataset")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == '__main__':
    print("--- Testing data_loader.py ---")
    try:
        utils.set_seeds()
    except NameError:
        print("Warning: utils.set_seeds() could not be called. Ensure utils.py is accessible.")
        np.random.seed(config.RANDOM_SEED_GLOBAL)
        tf.random.set_seed(config.RANDOM_SEED_GLOBAL)

    train_dataset, val_dataset, class_n_loaded, num_c_loaded = load_datasets()

    if train_dataset and val_dataset:
        print(f"Successfully loaded data. Number of classes from loader: {num_c_loaded}, Class names from loader: {class_n_loaded}")
        print(f"Train dataset element spec: {train_dataset.element_spec}")
        print(f"Validation dataset element spec: {val_dataset.element_spec}")
        if class_n_loaded:
            visualize_sample_data(train_dataset, class_n_loaded)
        else:
            print("Cannot visualize sample data because class names were not loaded.")
    else:
        print("Failed to load datasets during standalone test.")