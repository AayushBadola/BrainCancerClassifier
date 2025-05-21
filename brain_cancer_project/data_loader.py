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
        print(f"Error: Dataset directory '{config.DATASET_DIR}' (expected to be cleaned data) is empty or not found.")
        return None, None, [], 0

    try:
        inferred_class_names = sorted([d for d in os.listdir(config.DATASET_DIR) if os.path.isdir(os.path.join(config.DATASET_DIR, d))])
        if not inferred_class_names:
            print(f"Error: No subdirectories (classes) found in '{config.DATASET_DIR}'.")
            return None, None, [], 0
        
        num_classes = len(inferred_class_names)
        config.set_class_info(inferred_class_names, num_classes)
        print(f"Data Loader: Found {config.get_num_classes()} classes: {config.get_class_names()} from {config.DATASET_DIR}")

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

        if hasattr(train_ds, 'class_names') and train_ds.class_names != config.get_class_names():
            print(f"CRITICAL WARNING: Keras class name order {train_ds.class_names} "
                  f"differs from config's {config.get_class_names()}.")
            print(f"Re-setting class info based on actual training data source: {config.DATASET_DIR}")
            config.set_class_info(train_ds.class_names, len(train_ds.class_names))

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000, seed=config.RANDOM_SEED_GLOBAL, reshuffle_each_iteration=True).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        print(f"Training and validation datasets (from cleaned data) loaded and preprocessed from: {config.DATASET_DIR}")
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
        class_names_list = config.get_class_names()
        if not class_names_list:
            print("Class names list is empty, cannot map labels to names for visualization.")

    plt.figure(figsize=(10, 10))
    try:
        for images, labels in dataset.take(1):
            for i in range(min(num_images, images.shape[0])):
                ax = plt.subplot(3, 3, i + 1)
                display_image = images[i].numpy().astype("uint8")
                plt.imshow(display_image)
                label_index = np.argmax(labels[i])
                title_text = f"Label Idx: {label_index}"
                if class_names_list and 0 <= label_index < len(class_names_list):
                     title_text = class_names_list[label_index]
                plt.title(title_text)
                plt.axis("off")
        plt.suptitle("Sample Images from Dataset")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    except tf.errors.OutOfRangeError:
        print("Info: Reached end of dataset while trying to visualize samples (this is often normal for .take(1)).")
    except Exception as e:
        print(f"Error during data visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("--- Testing data_loader.py ---")
    try:
        if 'utils' in globals() and hasattr(utils, 'set_seeds'): utils.set_seeds()
    except NameError: pass

    train_dataset, val_dataset, class_n_loaded, num_c_loaded = load_datasets()
    if train_dataset and val_dataset:
        print(f"Successfully loaded data. Num classes: {num_c_loaded}, Names: {class_n_loaded}")
        visualize_sample_data(train_dataset, class_n_loaded)
