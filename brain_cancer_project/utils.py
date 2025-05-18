import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from . import config

def plot_training_history(history):
    if not history or not hasattr(history, 'history') or not history.history:
        print("No history object or history data found to plot.")
        return

    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss_hist = history.history.get('loss', [])
    val_loss_hist = history.history.get('val_loss', [])

    if acc:
        epochs_range = range(len(acc))
    elif loss_hist:
        epochs_range = range(len(loss_hist))
    else:
        print("No standard metrics (accuracy or loss) found in history to determine epoch range.")
        return

    plt.figure(figsize=(15, 6))

    plot_idx = 1
    has_accuracy = bool(acc and val_acc)
    has_loss = bool(loss_hist and val_loss_hist)

    if has_accuracy:
        plt.subplot(1, 2 if has_loss else 1, plot_idx)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plot_idx += 1

    if has_loss:
        plt.subplot(1, 2 if has_accuracy else 1, plot_idx)
        plt.plot(epochs_range, loss_hist, label='Training Loss')
        plt.plot(epochs_range, val_loss_hist, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

    if not has_accuracy and not has_loss:
        print("Neither accuracy nor loss metrics were found in history. Cannot plot.")
        plt.close()
        return

    plt.suptitle("Model Training History")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plot_save_path = os.path.join(config.BASE_DIR, "training_history.png")
    try:
        plt.savefig(plot_save_path)
        print(f"Training history plot saved to {plot_save_path}")
    except Exception as e:
        print(f"Error saving training history plot: {e}")
    plt.show()

def preprocess_image_for_prediction(image_path, img_height, img_width):
    try:
        img = tf.keras.utils.load_img(
            image_path, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        return img_array
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def set_seeds():
    os.environ['PYTHONHASHSEED'] = str(config.RANDOM_SEED_GLOBAL)
    np.random.seed(config.RANDOM_SEED_GLOBAL)
    tf.random.set_seed(config.RANDOM_SEED_GLOBAL)
    print(f"Global random seeds set to {config.RANDOM_SEED_GLOBAL}.")
