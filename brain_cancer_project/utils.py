import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import cv2
import imutils
from . import config

def crop_img(img_bgr_or_rgb):
    img_uint8 = img_bgr_or_rgb.astype(np.uint8)

    if img_uint8.ndim == 3 and img_uint8.shape[2] == 3:
        try:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        except cv2.error:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
    elif img_uint8.ndim == 2:
        gray = img_uint8
    else:
        print("Warning: crop_img received image with unexpected shape. Returning original.")
        return img_bgr_or_rgb

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if not cnts:
        print("Warning: No contours found in image for cropping. Returning original image.")
        return img_bgr_or_rgb

    c = max(cnts, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    h_orig, w_orig = img_bgr_or_rgb.shape[:2]
    x1, y1 = max(0, extLeft[0]), max(0, extTop[1])
    x2, y2 = min(w_orig, extRight[0]), min(h_orig, extBot[1])

    if y1 >= y2 or x1 >= x2:
        print("Warning: Invalid crop dimensions. Returning original image.")
        return img_bgr_or_rgb
            
    new_img_cropped = img_bgr_or_rgb[y1:y2, x1:x2].copy()
    
    return new_img_cropped

def plot_training_history(history):
    if not history or not hasattr(history, 'history') or not history.history:
        print("No history object or history data found to plot.")
        return

    acc_keys = [k for k in history.history.keys() if 'accuracy' in k and 'val' not in k]
    val_acc_keys = [k for k in history.history.keys() if 'val_accuracy' in k]
    loss_keys = [k for k in history.history.keys() if 'loss' in k and 'val' not in k]
    val_loss_keys = [k for k in history.history.keys() if 'val_loss' in k]

    acc = history.history.get(acc_keys[0] if acc_keys else 'accuracy', [])
    val_acc = history.history.get(val_acc_keys[0] if val_acc_keys else 'val_accuracy', [])
    loss_hist = history.history.get(loss_keys[0] if loss_keys else 'loss', [])
    val_loss_hist = history.history.get(val_loss_keys[0] if val_loss_keys else 'val_loss', [])

    if acc:
        epochs_range = range(len(acc))
    elif loss_hist:
        epochs_range = range(len(loss_hist))
    else:
        if hasattr(history, 'epoch') and history.epoch:
             epochs_range = history.epoch
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
        img_pil = tf.keras.utils.load_img(image_path)
        img_array_rgb = tf.keras.utils.img_to_array(img_pil)

        img_array_cropped = crop_img(img_array_rgb)

        if img_array_cropped.size == 0:
            print(f"Warning: Cropping resulted in an empty image for {image_path}. Cannot proceed with this image.")
            return None

        img_tensor_cropped_resized = tf.image.resize(img_array_cropped.astype(np.float32), [img_height, img_width])
        
        img_batch = tf.expand_dims(img_tensor_cropped_resized, 0)
        return img_batch
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def set_seeds():
    os.environ['PYTHONHASHSEED'] = str(config.RANDOM_SEED_GLOBAL)
    np.random.seed(config.RANDOM_SEED_GLOBAL)
    tf.random.set_seed(config.RANDOM_SEED_GLOBAL)
    print(f"Global random seeds set to {config.RANDOM_SEED_GLOBAL}.")

