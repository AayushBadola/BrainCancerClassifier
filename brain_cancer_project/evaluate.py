
import tensorflow as tf
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

try:
    from . import config
    from . import data_loader 
    from . import utils
except ImportError:
    import config
    import data_loader
    import utils


def evaluate_model(model_path=None, on_validation_data=True, on_test_data_dir=None):
    print("--- Starting Model Evaluation ---")
    if 'utils' in globals() and hasattr(utils, 'set_seeds'):
        utils.set_seeds()
    else:
        np.random.seed(config.RANDOM_SEED_GLOBAL)
        tf.random.set_seed(config.RANDOM_SEED_GLOBAL)


    if model_path is None:
        model_path = config.MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        print(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path, compile=True)
        model.summary()
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    class_names = config.get_class_names()
    num_classes = config.get_num_classes()

    if not class_names or num_classes == 0:
        print("Class names not found in config. Inferring from model output layer.")
        try:
            num_classes = model.output_shape[-1]
            class_names = [f"Class_{i}" for i in range(num_classes)]
            config.set_class_info(class_names, num_classes)
        except Exception as e_model_shape:
            print(f"Could not infer class names/num_classes from model output: {e_model_shape}")
            print("Evaluation cannot proceed without class information.")
            return
    
    print(f"Using {num_classes} classes for evaluation: {class_names}")


    dataset_to_evaluate = None
    dataset_name = ""
    true_labels_source_for_report = None

    if on_test_data_dir and os.path.exists(on_test_data_dir):
        print(f"\nLoading test data from: {on_test_data_dir}")
        dataset_name = "Test Set"
        try:
            test_ds = tf.keras.utils.image_dataset_from_directory(
                on_test_data_dir,
                seed=config.RANDOM_SEED_SPLIT,
                image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
                batch_size=config.BATCH_SIZE,
                label_mode='categorical',
                shuffle=False,
                class_names=class_names
            )
            if test_ds.class_names != class_names:
                 print(f"WARNING: Test set class names '{test_ds.class_names}' "
                       f"differ from expected '{class_names}'. This could lead to misinterpretation.")

            dataset_to_evaluate = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            true_labels_source_for_report = test_ds
            print(f"Test dataset '{on_test_data_dir}' loaded.")
        except Exception as e_test_load:
            print(f"Error loading test dataset from {on_test_data_dir}: {e_test_load}")
            return

    elif on_validation_data:
        print("\nLoading validation data for evaluation...")
        dataset_name = "Validation Set"
        if 'data_loader' in globals() and hasattr(data_loader, 'load_datasets'):
            _, val_ds, c_names_val, n_classes_val = data_loader.load_datasets()
            if val_ds:
                dataset_to_evaluate = val_ds
                true_labels_source_for_report = val_ds
                if not config.get_class_names() and c_names_val:
                    config.set_class_info(c_names_val, n_classes_val)
                    class_names = config.get_class_names()
                print("Validation dataset loaded.")
            else:
                print("Failed to load validation dataset. Aborting evaluation on validation set.")
                return
        else:
            print("data_loader module not available to load validation set. Cannot proceed.")
            return
    else:
        print("No dataset specified for evaluation.")
        return

    if dataset_to_evaluate and class_names:
        print(f"\n--- Evaluating model on {dataset_name} ---")
        eval_results = model.evaluate(dataset_to_evaluate, verbose=1, return_dict=True)
        print(f"\n{dataset_name} Accuracy: {eval_results.get('accuracy', 0)*100:.2f}%")
        print(f"{dataset_name} Loss: {eval_results.get('loss', 0):.4f}")

        print(f"\nGenerating report for {dataset_name}...")
        y_pred_probs = model.predict(dataset_to_evaluate)
        y_pred_indices = np.argmax(y_pred_probs, axis=1)
        
        y_true_indices = []
        if true_labels_source_for_report:
            for _, labels_batch in true_labels_source_for_report:
                y_true_indices.extend(np.argmax(labels_batch.numpy(), axis=1))
            y_true_indices = np.array(y_true_indices)
        else:
            print("Error: Cannot get true labels for report. Source dataset for labels is missing.")
            return

        if len(y_pred_indices) != len(y_true_indices):
            print(f"Warning: Mismatch in prediction ({len(y_pred_indices)}) and true label ({len(y_true_indices)}) counts.")
            min_len = min(len(y_pred_indices), len(y_true_indices))
            y_pred_indices = y_pred_indices[:min_len]
            y_true_indices = y_true_indices[:min_len]
            print(f"Adjusted to {min_len} samples for report.")

        try:
            report = classification_report(y_true_indices, y_pred_indices, target_names=class_names, zero_division=0)
            print("\nClassification Report:")
            print(report)
            report_save_path = os.path.join(config.BASE_DIR, f"{dataset_name.lower().replace(' ', '_')}_classification_report.txt")
            with open(report_save_path, 'w') as f: f.write(report)
            print(f"Classification report saved to {report_save_path}")
        except ValueError as ve:
            print(f"ValueError generating classification report: {ve}")
        except Exception as e_rep:
            print(f"Error generating classification report: {e_rep}")

        try:
            cm = confusion_matrix(y_true_indices, y_pred_indices, labels=np.arange(len(class_names)))
            plt.figure(figsize=(max(8, len(class_names)), max(6, int(len(class_names) * 0.8))))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix - {dataset_name}')
            cm_save_path = os.path.join(config.BASE_DIR, f"{dataset_name.lower().replace(' ', '_')}_confusion_matrix.png")
            plt.savefig(cm_save_path)
            print(f"Confusion matrix plot saved to {cm_save_path}")
            plt.show()
        except Exception as e_cm:
            print(f"Error generating confusion matrix: {e_cm}")

if __name__ == '__main__':
    print("--- Testing evaluate.py ---")
    evaluate_model(on_validation_data=True)