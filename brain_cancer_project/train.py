import tensorflow as tf
import os
try:
    from . import config
    from . import data_loader
    from . import model_builder
    from . import utils
except ImportError:
    import config
    import data_loader
    import model_builder
    import utils


def train_model():
    print("--- Starting Model Training Process ---")
    utils.set_seeds()

    print("Loading datasets...")
    train_ds, val_ds, class_names_loaded, num_classes_loaded = data_loader.load_datasets()

    if not train_ds or not val_ds or num_classes_loaded == 0:
        print("Failed to load data. Aborting training.")
        return None, None

    print(f"Data loaded. Training on {num_classes_loaded} classes: {class_names_loaded}")

    print("\nCreating model...")
    try:
        model = model_builder.create_model(num_classes_from_data=num_classes_loaded)
    except ValueError as e:
        print(f"Error creating model: {e}. Aborting training.")
        return None, None
    except Exception as e_create:
        print(f"An unexpected error occurred during model creation: {e_create}")
        import traceback
        traceback.print_exc()
        return None, None

    print("\nStarting model training...")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    )
    os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
    checkpoint_filepath = os.path.join(config.SAVED_MODELS_DIR, "best_epoch_weights.weights.h5")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1
    )
    callbacks_list = [early_stopping, model_checkpoint, reduce_lr]

    history = None
    try:
        history = model.fit(
            train_ds,
            epochs=config.EPOCHS,
            validation_data=val_ds,
            callbacks=callbacks_list
        )
    except Exception as e_fit:
        print(f"An error occurred during model.fit: {e_fit}")
        import traceback
        traceback.print_exc()
        return model if 'model' in locals() else None, history

    print("\n--- Model Training Finished ---")

    if os.path.exists(checkpoint_filepath):
        print(f"Loading best weights from {checkpoint_filepath}")
        model.load_weights(checkpoint_filepath)
    else:
        print("Warning: Best weights checkpoint file not found. Model will have weights from the last epoch.")

    try:
        model.save(config.MODEL_PATH)
        print(f"Final model (best weights or last epoch) saved to: {config.MODEL_PATH}")
    except Exception as e_save:
        print(f"Error saving final model: {e_save}")

    if history:
        print("\nPlotting training history...")
        utils.plot_training_history(history)

    if val_ds:
        print("\n--- Evaluating final model on validation data ---")
        loss, accuracy = model.evaluate(val_ds, verbose=1)
        print(f"Final Validation Accuracy: {accuracy*100:.2f}%")
        print(f"Final Validation Loss: {loss:.4f}")

    return model, history


if __name__ == '__main__':
    try:
        import utils as local_utils
        local_utils.set_seeds()
    except NameError:
        print("Warning: utils.set_seeds() could not be called. Ensure utils.py is accessible.")
        import numpy as np
        np.random.seed(config.RANDOM_SEED_GLOBAL)
        tf.random.set_seed(config.RANDOM_SEED_GLOBAL)

    train_model()
