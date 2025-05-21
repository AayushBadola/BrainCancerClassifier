import tensorflow as tf
from tensorflow.keras import optimizers
import os
import numpy as np

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

SKIP_STAGE_1_HEAD_TRAINING = False

PATH_TO_HEAD_MODEL_CHECKPOINT = os.path.join(config.SAVED_MODELS_DIR, "brain_cancer_classifier_HEAD.keras")

def get_class_weights(train_ds, num_classes, class_names):
    print("Calculating class weights...")
    labels_list = []
    temp_ds_for_counting = train_ds.unbatch().map(lambda x,y: y) 
    for labels_batch_unbatched in temp_ds_for_counting:
         labels_list.append(np.argmax(labels_batch_unbatched.numpy()))
    if not labels_list: return None
    unique_classes, counts = np.unique(labels_list, return_counts=True)
    class_weights = {}; total_samples = sum(counts)
    actual_counts_map = {cls_idx: count for cls_idx, count in zip(unique_classes, counts)}
    for i in range(num_classes):
        class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
        count_for_class_i = actual_counts_map.get(i, 0)
        if count_for_class_i == 0: weight = 1.0
        else: weight = total_samples / (num_classes * count_for_class_i)
        class_weights[i] = weight
    print(f"Class weights calculated: {class_weights}")
    return class_weights

def train_model():
    print("--- Starting Transfer Learning Model Training Process ---")
    utils.set_seeds()
    print("Loading datasets...")
    train_ds, val_ds, class_names_loaded, num_classes_loaded = data_loader.load_datasets()

    if not train_ds or not val_ds : return None, None
    print(f"Data loaded. Training on {num_classes_loaded} classes: {class_names_loaded} ({config.IMG_HEIGHT}x{config.IMG_WIDTH})")

    class_weights = None
    print("Not using class weights." if class_weights is None else "Using class weights.")

    model_for_fine_tuning = None
    history_head = None

    if not SKIP_STAGE_1_HEAD_TRAINING:
        print("\n--- Stage 1: Training the Classification Head ---")
        model_head = model_builder.create_model(
            num_classes_from_data=num_classes_loaded,
            learning_rate=config.LEARNING_RATE_HEAD,
            is_fine_tuning_stage=False 
        )
        callbacks_head = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=config.LEARNING_RATE_HEAD/20, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(filepath=PATH_TO_HEAD_MODEL_CHECKPOINT, save_best_only=True, monitor='val_loss', mode='min', verbose=1, save_weights_only=False)
        ]
        print(f"Training head for {config.EPOCHS_HEAD_TRAINING} epochs (LR={config.LEARNING_RATE_HEAD}). Head model will be saved to {PATH_TO_HEAD_MODEL_CHECKPOINT}")
        history_head = model_head.fit(train_ds, epochs=config.EPOCHS_HEAD_TRAINING, validation_data=val_ds, callbacks=callbacks_head, class_weight=class_weights)
        print("\n--- Head Training Finished ---")
        if history_head and hasattr(utils, 'plot_training_history'): utils.plot_training_history(history_head)
        if os.path.exists(PATH_TO_HEAD_MODEL_CHECKPOINT):
            print(f"Loading best head model from checkpoint: {PATH_TO_HEAD_MODEL_CHECKPOINT}")
            model_for_fine_tuning = tf.keras.models.load_model(PATH_TO_HEAD_MODEL_CHECKPOINT, compile=False)
        else:
            print(f"Warning: Head model checkpoint {PATH_TO_HEAD_MODEL_CHECKPOINT} not found. Using model from end of head training.")
            model_for_fine_tuning = model_head 
    else:
        print(f"\n--- Skipping Stage 1: Attempting to load pre-trained head model from: {PATH_TO_HEAD_MODEL_CHECKPOINT} ---")
        if not os.path.exists(PATH_TO_HEAD_MODEL_CHECKPOINT):
            print(f"ERROR: Pre-trained head model file not found at {PATH_TO_HEAD_MODEL_CHECKPOINT}. Cannot skip Stage 1.")
            return None, None
        try:
            model_for_fine_tuning = tf.keras.models.load_model(PATH_TO_HEAD_MODEL_CHECKPOINT, compile=False) 
            print(f"Successfully loaded pre-trained head model from {PATH_TO_HEAD_MODEL_CHECKPOINT}.")
        except Exception as e:
            print(f"Error loading pre-trained head model: {e}"); import traceback; traceback.print_exc(); return None, None

    if model_for_fine_tuning is None: print("CRITICAL: No model for fine-tuning. Exiting."); return None, history_head

    print("\n--- Stage 2: Fine-tuning the Model ---")
    base_model_layer_name = next((l.name for l in model_for_fine_tuning.layers if isinstance(l, tf.keras.Model) and any(n in l.name.lower() for n in ['efficientnet', 'resnet', 'densenet'])), None)
    if not base_model_layer_name:
        print("ERROR: Could not find base model layer by name."); model_for_fine_tuning.summary(); return model_for_fine_tuning, history_head
    
    print(f"Identified base model layer: {base_model_layer_name}")
    base_model_to_modify = model_for_fine_tuning.get_layer(name=base_model_layer_name)
    base_model_to_modify.trainable = True 
    num_base_layers = len(base_model_to_modify.layers)
    if config.NUM_LAYERS_TO_FINETUNE > 0 and config.NUM_LAYERS_TO_FINETUNE < num_base_layers:
        for layer in base_model_to_modify.layers[:-config.NUM_LAYERS_TO_FINETUNE]: layer.trainable = False
        print(f"Fine-tuning: Last {config.NUM_LAYERS_TO_FINETUNE} layers of {base_model_to_modify.name} trainable.")
    else: print(f"Fine-tuning: All {num_base_layers} layers of {base_model_to_modify.name} trainable.")

    model_for_fine_tuning.compile(optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE_FINETUNE), loss=config.LOSS_FUNCTION, metrics=config.METRICS)
    print(f"Model recompiled for fine-tuning (LR={config.LEARNING_RATE_FINETUNE}). Summary:")
    model_for_fine_tuning.summary()

    checkpoint_finetune_path = os.path.join(config.SAVED_MODELS_DIR, "best_finetuned_model.weights.h5")
    callbacks_finetune = [
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_finetune_path, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=max(1e-7, config.LEARNING_RATE_FINETUNE / 20), verbose=1)
    ]
    print(f"Fine-tuning for {config.EPOCHS_FINETUNING} epochs. Best weights will be saved to {checkpoint_finetune_path}")
    initial_epoch_ft = len(history_head.epoch) if history_head and hasattr(history_head, 'epoch') else 0
    history_finetune = model_for_fine_tuning.fit(train_ds, epochs=initial_epoch_ft + config.EPOCHS_FINETUNING, initial_epoch=initial_epoch_ft, validation_data=val_ds, callbacks=callbacks_finetune, class_weight=class_weights)
    print("\n--- Model Fine-tuning Finished ---")

    combined_history_data = {}
    history_to_plot_obj = None
    if history_head and history_finetune:
        for key in history_head.history:
            if key in history_finetune.history: combined_history_data[key] = history_head.history[key] + history_finetune.history[key]
    elif history_finetune: combined_history_data = history_finetune.history
    elif history_head: combined_history_data = history_head.history
        
    if combined_history_data and hasattr(utils, 'plot_training_history'):
        class TempHistory: epoch = []; history = {}
        history_to_plot_obj = TempHistory()
        history_to_plot_obj.history = combined_history_data
        first_key = next(iter(combined_history_data), None)
        if first_key: history_to_plot_obj.epoch = list(range(len(combined_history_data[first_key])))
        utils.plot_training_history(history_to_plot_obj)

    if os.path.exists(checkpoint_finetune_path):
        print(f"Loading best fine-tuned weights from {checkpoint_finetune_path}.")
        model_for_fine_tuning.load_weights(checkpoint_finetune_path)
    else: print("Warning: Best fine-tuned weights checkpoint not found.")

    final_model_save_path = config.MODEL_PATH
    try:
        model_for_fine_tuning.save(final_model_save_path) 
        print(f"Final fine-tuned model saved to: {final_model_save_path}")
    except Exception as e_save: print(f"Error saving final model: {e_save}")

    if val_ds:
        print("\n--- Evaluating final fine-tuned model on validation data ---")
        loss, accuracy = model_for_fine_tuning.evaluate(val_ds, verbose=1)
        print(f"Final Val Acc (fine-tuned): {accuracy*100:.2f}%, Loss: {loss:.4f}")

    return model_for_fine_tuning, history_to_plot_obj

if __name__ == '__main__':
    try: from . import utils as u; u.set_seeds()
    except: np.random.seed(config.RANDOM_SEED_GLOBAL); tf.random.set_seed(config.RANDOM_SEED_GLOBAL); print("Seeds set: Basic fallback.")
    train_model()
