import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input

try:
    from . import config
except ImportError:
    import config

def create_data_augmentation_layer():
    data_augmentation = Sequential(
        [
            layers.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS), name="augmentation_input"),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2), 
            layers.RandomZoom(0.2),   
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomBrightness(factor=0.2),
            layers.RandomContrast(factor=0.2) 
        ],
        name="data_augmentation"
    )
    return data_augmentation

def create_model(num_classes_from_data, 
                 print_summary=True, 
                 learning_rate=None, 
                 is_fine_tuning_stage=False):

    if num_classes_from_data is None or not isinstance(num_classes_from_data, int) or num_classes_from_data <= 0:
        raise ValueError(f"Number of classes must be a positive integer. Received: {num_classes_from_data}")

    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
        pooling=None 
    )
    base_model_name = base_model.name 

    if is_fine_tuning_stage:
        base_model.trainable = True
        if config.NUM_LAYERS_TO_FINETUNE > 0 and config.NUM_LAYERS_TO_FINETUNE < len(base_model.layers):
            for layer in base_model.layers[:-config.NUM_LAYERS_TO_FINETUNE]:
                layer.trainable = False
            print(f"Fine-tuning: Last {config.NUM_LAYERS_TO_FINETUNE} layers of {base_model_name} are trainable.")
        elif config.NUM_LAYERS_TO_FINETUNE == 0 : 
             print(f"Fine-tuning: All layers of {base_model_name} are trainable.")
        else: 
            print(f"Fine-tuning: All layers of {base_model_name} are trainable (NUM_LAYERS_TO_FINETUNE implies all).")
    else: 
        base_model.trainable = False
        print(f"{base_model_name} base model is FROZEN for head training.")

    inputs = layers.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS), name="input_image")
    x = create_data_augmentation_layer()(inputs)
    x = efficientnet_preprocess_input(x) 
    x = base_model(x, training=is_fine_tuning_stage) 
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(0.001), name="dense_head_1")(x)
    x = layers.BatchNormalization(name="bn_head_1")(x)
    x = layers.Activation('relu', name="relu_head_1")(x)
    x = layers.Dropout(0.5, name="dropout_head_1")(x)
    outputs = layers.Dense(num_classes_from_data, activation='softmax', name="output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"BrainCancer_{base_model_name}_Transfer")

    current_lr = learning_rate if learning_rate is not None else \
                 (config.LEARNING_RATE_FINETUNE if is_fine_tuning_stage else config.LEARNING_RATE_HEAD)

    optimizer = optimizers.Adam(learning_rate=current_lr)
    model.compile(optimizer=optimizer, loss=config.LOSS_FUNCTION, metrics=config.METRICS)
    
    if print_summary:
        print(f"--- Model Summary (Stage: {'Fine-tuning' if is_fine_tuning_stage else 'Head Training'}) ---")
        model.summary()
        if is_fine_tuning_stage:
            print(f"Number of trainable weights in base model ({base_model_name}): {len(base_model.trainable_weights)}")

    print(f"Transfer learning model ({base_model_name}) created. Stage: {'Fine-tuning' if is_fine_tuning_stage else 'Head Training'}.")
    return model

if __name__ == '__main__':
    print("--- Testing model_builder.py (Transfer Learning) ---")
    try:
        num_c_test = config.get_num_classes() 
        if num_c_test == 0: num_c_test = 4
        print("\nCreating model for HEAD TRAINING (base frozen):")
        create_model(num_classes_from_data=num_c_test, is_fine_tuning_stage=False)
        print("\nCreating model for FINE-TUNING (some base layers unfrozen):")
        create_model(num_classes_from_data=num_c_test, is_fine_tuning_stage=True)
    except Exception as e:
        print(f"Error: {e}"); import traceback; traceback.print_exc()
