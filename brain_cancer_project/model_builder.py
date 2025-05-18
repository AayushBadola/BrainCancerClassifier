import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers
try:
    from . import config
except ImportError:
    import config

def create_data_augmentation_layer():
    data_augmentation = Sequential(
        [
            layers.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS), name="augmentation_input"),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation"
    )
    return data_augmentation

def create_model(num_classes_from_data, print_summary=True):
    if num_classes_from_data is None or not isinstance(num_classes_from_data, int) or num_classes_from_data <= 0:
        raise ValueError(f"Number of classes must be a positive integer. Received: {num_classes_from_data}")

    input_layer = layers.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS), name="input_image")
    x = layers.Rescaling(1./255, name="rescaling")(input_layer)
    x = create_data_augmentation_layer()(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name="conv1")(x)
    x = layers.MaxPooling2D(name="pool1")(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name="conv2")(x)
    x = layers.MaxPooling2D(name="pool2")(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name="conv3")(x)
    x = layers.MaxPooling2D(name="pool3")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name="dense1")(x)
    x = layers.Dropout(0.5, name="dropout1")(x)
    output_layer = layers.Dense(num_classes_from_data, activation='softmax', name="output")(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="BrainCancerCNN_Functional")
    optimizer = optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss=config.LOSS_FUNCTION,
                  metrics=config.METRICS)
    
    if print_summary:
        model.summary()

    print("CNN model (Functional API) created and compiled successfully.")
    return model

if __name__ == '__main__':
    print("--- Testing model_builder.py ---")
    try:
        num_c_test = config.get_num_classes() 
        if num_c_test == 0:
            print("Warning: Number of classes is 0 or could not be determined from config/dataset for model_builder test. Using default of 3.")
            num_c_test = 3
        test_model = create_model(num_classes_from_data=num_c_test)
        if test_model:
            print("Model built successfully for testing.")
        else:
            print("Failed to build model for testing.")
    except Exception as e:
        print(f"Error during model_builder.py test: {e}")
        import traceback
        traceback.print_exc()
