import tensorflow as tf

# CNNs are good at local patterns but don’t understand long-term dependencies 
# Adding an LSTM or GRU layer after the CNN to learn sequential relationships.
# Since engine temperature changes are gradual but depend on short bursts of fan activation, a kernel size of 5 to 7 is a good starting point.

def my_model(input_shape):
    input_layer = tf.keras.layers.Input(input_shape)

    # First CNN block
    conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=15, padding="same")(input_layer) #5
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.ReLU()(conv1)
    conv1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)

    # Second CNN block
    conv2 = tf.keras.layers.Conv1D(filters=128, kernel_size=15, padding="same")(conv1) #5
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.ReLU()(conv2)
    conv2 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv2)

    # Third CNN block
    conv3 = tf.keras.layers.Conv1D(filters=128, kernel_size=7, padding="same")(conv2) #3
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.ReLU()(conv3)

    # LSTM layer for sequence learning
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(conv3)

    # Fully Connected Layers
    dense = tf.keras.layers.Dense(128, activation="relu")(lstm)
    dropout = tf.keras.layers.Dropout(0.3)(dense)

    # Binary classification output
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(dropout)  # Changed to sigmoid for binary classification

    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

