import numpy as np
import tensorflow as tf


def lstm():
    num_layers = 1


    num_repeticiones = 5
    cantidad_pasos_por_repeticion = 100
    num_features = 8
    W = 1

    num_pasos_disponibles = 4


    # Load the data into numpy arrays
    data = np.random.randn(
        num_repeticiones, 
        cantidad_pasos_por_repeticion, 
        num_features
    )
    labels = np.random.randint(
        0, 
        num_pasos_disponibles, 
        size=(num_repeticiones, cantidad_pasos_por_repeticion)
    )

    print(f"Shape data: {data.shape}")
    print(f"Shape labels: {labels.shape}")

    # Convert the labels to one-hot encoding
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes=num_pasos_disponibles)

    print(len(data))
    print(f"Shape labels_onehot: {labels_onehot.shape}")

    split_idx = int(0.8 * len(data))
    train_data = (data[:split_idx], labels_onehot[:split_idx])
    val_data = (data[split_idx:], labels_onehot[split_idx:])


    # Define the input layer
    inputs = tf.keras.Input(shape=(None, num_features))

    print(f"Shape inputs: {inputs.shape}")

    # Define the LSTM layers
    x = inputs
    for i in range(num_layers):
        lstm_cell = tf.keras.layers.LSTMCell(
            units=W * cantidad_pasos_por_repeticion * num_features,
            kernel_initializer=tf.keras.initializers.RandomUniform(),
            recurrent_initializer=tf.keras.initializers.RandomUniform(),
            bias_initializer=tf.keras.initializers.RandomUniform()
        )
        lstm_layer = tf.keras.layers.RNN(lstm_cell, return_sequences=True)
        x = lstm_layer(x)

    # Define the output layer
    outputs = tf.keras.layers.Dense(units=num_pasos_disponibles, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Define the loss function and optimizer
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    print(f"Shape train_data[0]: {train_data[0].shape}")
    print(f"Shape train_data[1]: {train_data[1].shape}")

    print(f"Shape val_data[0]: {val_data[0].shape}")
    print(f"Shape val_data[1]: {val_data[1].shape}")
    #print(f"Shape labels_onehot: {labels_onehot.shape}")


    # Reshape the input data to match the input layer shape
    #td = np.reshape(train_data[0], (-1, num_features))
    #vd = np.reshape(val_data[0], (-1, num_features))

    # Train the model
    model.fit(train_data[0], train_data[1], epochs=10, batch_size=num_repeticiones)

    # Evaluate the model on the validation data
    val_loss, val_accuracy = model.evaluate(val_data[0], val_data[1])

    # Print the validation loss and accuracy
    print('Validation Loss:', val_loss)
    print('Validation Accuracy:', val_accuracy)



if __name__ == "__main__":
    lstm()