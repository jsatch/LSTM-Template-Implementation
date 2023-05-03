import numpy as np
import pandas as pd
import tensorflow as tf

# Load the data
#url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
data = pd.read_csv(
    "E:\\development\\projects\\ULIMA\\ITLAB\\PeruParty\\DanceClassification\\UCIHARDataset\\train\\X_train.txt", 
    header=None, sep='\s+')

# Separate the data into inputs (X) and labels (y)
X = data.iloc[:, :-2].values
y = data.iloc[:, -1].values - 1  # subtract 1 to make labels 0-indexed

# Convert the labels to one-hot encoding
y_onehot = tf.keras.utils.to_categorical(y, num_classes=6)

# Reshape the input data to match the input shape of the model
X = X.reshape(-1, 128, 9)

# Split the data into training and validation sets
split_idx = int(0.8 * len(X))
train_data = (X[:split_idx], y_onehot[:split_idx])
val_data = (X[split_idx:], y_onehot[split_idx:])

# Define the LSTM model
model = tf.keras.Sequential([    tf.keras.layers.LSTM(64, input_shape=(128, 9)),
    tf.keras.layers.Dense(6, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',

              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data[0], train_data[1], epochs=10, batch_size=64, validation_data=val_data)