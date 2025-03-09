import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.image import resize


def build_lenet5():
    model = keras.Sequential([
        keras.Input(shape=(32, 32, 1)),
        layers.Conv2D(filters=6, kernel_size=(5, 5), activation='tanh', padding='same'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),

        layers.Conv2D(filters=16, kernel_size=(5, 5), activation='tanh'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),

        layers.Conv2D(filters=120, kernel_size=(5, 5), activation='tanh'),
        layers.Flatten(),

        layers.Dense(84, activation='tanh'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Load and preprocess dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Resize images to (32,32)
x_train = np.array([resize(img[..., np.newaxis], (32, 32)) for img in x_train])
x_test = np.array([resize(img[..., np.newaxis], (32, 32)) for img in x_test])

# Normalize the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Create the model
lenet5 = build_lenet5()

# Train the model
lenet5.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = lenet5.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Save the model
lenet5.save("lenet5_model.h5")
