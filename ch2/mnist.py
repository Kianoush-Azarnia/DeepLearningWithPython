from pathlib import Path
import tensorflow as tf 

from tensorflow import keras 
from tensorflow.keras import layers 

# Import mnist data stored in the following path: current directory -> mnist.npz
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path='mnist.npz') #(X_train, Y_train), (X_test, Y_test)
print("__________________________________________________________________________")
print(train_images.shape)

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"test_acc: {test_acc}")