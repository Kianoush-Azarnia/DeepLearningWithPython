from pathlib import Path
import tensorflow as tf 

from tensorflow import keras 
from tensorflow.keras import layers 

# Import mnist data stored in the following path: current directory -> mnist.npz
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path='mnist.npz') #(X_train, Y_train), (X_test, Y_test)

import matplotlib.pyplot as plt
print(len(train_images))
print(train_images.ndim, train_images.shape)
digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.savefig('./files/digit4.png')