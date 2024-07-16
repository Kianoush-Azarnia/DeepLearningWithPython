from pathlib import Path
import tensorflow as tf 

import warnings
warnings.filterwarnings('ignore')

# Import mnist data stored in the following path: current directory -> mnist.npz
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

print(len(X_train))