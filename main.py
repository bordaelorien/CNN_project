import numpy as np
from network import CNN
import tensorflow as tf



#création du réseau
cnn = CNN(img_shape=(28, 28, 1))
cnn.addLayer(type="CONV", num_filters=32)
cnn.addLayer(type="RELU", alpha=0.01)
cnn.addLayer(type="POOL")
cnn.addLayer(type="CONV", num_filters=64)
cnn.addLayer(type="RELU", alpha=0.01)
cnn.addLayer(type="POOL")
cnn.addLayer(type="FC", num_neurons=128)
cnn.addLayer(type="RELU", alpha=0.01)
cnn.addLayer(type="FC", num_neurons=10)

#importation des données
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

#normalisation
x_train = x_train.reshape(-1, 28, 28, 1) / 255
x_test = x_test.reshape(-1, 28, 28, 1) / 255

#entrainement réseau
cnn.train(x_train, y_train, x_test, y_test, epochs=10, batch_size=50, learning_rate=0.001)

