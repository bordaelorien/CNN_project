import numpy as np
from network import CNN
import tensorflow as tf



#création du réseau
cnn = CNN(img_shape=(32, 32, 3))
cnn.addLayer(type="CONV", num_filters=32)
cnn.addLayer(type="RELU", alpha=0.01)
cnn.addLayer(type="POOL")
cnn.addLayer(type="DROPOUT", dropout_rate=0.1)

cnn.addLayer(type="CONV", num_filters=90)
cnn.addLayer(type="RELU", alpha=0.01)
cnn.addLayer(type="POOL")
cnn.addLayer(type="DROPOUT", dropout_rate=0.2)

cnn.addLayer(type="FC", num_neurons=256)
cnn.addLayer(type="RELU", alpha=0.01)
cnn.addLayer(type="DROPOUT", dropout_rate=0.4)
cnn.addLayer(type="FC", num_neurons=20)

#importation des données
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="coarse")

#normalisation
x_train = np.astype(x_train, "float64") / 255
x_test = np.astype(x_test, "float64") / 255



#entrainement réseau
#cnn.train(x_train, y_train, x_test, y_test, epochs=10, batch_size=128, learning_rate=0.01)

#sauvegarde des poids du modèle entrainé
#cnn.saveModel()


#importation du modèle
cnn.loadModel()

#tests
success_rate = cnn.test(x_test, y_test)
print(f"Taux de réussite du CNN entraîné sur le dataset cifar100 : {success_rate} %.")
