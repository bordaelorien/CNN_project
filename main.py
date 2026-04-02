import numpy as np
from network import CNN
from ReadingMnist import MnistDataloader



#création du réseau
cnn = CNN(img_shape=(28, 28))
cnn.addLayer(type="CONV", num_filters=8)
cnn.addLayer(type="RELU")
cnn.addLayer(type="POOL")
cnn.addLayer(type="FC", num_neurons=10)

#importation des données
data_loader = MnistDataloader()
training_data, test_data = data_loader.load_data()
x_train, y_train = training_data
x_test, y_test = test_data

for i in range(len(x_test)):
    x_train[i], x_test[i] = np.array(x_train[i]).reshape(28, 28, 1) / 255, np.array(x_test[i]).reshape(28, 28, 1) / 255

#entrainement réseau
cnn.train(x_train, y_train, x_test, y_test, epochs=20, learning_rate=0.01)

