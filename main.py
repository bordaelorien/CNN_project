import numpy as np
from network import CNN



#création du réseau
net = CNN()
net.addLayer(type="CONV", num_filters=96, filter_size=5, stride=2, padding="None")
net.addLayer(type="RELU")
net.addLayer(type="POOL", pooling_size=2, stride=2)
net.addLayer(type="FC", num_neurons=10)

#prédiction sur une donnée
image = np.zeros(32, 32, 3)
prediction = net.predict(image)

