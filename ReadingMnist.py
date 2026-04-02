import numpy as np  # linear algebra
import struct
from array import array
from os.path import join
import random
import matplotlib.pyplot as plt
import pandas as pd

#
# MNIST Data Loader Class
#
class MnistDataloader(object):

    def __init__(self): # training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath):

        #
        # Set file paths based on added MNIST Datasets
        #
        input_path = 'Data'
        training_images_filepath = input_path + '/train-images.idx3-ubyte'
        training_labels_filepath = input_path + '/train-labels.idx1-ubyte'
        test_images_filepath = input_path + '/t10k-images.idx3-ubyte'
        test_labels_filepath = input_path + '/t10k-labels.idx1-ubyte'


        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)

#
# Verify Reading Dataset via MnistDataloader class
#






#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15)
        index += 1
    plt.show()

#
# Load MINST dataset
#


def load_emnist_csv(file_path):
    print("Chargement des données... (cela peut prendre une minute)")
    # On utilise pandas pour lire le CSV
    data = pd.read_csv(file_path, header=None)
    
    # Conversion en array NumPy
    data = np.array(data)
    
    # Mélanger les données (important pour l'entraînement)
    np.random.shuffle(data)
    
    # Séparation : étiquettes (Y) et pixels (X)
    Y = data[:, 0]
    X = data[:, 1:]
    
    # --- ÉTAPE CRUCIALE POUR EMNIST ---
    # Les images EMNIST sont souvent pivotées à 90° et inversées dans le CSV
    # Voici comment les remettre à l'endroit pour les visualiser ou les traiter
    X = X.reshape(-1, 28, 28)
    X = np.transpose(X, (0, 2, 1)) # Corrige la rotation
    X = X.reshape(-1, 784) # On ré-aplatit pour le réseau de neurones
    
    # Normalisation (0-255 -> 0-1)
    X = X / 255.0
    
    return X, Y

def check_data(X, Y, index):
    mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
    img = X[index].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {mapping[Y[index]]}")
    plt.show()


if __name__=="__main__":
    X, Y = load_emnist_csv("Data/emnist-balanced-test.csv")
    check_data(X, Y, 11000)
    print(np.max(Y))
