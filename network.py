import numpy as np

class CNN:
    """Réseau de neurones convolutif"""

    def __init__(self):
        pass
    

    def addLayer(self, type, *args):
        """Ajoute une couche de neurones au CNN.

        Args:
            type (str): type de couche à ajouter. 4 types: "CONV", "POOL", "RELU", "FC".
            Les autres arguments sont les hyperparamètres de la couche.
        """
        pass

    
    def forwardProp(self, input_image):
        """Forward propagation.

        Args:
            input_image (3D array - RGB image): Image en entrée.

        Returns:
            1D: Output du CNN pour l'image donnée.
                Applique un softmax à la dernière couche du réseau pour obtenir un vecteur de probabilités en sortie.
        """
        pass

    
    def predict(self, input_image):
        """Prédit le label d'une image.

        Args:
            input_image (3D array - RGB image): Image en entrée

        Returns:
            int: Label prédit par le CNN.
        """
        return np.argmax(self.forwardProp(input_image))







class convLayer:
    """Une couche de neurones convolutive"""

    def __init__(self, input_shape, num_filters, *, filter_size=3, stride=1, padding="zeros"):
        """
        Args:
            input_shape (tuple à trois éléments): Dimensions de la couche précédente
            num_filters (int): Nombre de filtres
            filter_size (int): Taille des filtres. Defaults to 3.
            stride (int): Stride. Defaults to 1.
            padding ("none", "zeros" ou "extend"): Type de padding appliqué à l'input. Le padding, s'il y en a,
                est calculé de façon à ce que l'output de la couche soit de la même taille que l'intput. Defaults to "zeros".
        """
        pass


    def forward(self, input):
        """Convolution avec l'input. 

        Args:
            input (array 3D de dim W1 x H1 x D1): Activations de la couche précédente

        Returns:
            array 3D de dim W2 x H2 x D2 : Valeurs d'activation de cette couche. 
                Si padding="none", alors l'output a les mêmes dimensions que l'input. 
                Sinon W2 = 1 + (W2 - filter_size)/stride, H2 = 1 + (H2 - filter_size)/stride, D2 = num_filters
        """
        pass



class poolLayer:
    """Une couche de max pooling"""

    def __init__(self, *, pooling_size=2, stride=2):
        """
        Args:
            pooling_size (int): Taille de la fenêtre de pooling. Defaults to 2.
            stride (int): Stride. Defaults to 2.
        """
        pass


    def forward(self, input):
        """Applique une opération de max pooling à l'input.

        Args:
            input (array 3D de dim W1 x H1 x D1): Activations de la couche précédente

        Returns:
            array 3D de dim W2 x H2 x D2 : Valeurs d'activation de cette couche.  
                W2 = 1 + (W2 - filter_size)/stride, H2 = 1 + (H2 - filter_size)/stride, D2 = D1
        """
        pass


class reluLayer:
    """Une couche ReLu"""

    def __init__(self):
        pass

    
    def forward(self, input):
        """Applique la fonction ReLU à l'input.

        Args:
            input (3D array): Activations de la couche précédente

        Returns:
            3D array: Valeurs d'activations de cette couche. Mêmes dimensions que l'input.
        """
        return 0


class FCLayer:
    """Une couche de neurones dense (fully-connected layer)"""

    def __init__(self, input_shape, num_neurons):
        """
        Args:
            input_shape (tuple à 3 éléments): Dimensions de la couche précédente.
            num_neurons (int): Nombre de neurones.
        """
        pass
    
    
    def forward(self, input):
        """Calcule la valeur d'activation de chaque neurone en fonction de l'activation de tous les neurones de la couche
        précédente.

        Args:
            input (3D array): Activations de la couche précédente

        Returns:
            1D array: Valeurs d'activations de cette couche. Taille = num_neurons.
        """
        pass