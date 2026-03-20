import numpy as np

class CNN:
    """Réseau de neurones convolutif"""

    def __init__(self):
        pass
    

    def addLayer(self, type, *args):
        pass

    
    def forwardProp(self, input_image):
        pass

    
    def predict(self, input_image):
        return np.argmax(self.forwardProp(input_image))







class convLayer:
    """Une couche de neurones convolutive"""

    def __init__(self, input_shape, num_filters, *, filter_size=3, stride=1, padding="zeros"):
        pass


    def forward(self, input):
        pass



class poolLayer:
    """Une couche de max-pooling"""

    def __init__(self, *, pooling_size=2, stride=2):
        pass


    def forward(self, input):
        pass


class reluLayer:
    """Une couche ReLu"""

    def __init__(self):
        pass

    
    def forward(self, input):
        pass


class FCLayer:
    """Une couche de neurones dense (fully-connected layer)"""

    def __init__(self, input_size, num_neurons):
        pass

    
    def forward(self, input):
        pass