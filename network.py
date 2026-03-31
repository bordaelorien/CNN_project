import numpy as np

class CNN:
    """Réseau de neurones convolutif"""

    def __init__(self, img_shape):
        self.layers = [] #liste des couches de neurones
        self.img_shape = (*img_shape, 3) #taille de l'image en entrée (image RGB donc la troisième dimension est 3)
    

    def addLayer(self, type, **kwargs):
        """Ajoute une couche de neurones au CNN.

        Args:
            type (str): type de couche à ajouter. 4 types: "CONV", "POOL", "RELU", "FC".
            Les autres arguments sont les hyperparamètres de la couche.
        """
        
        #création d'une nouvelle couche du type spécfifié
        input_shape = self.layers[-1].shape if len(self.layers) > 0 else self.img_shape
        if type == "CONV":
            #taille de couche précédente si elle existe, sinon taille de l'image en entrée
            new_layer = ConvLayer(input_shape, **kwargs)

        elif type == "POOL":
            new_layer = PoolLayer(input_shape, **kwargs)

        elif type == "RELU":
            new_layer = ReluLayer(input_shape)

        elif type == "FC":
            #taille de couche précédente si elle existe, sinon taille de l'image en entrée
            new_layer = FCLayer(input_shape, **kwargs)

        else:
            raise ValueError(f"'{type}' n'est pas un type de couche correcte. " \
                            "Les quatre types de couches sont 'CONV', 'POOL', 'RELU', 'FC'.")

        #ajout de la couche à la liste
        self.layers.append(new_layer)

    
    def forwardProp(self, input_image):
        """Forward propagation.

        Args:
            input_image (3D array - RGB image): Image en entrée.

        Returns:
            1D: Output du CNN pour l'image donnée.
                Applique un softmax à la dernière couche du réseau pour obtenir un vecteur de probabilités en sortie.
        """
        #vérification de la taille de l'image en entrée
        if input_image.shape != self.img_shape:
            raise ValueError(f"Image en entrée de taille {input_image.shape} alors " \
                             f"qu'elle devrait être de taille {self.img_shape}.")

        #forward propagation à travers toutes les couches du réseau
        activations = input_image
        for layer in self.layers:
            activations = layer.forward(activations)
        
        activations = activations.flatten()
        return self.softmax(activations)
        


    
    def predict(self, input_image):
        """Prédit le label d'une image.

        Args:
            input_image (3D array - RGB image): Image en entrée

        Returns:
            int: Label prédit par le CNN.
        """
        return np.argmax(self.forwardProp(input_image))
    

    def softmax(self, x):
        """Fonction softmax.

        Args:
            x (1D array): valeurs d'activations de la dernière couche (qui est sensée être une couche FC).

        Returns:
            1D array: version modifiée du vecteur en entrée qui permet de l'interpréter comme une
                distribution de probabilités.
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0)




class CNNLayer:
    """Une couche du CNN."""

    def im2col(self, input, output_shape, filter_size, stride, padding=(0, 0), padding_mode="none"):
        if padding_mode == "zeros":
            x = np.pad(input, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode="constant")

        elif padding_mode == "edge":
            x = np.pad(input, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode="edge")
        
        else:
            x = input
        
        #dimensions de l'input après padding 
        W, H, D = x.shape

        #dimenions de l'output
        F = filter_size
        W2, H2, D2 = output_shape

        x_col = np.zeros((F * F * D, W2 * H2))
        print(x_col.shape)

        colonne = 0
        for i in range(0, W - F + 1, stride):
            for j in range(0, H - F + 1, stride):
                x_col[:, colonne] = x[i:i+F, j:j+F, :].flatten()
                colonne += 1
        
        return x_col



class ConvLayer:
    """Une couche de neurones convolutive"""

    def __init__(self, input_shape, *, num_filters, filter_size=3, stride=1, padding="zeros"):
        """
        Args:
            input_shape (tuple à trois éléments): Dimensions de la couche précédente
            num_filters (int): Nombre de filtres
            filter_size (int): Taille des filtres. Defaults to 3.
            stride (int): Stride. Defaults to 1.
            padding ("none", "zeros" ou "edge"): Type de padding appliqué à l'input. Le padding, s'il y en a,
                est calculé de façon à ce que l'output de la couche soit de la même taille que l'intput. Defaults to "zeros".
        """
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding_mode = padding

        #calcul des dimensions de l'output et du padding
        W1, H1, D1 = input_shape

        if padding == "none":
            W2 = 1 + (W1 - filter_size) // stride
            H2 = 1 + (H1 - filter_size) // stride
            D2 = num_filters
            self.padding = (0, 0)

        elif padding == "zeros" or padding == "edge":
            W2, H2, D2 = W1, H1, num_filters
            pad_w = ((W1 - 1) * stride + filter_size - W1) // 2
            pad_h = ((H1 - 1) * stride + filter_size - H1) // 2
            self.padding = (pad_w, pad_h)

        self.shape = (W2, H2, D2)

        #initialisation des filtres et des biais
        self.filters = np.random.randn(num_filters, filter_size, filter_size, D1)
        self.biases = np.random.randn(num_filters, 1)


    def forward(self, input):
        """Convolution avec l'input. 

        Args:
            input (array 3D de dim W1 x H1 x D1): Activations de la couche précédente

        Returns:
            array 3D de dim W2 x H2 x D2 : Valeurs d'activation de cette couche. 
                Si padding="none", alors l'output a les mêmes dimensions que l'input. 
                Sinon W2 = 1 + (W1 - filter_size)/stride, H2 = 1 + (H1 - filter_size)/stride, D2 = num_filters
        """
        x_col = self.im2col(input, self.shape, self.filter_size, self.stride, self.padding, self.padding_mode)
        f_row = self.filters.reshape(self.num_filters, -1)
        output = f_row @ x_col + self.biases
        self.activations = output.reshape(self.shape)
        return self.activations
    

    def im2col(self, input, output_shape, filter_size, stride, padding=(0, 0), padding_mode="none"):
        if padding_mode == "zeros":
            x = np.pad(input, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode="constant")

        elif padding_mode == "edge":
            x = np.pad(input, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode="edge")
        
        else:
            x = input
        
        #dimensions de l'input après padding 
        W, H, D = x.shape

        #dimenions de l'output
        F = filter_size
        W2, H2, D2 = output_shape

        x_col = np.zeros((F * F * D, W2 * H2))

        colonne = 0
        for i in range(0, W - F + 1, stride):
            for j in range(0, H - F + 1, stride):
                x_col[:, colonne] = x[i:i+F, j:j+F, :].flatten()
                colonne += 1
        
        return x_col



class PoolLayer:
    """Une couche de max pooling"""

    def __init__(self, input_shape, *, pooling_size=2, stride=2):
        """
        Args:
            pooling_size (int): Taille de la fenêtre de pooling. Defaults to 2.
            stride (int): Stride. Defaults to 2.
        """
        W, H, D = input_shape
        self.pooling_size = pooling_size
        self.stride = stride

        #calcul des dimensions de l'output
        W2 = 1 + (W - self.pooling_size) // self.stride
        H2 = 1 + (H - self.pooling_size) // self.stride
        D2 = D
        self.shape = (W2, H2, D2)


    def forward(self, input):
        """Applique une opération de max pooling à l'input.

        Args:
            input (array 3D de dim W1 x H1 x D1): Activations de la couche précédente.

        Returns:
            array 3D de dim W2 x H2 x D2 : Valeurs d'activation de cette couche.  
                W2 = 1 + (W2 - pooling_size)/stride, H2 = 1 + (H2 - pooling_size)/stride, D2 = D1
        """ 
        W, H, D = input.shape
        F = self.pooling_size
        output = np.zeros(self.shape)

        for i in range(0, W - F + 1, self.stride):
            for j in range(0, H - F + 1, self.stride):
                output[i//self.stride, j//self.stride] = input[i:i+F, j:j+F, :].max(axis=(0, 1))

        self.activations = output
        return self.activations


class ReluLayer:
    """Une couche ReLu"""

    def __init__(self, input_shape, alpha=0):
        self.shape = input_shape
        self.alpha = alpha
    
    
    def forward(self, input):
        """Applique la fonction 'leaky ReLU' à l'input.

        Args:
            input (3D array): Activations de la couche précédente.

        Returns:
            3D array: Valeurs d'activations de cette couche. Mêmes dimensions que l'input.
        """
        self.activations = np.where(input < 0, self.alpha * input, input)
        return self.activations



class FCLayer:
    """Une couche de neurones dense (fully-connected layer)"""

    def __init__(self, input_shape, num_neurons):
        """
        Args:
            input_shape (tuple à 3 éléments): Dimensions de la couche précédente.
            num_neurons (int): Nombre de neurones.
        """
        W, H, D = input_shape
        self.len_input = W * H * D
        self.num_neurons = num_neurons
        
        #initialisation des poids et des biais
        self.weights = np.random.randn(self.num_neurons, self.len_input)
        self.biases = np.random.randn(self.num_neurons, 1)
    
    
    def forward(self, input):
        """Calcule la valeur d'activation de chaque neurone en fonction de l'activation de tous les neurones de la couche
        précédente.

        Args:
            input (3D array): Activations de la couche précédente.

        Returns:
            2D array: Valeurs d'activations de cette couche. Vecteur colonne de taille num_neurons.
        """
        x = input.reshape(-1, 1)
        self.activations = self.weights @ x + self.biases
        return self.activations



cnn = CNN(img_shape=(300,300))
cnn.addLayer(type="CONV", num_filters=36)
cnn.addLayer(type="POOL")
cnn.addLayer(type="RELU")
cnn.addLayer(type="FC", num_neurons=10)
img = np.zeros((300, 300, 3))
print(cnn.forwardProp(img))

m = np.asarray([[[1,3,5], [4,8,0]],
     [[16, 18, 20], [23, 45, 50]],
     [[100, 200, 30], [340, 234, 1000]]])

print(m[1, 1, 0])
print(m.max(axis=(1, 2)))

