import numpy as np
import matplotlib.pyplot as plt

class CNN:
    """Réseau de neurones convolutif"""

    def __init__(self, img_shape):
        self.layers = [] #liste des couches de neurones
        self.img_shape = img_shape #taille de l'image en entrée
    

    def addLayer(self, type, **kwargs):
        """Ajoute une couche de neurones au CNN.

        Args:
            type (str): type de couche à ajouter. 4 types: "CONV", "POOL", "RELU", "FC".
            Les autres arguments sont les hyperparamètres de la couche.
        """

        #taille de couche précédente si elle existe, sinon taille de l'image en entrée
        input_shape = self.layers[-1].shape if len(self.layers) > 0 else self.img_shape

        #création d'une nouvelle couche du type spécfifié
        if type == "CONV":
            new_layer = ConvLayer(input_shape, **kwargs)

        elif type == "POOL":
            new_layer = PoolLayer(input_shape, **kwargs)

        elif type == "RELU":
            new_layer = ReluLayer(input_shape, **kwargs)

        elif type == "FC":
            new_layer = FCLayer(input_shape, **kwargs)

        else:
            raise ValueError(f"'{type}' n'est pas un type de couche correcte. " \
                            "Les quatre types de couches sont 'CONV', 'POOL', 'RELU', 'FC'.")

        #ajout de la couche à la liste
        self.layers.append(new_layer)

    


    def train(self, x_train, y_train, x_test, y_test, batch_size=50, epochs=5, learning_rate=0.005):
        n_samples = len(x_train)
        success_rate_list = []
        
        for epoch in range(epochs):
            #le taux d'apprentissage est divisé par 2 toutes les 10 epochs
            if epoch > 0 and epoch % 4 == 0 :
                learning_rate /= 2

            #mélange aléatoire du dataset
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            
             #découpage en mini-batches
            for start in range(0, n_samples, batch_size):
                x_batch = x_shuffled[start:start + batch_size]
                y_batch = y_shuffled[start:start + batch_size]

                for image, label in zip(x_batch, y_batch):
                    #forward pass
                    output = self.forwardProp(image)
                    
                    #encodage one-hot
                    target = np.zeros_like(output)
                    target[label] = 1
                    
                    #backward pass
                    self.backwardProp(output, target, learning_rate / len(x_batch))

            #affichage du taux de réussite sur les images tests à la fin de chaque epoch
            success_rate = self.test(x_test, y_test)
            success_rate_list.append(success_rate)
            print(f"Époque {epoch+1}/{epochs} - Taux de réussite: {success_rate:.2f}")

        #représentation graphique de l'apprentissage
        indices = range(1, epochs+1)
        plt.xlabel("Époque")
        plt.ylabel("Taux de réussite")
        plt.title("Courbe d'apprentissage")
        plt.plot(idx, success_rate_list)
        plt.show()


    def test(self, x_test, y_test):
        #nombre d'images correctement identifiées
        score = 0

        for image, label in zip(x_test, y_test):
            output = np.argmax(self.forwardProp(image).flatten())
            if output == label:
                score += 1

        #taux de réussite  
        return score*100/len(x_test)

    
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
        for i, layer in enumerate(self.layers):
            activations = layer.forward(activations)
            #print(f"Layer {i} ({type(layer).__name__}) shape: {activations.shape}, mean: {np.mean(activations)}")
        
        return self.softmax(activations)
        

    def backwardProp(self, output, expected, learning_rate):
        #erreur de la dernière couche avec softmax + cross-entropy
        error = output - expected

        #calcul de l'erreur et mise à jour des poids par couche
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)
    

    def predict(self, input_image):
        """Prédit le label d'une image.

        Args:
            input_image (3D array - RGB image): Image en entrée

        Returns:
            int: Label prédit par le CNN.
        """
        return np.argmax(self.forwardProp(input_image).flatten())
    

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
        self.input_shape = input_shape

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
        std = np.sqrt(2 / (filter_size * filter_size * D1))
        self.filters = np.random.randn(num_filters, filter_size, filter_size, D1) * std
        self.biases = np.ones((num_filters, 1)) * 0.01


    def forward(self, input):
        """Convolution avec l'input. 

        Args:
            input (array 3D de dim W1 x H1 x D1): Activations de la couche précédente

        Returns:
            array 3D de dim W2 x H2 x D2 : Valeurs d'activation de cette couche. 
                Si padding="none", alors l'output a les mêmes dimensions que l'input. 
                Sinon W2 = 1 + (W1 - filter_size)/stride, H2 = 1 + (H1 - filter_size)/stride, D2 = num_filters
        """
        self.x_col = im2col(input, self.shape, self.filter_size, self.stride, self.padding, self.padding_mode)
        f_row = self.filters.reshape(self.num_filters, -1)
        output = f_row @ self.x_col + self.biases
        W2, H2, D2 = self.shape
        return output.reshape(D2, W2, H2).transpose(1, 2, 0)
    

    def backward(self, output_error, learning_rate):
        W2, H2, D2 = self.shape # = output_error.shape si tout va bien
        output_error = output_error.transpose(2, 0, 1).reshape(D2, -1)

        #gradient des filtres et des biais
        dF = output_error @ self.x_col.T
        dF = dF.reshape(self.filters.shape)
        dB = np.sum(output_error, axis=1).reshape(D2, 1)

        #mise à jour des poids
        self.filters -= learning_rate * dF
        self.biases -= learning_rate * dB

        #calcul de l'erreur de l'input
        f_row = self.filters.reshape(self.num_filters, -1)
        dx_col = f_row.T @ output_error

        #on la remet dans les bonnes dimensions puis on propage
        return col2im(dx_col, self.input_shape, self.filter_size, self.stride, self.padding)
        




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
        self.input_shape = input.shape
        F = self.pooling_size
        D = self.input_shape[2]
        W2, H2, D2 = self.shape
        
        x_col = im2col(input, self.shape, F, self.stride)
        x_col = x_col.reshape(F * F, -1)
        
        #calcul du max sur chaque région
        max_values = np.max(x_col, axis=0)
        
        #création d'un masque qui retient les indices des valeurs maximales, il est utile dans la backward
        self.mask = (x_col == max_values)
        
        #on renvoie les max
        return max_values.reshape(D, W2, H2).transpose(1, 2, 0)

    
    def backward(self, output_error, learning_rate):
        F = self.pooling_size
        D = self.input_shape[2]
        
        #on aplatit l'erreur
        output_error = output_error.transpose(2, 0, 1).flatten()
        
        #on multiplie le masque par l'erreur
        dx_col = self.mask * output_error
        
        #on remet au format (F*F*D, W2*H2) pour col2im
        dx_col = dx_col.reshape(F * F * D, -1)
        
        #reconstruction de l'image
        return col2im(dx_col, self.input_shape, F, self.stride)



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
        self.input = input
        return np.where(input < 0, self.alpha * input, input)
    
    
    def backward(self, output_error, learning_rate):
        grad = np.where(self.input < 0, self.alpha, 1)
        return output_error * grad



class FCLayer:
    """Une couche de neurones dense (fully connected layer)"""

    def __init__(self, input_shape, num_neurons):
        """
        Args:
            input_shape (tuple à 3 éléments): Dimensions de la couche précédente.
            num_neurons (int): Nombre de neurones.
        """
        if len(input_shape) == 3:
            W, H, D = input_shape
            self.len_input = W * H * D

        else:
            self.len_input = input_shape[0]
        
        self.num_neurons = num_neurons
        self.shape = (self.num_neurons,)
        
        #initialisation des poids et des biais
        self.weights = np.random.randn(self.num_neurons, self.len_input) * np.sqrt(2 / self.len_input)
        self.biases = np.zeros(self.num_neurons)
    
    
    def forward(self, input):
        """Calcule la valeur d'activation de chaque neurone en fonction de l'activation de tous les neurones de la couche
        précédente.

        Args:
            input (3D array): Activations de la couche précédente.

        Returns:
            1D array: Valeurs d'activations de cette couche. Taille = num_neurons.
        """
        self.input = input
        return self.weights @ input.flatten() + self.biases
    

    def backward(self, output_error, learning_rate):
        #erreur de l'input
        input_error = self.weights.T @ output_error

        #gradient des poids et des biais
        dW = np.outer(output_error, self.input.flatten())
        dB = output_error

        #mise à jour des poids
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * dB

        #propagation de l'erreur
        return input_error.reshape(self.input.shape)
    




# ----------------------------------------------------------------------------------------------------------------------------
#fonctions im2col et col2im
# ----------------------------------------------------------------------------------------------------------------------------

def im2col(input, output_shape, filter_size, stride, padding=(0, 0), padding_mode="none"):
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


def col2im(dx_col, input_shape, filter_size, stride, padding=(0, 0)):
    W, H, D = input_shape
    pad_w, pad_h = padding
    W_padded, H_padded = W + 2*pad_w, H + 2*pad_h
    dx_padded = np.zeros((W_padded, H_padded, D))
    
    F = filter_size
    colonne = 0
    for i in range(0, W_padded - F + 1, stride):
        for j in range(0, H_padded - F + 1, stride):
            #on ajoute (+=) car un pixel peut contribuer à plusieurs convolutions
            patch = dx_col[:, colonne].reshape(F, F, D)
            dx_padded[i:i+F, j:j+F, :] += patch
            colonne += 1
            
    #on retire le padding pour retrouver la taille d'origine
    if pad_w > 0 or pad_h > 0:
        return dx_padded[pad_w:-pad_w, pad_h:-pad_h, :]
    return dx_padded





if __name__ == "__main__":


    cnn = CNN(img_shape=(300,300, 3))
    cnn.addLayer(type="CONV", num_filters=36)
    cnn.addLayer(type="RELU")
    cnn.addLayer(type="POOL")
    cnn.addLayer(type="FC", num_neurons=10)
    img = np.random.uniform(0, 1, size=(300, 300, 3))
    print(cnn.forwardProp(img))

    idx = range(10)
    y = range(0, 20, 2)
    plt.plot(idx, y)
    plt.show()
