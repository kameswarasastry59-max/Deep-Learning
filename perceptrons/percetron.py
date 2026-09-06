import numpy as np 


class Perceptron:

    # We need __init__ here because every perceptron needs its own 
    # learning rate, number of passes (epochs), weights, and bias
    def __init__(self, learning_rate=0.1, epoches=10):
        self.learning_rate=learning_rate
        self.epoches=epoches
        self.weights=None
        self.bias=None 

    def fit(self, X, y):
        n_samples, n_features = X.shape

        rng=np.random.default_rng()
        self.weights=rng.random(n_features) * 0.01
        self.bias=rng.random() * 0.01


        for _ in range(self.epoches):
            for idx, x_i in enumerate(X):

                # Calculate the Output at that index
                linear_output = np.dot(x_i,self.weights)+self.bias

                # Make a prediction with the current weights
                y_predicted = np.where(linear_output >= 0, 1, 0)

                # Calculate the error (Real Answer - Predicted Answer)
                error = y[idx] - y_predicted

                self.weights += self.learning_rate * error * x_i
                self.bias += self.learning_rate * error


    def predict(self, X):
        # 1. Calculate the linear combination (w * x + b)
        linear_output = np.dot(X, self.weights) + self.bias
        
        # 2. Apply the activation function (our unit step function!)
        # Using >= 0 as the threshold, as we discussed earlier
        return np.where(linear_output >= 0, 1, 0)  

    