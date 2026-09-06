import numpy as np
from utils.activation_functions import Activation as Atv  # type: ignore

class MutliLayerPerceptron:
    def __init__(self, hidden_layers=[8], learning_rate=0.1):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

    def fit(self, X, y, epochs=1000):
        n_samples, input_size = X.shape
        _, output_size = y.shape

        # 1. CREATE A DYNAMIC ARCHITECTURE MAP
        layer_sizes = [input_size] + self.hidden_layers + [output_size]

        # 2. INITIALIZE WEIGHTS DYNAMICALLY (Xavier Initialization)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)

        # 3. THE TRAINING LOOP
        for epoch in range(epochs):
            
            # --- FORWARD PASS (Dynamic) ---
            activations = [X] 
            Z_vals = []       
            
            for i in range(len(self.weights)):
                current_input = activations[-1] 
                Z = np.dot(current_input, self.weights[i]) + self.biases[i]
                Z_vals.append(Z)
                
                if i == len(self.weights) - 1:
                    A = Atv.sigmoid(Z)
                else:
                    A = Atv.tanh(Z) # Swapped to Tanh for robustness!
                
                activations.append(A)

            # --- BACKWARD PASS (Dynamic) ---
            dW_list = []
            db_list = []
            
            final_output = activations[-1]
            error = final_output - y
            dZ = error * Atv.sigmoid_derivative(Z_vals[-1])
            
            for i in reversed(range(len(self.weights))):
                current_activation = activations[i] 
                
                dW = np.dot(current_activation.T, dZ)
                db = np.sum(dZ, axis=0, keepdims=True)
                
                dW_list.insert(0, dW)
                db_list.insert(0, db)
                
                if i > 0:
                    dA = np.dot(dZ, self.weights[i].T)
                    # Use Tanh derivative here!
                    dZ = dA * Atv.tanh_derivative(Z_vals[i-1])
            
            # --- UPDATE WEIGHTS ---
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dW_list[i] / n_samples
                self.biases[i] -= self.learning_rate * db_list[i] / n_samples
                
            # --- TRACK THE LOSS ---
            if epoch % 500 == 0:
                loss = np.mean(np.square(error))
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

    def predict(self, X):
        activations = [X]
        
        for i in range(len(self.weights)):
            current_input = activations[-1]
            Z = np.dot(current_input, self.weights[i]) + self.biases[i]
            
            if i == len(self.weights) - 1:
                A = Atv.sigmoid(Z)  
            else:
                A = Atv.tanh(Z) # Swapped to Tanh here too!
                
            activations.append(A)
            
        final_output = activations[-1]
        return np.where(final_output >= 0.5, 1, 0)