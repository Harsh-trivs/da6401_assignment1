import numpy as np

# Layer class
class Layer:
    def __init__(self, input_size, output_size, activation, weight_init='Xavier'):
        if weight_init == 'Xavier':
            scale = np.sqrt(1 / (input_size))
        elif weight_init == 'random':
            scale = 1
        self.weights = np.random.randn(input_size, output_size) * scale
        self.biases = np.zeros((1, output_size))
        self.activation_name = activation

    
    def activate(self, x):
        if self.activation_name == 'relu':
            return np.maximum(0, x)
        elif self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_name == 'tanh':
            return np.tanh(x)
        elif self.activation_name == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        elif self.activation_name == 'linear':
            return x
        else:
            raise ValueError("Unsupported activation function")
    
    def activation_derivative(self, x):
        if self.activation_name == 'relu':
            return (x > 0).astype(float)
        elif self.activation_name == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        elif self.activation_name == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_name == 'linear':
            return np.ones_like(x)
        else:
            raise ValueError("Unsupported activation function")
     

# Neural Network class
class FeedForwardNN:
    def __init__(self, input_size, hidden_layers, output_size, lr=0.01, optimizer="sgd",momentum=0.9,beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-6,alpha=0,batch_size=32,loss_function='cross_entropy',weight_init = 'Xavier'):  
        self.lr = lr
        self.batch_size = batch_size
        self.alpha=alpha
        self.optimizer = optimizer.lower()
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1  # Momentum parameter
        self.beta2 = beta2  # RMSProp parameter
        self.epsilon = epsilon  # Avoid division by zero
        self.t = 0  # Time step for Adam and Nadam
        self.loss_function = loss_function
        self.weight_init = weight_init
        self.layers = []
        self.optim_states = []  # Store optimization states

        layer_sizes = [input_size] + [layer[0] for layer in hidden_layers] + [output_size]
        if self.loss_function == 'cross_entropy':
            activations = [layer[1] for layer in hidden_layers] + ['softmax']
        elif self.loss_function == 'mean_squared_error':
            activations = [layer[1] for layer in hidden_layers] + ['linear']
        else:
            raise ValueError("Unsupported loss function")

        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i + 1], activations[i])
            self.layers.append(layer)
            
            self.optim_states.append({
                "v_weights": np.zeros_like(layer.weights),  # For momentum & Adam
                "v_biases": np.zeros_like(layer.biases),
                "s_weights": np.zeros_like(layer.weights),  # For RMSProp & Adam
                "s_biases": np.zeros_like(layer.biases)
            })
    
    def forward(self, X):
        self.a = [X]
        self.z = []
        
        for layer in self.layers:
            self.z.append(np.dot(self.a[-1], layer.weights) + layer.biases)
            self.a.append(layer.activate(self.z[-1]))
        
        return self.a[-1]
    
    def backward(self, X, y):
        m = X.shape[0]
        y_one_hot = np.eye(self.layers[-1].biases.shape[1])[y]
        if self.loss_function == 'cross_entropy':
            dz = self.a[-1] - y_one_hot
        elif self.loss_function == 'mean_squared_error':
            dz = 2 * (self.a[-1] - y_one_hot) / y.shape[0]
        self.t += 1  # Increase time step for Adam and Nadam

        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            state = self.optim_states[i]

            dw = np.dot(self.a[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m

            if self.optimizer == "sgd":
                # Vanilla SGD
                layer.weights -= self.lr * dw
                layer.biases -= self.lr * db

            elif self.optimizer == "momentum":
                # Momentum-based SGD
                state["v_weights"] = self.beta1 * state["v_weights"] + dw
                state["v_biases"] = self.beta1 * state["v_biases"] + db

                layer.weights -= self.lr * state["v_weights"]
                layer.biases -= self.lr * state["v_biases"]
            
            elif self.optimizer == "nag":
                v_prev_w = state["v_weights"]
                v_prev_b = state["v_biases"]
                
                state["v_weights"] = self.momentum * state["v_weights"] + self.lr * dw
                state["v_biases"] = self.momentum * state["v_biases"] + self.lr * db
                
                layer.weights -= self.momentum * v_prev_w + (1 - self.momentum) * state["v_weights"]
                layer.biases -= self.momentum * v_prev_b + (1 - self.momentum) * state["v_biases"]
            
            elif self.optimizer == "rmsprop":
                # RMSProp
                state["s_weights"] = self.beta * state["s_weights"] + (1 - self.beta) * (dw ** 2)
                state["s_biases"] = self.beta * state["s_biases"] + (1 - self.beta) * (db ** 2)

                layer.weights -= self.lr*0.1* dw / (np.sqrt(state["s_weights"]) + self.epsilon)
                layer.biases -= self.lr*0.1* db / (np.sqrt(state["s_biases"]) + self.epsilon)
            
            elif self.optimizer == "adam" or self.optimizer == "nadam":
                # Adam and Nadam
                state["v_weights"] = self.beta1 * state["v_weights"] + (1 - self.beta1) * dw
                state["v_biases"] = self.beta1 * state["v_biases"] + (1 - self.beta1) * db

                state["s_weights"] = self.beta2 * state["s_weights"] + (1 - self.beta2) * (dw ** 2)
                state["s_biases"] = self.beta2 * state["s_biases"] + (1 - self.beta2) * (db ** 2)

                # Bias correction
                v_w_corr = state["v_weights"] / (1 - self.beta1 ** self.t)
                v_b_corr = state["v_biases"] / (1 - self.beta1 ** self.t)
                s_w_corr = state["s_weights"] / (1 - self.beta2 ** self.t)
                s_b_corr = state["s_biases"] / (1 - self.beta2 ** self.t)

                if self.optimizer == "nadam":
                    # Nadam update
                    v_w_corr = self.beta1 * v_w_corr + (1 - self.beta1) * dw / (1 - self.beta1 ** self.t)
                    v_b_corr = self.beta1 * v_b_corr + (1 - self.beta1) * db / (1 - self.beta1 ** self.t)

                layer.weights -= self.lr * v_w_corr / (np.sqrt(s_w_corr) + self.epsilon)
                layer.biases -= self.lr * v_b_corr / (np.sqrt(s_b_corr) + self.epsilon)
            
            # L2 Regularization
            layer.weights -= self.lr*self.alpha * layer.weights

            if i > 0:
                dz = np.dot(dz, layer.weights.T) * self.layers[i - 1].activation_derivative(self.z[i - 1])
    
    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                self.forward(X_batch)
                self.backward(X_batch, y_batch)
            # Compute loss for the entire dataset after weight updates
            loss = self.loss(X, y)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
    
    def loss(self,X,y):
        self.forward(X)
        if self.loss_function == 'cross_entropy':
            return -np.mean(np.log(self.a[-1][range(y.shape[0]), y] + 1e-8))
        elif self.loss_function == 'mean_squared_error':
            y_one_hot = np.eye(self.layers[-1].biases.shape[1])[y]  # One-hot encode labels
            return np.mean((self.a[-1] - y_one_hot) ** 2)
