import sympy as sp
import numpy as np

class symbolicMLP:
    def __init__(self, np_module, nonlinear_terms, l2_lambda, lr, hidden_layers, solver,activation_function, scale_factor=2.0, offset=0.0 ):
        self.np = np_module
        self.nonlinear_weights = nonlinear_terms
        self.scale_factor = scale_factor
        self.offset = offset
        self.epoch_count = 0
        self.lr = lr
        self.status = "running..."
        self.stop_flag = False
        self.counter = 0
        self.patience = 10
        self.loss_history = []
        self.best_loss = float('inf')
        self.best_weights = {}
        self.l2_lambda = l2_lambda
        self.hidden_layers = hidden_layers if isinstance(hidden_layers, list) else [hidden_layers]
        self.solver = solver
       

        # Determine output activation based on nonlinear terms
        if activation_function == 'placeholder':
          self.output_activation = next((f for f, terms in nonlinear_terms.items() if terms), None)
        else: 
            self.output_activation =activation_function
        
    
        # Initialize weights and biases as matrices and vectors
        scale = 0.1
        input_size = 3  # For x1, x2, x3
        self.weights = []
        self.biases = []
        prev_size = input_size
        for neurons in self.hidden_layers:
            self.weights.append(self.np.random.randn(neurons, prev_size) * scale)
            self.biases.append(self.np.random.randn(neurons, 1) * scale)
            prev_size = neurons
        # Output layer (1 neuron)
        self.weights.append(self.np.random.randn(1, prev_size) * scale)
        self.biases.append(self.np.random.randn(1, 1) * scale)

        # Adam optimization parameters
        self.m_w = [self.np.zeros_like(w) for w in self.weights]
        self.v_w = [self.np.zeros_like(w) for w in self.weights]
        self.m_b = [self.np.zeros_like(b) for b in self.biases]
        self.v_b = [self.np.zeros_like(b) for b in self.biases]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    def _apply_activation(self, func, z):
        """Applies the specified activation function to z."""
        if func == sp.sin or'sin'  :
            return self.np.sin(z)
        elif func == sp.cos or 'cos'  :
            return self.np.cos(z)
        elif func == sp.exp or 'exp'  :
            return self.np.exp(z)
        elif func == sp.tanh or 'tanh': 
            return self.np.tanh(z)
        elif func == 'sigmoid':
            return 1/(1 + self.np.exp(-z))  
        elif func == 'relu':
            return self.np.where(z > 0, z, 0)  
        else:
            return self.np.maximum(0.01 * z, z)  # Leaky ReLU as default

    def _apply_derivative(self, z, func):
        """Applies the derivative of the specified activation function."""
        if func == sp.sin or'sin':
            return self.np.cos(z)
        elif func == sp.cos or 'cos':
            return -self.np.sin(z)
        elif func == sp.exp or 'exp'  :
            return self.np.exp(z)
        elif func == sp.tanh or 'tanh':
            return 1 - self.np.tanh(z)**2
        elif func == 'sigmoid':
            return self.np.exp(z) / (1 + self.np.exp(z))**2  # sigmoid derivative
        elif func == 'relu':
            return self.np.where(z > 0, 1, 0.01)   
        else:
            return self.np.where(z > 0, 1, 0.01)  # Leaky ReLU derivative

    def forward(self, x1, x2, x3):
        """Computes the forward pass through the network."""
        x = self.np.column_stack((x1, x2, x3)).T  # Shape: (3, n_samples)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = w @ x + b
            x = self._apply_activation(self.output_activation, z)
        z_out = self.weights[-1] @ x + self.biases[-1]
        return self._apply_activation(self.output_activation, z_out)

    def backward(self, x1, x2, x3, y_true):
        y_true = self.np.array(y_true).reshape(1, -1)
        x = self.np.column_stack((x1, x2, x3)).T  # Shape: (3, n_samples)
        layers = [x]
        zs = []

        # Forward pass to store intermediates
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = w @ x + b
            zs.append(z)
            x = self._apply_activation(self.output_activation, z)
            layers.append(x)  # layers[1] is first hidden layer, layers[-2] is last hidden layer
        z_out = self.weights[-1] @ x + self.biases[-1]
        zs.append(z_out)
        layers.append(z_out)  # layers[-1] is z_out

        y_pred = self._apply_activation(self.output_activation, z_out)

        # Compute loss with L2 regularization
        loss = 0.5 * self.np.mean((y_pred - y_true) ** 2)
        l2_reg = sum(self.np.sum(w**2) for w in self.weights)
        loss += 0.5 * self.l2_lambda * l2_reg
        self.loss_history.append(loss)

        # Backward pass
        dL_dy = (y_pred - y_true) / y_true.size  # Shape: (1, n_samples)
        dL_dz = dL_dy * self._apply_derivative(z_out, self.output_activation)

        # Output layer gradients
        delta = dL_dz  # Shape: (1, n_samples)
        grad_w = delta @ layers[-2].T  # layers[-2] should be (50, n_samples), result should be (1, 50)
        grad_w = grad_w / x1.size  # Normalize by batch size
        grad_b = self.np.sum(delta, axis=1, keepdims=True)  # Shape: (1, 1)
        self.m_w[-1], self.v_w[-1] = self._update_weights(self.weights[-1], grad_w, self.m_w[-1], self.v_w[-1])
        self.m_b[-1], self.v_b[-1] = self._update_biases(self.biases[-1], grad_b, self.m_b[-1], self.v_b[-1])

        # Hidden layers gradients
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            delta = (self.weights[i+1].T @ delta) * self._apply_derivative(zs[i], sp.tanh)
            grad_w = delta @ layers[i].T  # Shape should match weights[i]
            grad_w = grad_w / x1.size
            grad_b = self.np.sum(delta, axis=1, keepdims=True)
            self.m_w[i], self.v_w[i] = self._update_weights(self.weights[i], grad_w, self.m_w[i], self.v_w[i])
            self.m_b[i], self.v_b[i] = self._update_biases(self.biases[i], grad_b, self.m_b[i], self.v_b[i])

        # Early stopping and best weights tracking
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_weights = {'weights': [w.copy() for w in self.weights], 'biases': [b.copy() for b in self.biases]}
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.stop_flag = True

        self.epoch_count += 1
        return loss
    
    

    def _update_weights(self, w, grad_w, m_w, v_w):
        """Updates weights using chosen optimizer."""
       
        if self.solver in['adam' ,'Adam']:
            self.t += 1
            m_w = self.beta1 * m_w + (1 - self.beta1) * grad_w
            v_w = self.beta2 * v_w + (1 - self.beta2) * (grad_w ** 2)
            m_hat = m_w / (1 - self.beta1 ** self.t)
            v_hat = v_w / (1 - self.beta2 ** self.t)
            w -= self.lr * m_hat / (self.np.sqrt(v_hat) + self.epsilon)
            return m_w, v_w
        elif self.solver in['sgd' , 'SGD']:
            self.t += 1
            w -= self.lr * grad_w
            return m_w, v_w 
        elif self.solver in ['rmsprop', 'RMSprop']:
            self.t += 1
            v_w= self.beta2*v_w+(1-self.beta2) * (grad_w ** 2)
            w -= grad_w*self.lr/(self.np.sqrt(v_w)+self.epsilon)
            return m_w, v_w
        
    
        

    def _update_biases(self, b, grad_b, m_b, v_b):
        """Updates biases using chosen optimizer."""
        if self.solver in ['adam' ,'Adam']:
            m_b = self.beta1 * m_b + (1 - self.beta1) * grad_b
            v_b = self.beta2 * v_b + (1 - self.beta2) * (grad_b ** 2)
            m_hat = m_b / (1 - self.beta1 ** self.t)
            v_hat = v_b / (1 - self.beta2 ** self.t)
            b -= self.lr * m_hat / (self.np.sqrt(v_hat) + self.epsilon)
            return m_b, v_b
        elif self.solver in['sgd' , 'SGD']:
            b -= self.lr * grad_b
            return m_b, v_b
        elif self.solver in ['rmsprop', 'RMSprop']:
            v_b= self.beta2*v_b+(1-self.beta2) * (grad_b ** 2)
            b -= grad_b*self.lr/(self.np.sqrt(v_b)+self.epsilon)
            return m_b, v_b

    def restore_best_weights(self):
        """Restores the best weights found during training."""
        if self.best_weights:
            self.weights = [w.copy() for w in self.best_weights['weights']]
            self.biases = [b.copy() for b in self.best_weights['biases']]


  
    # Activation functions
    def tanh_activation(self, z): return self.np.tanh(z)
    def sin_activation(self, z): return self.np.sin(z)
    def cos_activation(self, z): return self.np.cos(z)
    def exp_activation(self, z): return self.np.exp(z)
    def relu(self, z, alpha=0.01): return self.np.maximum(alpha * z, z)

    # Derivative functions
    def tanh_derivative(self, z): return 1 - (self.np.tanh(z)) ** 2
    def sin_derivative(self, z): return self.np.cos(z)
    def cos_derivative(self, z): return -self.np.sin(z)
    def exp_derivative(self, z): return self.np.exp(z)
    def relu_derivative(self, z, alpha=0.01): return self.np.where(z > 0, 1.0, alpha)
 