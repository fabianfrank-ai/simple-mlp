import sympy as sp
import numpy as np

class symbolicMLP:
    def __init__(self, np_module, nonlinear_weights):
        self.np = np_module
        self.nonlinear_weights = nonlinear_weights

        # Symbolic weights
        self.x1, self.x2, self.x3 = sp.symbols('x1 x2 x3')
        self.w_symbols = {
            'w1': sp.Symbol('w1'), 'w2': sp.Symbol('w2'), 'w3': sp.Symbol('w3'),
            'w4': sp.Symbol('w4'), 'b': sp.Symbol('b')
        }
        self.w_symbols.update({str(sym): sym for func in nonlinear_weights.values() for _, sym in func.items()})

        # initialise weights
        scale = 0.1
        self.w_vals = {k: self.np.random.randn() * scale for k in self.w_symbols.keys()}

        # Symbolic function
        self.z = (self.w_symbols['w1'] * self.x1 + self.w_symbols['w2'] * self.x2 +
                  self.w_symbols['w3'] * self.x3 + self.w_symbols['w4'] * self.x1 * self.x2 +
                  self.w_symbols['b'])
        for func, wdict in nonlinear_weights.items():
            for var, w_sym in wdict.items():
                self.z += w_sym * func(var)

        # Lambdify fo efficiency
        self.func_forward = sp.lambdify(
            [self.x1, self.x2, self.x3] + list(self.w_symbols.values()),
            self.z, modules=self.np
        )

    def _apply_activation(self, func, z):
        """Wählt die Aktivierungsfunktion basierend auf der Funktion."""
        activation_map = {
            sp.sin: self.sin_activation,
            sp.cos: self.cos_activation,
            sp.exp: self.exp_activation,
            sp.tanh: self.tanh_activation
        }
        return activation_map.get(func, lambda x: x)(z)

    def _apply_derivative(self, func, z):
        """Wählt die Ableitung basierend auf der Funktion."""
        derivative_map = {
            sp.sin: self.sin_derivative,
            sp.cos: self.cos_derivative,
            sp.exp: self.exp_derivative,
            sp.tanh: self.tanh_derivative
        }
        return derivative_map.get(func, lambda x: 1.0)(z)

    def forward(self, x1, x2, x3):
        """Efficient forwardpropagation"""
        weights = [self.w_vals[k] for k in self.w_symbols.keys()]
        z = self.func_forward(x1, x2, x3, *weights)
        return self.tanh_activation(z)  # Konsistente Aktivierung mit tanh

    def _clip(self, grad, threshold):
        """Clips dthe gradient"""
        return self.np.clip(grad, -threshold, threshold)

    def backward(self, x1, x2, x3, y_true, lr, y_pred, clip_grad=1):
        """Optimised backpropagation"""
        y_true = self.np.array(y_true)
        weights = [self.w_vals[k] for k in self.w_symbols.keys()]
        z = self.func_forward(x1, x2, x3, *weights)

         #Loss and dericatice
        dL_dy = 2 * (y_pred - y_true) / y_true.size  # MSE-derivative
        dL_dz = dL_dy * self.tanh_derivative(z)

        # Calculate gradients
        grad_w1 = self.np.mean(dL_dz * x1)
        grad_w2 = self.np.mean(dL_dz * x2)
        grad_w3 = self.np.mean(dL_dz * x3)
        grad_w4 = self.np.mean(dL_dz * x1 * x2)
        grad_b = self.np.mean(dL_dz)

        # Gradients for non-linear weights
        grad_nonlinear = {}
        for func, wdict in self.nonlinear_weights.items():
            for var_sym, w_sym in wdict.items():
                x = {'x1': x1, 'x2': x2, 'x3': x3}[str(var_sym)]
                grad_nonlinear[str(w_sym)] = self.np.mean(dL_dz * self._apply_derivative(func, x))

        # Update weights
        updates = {
            'w1': grad_w1, 'w2': grad_w2, 'w3': grad_w3, 'w4': grad_w4, 'b': grad_b
        }
        updates.update(grad_nonlinear)
        for w, grad in updates.items():
            self.w_vals[w] -= lr * self._clip(grad, clip_grad)

        # Calculate loss
        loss = 0.5 * self.np.mean((y_pred - y_true) ** 2)
        print(f"grad_w1: {grad_w1}, grad_w2: {grad_w2}, grad_w3: {grad_w3}, grad_w4: {grad_w4}, grad_b: {grad_b}")
        for w_sym, grad in grad_nonlinear.items():
            print(f"grad for {w_sym}: {grad}")
        print(self.w_vals)
        return loss

    # Activations
    def tanh_activation(self, z): return self.np.tanh(z)
    def sin_activation(self, z): return self.np.sin(z)
    def cos_activation(self, z): return self.np.cos(z)
    def exp_activation(self, z): return self.np.exp(z)

    # Derivatives
    def tanh_derivative(self, z): return 1 - (self.np.tanh(z)) ** 2
    def sin_derivative(self, z): return self.np.cos(z)
    def cos_derivative(self, z): return -self.np.sin(z)
    def exp_derivative(self, z): return self.np.exp(z)