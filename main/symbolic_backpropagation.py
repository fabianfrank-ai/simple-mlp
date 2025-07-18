#f(x1,x2)=w3*relu(w1*x1+w2+x2+w3+x3+b1)+b
import sympy as sp
import cupy as cp
class symbolicMLP():
      def __init__(self, np_module):
        self.np = np_module
        self.x1, self.x2, self.x3, self.w1, self.w2, self.w3, self.b = sp.symbols('x1 x2 x3 w1 w2 w3 b')
        self.w1_val = self.np.array(self.np.random.randn() * 0.1)
        self.w2_val = self.np.array(self.np.random.randn() * 0.1)
        self.w3_val = self.np.array(self.np.random.randn() * 0.1)
        self.b_val = self.np.array(self.np.random.randn() * 0.1)

        self.z = self.w3 * self.x3 + self.w2 * self.x2 + self.w1 * self.x1 + self.b

       
        self.f_linear = sp.lambdify([self.x1, self.x2, self.x3, self.w1, self.w2, self.w3, self.b], self.z, modules=np_module)

      def relu(self, z):
        
        return self.np.maximum(z, 0)
      def relu_derivative(self, z):
 
            return (z > 0).astype(z.dtype)
      
      
      def forward(self, x1, x2, x3):
        x1 = self.np.array(x1)
        x2 = self.np.array(x2)
        x3 = self.np.array(x3)

        z_val = self.f_linear(x1, x2, x3, self.w1_val, self.w2_val, self.w3_val, self.b_val)
        y_pred = self.relu(z_val)
        return y_pred

      def backward(self, x1, x2, x3, y_true, lr):
            x1 = self.np.array(x1)
            x2 = self.np.array(x2)
            x3 = self.np.array(x3)
            y_true = self.np.array(y_true)

            # Forward pass
            z_val = self.f_linear(x1, x2, x3, self.w1_val, self.w2_val, self.w3_val, self.b_val)
            y_pred = self.relu(z_val)

            # Loss
            loss = 0.5 * self.np.mean((y_pred - y_true) ** 2)

            # Backprop numeric
            dL_dy = (y_pred - y_true) / y_true.size  # MSE Gradient

            dy_dz = self.relu_derivative(z_val)

            dL_dz = dL_dy * dy_dz  # Elementweise Multiplikation

            # Gradients
            grad_w1 = self.np.mean(dL_dz * x1)
            grad_w2 = self.np.mean(dL_dz * x2)
            grad_w3 = self.np.mean(dL_dz * x3)
            grad_b = self.np.mean(dL_dz)

            # Weights update
            self.w1_val -= lr * grad_w1
            self.w2_val -= lr * grad_w2
            self.w3_val -= lr * grad_w3
            self.b_val -= lr * grad_b

            return float(loss)
            
  