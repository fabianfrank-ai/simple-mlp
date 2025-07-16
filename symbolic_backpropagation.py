#f(x1,x2)=w3*relu(w1*x1+w2+x2+w3+x3+b1)+b
import sympy as sp

class symbolicMLP():
    def __init__(self):
       self.x1, self.x2, self.x3, self.w1,self.w2,self.w3, self.b= sp.symbols('x1 x2 x3 w1 w2 w3 b')

    def forward(self):
        self.z= self.w3*self.x3+self.w2*self.x2+self.w1*self.x1+self.b
        output= self.relu(self.z)
        return self.z, output
           
    def relu(self,z):
        return sp.Piecewise((0, z <= 0), (z, z > 0))
    
    def relu_derivate(self,z):
         return sp.Piecewise((0, z <= 0), (z, z > 0))
    def backward(self, y_true):
     y_pred = self.relu(self.z)
     loss = 0.5 * (y_pred - y_true)**2

     dL_dy = sp.diff(loss, y_pred)        # ∂L/∂y_pred
     dy_dz = self.relu_derivative(self.z) # ∂y_pred/∂z

     dz_dw1 = self.x1
     dz_dw2 = self.x2
     dz_dw3 = self.x3
     dz_db = 1

     dL_dw1 = dL_dy * dy_dz * dz_dw1
     dL_dw2 = dL_dy * dy_dz * dz_dw2
     dL_dw3 = dL_dy * dy_dz * dz_dw3
     dL_db = dL_dy * dy_dz * dz_db

     return dL_dw1, dL_dw2, dL_dw3, dL_db

    
  