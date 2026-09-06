import numpy as np

class Activation:


   @staticmethod
   def unit_step_func(x):
      return np.where(x > 1, 1, 0)

   @staticmethod
   def relu(x):
      return np.maximum(0,x)

   @staticmethod
   def relu_derivate(x):
      return np.where(x > 0, 1, 0)

   @staticmethod
   def sigmoid(x):
      x = np.clip(x, -500, 500)
      return 1 / (1 + np.exp(-x))

   @staticmethod
   def sigmoid_derivative(x):
      s = Activation.sigmoid(x)
      return s * (1 - s)

   @staticmethod
   def tanh(x):
      return np.tanh(x)

   @staticmethod
   def tanh_derivative(x):
      return 1.0 - np.tanh(x) ** 2


   


   