import sys, os
sys.path.append(os.pardir)
import numpy as np
import grad
import error

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
    
    def predict(self, x):
        return x @ self.W

    def loss(self, x, t):
        y = self.predict(self, x)
        loss = error.cross_entropy_error(y, t)
        return loss

if __name__ == '__main__':
    net = simpleNet()
    