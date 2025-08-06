import numpy as np

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.X = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dout):
        dX = self.W.T @ dout
        self.dW = self.X.T @ dout
        self.db = np.sum(dout, axis=0)
        return dX

    def grad_descent(self, lr=0.01):
        self.W -= lr * self.dW
        self.b -= lr * self.db

class Sigmoid:
    def __init__(self):
        self.Y = None

    def forward(self, X):
        self.Y = 1 / (1 + np.exp(-X))
        return self.Y

    def backward(self, dout):
        return dout * self.Y * (1 - self.Y)

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, X):
        self.mask = (X <= 0)
        out = X.copy()[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout

class SoftmaxWithLoss:
    def __init__(self):
        self.Y = None
        self.T = None

    def forward(self, X, T):
        c = np.max(X, axis=1, keepdims=True)
        tmp = np.exp(X - c)
        self.Y = tmp / np.sum(tmp, axis=1, keepdims=True) 
        self.T = T
        loss = -np.sum(T * np.log(self.Y * 1e-7)) / len(T)
        return loss

    def backward(self, dout=1):
        return (self.Y - self.T) / len(self.T) 

