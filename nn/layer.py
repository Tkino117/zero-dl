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
        # print(f"class(X, W, b): {type(X)} {type(self.W)} {type(self.b)}")
        # print(f"size(X, W, b): {X.shape} {self.W.shape} {self.b.shape}")
        return X @ self.W + self.b

    def backward(self, dout):
        # print(f"class(W, dout): {type(self.W)} {type(dout)}")
        # print(f"size(W, dout): {self.W.shape} {dout.shape}")
        dX = dout @ self.W.T
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
        out = X.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout

class SoftmaxWithLoss:
    def __init__(self):
        self.Y = None
        self.T = None

    def _softmax(self, X):
        c = np.max(X, axis=1, keepdims=True)
        tmp = np.exp(X - c)
        return tmp / np.sum(tmp, axis=1, keepdims=True) 

    def _loss(self, Y, T):
        return -np.sum(T * np.log(Y + 1e-7)) / len(T)

    def forward(self, X, T):
        self.Y = self._softmax(X)
        self.T = T
        return self._loss(self.Y, self.T)

    def accuracy(self, X, T):
        Y = self._softmax(X)
        predict = np.argmax(Y, axis=1)
        true = np.argmax(T, axis=1)
        matches = predict == true
        return np.mean(matches)

    def backward(self, dout=1):
        return (self.Y - self.T) / len(self.T) 

