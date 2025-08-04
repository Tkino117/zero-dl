import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from dataset.testdata import load_data
import numpy as np
from util import *

class TwoLayerNN:
    def __init__(self, W1, h1, W2):
        self.W1 = W1
        self.h1 = h1
        self.W2 = W2
    
    def forward(self, x):
        return self._forward(x, self.W1, self.W2)
    
    def _forward(self, x, W1, W2):
        l1 = self.h1(x @ W1)
        # l2 = softmax(l1 @ W2)
        l2 = l1 @ W2   # 出力が1次元で回帰として扱ってるのでリニア
        return l2

    def _error(self, x, t, W1, W2):
        return mean_squared_error(self._forward(x, W1, W2), t)

    def loss(self, X, T):
        return self._loss(X, T, self.W1, self.W2)

    def _loss(self, X, T, W1, W2):
        tmp = np.zeros(len(T))
        for i in range(len(T)):
            tmp[i] = self._error(X[i], T[i], W1, W2)
        return np.average(tmp)

    def grad(self, W1, W2, X, T):
        f1 = lambda W: self._loss(X, T, W, W2)
        f2 = lambda W: self._loss(X, T, W1, W)
        g1 = self._grad(f1, W1)
        g2 = self._grad(f2, W2)
        return (g1, g2)

    def _grad(self, f, W):
        res = np.zeros_like(W)
        d = 1e-4
        if W.ndim == 0:
            res = (f(W + d) - f(W - d)) / (2 * d)
        elif W.ndim == 1:
            for i in range(len(W)):
                dx = np.zeros_like(W)
                dx[i] += d
                res[i] = (f(W + dx) - f(W - dx)) / (2 * d)
        elif W.ndim == 2:
            for i in range(len(W)):
                for j in range(len(W[0])):
                    dx = np.zeros_like(W)
                    dx[i, j] += d
                    res[i, j] = (f(W + dx) - f(W - dx)) / (2 * d)
        return res

    def grad_descent(self, X, T, lr=0.01, step_num=100):
        for i in range(step_num):
            g1, g2 = self.grad(self.W1, self.W2, X, T)
            self.W1 -= lr * g1
            self.W2 -= lr * g2
            if i % 10 == 0:
                print(f"loss: {self.loss(X, T)} (step = {i})")

    def accuracy(self, X, T):
        # 暫定。変えてok
        correct = np.zeros(len(X))
        for i in range(len(X)):
            ans = 1 if self.forward(X[i]) > 0.5 else 0
            correct[i] = 1 if T[i] == ans else 0
        return np.average(correct)

dataset = load_data()
x_train = dataset['x_train']
t_train = dataset['t_train']
W1 = np.random.randn(3, 10)
W2 = np.random.randn(10, 1)
h1 = sigmoid_func

nn = TwoLayerNN(W1, h1, W2)

print(f"loss: {nn.loss(x_train, t_train)}")
print(f"accuracy: {nn.accuracy(x_train, t_train)}")
nn.grad_descent(x_train, t_train, lr=0.3, step_num=500)
print(f"loss: {nn.loss(x_train, t_train)}")
print(f"accuracy: {nn.accuracy(x_train, t_train)}")


