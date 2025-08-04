import sys, os
sys.path.append(os.pardir)
from testdata import load_data
import numpy as np
from util import *

# バッチ処理に対応してみる（めっちゃ早くなった）
class TwoLayerNN:
    def __init__(self, W1, b1, h1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.h1 = h1
        self.W2 = W2
        self.b2 = b2
    
    def forward(self, X):
        return self._forward(X, self.W1, self.W2, self.b1, self.b2)
    
    def _forward(self, X, W1, W2, b1, b2):
        l1 = self.h1(X @ W1 + b1)
        # l2 = softmax(l1 @ W2)
        l2 = sigmoid_func(l1 @ W2 + b2) # 回帰なのでシグモイドにした
        return l2

    def loss(self, X, T):
        return self._loss(X, T, self.W1, self.W2, self.b1, self.b2)

    def _loss(self, X, T, W1, W2, b1, b2):
        Y = self._forward(X, W1, W2, b1, b2)[:, 0]
        return mean_squared_error(Y, T)

    def grad(self, W1, W2, b1, b2, X, T):
        f1 = lambda W: self._loss(X, T, W, W2, b1, b2)
        f2 = lambda W: self._loss(X, T, W1, W, b1, b2)
        f3 = lambda b: self._loss(X, T, W1, W2, b, b2)
        f4 = lambda b: self._loss(X, T, W1, W2, b1, b)
        g1 = self._grad(f1, W1)
        g2 = self._grad(f2, W2)
        g3 = self._grad(f3, b1)
        g4 = self._grad(f4, b2)
        return (g1, g2, g3, g4)

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
            g1, g2, g3, g4 = self.grad(self.W1, self.W2, self.b1, self.b2, X, T)
            self.W1 -= lr * g1
            self.W2 -= lr * g2
            self.b1 -= lr * g3
            self.b2 -= lr * g4
            if i % 1000 == 0:
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
x_test = dataset['x_test']
t_test = dataset['t_test']
W1 = np.random.randn(3, 10) 
W2 = np.random.randn(10, 1) 
b1 = np.zeros(10)
b2 = np.zeros(1)
h1 = relu_func

nn = TwoLayerNN(W1, b1, h1, W2, b2)

print(f"before train")
print(f"loss: {nn.loss(x_train, t_train)}")
print(f"accuracy: {nn.accuracy(x_train, t_train)}")
# print(np.transpose([nn.forward(x_train)[:, 0], t_train]))
print(f"after train")
nn.grad_descent(x_train, t_train, lr=0.05, step_num=10000)
print(f"loss: {nn.loss(x_train, t_train)}")
print(f"accuracy: {nn.accuracy(x_train, t_train)}")
# print(np.transpose([nn.forward(x_train)[:, 0], t_train]))
print(f"test")
print(f"loss: {nn.loss(x_test, t_test)}")
print(f"accuracy: {nn.accuracy(x_test, t_test)}")



