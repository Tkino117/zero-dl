import numpy as np
from typing import Callable

### Errors
def cross_entropy_error(y, t):
    delta = 1e-7   # log(0) 回避
    return -np.sum(t * np.log(y + delta))

def mean_squared_error(y, t):
    return np.sum((y - t) ** 2) / 2

### Gradients
# f: W -> R (must be W.ndim <= 2)
def grad(f: Callable[[np.ndarray], np.ndarray], W: np.ndarray):
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

# update W to minimize f(W)
def grad_descent(f: Callable[[np.ndarray], np.ndarray], W: np.ndarray, lr=0.01, step_num=100):
    for i in range(step_num):
        W -= lr * grad(f, W)

def grad_descent_mul(f: Callable[[np.ndarray], np.ndarray], W1: np.ndarray, W2, lr=0.01, step_num=100):
    for i in range(step_num):
        W1 -= lr * grad(f, W1)
        W2 -= lr * grad(f, W2)

### Activations
def sigmoid_func(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

def relu_func(x: np.ndarray):
    return np.maximum(0, x)

def id_func(x: np.ndarray):
    return x

def softmax(x: np.ndarray):
    c = np.max(x)     # オーバーフロー対策
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x)

if __name__ == '__main__':
    f = lambda W: (W * W).sum()
    W = np.float64([[1, 2], [3, 4]])
    grad_descent(f, W, lr=0.03, step_num=1000)
    print(W)
