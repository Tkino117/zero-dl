import numpy as np
from typing import Callable

def diff(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray):
    dx = 1e-4 
    return (f(x + dx) - f(x - dx)) / (2 * dx)

def grad_naive(f: Callable[[np.ndarray], float], x: np.ndarray):
    x = x.astype(np.float64)
    d = 1e-4
    res = np.zeros_like(x)
    for i in range(len(x)):
        for j in range(len(x[0])):
            dx = np.zeros_like(x[i])
            dx[j] += d
            res[i][j] = (f(x[i] + dx) - f(x[i] - dx)) / (2 * d)
    return res

# f: R^n -> R (supports multiple input), x: (.., n), res: (..)
def grad(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray):
    x = x.astype(np.float64)
    d = 1e-4
    res = np.zeros_like(x)
    # 入力が1つのとき, 複数の時
    if (x.ndim == 1):
        for j in range(len(x)):
            dx = np.zeros_like(x)
            dx[j] += d
            res[j] = (f(x + dx) - f(x - dx)) / (2 * d)
    else:
        for j in range(len(x[0])):
            dx = np.zeros_like(x)
            dx[:, j] += d
            res[:, j] = (f(x + dx) - f(x - dx)) / (2 * d)
    return res

def grad_descent(f: Callable[[np.ndarray], np.ndarray], init_x: np.ndarray, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        x -= grad(f, np.array(x)) * lr
    return x

if __name__ == '__main__':
    f = lambda x: np.sum(x**2, axis=np.ndim(x)-1)
    print(grad_descent(f, [10000, 10000], step_num=10000))
    # X, Y = np.meshgrid(np.arange(-2, 3, 1), np.arange(-2, 3, 1))
    # points = np.transpose([X.flatten(), Y.flatten()])
    # print(grad(f, points))

