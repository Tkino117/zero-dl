import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from activations import *

class perceptron:
    def __init__(self, w: np.ndarray, b: np.ndarray, act: Callable[[np.ndarray], np.ndarray]):
        self.w = w
        self.b = b
        self.act = act
    def cul(self, x: np.ndarray):
        return self.act(x @ self.w + self.b)

w1 = np.array([[1, 2, 3], [1, -2, 6]])
b1 = np.array([-5, -5, 4])
p1 = perceptron(w1, b1, sigmoid_func)

x = np.array([1, 2])
print(p.cul(x))

# act(xw + b)