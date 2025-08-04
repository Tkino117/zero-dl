import numpy as np
import matplotlib.pyplot as plt

def step_func(x: np.ndarray):
    return (x > 0).astype(np.int16)

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

def main():
    print(softmax(np.array([0.3, 2.9, 4.0])))

if __name__ == "__main__":
    main()