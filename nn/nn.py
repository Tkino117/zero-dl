import sys, os
sys.path.append(os.pardir)
import numpy as np
from layer import *
from testdata import load_data

class NN:
    def __init__(self, layers, losslayer):
        self.layers = layers
        self.losslayer = losslayer

    def loss(self, X, T):
        for layer in self.layers:
            X = layer.forward(X)
        loss = self.losslayer.forward(X, T)
        return loss

    def accuracy(self, X, T):
        for layer in self.layers:
            X = layer.forward(X)
        acc = self.losslayer.accuracy(X, T)
        return acc

    def train(self, X, T, lr=0.01, step_num=1000):
        for i in range(step_num):
            loss = self._train(X, T, lr)
            # if i % 10 == 0:
                # print(f"loss: {loss} (step={i})")

    def _train(self, X, T, lr=0.01):
        # 純伝搬
        loss = self.loss(X, T)

        # 逆伝搬
        dX = self.losslayer.backward()
        for layer in self.layers[::-1]:
            dX = layer.backward(dX)
            
        # 勾配降下
        for layer in self.layers[::-1]:
            if isinstance(layer, Affine):
                layer.grad_descent(lr=lr)
        
        return loss


if __name__ == '__main__':
    dataset = load_data()
    x_train = dataset['x_train']
    t_train = dataset['t_train']
    x_test = dataset['x_test']
    t_test = dataset['t_test']

    l1 = 3
    l2 = 20
    l3 = 7
    l4 = 2
    W1 = np.random.randn(l1, l2) 
    W2 = np.random.randn(l2, l3)
    W3 = np.random.randn(l3, l4) 
    b1 = np.zeros(l2)
    b2 = np.zeros(l3)
    b3 = np.zeros(l4)

    layers = [Affine(W1, b1), Relu(), Affine(W2, b2), Relu(), Affine(W3, b3)]
    losslayer = SoftmaxWithLoss()
    nn = NN(layers, losslayer)


    print(f"before train")
    print(f"loss: {nn.loss(x_train, t_train)}")
    print(f"accuracy: {nn.accuracy(x_train, t_train)}")
    # print(np.transpose([nn.forward(x_train)[:, 0], t_train]))
    nn.train(x_train, t_train, lr=0.05, step_num=10000)
    print()
    print(f"after train")
    print(f"loss: {nn.loss(x_train, t_train)}")
    print(f"accuracy: {nn.accuracy(x_train, t_train)}")
    # print(np.transpose([nn.forward(x_train)[:, 0], t_train]))
    print()
    print(f"test")
    print(f"loss: {nn.loss(x_test, t_test)}")
    print(f"accuracy: {nn.accuracy(x_test, t_test)}")



