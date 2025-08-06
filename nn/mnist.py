import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
from nn import NN
from layer import *

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

l1 = 784 
l2 = 100
l3 = 50
l4 = 10
W1 = np.random.randn(l1, l2)  * np.sqrt(2.0 / l1)
W2 = np.random.randn(l2, l3) * np.sqrt(2.0 / l2)
W3 = np.random.randn(l3, l4) * np.sqrt(2.0 / l3)
b1 = np.zeros(l2)
b2 = np.zeros(l3)
b3 = np.zeros(l4)

layers = [Affine(W1, b1), Relu(), Affine(W2, b2), Relu(), Affine(W3, b3)]
losslayer = SoftmaxWithLoss()
nn = NN(layers, losslayer)


print(f"before train")
print(f"loss: {nn.loss(x_train, t_train)}")
print(f"accuracy: {nn.accuracy(x_train, t_train)}")

# train
batch_size = 64
epochs = 30
for epoch in range(epochs):
    indices = np.random.permutation(x_train.shape[0])
    x_train_shuffled = x_train[indices]
    t_train_shuffled = t_train[indices]
    for batch_idx in range(0, x_train.shape[0], batch_size):
        x_batch = x_train_shuffled[batch_idx:batch_idx+batch_size]
        t_batch = t_train_shuffled[batch_idx:batch_idx+batch_size]
        nn.train(x_batch, t_batch, lr=0.01, step_num=1)
    print(f"epoch: {epoch+1}, loss: {nn.loss(x_train, t_train)}")

print()
print(f"after train")
print(f"loss: {nn.loss(x_train, t_train)}")
print(f"accuracy: {nn.accuracy(x_train, t_train)}")

# test
print()
print(f"test")
print(f"loss: {nn.loss(x_test, t_test)}")
print(f"accuracy: {nn.accuracy(x_test, t_test)}")