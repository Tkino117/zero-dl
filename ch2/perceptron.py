import numpy as np
class perceptron:
    def __init__(self, w1, w2, b):
        self.w = np.array([w1, w2])
        self.b = b
    def cul(self, x, y):
        input = np.array([x, y])
        return 1 if np.dot(self.w, input) + self.b > 0 else 0

p_and = perceptron(0.5, 0.5, -0.7)
p_or = perceptron(0.5, 0.5, -0.3)
p_nand = perceptron(-0.5, -0.5, 0.7)

def p_xor(x, y):
    nx = p_nand.cul(x, y)
    ny = p_or.cul(x, y)
    return p_and.cul(nx, ny)

input = [[0, 0], [0, 1], [1, 0], [1, 1]]
for i in input:
    print(p_xor(i[0], i[1]))