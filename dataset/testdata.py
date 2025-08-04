import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

save_file = './testdata.pkl'
def make_data():
    x = np.random.rand(3 * 100).reshape(-1, 3) * 2 - 1
    t = np.array([1 if (xi**2).sum() > 1 else 0 for xi in x])
    split = int(0.8 * len(x)) 
    dataset = {}
    dataset['x_train'] = x[:split]
    dataset['t_train'] = t[:split]
    dataset['x_test'] = x[split:]
    dataset['t_test'] = t[split:]
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
        print("made data and dumpped!")
    # ax = plt.axes(projection='3d')
    # ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=t)
    # plt.show()

def load_data():
    if not os.path.exists(save_file):
        make_data()
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

if __name__ == '__main__':
    dataset = load_data()
    ax = plt.axes(projection='3d')
    x = dataset['x_test']
    t = dataset['t_test']
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=t)
    plt.show()