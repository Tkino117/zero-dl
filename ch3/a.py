import sys, os
sys.path.append(os.pardir) # 親ディレクトリをパスに追加
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def show_img(img: np.ndarray):
    """ img: 2dim array """
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)
img = x_train[0].reshape(28, -1)
pil_img = Image.fromarray(img)
resized_img = pil_img.resize((500, 500))
resized_img.show()