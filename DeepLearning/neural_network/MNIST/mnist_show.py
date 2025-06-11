import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from mnist import load_mnist
from icecream import ic
from PIL import Image

# Option1
# def show_image(image):
#     image_scaled = (image * 255).astype(np.uint8)
#     pil_image = Image.fromarray(image_scaled)
#     pil_image.show()

# Option2
def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    """
    normalize: if true, convert 0 ~ 255 to 0 ~ 1
    flatten: if false, remain 3-dimension shape (1, 28, 28); if true, convert to 1-dimension (784,)  28 x 28 = 784
    one_hot_label: if true, convert to one-hot encoding e.g. 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    """
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)

    ic(x_train.shape)
    ic(t_train.shape)
    ic(x_test.shape)
    ic(x_test.shape)

    image = x_train[0]
    label = t_train[0]
    ic(label)

    ic(image.shape)
    image = image.reshape(28, 28) # convert to image original shape: (784,) -> (28, 28)
    ic(image.shape)

    show_image(image)
