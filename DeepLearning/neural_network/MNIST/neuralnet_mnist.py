import sys
import os
import pickle
import numpy as np
from icecream import ic
from rich import print
from mnist import load_mnist
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activation_function import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

def analyze_weights(network) -> None:
    weights = network['W1'], network['W2'], network['W3']
    print({f'W{i}.shape': W.shape for i, W in enumerate(weights, 1)})


def evaluate(x_test: np.ndarray, t_test: np.ndarray, network: dict, batch_size: int) -> float:
    accuracy_cnt = 0
    for i in range(0, len(x_test), batch_size):
        x_batch = x_test[i: i + batch_size]
        
        # y: an array of probability of each number
        y_batch = predict(network, x_batch)

        #  p: index of max value in array y
        p = np.argmax(y_batch, axis=1)
        print(f"p: {p}, t: {t_test[i: i + batch_size]}")

        accuracy_cnt += np.sum(p == t_test[i: i + batch_size])

    return float(accuracy_cnt / len(x_test))

    


if __name__ == "__main__":
    x, t = get_data()
    network = init_network()

    ic(x.shape)
    analyze_weights(network)

    accuracy = evaluate(x, t, network, batch_size=100)
    print(f"Accuracy: {accuracy}")
