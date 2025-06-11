# coding: utf-8
import urllib.request
import os.path
import gzip
import pickle
import os
import numpy as np
# url_base = 'http://yann.lecun.com/exdb/mnist/'
url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'  # 鏡像網站
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}
dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"
train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    if os.path.exists(file_path):
        return
    print("正在下載 " + file_name + " ... ")
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0"}
    request = urllib.request.Request(url_base+file_name, headers=headers)
    response = urllib.request.urlopen(request).read()
    with open(file_path, mode='wb') as f:
        f.write(response)
    print("完成")


def download_mnist():
    for v in key_file.values():
       _download(v)


def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    print("正在將 " + file_name + " 轉換為 NumPy 陣列...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("完成")
    return labels


def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    print("正在將 " + file_name + " 轉換為 NumPy 陣列...")
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("完成")
    return data


def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    return dataset


def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("正在建立 pickle 檔案...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("完成！")


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """載入 MNIST 資料集
    參數
    ----------
    normalize : 將影像像素值正規化為 0.0~1.0
    one_hot_label :
        當 one_hot_label 為 True 時，標籤以 one-hot 陣列形式回傳
        one-hot 陣列是指像 [0,0,1,0,0,0,0,0,0,0] 這樣的陣列
    flatten : 是否將影像平坦化為一維陣列
    回傳值
    -------
    (訓練影像, 訓練標籤), (測試影像, 測試標籤)
    """
    if not os.path.exists(save_file):
        init_mnist()
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
    init_mnist()
