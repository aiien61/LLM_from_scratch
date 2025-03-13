from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from rich import print
import torch
import pandas as pd

def use_dataset():
    # Define transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    # Download MNIST dataset
    mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    print(mnist_dataset)

def customize_dataset():
    class CustomDataset(Dataset):
        def __init__(self, csv_file):
            self.data = pd.read_csv(csv_file)

        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx: int):
            features = torch.tensor(self.data.iloc[idx, :-1].values, dtype=torch.float32)
            label = torch.tensor(self.data.iloc[idx, -1], dtype=torch.long)
            return features, label
    
    dataset = CustomDataset('data.csv')

def use_dataloader(dataset):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True):
    for batch in dataloader:
        features, labels = batch
        print(features.shape, labels.shape)  # e.g. (32, n_features), (32,)

    