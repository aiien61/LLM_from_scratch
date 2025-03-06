import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from icecream import ic

def make_data() -> dict:
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])

    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6]
    ])

    y_test = torch.tensor([0, 1])

    return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}

# Defines a dataset class
class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index: int) -> Tuple[list]:
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    
    def __len__(self):
        return self.labels.shape[0]
    

def load_data(train_ds: Dataset, test_ds: Dataset):
    """Loads dataset into data loader in order to send equally partial data in batch into training model"""
    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # number of process
        drop_last=True  # drop batch in the last epoch in case inconsistent size
    )

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    print("train loader:")
    for index, (x, y) in enumerate(train_loader):
        print(f"Batch {index + 1}: {x}, {y}")
    
    print("test loader:")
    for index, (x, y) in enumerate(test_loader):
        print(f"Batch {index + 1}: {x}, {y}")


    
if __name__ == "__main__":
    data: dict = make_data()

    train_ds = ToyDataset(data['X_train'], data['y_train'])
    test_ds = ToyDataset(data['X_test'], data['y_test'])

    ic(len(train_ds))
    ic(train_ds)

    load_data(train_ds, test_ds)