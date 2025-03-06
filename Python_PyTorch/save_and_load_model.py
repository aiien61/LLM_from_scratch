import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from icecream import ic
from typing import Tuple


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 30),

            torch.nn.ReLU(),

            torch.nn.Linear(30, 20),

            torch.nn.ReLU(),

            torch.nn.Linear(20, num_outputs)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


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

    return train_loader, test_loader


def train_model(model: NeuralNetwork, data_loader: DataLoader):
    torch.manual_seed(123)
    data: dict = make_data()
    train_loader = data_loader

    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    num_epochs: int = 3
    for epoch in range(num_epochs):
        model.train()

        for batch_index, (features, labels) in enumerate(train_loader):
            logits = model(features)

            loss = F.cross_entropy(logits, labels)  # Loss function

            # set grad = 0 in the previous epoch in order to prevent unexpected gradient accumulation
            optimizer.zero_grad()

            # computes the gradients of parameters in the loss function
            loss.backward()

            # optimiser can use the gradients to update model parameters
            optimizer.step()

            # logging
            print(f"Epoch: {epoch + 1:03d} / {num_epochs:03d}"
                  f" | Batch {batch_index:03d} / {len(train_loader):03d}"
                  f" | Train/Val Loss: {loss:.2f}")

        model.eval()

    return model


def compute_accuracy(model: NeuralNetwork, dataloader: DataLoader):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for index, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions

        correct += torch.sum(compare)

        total_examples += len(compare)

    return (correct / total_examples).item()

def save_model(model: NeuralNetwork) -> bool:
    torch.save(model.state_dict(), "./model.pth")
    return True

def load_model_state(model: NeuralNetwork, state_model_path: str) -> bool:
    model.load_state_dict(torch.load(state_model_path))
    return True

def main():
    data: dict = make_data()
    train_ds = ToyDataset(data['X_train'], data['y_train'])
    test_ds = ToyDataset(data['X_test'], data['y_test'])
    train_loader, test_loader = load_data(train_ds, test_ds)

    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model = train_model(model, train_loader)

    ic(compute_accuracy(model, train_loader))
    ic(compute_accuracy(model, test_loader))

    save_model(model)

    new_model = NeuralNetwork(2, 2)
    load_model_state(new_model, "./model.pth")
    ic(compute_accuracy(new_model, train_loader))
    ic(compute_accuracy(new_model, test_loader))

if __name__ == "__main__":
    main()
