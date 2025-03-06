import torch
from icecream import ic

class NeuralNetwork(torch.nn.Module):
    """forward() method must be implemented when torch.nn.Module is being inherited and layers is 
    being generated by Sequential() method
    """

    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),  # Arguments for the Linear layer is the number of input and output

            torch.nn.ReLU(),  # the nonlinear activatation function in between hidden layers

            # 2nd hidden layer
            torch.nn.Linear(30, 20),

            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits
    
def weights_and_bias_matrices():
    # set random seed to make the weights matrix reproduciable
    torch.manual_seed(123)

    model = NeuralNetwork(50, 3)
    ic(model)

    # See how many trainable parameters in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable model parameters: {num_params}")

    # See weights matrix in the 1st layer
    ic(model.layers[0].weight)

    # See the shape of the weights matrix in the 1st layer
    ic(model.layers[0].weight.shape)

    # See bias matrix in the 1st layer
    ic(model.layers[0].bias)

    # See the shape of the bias matrix in the 1st layer
    ic(model.layers[0].bias.shape)


def how_forward_works_with_grad_retained():
    torch.manual_seed(123)
    model = NeuralNetwork(50, 3)

    # Randomly generating a list of training dataset X
    torch.manual_seed(123)
    X = torch.rand((1, 50))
    out = model(X)
    ic(out)


def how_forward_works_without_grad_retained():
    torch.manual_seed(123)
    model = NeuralNetwork(50, 3)

    # Randomly generating a list of training dataset X
    torch.manual_seed(123)
    X = torch.rand((1, 50))

    with torch.no_grad():
        out = model(X)
    ic(out)

def how_forward_works_in_prob_without_grad_retained():
    torch.manual_seed(123)
    model = NeuralNetwork(50, 3)

    # Randomly generating a list of training dataset X
    torch.manual_seed(123)
    X = torch.rand((1, 50))

    with torch.no_grad():
        out = torch.softmax(model(X), dim=1)
    ic(out)

if __name__ == "__main__":
    # weights_and_bias_matrices()
    # how_forward_works_with_grad_retained()
    # how_forward_works_without_grad_retained()
    how_forward_works_in_prob_without_grad_retained()  # each returned values can be seen to be the prob of each class
