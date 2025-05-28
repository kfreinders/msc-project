import torch.nn as nn


class NeuralNetwork(nn.Module):
    """
    Fully connected feedforward neural network.

    This network consists of a configurable number of hidden layers with
    ReLU activation functions and dropout regularization. It is designed
    for regression tasks, such as predicting continuous parameters from
    input features.

    Parameters
    ----------
    input_dim : int
        Dimension of the input features.
    output_dim : int
        Dimension of the output predictions.
    hidden_size : int, optional
        Number of neurons in each hidden layer (default is 128).
    num_hidden_layers : int, optional
        Number of fully connected layers between the input and output layers
        (default is 3).
    dropout_rate : float, optional
        Dropout probability applied after each hidden layer (default is 0.2).

    Attributes
    ----------
    net : torch.nn.Sequential
        Sequential container holding the entire network architecture.

    Methods
    -------
    forward(x)
        Perform a forward pass through the network.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_size=128,
        num_hidden_layers=3,
        dropout_rate=0.2
    ):
        super(NeuralNetwork, self).__init__()

        layers = []
        for i in range(num_hidden_layers):
            in_dim = input_dim if i == 0 else hidden_size
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_size, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
