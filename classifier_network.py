import torch.nn.functional as F
from torch import nn


# Model class
# Make the hidden layers to be some parameters to be able to test multiple configurations
class ClassifierNetwork(nn.Module):
    """Classifier model

    Args:
        input_size (int): Size of the input
        output_size (int): Size of the output
        hidden_layers (list of int): A list of the sizes of the hidden layers
        drop_p (float, optional): The dropout of the layers in the  network
    """

    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        super().__init__()

        # build the last layer of the network
        self.input = nn.Linear(input_size, hidden_layers[0])

        self.hidden_layers = nn.ModuleList([])
        layers_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(l1, l2) for l1, l2 in layers_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)

        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)

        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        return x
