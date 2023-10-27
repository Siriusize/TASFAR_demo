import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, dropout=0.5):
        super().__init__()
        """
        A Simple Artificial Neural Network
        Args:
            input_size: input size of ANN (8)
            output_size: output size of ANN (1)
            hidden_sizes: neuron numbers of hidden layers
            dropout: dropout probability of ANN
        """
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.linear1 = nn.Linear(input_size, hidden_sizes[0])
        self.linear2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.linear3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.output_layer = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.linear2(x)
        x = nn.ReLU()(x)
        x = self.dropout_layer(x)
        x = self.linear3(x)
        x = nn.ReLU()(x)
        output = self.output_layer(x)
        return output
