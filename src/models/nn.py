import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, bias=True):
        """A Multilayer Perceptron class

        Parameters
        ----------
            input_dim (int): length of input data
            hidden_dims (list of ints): size of each hidden layer
            output_dim (int): dimensionality of output
        """
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        in_dims = [input_dim] + self.hidden_dims
        out_dims = self.hidden_dims + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(i, o, bias=bias)
                                     for i,o in zip(in_dims, out_dims)])

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = F.leaky_relu(self.layers[i](x)) # avoid dying relu issue
        x = self.layers[-1](x) #output is on the reals

        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
