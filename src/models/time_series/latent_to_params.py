import torch
import torch.nn as nn
import torch.nn.functional as F


class CatMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, out_dim):
        # Initialise MLP model w/correct dimensions
        super(CatMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim

        in_dims = [input_dim] + self.hidden_dims
        out_dims = self.hidden_dims + [out_dim]
        self.layers = nn.ModuleList([nn.Linear(i, o, bias=True)
                                     for i, o in zip(in_dims, out_dims)])

    def mlp_forward(self, x):
        for i in range(len(self.layers)-1):
            x = F.leaky_relu(self.layers[i](x))  # avoid dying relu issue
        x = self.layers[-1](x)  # output is on the reals
        return x

    def forward(self, user_vals, item_vals, user_bias, item_bias):
        # Concatenate as appropriate and call MLP's forward method
        latents = torch.cat((user_vals, item_vals), dim=1)
        if user_bias is not None:
            latents = torch.cat((latents, user_bias))
        if item_bias is not None:
            latents = torch.cat((latents, item_bias))

        return self.mlp_forward(latents)
