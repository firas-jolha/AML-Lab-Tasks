import torch.nn as nn
import torch

class Regressor(nn.Module):
    def __init__(self, input_dim=50, output_dim=1):
        super(Regressor, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 25)
        self.output = nn.Linear(25, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x)) 
        x = torch.relu(self.layer2(x))
        x = self.output(x)
        return x
