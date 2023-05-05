import math
import torch
from torch_geometric.nn import SAGEConv 

class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.fullyconnected1=torch.nn.Linear(in_features=3, out_features=256, bias=True)
        self.fullyconnected2=torch.nn.Linear(in_features=256,out_features=1, bias=True)

    def forward(self, x):
        x=self.fullyconnected1(x)
        x=torch.nn.functional.relu(x)
        x=self.fullyconnected2(x)

        output=torch.sigmoid(x)
        return output
