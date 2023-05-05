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
    

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim,dtype=torch.float64)

    def forward(self, inputs):
        x = self.linear(inputs)
        out=torch.sigmoid(x)
        return out
