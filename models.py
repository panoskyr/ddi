import math
import torch
from torch_geometric.nn import SAGEConv 
import torch.nn.functional as F

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
    
def BinaryAccuracy(preds, true_labels):
    rounded_preds = torch.round(preds)
    # the equality operator on tensors returns True/False
    correct = (rounded_preds == true_labels).float()
    acc = correct.sum() / len(correct)
    return acc

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, aggr="add"):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=True, aggr=aggr))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=True, aggr=aggr))
        self.convs.append(SAGEConv(hidden_channels, out_channels, normalize=True, aggr=aggr))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class DotProductLinkPredictor(torch.nn.Module):
    def __init__(self):
        super(DotProductLinkPredictor, self).__init__()

    def forward(self, x_i, x_j):
        out = (x_i*x_j).sum(-1)
        return torch.sigmoid(out)
    
    def reset_parameters(self):
      pass