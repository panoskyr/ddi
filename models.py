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
    return acc.item()

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

## the predictor takes two tensors
## of node embeding h1 h2
## with size [batch_size, embed_dim]
class DotProductLinkPredictor(torch.nn.Module):
    def __init__(self):
        super(DotProductLinkPredictor, self).__init__()

    def forward(self, x_i, x_j):
        #print("preidctore input: ", x_i.shape,x_j.shape)
        # multiply the embeddings of a src_node and a dest_node
        # for each edge in the batch
        # then sum the result per edge
        # example of x_i*x_j:
        # the first half are real edges and the second half are fake edges
        # x_i=[emb1, emb2, emb3, emb4]
        # x_j=[emb5, emb6, emb7, emb8]
        # x_i*x_j=[emb1*emb5, emb2*emb6, emb3*emb7]
        # out=[sum(emb1*emb5), sum(emb2*emb6), sum(emb3*emb7)]
        # this is the way to combine the info (in the embeddings) that two nodes are connected or not
        # then pass the result to the sigmoid
        out = (x_i*x_j).sum(-1)
        return torch.sigmoid(out)
    
    def reset_parameters(self):
      pass


# a simple NN with at least 2 fully connected layers
class NeuralLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(NeuralLinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        print("squeeze shape: " ,torch.sigmoid(x).squeeze().shape,"nos queeze shape: ", torch.sigmoid(x).shape)
        return torch.sigmoid(x).squeeze()
