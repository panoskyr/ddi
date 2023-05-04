import math
import torch
from torch_geometric.nn import SAGEConv 

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv=SAGEConv(
            
        )