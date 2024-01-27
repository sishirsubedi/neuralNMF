import torch.nn as nn

from .nmf import nmfl


class nmfNET(nn.Module):
    def __init__(self,layers):
        super(nmfNET, self).__init__()
        self.layers = layers
        self.depth = len(self.layers)
        self.nmflist = nn.ModuleList([nmfl(self.layers[i], self.layers[i+1]) for i in range(self.depth-1)])
    
    def forward(self,X):
        betalist = []
        for lindex in range(self.depth-1):
            X = self.nmflist[lindex](X)
            betalist.append(X)
        return betalist
    