import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class Recon_Loss_Func(nn.Module):

    def __init__(self, lambd = 0):
        super(Recon_Loss_Func, self).__init__()
        self.lambd = lambd
        self.criterion = Fro_Norm()
            
    def forward(self, net, X, beta_list):
        """
        Runs the forward pass of the energy loss function.
        Parameters
        ----------
        net: Pytorch module Neural NMF object
            The Neural NMF object for which the loss is calculated.
        X: Pytorch tensor
            The input to the Neural NMF network (matrix to be factorized).
        S_lst: list
            All S matrices ([S_0, S_1, ..., S_L]) that were returned by the forward pass of the Neural 
            NMF object.
        Returns
        -------
        reconstructionloss: Pytorch tensor
            The total energy loss from X, the S matrices, and the A matrices, stored in a 1x1 Pytorch 
            tensor to preserve information for backpropagation.
        """

        depth = net.depth
        
        X_approx = beta_list[-1]
        for i in range(depth-2, -1, -1):
            X_approx = torch.mm(net.nmflist[i].theta,X_approx)
        
        reconstructionloss = self.criterion(X_approx, X)

        return reconstructionloss




class Fro_Norm(nn.Module):
    """
    Calculate the Frobenius norm between two matrices of the same size. This function actually returns 
    the entrywise average of the square of the Frobenius norm. 
    Examples
    --------
    >>> criterion = Fro_Norm()
    >>> loss = criterion(X1,X2)
    """
    def __init__(self):
        super(Fro_Norm, self).__init__()
        self.criterion = nn.MSELoss()
    def forward(self,X1, X2):
        """
        Runs the forward pass of the Frobenius norm module
        Parameters
        ----------
        X1: Pytorch tensor
            The first input to the Frobenius norm loss function
        X2: Pytorch tensor
            The second input to the Frobenius norm loss function
        Returns
        -------
        loss: Pytorch tensor
            The Frobenius norm of X1 and X2, stored in a 1x1 Pytorch tensor to preserve information for 
            backpropagation.
        """
        len1 = torch.numel(X1.data)
        len2 = torch.numel(X2.data)
        assert(len1 == len2)
        X = X1 - X2
        #X.contiguous()
        loss =  self.criterion(X.reshape(len1), Variable(torch.zeros(len1).double().to('cpu')))
        return loss
