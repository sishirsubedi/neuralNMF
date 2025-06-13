import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable


import numpy as np
from scipy.optimize import nnls 

class factorization(torch.autograd.Function):

    @staticmethod
    def forward(ctx,X,theta):
        
        beta = calculate_beta(theta.data,X.data)
        
        ctx.save_for_backward(X, theta, None)
        ctx.intermediate = beta
        
        return beta

    @staticmethod
    # TODO understand this once_differentiable function
    @once_differentiable #without this line then in backprop all the operations should be differentiable
    def backward(ctx, grad_output):
        X, theta, _ = ctx.saved_tensors
        grad_X = grad_theta = None
        output = ctx.intermediate
        if ctx.needs_input_grad[0]:
            grad_X = calc_grad_X(grad_output, theta.data, output)
        if ctx.needs_input_grad[1]:
            grad_theta = calc_grad_A(grad_output, theta.data, output, X.data)
        return grad_X, grad_theta, None
    
def calculate_beta(theta,X,mode='lsq'):
    X = X.numpy()
    theta = theta.numpy()
    m = X.shape[0]
    n = X.shape[1]
    k = theta.shape[1]
    
    beta = np.zeros([k,n])
        
    if mode =='lsq':
        for i in range(n):
            x = X[:,i]
            [b,res] = nnls(theta,x)
            beta[:,i] = b
            
    elif mode == 'pmf':
        from .pmf import altPMF
        model = altPMF(n_components=theta.shape[1],max_iter=25)
        res_beta = model.predict_beta(X,prev_theta=theta)
        beta = res_beta['ebeta']
                
    # mean = beta.mean(axis=1)
    # var = beta.var(axis=1, ddof=0)  
    # scaler = np.sqrt(1 / var)
    # beta_std = (beta.T - mean) * scaler + mean
    # beta_std = beta_std.T
        
    beta = torch.from_numpy(beta).double()
    
    return beta

def calc_grad_X(grad_S, A, S):
    """
    Calculates the gradient of q(X,A) with respect to X.

    Parameters
    ---------
    grad_S: PyTorch tensor
        The gradient of S = q(X,A) passed on from the last layer.
    A: PyTorch tensor
        The A matrix used to compute q(X,A).
    S: PyTorch tensor
        The output S = q(X,A).

    Returns
    -------
    grad_X: PyTorch tensor
        The gradient of q(X,A) with respsect to X.

    """
    A_np = A.numpy()
    S_np = S.numpy()
    grad_S_np = grad_S.numpy()
    m = A.shape[0]
    k = A.shape[1]
    n = S.shape[1]
    grad_X = np.zeros([m,n])
    for i in range(n):
        s = S_np[:,i]
        supp = s!=0
        grad_s_supp = grad_S_np[supp,i]
        A_supp = A_np[:,supp]
        grad_X[:,i] = np.linalg.pinv(A_supp).T@grad_s_supp
    grad_X = torch.from_numpy(grad_X).double()
    return grad_X

def calc_grad_A(grad_S, A, S, X):
    """
    Calculates the gradient of q(X,A) with respect to A.

    Parameters
    ---------
    grad_S: PyTorch tensor
        The gradient of S = q(X,A) passed on from the last layer.
    A: PyTorch tensor
        The A matrix used to compute q(X,A).
    S: PyTorch tensor
        The output S = q(X,A).
    X: PyTorch tensor
        The X matrix used to compute q(X,A).

    Returns
    -------
    grad_A: PyTorch tensor
        The gradient of q(X,A) with respsect to A.

    """

    A_np = A.numpy()
    S_np = S.numpy()
    grad_S_np = grad_S.numpy()
    X_np = X.cpu().numpy()
    m = A.shape[0]
    k = A.shape[1]
    n = S.shape[1]
    grad_A = np.zeros([m,k])
    for l in range(n):
        s = S_np[:,l]
        supp = s!=0
        A_supp = A_np[:,supp]
        grad_s_supp = grad_S_np[supp,l:l+1]
        x = X_np[:,l:l+1]
        A_supp_inv = np.linalg.pinv(A_supp)
        part1 = -(A_supp_inv.T@grad_s_supp)@(x.T@A_supp_inv.T)
        part2 = (x - A_supp@(A_supp_inv@x))@((grad_s_supp.T@A_supp_inv)@A_supp_inv.T)
        grad_A[:,supp] += part1 + part2
    grad_A = torch.from_numpy(grad_A).double()
    return grad_A


class nmfl(nn.Module):
    def __init__(self,m,k):
        super(nmfl, self).__init__()
        self.m = m
        self.k = k
        self.theta = nn.Parameter(torch.DoubleTensor(m,k))
        self.theta.data = torch.abs(torch.rand(m,k,dtype = torch.double)) 
        
    def forward(self,X):
        
        # mean = self.theta.data.mean(dim=0)
        # var = self.theta.data.var(dim=0, unbiased=False)  
        # scaler = torch.sqrt(1 / var)
        # theta_std = (self.theta.data - mean) * scaler + mean
        # self.theta.data = theta_std

        self.theta.data = torch.clamp(self.theta.data, min=0)
        
        return factorization.apply(X, self.theta)

