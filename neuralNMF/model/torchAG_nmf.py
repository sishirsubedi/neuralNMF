'''
A PyTorch implementation on Non-negative Matrix Factorization
using torch AUTOGRAD.

source-
PyTorch-NMF based on torch.nn.Module, so the models can be moved freely among CPU/GPU devices and utilize parallel computation of cuda. We also utilize the computational graph from torch.autograd to derive updated coefficients so the amount of codes is reduced and easy to maintain.

https://github.com/yoyololicon/pytorch-NMF/blob/master/torchnmf/nmf.py

'''

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from typing import Union, Iterable, Optional, Tuple
from .nmf_models._metrics import beta_div
from tqdm import tqdm
from .nmf_models._constants import eps

__all__ = [
    'BaseComponent', 'NMF'
]


@torch.jit.script
def _proj_func(s: Tensor,
               k1: float,
               k2: float) -> Tensor:
    s_shape = s.size()
    s = s.reshape(-1)
    N = s.numel()
    v = s + (k1 - s.sum()) / N

    zero_coef = torch.zeros(N, dtype=torch.bool, device=s.device)
    while True:
        m = k1 / (N - zero_coef.count_nonzero())
        w = torch.where(~zero_coef, v - m, v)
        a = w @ w
        b = 2 * w @ v
        c = v @ v - k2
        alphap = (-b + (b * b - 4 * a * c).relu().sqrt()) * 0.5 / a
        v.add_(w, alpha=alphap.item())

        mask = v < 0
        if not torch.any(mask):
            break

        zero_coef |= mask
        v.relu_()
        v += (k1 - v.sum()) / (N - zero_coef.count_nonzero())
        v.relu_()

    return v.view(s_shape)


def _double_backward_update(V: Tensor,
                            WH: Tensor,
                            param: Parameter,
                            beta: float,
                            gamma: float,
                            l1_reg: float,
                            l2_reg: float,
                            pos: Tensor = None):
    param.grad = None
    if beta == 2:
        output_neg = V
        output_pos = WH
    elif beta == 1:
        output_neg = V / WH.add(eps)
        output_pos = None
    elif beta == 0:
        WH_eps = WH.add(eps)
        output_pos = WH_eps.reciprocal_()
        output_neg = output_pos.square().mul_(V)
    else:
        WH_eps = WH.add(eps)
        output_neg = WH_eps.pow(beta - 2).mul_(V)
        output_pos = WH_eps.pow_(beta - 1)

    # first backward
    WH.backward(output_neg, retain_graph=pos is None)
    neg = param.grad.relu_().add_(eps)

    if pos is None:
        param.grad = None
        WH.backward(output_pos)
        pos = param.grad.relu_().add_(eps)

    if l1_reg > 0:
        pos.add_(l1_reg)
    if l2_reg > 0:
        pos = pos.add(param.data, alpha=l2_reg)
    multiplier = neg.div_(pos)
    if gamma != 1:
        multiplier.pow_(gamma)
    param.data.mul_(multiplier)


def _sp_double_backward_update(pos_out: Tensor,
                               neg_out: Tensor,
                               param: Parameter,
                               gamma: float,
                               l1_reg: float,
                               l2_reg: float,
                               pos: Tensor = None):
    param.grad = None
    # first backward
    neg_out.backward()
    neg = param.grad.relu_().add_(eps)

    if pos is None:
        param.grad = None
        pos_out.backward()
        pos = param.grad.relu_().add_(eps)

    if l1_reg > 0:
        pos.add_(l1_reg)
    if l2_reg > 0:
        pos = pos.add(param.data, alpha=l2_reg)
    multiplier = neg.div_(pos)
    if gamma != 1:
        multiplier.pow_(gamma)
    param.data.mul_(multiplier)


def _get_W_kl_positive(H: Tensor) -> Tensor:
    sum_dims = list(range(H.dim()))
    sum_dims.remove(1)
    return H.sum(sum_dims, keepdims=True)


def _get_H_kl_positive(W: Tensor) -> Tensor:
    sum_dims = list(range(W.dim()))
    sum_dims.remove(1)
    return W.sum(sum_dims, keepdims=True).squeeze(0)


def _get_norm(x: Tensor,
              axis: int = 1) -> Tensor:
    x2 = x * x
    sum_dims = list(range(x2.dim()))
    sum_dims.remove(axis)
    return x2.sum(sum_dims).sqrt()


@torch.no_grad()
def _renorm(W: Tensor,
            H: Tensor,
            unit_norm='W'):
    if unit_norm == 'W':
        W_norm = _get_norm(W)
        slicer = (slice(None),) + (None,) * (W.dim() - 2)
        W /= W_norm[slicer]
        slicer = (slice(None),) + (None,) * (H.dim() - 2)
        H *= W_norm[slicer]
    elif unit_norm == 'H':
        H_norm = _get_norm(H)
        slicer = (slice(None),) + (None,) * (H.dim() - 2)
        H /= H_norm[slicer]
        slicer = (slice(None),) + (None,) * (W.dim() - 2)
        W *= H_norm[slicer]
    else:
        raise ValueError("Input type isn't valid!")


def _get_V_norm(V: Tensor, beta: float):
    assert V.is_coalesced()
    if beta == 2:
        return V.values() @ V.values() * 0.5
    elif beta == 1:
        return V.values() @ V.values().log() - V.values().sum()
    else:
        V_vals = V.values()
        return V_vals.pow(beta).sum() / beta / (beta - 1)


class BaseComponent(torch.nn.Module):
    r"""Base class for all NMF modules.

    You can't use this module directly.
    Your models should also subclass this class.

    Args:
        rank (int): size of hidden dimension
        W (tuple or Tensor): size or initial weights of template tensor W
        H (tuple or Tensor): size or initial weights of activation tensor H
        trainable_W (bool):  controls whether template tensor W is trainable when initial weights is given. Default: ``True``
        trainable_H (bool):  controls whether activation tensor H is trainable when initial weights is given. Default: ``True``

    Attributes:
        W (Tensor or None): the template tensor of the module if corresponding argument is given.
            If size is given, values are initialized non-negatively.
        H (Tensor or None): the activation tensor of the module if corresponding argument is given.
            If size is given, values are initialized non-negatively.

       """
    __constants__ = ['rank']
    __annotations__ = {'W': Optional[Tensor],
                       'H': Optional[Tensor],
                       'out_channels': Optional[int],
                       'kernel_size': Optional[Tuple[int, ...]]}

    rank: int
    W: Optional[Tensor]
    H: Optional[Tensor]
    out_channels: Optional[int]
    kernel_size: Optional[Tuple[int, ...]]

    def __init__(self,
                 rank: int = None,
                 W: Union[Iterable[int], Tensor] = None,
                 H: Union[Iterable[int], Tensor] = None,
                 trainable_W: bool = True,
                 trainable_H: bool = True):
        super().__init__()

        infer_rank = None
        if isinstance(W, Tensor):
            assert torch.all(W >= 0.), "Tensor W should be non-negative."
            self.register_parameter('W', Parameter(
                torch.empty(*W.size()), requires_grad=trainable_W))
            self.W.data.copy_(W)
            infer_rank = self.W.shape[1]

        if isinstance(H, Tensor):
            assert torch.all(H >= 0.), "Tensor H should be non-negative."
            H_shape = H.shape
            self.register_parameter('H', Parameter(
                torch.empty(*H_shape), requires_grad=trainable_H))
            self.H.data.copy_(H)
            infer_rank = self.H.shape[1]

        if infer_rank is None:
            assert rank, "A rank should be given when W and H are not available!"
        else:
            if getattr(self, "H") is not None:
                assert self.H.shape[1] == infer_rank, "Latent size of H does not match with others!"
            if getattr(self, "W") is not None:
                assert self.W.shape[1] == infer_rank, "Latent size of W does not match with others!"
                self.out_channels = self.W.shape[0]
                if self.W.ndim > 2:
                    self.kernel_size = self.W.shape[2:]
            rank = infer_rank

        self.rank = rank

    def extra_repr(self) -> str:
        s = ('{rank}')
        if self.W is not None:
            s += ', out_channels={out_channels}'
            if hasattr(self, 'kernel_size'):
                s += ', kernel_size={kernel_size}'
        return s.format(**self.__dict__)

    def forward(self, H: Tensor = None, W: Tensor = None) -> Tensor:
        r"""An outer wrapper of :meth:`self.reconstruct(H,W) <torchnmf.nmf.BaseComponent.reconstruct>`.

        .. note::
                Should call the :class:`BaseComponent` instance afterwards
                instead of this since the former takes care of running the
                registered hooks while the latter silently ignores them.

        Args:
            H(Tensor, optional): input activation tensor H. If no tensor was given will use :attr:`H` from this module
                                instead
            W(Tensor, optional): input template tensor W. If no tensor was given will use :attr:`W` from this module
                                instead

        Returns:
            Tensor: tensor
        """
        if H is None:
            H = self.H
        if W is None:
            W = self.W
        assert H is not None
        assert W is not None
        return self.reconstruct(H, W)

    @staticmethod
    def reconstruct(H: Tensor, W: Tensor) -> Tensor:
        r"""Defines the computation performed at every call.

            Should be overridden by all subclasses.
            """
        raise NotImplementedError

    def _sp_recon_beta_pos_neg(self, V, H, W, beta):
        raise NotImplementedError

    @torch.jit.ignore
    def fit(self,
            V: Tensor,
            beta: float = 1,
            tol: float = 1e-4,
            max_iter: int = 200,
            verbose: bool = False,
            alpha: float = 0,
            l1_ratio: float = 0
            ) -> int:
        r"""Learn a NMF model for the data V by minimizing beta divergence.

        To invoke this function, attributes :meth:`H <torchnmf.nmf.BaseComponent.H>` and
        :meth:`W <torchnmf.nmf.BaseComponent.W>` should be presented in this module.

        Args:
            V (Tensor): data tensor to be decomposed. Can be a sparse tensor returned by :func:`torch.sparse_coo_tensor` 
            beta (float): beta divergence to be minimized, measuring the distance between V and the NMF model.
                        Default: ``1.``
            tol (float): tolerance of the stopping condition. Default: ``1e-4``
            max_iter (int): maximum number of iterations before timing out. Default: ``200``
            verbose (bool): whether to be verbose. Default: ``False``
            alpha (float): constant that multiplies the regularization terms. Set it to zero to have no regularization
                            Default: ``0``
            l1_ratio (float):  the regularization mixing parameter, with 0 <= l1_ratio <= 1.
                                For l1_ratio = 0 the penalty is an elementwise L2 penalty (aka Frobenius Norm).
                                For l1_ratio = 1 it is an elementwise L1 penalty.
                                For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2. Default: ``0``

        Returns:
            int: total number of iterations
        """
        assert torch.all((V._values() if V.is_sparse else V) >=
                         0.), "Target should be non-negative."

        if (V.is_sparse or V.min() == 0) and beta <= 0:
            raise ValueError("When beta <= 0 and V contains zeros, "
                             "the training process may diverge. "
                             "Please add small values to "
                             "V, or use a positive beta value.")

        W = self.W
        H = self.H

        if beta < 1:
            gamma = 1 / (2 - beta)
        elif beta > 2:
            gamma = 1 / (beta - 1)
        else:
            gamma = 1

        l1_reg = alpha * l1_ratio
        l2_reg = alpha * (1 - l1_ratio)

        if V.is_sparse:
            V = V.coalesce()
            V_norm = _get_V_norm(V, beta)

        with torch.no_grad():
            if V.is_sparse:
                pos, neg = self._sp_recon_beta_pos_neg(V, H, W, beta)
                loss_init = V_norm + pos - neg
            else:
                WH = self.reconstruct(H, W)
                loss_init = beta_div(WH, V, beta)
        loss_init = loss_init.mul(2).sqrt().item()
        previous_loss = loss_init

        with tqdm(total=max_iter, disable=not verbose) as pbar:
            for n_iter in range(max_iter):
                if W.requires_grad:
                    precomputed_pos = _get_W_kl_positive(
                        H.detach()) if beta == 1 else None
                    if V.is_sparse:
                        pos, neg = self._sp_recon_beta_pos_neg(
                            V, H.detach(), W, beta)
                        _sp_double_backward_update(
                            pos, neg, W, gamma, l1_reg, l2_reg, precomputed_pos)
                    else:
                        WH = self.reconstruct(H.detach(), W)
                        _double_backward_update(
                            V, WH, W, beta, gamma, l1_reg, l2_reg, precomputed_pos)

                if H.requires_grad:
                    precomputed_pos = _get_H_kl_positive(
                        W.detach()) if beta == 1 else None
                    if V.is_sparse:
                        pos, neg = self._sp_recon_beta_pos_neg(
                            V, H, W.detach(), beta)
                        _sp_double_backward_update(
                            pos, neg, H, gamma, l1_reg, l2_reg, precomputed_pos)
                    else:
                        WH = self.reconstruct(H, W.detach())
                        _double_backward_update(
                            V, WH, H, beta, gamma, l1_reg, l2_reg, precomputed_pos)

                if n_iter % 10 == 9:
                    with torch.no_grad():
                        if V.is_sparse:
                            pos, neg = self._sp_recon_beta_pos_neg(
                                V, H, W, beta)
                            loss = V_norm + pos - neg
                        else:
                            WH = self.reconstruct(H, W)
                            loss = beta_div(WH, V, beta)
                        loss = loss.mul(2).sqrt().item()
                    pbar.set_postfix(loss=loss)
                    pbar.update(10)
                    if (previous_loss - loss) / loss_init < tol:
                        break
                    previous_loss = loss

        return n_iter + 1

    @torch.jit.ignore
    def sparse_fit(self,
                   V,
                   beta=2,
                   max_iter=200,
                   verbose=False,
                   sW=None,
                   sH=None,
                   ) -> int:
        r"""Learn a NMF model for the data V by minimizing beta divergence with sparseness constraints proposed in
        `Non-negative Matrix Factorization with Sparseness Constraints`_.

        To invoke this function, attributes :meth:`H <torchnmf.nmf.BaseComponent.H>` and
        :meth:`W <torchnmf.nmf.BaseComponent.W>` should be presented in this module.


        Note:
            Although the value range of ``beta`` is unrestricted, the original implementation only use Euclidean Distance 
            (which means ``beta=2``) as their loss function, and we have no gaurantee on other values besides 2.

        .. _`Non-negative Matrix Factorization with Sparseness Constraints`:
            https://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf

        Args:
            V (Tensor): data tensor to be decomposed. Can be a sparse tensor returned by :func:`torch.sparse_coo_tensor` 
            beta (float): beta divergence to be minimized, measuring the distance between V and the NMF model
                        Default: ``1.``
            max_iter (int): maximum number of iterations before timing out. Default: ``200``
            verbose (bool): whether to be verbose. Default: ``False``
            sW (float or None): the target sparseness for template tensor :attr:`W` , with 0 < sW < 1. Set it to ``None``
                will have no constraint. Default: ``None``
            sH (float or None): the target sparseness for activation tensor :attr:`H` , with 0 < sH < 1. Set it to ``None``
                will have no constraint. Default: ``None``

        Returns:
            int: total number of iterations
        """
        assert torch.all((V._values() if V.is_sparse else V) >=
                         0.), "Target should be non-negative."

        if (V.is_sparse or V.min() == 0) and beta <= 0:
            raise ValueError("When beta <= 0 and V contains zeros, "
                             "the training process may diverge. "
                             "Please add small values to "
                             "V, or use a positive beta value.")
        W = self.W
        H = self.H

        if sW is not None and W.requires_grad:
            dim = W[:, 0].numel()
            L1a = dim ** 0.5 * (1 - sW) + sW
            with torch.no_grad():
                for i in range(W.shape[1]):
                    W[:, i] = _proj_func(W[:, i], L1a, 1)
        else:
            L1a = None

        if sH is not None and H.requires_grad:
            dim = H[:, 0].numel()
            L1s = dim ** 0.5 * (1 - sH) + sH
            with torch.no_grad():
                for j in range(H.shape[1]):
                    H[:, j] = _proj_func(H[:, j], L1s, 1)
        else:
            L1s = None

        if beta < 1:
            gamma = 1 / (2 - beta)
        elif beta > 2:
            gamma = 1 / (beta - 1)
        else:
            gamma = 1

        if V.is_sparse:
            V = V.coalesce()
            V_norm = _get_V_norm(V, beta)

        stepsize_W, stepsize_H = 1, 1
        with tqdm(total=max_iter, disable=not verbose) as pbar:
            for n_iter in range(max_iter):
                if W.requires_grad:
                    if V.is_sparse:
                        pos, neg = self._sp_recon_beta_pos_neg(
                            V, H.detach(), W, beta)
                        WH = None
                    else:
                        WH = self.reconstruct(H.detach(), W)
                        pos, neg = None, None
                    if sW is None:
                        precomputed_pos = _get_W_kl_positive(
                            H.detach()) if beta == 1 else None
                        if V.is_sparse:
                            _sp_double_backward_update(
                                pos, neg, W, gamma, 0, 0, precomputed_pos)
                        else:
                            _double_backward_update(
                                V, WH, W, beta, gamma, 0, 0, precomputed_pos)
                    else:
                        W.grad = None
                        if V.is_sparse:
                            loss = V_norm + pos - neg
                        else:
                            loss = beta_div(WH, V, beta)
                        loss.backward()
                        with torch.no_grad():
                            for i in range(10):
                                Wnew = W - stepsize_W * W.grad
                                norms = _get_norm(Wnew)
                                for j in range(Wnew.shape[1]):
                                    Wnew[:, j] = _proj_func(
                                        Wnew[:, j], L1a * norms[j], norms[j] ** 2)
                                if V.is_sparse:
                                    new_pos, new_neg = self._sp_recon_beta_pos_neg(
                                        V, H, Wnew, beta)
                                    new_loss = V_norm + new_pos - new_neg
                                else:
                                    new_loss = beta_div(self.reconstruct(H, Wnew),
                                                        V, beta)
                                if new_loss <= loss:
                                    break

                                stepsize_W *= 0.5

                            stepsize_W *= 1.2
                            W.copy_(Wnew)

                if H.requires_grad:
                    if V.is_sparse:
                        pos, neg = self._sp_recon_beta_pos_neg(
                            V, H, W.detach(), beta)
                        WH = None
                    else:
                        WH = self.reconstruct(H, W.detach())
                        pos, neg = None, None
                    if sH is None:
                        precomputed_pos = _get_H_kl_positive(
                            W.detach()) if beta == 1 else None
                        if V.is_sparse:
                            _sp_double_backward_update(
                                pos, neg, H, gamma, 0, 0, precomputed_pos)
                        else:
                            _double_backward_update(
                                V, WH, H, beta, gamma, 0, 0, precomputed_pos)

                    else:
                        H.grad = None
                        if V.is_sparse:
                            loss = V_norm + pos - neg
                        else:
                            loss = beta_div(WH, V, beta)
                        loss.backward()

                        with torch.no_grad():
                            for i in range(10):
                                Hnew = H - stepsize_H * H.grad
                                norms = _get_norm(Hnew)
                                for j in range(H.shape[1]):
                                    Hnew[:, j] = _proj_func(
                                        Hnew[:, j], L1s * norms[j], norms[j] ** 2)

                                if V.is_sparse:
                                    new_pos, new_neg = self._sp_recon_beta_pos_neg(
                                        V, Hnew, W, beta)
                                    new_loss = V_norm + new_pos - new_neg
                                else:
                                    new_loss = beta_div(self.reconstruct(Hnew, W),
                                                        V, beta)
                                if new_loss <= loss:
                                    break

                                stepsize_H *= 0.5

                            stepsize_H *= 1.2
                            H.copy_(Hnew)
                    _renorm(W, H, 'H')

                if n_iter % 10 == 9:
                    with torch.no_grad():
                        if V.is_sparse:
                            pos, neg = self._sp_recon_beta_pos_neg(
                                V, H, W, beta)
                            loss = V_norm + pos - neg
                        else:
                            loss = beta_div(self.reconstruct(H, W),
                                            V, beta)
                        loss = loss.mul(2).sqrt().item()
                    pbar.set_postfix(loss=loss)
                    pbar.update(10)
        return n_iter + 1


@torch.jit.script
def _nmf_sparse_reconstruct(H: Tensor, W: Tensor, indices: Tensor):
    ii, jj = indices[0], indices[1]
    n_vals = indices.shape[1]
    rank = W.shape[1]
    batch_size = max(rank, n_vals // rank)

    dot_vals = torch.empty(n_vals, dtype=H.dtype, device=H.device)
    for start in range(0, n_vals, batch_size):
        batch = slice(start, start + batch_size)
        dot_vals[batch] = (W[jj[batch], :] * H[ii[batch], :]).sum(1)

    return dot_vals


@torch.jit.script
def _nmf_sp_recon_beta_pos_neg(V: Tensor, H: Tensor, W: Tensor, beta: float, eps: float):
    V_idx = V.indices()
    V_vals = V.values()
    if beta == 2:
        pos = torch.chain_matmul(H, W.T, W).view(-1) @ H.view(-1) * 0.5
        neg = (V.t() @ H).view(-1) @ W.view(-1)
        return pos, neg

    WH_vals = _nmf_sparse_reconstruct(H, W, V_idx)

    if beta == 1:
        pos = W.sum(0) @ H.sum(0)
        neg = V_vals @ WH_vals.add(eps).log()
    else:
        bminus = beta - 1
        pos = (W @ H[0] + eps).pow(beta).sum()
        for i in range(1, H.shape[0]):
            pos += (W @ H[i] + eps).pow(beta).sum()
        pos /= beta
        neg = V_vals @ WH_vals.add(eps).pow(bminus) / bminus
    return pos, neg


class NMF(BaseComponent):
    r"""Non-Negative Matrix Factorization (NMF).

    Find two non-negative matrices (W, H) whose product approximates the non-
    negative matrix V: :math:`V \approx HW^T`.

    This factorization can be used for example for dimensionality reduction, source separation or topic extraction.

    Note:
        If `Vshape` argument is given, the model will try to infer the size of :meth:`W <torchnmf.nmf.BaseComponent.W>` and
        :meth:`H <torchnmf.nmf.BaseComponent.H>`, and override arguments passed through to :meth:`BaseComponent <torchnmf.nmf.BaseComponent>`.

    Args:
        Vshape (tuple, optional): size of target matrix V
        rank (int, optional): size of hidden dimension
        **kwargs: arguments passed through to :meth:`BaseComponent <torchnmf.nmf.BaseComponent>`


    Shape:
        - V: :math:`(N, C)`
        - W: :math:`(C, R)`
        - H: :math:`(N, R)`


    """

    def __init__(self,
                 Vshape: Iterable[int] = None,
                 rank: int = None,
                 **kwargs):

        super().__init__(rank, **kwargs)

    @staticmethod
    def reconstruct(H, W):
        return F.linear(H, W)

    def _sp_recon_beta_pos_neg(self, V, H, W, beta):
        assert V.is_sparse
        return _nmf_sp_recon_beta_pos_neg(V, H, W, beta, eps)

'''
import neuralNMF
import scipy
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np 
from neuralNMF.model.torchAG_nmf import NMF
import torch

H,W,X = neuralNMF.generate_data(N=100,K=10,M=200,mode='block')

X = X.astype(float)
X = torch.from_numpy(X)

model = NMF(X.shape, rank=10).cuda()
print(model)
model.fit(X.cuda(),verbose=True)

H_hat = model.H.detach().cpu().numpy()
W_hat = model.W.detach().cpu().numpy().T
x_hat = H_hat @ W_hat

correlation_matrix = np.corrcoef(X, x_hat)
correlation_rows = correlation_matrix[:X.shape[0], x_hat.shape[0]:]
sns.heatmap(correlation_rows)
plt.savefig('torchADnmf_corr.png');plt.close()

'''