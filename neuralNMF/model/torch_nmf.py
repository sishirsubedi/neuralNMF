
'''
source - 
A PyTorch implementation on Non-negative Matrix Factorization.

https://github.com/lilab-bcb/nmf-torch/blob/main/nmf/nmf.py

'''
import numpy as np
import torch
from typing import List, Union, Tuple, Optional

from .nmf_models import NMFBatchMU, NMFBatchHALS

def run_nmf(
    X: Union[np.array, torch.tensor],
    n_components: int,
    init: str = "nndsvdar",
    beta_loss: Union[str, float] = "frobenius",
    algo: str = "halsvar",
    mode: str = "batch",
    tol: float = 1e-4,
    n_jobs: int = -1,
    random_state: int = 0,
    use_gpu: bool = False,
    alpha_W: float = 0.0,
    l1_ratio_W: float = 0.0,
    alpha_H: float = 0.0,
    l1_ratio_H: float = 0.0,
    fp_precision: Union[str, torch.dtype] = "float",
    batch_max_iter: int = 500,
    batch_hals_tol: float = 0.05,
    batch_hals_max_iter: int = 200
) -> Tuple[np.array, np.array, float]:
    """
    Perform Non-negative Matrix Factorization (NMF).

    Decompose a non-negative matrix X into an approximation of the product of two matrices H and W of smaller ranks.
    It is useful for dimension reduction, topic modeling, gene program extraction in Genomics, etc.

    The objective function is

        .. math::

            ||X - HW||_{beta} + alpha_H * l1_{ratio, H} * ||vec(H)||_1

            + 0.5 * alpha_H * (1 - l1_{ratio, H}) * ||H||_{Fro}^2

            + alpha_W * l1_{ratio, W} * ||vec(W)||_1

            + 0.5 * alpha_W * (1 - l1_{ratio, W}) * ||W||_{Fro}^2

    where

    :math:`||A||_{beta} = \\frac{1}{beta * (beta - 1)} \\sum_{i, j} X_{ij}^{beta} - beta * X_{ij} * Y_{ij}^{beta - 1} + (beta - 1) * Y_{ij}^{beta}` (Beta divergence)

    :math:`||vec(A)||_1 = \\sum_{i, j} abs(A_{ij})` (Element-wise L1 norm)

    :math:`||A||_{Fro}^2 = \\sum_{i, j} A_{ij}^2` (Frobenius norm)

    NMF uses various solvers (specified in ``algo`` parameter), in either batch or online mode (specified in ``mode`` parameter), to minimize this objective function.

    Parameters
    ----------

    X: ``numpy.array`` or ``torch.tensor``
        The input non-negative matrix of shape (n_samples, n_features).
    n_components: ``int``
        Number of components.
    init: ``str``, optional, default: ``nndsvdar``
        Method for initialization on H and W matrices. Available options are: ``random``, ``nndsvd``, ``nndsvda``, ``nndsvdar``.
    beta_loss: ``str`` or ``float``
        Beta loss between the given matrix X and its approximation calculated by HW, which is used as the metric to be minimized during the computation.
        It can be a string from options:
            - ``frobenius``: L2 distance, same as ``beta_loss=2.0``.
            - ``kullback-leibler``:KL divergence, same as ``beta_loss=1.0``.
            - ``itakura-saito``: Itakura-Saito divergence, same as ``beta_loss=0``.
        Alternatively, it can also be a float number, which gives the beta parameter of the beta loss to be used.
    algo: ``str``, optional, default: ``halsvar``
        Choose from ``mu`` (Multiplicative Update), ``hals`` (Hierarchical Alternative Least Square), ``halsvar`` (HALS variant) and ``bpp`` (alternative non-negative least squares with Block Principal Pivoting method).
        ``hals`` refers to the standard HALS algorithm and sets batch_hals_max_iter = 1. ``halsvar`` is the HALS variant that tries to mimic ``bpp`` and uses batch_hals_max_iter to tune the HALS iterations over H/W.
        If mode is online, there is no difference between ``hals`` and ``halsvar``.
    mode: ``str``, optional, default: ``batch``
        Learning mode. Choose from ``batch`` and ``online``. Notice that ``online`` only works when ``beta=2.0``. For other beta loss, it switches back to ``batch`` method.
    tol: ``float``, optional, default: ``1e-4``
        The toleration used for convergence check.
    n_jobs: ``int``, optional, default: ``-1``
        Number of cpu threads to use. If -1, use PyTorch's default setting.
    random_state: ``int``, optional, default: ``0``
        The random state used for reproducibility on the results.
    use_gpu: ``bool``, optional, default: ``False``
        If ``True``, use GPU if available. Otherwise, use CPU only.
    alpha_W: ``float``, optional, default: ``0.0``
        A numeric scale factor which multiplies the regularization terms related to W.
        If zero or negative, no regularization regarding W is considered.
    l1_ratio_W: ``float``, optional, default: ``0.0``
        The ratio of L1 penalty on W, must be between 0 and 1. And thus the ratio of L2 penalty on W is (1 - l1_ratio_W).
    alpha_H: ``float``, optional, default: ``0.0``
        A numeric scale factor which multiplies the regularization terms related to H.
        If zero or negative, no regularization regarding H is considered.
    l1_ratio_H: ``float``, optional, default: ``0.0``
        The ratio of L1 penalty on W, must be between 0 and 1. And thus the ratio of L2 penalty on H is (1 - l1_ratio_H).
    fp_precision: ``str``, optional, default: ``float``
        The numeric precision on the results.
        If ``float``, set precision to ``torch.float``; if ``double``, set precision to ``torch.double``.
        Alternatively, choose Pytorch's `torch dtype <https://pytorch.org/docs/stable/tensor_attributes.html>`_ of your own.
    batch_max_iter: ``int``, optional, default: ``500``
        The maximum number of iterations to perform for batch learning.
    batch_hals_tol: ``float``, optional, default: ``0.05``
        For HALS, we have the option of using HALS to mimic BPP for a possible better loss. The mimic works as follows: update H by HALS several iterations until the maximal relative change < batch_hals_tol. Then update W similarly.
    batch_hals_max_iter: ``int``, optional, default: ``200``
        Maximal iterations of updating H & W for mimic BPP. If this parameter set to 1, it is the standard HALS.
    online_max_pass: ``int``, optional, default: ``20``
        The maximum number of online passes of all data to perform.
    online_chunk_size: ``int``, optional, default: ``5000``
        The chunk / mini-batch size for online learning.
    online_chunk_max_iter: ``int``, optional, default: ``200``
        The maximum number of iterations for updating H or W in online learning.
    online_h_tol: ``float``, optional, default: 0.05
        The tolerance for updating H in each chunk in online learning.
    online_w_tol: ``float``, optional, default: 0.05
        The tolerance for updating W in each chunk in online learning.

    Returns
    -------
    H: ``numpy.array``
        One of the resulting decomposed matrix of shape (n_samples, n_components). It represents the transformed coordinates of samples regarding components.
    W: ``numpy.array``
        The other resulting decomposed matrix of shape (n_components, n_features). It represents the composition of each component in terms of features.
    reconstruction_error: ``float``
        The Beta Loss between the origin matrix X and its approximation HW after NMF.


    """
    if beta_loss == 'frobenius':
        beta_loss = 2
    elif beta_loss == 'kullback-leibler':
        beta_loss = 1
    elif beta_loss == 'itakura-saito':
        beta_loss = 0
    elif not (isinstance(beta_loss, int) or isinstance(beta_loss, float)):
        raise ValueError("beta_loss must be a valid value: either from ['frobenius', 'kullback-leibler', 'itakura-saito'], or a numeric value.")

    device_type = 'cpu'
    if use_gpu:
        if torch.cuda.is_available():
            device_type = 'cuda'
            print("Use GPU mode.")
        else:
            print("CUDA is not available on your machine. Use CPU mode instead.")

    if algo not in {'mu', 'hals', 'halsvar', 'bpp'}:
        raise ValueError("Parameter algo must be a valid value from ['mu', 'hals', 'halsvar', 'bpp']!")
    if mode not in {'batch', 'online'}:
        raise ValueError("Parameter mode must be a valid value from ['batch', 'online']!")
    if beta_loss != 2 and mode == 'online':
        print("Cannot perform online update when beta not equal to 2. Switch to batch update method.")
        mode = 'batch'

    if algo == 'hals':
        batch_hals_max_iter = 1

    model_class = None
    kwargs = {'alpha_W': alpha_W, 'l1_ratio_W': l1_ratio_W, 'alpha_H': alpha_H, 'l1_ratio_H': l1_ratio_H, 'fp_precision': fp_precision, 'device_type': device_type}

    if mode == 'batch':
        kwargs['max_iter'] = batch_max_iter
        if algo == 'mu':
            model_class = NMFBatchMU
        elif algo == 'hals' or algo == 'halsvar':
            model_class = NMFBatchHALS
            kwargs['hals_tol'] = batch_hals_tol
            kwargs['hals_max_iter'] = batch_hals_max_iter
    else:
        print('online version not added..')
        
    model = model_class(
                n_components=n_components,
                init=init,
                beta_loss=beta_loss,
                tol=tol,
                n_jobs=n_jobs,
                random_state=random_state,
                **kwargs
            )

    H = model.fit_transform(X)
    W = model.W
    err = model.reconstruction_err

    return H.cpu().numpy(), W.cpu().numpy(), err.item()

'''
import neuralNMF
import scipy
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np 
from neuralNMF.model.torch_nmf import run_nmf


H,W,X = neuralNMF.generate_data(N=100,K=10,M=200,mode='block')



H_hat, W_hat, err = run_nmf(X, n_components=10, init='random', algo='mu', mode='batch')

x_hat = H_hat @ W_hat

correlation_matrix = np.corrcoef(X, x_hat)
correlation_rows = correlation_matrix[:X.shape[0], x_hat.shape[0]:]
sns.heatmap(correlation_rows)
plt.savefig('torchnmf_corr.png');plt.close()

'''