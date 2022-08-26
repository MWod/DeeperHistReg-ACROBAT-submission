import sys
current_file = sys.modules[__name__]

import math
import numpy as np
import torch as tc
import torch.nn.functional as F

def ncc_local_tc(sources: tc.Tensor, targets: tc.Tensor, device: str=None, **params):
    """
    Local normalized cross-correlation (as cost function) using PyTorch tensors.

    Implementation inspired by VoxelMorph (with some modifications).

    Parameters
    ----------
    sources : tc.Tensor(Bx1xMxN)
        The source tensor
    targest : tc.Tensor (Bx1xMxN)
        The target target
    device : str
        The device where source/target are placed
    params : dict
        Additional cost function parameters

    Returns
    ----------
    ncc : float
        The negative of normalized cross-correlation (average across batches)
    
    """
    ndim = len(sources.size()) - 2
    if ndim not in [2, 3]:
        raise ValueError("Unsupported number of dimensions.")
    try:
        win_size = params['win_size']
    except:
        win_size = 3
    try:
        mask = params['mask']
    except:
        mask = None
    window = (win_size, ) * ndim
    if device is None:
        sum_filt = tc.ones([1, 1, *window]).type_as(sources)
    else:
        sum_filt = tc.ones([1, 1, *window], device=device)

    if mask is not None:
        targets = targets * mask

    pad_no = math.floor(window[0] / 2)
    stride = ndim * (1,)
    padding = ndim * (pad_no,)
    conv_fn = getattr(F, 'conv%dd' % ndim)
    sources_denom = sources**2
    targets_denom = targets**2
    numerator = sources*targets
    sources_sum = conv_fn(sources, sum_filt, stride=stride, padding=padding)
    targets_sum = conv_fn(targets, sum_filt, stride=stride, padding=padding)
    sources_denom_sum = conv_fn(sources_denom, sum_filt, stride=stride, padding=padding)
    targets_denom_sum = conv_fn(targets_denom, sum_filt, stride=stride, padding=padding)
    numerator_sum = conv_fn(numerator, sum_filt, stride=stride, padding=padding)
    size = np.prod(window)
    u_sources = sources_sum / size
    u_targets = targets_sum / size
    cross = numerator_sum - u_targets * sources_sum - u_sources * targets_sum + u_sources * u_targets * size
    sources_var = sources_denom_sum - 2 * u_sources * sources_sum + u_sources * u_sources * size
    targets_var = targets_denom_sum - 2 * u_targets * targets_sum + u_targets * u_targets * size
    ncc = cross * cross / (sources_var * targets_var + 1e-5)
    return -tc.mean(ncc)

def sparse_ncc_tc(sources: tc.Tensor, targets: tc.Tensor, device: str=None, **params):
    """
    TODO - documentation
    """
    keypoints = params['keypoints']
    win_size = params['win_size']
    scores = tc.zeros(len(keypoints), device=sources.device)
    _, _, y_size, x_size = sources.shape
    for i in range(len(keypoints)):
        keypoint = int(keypoints[i].pt[0]), int(keypoints[i].pt[1])
        b_y, e_y = max(min(keypoint[1] - int(win_size // 2), y_size), 0), max(min(keypoint[1] + int(win_size // 2) + 1, y_size), 0)
        b_x, e_x = max(min(keypoint[0] - int(win_size // 2), x_size), 0), max(min(keypoint[0] + int(win_size // 2) + 1, x_size), 0)
        cs = sources[:, :, b_y:e_y, b_x:e_x]
        ts = targets[:, :, b_y:e_y, b_x:e_x]
        scores[i] = ncc_global_tc(cs, ts)
    scores = scores[scores != 1]
    return tc.mean(scores)

def ncc_global_tc(sources: tc.Tensor, targets: tc.Tensor, device: str="cpu", **params):
    """
    Global normalized cross-correlation (as cost function) using PyTorch tensors.

    Parameters
    ----------
    sources : tc.Tensor(Bx1xMxN)
        The source tensor
    targest : tc.Tensor (Bx1xMxN)
        The target target
    device : str
        The device where source/target are placed
    params : dict
        Additional cost function parameters

    Returns
    ----------
    ncc : float
        The negative of normalized cross-correlation (average across batches)
    """
    sources = (sources - tc.min(sources)) / (tc.max(sources) - tc.min(sources))
    targets = (targets - tc.min(targets)) / (tc.max(targets) - tc.min(targets))
    if sources.size() != targets.size():
        raise ValueError("Shape of both the tensors must be the same.")
    size = sources.size()
    prod_size = tc.prod(tc.Tensor(list(size[1:])))
    sources_mean = tc.mean(sources, dim=list(range(1, len(size)))).view((sources.size(0),) + (len(size)-1)*(1,))
    targets_mean = tc.mean(targets, dim=list(range(1, len(size)))).view((targets.size(0),) + (len(size)-1)*(1,))
    sources_std = tc.std(sources, dim=list(range(1, len(size))), unbiased=False).view((sources.size(0),) + (len(size)-1)*(1,))
    targets_std = tc.std(targets, dim=list(range(1, len(size))), unbiased=False).view((targets.size(0),) + (len(size)-1)*(1,))
    ncc = (1 / prod_size) * tc.sum((sources - sources_mean) * (targets - targets_mean) / (sources_std * targets_std), dim=list(range(1, len(size))))
    ncc = tc.mean(ncc)
    if ncc != ncc:
        ncc = tc.autograd.Variable(tc.Tensor([-1]), requires_grad=True).to(device)
    return -ncc




def get_function(function_name):
    return getattr(current_file, function_name)