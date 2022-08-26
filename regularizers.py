import sys
current_file = sys.modules[__name__]

import torch as tc


def diffusion_relative_tc(displacement_field: tc.Tensor, device: str="cpu", **params):
    """
    Relative diffusion regularization (with respect to the input size) (PyTorch).
    Parameters
    ----------
    displacement_field : tc.Tensor
        The input displacment field (2-D or 3-D) (B x size x ndim)
    params : dict
        Additional parameters
    Returns
    ----------
    diffusion_reg : float
        The value denoting the decrease of displacement field smoothness
    """
    ndim = len(displacement_field.size()) - 2
    if ndim == 2:
        dx = ((displacement_field[:, 1:, :, :] - displacement_field[:, :-1, :, :])*displacement_field.shape[1])**2
        dy = ((displacement_field[:, :, 1:, :] - displacement_field[:, :, :-1, :])*displacement_field.shape[2])**2
        diffusion_reg = (tc.mean(dx) + tc.mean(dy)) / 2
    elif ndim == 3:
        dx = ((displacement_field[:, 1:, :, :, :] - displacement_field[:, :-1, :, :, :])*displacement_field.shape[1])**2
        dy = ((displacement_field[:, :, 1:, :, :] - displacement_field[:, :, :-1, :, :])*displacement_field.shape[2])**2
        dz = ((displacement_field[:, :, :, 1:, :] - displacement_field[:, :, :, :-1, :])*displacement_field.shape[3])**2
        diffusion_reg = (tc.mean(dx) + tc.mean(dy) + tc.mean(dz)) / 3
    else:
        raise ValueError("Unsupported number of dimensions.")
    return diffusion_reg



def get_function(function_name):
    return getattr(current_file, function_name)