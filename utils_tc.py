import math
import numpy as np
import torch as tc
import torch.nn.functional as F
import torchvision.transforms as tr


def normalize(tensor : tc.Tensor):
    if len(tensor.size()) - 2 == 2:
        num_channels = tensor.size(1)
        normalized_tensor = tc.zeros_like(tensor)
        for i in range(num_channels):
            mins, _ = tc.min(tc.min(tensor[:, i, :, :] , dim=1, keepdim=True)[0], dim=2, keepdim=True) # TODO - find better approach
            maxs, _ = tc.max(tc.max(tensor[:, i, :, :] , dim=1, keepdim=True)[0], dim=2, keepdim=True)
            normalized_tensor[:, i, :, :] = (tensor[:, i, :, :] - mins) / (maxs - mins)
        return normalized_tensor
    else:
        raise ValueError("Unsupported number of channels.")

def resample(tensor, resample_ratio, cval=0, order=3):
    resample_dict = {0: "nearest", 1: "bilinear", 3: "bicubic"}
    try:
        mode = resample_dict[order]
    except KeyError:
        mode = "bicubic"
    return F.interpolate(tensor, scale_factor = 1 / resample_ratio, mode=mode, recompute_scale_factor=False, align_corners=False)

def resample_displacement_field(displacement_field, resample_ratio, cval=0, order=3):
    resample_dict = {0: "nearest", 1: "bilinear", 3: "bicubic"}
    try:
        mode = resample_dict[order]
    except KeyError:
        mode = "bicubic"
    displacement_field = displacement_field.permute(0, 3, 1, 2)
    resampled_displacement_field = F.interpolate(displacement_field, scale_factor = 1 / resample_ratio, mode=mode, recompute_scale_factor=False, align_corners=False)
    return resampled_displacement_field.permute(0, 2, 3, 1)

def resample_tensor_to_size(tensor: tc.Tensor, new_size: tc.Tensor, mode: str='bilinear'):
    return F.interpolate(tensor, size=new_size, mode=mode, align_corners=False)

def resample_displacement_field_to_size(displacement_field: tc.Tensor, new_size: tc.Tensor, mode: str='bilinear'):
    return F.interpolate(displacement_field.permute(0, 3, 1, 2), size=new_size, mode=mode, align_corners=False).permute(0, 2, 3, 1)

def gaussian_smoothing(tensor, sigma):
    with tc.set_grad_enabled(False):
        kernel_size = int(sigma * 2.54) + 1 if int(sigma * 2.54) % 2 == 0 else int(sigma * 2.54)
        return tr.GaussianBlur(kernel_size, sigma)(tensor)

def gaussian_smoothing_patch(tensor, sigma, patch_size=(2048, 2048), offset=(50, 50)):
    smoothed_tensor = tc.zeros_like(tensor)
    with tc.set_grad_enabled(False):
        y_size, x_size = tensor.size(2), tensor.size(3)
        rows, cols = int(np.ceil(y_size / patch_size[0])), int(np.ceil(x_size / patch_size[1]))
        for row in range(rows):
            for col in range(cols):
                b_x = max(0, min(x_size, col*patch_size[1]))
                b_y = max(0, min(y_size, row*patch_size[0]))
                e_x = max(0, min(x_size, (col+1)*patch_size[1]))
                e_y = max(0, min(y_size, (row+1)*patch_size[0]))
                ob_x = max(0, min(x_size, b_x - offset[1]))
                oe_x = max(0, min(x_size, e_x + offset[1]))
                ob_y = max(0, min(y_size, b_y - offset[0]))
                oe_y =  max(0, min(y_size, e_y + offset[0]))
                diff_bx = b_x - ob_x
                diff_by = b_y - ob_y
                smoothed_tensor[:, :, b_y:e_y, b_x:e_x] = gaussian_smoothing(tensor[:, :, ob_y:oe_y, ob_x:oe_x], sigma)[:, :, diff_by:diff_by+patch_size[0], diff_bx:diff_bx+patch_size[1]]
    return smoothed_tensor

def compose_displacement_fields(displacement_field_1, displacement_field_2):
    sampling_grid = generate_grid(tensor_size=(displacement_field_1.size(0), 1, displacement_field_1.size(1), displacement_field_1.size(2)), device=displacement_field_1.device)
    composed_displacement_field = F.grid_sample((sampling_grid + displacement_field_1).permute(0, 3, 1, 2), sampling_grid + displacement_field_2, padding_mode='border', align_corners=False).permute(0, 2, 3, 1)
    composed_displacement_field = composed_displacement_field - sampling_grid
    return composed_displacement_field

def get_combined_size(tensor_1 : tc.Tensor, tensor_2 : tc.Tensor):
    tensor_1_y_size, tensor_1_x_size = tensor_1.size(2), tensor_1.size(3)
    tensor_2_y_size, tensor_2_x_size = tensor_2.size(2), tensor_2.size(3)
    return tensor_1_y_size, tensor_1_x_size, tensor_2_y_size, tensor_2_x_size

def generate_grid(tensor : tc.Tensor=None, tensor_size: tc.Tensor=None, device: str=None):
    """
    Generates the identity grid for a given tensor size.

    Parameters
    ----------
    tensor : tc.Tensor
        The tensor to be used as template
    tensor_size : tc.Tensor or tc.Size
        The tensor size used to generate the regular grid
    device : str
        The device to generate the grid on
    Returns
    ----------
    grid : tc.Tensor
        The regular grid (relative for warp_tensor with align_corners=False)
    """
    if tensor is not None:
        tensor_size = tensor.size()
    if device is None:
        identity_transform = tc.eye(len(tensor_size)-1)[:-1, :].unsqueeze(0).type_as(tensor)
    else:
        identity_transform = tc.eye(len(tensor_size)-1, device=device)[:-1, :].unsqueeze(0)
    identity_transform = tc.repeat_interleave(identity_transform, tensor_size[0], dim=0)
    grid = F.affine_grid(identity_transform, tensor_size, align_corners=False)
    return grid

def create_identity_displacement_field(tensor : tc.Tensor):
    return tc.zeros((tensor.size(0), tensor.size(2), tensor.size(3)) + (2,)).type_as(tensor)

def warp_tensor(tensor: tc.Tensor, displacement_field: tc.Tensor, grid: tc.Tensor=None, mode: str='bilinear', padding_mode: str='zeros', device: str=None):
    """
    Transforms a tensor with a given displacement field.
    Uses F.grid_sample for the structured data interpolation (only linear and nearest supported).
    Be careful - the autogradient calculation is possible only with mode set to "bilinear".

    Parameters
    ----------
    tensor : tc.Tensor
        The tensor to be transformed (BxYxXxZxD)
    displacement_field : tc.Tensor
        The PyTorch displacement field (BxYxXxZxD)
    grid : tc.Tensor (optional)
        The input identity grid (optional - may be provided to speed-up the calculation for iterative algorithms)
    mode : str
        The interpolation mode ("bilinear" or "nearest")
    device : str
        The device to generate the warping grid if not provided
    Returns
    ----------
    transformed_tensor : tc.Tensor
        The transformed tensor (BxYxXxZxD)
    """
    if grid is None:
        grid = generate_grid(tensor=tensor, device=device)
    sampling_grid = grid + displacement_field
    transformed_tensor = F.grid_sample(tensor, sampling_grid, mode=mode, padding_mode=padding_mode, align_corners=False)
    return transformed_tensor

def transform_tensor(tensor: tc.Tensor, sampling_grid: tc.Tensor, grid: tc.Tensor=None, device: str="cpu", mode: str='bilinear'):
    """
    Transforms a tensor with a given sampling grid.
    Uses F.grid_sample for the structured data interpolation (only linear and nearest supported).
    Be careful - the autogradient calculation is possible only with mode set to "bilinear".

    Parameters
    ----------
    tensor : tc.Tensor
        The tensor to be transformed (BxYxXxZxD)
    sampling_grid : tc.Tensor
        The PyTorch sampling grid
    grid : tc.Tensor (optional)
        The input identity grid (optional - may be provided to speed-up the calculation for iterative algorithms)
    device : str
        The device used for warping (e.g. "cpu" or "cuda:0")
    mode : str
        The interpolation mode ("bilinear" or "nearest")

    Returns
    ----------
    transformed_tensor : tc.Tensor
        The transformed tensor (BxYxXxZxD)
    """
    transformed_tensor = F.grid_sample(tensor, sampling_grid, mode=mode, padding_mode='zeros', align_corners=False)
    return transformed_tensor

def pad_to_same_size(image_1 : tc.Tensor, image_2 : tc.Tensor, pad_value : float=1.0):
    y_size_1, x_size_1 = image_1.size(2), image_1.size(3)
    y_size_2, x_size_2 = image_2.size(2), image_2.size(3)
    pad_1 = [(0, 0), (0, 0)]
    pad_2 = [(0, 0), (0, 0)]
    if y_size_1 > y_size_2:
        pad_size = y_size_1 - y_size_2
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_2[0] = pad
    elif y_size_1 < y_size_2:
        pad_size = y_size_2 - y_size_1
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_1[0] = pad
    else:
        pass
    if x_size_1 > x_size_2:
        pad_size = x_size_1 - x_size_2
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_2[1] = pad
    elif x_size_1 < x_size_2:
        pad_size = x_size_2 - x_size_1
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_1[1] = pad
    else:
        pass
    image_1 = F.pad(image_1, pad_1[1] + pad_1[0], mode='constant', value=pad_value)
    image_2 = F.pad(image_2, pad_2[1] + pad_2[0], mode='constant', value=pad_value)
    padding_params = dict()
    padding_params['pad_1'] = pad_1
    padding_params['pad_2'] = pad_2
    return image_1, image_2, padding_params

def calculate_diagonal(tensor : tc.Tensor):
    return math.sqrt(tensor.size(2)**2 + tensor.size(3)**2)

def convert_to_gray(image : tc.Tensor):
    return tr.Grayscale()(image)

def unpad_displacement_field(displacement_field : tc.Tensor, padding_params : dict):
    # TODO - documentation
    pad = padding_params['pad_1']
    y_pad, x_pad = pad
    if y_pad[1] == 0:
        displacement_field = displacement_field[:, y_pad[0]:, :, :]
    else:
        displacement_field = displacement_field[:, y_pad[0]:-y_pad[1], :, :]
    if x_pad[1] == 0:
        displacement_field = displacement_field[:, :, x_pad[0]:, :]
    else:
        displacement_field = displacement_field[:, :, x_pad[0]:-x_pad[1], :]
    return displacement_field

def center_of_mass(tensor):
    y_size, x_size = tensor.size(2), tensor.size(3)
    gy, gx = tc.meshgrid(tc.arange(y_size).type_as(tensor), tc.arange(x_size).type_as(tensor), indexing='ij')
    m00 = tc.sum(tensor).item()
    m10 = tc.sum(gx*tensor).item()
    m01 = tc.sum(gy*tensor).item()
    com_x = m10 / m00
    com_y = m01 / m00
    return com_x, com_y

def tc_transform_to_tc_df(transformation: tc.Tensor, size: tc.Size):
    """
    Transforms the transformation tensor into the displacement field tensor.

    Parameters
    ----------
    transformation : tc.Tensor
        The transformation tensor (B x transformation size (2x3 or 3x4))
    size : tc.Tensor (or list, or tuple)
        The desired displacement field size
    Returns
    ----------
    resampled_displacement_field: tc.Tensor
        The resampled displacement field (BxYxXxZxD)
    """
    deformation_field = F.affine_grid(transformation, size=size, align_corners=False)
    size = (deformation_field.size(0), 1) + deformation_field.size()[1:-1]
    grid = generate_grid(tensor_size=size, device=transformation.device)
    displacement_field = deformation_field - grid
    return displacement_field

def affine2theta(affine, shape):
    h, w = shape[0], shape[1]
    temp = affine
    theta = tc.zeros([2, 3])
    theta[0, 0] = temp[0, 0]
    theta[0, 1] = temp[0, 1]*h/w
    theta[0, 2] = temp[0, 2]*2/w + theta[0, 0] + theta[0, 1] - 1
    theta[1, 0] = temp[1, 0]*w/h
    theta[1, 1] = temp[1, 1]
    theta[1, 2] = temp[1, 2]*2/h + theta[1, 0] + theta[1, 1] - 1
    return theta

def theta2affine(theta, shape):
    h, w = shape[0], shape[1]
    temp = theta
    affine = np.zeros((2, 3))
    affine[1, 2] = (temp[1, 2] - temp[1, 0] - temp[1, 1] + 1)*h/2
    affine[1, 1] = temp[1, 1]
    affine[1, 0] = temp[1, 0]*h/w
    affine[0, 2] = (temp[0, 2] - temp[0, 0] - temp[0, 1] + 1)*w/2
    affine[0, 1] = temp[0, 1]*w/h
    affine[0, 0] = temp[0, 0]
    return affine

def compose_transforms(t1, t2):
    tr1 = tc.zeros((3, 3)).type_as(t1)
    tr2 = tc.zeros((3, 3)).type_as(t2)
    tr1[0:2, :] = t1
    tr2[0:2, :] = t2
    tr1[2, 2] = 1
    tr2[2, 2] = 1
    result = tc.mm(tr1, tr2)
    return result[0:2, :]

def generate_rigid_matrix(angle, x0, y0, tx, ty):
    angle = angle * np.pi/180
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    cm1 = np.array([
        [1, 0, x0],
        [0, 1, y0],
        [0, 0, 1]
    ])
    cm2 = np.array([
        [1, 0, -x0],
        [0, 1, -y0],
        [0, 0, 1]
    ])
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    transform = cm1 @ rotation_matrix @ cm2 @ translation_matrix
    return transform[0:2, :]


def create_pyramid(tensor: tc.Tensor, num_levels: int, mode: str='bilinear'):
    """
    Creates the resolution pyramid of the input tensor (assuming uniform resampling step = 2).

    Parameters
    ----------
    tensor : tc.Tensor
        The input tensor
    num_levels: int
        The number of output levels
    mode : str
        The interpolation mode ("bilinear" or "nearest")
    
    Returns
    ----------
    pyramid: list of tc.Tensor
        The created resolution pyramid

    """
    pyramid = [None]*num_levels
    for i in range(num_levels - 1, -1, -1):
        if i == num_levels - 1:
            pyramid[i] = tensor
        else:
            current_size = pyramid[i+1].size()
            new_size = (int(current_size[j] / 2) if j > 1 else current_size[j] for j in range(len(current_size)))
            new_size = tc.Size(new_size)[2:]
            new_tensor = resample_tensor_to_size(gaussian_smoothing(pyramid[i+1], 1), new_size, mode=mode)
            pyramid[i] = new_tensor
    return pyramid
