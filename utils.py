import os
import pathlib

import numpy as np
import scipy.ndimage as nd
import pandas as pd
import torch as tc
import torch.nn.functional as F
import SimpleITK as sitk
import PIL

import paths as p
import utils_np as unp
import utils_tc as utc

try:
    OPENSLIDE_PATH = p.openslide_path
    import os
    if hasattr(os, 'add_dll_directory'):
        with os.add_dll_directory(OPENSLIDE_PATH):
            import openslide
    else:
        import openslide
except:
    print("OpenSlide not available.")


def load_slide(load_path, level, load_slide=False):
    slide = openslide.OpenSlide(load_path)
    dimension = slide.level_dimensions[level]
    image = slide.read_region((0, 0), level, dimension)
    image = np.asarray(image)[:, :, 0:3].astype(np.float32)
    image = normalize(image)
    if load_slide:
        return image, slide
    else:
        return image
        
def save_image(image, save_path, renormalize=True):
    if not save_path.parents[0].exists():
        save_path.parents[0].mkdir(parents=True, exist_ok=True)
    extension = os.path.splitext(save_path)[1]
    if image.shape[2] == 3:
        if extension == ".jpg" or extension == ".jpeg":
            if renormalize:
                image = (image * 255)
            image = image.astype(np.uint8)
            image = PIL.Image.fromarray(image)
            image.save(str(save_path))
        else:
            sitk.WriteImage(sitk.GetImageFromArray(image), str(save_path))
    elif image.shape[2] == 1:
        if renormalize:
            image = (image[:, :, 0]*255)
            image = image.astype(np.uint8)
        else:
            image = (image[:, :, 0])
            image = image.astype(np.float32)
        sitk.WriteImage(sitk.GetImageFromArray(image), str(save_path))
    else:
        raise ValueError("Unsupported image format.")

def load_landmarks(load_path, mode=None, case_id=0):
    if mode is None:
        landmarks = pd.read_csv(load_path)
        landmarks = landmarks.to_numpy()[1:, 1:]
        return landmarks
    elif mode == "ANHIR":
        landmarks = pd.read_csv(load_path)
        landmarks = landmarks.to_numpy()[:, 1:]
        return landmarks

def save_landmarks(landmarks, save_path, mode=None):
    if save_path is not None:
        if not save_path.parents[0].exists():
            save_path.parents[0].mkdir(parents=True, exist_ok=True)
        if mode is None:
            column_names = ['ID', 'X', 'Y']
            landmarks = np.concatenate((np.arange(len(landmarks))[:, np.newaxis], landmarks), axis=1)
            output_df = pd.DataFrame(landmarks, columns=column_names)
            output_df.to_csv(save_path, index=False)
        elif mode == "ANHIR":
            df = pd.DataFrame(landmarks)
            df.to_csv(save_path)

def normalize(image):
    if isinstance(image, np.ndarray):
        return unp.normalize(image)
    if isinstance(image, tc.Tensor):
        return utc.normalize(image)
    else:
        raise ValueError("Unsupported type.")

def create_identity_displacement_field(image):
    if isinstance(image, np.ndarray):
        return unp.create_identity_displacement_field(image)
    elif isinstance(image, tc.Tensor):
        return utc.create_identity_displacement_field(image)
    else:
        raise TypeError("Unsupported type.")

def warp_image(image, displacement_field, cval=1.0, order=1, padding_mode='zeros'):
    if isinstance(image, np.ndarray):
        return unp.warp_image(image, displacement_field, cval=cval, order=order)
    elif isinstance(image, tc.Tensor):
        warping_dict = {0: "nearest", 1: "bilinear", 3: "bicubic"}
        return utc.warp_tensor(image, displacement_field, mode=warping_dict[order], padding_mode=padding_mode)
    else:
        raise TypeError("Unsupported type.")

def warp_landmarks(landmarks, displacement_field):
    landmarks_x = landmarks[:, 0]
    landmarks_y = landmarks[:, 1]
    ux = nd.map_coordinates(displacement_field[0, :, :], [landmarks_y, landmarks_x], mode='nearest')
    uy = nd.map_coordinates(displacement_field[1, :, :], [landmarks_y, landmarks_x], mode='nearest')
    new_landmarks = np.stack((landmarks_x + ux, landmarks_y + uy), axis=1)
    return new_landmarks

def calculate_resampling_ratio(x_sizes, y_sizes, min_resolution):
    x_size, y_size = max(x_sizes), max(y_sizes)
    min_size = min(x_size, y_size)
    if min_resolution > min_size:
        resampling_ratio = 1
    else:
        resampling_ratio = min_size / min_resolution
    return resampling_ratio

def resample(image, resample_ratio, cval=0, order=3):
    if isinstance(image, np.ndarray):
        return unp.resample(image, resample_ratio, cval=cval, order=order)
    elif isinstance(image, tc.Tensor):
        return utc.resample(image, resample_ratio, cval=cval, order=order)
    else:
        raise TypeError("Unsupported type.")

def calculate_new_shape_min(current_shape, min_size):
    if current_shape[0] > current_shape[1]:
        divider = current_shape[1] / min_size
    else:
        divider = current_shape[0] / min_size
    new_shape = (int(current_shape[0] / divider), int(current_shape[1] / divider))
    return new_shape

def resample_displacement_field(displacement_field, resample_ratio, cval=0, order=3):
    if isinstance(displacement_field, np.ndarray):
        return unp.resample_displacement_field(displacement_field, resample_ratio, cval=cval, order=order)
    elif isinstance(displacement_field, tc.Tensor):
        return utc.resample_displacement_field(displacement_field, resample_ratio, cval=cval, order=order)
    else:
        raise TypeError("Unsupported type.")  

def resample_displacement_field_to_size(displacement_field, new_shape, cval=0, order=3):
    if isinstance(displacement_field, np.ndarray):
        return unp.resample_displacement_field_to_size(displacement_field, new_shape, cval=cval, order=order)
    elif isinstance(displacement_field, tc.Tensor):
        warping_dict = {0: "nearest", 1: "bilinear", 3: "bicubic"}
        return utc.resample_displacement_field_to_size(displacement_field, new_shape, mode=warping_dict[order])
    else:
        raise TypeError("Unsupported type.")  

def unpad_displacement_field(displacement_field, padding_params):
    if isinstance(displacement_field, np.ndarray):
        return unp.unpad_displacement_field(displacement_field, padding_params)
    elif isinstance(displacement_field, tc.Tensor):
        return utc.unpad_displacement_field(displacement_field, padding_params)
    else:
        raise TypeError("Unsupported type.")  

def gaussian_smoothing(image, sigma):
    if isinstance(image, np.ndarray):
        return unp.gaussian_smoothing(image, sigma)
    elif isinstance(image, tc.Tensor):
        diagonal = calculate_diagonal(image)
        if diagonal > 10000:
            return utc.gaussian_smoothing_patch(image, sigma)
        else:
            return utc.gaussian_smoothing(image, sigma)
    else:
        raise TypeError("Unsupported type.")

def image_to_tensor(image : np.ndarray, device : str="cpu"):
    if len(image.shape) == 3:
        return tc.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    elif len(image.shape) == 2:
        return tc.from_numpy(image).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)

def tensor_to_image(tensor : tc.Tensor):
    if tensor.size(0) == 1:
        return tensor[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
    else:
        return tensor.permute(0, 2, 3, 1).detach().cpu().numpy()

def convert_to_gray(image):
    if isinstance(image, np.ndarray):
        return unp.convert_to_gray(image)
    elif isinstance(image, tc.Tensor):
        return utc.convert_to_gray(image)
    else:
        raise TypeError("Unsupported type.")  

def compose_displacement_fields(df_1, df_2):
    if isinstance(df_1, tc.Tensor) and isinstance(df_2, tc.Tensor):
        return unp.compose_displacement_fields(df_1, df_2)
    elif isinstance(df_1, np.ndarray) and isinstance(df_2, np.ndarray):
        return utc.compose_displacement_fields(df_1, df_2)

def replace_extension(path : pathlib.Path, to_replace=".jpg"):
    return pathlib.Path(os.path.splitext(path)[0] + to_replace)

def get_filename(path : pathlib.Path):
    return path.name

def get_combined_size(source, target):
    if isinstance(source, np.ndarray) and isinstance(target, np.ndarray):
        return unp.get_combined_size(source, target)
    elif isinstance(source, tc.Tensor) and isinstance(target, tc.Tensor):
        return utc.get_combined_size(source, target)
    else:
        raise TypeError("Unsupported type.")

def pad_to_same_size(source, target, pad_value=1.0):
    if isinstance(source, np.ndarray) and isinstance(target, np.ndarray):
        return unp.pad_to_same_size(source, target, pad_value=pad_value)
    elif isinstance(source, tc.Tensor) and isinstance(target, tc.Tensor):
        return utc.pad_to_same_size(source, target, pad_value=pad_value)
    else:
        raise TypeError("Unsupported type.")    

def pad_landmarks(landmarks, padding_size):
    y_pad = padding_size[0]
    x_pad = padding_size[1]
    landmarks[:, 0] = landmarks[:, 0] + x_pad[0]
    landmarks[:, 1] = landmarks[:, 1] + y_pad[0]
    return landmarks

def unpad_landmarks(landmarks, padding_size):
    y_pad = padding_size[0]
    x_pad = padding_size[1]
    landmarks[:, 0] = landmarks[:, 0] - x_pad[0]
    landmarks[:, 1] = landmarks[:, 1] - y_pad[0]
    return landmarks

def calculate_diagonal(image):
    if isinstance(image, np.ndarray):
        return unp.calculate_diagonal(image)
    elif isinstance(image, tc.Tensor):
        return utc.calculate_diagonal(image)
    else:
        raise TypeError("Unsupported type.")  

def np_df_to_tc_df(displacement_field_np: np.ndarray, device: str="cpu"):
    shape = displacement_field_np.shape
    ndim = len(shape) - 1
    if ndim == 2:
        displacement_field_tc = tc.from_numpy(displacement_field_np.copy())
        displacement_field_tc = displacement_field_tc.permute(1, 2, 0).unsqueeze(0)
        temp_df_copy = displacement_field_tc.clone()
        displacement_field_tc[:, :, :, 0] = temp_df_copy[:, :, :, 0] / (shape[2]) * 2.0
        displacement_field_tc[:, :, :, 1] = temp_df_copy[:, :, :, 1] / (shape[1]) * 2.0
    if ndim == 3:
        displacement_field_tc = tc.from_numpy(displacement_field_np.copy())
        displacement_field_tc = displacement_field_tc.permute(1, 2, 3, 0).unsqueeze(0)
        temp_df_copy = displacement_field_tc.clone()
        displacement_field_tc[:, :, :, :, 0] = temp_df_copy[:, :, :, :, 2] / (shape[3]) * 2.0
        displacement_field_tc[:, :, :, :, 1] = temp_df_copy[:, :, :, :, 0] / (shape[2]) * 2.0
        displacement_field_tc[:, :, :, :, 2] = temp_df_copy[:, :, :, :, 1] / (shape[1]) * 2.0
    return displacement_field_tc.to(device)

def tc_df_to_np_df(displacement_field_tc: tc.Tensor):
    ndim = len(displacement_field_tc.size()) - 2
    if ndim == 2:
        displacement_field_np = displacement_field_tc.detach().clone().cpu()[0].permute(2, 0, 1).numpy()
        shape = displacement_field_np.shape
        temp_df_copy = displacement_field_np.copy()
        displacement_field_np[0, :, :] = temp_df_copy[0, :, :] / 2.0 * (shape[2])
        displacement_field_np[1, :, :] = temp_df_copy[1, :, :] / 2.0 * (shape[1])
    elif ndim == 3:
        displacement_field_np = displacement_field_tc.detach().clone().cpu()[0].permute(3, 0, 1, 2).numpy()
        shape = displacement_field_np.shape
        temp_df_copy = displacement_field_np.copy()
        displacement_field_np[0, :, :, :] = temp_df_copy[1, :, :, :] / 2.0 * (shape[2])
        displacement_field_np[1, :, :, :] = temp_df_copy[2, :, :, :] / 2.0 * (shape[1])
        displacement_field_np[2, :, :, :] = temp_df_copy[0, :, :, :] / 2.0 * (shape[3])
    return displacement_field_np

def round_up_to_odd(value):
    return int(np.ceil(value) // 2 * 2 + 1)

def points_to_homogeneous_representation(points: np.ndarray):
    homogenous_points = np.concatenate((points, np.ones((points.shape[0], 1), dtype=points.dtype)), axis=1)
    return homogenous_points

def np_transform_to_np_df(transformation: np.ndarray, array_shape: tuple):
    grid_x, grid_y = np.meshgrid(np.arange(array_shape[1]), np.arange(array_shape[0]))
    y_size, x_size = array_shape

    points = np.vstack((grid_x.ravel(), grid_y.ravel(), np.ones(array_shape).ravel()))
    transformed_points = transformation @ points
    u_x = np.reshape(transformed_points[0, :], (y_size, x_size)) - grid_x
    u_y = np.reshape(transformed_points[1, :], (y_size, x_size)) - grid_y
    displacement_field = np.zeros((2, array_shape[0], array_shape[1]), dtype=np.float32)
    displacement_field[0, :, :] = u_x
    displacement_field[1, :, :] = u_y
    return displacement_field

def np_transform_to_tc_displacement_field(transform, input_template, output_template):
    df_np = np_transform_to_np_df(transform, (input_template.shape[0], input_template.shape[1]))
    displacement_field = np_df_to_tc_df(df_np).type_as(output_template)
    displacement_field = utc.resample_displacement_field_to_size(displacement_field, (output_template.size(2), output_template.size(3)))
    return displacement_field

def initial_resampling(source, target, resolution):
    source_y_size, source_x_size, target_y_size, target_x_size = get_combined_size(source, target)
    resample_ratio = calculate_resampling_ratio((source_x_size, target_x_size), (source_y_size, target_y_size), resolution)
    resampled_source = resample(gaussian_smoothing(source, min(max(resample_ratio -1, 0.1), 10)), resample_ratio, cval=0.0)
    resampled_target = resample(gaussian_smoothing(target, min(max(resample_ratio -1, 0.1), 10)), resample_ratio, cval=0.0)
    return resampled_source, resampled_target

def calculate_affine_transform(source_points, target_points):
    transform, _, _, _ = np.linalg.lstsq(source_points, target_points, rcond=None)
    transform = transform.T    
    return transform

def calculate_rigid_transform(source_points, target_points):
    target = target_points.ravel()
    source_homogenous_points = points_to_homogeneous_representation(source_points)
    source = np.zeros((2*source_points.shape[0], 2*source_points.shape[1]))
    source[0::2, 0:target_points.shape[1]+1] = source_homogenous_points[:, :]
    source[1::2, 0:target_points.shape[1]+1] = source_homogenous_points[:, :]
    source[1::2, 1], source[1::2, 0] = (-1) * source.copy()[1::2, 0], source.copy()[1::2, 1]
    source[1::2, 2] = 0
    source[1::2, 3] = 1
    inv_source = np.linalg.pinv(source)
    params = inv_source @ target 
    transform = np.array([
        [params[0], params[1], params[2]],
        [-params[1], params[0], params[3]],
        [0, 0, 1],
    ], dtype=source_points.dtype)
    return transform


