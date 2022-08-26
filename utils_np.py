import numpy as np
import scipy.ndimage as nd


def normalize(image : np.ndarray):
    if len(image.shape) == 2:
        return (image - np.min(image)) / (np.max(image) - np.min(image))
    elif len(image.shape) == 3:
        normalized_image = np.zeros_like(image)
        for i in range(normalized_image.shape[2]):
            normalized_image[:, :, i] = normalize(image[:, :, i])
        return normalized_image
    else:
        raise ValueError("Unsupported number of channels.")

def resample(image, resample_ratio, cval=0, order=3):
    if len(image.shape) == 2:
        y_size, x_size = image.shape
        new_y_size, new_x_size = int(y_size / resample_ratio), int(x_size / resample_ratio)
        grid_x, grid_y = np.meshgrid(np.arange(new_x_size), np.arange(new_y_size))
        grid_x = grid_x * (x_size / new_x_size)
        grid_y = grid_y * (y_size / new_y_size)
        resampled_image = nd.map_coordinates(image, [grid_y, grid_x], cval=cval, order=3) 
    elif len(image.shape) == 3:
        y_size, x_size, num_channels = image.shape
        new_y_size, new_x_size = int(y_size / resample_ratio), int(x_size / resample_ratio)
        grid_x, grid_y = np.meshgrid(np.arange(new_x_size), np.arange(new_y_size))
        grid_x = grid_x * (x_size / new_x_size)
        grid_y = grid_y * (y_size / new_y_size)
        resampled_image = np.zeros((grid_x.shape[0], grid_x.shape[1], num_channels))
        for i in range(num_channels):
            resampled_image[:, :, i] = nd.map_coordinates(image[:, :, i], [grid_y, grid_x], cval=cval, order=order) 
    else:
        raise ValueError("Unsupported number of channels.")
    return resampled_image

def resample_displacement_field(displacement_field, resample_ratio, cval=0, order=3):
    raise NotImplementedError

def unpad_displacement_field(displacement_field, padding_params):
    raise NotImplementedError

def gaussian_smoothing(image, sigma):
    if len(image.shape) == 2:
       return nd.gaussian_filter(image, sigma)
    elif len(image.shape) == 3:
        _, _, num_channels = image.shape
        smoothed_image = np.zeros_like(image)
        for i in range(num_channels):
            smoothed_image[:, :, i] = nd.gaussian_filter(image[:, :, i], sigma)
        return smoothed_image
    else:
        raise ValueError("Unsupported number of channels.")

def create_identity_displacement_field(image : np.ndarray):
    raise NotImplementedError # TODO

def compose_displacement_fields(df_1 : np.ndarray, df_2 : np.ndarray):
    pass #TODO

def warp_image(image, displacement_field, cval=1.0, order=1):
    raise NotImplementedError #TODO

def get_combined_size(image_1 : np.ndarray, image_2 : np.ndarray):
    image_1_y_size, image_1_x_size = image_1.shape[0:2]
    image_2_y_size, image_2_x_size = image_2.shape[0:2]
    return image_1_y_size, image_1_x_size, image_2_y_size, image_2_x_size

def pad_to_same_size(image_1 : np.ndarray, image_2 : np.ndarray, pad_value : float=1.0):
    raise NotImplementedError()

def calculate_diagonal(image : np.ndarray):
    raise NotImplementedError()

def convert_to_gray(image : np.ndarray):
    raise NotImplementedError()