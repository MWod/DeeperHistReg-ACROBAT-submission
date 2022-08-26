import sys
current_file = sys.modules[__name__]

import utils as u

def target_landmarks_postprocessing(target_landmarks, params):
    late_resample = params['late_resample']
    if late_resample:
        late_resample_ratio = params['late_resample_ratio']
        target_landmarks = target_landmarks * late_resample_ratio

    pad_to_same_size = params['pad_to_same_size']
    if pad_to_same_size:
        padding_params = params['padding_params']
        print(f"Padding params: {padding_params}")
        target_landmarks = u.unpad_landmarks(target_landmarks, padding_params['pad_1'])

    initial_resampling = params['initial_resampling']
    if initial_resampling:
        initial_resample_ratio = params['initial_resample_ratio']
        target_landmarks = target_landmarks * initial_resample_ratio
    return target_landmarks

def get_function(function_name):
    return getattr(current_file, function_name)