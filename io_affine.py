import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import instance_optimization as io

import cost_functions as cf
import utils as u

def instance_optimization_affine_registration(source, target, initial_transform, params):
    device = params['device']
    echo = params['echo']
    cost_function = params['cost_function']
    cost_function_params = params['cost_function_params']
    resolution = params['registration_size']
    num_levels = params['num_levels']
    used_levels = params['used_levels']
    iterations = params['iterations']
    learning_rate = params['learning_rate']

    if type(cost_function) == str:
        cost_function = cf.get_function(cost_function)

    ### Initial resampling ###
    resampled_source, resampled_target = u.initial_resampling(source, target, resolution)
    if echo:
        print(f"Resampled source size: {resampled_source.size()}")
        print(f"Resampled target size: {resampled_target.size()}")
    initial_cost_function = cost_function(resampled_source, resampled_target, device=device, **cost_function_params)
    if echo:
        print(f"Initial objective function: {initial_cost_function.item()}")

    ### Affine Registration ###
    transform = io.affine_registration(resampled_source, resampled_target, num_levels, used_levels, iterations, learning_rate, cost_function, cost_function_params, device=device, initial_transform=initial_transform, echo=echo, return_best=True)
    if echo:
        print(f"Final transform: {transform}")
    return transform
