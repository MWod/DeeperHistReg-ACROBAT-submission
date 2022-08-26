import sys
current_file = sys.modules[__name__]

import utils as u

import io_nonrigid as ion

def identity_nonrigid_registration(source, target, initial_displacement_field,params):
    return u.create_identity_displacement_field(source)

def instance_optimization_nonrigid_registration(source, target, initial_displacement_field, params):
    return ion.instance_optimization_nonrigid_registration(source, target, initial_displacement_field, params)

def get_function(function_name):
    return getattr(current_file, function_name)