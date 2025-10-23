import xarray as xr 
from pathlib import Path
import yaml
import torch 
import random
import os 
import numpy as np 

def load_data(data):
    """
    Takes either a pathlib.Path, str or an xarray.Dataset.

    Parameters:
        data (Path or xr.Dataset): The input data.

    Returns:
        xr.Dataset: The loaded or unchanged dataset.
    """
    if isinstance(data, (Path, str)):
        return xr.open_dataset(data).load()
    elif isinstance(data, xr.Dataset):
        return data
    else:
        raise TypeError("Input must be either a pathlib.Path, Str, or an xarray.Dataset")
    
def load_yaml(file_path):
    with open(file_path) as fd:
        settings = yaml.load(fd, Loader=yaml.SafeLoader)
    return settings

def write_yaml(yaml_obj, save_path):
    with open(save_path, 'w') as outfile:
        yaml.dump(yaml_obj, outfile, default_flow_style=False)

