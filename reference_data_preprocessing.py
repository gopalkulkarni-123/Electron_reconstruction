import os
import random
import numpy as np
import torch
#import wandb

def prepare_paths(dir_in):
    '''
    Create list of paths to all files from simulations in dir_in(string)
    '''
    dirs = [os.path.join(dir_in, o) for o in os.listdir(dir_in) 
                    if os.path.isdir(os.path.join(dir_in, o))]

    files = []
    for dir_ in dirs:
        for x in os.listdir(dir_):
            files.append(dir_ + '/' + x)
    return files

def normalize_point(point, vmin, vmax, a=0., b=1.):
    '''
    Normalize point from a set of points with vmin(minimum) and vmax(maximum)
    to be in a range [a, b]
    '''
    return (a + (point - vmin) * (b - a) / ( vmax - vmin))

def denormalize_point(point, vmin, vmax, a=0., b=1.):
    '''
    Denormalize point from range [a, b]
    to be in set of points with vmin(minimum) and vmax(maximum)
    '''
    return ((point - a) * (vmax - vmin) / (b - a) + vmin)

def get_vmin_vmax(items):
    '''
    Find minima/maxima in all columns among a complete data(all simulation files)
    Args:
        items(list of string): list of paths to all electron clouds
        positions(list of string): suffixes of files that correspond to positions in beamline, where electron clouds were observed
        time_stamp(integer): index of column that corresponds to time stamp
        num_inputs(integer): number of input variables, that start from index 0 until num_inputs-1 in dimension 2
    
    returns torch tensors with minima and maxima
    '''
    
    for item in items:
        arr = np.load(item)

        #convert time to a spacial coordinate and center around 0
        #arr[:, time_stamp] = arr[:, time_stamp]*3e8 - np.mean(arr[:, time_stamp]*3e8)
        #arr = extend_to_hotvec(item, arr, positions, num_inputs)

        if item == items[0]:
            vmin = [np.min(arr[:, i]) for i in range(arr.shape[1])]
            vmax = [np.max(arr[:, i]) for i in range(arr.shape[1])]
        else:
            vmin = [min(np.min(arr[:, i]), vmin[i]) for i in range(arr.shape[1])]
            vmax = [max(np.max(arr[:, i]), vmax[i]) for i in range(arr.shape[1])]
    return torch.Tensor(vmin), torch.Tensor(vmax)

paths = prepare_paths(r'D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\data\validation_data_npy')

vmin, vmax = get_vmin_vmax(paths)

print("")