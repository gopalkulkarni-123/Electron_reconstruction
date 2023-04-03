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

paths = prepare_paths(r'D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\go_with_the_flows\Data\validation_data_npy')
list = []
electron_dataset = []

for i in range(len(paths)):
    list.append(torch.tensor(np.load(paths[i])))

for j in range(len(list)):
    data = {}
    ind = torch.randint(1, max(list[j].shape), (100000,))
    list[j] = list[j][ind]
    ind_cloud = torch.randint(1, 50000, (50000,))
    ind_eval_cloud = torch.randint(50000, max(list[j].shape), (50000,))
    data['cloud'] = list[j][ind_cloud].to(torch.float32).T
    data['eval_cloud'] = list[j][ind_eval_cloud].to(torch.float32).T
    data['label'] = j - 8*int(j/8)
    electron_dataset.append(data)

#print('end')