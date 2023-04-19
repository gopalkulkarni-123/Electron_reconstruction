import os
import numpy as np
import torch
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from torch.utils.data import random_split
#import wandb

def plot(dataset_1, title):

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    
    for j in range(2):
        ax.scatter(dataset_1[j+0], dataset_1[j+1], dataset_1[j+2], label='sampled cloud')
        #ax.scatter(dataset_2[j+4], dataset_2[j+5], dataset_2[j+6], label='reference cloud')
        ax.legend()
        ax.set_title(title)
        plt.show()

def prepare_paths(dir_in):
    '''
    Create list of paths to all files from simulations in dir_in(string)
    '''
    files = []
    for x in os.listdir(dir_in):
        files.append(dir_in + "\\" + x)
    
    return files

def normalize_tensor(data):
    
    for j in range (data.shape[0]):
        for i in range(data.shape[1]):
            vmax = torch.max(data[j][i])
            vmin = torch.min(data[j][i])

            data[j][i] = (0. + (data[j][i] - vmin) * (1. - 0.) / (vmax - vmin))
    return data

num_particles = 10000
num_dimensions = 6

paths = prepare_paths(r'D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\data\validation_data_npy\Track_1.017e-08_1.017e-08_1.778e-03_1.778e-03')
list = []
electron_data = torch.zeros(len(paths), num_dimensions, num_particles )

for i in range(len(paths)):
    list.append(torch.tensor(np.load(paths[i])))

for j in range(len(list)):

    ind = torch.randint(1, max(list[j].shape), (num_particles,))
    list[j] = list[j][ind]

    ind_cloud = torch.randint(1, num_particles, (num_particles,))
    electron_data[j] = (list[j][ind_cloud][:, 4:-1].to(torch.float32).T)
    #electron_dataset[j][4] = electron_dataset[j][4]*(3*10**8)

electron_data[:, 4, :] = ((electron_data[:, 4, :] * 3e8) - torch.mean((electron_data[:, 4, :] * 3e8)))

e_data = normalize_tensor(electron_data)

electron_dataset = []
for i in range(e_data.shape[0]):
    electron_data_processed = {'cloud':0, 'eval_cloud':0, 'label':0}
    f = e_data[i].T
    ind = torch.randn(10000, 2)
    half_size = ind.size(0) // 2
    cloud = f[:half_size]
    eval_cloud = f[half_size:]
    electron_data_processed['cloud'] = cloud.T
    electron_data_processed['eval_cloud'] = eval_cloud.T
    electron_data_processed['label'] = i
    electron_dataset.append(electron_data_processed)