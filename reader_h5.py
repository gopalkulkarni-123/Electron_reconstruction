import h5py
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import entropy, wasserstein_distance

def jsd(p, q):
    """
    Jensen-Shannon divergence between two probability distributions
    """
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2

def emd(p, q):
    """
    Earth Mover's Distance between two probability distributions
    """
    return wasserstein_distance(p, q)

def plot(dataset_1, dataset_2, j, title):

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(dataset_1[j+0], dataset_1[j+1], dataset_1[j+2], label='sampled cloud')
    ax.scatter(dataset_2[j+4], dataset_2[j+5], dataset_2[j+6], label='reference cloud')
    ax.legend()
    ax.set_title(title)
    plt.show()

filename = r'D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\Electron_reconstruction\results\airplane_gen_model\electron_250_v1_Img2.h5'
ref = r'D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\data\validation_data_npy\Track_1.017e-08_1.017e-08_1.778e-03_1.778e-03\Track_1.017e-08_1.017e-08_1.778e-03_1.778e-03-Img2.npy'
with h5py.File(filename, 'r') as file:
    sampled_clouds=np.array(file.get('sampled_clouds'))

ref_cloud = np.load(ref)
num_points = 5000
random_indices_ref = np.random.choice(ref_cloud.shape[0], num_points, replace=False)
random_indices_sampled = np.random.choice(sampled_clouds.shape[2], num_points, replace=False)

random_points = ref_cloud[random_indices_ref]
sampled_clouds = sampled_clouds[0].T[random_indices_sampled]

#ranges_sample = np.array((min(random_points[0]), min(random_points[1]), min(random_points[2]), max(random_points[0]), max(random_points[1]), max(random_points[2])))

plot(sampled_clouds.T, random_points.T, 0,'title' )
