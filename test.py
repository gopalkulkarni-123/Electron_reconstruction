import h5py
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def plot(dataset_1):

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(dataset_1[4], dataset_1[5], dataset_1[6])
    plt.show()

ref = r'D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\data\validation_data_npy\Track_1.017e-08_1.017e-08_1.778e-03_2.556e-03\Track_1.017e-08_1.017e-08_1.778e-03_2.556e-03-Img1.npy'
ref_ = np.load(ref)
ref_ =  ref_.T
plot(ref_)