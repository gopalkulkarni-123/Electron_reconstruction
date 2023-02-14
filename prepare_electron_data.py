import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,3], cmap='viridis')
    ax.set_title("4D data with color representing the fourth dimension")

    # Add a color bar to show the mapping between colors and values
    cbar = fig.colorbar(scatter)
    cbar.set_label("Fourth dimension value")

    # Show the plot
    plt.show()


Img1 = np.load(r'D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\go_with_the_flows\Data\validation_data_npy/Track_1.017e-08_1.017e-08_1.778e-03_1.778e-03/Track_1.017e-08_1.017e-08_1.778e-03_1.778e-03-Img1.npy')

#    Img2 = np.load(r'D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\go_with_the_flows\Data\validation_data_npy/Track_1.017e-08_1.017e-08_1.778e-03_1.778e-03/Track_1.017e-08_1.017e-08_1.778e-03_1.778e-03-Img2.npy')
#    Img4 = np.load(r'D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\go_with_the_flows\Data\validation_data_npy/Track_1.017e-08_1.017e-08_1.778e-03_1.778e-03/Track_1.017e-08_1.017e-08_1.778e-03_1.778e-03-Img4.npy')
#    Img5 = np.load(r'D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\go_with_the_flows\Data\validation_data_npy/Track_1.017e-08_1.017e-08_1.778e-03_1.778e-03/Track_1.017e-08_1.017e-08_1.778e-03_1.778e-03-Img5.npy')
#    Img6 = np.load(r'D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\go_with_the_flows\Data\validation_data_npy/Track_1.017e-08_1.017e-08_1.778e-03_1.778e-03/Track_1.017e-08_1.017e-08_1.778e-03_1.778e-03-Img6.npy')
#    lpa = np.load(r'D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\go_with_the_flows\Data\validation_data_npy/Track_1.017e-08_1.017e-08_1.778e-03_1.778e-03/Track_1.017e-08_1.017e-08_1.778e-03_1.778e-03-lpa.npy')
#    UndEnt = np.load(r'D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\go_with_the_flows\Data\validation_data_npy/Track_1.017e-08_1.017e-08_1.778e-03_1.778e-03/Track_1.017e-08_1.017e-08_1.778e-03_1.778e-03-UndEnt.npy')
#    UndMid = np.load(r'D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\go_with_the_flows\Data\validation_data_npy/Track_1.017e-08_1.017e-08_1.778e-03_1.778e-03/Track_1.017e-08_1.017e-08_1.778e-03_1.778e-03-Img1.npy')


electron_data_list = []
electron_data_list.append(Img1)
"""
electron_data_list.append(Img2)
electron_data_list.append(Img4)
electron_data_list.append(Img5)
electron_data_list.append(Img6)
electron_data_list.append(lpa)
electron_data_list.append(UndEnt)
electron_data_list.append(UndMid)
"""

def get_data():
    data = []
    for i in range (len(electron_data_list)):
        half_length = electron_data_list[i].shape[1] // 2
        cloud, eval_cloud = np.split(electron_data_list[i][:,4:8], [half_length], axis=0)
        temp = {'cloud':cloud.T, 'eval_cloud':eval_cloud.T}
        data.append(temp)

    return data