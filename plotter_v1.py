#import h5py
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
"""
with h5py.File('ShapeNetCore55v2_meshes_resampled.h5', 'r') as f:
    ls = list(f.keys())
    data = f.get('test_vertices_c')
    datasets = np.array(data)
    #print(f.keys.test_vertices_c)
"""
def plot(datasets):
    z = datasets[:,0]
    #print(z)
    x = datasets[:,1]
    #print(x)
    y = datasets[:,2]
    #print(y)

    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    
    # Creating plot
    ax.scatter3D(x, y, z, color = "green")
    plt.title("simple 3D scatter plot")
    
    # show plot
    plt.show()

"""

with open('airplane_data_v2.txt', 'w') as k:

    for i  in range (datasets.shape[0]):
       k.write(str(datasets[i])+"\n") 
       
"""
       