
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


def plot(dataset):
    
    x = dataset[0]
    z = dataset[1]
    y = dataset[2]
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    plt.show()