import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


def plot(dataset, title):
    for i in range(len(dataset)):
        dataset_plot = dataset[i]['cloud']
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)

        ax.scatter(dataset_plot[0],dataset_plot[1],dataset_plot[2])

        #ax.scatter(dataset_1[0], dataset_1[1], dataset_1[2])
        #ax.scatter(dataset_2[0], dataset_2[1], dataset_2[2])
        plt.show()