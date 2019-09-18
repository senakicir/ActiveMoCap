# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm, colors
from sklearn.cluster import KMeans
import os

input_file_loc = "/Users/kicirogl/Documents/temp_main/grid_search_len1_ver3/errors.txt"

dir_name = "/Users/kicirogl/Documents/temp_main/grid_search_len1_ver3/grid_search_res" 
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

whole_file = pd.read_csv(input_file_loc, sep='\t', header=None).values
errors = whole_file[:,-1]
index=np.argmin(errors)
param = whole_file[index, :-1]

print("param=", param)
param_dict = {0:"proj", 1:"smooth", 2:"bone", 3:"lift"}

for param_ind in range(4):
    current_param = param[param_ind]
    other_param = list(range(4))
    other_param.remove(param_ind)

    indices_1 = [whole_file[:, other_param[0]] == param[other_param[0]]]
    indices_2 = [whole_file[:, other_param[1]] == param[other_param[1]]] 
    indices_3 = [whole_file[:, other_param[2]] == param[other_param[2]]]
    indices = np.logical_and(indices_1[0], indices_2[0], indices_3[0])

    errors = whole_file[indices, -1]
    fig = plt.figure()
    plt.plot(list(range(errors.shape[0])), errors, marker = "*")
    plt.savefig(dir_name+"/param_"+ param_dict[param_ind] +".png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)


