# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm, colors
from sklearn.cluster import KMeans
import os

input_file_loc = "/Users/kicirogl/Documents/temp_main/2019-09-04-14-27/02_01/0/trajectory_list.txt"
f_traj = open(input_file_loc, "r")

dir_name = "uncertainties_vis" 
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

for plot_num in range(15):
    uncertainty_matrix = np.zeros([8,8])

    #read file 

    first_word = ""
    #stop when you see chosen
    counter = 0
    while first_word != "Goal":
        #save when you see uncertainty mark
        line = f_traj.readline()
        word_list = line.split()
        stripped_word_list = [word.strip() for word in word_list]
        first_word = stripped_word_list[0]
        if first_word == "Linecount:":
            linecount = stripped_word_list[-1]
            #print("Linecount is", linecount)
        if first_word == "Trajectory":
            uncertainty = float(stripped_word_list[-1])
            uncertainty_matrix[counter//8, counter%8] = uncertainty
            #print("uncertainty is", uncertainty)
            counter += 1

    #plot
    fig = plt.figure()
    im =plt.imshow(uncertainty_matrix)
    file_name = dir_name + "/uncertainties_" + linecount + ".png"
    plt.colorbar(im)
    plt.savefig(file_name)
    plt.close(fig)