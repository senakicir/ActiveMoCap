import os
os.environ["MKL_NUM_THREADS"] = "8" 
os.environ["NUMEXPR_NUM_THREADS"] = "8" 
os.environ["OMP_NUM_THREADS"] = "8" 

from run import run_simulation
from helpers import reset_all_folders, fill_notes, append_error_notes
from math import radians
import numpy as np
import torch as torch
torch.set_num_threads(8)

import time, os
import yaml
import sys 

if __name__ == "__main__":

    # if you want to run multiple simulators on different ports, then you need to specify the port_num
    # this needs to match with the port_num you specify in the AirSim settings
    port_num = 45415
    if (len(sys.argv)>1):
        port_num = sys.argv[1]

    #specify which yaml file to use here
    with open("config_files/config_file_mpi.yaml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    parameters = cfg["parameters"]
    active_parameters =  cfg["active_parameters"]
    energy_parameters = cfg["energy_parameters"]
    
    #create a timestamped output folder
    base_folder = parameters["simulation_output_path"] + time.strftime("%Y-%m-%d-%H-%M")
    saved_vals_loc = parameters["test_sets_path"]
    test_sets_loc = parameters["test_sets_path"]
    while os.path.exists(base_folder):
        base_folder += "_b_"
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)  
            break
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)     

    
    parameters["PORT"] = int(port_num)
    ANIMATIONS = parameters["ANIMATION_LIST"]
    SEED_LIST = parameters["SEED_LIST"]
    TRAJECTORY_LIST = parameters["TRAJECTORY_LIST"]

    theta_list = [270]
    phi_list = list(range(0, 360, 20))
    active_parameters["POSITION_GRID"] = [[radians(theta),  radians(phi)] for theta in theta_list for phi in phi_list]


    #each trajectory is a separate experiment. (active, constant rotation etc.)
    num_of_experiments = len(TRAJECTORY_LIST)

    for experiment_ind in range(num_of_experiments):

        file_names, folder_names, f_notes_name, _ = reset_all_folders(ANIMATIONS, SEED_LIST, base_folder, saved_vals_loc, test_sets_loc)

        active_parameters["TRAJECTORY"] = TRAJECTORY_LIST[experiment_ind]
        parameters["FILE_NAMES"] = file_names
        parameters["FOLDER_NAMES"] = folder_names
        energy_parameters["WEIGHTS_FUTURE"] = energy_parameters["WEIGHTS"].copy() 

        fill_notes(f_notes_name, parameters, energy_parameters, active_parameters)   

        #we run each experiment for all animations specified
        for animation in ANIMATIONS:
            many_runs_current = []
            many_runs_middle = []
            many_runs_pastmost = []
            many_runs_overall = []

            #for the number of seeds specified
            for ind, seed in enumerate(SEED_LIST):
                parameters["ANIMATION_NUM"]=  animation
                parameters["SEED"] = seed
                parameters["EXPERIMENT_NUMBER"] = ind
                parameters["EXPERIMENT_NAME"] = str(animation) + "_" + str(ind)
                ave_current_error, ave_middle_error, ave_pastmost_error, ave_overall_error = run_simulation(kalman_arguments, parameters, energy_parameters, active_parameters)
                many_runs_current.append(ave_current_error)
                many_runs_middle.append(ave_middle_error)
                many_runs_pastmost.append(ave_pastmost_error)
                many_runs_overall.append(ave_overall_error)

            #we calculate the average error and std of all seeds for a specific animation and record it.
            append_error_notes(f_notes_name, animation, many_runs_current, many_runs_middle, many_runs_pastmost, many_runs_overall)
