import os
os.environ["MKL_NUM_THREADS"] = "8" 
os.environ["NUMEXPR_NUM_THREADS"] = "8" 
os.environ["OMP_NUM_THREADS"] = "8" 

from run import run_simulation
from helpers import reset_all_folders, normalize_weights, fill_notes, append_error_notes
from math import radians
import numpy as np
import torch as torch
torch.set_num_threads(8)

import time, os
import yaml
import sys 

if __name__ == "__main__":
    port_num = sys.argv[1]
    SEED_LIST = [5,41,3,10,12]#[5, 41, 3, 10, 12]#, 12, 1995, 100, 150, 200, 190, 0]

    ANIMATIONS =  ["38_03"]#, "02_01", "38_03"]#["05_08", "02_01", "38_03"]#["02_01","38_03","05_08"]#, "38_03", "05_08"]#,"38_03", "05_08"]#["05_08", "02_01", "38_03"]#["02_01", "05_08", "38_03"]#["02_01", "05_08", "38_03"]#,"02_01", "38_03"]#, "02_01", "38_03"]#, "14_32"]#["05_08", "02_01", "38_03"]#["05_08", "02_01", "14_32"]#["05_08", "02_01", "14_32"]#, "38_03", ]#, "14_32"]#, "14_32"]#["05_08", "02_01", "38_03"] #, "06_13", "13_06"]#, "13_06", "28_19"]#["06_13"]#,"13_06"]#,"28_19"] #, "05_08", "28_19"]#,"02_01"]#, "38_03"]#, "14_32"]#["05_08", "02_01", "38_03"]#, "02_01", "38_03", "14_32"]#, "02_01", "38_03"]#["05_08", "02_01", "38_03"]#, "02_01", "38_03"]#, "14_32"]#, "05_08", "38_03"]#, "14_32", "06_13", "13_06", "28_19"]#, "13_06", "28_19"]#, "06_13", "13_06", "28_19"]#,"05_08", "38_03"]#, "64_06", "06_03", "05_11", "05_15", "06_09", "07_10",
                  #"07_05", "64_11", "64_22", "64_26", "13_06", "14_32", "06_13", "14_01", "28_19"]
                  #validation set: ["06_13", "13_06", "28_19"]
                  #test set: ["05_08", "02_01", "38_03", "14_32"]
                  #new test set, jitter: ["05_08", "02_01", "38_03"]
                  #"mpi_inf_3dhp"
    #animations = {"02_01": len(SEED_LIST)}

    with open("config_file.yaml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    #never play with this section
    kalman_arguments = cfg["kalman_arguments"]
    parameters = cfg["parameters"]
    active_parameters =  cfg["active_parameters"]
    energy_parameters = cfg["energy_parameters"]
    date_time_name = time.strftime("%Y-%m-%d-%H-%M")
    if (parameters["run_loc"] == "local"):
        base_folder = "/Users/kicirogl/Documents/simulation/simulation_results/experiments_" + date_time_name
        saved_vals_loc = "/Users/kicirogl/workspace/cvlabsrc1/home/kicirogl/ActiveDrone/saved_vals"
        test_sets_loc = "/Users/kicirogl/workspace/cvlabsrc1/home/kicirogl/ActiveDrone/test_sets"
    elif (parameters["run_loc"] == "server"):
        base_folder = "/cvlabsrc1/home/kicirogl/ActiveDrone/simulation_results/experiments_" + date_time_name
        saved_vals_loc = "/cvlabsrc1/home/kicirogl/ActiveDrone/saved_vals"
        test_sets_loc = "/cvlabsrc1/home/kicirogl/ActiveDrone/test_sets"
    while os.path.exists(base_folder):
        base_folder += "_b_"
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)  
            break
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)      
    if energy_parameters["PROJECTION_METHOD"] == "scaled":
        energy_parameters["WEIGHTS"] =  {'proj': 0.25, 'smooth': 0.25, 'bone': 0.25, 'lift': 0.25}
    parameters["PORT"] = int(port_num)

    theta_list = [270]#, 250, 230]#list(range(270, 190, -35))#list(range(270, 235, -20))
    phi_list = list(range(0, 360, 20))
    active_parameters["POSITION_GRID"] = [[radians(theta),  radians(phi)] for theta in theta_list for phi in phi_list]
    #####


    #trajectory = 0-active, 1-constant_rotation, 2-random, 3-constant_angle, 4-wobbly_rotation, 5-updown, 6-leftright, 7-oracle, 8-go_to_worst
    TRAJECTORY_LIST = ["active"]#["random", "constant_angle"]#["constant_rotation", "active", "random", "constant_angle"]#["oracle"]#["active", "constant_rotation", "random", "constant_angle"]#, "random", "constant_angle"]#["active"]#["active"]#["constant_rotation", "random", "constant_angle"]

    ablation_study = False
    find_weights = True

    if ablation_study:
        num_of_experiments = 5
        is_quiet = False
        TRAJECTORY_LIST = ["active", "constant_rotation"]
    else:
        num_of_experiments = len(TRAJECTORY_LIST)


    num_of_experiments_layer_2 = 1
    if find_weights:
        num_of_experiments_layer_2 = 2

    for exp_ind_2 in range(num_of_experiments_layer_2):
        # if find_weights:
        #     if exp_ind_2 == 0:
        #         energy_parameters["WEIGHTS_FUTURE"]["proj"]=0.0001
        #     if exp_ind_2 == 1:
        #         energy_parameters["WEIGHTS_FUTURE"]["proj"]=0.0001
        #     if exp_ind_2 == 2:
        #         energy_parameters["WEIGHTS_FUTURE"]["proj"]=0.00001
        #     if exp_ind_2 == 3:
        #         energy_parameters["WEIGHTS_FUTURE"]["proj"]=0.00001
        if exp_ind_2 == 0:
            active_parameters["PRIMARY_ROTATION_DIR"] = "r"
        elif exp_ind_2 == 1:
            active_parameters["PRIMARY_ROTATION_DIR"] = "l"

        for experiment_ind in range(num_of_experiments):

            active_parameters["TRAJECTORY"] = TRAJECTORY_LIST[experiment_ind]
            #if energy_parameters["MODES"]["mode_2d"] == "openpose" and energy_parameters["MODES"]["mode_lift"] == "lift":
                #if active_parameters["TRAJECTORY"] == "random":
                #    SEED_LIST = [41, 5, 2, 3, 10]
                #elif active_parameters["TRAJECTORY"] == "active":
                #    SEED_LIST = [41, 5, 2, 3, 10]
                #else:
                #SEED_LIST = [41, 5, 2, 3, 10]


            file_names, folder_names, f_notes_name, _ = reset_all_folders(ANIMATIONS, SEED_LIST, base_folder, saved_vals_loc, test_sets_loc)
            
            parameters["FILE_NAMES"] = file_names
            parameters["FOLDER_NAMES"] = folder_names
            
            if ablation_study:
                energy_parameters["WEIGHTS_FUTURE"] = energy_parameters["WEIGHTS"].copy() 
                if experiment_ind == 0:
                    energy_parameters["WEIGHTS_FUTURE"]["proj"]=0.001
                if experiment_ind == 4:
                    energy_parameters["WEIGHTS_FUTURE"]["proj"]=0.01
                elif experiment_ind == 3:
                    energy_parameters["WEIGHTS_FUTURE"]["smooth"]=0
                elif experiment_ind == 2:
                    energy_parameters["WEIGHTS_FUTURE"]["bone"]=0
                    energy_parameters["WEIGHTS_FUTURE"]["lift"]=0
                elif experiment_ind == 1:
                    energy_parameters["WEIGHTS_FUTURE"]["smooth"]=0
                    energy_parameters["WEIGHTS_FUTURE"]["bone"]=0
                    energy_parameters["WEIGHTS_FUTURE"]["lift"]=0

            
            fill_notes(f_notes_name, parameters, energy_parameters, active_parameters)   

            for animation in ANIMATIONS:
                many_runs_current = []
                many_runs_middle = []
                many_runs_pastmost = []
                many_runs_overall = []
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

                append_error_notes(f_notes_name, animation, many_runs_current, many_runs_middle, many_runs_pastmost, many_runs_overall)
