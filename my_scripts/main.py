from run import run_simulation
from helpers import reset_all_folders, normalize_weights, fill_notes, append_error_notes
from math import radians
import numpy as np
import torch as torch
import time, os
import yaml
import sys

if __name__ == "__main__":
    port_num = sys.argv[1]

    SEED_LIST = [41, 5, 2, 3, 10]#, 12, 1995]#, 100, 150, 200, 190, 0]
    ANIMATIONS = ["02_01", "05_08", "38_03"]#, "64_06", "06_03", "05_11", "05_15", "06_09", "07_10",
                 # "07_05", "64_11", "64_22", "64_26", "13_06", "14_32", "06_13", "14_01", "28_19"]
    #animations = {"02_01": len(SEED_LIST)}

    with open("config_file.yaml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    kalman_arguments = cfg["kalman_arguments"]
    parameters = cfg["parameters"]
    active_parameters =  cfg["active_parameters"]
    energy_parameters = cfg["energy_parameters"]

    date_time_name = time.strftime("%Y-%m-%d-%H-%M")
    if (parameters["run_loc"] == "local"):
        base_folder = "/Users/kicirogl/Documents/simulation/simulation_results"
        saved_vals_loc = "/Users/kicirogl/workspace/cvlabdata2/home/kicirogl/ActiveDrone/saved_vals"
    elif (parameters["run_loc"] == "server"):
        base_folder = "/cvlabdata2/home/kicirogl/ActiveDrone/simulation_results"
        saved_vals_loc = "/cvlabdata2/home/kicirogl/ActiveDrone/saved_vals"

    if energy_parameters["PROJECTION_METHOD"] == "scaled":
        energy_parameters["WEIGHTS"] =  {'proj': 0.25, 'smooth': 0.25, 'bone': 0.25, 'lift': 0.25}
    parameters["PORT"] = int(port_num)

    theta_list = [270]#list(range(270, 190, -35))#list(range(270, 235, -20))
    phi_list = list(range(0, 360, 45))
    active_parameters["POSITION_GRID"] = [[radians(theta),  radians(phi)] for theta in theta_list for phi in phi_list]

    #trajectory = 0-active, 1-constant_rotation, 2-random, 3-constant_angle, 4-wobbly_rotation, 5-updown, 6-leftright, 7-go_to_best, 8-go_to_worst
    TRAJECTORY_LIST = ["active"]

    ablation_study = False
    if ablation_study:
        num_of_experiments = 4
        is_quiet = True
        TRAJECTORY_LIST = ["active"]
    else:
        num_of_experiments = len(TRAJECTORY_LIST)

    for experiment_ind in range(num_of_experiments):
        file_names, folder_names, f_notes_name, _ = reset_all_folders(ANIMATIONS, SEED_LIST, base_folder, saved_vals_loc)
        
        parameters["FILE_NAMES"] = file_names
        parameters["FOLDER_NAMES"] = folder_names
        
        energy_parameters["WEIGHTS_FUTURE"] = energy_parameters["WEIGHTS"] 

        if ablation_study:
            if experiment_ind == 0:
                energy_parameters["WEIGHTS_FUTURE"]["proj"]=0
            elif experiment_ind == 1:
                energy_parameters["WEIGHTS_FUTURE"]["smooth"]=0
            elif experiment_ind == 2:
                energy_parameters["WEIGHTS_FUTURE"]["bone"]=0
            elif experiment_ind == 3:
                energy_parameters["WEIGHTS_FUTURE"]["lift"]=0

        active_parameters["TRAJECTORY"] = TRAJECTORY_LIST[experiment_ind]

        fill_notes(f_notes_name, parameters, energy_parameters, active_parameters)   

        for animation in ANIMATIONS:
            many_runs_current = []
            many_runs_middle = []
            many_runs_pastmost = []
            many_runs_overall = []
            for ind, seed in enumerate(SEED_LIST):
                parameters["ANIMATION_NUM"]=  animation
                parameters["SEED"] = seed
                parameters["EXPERIMENT_NAME"] = str(animation) + "_" + str(ind)
                ave_current_error, ave_middle_error, ave_pastmost_error, ave_overall_error = run_simulation(kalman_arguments, parameters, energy_parameters, active_parameters)
                many_runs_current.append(ave_current_error)
                many_runs_middle.append(ave_middle_error)
                many_runs_pastmost.append(ave_pastmost_error)
                many_runs_overall.append(ave_overall_error)

            append_error_notes(f_notes_name, animation, many_runs_current, many_runs_middle, many_runs_pastmost, many_runs_overall)
