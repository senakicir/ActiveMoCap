from run import run_simulation
from helpers import reset_all_folders, normalize_weights, fill_notes, append_error_notes
from math import radians
import numpy as np
import torch as torch
import time, os
import sys
import yaml

if __name__ == "__main__":
    port_num = sys.argv[1]

    SEED_LIST = [200, 3]
    ANIMATIONS = ["07_05", "06_03", "05_11"]

    with open("config_file.yaml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    kalman_arguments = cfg["kalman_arguments"]
    parameters = cfg["parameters"]
    active_parameters =  cfg["active_parameters"]
    energy_parameters = cfg["energy_parameters"]

    # run_loc = "server", "local"
    run_loc = "server"
    date_time_name = time.strftime("%Y-%m-%d-%H-%M")
    if (run_loc == "local"):
        base_folder = "/Users/kicirogl/Documents/simulation/grid_search_results/gs_" + date_time_name
        saved_vals_loc = "/Users/kicirogl/workspace/cvlabdata2/home/kicirogl/ActiveDrone/saved_vals"
    elif (run_loc == "server"):
        base_folder = "/cvlabdata2/home/kicirogl/ActiveDrone/grid_search_results/gs_" + date_time_name
        saved_vals_loc = "/cvlabdata2/home/kicirogl/ActiveDrone/saved_vals"

    while os.path.exists(base_folder):
        base_folder += "_b_"
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)  
            break
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)      

    parameters["PORT"] = int(port_num)
    parameters["LENGTH_OF_SIMULATION"] = 100
    energy_parameters["QUIET"] = True
    energy_parameters["ESTIMATION_WINDOW_SIZE"] = 5
    energy_parameters["FUTURE_WINDOW_SIZE"] = 3
    energy_parameters["INIT_POSE_MODE"] = "gt_with_noise"
    active_parameters["TRAJECTORY"] = "active"

    theta_list = [270]#list(range(270, 190, -35))#list(range(270, 235, -20))
    phi_list = list(range(0, 360, 45))
    active_parameters["POSITION_GRID"] = [[radians(theta),  radians(phi)] for theta in theta_list for phi in phi_list]

    
    file_errors = open(base_folder+"/errors.txt", "w")

    for weight_proj in np.logspace(-3,-1,3):
        for  weight_smooth in np.logspace(-2,0,3):
            for weight_bone  in np.logspace(-2,0,3):
                for weight_lift  in [0.1]:#np.logspace(-2,0,3):

                    file_names, folder_names, f_notes_name, _ = reset_all_folders(ANIMATIONS, SEED_LIST, base_folder, saved_vals_loc)
                    
                    parameters["FILE_NAMES"] = file_names
                    parameters["FOLDER_NAMES"] = folder_names
            
                    energy_parameters["WEIGHTS"] = {'proj': weight_proj, 'smooth': weight_smooth, 'bone': weight_bone, 'lift': weight_lift}
                    energy_parameters["WEIGHTS_FUTURE"] = energy_parameters["WEIGHTS"]

                    fill_notes(f_notes_name, parameters, energy_parameters, active_parameters)   

                    error_list =  []
                    for animation in ANIMATIONS:
                        many_runs_current = []
                        many_runs_middle = []
                        for ind, seed in enumerate(SEED_LIST):
                            parameters["ANIMATION_NUM"]=  animation
                            parameters["SEED"] = seed
                            parameters["EXPERIMENT_NAME"] = str(animation) + "_" + str(ind)
                            ave_current_error, ave_middle_error  = run_simulation(kalman_arguments, parameters, energy_parameters, active_parameters)
                            many_runs_current.append(ave_current_error)
                            many_runs_middle.append(ave_middle_error)
                            error_list.append(ave_middle_error)

                        append_error_notes(f_notes_name, many_runs_current, many_runs_middle, animation)

                    ave_error = sum(error_list)/len(error_list)

                    error_string =  str(weight_proj) + "\t" + str(weight_smooth) + "\t" + str(weight_bone) +"\t" + str(weight_lift) + "\t" + str(ave_error) + "\n"
                    file_errors.write(error_string)
