from run import run_simulation
from helpers import reset_all_folders, normalize_weights, fill_notes, append_error_notes
from math import radians
import numpy as np
import torch as torch

if __name__ == "__main__":
    kalman_arguments = {"KALMAN_PROCESS_NOISE_AMOUNT" :1, "KALMAN_MEASUREMENT_NOISE_AMOUNT_XY" : 1e-3}
    kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_Z"] = 1000 * kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_XY"]
    use_trackbar = False
    
    # simulation mode = "use_airsim", "saved_simulation"
    simulation_mode = "use_airsim"
    if (simulation_mode == "use_airsim"):
        base_folder = "/Users/kicirogl/Documents/temp_main"
    elif (simulation_mode == "saved_simulation"):
        base_folder = "/cvlabdata2/home/kicirogl/ActiveDrone/my_scripts/temp_main"

    #loop_mode = 0-normal, normal_teleport, 1-openpose, 2-teleport, 3-create_dataset
    loop_mode = "normal_teleport"
    #hessian_part: 0-future, 1-middle, 2-whole
    hessian_part = "whole"
    #uncertainty_calc_method: 0-sum_eig 1-add_diag 2-multip_eig 3-determinant 4-max_eig 5-root_six
    uncertainty_calc_method = "sum_eig"

    minmax = True #True-min, False-max
    SEED_LIST = [2, 12, 1995]#, 100, 150, 200, 190, 0]
    WOBBLE_FREQ_LIST = [0.5]#, 1, 2, 5, 20]
    delta_t = 0.1
    upper_lim = -3
    lower_lim = -0.5 #-2.5
    top_speed = 1.5
    go_distance = 2
    LOOKAHEAD_LIST = [0.5]
    if loop_mode == "normal_teleport":
        go_distance = 0.5
        LOOKAHEAD_LIST = [0.5]#[6]#[0.5]

    UPDOWN_LIM_LIST = [[upper_lim, lower_lim]]
    ftol = 1e-3

    is_quiet = False
    
    online_window_size = 6
    calibration_length = 30
    calibration_window_size = 20
    precalibration_length = 10
    length_of_simulation = 120
    
    #init_pose_mode: 0- "gt", "zeros", "backproj", "gt_with_noise"
    init_pose_mode = "backproj"

    find_best_traj = False
    num_of_noise_trials = 5

    noise_2d_std = 3
    noise_lift_std = 0.05

    noise_3d_init_std = 0.2
    drone_pos_jitter_noise_std = 0.5
    predefined_traj_len = 0

    use_symmetry_term = True
    use_single_joint = False
    use_bone_term = True
    use_lift_term = True
    use_trajectory_basis = False
    num_of_trajectory_param = 5
    #smoothness_mode: 0-velocity, 1-position, 2-all_connected, 3-only_velo_connected, 4-none
    smoothness_mode = "velocity"
    #lift_method: "simple", "complex"
    lift_method = "simple" 
    #bone_len_method: "no_sqrt, sqrt"
    bone_len_method = "sqrt" 
    #projection_method: "scaled, normal, normalized"
    projection_method = "normalized" 
    if projection_method == "normal" or projection_method == "normalized":
        weights =  {'proj': 0.000333, 'smooth': 0.333, 'bone': 0.333, 'lift': 0.333}
    elif projection_method == "scaled":
        weights =  {'proj': 0.25, 'smooth': 0.25, 'bone': 0.25, 'lift': 0.25}

    #modes: 0- "constant", "adaptive"
    reconstruction_energy = "constant"
    uncertainty_energy = "constant"

    parameters = {"USE_TRACKBAR": use_trackbar, "SIMULATION_MODE": simulation_mode,"LOOP_MODE":loop_mode, 
                  "FIND_BEST_TRAJ": find_best_traj, "PREDEFINED_TRAJ_LEN": predefined_traj_len, 
                  "NUM_OF_NOISE_TRIALS": num_of_noise_trials, "LENGTH_OF_SIMULATION":length_of_simulation}

    #mode_3d: 0- gt, 1- naiveback, 2- energy pytorch, 3-energy scipy
    #mode_2d: 0- gt, 1- gt_with_noise, 2- openpose
    #mode_lift: 0- gt, 1- gt_with_noise, 2-lift
    #bone_len: 0-gt, 1- calib_res
    modes = {"mode_3d":"scipy", "mode_2d":"gt_with_noise", "mode_lift":"gt_with_noise", "bone_len": "calib_res"}


    param_read_M = False
    if modes["mode_2d"] == "openpose":
        param_read_M = True

    param_find_M = False

    ANIMATIONS = ["38_03"]# ["02_01","38_03"]# "64_06", "06_03", "05_11", "05_15", "06_09", "07_10",
                 # "07_05", "64_11", "64_22", "64_26", "13_06", "14_32", "06_13", "14_01", "28_19"]
    #animations = {"02_01": len(SEED_LIST)}

    theta_list = list(range(270, 190, -35))#list(range(270, 235, -20))#list(range(270, 180, -40)) #list(range(270, 180, -20))
    phi_list = list(range(0, 360, 20))
    position_grid = [[radians(theta),  radians(phi)] for theta in theta_list for phi in phi_list]
    #active_sampling = "ellipse", "uniform"

    active_sampling_mode = "uniform"


    active_parameters ={"HESSIAN_PART":hessian_part, "UNCERTAINTY_CALC_METHOD":uncertainty_calc_method, 
                        "MINMAX":minmax, "THETA_LIST":theta_list, "PHI_LIST":phi_list, "POSITION_GRID":position_grid, 
                        "GO_DISTANCE":go_distance, "UPPER_LIM":upper_lim, "LOWER_LIM":lower_lim, "ACTIVE_SAMPLING_MODE":active_sampling_mode,
                        "TOP_SPEED": top_speed, "DELTA_T": delta_t}
    Z_POS_LIST = [-2.5]#, -4, -5, -6]
    

    #trajectory = 0-active, 1-constant_rotation, 2-random, 3-constant_angle, 4-wobbly_rotation, 5-updown, 6-leftright, 7-go_to_best, 8-go_to_worst
    TRAJECTORY_LIST = [ "active"]#, "constant_angle"]
    ablation_study = False
    grid_search = False
    
    if ablation_study:
        num_of_experiments = 4
        is_quiet = True
        TRAJECTORY_LIST = ["active"]
    elif grid_search:
        num_of_experiments = 8
        smooth_weights = torch.logspace(-4,-1,num_of_experiments)
        is_quiet = True
        TRAJECTORY_LIST = ["constant_rotation"]
        SEED_LIST = [41, 5, 2]
    else:
        num_of_experiments = len(TRAJECTORY_LIST)

    for experiment_ind in range(num_of_experiments):
        file_names, folder_names, f_notes_name, _ = reset_all_folders(ANIMATIONS, SEED_LIST, base_folder)
        
        parameters["FILE_NAMES"] = file_names
        parameters["FOLDER_NAMES"] = folder_names
        
        weights_future =  {'proj': 0.000333, 'smooth': 0.33, 'bone': 0, 'lift': 0.33}
        #weights_future =  {'proj': 0.33, 'smooth': 0.33, 'bone': 0, 'lift': 0.33}

        if grid_search:
            weights_future['proj'] = smooth_weights[experiment_ind]
            weights['proj'] =  smooth_weights[experiment_ind]
        if ablation_study:
            weights_future["bone"]=0.33
            if experiment_ind == 0:
                weights_future["proj"]=0
            elif experiment_ind == 1:
                weights_future["smooth"]=0
            elif experiment_ind == 2:
                weights_future["bone"]=0
            elif experiment_ind == 3:
                weights_future["lift"]=0
            

        energy_parameters = {"LIFT_METHOD":lift_method, "BONE_LEN_METHOD":bone_len_method, "ONLINE_WINDOW_SIZE": online_window_size, 
                            "CALIBRATION_WINDOW_SIZE": calibration_window_size, "CALIBRATION_LENGTH": calibration_length, 
                            "PRECALIBRATION_LENGTH": precalibration_length, "PARAM_FIND_M": param_find_M, "PARAM_READ_M": param_read_M, 
                            "QUIET": is_quiet, "MODES": modes, "MODEL": "mpi", "METHOD": "trf", "FTOL": ftol, "WEIGHTS": weights,
                            "INIT_POSE_MODE": init_pose_mode, "NOISE_2D_STD": noise_2d_std, "USE_SYMMETRY_TERM": use_symmetry_term, 
                            "USE_SINGLE_JOINT": use_single_joint, "SMOOTHNESS_MODE": smoothness_mode, "USE_TRAJECTORY_BASIS": use_trajectory_basis,
                            "NUMBER_OF_TRAJ_PARAM": num_of_trajectory_param, "NOISE_LIFT_STD": noise_lift_std, "NOISE_3D_INIT_STD": noise_3d_init_std,
                            "PROJECTION_METHOD" :projection_method, "WEIGHTS_FUTURE":weights_future}
        energy_parameters["USE_LIFT_TERM"] = use_lift_term
        energy_parameters["USE_BONE_TERM"] = use_bone_term

        active_parameters["UPDOWN_LIM"] = UPDOWN_LIM_LIST[0]
        active_parameters["WOBBLE_FREQ"] = WOBBLE_FREQ_LIST[0]
        active_parameters["Z_POS"] = Z_POS_LIST[0]
        active_parameters["LOOKAHEAD"] = LOOKAHEAD_LIST[0]

        active_parameters["TRAJECTORY"] = TRAJECTORY_LIST[experiment_ind]

        fill_notes(f_notes_name, parameters, energy_parameters, active_parameters)   

        for animation in ANIMATIONS:
            many_runs_last = []
            many_runs_middle = []
            for ind, seed in enumerate(SEED_LIST):
                parameters["ANIMATION_NUM"]=  animation
                parameters["SEED"] = seed
                parameters["EXPERIMENT_NAME"] = str(animation) + "_" + str(ind)
                errors = run_simulation(kalman_arguments, parameters, energy_parameters, active_parameters)
                many_runs_last.append(errors["ave_3d_err"] )
                many_runs_middle.append(errors["middle_3d_err"] )

            append_error_notes(f_notes_name, many_runs_last, many_runs_middle, animation)
