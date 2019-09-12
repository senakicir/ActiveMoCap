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
        base_folder = "/Users/kicirogl/Documents/temp_main/grid_search"
    elif (simulation_mode == "saved_simulation"):
        base_folder = "/cvlabdata2/home/kicirogl/ActiveDrone/my_scripts/temp_main"

    #loop_mode = 0-normal_simulation, teleport_simulation, 1-openpose, 2-toy_example, 3-create_dataset
    loop_mode = "teleport_simulation"
    #hessian_part: 0-future, 1-middle, 2-whole
    hessian_part = "whole"
    #uncertainty_calc_method: 0-sum_eig 1-add_diag 2-multip_eig 3-determinant 4-max_eig 5-root_six
    uncertainty_calc_method = "sum_eig"

    minmax = True #True-min, False-max
    SEED_LIST = [0, 5]#,5,3]#[41, 5, 2]#, 100, 150, 200, 190, 0]
    wobble_freq = 0.5#, 1, 2, 5, 20]
    delta_t = 0.1
    upper_lim = -3
    lower_lim = -0.5 #-2.5
    updown_lim = [upper_lim, lower_lim]

    top_speed = 3
    z_pos =  -2.5

    go_distance = 0.5
    lookahead = 0.5

    ftol = 1e-3

    is_quiet = True
    
    estimation_window_size = 5
    future_window_size = 1
    calibration_length = 30
    calibration_window_size = 20
    precalibration_length = 10
    length_of_simulation = 100
    
    #init_pose_mode: 0- "gt", "zeros", "backproj", "gt_with_noise"
    init_pose_mode = "gt_with_noise"

    find_best_traj = False
    num_of_noise_trials = 5

    noise_2d_std = 3
    noise_lift_std = 0.05

    noise_3d_init_std = 0.5
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
    projection_method = "normal" 

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
    param_find_M = False

    ANIMATIONS = ["07_05", "06_03", "05_11"]#["02_01", "05_08", "14_32,  "64_06", "38_03"]#
                 #  "02_01", "05_08", "38_03", "64_06", "06_03", "05_11", "05_15", "06_09", "07_10",
                 # "07_05", "64_11", "64_22", "64_26", "13_06", "14_32", "06_13", "14_01", "28_19"]

    theta_list = [270]#list(range(270, 190, -35))#list(range(270, 235, -20))#list(range(270, 180, -40)) #list(range(270, 180, -20))
    phi_list = list(range(0, 360, 45))
    position_grid = [[radians(theta),  radians(phi)] for theta in theta_list for phi in phi_list]
    
    #active_sampling = "ellipse", "uniform"
    active_sampling_mode = "uniform"

    active_parameters ={"HESSIAN_PART":hessian_part, "UNCERTAINTY_CALC_METHOD":uncertainty_calc_method, 
                        "MINMAX":minmax, "THETA_LIST":theta_list, "PHI_LIST":phi_list, "POSITION_GRID":position_grid, 
                        "GO_DISTANCE":go_distance, "UPPER_LIM":upper_lim, "LOWER_LIM":lower_lim, "ACTIVE_SAMPLING_MODE":active_sampling_mode,
                        "TOP_SPEED": top_speed, "DELTA_T": delta_t, "LOOKAHEAD": lookahead,  "Z_POS": z_pos, "WOBBLE_FREQ": wobble_freq, 
                        "UPDOWN_LIM": updown_lim}
    
    #trajectory = 0-active, 1-constant_rotation, 2-random, 3-constant_angle, 4-wobbly_rotation, 5-updown, 6-leftright, 7-go_to_best, 8-go_to_worst
    TRAJECTORY_LIST = ["active"]
    grid_search = False


    file_errors = open(base_folder+"/errors.txt", "w")

    for weight_proj in np.logspace(-3,-1,3):
        for  weight_smooth in np.logspace(-2,0,3):
            for weight_bone  in  np.logspace(-2,0,3):

                weight_lift = 2.1-(weight_proj+weight_smooth+weight_bone)/2

                file_names, folder_names, f_notes_name, _ = reset_all_folders(ANIMATIONS, SEED_LIST, base_folder)
                
                parameters["FILE_NAMES"] = file_names
                parameters["FOLDER_NAMES"] = folder_names
        
                weights =  {'proj': weight_proj, 'smooth': weight_smooth, 'bone': weight_bone, 'lift': weight_lift}
                weights_future =  weights

                energy_parameters = {"LIFT_METHOD":lift_method, "BONE_LEN_METHOD":bone_len_method, "ESTIMATION_WINDOW_SIZE": estimation_window_size, 
                            "FUTURE_WINDOW_SIZE": future_window_size, "CALIBRATION_WINDOW_SIZE": calibration_window_size, 
                            "CALIBRATION_LENGTH": calibration_length, "PRECALIBRATION_LENGTH": precalibration_length, 
                            "PARAM_FIND_M": param_find_M, "PARAM_READ_M": param_read_M, 
                            "QUIET": is_quiet, "MODES": modes, "MODEL": "mpi", "METHOD": "trf", "FTOL": ftol, "WEIGHTS": weights,
                            "INIT_POSE_MODE": init_pose_mode, "NOISE_2D_STD": noise_2d_std, "USE_SYMMETRY_TERM": use_symmetry_term, 
                            "USE_SINGLE_JOINT": use_single_joint, "SMOOTHNESS_MODE": smoothness_mode, "USE_TRAJECTORY_BASIS": use_trajectory_basis,
                            "NUMBER_OF_TRAJ_PARAM": num_of_trajectory_param, "NOISE_LIFT_STD": noise_lift_std, "NOISE_3D_INIT_STD": noise_3d_init_std,
                            "PROJECTION_METHOD" :projection_method, "WEIGHTS_FUTURE":weights_future, "USE_LIFT_TERM": use_lift_term, "USE_BONE_TERM": use_bone_term,
                            }

                active_parameters["TRAJECTORY"] = TRAJECTORY_LIST[0]

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