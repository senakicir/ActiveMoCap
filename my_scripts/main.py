from run import run_simulation
from helpers import reset_all_folders, normalize_weights, fill_notes, TEST_SETS, append_error_notes
from math import radians
import numpy as np

if __name__ == "__main__":
    kalman_arguments = {"KALMAN_PROCESS_NOISE_AMOUNT" :1, "KALMAN_MEASUREMENT_NOISE_AMOUNT_XY" : 1e-3}
    kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_Z"] = 1000 * kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_XY"]
    use_trackbar = False
    
    #simulation mode = "use_airsim", "drone_flight_data"
    simulation_mode = "use_airsim"
    base_folder = "/Users/kicirogl/Documents/temp_main"
    #trajectory = 0-active, 1-constant_rotation, 2-random, 3-constant_angle, 4-wobbly_rotation, 5-updown, 6-leftright
    trajectory = "active"
    #loop_mode = 0-normal, 1-openpose, 2-teleport
    loop_mode = "teleport"
    #hessian_part: 0-future, 1-middle, 2-whole
    hessian_part = "whole"
    #uncertainty_calc_method: 0-sum_eig 1-add_diag 2-multip_eig 3-determinant 4-random
    uncertainty_calc_method = "sum_eig"

    minmax = True #True-min, False-max
    SEED_LIST = [41]#, 5, 2, 12, 1995]
    WOBBLE_FREQ_LIST = [0.5, 1, 2, 5, 20]
    UPDOWN_LIM_LIST = [[-3, -1]]
    LOOKAHEAD_LIST = [0.3]
    go_distance = 3
    upper_lim = -3
    lower_lim = -1 #-2.5

    param_read_M = False
    param_find_M = False
    is_quiet = False
    
    online_window_size = 6
    calibration_length = 0
    calibration_window_size = 200

    precalibration_length = 0
    init_pose_with_gt = True
    find_best_traj = True
    noise_2d_std = 3
    predefined_traj_len = 0

    use_symmetry_term = True
    use_single_joint = False
    #smoothness_mode: 0-velocity, 1-position, 2-all_connected, 3-only_velo_connected, 4-none
    smoothness_mode = "velocity"
    #use_bone_term = True
    #use_lift_term = False
    use_trajectory_basis = False
    num_of_trajectory_param = 5
    num_of_noise_trials = 30
    pose_noise_3d_std = 0.1

    parameters = {"USE_TRACKBAR": use_trackbar, "SIMULATION_MODE": simulation_mode, "LOOP_MODE":loop_mode, "FIND_BEST_TRAJ": find_best_traj, "PREDEFINED_TRAJ_LEN": predefined_traj_len, "NUM_OF_NOISE_TRIALS": num_of_noise_trials, "POSE_NOISE_3D_STD": pose_noise_3d_std}

    #mode_3d: 0- gt, 1- naiveback, 2- energy pytorch, 3-energy scipy
    #mode_2d: 0- gt, 1- gt_with_noise, 2- openpose
    #mode_lift: 0- gt, 1- lift
    modes = {"mode_3d":"scipy", "mode_2d":"gt_with_noise", "mode_lift":"gt"}
   
    #animations = {"02_01": len(SEED_LIST), "05_08": len(SEED_LIST)}
    animations = {"02_01": len(SEED_LIST)}

    theta_list = [270]#list(range(270, 180, -40)) #list(range(270, 180, -20))
    phi_list = list(range(0, 360, 20))
    position_grid = [[radians(theta),  radians(phi)] for theta in theta_list for phi in phi_list]
    #position_grid.append([radians(180), radians(0)])

    active_parameters = {"TRAJECTORY":trajectory, "HESSIAN_PART":hessian_part, "UNCERTAINTY_CALC_METHOD":uncertainty_calc_method, "MINMAX":minmax, "THETA_LIST":theta_list, "PHI_LIST":phi_list, "POSITION_GRID":position_grid, "GO_DISTANCE":go_distance, "UPPER_LIM":upper_lim, "LOWER_LIM":lower_lim}
    Z_POS_LIST = [-2.5]#, -4, -5, -6]
    
    lift_bone_term_grid = [[False, True],[True, True], [True, False], [False, False]]
    num_of_experiments = len(lift_bone_term_grid)
    for experiment_ind in range(num_of_experiments):

        file_names, folder_names, f_notes_name, _ = reset_all_folders(animations, base_folder)
        
        parameters["FILE_NAMES"] = file_names
        parameters["FOLDER_NAMES"] = folder_names
        
        weights =  {'proj': 0.0003332222592469177, 'smooth': 0.3332222592469177, 'bone': 0.3332222592469177, 'lift': 0.3332222592469177}

        energy_parameters = {"ONLINE_WINDOW_SIZE": online_window_size, "CALIBRATION_WINDOW_SIZE": calibration_window_size, "CALIBRATION_LENGTH": calibration_length, "PRECALIBRATION_LENGTH": precalibration_length, "PARAM_FIND_M": param_find_M, "PARAM_READ_M": param_read_M, "QUIET": is_quiet, "MODES": modes, "MODEL": "mpi", "METHOD": "trf", "FTOL": 1e-3, "WEIGHTS": weights, "INIT_POSE_WITH_GT": init_pose_with_gt, "NOISE_2D_STD": noise_2d_std, "USE_SYMMETRY_TERM": use_symmetry_term, "USE_SINGLE_JOINT": use_single_joint, "SMOOTHNESS_MODE": smoothness_mode, "USE_TRAJECTORY_BASIS": use_trajectory_basis, "NUMBER_OF_TRAJ_PARAM": num_of_trajectory_param}
        energy_parameters["USE_LIFT_TERM"] = lift_bone_term_grid[experiment_ind][0]
        energy_parameters["USE_BONE_TERM"] = lift_bone_term_grid[experiment_ind][1]

        active_parameters["UPDOWN_LIM"] = UPDOWN_LIM_LIST[0]
        active_parameters["WOBBLE_FREQ"] = WOBBLE_FREQ_LIST[0]
        active_parameters["Z_POS"] = Z_POS_LIST[0]
        active_parameters["LOOKAHEAD"] = LOOKAHEAD_LIST[0]

        fill_notes(f_notes_name, parameters, energy_parameters, active_parameters)   

        many_runs_last = []
        many_runs_middle = []
        #if (use_airsim):
        for animation in animations:
            for ind in range(animations[animation]):
                parameters["ANIMATION_NUM"]=  animation
                energy_parameters["SEED"] = SEED_LIST[ind]
                parameters["EXPERIMENT_NAME"] = str(animation) + "_" + str(ind)
                parameters["TEST_SET_NAME"]= ""
                errors = run_simulation(kalman_arguments, parameters, energy_parameters, active_parameters)
                many_runs_last.append(errors["ave_3d_err"] )
                many_runs_middle.append(errors["middle_3d_err"] )

        #else:
        #    ind = 0
        #    for animation in animations:
        #        parameters["ANIMATION_NUM"]=  animation
        #        parameters["EXPERIMENT_NAME"] = animation + "_" + str(ind)
        #        parameters["TEST_SET_NAME"]= TEST_SETS[animation]
        #        errors = run_simulation(kalman_arguments, parameters, energy_parameters, active_parameters)
        
        append_error_notes(f_notes_name, many_runs_last, many_runs_middle)
