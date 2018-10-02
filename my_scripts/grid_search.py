from main import *

def run_simulation(weights, animation_num, kalman_arguments, energy_parameters):
    file_names, folder_names, f_notes_name, date_time_name = reset_all_folders([animation_num], base="grid_search")
    test_set = TEST_SETS[animation_num]
    parameters = {"USE_TRACKBAR": False, "USE_AIRSIM": False, "FILE_NAMES": file_names, "FOLDER_NAMES": folder_names, "ANIMATION_NUM": animation_num, "TEST_SET_NAME":test_set}
    normalized_weights = normalize_weights(weights)
    energy_parameters["WEIGHTS"] = normalized_weights
    fill_notes(f_notes_name, parameters, energy_parameters)   

    energy_parameters["PARAM_READ_M"] = False
    energy_parameters["PARAM_FIND_M"] = True
    errors = main(kalman_arguments, parameters, energy_parameters)
    move_M(folder_names[animation_num]["estimates"])
    return errors, date_time_name   

def run_simulation_2(weights, animation_num, kalman_arguments, energy_parameters):
    file_names, folder_names, f_notes_name, date_time_name = reset_all_folders([animation_num], base="grid_search")
    test_set = TEST_SETS[animation_num]
    parameters = {"USE_TRACKBAR": False, "USE_AIRSIM": False, "FILE_NAMES": file_names, "FOLDER_NAMES": folder_names, "ANIMATION_NUM": animation_num, "TEST_SET_NAME":test_set}
    normalized_weights = normalize_weights(weights)
    energy_parameters["WEIGHTS"] = normalized_weights
    fill_notes(f_notes_name, parameters, energy_parameters)   

    energy_parameters["PARAM_READ_M"] = False
    energy_parameters["PARAM_FIND_M"] = True
    main(kalman_arguments, parameters, energy_parameters)

    energy_parameters["PARAM_READ_M"] = True
    energy_parameters["PARAM_FIND_M"] = False
    errors = main(kalman_arguments, parameters, energy_parameters)
    move_M(folder_names[animation_num]["estimates"])
    return errors, date_time_name   

def weight_search(with_M = True): 
    if (with_M):
        sim_func = run_simulation_2
    else:
        sim_func = run_simulation

    kalman_arguments = {"KALMAN_PROCESS_NOISE_AMOUNT" : 5.17947467923e-10, "KALMAN_MEASUREMENT_NOISE_AMOUNT_XY" : 1.38949549437e-08}
    kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_Z"] = 517.947467923 * kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_XY"]
    param_read_M = True
    param_find_M = False
    is_quiet = True
    calculate_hess = False
    flight_window_size = 6
    calibration_length = 35
    #mode_3d: 0- gt, 1- naiveback, 2- energy pytorch, 3-energy scipy
    #mode_2d: 0- gt, 1- openpose
    #mode_lift: 0- gt, 1- lift
    modes = {"mode_3d":3, "mode_2d":1, "mode_lift":1} 

    animation_num = "02_01"

    weight_list = np.logspace(-2, 2, 4)
    if os.path.exists("grid_search"):
        shutil.rmtree("grid_search")

    ave_errors_pos = np.zeros([len(weight_list),len(weight_list), len(weight_list)])
    folder_name_grid = []          

    energy_parameters = {"CALCULATE_HESSIAN":calculate_hess, "FLIGHT_WINDOW_SIZE": flight_window_size, "CALIBRATION_LENGTH": calibration_length, "PARAM_FIND_M": param_find_M, "PARAM_READ_M": param_read_M, "QUIET": is_quiet, "MODES": modes, "MODEL": "mpi", "METHOD": "trf", "FTOL": 1e-3}

    for smooth_ind, smooth_weights in enumerate(weight_list):
        for bone_ind, bone_weights in enumerate(weight_list):
            for lift_ind, lift_weights in enumerate(weight_list):
                weights_ = {'proj': 0.1, 'smooth': float(smooth_weights), 'bone': float(bone_weights), 'lift': float(lift_weights)}#'smoothpose': 0.01,}
                errors, date_time = sim_func(weights_, animation_num, kalman_arguments, energy_parameters)
                folder_name_grid.append(date_time)
                ave_errors_pos[smooth_ind, bone_ind, lift_ind] = errors["ave_3d_err"]
                print(errors)
            simple_plot2(weight_list, ave_errors_pos[smooth_ind, bone_ind, :], "grid_search", "overall_err"+str(lift_ind), plot_title="For " + str(smooth_weights) +" " + str(bone_weights), x_label="Lift weight", y_label="Error")

    overall_errors = ave_errors_pos
    min_err_ind = np.argmin(overall_errors)
    print("folder name:", folder_name_grid[min_err_ind])
    ind = np.unravel_index(min_err_ind, overall_errors.shape)
    print("min error", np.amin(overall_errors))
    good_weights = {'proj': 0.1, 'smooth': weight_list[ind[0]], 'bone': weight_list[ind[1]], 'lift':weight_list[ind[2]]}
    normalized_good_weights = normalize_weights(good_weights)
    print("weights", normalized_good_weights)

def oneaxis_weight_search(): 
    kalman_arguments = {"KALMAN_PROCESS_NOISE_AMOUNT" : 5.17947467923e-10, "KALMAN_MEASUREMENT_NOISE_AMOUNT_XY" : 1.38949549437e-08}
    kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_Z"] = 517.947467923 * kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_XY"]
    param_read_M = True
    param_find_M = False
    is_quiet = True
    calculate_hess = False
    flight_window_size = 6
    calibration_length = 35
    #mode_3d: 0- gt, 1- naiveback, 2- energy pytorch, 3-energy scipy
    #mode_2d: 0- gt, 1- openpose
    #mode_lift: 0- gt, 1- lift
    modes = {"mode_3d":3, "mode_2d":1, "mode_lift":1} 
    animation_num = "02_01"

    if os.path.exists("grid_search"):
        shutil.rmtree("grid_search")

    energy_parameters = {"CALCULATE_HESSIAN":calculate_hess, "FLIGHT_WINDOW_SIZE": flight_window_size, "CALIBRATION_LENGTH": calibration_length, "PARAM_FIND_M": param_find_M, "PARAM_READ_M": param_read_M, "QUIET": is_quiet, "MODES": modes, "MODEL": "mpi", "METHOD": "trf", "FTOL": 1e-3}

    centered = {'proj': 0.01, 'smooth': 100, 'bone': 4.6415888336127775, 'lift': 100}#'smoothpose': 0.01,}

    def generate_weight_list(mean):
        return [mean/10, mean/5, mean, mean*5, mean*10]

    for _ in range(0,2):
        weights_ = centered
        line_error_list = np.zeros([5,])
        proj_weight_list = generate_weight_list(centered["proj"])
        for proj_ind, proj_weights in enumerate(proj_weight_list):
            weights_["proj"] = proj_weights
            errors, date_time = run_simulation(weights_, animation_num, kalman_arguments, energy_parameters)
            line_error_list[proj_ind] = errors["ave_3d_err"]
        centered["proj"]=proj_weight_list[np.argmin(line_error_list)]

        smooth_weight_list = generate_weight_list(centered["smooth"])
        weights_ = centered
        for smooth_ind, smooth_weights in enumerate(smooth_weight_list):
            weights_["smooth"] = smooth_weights
            errors, date_time = run_simulation(weights_, animation_num, kalman_arguments, energy_parameters)
            line_error_list[smooth_ind] = errors["ave_3d_err"]
        centered["smooth"]=smooth_weight_list[np.argmin(line_error_list)]

        bone_weight_list = generate_weight_list(centered["bone"])
        weights_ = centered
        for bone_ind, bone_weights in enumerate(bone_weight_list):
            weights_["bone"] = bone_weights
            errors, date_time = run_simulation(weights_, animation_num, kalman_arguments, energy_parameters)
            line_error_list[bone_ind] = errors["ave_3d_err"]
        centered["bone"]=bone_weight_list[np.argmin(line_error_list)]

        lift_weight_list = generate_weight_list(centered["lift"])
        weights_ = centered
        for lift_ind, lift_weights in enumerate(lift_weight_list):
            weights_["lift"] = lift_weights
            errors, date_time = run_simulation(weights_, animation_num, kalman_arguments, energy_parameters)
            line_error_list[lift_ind] = errors["ave_3d_err"]
        centered["lift"]=lift_weight_list[np.argmin(line_error_list)]       

    print(centered)
    print(np.min(line_error_list))


if __name__ == "__main__":
    weight_search(with_M= False)

