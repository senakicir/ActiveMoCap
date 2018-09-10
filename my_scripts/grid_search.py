from main import *

def grid_search():
    #USE_TRACKBAR, USE_GROUNDTRUTH, USE_AIRSIM, PLOT_EVERYTHING, SAVE_VALUES
    animations = [0]
    file_names, folder_names = reset_all_folders(animations)
    parameters = {"USE_TRACKBAR": False, "USE_GROUNDTRUTH": 1, "USE_AIRSIM": False, "ANIMATION_NUM": 0, "TEST_SET_NAME":"test_set_1", "FILE_NAMES": file_names, "FOLDER_NAMES": folder_names,  "LR_MU": [lr, mu]}
    energy_parameters = {"LR_MU": [0.2, 0.8], "ITER": 3000, "WEIGHTS": {"proj":1,"smooth":0.5, "bone":3}}

    process_noise_list = np.logspace(-15, -7, 15)
    xy_measurement_noise_list = np.logspace(-15, -5, 15)
    z_xy_ratio_list = np.logspace(1, 5, 15)
    ave_errors_pos = np.zeros([len(process_noise_list),len(xy_measurement_noise_list),len(z_xy_ratio_list)])
    ave_errors_vel = np.zeros([len(process_noise_list),len(xy_measurement_noise_list),len(z_xy_ratio_list)])
    for i, process_noise in enumerate(process_noise_list):
        for j, xy_measurement_noise in enumerate(xy_measurement_noise_list):
            for k, z_xy_ratio in enumerate(z_xy_ratio_list):
                kalman_arguments = {"KALMAN_PROCESS_NOISE_AMOUNT" : process_noise, "KALMAN_MEASUREMENT_NOISE_AMOUNT_XY" : xy_measurement_noise}
                kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_Z"] = z_xy_ratio * kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_XY"]
                errors = main(kalman_arguments, parameters)
                ave_errors_pos[i,j,k], ave_errors_vel[i,j,k]= errors["error_ave_pos"], errors["error_ave_vel"]

    overall_errors = 0.5*ave_errors_pos + 0.5*ave_errors_vel
    ind = np.unravel_index(np.argmin(overall_errors), overall_errors.shape)
    print(np.amin(overall_errors))
    print(process_noise_list[ind[0]], xy_measurement_noise_list[ind[1]], z_xy_ratio_list[ind[2]])

def learning_rate_search(): 
    #USE_TRACKBAR, USE_GROUNDTRUTH, USE_AIRSIM, PLOT_EVERYTHING, SAVE_VALUES
    animations = [0]

    kalman_arguments = {"KALMAN_PROCESS_NOISE_AMOUNT" : 5.17947467923e-10, "KALMAN_MEASUREMENT_NOISE_AMOUNT_XY" : 1.38949549437e-08}
    kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_Z"] = 517.947467923 * kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_XY"]

    lr_list = np.linspace(0.08, 1, 5)
    mu_list = np.linspace(0.5, 0.99, 5)
    ave_errors_pos = np.zeros([len(lr_list),len(mu_list)])
    ave_errors_vel = np.zeros([len(lr_list),len(mu_list)])

    for lr_ind, lr in enumerate(lr_list):
        for mu_ind, mu in enumerate(mu_list):
            foldername_param = "lr" + str(lr) + "_mu" + str(mu)
            file_names, folder_names = my_helpers.reset_all_folders(animations, foldername_param)
            parameters = {"USE_TRACKBAR": False, "USE_GROUNDTRUTH": 3, "USE_AIRSIM": False, "ANIMATION_NUM": 0, "TEST_SET_NAME":"test_set_1", "FILE_NAMES": file_names, "FOLDER_NAMES": folder_names}
            energy_parameters = {"LR_MU": [lr, mu], "ITER": 3000, "WEIGHTS": {"proj":1,"smooth":0.5, "bone":3}}
            errors = main(kalman_arguments, parameters)
            ave_errors_pos[lr_ind, mu_ind], ave_errors_vel[lr_ind, mu_ind]= errors["error_ave_pos"], errors["error_ave_vel"]
            print(errors)

def weight_search(): 
    kalman_arguments = {"KALMAN_PROCESS_NOISE_AMOUNT" : 5.17947467923e-10, "KALMAN_MEASUREMENT_NOISE_AMOUNT_XY" : 1.38949549437e-08}
    kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_Z"] = 517.947467923 * kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_XY"]
    use_airsim = False
    #mode_3d: 0- gt, 1- naiveback, 2- energy pytorch, 3-energy scipy
    #mode_2d: 0- gt, 1- openpose
    #mode_lift: 0- gt, 1- lift
    modes = {"mode_3d":3, "mode_2d":1, "mode_lift":1} 
   
    use_trackbar = False

    animation_num = "02_01"
    test_set = TEST_SETS[animation_num]

    weight_list = np.logspace(-2, 2, 4)
    if os.path.exists("grid_search"):
        shutil.rmtree("grid_search")

    ave_errors_pos = np.zeros([len(weight_list),len(weight_list), len(weight_list)])
    ave_errors_vel = np.zeros([len(weight_list),len(weight_list), len(weight_list)])

    for smooth_ind, smooth_weights in enumerate(weight_list):
        for bone_ind, bone_weights in enumerate(weight_list):
            for lift_ind, lift_weights in enumerate(weight_list):
                weights_ = {'proj': 0.1, 'smooth': float(smooth_weights), 'bone': float(bone_weights), 'lift': float(lift_weights)}#'smoothpose': 0.01,}
                weights = {}
                weights_sum = sum(weights_.values())
                for loss_key in LOSSES:
                    weights[loss_key] = weights_[loss_key]/weights_sum
                energy_parameters = {"METHOD": "trf", "FTOL": 1e-3, "WEIGHTS": weights}

                file_names, folder_names, f_notes_name = reset_all_folders([animation_num], base="grid_search")
                parameters = {"QUIET": True, "ANIMATION_NUM": animation_num, "TEST_SET_NAME": test_set, "USE_TRACKBAR": use_trackbar, "MODES": modes, "USE_AIRSIM": use_airsim, "FILE_NAMES": file_names, "FOLDER_NAMES": folder_names, "MODEL": "mpi"}
                fill_notes(f_notes_name, parameters, energy_parameters)   

                errors = main(kalman_arguments, parameters, energy_parameters)
                ave_errors_pos[smooth_ind, bone_ind, lift_ind] = errors["error_ave_pos"], errors["error_ave_vel"]
                print(errors)
            simple_plot2(weight_list, ave_errors_pos[smooth_ind, bone_ind, :], "grid_search", "overall_err"+str(lift_ind), plot_title="For " + str(smooth_weights) +" " + str(bone_weights), x_label="Lift weight", y_label="Error")
                


    overall_errors = ave_errors_pos
    ind = np.unravel_index(np.argmin(overall_errors), overall_errors.shape)
    print(np.amin(overall_errors))
    print(weight_list[ind[0]], weight_list[ind[1]], weight_list[ind[2]])


if __name__ == "__main__":
    weight_search()

