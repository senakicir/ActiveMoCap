import numpy as np

def drone_flight_filenames(date_time_name="", mode=""):
    if date_time_name == "":
        date_time_name = '2019-05-23-20-35'
    if mode == "":
        mode = "ransac"

    main_dir = "/cvlabdata2/home/kicirogl/ActiveDrone/drone_flight/2019_02_isinsu/video_1_full_framerate_2_trial_2"
    general_output_folder = main_dir + "/drone_flight_dataset/" 
    gt_folder_dir = general_output_folder + date_time_name + "_" + mode + '/'
    input_image_dir = gt_folder_dir
    openpose_liftnet_image_dir = general_output_folder + "openpose_liftnet_images"

    drone_flight_filenames = {"input_image_dir": input_image_dir, 
            "openpose_liftnet_image_dir": openpose_liftnet_image_dir, 
            "gt_folder_dir": gt_folder_dir,
            "f_drone_pos": gt_folder_dir + "drone_pos_reoriented.txt", 
            "f_groundtruth": gt_folder_dir + "groundtruth_reoriented.txt", 
            "f_pose_2d": general_output_folder + "pose_2d.txt", 
            "f_pose_lift": general_output_folder + "pose_lift.txt",
            "f_intrinsics": general_output_folder + "intrinsics.txt"}

    return drone_flight_filenames

def get_airsim_testset(anim):
    main_dir = "/cvlabdata2/home/kicirogl/ActiveDrone/my_scripts/test_set/" +anim + "/0/"
    #main_dir = "/Users/kicirogl/Documents/temp_main/test_set/" + anim + "/0/"
    input_image_dir = main_dir+ "images/"

    drone_flight_filenames = {"input_image_dir": input_image_dir, 
            "f_drone_pos": main_dir + "drone_pos.txt", 
            "f_groundtruth": main_dir + "groundtruth.txt", 
            "f_pose_2d": main_dir + "openpose_results.txt", 
            "f_pose_lift": main_dir + "liftnet_results.txt",
            "f_intrinsics": None}
    return drone_flight_filenames

def get_filenames(test_set_name):
    if test_set_name == "drone_flight":
        return drone_flight_filenames()
    else: 
        return get_airsim_testset(test_set_name)


class FileManager(object):
    def __init__(self, parameters):
        self.anim_num = parameters["ANIMATION_NUM"]
        self.experiment_name = parameters["EXPERIMENT_NAME"]
        self.test_set_name = self.anim_num

        self.file_names = parameters["FILE_NAMES"]
        self.folder_names = parameters["FOLDER_NAMES"]
        self.simulation_mode = parameters["SIMULATION_MODE"]
        self.loop_mode = parameters["LOOP_MODE"]

        self.foldernames_anim = self.folder_names[self.experiment_name]
        self.filenames_anim = self.file_names[self.experiment_name]

        #open files
        self.f_drone_pos = open(self.filenames_anim["f_drone_pos"], 'w')
        self.f_groundtruth = open(self.filenames_anim["f_groundtruth"], 'w')
        self.f_reconstruction = open(self.filenames_anim["f_reconstruction"], 'w')
        self.f_error = open(self.filenames_anim["f_error"], 'w')
        self.f_uncertainty = open(self.filenames_anim["f_uncertainty"], 'w')

        self.f_average_error = open(self.filenames_anim["f_average_error"], 'w')
        self.f_average_error.write("linecount" + '\t' +  "overall error'\toverall ave error\tcurrent error\tcurrent ave error\tmiddle error\tmiddle ave error"  + '\n')

        self.f_correlations = open(self.filenames_anim["f_correlations"], 'w')
        self.f_correlations.write("linecount" + '\t' +  "current corr" + '\t' + "running ave corr" + '\t' +  "current cos sim" + '\t' + 'running ave cos sim' + '\n')

        self.f_initial_drone_pos = open(self.filenames_anim["f_initial_drone_pos"], 'w')
        self.f_openpose_results = open(self.filenames_anim["f_openpose_results"], 'w')
        self.f_liftnet_results = open(self.filenames_anim["f_liftnet_results"], 'w')

        if self.loop_mode ==  "openpose":
            self.f_openpose_error = open(self.filenames_anim["f_openpose_error"], 'w')
            self.f_openpose_arm_error = open(self.filenames_anim["f_openpose_arm_error"], 'w')
            self.f_openpose_leg_error = open(self.filenames_anim["f_openpose_leg_error"], 'w')

        self.plot_loc = self.foldernames_anim["superimposed_images"]
        self.photo_loc = ""
        self.take_photo_loc =  self.foldernames_anim["images"]
        self.estimate_folder_name = self.foldernames_anim["estimates"]

        self.openpose_err_str = ""
        self.openpose_err_arm_str = ""
        self.openpose_err_leg_str = ""

        if (self.simulation_mode == "saved_simulation"):
            self.non_simulation_filenames = get_filenames(self.test_set_name)  
            self.label_list = []
            self.non_simulation_files =  {"f_intrinsics":None,
                                        "f_pose_2d":open(self.non_simulation_filenames["f_pose_2d"], "r"),
                                        "f_pose_lift":open(self.non_simulation_filenames["f_pose_lift"], "r"),
                                        "f_groundtruth": open(self.non_simulation_filenames["f_groundtruth"], "r"),
                                        "f_drone_pos": open(self.non_simulation_filenames["f_drone_pos"], "r")}
            if self.non_simulation_filenames["f_intrinsics"] != None:
                self.non_simulation_files["f_intrinsics"] = open(self.non_simulation_filenames["f_intrinsics"], "r")
            

    def save_initial_drone_pos(self, airsim_client):
        initial_drone_pos_str = 'drone init pos\t' + str(airsim_client.DRONE_INITIAL_POS[0,]) + "\t" + str(airsim_client.DRONE_INITIAL_POS[1,]) + "\t" + str(airsim_client.DRONE_INITIAL_POS[2,])
        self.f_initial_drone_pos.write(initial_drone_pos_str + '\n')

    def update_photo_loc(self, linecount, viewpoint):
        if (self.simulation_mode == "use_airsim"):
            if viewpoint == "":
                self.photo_loc = self.foldernames_anim["images"] + '/img_' + str(linecount) + '.png'
            else:
                self.photo_loc = self.foldernames_anim["images"] + '/img_' + str(linecount) + "_viewpoint_" + str(viewpoint) + '.png'
        elif (self.simulation_mode == "saved_simulation"):
            if self.test_set_name == "drone_flight":
                linecount = 0
            self.photo_loc = self.non_simulation_filenames["input_image_dir"] + '/img_' + str(linecount) + "_viewpoint_" + str(viewpoint) + '.png'
        return self.photo_loc

    def get_photo_loc(self):
        return self.photo_loc

    def get_photo_locs(self):
        if self.loop_mode == "normal":
            return [self.photo_loc]*9

        photo_locs = []
        for viewpoint in range(16):
            photo_locs.append(self.non_simulation_filenames["input_image_dir"] + '/img_' + str(0) + "_viewpoint_" + str(viewpoint) + '.png')
        return photo_locs

    
    def close_files(self):
        self.f_drone_pos.close()
        self.f_groundtruth.close()
        self.f_reconstruction.close()
        self.f_error.close()
        self.f_uncertainty.close()
        if self.loop_mode ==  1:
            self.f_openpose_error.close()
            self.f_openpose_arm_error.close()
            self.f_openpose_leg_error.close()

    def write_openpose_prefix(self,THETA_LIST, PHI_LIST, num_of_joints):
        prefix_string = ""
        for new_theta_deg in THETA_LIST:
            for new_phi_deg in PHI_LIST:
                prefix_string += str(new_theta_deg) + ", " + str(new_phi_deg) + '\t'
        self.f_openpose_arm_error.write(prefix_string + "\n")
        self.f_openpose_leg_error.write(prefix_string + "\n")

        for _ in range(num_of_joints):
            prefix_string += '\t'
        self.f_openpose_error.write(prefix_string + "\n")

    def append_openpose_error(self, err, arm_err, leg_err):
        self.openpose_err_str += str(err) + "\t"
        self.openpose_err_arm_str += str(arm_err) + "\t"
        self.openpose_err_leg_str += str(leg_err) + "\t"

    def write_openpose_error(self, human_pose):
        for i in range(human_pose.shape[1]):
            self.openpose_err_str += str(human_pose[0,i]) + "\t" + str(human_pose[1,i]) + "\t" + str(human_pose[2,i]) + "\t"

        self.f_openpose_error.write(self.openpose_err_str + "\n")
        self.f_openpose_arm_error.write(self.openpose_err_arm_str + "\n")
        self.f_openpose_leg_error.write(self.openpose_err_leg_str + "\n")

        self.openpose_err_str = ""
        self.openpose_err_arm_str = ""
        self.openpose_err_leg_str = ""

    def prepare_test_set(self, current_state, openpose_res, liftnet_res, linecount, state_ind):
        f_drone_pos_str = ""
        flattened_transformation_matrix = np.reshape(current_state.drone_transformation_matrix.numpy(), (16, ))
        for i in range (16):
            f_drone_pos_str += str(float(flattened_transformation_matrix[i])) + '\t'

        self.f_drone_pos.write(str(linecount)+ '\t' + str(state_ind) + '\t' + f_drone_pos_str + '\n')
    
        openpose_str = ""
        for i in range(openpose_res.shape[1]):
            openpose_str += str(openpose_res[0, i].item()) + '\t' + str(openpose_res[1, i].item()) + '\t'
        self.f_openpose_results.write(str(linecount)+ '\t'+ str(state_ind) + '\t' + openpose_str + '\n')

        liftnet_str = ""
        for i in range(liftnet_res.shape[1]):
            liftnet_str += str(liftnet_res[0, i].item()) + '\t' + str(liftnet_res[1, i].item()) + '\t' + str(liftnet_res[2, i].item())
        self.f_liftnet_results.write(str(linecount)+ '\t'+ str(state_ind) + '\t' + liftnet_str + '\n')

    def prepare_test_set_gt(self, current_state, linecount, state_ind):
        f_drone_pos_str = ""
        flattened_transformation_matrix = np.reshape(current_state.drone_transformation_matrix.numpy(), (16, ))
        for i in range (16):
            f_drone_pos_str += str(float(flattened_transformation_matrix[i])) + '\t'

        self.f_drone_pos.write(str(linecount)+ '\t' + str(state_ind) + '\t' + f_drone_pos_str + '\n')

    def record_gt_pose(self, gt_3d_pose, linecount):
        f_groundtruth_str = ""
        for i in range(gt_3d_pose.shape[1]):
            f_groundtruth_str += str(gt_3d_pose[0, i].item()) + '\t' + str(gt_3d_pose[1, i].item()) + '\t' +  str(gt_3d_pose[2, i].item()) + '\t'
        self.f_groundtruth.write(str(linecount)+ '\t' + f_groundtruth_str + '\n')


    def write_reconstruction_values(self, pose_3d, pose_3d_gt, drone_pos, drone_orient, linecount, num_of_joints):
        f_reconstruction_str = ""
        f_groundtruth_str = ""
        for i in range(num_of_joints):
            f_reconstruction_str += str(pose_3d[0,i]) + '\t' + str(pose_3d[1,i]) + '\t' + str(pose_3d[2,i]) + '\t'
            f_groundtruth_str += str(pose_3d_gt[0, i]) + '\t' + str(pose_3d_gt[1, i]) + '\t' +  str(pose_3d_gt[2, i]) + '\t'

        self.f_reconstruction.write(str(linecount)+ '\t' + f_reconstruction_str + '\n')
        self.f_groundtruth.write(str(linecount)+ '\t' + f_groundtruth_str + '\n')

        f_drone_pos_str = ""
        for i in range(3):
            f_drone_pos_str += str(drone_pos[i].item()) + '\t'
        for i in range(3):
            for j in range(3):
                f_drone_pos_str += str(drone_orient[i][j].item()) + '\t'
        self.f_drone_pos.write(str(linecount)+ '\t' + f_drone_pos_str + '\n')

    def write_error_values(self, errors, linecount):
        f_error_str = ""
        for error in errors:
            f_error_str += str(error) + '\t'
        self.f_error.write(str(linecount)+ '\t' + f_error_str + '\n')

    def write_uncertainty_values(self, uncertainties, linecount):
        f_uncertainty_str = ""
        for uncertainty in uncertainties:
            f_uncertainty_str += str(uncertainty) + '\t'
        self.f_uncertainty.write(str(linecount)+ '\t' + f_uncertainty_str + '\n')

    def write_average_error_over_trials(self, linecount, potential_error_finder):
        overall_error, overall_ave_error = potential_error_finder.overall_error_list[-1], potential_error_finder.final_overall_error
        current_error, current_ave_error = potential_error_finder.current_error_list[-1], potential_error_finder.final_current_error
        middle_error, middle_ave_error = potential_error_finder.middle_error_list[-1], potential_error_finder.final_middle_error

        self.f_average_error.write(str(linecount) + '\t' + str(overall_error) + '\t' + str(overall_ave_error) + '\t' + str(current_error) + 
                                   '\t' + str(current_ave_error) + '\t' + str(middle_error) + '\t' + str(middle_ave_error) + '\n')

    def write_correlation_values(self, linecount, corr_array, cosine_array):
        self.f_correlations.write(str(linecount) + '\t' + str(corr_array[-1]) + '\t' + str(np.mean(corr_array)) + '\t' + str(cosine_array[-1]) + '\t' + str(np.mean(cosine_array)) + '\n')


    def write_hessians(self, hessians, linecount):
        pass
        #TO DO 
