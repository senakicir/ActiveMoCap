import numpy as np
import pandas as pd
import os

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

def get_bone_len_file_name(modes):
    return "/bone_len_mode2d_" + modes["mode_2d"]  + "__modelift_" + modes["mode_lift"] + ".txt"


class FileManager(object):
    def __init__(self, parameters, bone_len_file_name):
        self.anim_num = parameters["ANIMATION_NUM"]
        self.experiment_name = parameters["EXPERIMENT_NAME"]
        self.test_set_name = self.anim_num

        self.file_names = parameters["FILE_NAMES"]
        self.folder_names = parameters["FOLDER_NAMES"]
        self.simulation_mode = parameters["SIMULATION_MODE"]
        self.loop_mode = parameters["LOOP_MODE"]
        self.calibration_mode = parameters["CALIBRATION_MODE"]

        self.foldernames_anim = self.folder_names[self.experiment_name]
        self.filenames_anim = self.file_names[self.experiment_name]
        self.main_folder = self.file_names["main_folder"]
        self.saved_vals_loc = self.file_names["saved_vals_loc"]
        self.test_sets_loc = self.file_names["test_sets_loc"]


        #open files
        self.f_drone_pos = open(self.filenames_anim["f_drone_pos"], 'w')
        self.f_drone_pos.write("linecount" + '\t' + "posx"+'\t' + "posy"+'\t' + "posz"+'\t' + "orient1"+'\t' + "orient2"+'\t' + "orient3" + '\n')

        self.f_groundtruth = open(self.filenames_anim["f_groundtruth"], 'w')
        self.f_reconstruction = open(self.filenames_anim["f_reconstruction"], 'w')
        self.f_error = open(self.filenames_anim["f_error"], 'w')
        self.f_uncertainty = open(self.filenames_anim["f_uncertainty"], 'w')
        self.f_oracle_errors = open(self.filenames_anim["f_oracle_errors"], 'w')
        self.f_chosen_traj = open(self.filenames_anim["f_chosen_traj"], 'w')
        self.f_distance = open(self.filenames_anim["f_distance"], 'w')


        #empty
        self.f_average_error = open(self.filenames_anim["f_average_error"], 'w')

        self.f_correlations = open(self.filenames_anim["f_correlations"], 'w')
        self.f_correlations.write("linecount" + '\t' +  "current corr" + '\t' + "running ave corr" + '\t' +  "current cos sim" + '\t' + 'running ave cos sim' + '\n')

        self.f_initial_drone_pos = open(self.filenames_anim["f_initial_drone_pos"], 'w')
        self.f_openpose_results = open(self.filenames_anim["f_openpose_results"], 'w')
        self.f_liftnet_results = open(self.filenames_anim["f_liftnet_results"], 'w')
        self.f_projection_est = open(self.filenames_anim["f_projection_est"], 'w')
        self.f_trajectory_list = open(self.filenames_anim["f_trajectory_list"], 'w')
        self.f_yolo_res = open(self.filenames_anim["f_yolo_res"], 'w')
        self.f_openpose_error = open(self.filenames_anim["f_openpose_error"], 'w')
        #self.f_liftnet_error = open(self.filenames_anim["f_liftnet_error"], 'w')



        if self.loop_mode ==  "openpose":
            self.f_openpose_arm_error = open(self.filenames_anim["f_openpose_arm_error"], 'w')
            self.f_openpose_leg_error = open(self.filenames_anim["f_openpose_leg_error"], 'w')

        self.plot_loc = self.foldernames_anim["superimposed_images"]
        self.photo_loc = ""
        self.take_photo_loc =  self.foldernames_anim["images"]
        self.estimate_folder_name = self.foldernames_anim["estimates"]

        self.openpose_err_str = ""
        self.openpose_err_arm_str = ""
        self.openpose_err_leg_str = ""
        self.saved_anim_time = []

        saved_vals_loc_anim = self.saved_vals_loc + "/" + str(self.anim_num)
        if not os.path.exists(saved_vals_loc_anim):
            os.makedirs(saved_vals_loc_anim) 

        if self.loop_mode == "save_gt_poses":
            self.f_anim_gt = open(saved_vals_loc_anim + "/gt_poses.txt", "w")
            self.f_anim_gt_array = None
            self.f_anim_gt.write("time\tgt_poses\n")
            self.bone_lengths_dict = None
        else:
            #read into matrix
            self.f_anim_gt_array =  pd.read_csv(saved_vals_loc_anim + "/gt_poses.txt", sep='\t', header=None, skiprows=[0]).to_numpy()[:,:-1].astype('float')

            if self.calibration_mode:
                self.f_bone_len = open(saved_vals_loc_anim + bone_len_file_name, "w")
                self.f_bone_len.write("bone_lengths\n")
                self.bone_lengths_dict = None
            else:
                #read into matrix
                self.bone_lengths_dict = {}
                bone_len_file =  pd.read_csv(saved_vals_loc_anim + bone_len_file_name, sep='\t', header=None, skiprows=[0]).to_numpy()
                for i in range(2):
                    self.bone_lengths_dict[bone_len_file[i,0]] = bone_len_file[i,1:-1].astype('float')

    def update_anim_info(self, animation):
        self.anim_num = animation

    def save_initial_drone_pos(self, airsim_client):
        initial_drone_pos_str = 'drone init pos\t' + str(airsim_client.DRONE_INITIAL_POS[0,]) + "\t" + str(airsim_client.DRONE_INITIAL_POS[1,]) + "\t" + str(airsim_client.DRONE_INITIAL_POS[2,])
        self.f_initial_drone_pos.write(initial_drone_pos_str + '\n')

    def init_photo_loc_dir(self, photo_loc_dir):
        self.take_photo_loc = photo_loc_dir

    def update_photo_loc(self, linecount, viewpoint):
        if (self.simulation_mode == "use_airsim"):
            if viewpoint == "":
                self.photo_loc = self.take_photo_loc + '/img_' + str(linecount) + '.png'
            else:
                self.photo_loc =  self.take_photo_loc + '/camera_' + str(viewpoint) + "/img_" + str(linecount) + '.png'
        elif (self.simulation_mode == "saved_simulation"):
            if self.test_set_name == "drone_flight":
                #linecount = 0 not necessary anymore?
                self.photo_loc = self.take_photo_loc + '/img_' + str(linecount) + "_viewpoint_" + str(viewpoint) + '.png'
            elif self.test_set_name == "mpi_inf_3dhp":
                self.photo_loc =  self.take_photo_loc + '/camera_' + str(viewpoint) + "/img_" + str(linecount) + '.jpg'
            else:
                self.photo_loc =  self.take_photo_loc + '/camera_' + str(viewpoint) + "/img_" + str(linecount) + '.png'

        return self.photo_loc

    def write_distance_values(self, distances_travelled, total_distance_travelled, linecount):
        if len(distances_travelled)>0:
            f_distance_str = str(linecount)+'\t'+str(distances_travelled[-1])+'\t'+str(total_distance_travelled)+'\n'
            self.f_distance.write(f_distance_str)

    def get_photo_loc(self):
        return self.photo_loc

    def get_photo_locs_for_all_viewpoints(self, linecount, viewpoint_list):
        assert self.test_set_name == "mpi_inf_3dhp"
        photo_loc_list = []
        for viewpoint, _ in viewpoint_list:
            photo_loc_list.append(self.take_photo_loc + '/camera_' + str(viewpoint) + "/img_" + str(linecount) + '.jpg')
        return photo_loc_list

    def get_photo_locs(self):
        if self.loop_mode == "normal_simulation":
            return [self.photo_loc]*9

        photo_locs = []
        for viewpoint in range(16):
            photo_locs.append(self.non_simulation_filenames["input_image_dir"] + '/img_' + str(0) + "_viewpoint_" + str(viewpoint) + '.png')
        return photo_locs

    def save_pose_2d(self, pose_2d, linecount):
        openpose_str = ""
        for i in range(pose_2d.shape[1]):
            openpose_str += str(pose_2d[0, i].item()) + '\t' + str(pose_2d[1, i].item()) + '\t'
        self.f_openpose_results.write(str(linecount)+ '\t' + openpose_str + '\n')

    def save_lift(self, pose3d_lift_directions, linecount):
        if pose3d_lift_directions is not None:
            liftnet_str = ""
            for i in range(pose3d_lift_directions.shape[1]):
                liftnet_str += str(pose3d_lift_directions[0, i].item()) + '\t' + str(pose3d_lift_directions[1, i].item()) + '\t' + str(pose3d_lift_directions[2, i].item())
            self.f_liftnet_results.write(str(linecount)+ '\t'+ liftnet_str + '\n')

    
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
        self.f_trajectory_list.close()

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
        #for i in range(human_pose.shape[1]):
           # self.openpose_err_str += str(human_pose[0,i]) + "\t" + str(human_pose[1,i]) + "\t" + str(human_pose[2,i]) + "\t"

        self.f_openpose_error.write(self.openpose_err_str + "\n")
        self.f_openpose_arm_error.write(self.openpose_err_arm_str + "\n")
        self.f_openpose_leg_error.write(self.openpose_err_leg_str + "\n")

        self.openpose_err_str = ""
        self.openpose_err_arm_str = ""
        self.openpose_err_leg_str = ""

    def write_openpose_error2(self, error):
        self.f_openpose_error.write(str(error) + "\n")

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

    def prepare_test_set_gt(self, current_state, linecount, state_ind, drone_pos_file):
        f_drone_pos_str = ""
        flattened_transformation_matrix = np.reshape(current_state.drone_transformation_matrix.numpy(), (16, ))
        for i in range (16):
            f_drone_pos_str += str(float(flattened_transformation_matrix[i])) + '\t'
        drone_pos_file.write(str(linecount)+ '\t' + str(state_ind) + '\t' + f_drone_pos_str + '\n')

    def record_gt_pose(self, gt_3d_pose, linecount):
        f_groundtruth_str = ""
        for i in range(gt_3d_pose.shape[1]):
            f_groundtruth_str += str(gt_3d_pose[0, i].item()) + '\t' + str(gt_3d_pose[1, i].item()) + '\t' +  str(gt_3d_pose[2, i].item()) + '\t'
        self.f_groundtruth.write(str(linecount)+ '\t' + f_groundtruth_str + '\n')

    def record_yolo_results(self, linecount, yolo_confidence, bbox):
        f_yolo_str = str(linecount)+ '\t' + str(yolo_confidence) + '\t'
        if bbox is not None:
            for ele in bbox:
                f_yolo_str += str(ele) + '\t'
        self.f_yolo_res.write(f_yolo_str + '\n')

    def save_flight_curves(self, x_curr, v_curr, directions, delta_ts, x_actual):
        flight_curves_loc = self.saved_vals_loc + "/flight_curves"

        np.save(flight_curves_loc+"/x_curr", x_curr)
        np.save(flight_curves_loc+"/v_curr", v_curr)
        np.save(flight_curves_loc+"/directions", directions)
        np.save(flight_curves_loc+"/delta_ts", delta_ts)
        np.save(flight_curves_loc+"/x_actual", x_actual)

    def save_openpose_and_gt2d(self, openpose, pose_2d_gt):
        openpose_liftnet_loc =  self.saved_vals_loc + "/openpose_liftnet"
        np.save(openpose_liftnet_loc+"/openpose", openpose)
        np.save(openpose_liftnet_loc+"/pose_2d_gt", pose_2d_gt)

    def save_lift_and_gtlift(self, pose_lift, pose_lift_gt):
        openpose_liftnet_loc =  self.saved_vals_loc + "/openpose_liftnet"
        np.save(openpose_liftnet_loc+"/pose_lift", pose_lift)
        np.save(openpose_liftnet_loc+"/pose_lift_gt", pose_lift_gt)

    def read_flight_curves(self):
        flight_curves_dict = {}
        flight_curves_loc = self.saved_vals_loc + "/flight_curves"
        keys = ["x_curr", "v_curr", "directions", "delta_ts", "x_actual"]
        for key in keys:
            flight_curves_dict[key] = np.load(flight_curves_loc+ "/"+ key + ".npy") 
        return flight_curves_dict

    def save_intrinsics(self, intrinsics_dict, f_intrinsics_loc):
        f_intrinsics = open(f_intrinsics_loc, "w")
        f_intrinsics.write("focal_length\tpx\tpy\tsize_x\tsize_y\n")
        f_intrinsics.write( str(intrinsics_dict["f"] )
                            +"\t"+str(intrinsics_dict["px"])
                            +"\t"+str(intrinsics_dict["py"])
                            +"\t"+str(intrinsics_dict["size_x"])
                            +"\t"+str(intrinsics_dict["size_y"]))
        f_intrinsics.close()

    def save_gt_values_dataset(self, linecount, anim_time, pos_3d_gt, f_poses_gt):
        f_groundtruth_str = ""
        for i in range(pos_3d_gt.shape[1]):
            f_groundtruth_str += str(pos_3d_gt[0, i]) + '\t' + str(pos_3d_gt[1, i]) + '\t' +  str(pos_3d_gt[2, i]) + '\t'
        f_poses_gt.write(str(linecount) + '\t'+ str(anim_time)+ '\t' + f_groundtruth_str + '\n')


    def write_gt_pose_values(self, anim_time, pos_3d_gt):
        if anim_time not in self.saved_anim_time:
            self.saved_anim_time.append(anim_time)
            f_groundtruth_str = ""
            for i in range(pos_3d_gt.shape[1]):
                f_groundtruth_str += str(pos_3d_gt[0, i]) + '\t' + str(pos_3d_gt[1, i]) + '\t' +  str(pos_3d_gt[2, i]) + '\t'
            self.f_anim_gt.write(str(linecount) + '\t' + str(anim_time)+ '\t' + f_groundtruth_str + '\n')

    def record_drone_info(self, drone_pos, drone_orient, linecount):
        f_drone_pos_str = ""
        for i in range(3):
            f_drone_pos_str += str(drone_pos[i]) + '\t'
        for i in range(3):
            f_drone_pos_str += str(drone_orient[i]) + '\t'
        self.f_drone_pos.write(str(linecount)+ '\t' + f_drone_pos_str + '\n')

    def record_reconstruction_values(self, pose_3d, linecount):
        f_reconstruction_str = ""
        for i in range(pose_3d.shape[1]):
            f_reconstruction_str += str(pose_3d[0, i].item()) + '\t' + str(pose_3d[1, i].item()) + '\t' +  str(pose_3d[2, i].item()) + '\t'
        self.f_reconstruction.write(str(linecount)+ '\t' + f_reconstruction_str + '\n')

    def record_projection_est_values(self, pose_2d, linecount):
        f_projection_est_str = ""
        for i in range(pose_2d.shape[1]):
            f_projection_est_str += str(pose_2d[0, i].item()) + '\t' + str(pose_2d[1, i].item()) + '\t' 
        self.f_projection_est.write(str(linecount)+ '\t' + f_projection_est_str + '\n')

    def record_chosen_trajectory(self, linecount, index):
        self.f_chosen_traj.write(str(linecount) + '\t' + str(index) + '\n')

    def write_all_values(self, pose_3d, pose_3d_gt, drone_pos, drone_orient, linecount, num_of_joints):
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

    def write_gt_pose_values(self, anim_time, pos_3d_gt):
        if anim_time not in self.saved_anim_time:
            self.saved_anim_time.append(anim_time)
            f_groundtruth_str = ""
            for i in range(pos_3d_gt.shape[1]):
                f_groundtruth_str += str(pos_3d_gt[0, i]) + '\t' + str(pos_3d_gt[1, i]) + '\t' +  str(pos_3d_gt[2, i]) + '\t'
            self.f_anim_gt.write(str(anim_time)+ '\t' + f_groundtruth_str + '\n')



    def save_bone_lengths(self, bone_lengths_dict):
        for key, bone_lengths in bone_lengths_dict.items():
            f_bone_len_str = key+"\t"
            for i in range(bone_lengths.shape[0]):
                f_bone_len_str += str(bone_lengths[i].item()) + '\t'
            self.f_bone_len.write(f_bone_len_str + '\n')

   # def read_gt_pose_values(self, anim_time):
     #   return self.f_anim_gt_array[abs(self.f_anim_gt_array[:,0]-anim_time)<1e-4, 1:]

    def write_error_values(self, ave_errors, linecount):
        f_error_str=""
        for error_ind, error_value in ave_errors.items():
            f_error_str += str(error_value) + '\t'
        self.f_error.write(str(linecount)+ '\t' + f_error_str + '\n')

    def record_toy_example_results(self, linecount, potential_trajectory_list, uncertainty_dict, goal_trajectory):
        ## Traj i: a-b-c-d , uncertainty:x
        ## ...
        ## Chosen traj: x
        ## **
        traj_str = "Linecount: " + str(linecount) + "\n"
        for potential_trajectory in potential_trajectory_list:
            trajectory_index = potential_trajectory.trajectory_index
            traj_str += "\tTrajectory " + str(trajectory_index) + " : "
            for future_ind, state in potential_trajectory.states.items():
                state_index = state.index
                traj_str += str(state_index) + ", "
            traj_str += "uncertainty: " + str(potential_trajectory.uncertainty) + '\n'
        traj_str += "Goal trajectory is trajectory " + str(trajectory_index) + " :"
        for future_ind, state in goal_trajectory.states.items():
            state_index = state.index
            traj_str += str(state_index) + ", "
        traj_str += "lowest uncertainty: " + str(goal_trajectory.uncertainty) + '\n'
        traj_str += "*******\n"
        self.f_trajectory_list.write(traj_str)

    def record_oracle_errors(self, linecount, potential_trajectory_list):
        traj_str = str(linecount) + '\t'
        for potential_trajectory in potential_trajectory_list:
            traj_str += str(potential_trajectory.error_middle) + '\t'
        self.f_oracle_errors.write(traj_str+'\n')

    def write_uncertainty_values(self, uncertainties, linecount):
        f_uncertainty_str = ""
        for key, uncertainty in uncertainties.items():
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
