from helpers import * 
from PoseEstimationClient import PoseEstimationClient
from pose3d_optimizer import *
from Projection_Client import Projection_Client
from determine_positions import *
from PotentialStatesFetcher import PotentialStatesFetcher, PotentialState, Potential_Trajectory
from State import State, find_pose_and_frame_at_time
from file_manager import FileManager, get_bone_len_file_name
from drone_flight_client import DroneFlightClient
from mpi_dataset_client import MPI_Dataset_Client
from synth_dataset_client import Synth_Dataset_Client, get_synth_dataset_filenames
from math import radians, degrees
from rng_object import rng_object
from simulator_data_processor import get_client_gt_values, airsim_retrieve_gt, take_photo, get_simulator_responses, airsim_retrieve_poses_gt
import copy
from flight_loops import generate_new_goal_pos_random, generate_new_goal_pos_same_dir
from PIL import Image

import pprint
import os
import cv2 as cv


def run_simulation(kalman_arguments, parameters, energy_parameters, active_parameters):
    """
    Description: 
        Main function that runs simulation

    Inputs: 
        kalman_arguments: dict of kalman filter parameters that are now useless
        parameters: dict of general parameters
        energy_parameters: dict of pose estimation parameters
        active_parameters: dict of active mode parameters
    Returns:
        errors: dict of errors
    """
    # energy_parameters["MODES"]["mode_2d"] = "gt_with_noise"
    bone_len_file_name = get_bone_len_file_name(energy_parameters["MODES"])
    file_manager = FileManager(parameters, bone_len_file_name)   

    date_time_name = time.strftime("%Y-%m-%d-%H-%M")
    print("experiment began at:", date_time_name)

    length_of_simulation = parameters["LENGTH_OF_SIMULATION"]
    loop_mode = parameters["LOOP_MODE"]
    calibration_mode = parameters["CALIBRATION_MODE"]
    port =  parameters["PORT"]
    simulation_mode = parameters["SIMULATION_MODE"]
    experiment_ind = parameters["EXPERIMENT_NUMBER"]

    #set random seeds once and for all
    my_rng = rng_object(parameters["SEED"], file_manager.saved_vals_loc, active_parameters["DRONE_POS_JITTER_NOISE_STD"], energy_parameters["NOISE_3D_INIT_STD"])

    #connect to the AirSim simulator
    if simulation_mode == "use_airsim":
        airsim_client = airsim.MultirotorClient(length_of_simulation, port=port)
        airsim_client.confirmConnection()
        airsim_client.reset()
        if loop_mode == "flight_simulation" or loop_mode == "try_controller_control":
            airsim_client.enableApiControl(True)
            airsim_client.armDisarm(True)
            camera_offset_x = 45/100
        else:
            camera_offset_x = 0 
        airsim_client.initInitialDronePos()
        airsim_client.changeAnimation(ANIM_TO_UNREAL[file_manager.anim_num])
        if loop_mode == "flight_simulation" or loop_mode == "try_controller_control":
            airsim_client.takeoffAsync(timeout_sec = 15).join()
            airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(0, 0, 0))
    elif simulation_mode == "saved_simulation":
        camera_offset_x = 0 
        if file_manager.anim_num == "drone_flight":
            airsim_client = DroneFlightClient(length_of_simulation, file_manager.anim_num, file_manager.non_simulation_files)
        elif file_manager.anim_num == "mpi_inf_3dhp":
            airsim_client = MPI_Dataset_Client(length_of_simulation, file_manager.test_sets_loc, experiment_ind)
        else:
            airsim_client = Synth_Dataset_Client(length_of_simulation, file_manager.test_sets_loc, file_manager.anim_num, experiment_ind)
        file_manager.init_photo_loc_dir(airsim_client.image_main_dir)

        #file_manager.label_list = airsim_client.label_list
    #pause airsim until we set stuff up 
    airsim_client.simPause(True, loop_mode)

    pose_client = PoseEstimationClient(param=energy_parameters, general_param=parameters, 
                                        intrinsics=airsim_client.intrinsics)
    current_state = State(use_single_joint=pose_client.USE_SINGLE_JOINT, active_parameters=active_parameters,
                         model_settings=pose_client.model_settings(), anim_gt_array=file_manager.f_anim_gt_array, 
                         future_window_size=pose_client.FUTURE_WINDOW_SIZE,  initial_drone_pos=airsim_client.DRONE_INITIAL_POS,
                         camera_offset_x = camera_offset_x)
    potential_states_fetcher = PotentialStatesFetcher(airsim_client=airsim_client, pose_client=pose_client, 
                                active_parameters=active_parameters, loop_mode=loop_mode)

    current_state.init_anim_time(airsim_client.default_initial_anim_time, file_manager.anim_num)
    set_animation_to_frame(airsim_client, pose_client, current_state, airsim_client.default_initial_anim_time)
    if airsim_client.is_using_airsim and not (loop_mode == "flight_simulation" or loop_mode == "try_controller_control"):
        move_drone_to_front(airsim_client, pose_client, current_state.radius, current_state.C_cam_torch)

    airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
    time.sleep(0.5)
    if not calibration_mode:
        pose_client.read_bone_lengths_from_file(file_manager, current_state.bone_pos_gt)

    file_manager.save_initial_drone_pos(airsim_client)

    #shoulder_vector = initial_positions[R_SHOULDER_IND, :] - initial_positions[L_SHOULDER_IND, :] #find initial human orientation!
    #INITIAL_HUMAN_ORIENTATION = np.arctan2(-shoulder_vector[0], shoulder_vector[1]) #in unreal coordinates

    ################
    if loop_mode == "flight_simulation" or loop_mode == "teleport_simulation" or loop_mode == "toy_example":
        general_simulation_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, parameters, my_rng)
    elif loop_mode == "openpose":
        openpose_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, my_rng)
    #elif loop_mode == "teleport":
     #   teleport_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, loop_mode, parameters)
    elif loop_mode == "save_gt_poses":
        save_gt_poses_loop(current_state, pose_client, airsim_client, file_manager)
    elif loop_mode == "create_dataset":
        create_test_set(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, my_rng)
    elif loop_mode == "try_controller_control":
        try_controller_control_loop(current_state, pose_client, airsim_client, file_manager, potential_states_fetcher, loop_mode)
    elif loop_mode == "openpose_liftnet":
        openpose_liftnet(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, parameters, my_rng)
    ################

    #calculate errors
    airsim_client.simPause(False, loop_mode)
    average_errors = pose_client.average_errors
    ave_current_error, ave_middle_error, ave_pastmost_error, ave_overall_error = pose_client.average_errors[pose_client.CURRENT_POSE_INDEX],  pose_client.average_errors[pose_client.MIDDLE_POSE_INDEX], pose_client.average_errors[pose_client.PASTMOST_POSE_INDEX], pose_client.ave_overall_error
    plot_distance_values(current_state.distances_travelled, file_manager.plot_loc)

    if calibration_mode:
        file_manager.save_bone_lengths(pose_client.boneLengths)

    print('End it!')
    pose_client.reset(file_manager.plot_loc)
    file_manager.close_files()

    return ave_current_error, ave_middle_error, ave_pastmost_error, ave_overall_error 

def general_simulation_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, parameters, my_rng):
    """
    Description:
        General simulation loop
    Inputs: 
        current_state: an object of type State
        pose_client: an object of type PoseEstimationClient
        airsim_client: an object of type VehicleClient or DroneFlightClient
        potential_states_fetcher: an object of type PotentialStatesFetcher
        file_manager: object of type FileManager
    """
    num_of_noise_trials = parameters["NUM_OF_NOISE_TRIALS"]
    if pose_client.modes["mode_2d"] == "openpose":
        num_of_noise_trials = 1
    find_best_traj = parameters["FIND_BEST_TRAJ"]

    take_photo(airsim_client, pose_client, current_state, file_manager)
    initialize_empty_frames(airsim_client.linecount, pose_client, current_state, file_manager, my_rng)
    set_animation_to_frame(airsim_client, pose_client, current_state, airsim_client.default_initial_anim_time)

    potential_states_fetcher.reset(pose_client, airsim_client, current_state)
    potential_states_fetcher.get_potential_positions(airsim_client.linecount)
    while airsim_client.linecount < airsim_client.length_of_simulation:   
        #### if we have the best traj finder 
        if find_best_traj and not pose_client.is_calibrating_energy:
            start_best_sim = time.time()   
            my_rng.freeze_all_rng_states()
            current_anim_time = airsim_client.getAnimationTime()
            state_copy = current_state.deepcopy_state()
            update_animation(airsim_client, pose_client.deepcopy_PEC(0),  state_copy)
            future_anim_time = state_copy.anim_time
            #find goal location
            for trajectory_ind in range(0, len(potential_states_fetcher.potential_trajectory_list)):
                #print("* trajectory_ind", trajectory_ind)
                my_rng.reload_all_rng_states() 
                goal_traj = potential_states_fetcher.choose_trajectory_using_trajind(trajectory_ind)
                #update_animation(airsim_client, pose_client_copy, state_copy)
                #my_rng_oracle = rng_object(101, file_manager.saved_vals_loc, potential_states_fetcher.DRONE_POS_JITTER_NOISE_STD, pose_client.NOISE_3D_INIT_STD)
                my_rng_oracle = my_rng
                for trial_ind in range(num_of_noise_trials):
                    #print("** trial ind", trial_ind)
                    potential_states_fetcher.restart_trajectory()
                    pose_client_copy = pose_client.deepcopy_PEC(trial_ind)
                    state_copy = current_state.deepcopy_state()
                    state_copy.update_anim_time(future_anim_time)
                    #set_animation_to_frame(airsim_client, pose_client, state_copy, current_anim_time)
                    #print("*** future_ind", future_ind)
                    goal_trajectory = potential_states_fetcher.goal_trajectory
                    #goal_state = potential_states_fetcher.move_along_trajectory()
                    #update_animation(airsim_client, pose_client_copy, state_copy)
                    set_position(goal_trajectory, airsim_client, state_copy, pose_client_copy, potential_states_fetcher, loop_mode=potential_states_fetcher.loop_mode)
                
                    take_photo(airsim_client, pose_client_copy, state_copy, file_manager)           
                    determine_positions(airsim_client.linecount, pose_client_copy, state_copy, file_manager, my_rng_oracle) 
                    #print("error", pose_client_copy.average_errors[pose_client_copy.MIDDLE_POSE_INDEX])
                    goal_traj.record_error_for_trial(pose_client_copy.FUTURE_WINDOW_SIZE-1, pose_client_copy.average_errors[pose_client_copy.MIDDLE_POSE_INDEX], pose_client_copy.overall_error)
                goal_traj.find_overall_error()

            file_manager.record_oracle_errors(airsim_client.linecount, potential_states_fetcher.potential_trajectory_list)
            potential_states_fetcher.restart_trajectory()
            my_rng.reload_all_rng_states()
            set_animation_to_frame(airsim_client, pose_client, current_state, current_anim_time)
            end_best_sim = time.time()
            print("Simulating errors for all locations took", end_best_sim-start_best_sim, "seconds")

            if potential_states_fetcher.trajectory == "oracle":
                min_error = np.inf
                for trajectory_ind in range(0, len(potential_states_fetcher.potential_trajectory_list)):
                    traj = potential_states_fetcher.potential_trajectory_list[trajectory_ind]
                    if traj.error_middle < min_error:
                        potential_states_fetcher.oracle_traj_ind=trajectory_ind
                        min_error = traj.error_middle
                    print("traj ind", trajectory_ind, "error", traj.error_middle)
                print("chosen traj", potential_states_fetcher.oracle_traj_ind, "with error", min_error)


        #find goal location
        if airsim_client.linecount != 0:
            start2=time.time()
            goal_trajectory = potential_states_fetcher.choose_trajectory(pose_client, airsim_client.linecount, airsim_client.online_linecount, file_manager, my_rng)
            #move there
            update_animation(airsim_client, pose_client, current_state)
            set_position(goal_trajectory, airsim_client, current_state, pose_client, potential_states_fetcher, loop_mode=potential_states_fetcher.loop_mode)
            end2= time.time()
            file_manager.record_chosen_trajectory(airsim_client.linecount, goal_trajectory.trajectory_index)
            #print("Choosing a trajectory took", end2-start2, "seconds")

        #update state values read from AirSim and take picture
        take_photo(airsim_client, pose_client, current_state, file_manager)        

        #find human pose 
        start3=time.time() 
        determine_positions(airsim_client.linecount, pose_client, current_state, file_manager, my_rng)
        end3=time.time()
        print("finding human pose took", end3-start3, "seconds")

        #plotting
        start4=time.time() 
        if not pose_client.quiet and airsim_client.linecount > 0:
            plot_drone_traj(pose_client, file_manager.plot_loc, airsim_client.linecount,  pose_client.animation)
        file_manager.write_error_values(pose_client.average_errors, airsim_client.linecount)
        file_manager.write_distance_values(current_state.distances_travelled, current_state.total_distance_travelled, airsim_client.linecount)
        end4=time.time()
        #print("plotting and recording error took ", end4-start4, "seconds")

    #    if not pose_client.is_calibrating_energy and not pose_client.quiet and file_manager.loop_mode == "toy_example":
    #       plot_potential_errors_and_uncertainties_matrix(airsim_client.linecount, potential_states_fetcher.potential_trajectory_list,
      #                                                      potential_states_fetcher.goal_trajectory, find_best_traj, file_manager.plot_loc)
        potential_states_fetcher.reset(pose_client, airsim_client, current_state)
        potential_states_fetcher.get_potential_positions(airsim_client.linecount)

        ### Debugging
        if not pose_client.quiet and pose_client.animation == "mpi_inf_3dhp":
            _, frame = find_pose_and_frame_at_time (current_state.anim_time+current_state.DELTA_T, current_state.anim_gt_array, current_state.num_of_joints)
            photo_locs = file_manager.get_photo_locs_for_all_viewpoints(frame, potential_states_fetcher.thrown_view_list)
            plot_thrown_views(potential_states_fetcher.thrown_view_list, file_manager.plot_loc, photo_locs, airsim_client.linecount, pose_client.bone_connections)

        airsim_client.increment_linecount(pose_client.is_calibrating_energy)


def save_gt_poses_loop(current_state, pose_client, airsim_client, file_manager):
    prev_pose_3d_gt = np.zeros([3, pose_client.num_of_joints])
    while airsim_client.linecount < 280:    
        #update state values read from AirSim and take picture
        airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
        take_photo(airsim_client, pose_client, current_state, file_manager)
        assert not np.allclose(prev_pose_3d_gt, current_state.bone_pos_gt)
        anim_time = airsim_client.getAnimationTime()
        file_manager.write_gt_pose_values(anim_time, current_state.bone_pos_gt)

        print(current_state.bone_pos_gt[:,0])
        prev_pose_3d_gt = current_state.bone_pos_gt.copy()
        start1=time.time()
        update_animation(airsim_client, pose_client, current_state, delta_t=0.1)
        print("updating animation took", time.time()-start1)

        airsim_client.increment_linecount(pose_client.is_calibrating_energy)

def openpose_liftnet(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, parameters, rng_object):
    # airsim_client.framecount = 940
    # airsim_client.internal_anim_time = 
    # airsim_client.chosen_cam_view = 8
    image_dir = "/cvlabsrc1/home/kicirogl/ActiveDrone/test_sets/mpi_inf_3dhp/camera_"+str(airsim_client.chosen_cam_view) +"/img_" +str(airsim_client.framecount) +".jpg"
    print(file_manager.photo_loc)
    take_photo(airsim_client, pose_client, current_state, file_manager)
    
    pose_client.modes["mode_2d"] = "gt_with_noise" #set this back
    noisy_gt, pose_2d_gt_cropped, heatmap_2d, cropped_image = determine_2d_positions(pose_client=pose_client, 
                                                        current_state=current_state, my_rng=rng_object, 
                                                        file_manager=file_manager, linecount=airsim_client.linecount)
    
    
    pose_client.modes["mode_2d"] = "openpose" #set this back
    openpose_res, pose_2d_gt_cropped, heatmap_2d, cropped_image = determine_2d_positions(pose_client=pose_client, 
                                                        current_state=current_state, my_rng=rng_object, 
                                                        file_manager=file_manager, linecount=airsim_client.linecount)
    
    #find relative 3d pose using liftnet
    pose_client.modes["mode_lift"] = "lift" #set this back
    pose3d_lift_directions = determine_relative_3d_pose(pose_client=pose_client, current_state=current_state, 
                                                        my_rng=rng_object, pose_2d=openpose_res, cropped_image=cropped_image, 
                                                        heatmap_2d=heatmap_2d, file_manager=file_manager)
    pose_client.modes["mode_lift"] = "gt"
    #find relative 3d pose using gt
    pose3d_lift_directions_gt = determine_relative_3d_pose(pose_client=pose_client, current_state=current_state, 
                                                        my_rng=rng_object, pose_2d=noisy_gt, cropped_image=cropped_image, 
                                                        heatmap_2d=heatmap_2d, file_manager=file_manager)
    pose_client.modes["mode_lift"] = "gt_with_noise"
    pose3d_lift_directions_gt_with_noise = determine_relative_3d_pose(pose_client=pose_client, current_state=current_state, 
                                                        my_rng=rng_object, pose_2d=noisy_gt, cropped_image=cropped_image, 
                                                        heatmap_2d=heatmap_2d, file_manager=file_manager)

    plot_single_human(pose3d_lift_directions_gt.numpy(), file_manager.plot_loc, "lift_gt",  pose_client.bone_connections)
    plot_single_human(pose3d_lift_directions_gt_with_noise.numpy(), file_manager.plot_loc, "lift_gt_with_noise",  pose_client.bone_connections)
    plot_single_human(pose3d_lift_directions.numpy(), file_manager.plot_loc, "liftnet",  pose_client.bone_connections)

    plot_2d_projection(noisy_gt, file_manager.plot_loc, 0, pose_client.bone_connections, custom_name="2d_noisy_gt")
    plot_2d_projection(pose_2d_gt_cropped, file_manager.plot_loc, 0, pose_client.bone_connections, custom_name="2d_gt")
    plot_2d_projection(openpose_res, file_manager.plot_loc, 0, pose_client.bone_connections, custom_name="openpose")
    

def openpose_liftnet_other(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, parameters):
    
    potential_states_fetcher.reset(pose_client, airsim_client, current_state)
    potential_states_fetcher.get_potential_positions(airsim_client.linecount)
    num_of_loc = len(potential_states_fetcher.potential_trajectory_list)
    len_of_sim = 80 
    validation_anims = ["28_19", "06_13", "13_06"]

    openpose_array = np.zeros([len(validation_anims)*len_of_sim*num_of_loc, 2, 15])
    gt_2d_array = np.zeros([len(validation_anims)*len_of_sim*num_of_loc, 2, 15])
    lift_array = np.zeros([len(validation_anims)*len_of_sim*num_of_loc, 3, 15])
    gt_lift_array = np.zeros([len(validation_anims)*len_of_sim*num_of_loc, 3, 15])

    for anim_num in range(len(validation_anims)):
        parameters["ANIMATION_NUM"] = validation_anims[anim_num]
        file_manager = FileManager(parameters, get_bone_len_file_name(pose_client.modes))   
        current_state.update_animation_gt_array(file_manager.f_anim_gt_array)

        airsim_client.changeAnimation(ANIM_TO_UNREAL[file_manager.anim_num])
        current_state.init_anim_time(airsim_client.default_initial_anim_time, file_manager.anim_num)
        set_animation_to_frame(airsim_client, pose_client, current_state, airsim_client.default_initial_anim_time)
        prev_drone_pose = torch.zeros([3,1])
        airsim_client.linecount = 0

        while airsim_client.linecount < len_of_sim and anim_num != 0:  
            airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
            pose_client.immediate_future_pose = current_state.bone_pos_gt
            pose_client.current_pose = current_state.bone_pos_gt
            potential_states_fetcher.reset(pose_client, airsim_client, current_state)
            potential_states_fetcher.get_potential_positions(airsim_client.linecount)
            anim_time = airsim_client.getAnimationTime()

            for trajectory_ind in range(num_of_loc):
                # print("  ** traj ind", trajectory_ind)
                index = anim_num*len_of_sim*num_of_loc+airsim_client.linecount*num_of_loc+trajectory_ind
                #move
                potential_states_fetcher.restart_trajectory()
                goal_traj = potential_states_fetcher.choose_trajectory_using_trajind(trajectory_ind)
                set_position(goal_traj, airsim_client, current_state, pose_client, potential_states_fetcher, loop_mode=potential_states_fetcher.loop_mode)

                #update state values read from AirSim and take picture
                take_photo(airsim_client, pose_client, current_state, file_manager)
                new_drone_pose = current_state.C_drone_gt.clone()
                assert not torch.allclose(prev_drone_pose, new_drone_pose)
                prev_drone_pose = new_drone_pose.clone()

                #find 2d pose (using openpose and gt)
                pose_2d_cropped, pose_2d_gt_cropped, heatmap_2d, cropped_image = determine_2d_positions(pose_client=pose_client, 
                                                        current_state=current_state, my_rng=None, 
                                                        file_manager=file_manager, linecount=airsim_client.linecount)
                
                openpose_array[index, :, :] = pose_2d_cropped.numpy().copy()
                gt_2d_array[index, :, :] = pose_2d_gt_cropped.numpy().copy()

                #find relative 3d pose using liftnet
                pose_client.modes["mode_lift"] = "lift" #set this back
                pose3d_lift_directions = determine_relative_3d_pose(pose_client=pose_client, current_state=current_state, 
                                                                    my_rng=None, pose_2d=pose_2d_cropped, cropped_image=cropped_image, 
                                                                    heatmap_2d=heatmap_2d, file_manager=file_manager)
                lift_array[index, :, :] = pose3d_lift_directions.numpy().copy()
                pose_client.modes["mode_lift"] = "gt"
                #find relative 3d pose using gt
                pose3d_lift_directions_gt = determine_relative_3d_pose(pose_client=pose_client, current_state=current_state, 
                                                                    my_rng=None, pose_2d=pose_2d_cropped, cropped_image=cropped_image, 
                                                                    heatmap_2d=heatmap_2d, file_manager=file_manager)
                gt_lift_array[index, :, :] = pose3d_lift_directions_gt.numpy().copy()


            file_manager.save_openpose_and_gt2d(openpose_array, gt_2d_array)
            file_manager.save_lift_and_gtlift(lift_array, gt_lift_array)

            update_animation(airsim_client, pose_client, current_state, delta_t=0.2)
            airsim_client.increment_linecount(pose_client.is_calibrating_energy)

def create_test_set(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, my_rng):
    date_time_name = time.strftime("%Y-%m-%d-%H-%M")
    print("experiment began at:", date_time_name)

    #synth_dataset_client = Synth_Dataset_Client(airsim_client.length_of_simulation, file_manager.test_sets_loc)
    files = get_synth_dataset_filenames(file_manager.test_sets_loc, file_manager.anim_num)

    gt_poses_file = open( files["f_groundtruth_poses"], "w")
    gt_poses_file.write("linecount\tanim_time\n")
    camera_pos_file = open(files["f_camera_pos"], "w")

    file_manager.init_photo_loc_dir(files["image_main_dir"])
    for viewpoint in range(18):
        if not os.path.exists(files["image_main_dir"]+"/camera_"+str(viewpoint)):
            os.makedirs(files["image_main_dir"]+"/camera_"+str(viewpoint)) 

    file_manager.save_intrinsics(airsim_client.intrinsics, files["f_intrinsics"])

    airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
    pose_client.immediate_future_pose = current_state.bone_pos_gt
    pose_client.current_pose = current_state.bone_pos_gt
    potential_states_fetcher.reset(pose_client, airsim_client, current_state)
    potential_states_fetcher.get_potential_positions(airsim_client.linecount)
    anim_time = airsim_client.getAnimationTime()
    prev_drone_pose = torch.zeros([3,1])
    num_of_loc = len(potential_states_fetcher.potential_trajectory_list)
    while airsim_client.linecount < airsim_client.length_of_simulation:  
        file_manager.save_gt_values_dataset(airsim_client.linecount, anim_time, current_state.bone_pos_gt, gt_poses_file)
        for trajectory_ind in range(num_of_loc):
            potential_states_fetcher.restart_trajectory()
            goal_traj = potential_states_fetcher.choose_trajectory_using_trajind(trajectory_ind)
            set_position(goal_traj, airsim_client, current_state, pose_client, potential_states_fetcher, loop_mode=potential_states_fetcher.loop_mode)        

            take_photo(airsim_client, pose_client, current_state, file_manager, trajectory_ind)
            new_drone_pose = current_state.C_drone_gt.clone()
            assert not torch.allclose(prev_drone_pose, new_drone_pose)
            prev_drone_pose = new_drone_pose.clone()

            file_manager.prepare_test_set_gt(current_state, airsim_client.linecount, trajectory_ind, camera_pos_file)

        airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
        pose_client.immediate_future_pose = current_state.bone_pos_gt
        pose_client.current_pose = current_state.bone_pos_gt    
        potential_states_fetcher.reset(pose_client, airsim_client, current_state)
        potential_states_fetcher.get_potential_positions(airsim_client.linecount)
        airsim_client.increment_linecount(pose_client.is_calibrating_energy)
        update_animation(airsim_client, pose_client, current_state)
        anim_time = airsim_client.getAnimationTime()


def move_drone_to_front(airsim_client, pose_client, radius, C_cam_torch):
    _, response_poses = get_simulator_responses(airsim_client, pose_client.loop_mode)
    pose_3d_gt,_,_ = get_client_gt_values(airsim_client, pose_client, response_poses)
    assert pose_3d_gt.shape == (3, pose_client.num_of_joints)

    left_arm_ind = pose_client.joint_names.index('left_arm')
    right_arm_ind = pose_client.joint_names.index('right_arm')
    human_orientation_GT = find_human_pose_orientation(pose_3d_gt, left_arm_ind, right_arm_ind)

    print("Initial human orientation is:", human_orientation_GT)

    new_yaw = human_orientation_GT#+pi/4
    new_theta = 3*pi/2
    x = radius*cos(new_yaw)*sin(new_theta) + pose_3d_gt[0, pose_client.hip_index]
    y = radius*sin(new_yaw)*sin(new_theta) + pose_3d_gt[1, pose_client.hip_index]
    z = radius*cos(new_theta)+ pose_3d_gt[2, pose_client.hip_index]
    drone_pos = np.array([x, y, z])
    _, new_phi_go = find_current_polar_info(drone_pos, pose_3d_gt[:, pose_client.hip_index])
    goal_state = PotentialState(position=drone_pos.copy(), orientation=new_phi_go+pi, pitch=new_theta+pi/2, index=0, C_cam_torch=C_cam_torch)
    airsim_client.simSetVehiclePose(goal_state)
    airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(0, 0, 0))

def update_animation(airsim_client, pose_client, current_state, delta_t=None):
    if delta_t is None:
        delta_t = current_state.DELTA_T
    set_animation_to_frame(airsim_client, pose_client, current_state, current_state.anim_time+delta_t)

def set_animation_to_frame(airsim_client, pose_client, current_state, anim_frame):
    current_anim_time = airsim_client.getAnimationTime()
    if not pose_client.is_calibrating_energy and anim_frame != current_anim_time:
        prev_gt_pose = current_state.bone_pos_gt.copy()
        #airsim_client.simPause(False, pose_client.loop_mode)
        airsim_client.setAnimationTime(anim_frame)
        #time.sleep(5)
        #airsim_client.simContinueForTime(0.01)
        #airsim_client.simPause(True, pose_client.loop_mode)
        new_gt_pose = airsim_retrieve_poses_gt(airsim_client, pose_client)
        i = 0
        if airsim_client.is_using_airsim:
            while (np.allclose(prev_gt_pose, new_gt_pose) and i < 200):
                time.sleep(0.05)
                new_gt_pose = airsim_retrieve_poses_gt(airsim_client, pose_client)
                i += 1
                #try again
                if i%10 == 0:
                    # airsim_client.simPause(False, pose_client.loop_mode)
                    airsim_client.setAnimationTime(anim_frame)
                    # airsim_client.simPause(True, pose_client.loop_mode)
                fail_msg = "waited until too long, i =200"
                assert i != 200, fail_msg

        anim_time = airsim_client.getAnimationTime()
        assert anim_time != 0
        current_state.update_anim_time(anim_time)
        #print(anim_time, "is anim_time")

def set_position(goal_trajectory, airsim_client, current_state, pose_client, potential_states_fetcher, loop_mode):
    if loop_mode == "teleport_simulation" or loop_mode == "toy_example" or loop_mode ==  "create_dataset" or loop_mode =="openpose_liftnet":
        goal_state = potential_states_fetcher.move_along_trajectory()
        airsim_client.simSetVehiclePose(goal_state)
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(goal_state.pitch, 0, 0))
        current_state.set_cam_pitch(goal_state.pitch)

    elif loop_mode == "flight_simulation":
        desired_dir = goal_trajectory.get_movement_direction(potential_states_fetcher.motion_predictor.future_ind)
        potential_states_fetcher.motion_predictor.update_last_direction(desired_dir)
        goal_state = potential_states_fetcher.move_along_trajectory()
        desired_yaw_deg, _ =  goal_state.get_goal_yaw_pitch(current_state.drone_orientation_gt)
        cam_pitch = current_state.get_required_pitch()

        # go_dist = np.linalg.norm(desired_pos[:, np.newaxis]-current_state.C_drone_gt.numpy()) 

        # print("go_dist is", go_dist)
        # if airsim_client.linecount < pose_client.CALIBRATION_LENGTH:
        # drone_speed = go_dist
        # if drone_speed > current_state.TOP_SPEED:
        #     drone_speed = current_state.TOP_SPEED
        # airsim_client.moveToPositionAsync(desired_pos[0], desired_pos[1], desired_pos[2], 
        #                                   drone_speed, current_state.DELTA_T, airsim.DrivetrainType.MaxDegreeOfFreedom, 
        #                                   airsim.YawMode(is_rate=False, yaw_or_rate=desired_yaw_deg), lookahead=-1, adaptive_lookahead=0).join()
        # airsim_client.moveToPositionAsync(desired_pos[0], desired_pos[1], desired_pos[2], 
        #                                   drone_speed, current_state.DELTA_T, airsim.DrivetrainType.MaxDegreeOfFreedom, 
        #                                   airsim.YawMode(is_rate=False, yaw_or_rate=desired_yaw_deg), lookahead=-1, adaptive_lookahead=0).join()
        if potential_states_fetcher.movement_mode == "position":
            drone_speed = current_state.TOP_SPEED
            airsim_client.moveToPositionAsync(desired_dir[0], desired_dir[1], desired_dir[2], 
                                            drone_speed, 1, airsim.DrivetrainType.MaxDegreeOfFreedom, 
                                            airsim.YawMode(is_rate=False, yaw_or_rate=desired_yaw_deg), 
                                            lookahead=-1, adaptive_lookahead=0)
    
        if potential_states_fetcher.movement_mode == "velocity":
            # total_time_passed = 0
            # while(total_time_passed < current_state.DELTA_T-0.05):
                # time_remaining = current_state.DELTA_T - total_time_passed
            airsim_client.moveByVelocityAsync(desired_dir[0], desired_dir[1], desired_dir[2],  
                                            1, airsim.DrivetrainType.MaxDegreeOfFreedom, 
                                            airsim.YawMode(is_rate=False, yaw_or_rate=desired_yaw_deg))
                # time_passed = end_move-start_move
                # total_time_passed += time_passed

        time.sleep(0.01)
        airsim_client.simContinueForTime(current_state.DELTA_T)
        while(not airsim_client.simIsPause()):
            time.sleep(0.01)
        # airsim_client.simPause(True, loop_mode)
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(cam_pitch, 0, 0))
        current_state.set_cam_pitch(cam_pitch)

    elif loop_mode == "try_controller_control":
        desired_dir = goal_trajectory.get_movement_direction(potential_states_fetcher.motion_predictor.future_ind)
        potential_states_fetcher.motion_predictor.future_ind -= 1
        potential_states_fetcher.motion_predictor.update_last_direction(desired_dir)

        cam_pitch = current_state.get_required_pitch()


        # go_dist = np.linalg.norm(desired_pos[:, np.newaxis]-current_state.C_drone_gt.numpy()) 

        # drone_speed = go_dist
        # if drone_speed > current_state.TOP_SPEED:
        #     drone_speed = current_state.TOP_SPEED
        # print("desired pos is", go_dist)

        start_move = time.time()
        if potential_states_fetcher.movement_mode == "position":
            drone_speed = current_state.TOP_SPEED
            airsim_client.moveToPositionAsync(desired_pos[0], desired_pos[1], desired_pos[2], 
                                            drone_speed, 1, airsim.DrivetrainType.MaxDegreeOfFreedom, 
                                            airsim.YawMode(is_rate=False, yaw_or_rate=0), 
                                            lookahead=-1, adaptive_lookahead=0)
    
        if potential_states_fetcher.movement_mode == "velocity":
            airsim_client.moveByVelocityAsync(desired_dir[0], desired_dir[1], desired_dir[2],  
                                            1, airsim.DrivetrainType.MaxDegreeOfFreedom, 
                                            airsim.YawMode(is_rate=False, yaw_or_rate=0))

        time.sleep(0.01)
        airsim_client.simContinueForTime(current_state.DELTA_T)
        while(not airsim_client.simIsPause()):
            time.sleep(0.01)
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(cam_pitch, 0, 0))
        current_state.set_cam_pitch(cam_pitch)

def try_controller_control_loop(current_state, pose_client, airsim_client, file_manager, potential_states_fetcher, loop_mode):
    
    airsim_client.simPause(False, loop_mode)
    airsim_client.moveToZAsync(z=-40, velocity=5, timeout_sec=10).join()
    airsim_client.simPause(True, loop_mode)
    airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)

    FUTURE_WINDOW_SIZE = 3
    pose_client.is_calibrating_energy = True
    motion_predictor = potential_states_fetcher.motion_predictor
    errors = []

    prev_pos = np.zeros((0,3))
    x_curr = np.zeros((0,3))
    v_curr = np.zeros((0,3))
    v_final_arr = np.zeros((0,3))
    directions_arr = np.zeros((0,3))
    x_actual = np.zeros((0,3))
    delta_ts = np.zeros((0))

    for ind in range(500):        
        print("ind", ind)
        airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)

        current_drone_pos = np.squeeze(current_state.C_drone_gt.numpy())
        current_drone_vel = current_state.current_drone_vel

        if ind == 0:
            choose_sth_random = True
        else:
            choose_sth_random = False

        x_goal_desired, chosen_dir = generate_new_goal_pos_random(current_drone_pos, motion_predictor.prev_direction, potential_states_fetcher.TOP_SPEED, choose_sth_random)
        #print("DISTANCES:", np.linalg.norm(current_drone_pos[np.newaxis]-directions, axis=1))
        new_directions = motion_predictor.determine_new_direction(x_goal_desired)
        
        goal_trajectory = Potential_Trajectory(0, FUTURE_WINDOW_SIZE, new_directions)
        potential_pos = motion_predictor.predict_potential_positions_func(x_goal_desired, current_drone_pos, current_drone_vel)
        actual_pos = np.zeros([FUTURE_WINDOW_SIZE,3])
        x_curr = np.concatenate((current_drone_pos[np.newaxis].repeat(3, axis=0), x_curr), axis=0)
        v_curr = np.concatenate((current_drone_vel[np.newaxis].repeat(3, axis=0), v_curr), axis=0)
        directions_arr = np.concatenate((new_directions, directions_arr), axis=0)
        delta_ts = np.concatenate((np.array([0.6, 0.4, 0.2]), delta_ts), axis=0)

        potential_states_fetcher.motion_predictor.future_ind=FUTURE_WINDOW_SIZE-1
        for i in range(FUTURE_WINDOW_SIZE):
            set_position(goal_trajectory, airsim_client, current_state, pose_client, potential_states_fetcher, loop_mode)
            airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
            actual_pos[FUTURE_WINDOW_SIZE-i-1, :] = np.squeeze(current_state.C_drone_gt.numpy())
            errors.append(np.linalg.norm(actual_pos[FUTURE_WINDOW_SIZE-i-1, :] - potential_pos[FUTURE_WINDOW_SIZE-i-1, :]))

        airsim_client.increment_linecount(pose_client.is_calibrating_energy)
        x_actual = np.concatenate((actual_pos, x_actual), axis=0)
        plot_flight_positions_and_error(file_manager.plot_loc, prev_pos, current_drone_pos, x_goal_desired, potential_pos, actual_pos, airsim_client.linecount, errors, chosen_dir)
        prev_pos = np.concatenate((prev_pos, actual_pos), axis=0)
    
        if ind % 100 == 0:
            file_manager.save_flight_curves(x_curr, v_curr, directions_arr, delta_ts, x_actual)