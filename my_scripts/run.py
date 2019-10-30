from helpers import * 
from PoseEstimationClient import PoseEstimationClient
from pose3d_optimizer import *
from project_bones import *
from determine_positions import *
from PotentialStatesFetcher import PotentialStatesFetcher, PotentialState
from State import State
from file_manager import FileManager, get_bone_len_file_name
from drone_flight_client import DroneFlightClient
from mpi_dataset_client import MPI_Dataset_Client
from rng_object import rng_object
from simulator_data_processor import get_client_gt_values, airsim_retrieve_gt, take_photo, get_simulator_responses, airsim_retrieve_poses_gt
import copy
from PIL import Image

import pprint
import os
import cv2 as cv

gt_hv = []
est_hv = []
photo_time = 0

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
    bone_len_file_name = get_bone_len_file_name(energy_parameters["MODES"])
    file_manager = FileManager(parameters, bone_len_file_name)   

    date_time_name = time.strftime("%Y-%m-%d-%H-%M")
    print("experiment began at:", date_time_name)

    simulation_mode = parameters["SIMULATION_MODE"]
    length_of_simulation = parameters["LENGTH_OF_SIMULATION"]
    loop_mode = parameters["LOOP_MODE"]
    calibration_mode = parameters["CALIBRATION_MODE"]
    port =  parameters["PORT"]

    #set random seeds once and for all
    my_rng = rng_object(parameters["SEED"])

    #connect to the AirSim simulator
    if simulation_mode == "use_airsim":
        airsim_client = airsim.MultirotorClient(length_of_simulation, port=port)
        airsim_client.confirmConnection()
        airsim_client.reset()
        if loop_mode == "normal_simulation":
            airsim_client.enableApiControl(True)
            airsim_client.armDisarm(True)
        airsim_client.initInitialDronePos()
        airsim_client.changeAnimation(ANIM_TO_UNREAL[file_manager.anim_num])
        if loop_mode == "normal_simulation":
            airsim_client.takeoffAsync(timeout_sec = 15)#.join()

    elif simulation_mode == "saved_simulation":
        if file_manager.anim_num == "drone_flight":
            airsim_client = DroneFlightClient(length_of_simulation, file_manager.anim_num, file_manager.non_simulation_files)
        elif file_manager.anim_num == "mpi_inf_3dhp":
            airsim_client = MPI_Dataset_Client(length_of_simulation, file_manager.test_sets_loc)
        file_manager.init_photo_loc_dir(airsim_client.image_main_dir)

        #file_manager.label_list = airsim_client.label_list
    #pause airsim until we set stuff up 
    airsim_client.simPause(True)

    pose_client = PoseEstimationClient(param=energy_parameters, general_param=parameters, 
                                        intrinsics=airsim_client.intrinsics)
    current_state = State(use_single_joint=pose_client.USE_SINGLE_JOINT, active_parameters=active_parameters,
                         model_settings=pose_client.model_settings(), anim_gt_array=file_manager.f_anim_gt_array, 
                         future_window_size=pose_client.FUTURE_WINDOW_SIZE)
    potential_states_fetcher = PotentialStatesFetcher(airsim_client=airsim_client, pose_client=pose_client, 
                                active_parameters=active_parameters, loop_mode=loop_mode)

    set_animation_to_frame(airsim_client, pose_client, current_state, airsim_client.default_initial_anim_time)
    current_state.init_anim_time(airsim_client.default_initial_anim_time)
    if airsim_client.is_using_airsim:
        move_drone_to_front(airsim_client, pose_client, current_state.radius)
    #airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(CAMERA_PITCH_OFFSET, 0, 0))

    if not calibration_mode:
        pose_client.read_bone_lengths_from_file(file_manager)
    file_manager.save_initial_drone_pos(airsim_client)

    #if pose_client.modes["mode_2d"] == "openpose":
    #    import openpose as openpose_module
    #if pose_client.modes["mode_lift"] == "lift":
    #    import liftnet as liftnet_module

    #shoulder_vector = initial_positions[R_SHOULDER_IND, :] - initial_positions[L_SHOULDER_IND, :] #find initial human orientation!
    #INITIAL_HUMAN_ORIENTATION = np.arctan2(-shoulder_vector[0], shoulder_vector[1]) #in unreal coordinates

    ################
    if loop_mode == "normal_simulation" or loop_mode == "teleport_simulation" or loop_mode == "toy_example":
        general_simulation_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, parameters, my_rng)
    elif loop_mode == "openpose":
        openpose_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, my_rng)
    #elif loop_mode == "teleport":
     #   teleport_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, loop_mode, parameters)
    elif loop_mode == "save_gt_poses":
        save_gt_poses_loop(current_state, pose_client, airsim_client, file_manager)
    elif loop_mode == "create_dataset":
        create_test_set(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, my_rng)
    ################

    #calculate errors
    airsim_client.simPause(True)
    average_errors = pose_client.average_errors
    ave_current_error, ave_middle_error, ave_pastmost_error, ave_overall_error = pose_client.average_errors[pose_client.CURRENT_POSE_INDEX],  pose_client.average_errors[pose_client.MIDDLE_POSE_INDEX], pose_client.average_errors[pose_client.PASTMOST_POSE_INDEX], pose_client.ave_overall_error

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
    find_best_traj = parameters["FIND_BEST_TRAJ"]

    airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
    time.sleep(0.5)

    take_photo(airsim_client, pose_client, current_state, file_manager)
    
    initialize_empty_frames(airsim_client.linecount, pose_client, current_state, file_manager, my_rng)

    airsim_client.simPause(True)
    potential_states_fetcher.reset(pose_client, airsim_client, current_state)
    potential_states_fetcher.get_potential_positions(pose_client.is_calibrating_energy)
    while airsim_client.linecount < airsim_client.length_of_simulation:   
        #### if we have the best traj finder 
        if find_best_traj and not pose_client.is_calibrating_energy:
            start_best_sim = time.time()   
            my_rng.freeze_all_rng_states()
            current_anim_time = airsim_client.getAnimationTime()

            #find goal location
            for trajectory_ind in range(0, len(potential_states_fetcher.potential_trajectory_list)):
                print("* trajectory_ind", trajectory_ind)
                my_rng.reload_all_rng_states() 
                goal_traj = potential_states_fetcher.choose_trajectory_using_trajind(trajectory_ind)
                for trial_ind in range(num_of_noise_trials):
                    #print("** trial ind", trial_ind)
                    potential_states_fetcher.restart_trajectory()
                    pose_client_copy = pose_client.deepcopy_PEC(trial_ind)
                    state_copy = current_state.deepcopy_state()
                    set_animation_to_frame(airsim_client, pose_client, state_copy, current_anim_time)
                    for future_ind in range(pose_client_copy.FUTURE_WINDOW_SIZE-1,0,-1):
                        #print("*** future_ind", future_ind)
                        goal_state = potential_states_fetcher.move_along_trajectory()
                        #set position also updates animation
                        set_position(goal_state, airsim_client, state_copy, pose_client_copy, loop_mode=potential_states_fetcher.loop_mode)
                    
                        take_photo(airsim_client, pose_client_copy, state_copy, file_manager)           
                        determine_positions(airsim_client.linecount, pose_client_copy, state_copy, file_manager, my_rng) 
                        goal_traj.record_error_for_trial(future_ind, pose_client_copy.errors[pose_client_copy.MIDDLE_POSE_INDEX], pose_client_copy.overall_error)
                goal_traj.find_overall_error()
            #file_manager.record_toy_example_results_error(linecount, self.potential_trajectory_list, self.goal_trajectory)
            potential_states_fetcher.restart_trajectory()
            my_rng.reload_all_rng_states()
            set_animation_to_frame(airsim_client, pose_client, current_state, current_anim_time)
            end_best_sim = time.time()
            print("Simulating errors for all locations took", end_best_sim-start_best_sim, "seconds")

        #find goal location
        if airsim_client.linecount != 0:
            start2=time.time()
            potential_states_fetcher.choose_trajectory(pose_client, airsim_client.linecount, airsim_client.online_linecount, file_manager, my_rng)
            goal_state = potential_states_fetcher.move_along_trajectory()
            #move there
            set_position(goal_state, airsim_client, current_state, pose_client, loop_mode=potential_states_fetcher.loop_mode)
            end2= time.time()
            print("Choosing a trajectory took", end2-start2, "seconds")

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
        end4=time.time()
        print("plotting and recording error took ", end4-start4, "seconds")

    #    if not pose_client.is_calibrating_energy and not pose_client.quiet and file_manager.loop_mode == "toy_example":
    #       plot_potential_errors_and_uncertainties_matrix(airsim_client.linecount, potential_states_fetcher.potential_trajectory_list,
      #                                                      potential_states_fetcher.goal_trajectory, find_best_traj, file_manager.plot_loc)
        airsim_client.increment_linecount(pose_client.is_calibrating_energy)

        potential_states_fetcher.reset(pose_client, airsim_client, current_state)
        potential_states_fetcher.get_potential_positions(pose_client.is_calibrating_energy)

def openpose_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, my_rng):
    #animations_to_test = ["64_06", "02_01", "05_08", "38_03"]
    file_manager.write_openpose_prefix(potential_states_fetcher.THETA_LIST, potential_states_fetcher.PHI_LIST, pose_client.num_of_joints)

    for animation in range(1,19):
        airsim_client.simPause(True)
        #airsim_client.changeAnimation(ANIM_TO_UNREAL[animation])
        airsim_client.changeAnimation(animation)
        print("Animation:", animation)
        time.sleep(1)
        for _ in range(150): 
            airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
            potential_states_fetcher.reset(pose_client, airsim_client, current_state, airsim_client.linecount, loop_mode)
            potential_states_fetcher.dome_experiment()

            for sample_ind in range(potential_states_fetcher.number_of_samples):
                photo_loc = file_manager.get_photo_loc(airsim_client.linecount)
                goal_state = potential_states_fetcher.potential_states_try[sample_ind]

                sim_pos = goal_state.position

                airsim_client.simPause(False)
                airsim_client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(sim_pos[0],sim_pos[1],sim_pos[2]), airsim.to_quaternion(0, 0, goal_state.orientation)), True)
                airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(goal_state['pitch'], 0, 0))
                current_state.cam_pitch = goal_state.pitch
                
                take_photo(airsim_client, pose_client, current_state, file_manager)
                airsim_client.simPause(True)
                
                determine_openpose_error(airsim_client.linecount, pose_client, current_state, file_manager, my_rng)

                #SAVE ALL VALUES OF THIS SIMULATION
                file_manager.append_openpose_error(pose_client.openpose_error, pose_client.openpose_arm_error,  pose_client.openpose_leg_error)
                #file_manager.save_simulation_values(airsim_client, pose_client)

                airsim_client.increment_linecount(pose_client.is_calibrating_energy)
                #print('linecount', airsim_client.linecount)

            #print("WRITING ERROR NOW!")
            file_manager.write_openpose_error(current_state.bone_pos_gt)
            plot_drone_traj(pose_client, file_manager.plot_loc, airsim_client.linecount,  pose_client.animation)
            pose_client.calib_res_list.clear()

            #implement a human pause function in airsim
            update_animation(airsim_client, pose_client, current_state, delta_t=0.3)
    date_time_name = time.strftime("%Y-%m-%d-%H-%M")
    print("experiment ended at:", date_time_name)

def save_gt_poses_loop(current_state, pose_client, airsim_client, file_manager):
    airsim_client.simPause(False)
    prev_pose_3d_gt = np.zeros([3, pose_client.num_of_joints])
    while airsim_client.linecount < 140:    
        #update state values read from AirSim and take picture
        airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
        take_photo(airsim_client, pose_client, current_state, file_manager)
        assert not np.allclose(prev_pose_3d_gt, current_state.bone_pos_gt)
        anim_time = airsim_client.getAnimationTime()
        file_manager.write_gt_pose_values(anim_time, current_state.bone_pos_gt)

        print(current_state.bone_pos_gt[:,0])
        prev_pose_3d_gt = current_state.bone_pos_gt.copy()
        start1=time.time()
        update_animation(airsim_client, pose_client, current_state, delta_t=0.2)
        print("updating animation took", time.time()-start1)

        airsim_client.increment_linecount(pose_client.is_calibrating_energy)


def create_test_set(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, my_rng):
    date_time_name = time.strftime("%Y-%m-%d-%H-%M")
    print("experiment began at:", date_time_name)
    airsim_client.simPause(False)

    for _ in range(62):        
        update_animation(airsim_client, pose_client, current_state)

        airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
        pose_client.future_pose = current_state.bone_pos_gt
        pose_client.current_pose = current_state.bone_pos_gt
        potential_states_fetcher.reset(pose_client, airsim_client, current_state)
        potential_states_try = potential_states_fetcher.dome_experiment()
        file_manager.record_gt_pose(current_state.bone_pos_gt, airsim_client.linecount)

        for state_ind in range(len(potential_states_try)):
            goal_state = potential_states_try[state_ind]
            airsim_client.simSetVehiclePose(goal_state)
            airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(goal_state.pitch, 0, 0))
            current_state.cam_pitch = goal_state.pitch

            take_photo(airsim_client, pose_client, current_state, file_manager, state_ind)
            #openpose_res, liftnet_res = determine_openpose_error(airsim_client.linecount, pose_client, current_state, file_manager.plot_loc, file_manager.get_photo_loc())
            file_manager.prepare_test_set_gt(current_state, airsim_client.linecount, state_ind)

        airsim_client.increment_linecount(pose_client.is_calibrating_energy)


def create_trackbars(radius, z_pos):
    # create trackbars for angle change
    cv2.namedWindow('Drone Control')
    cv2.createTrackbar('Angle','Drone Control', 0, 360, do_nothing)
    #cv2.setTrackbarPos('Angle', 'Angle Control', int(degrees(some_angle-INITIAL_HUMAN_ORIENTATION)))
    #cv2.setTrackbarPos('Angle', 'Drone Control', int(degrees(current_state.some_angle)))

    cv2.createTrackbar('Radius','Drone Control', 3, 10, do_nothing)
    cv2.setTrackbarPos('Radius', 'Drone Control', int(radius))

    cv2.createTrackbar('Z','Drone Control', 3, 20, do_nothing)
    cv2.setTrackbarPos('Z', 'Drone Control', z_pos)

#if (pose_client.modes["mode_3d"] == 3 and USE_AIRSIM):
#    cv2.namedWindow('Calibration for 3d pose')
#    cv2.createTrackbar('Calibration mode','Calibration for 3d pose', 0, 1, do_nothing)
#    cv2.setTrackbarPos('Calibration mode','Calibration for 3d pose', 1)

def move_drone_to_front(airsim_client, pose_client, radius):
    _, response_poses = get_simulator_responses(airsim_client)
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
    goal_state = PotentialState(position=drone_pos.copy(), orientation=new_phi_go+pi, pitch=new_theta+pi/2, index=0)
    airsim_client.simSetVehiclePose(goal_state)
    airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(0, 0, 0))

def update_animation(airsim_client, pose_client, current_state, delta_t=None):
    if delta_t is None:
        delta_t = current_state.DELTA_T
    set_animation_to_frame(airsim_client, pose_client, current_state, current_state.anim_time+delta_t)

def set_animation_to_frame(airsim_client, pose_client, current_state, anim_frame):
    if not pose_client.is_calibrating_energy:
        prev_gt_pose = current_state.bone_pos_gt.copy()
        airsim_client.setAnimationTime(anim_frame)
        new_gt_pose = airsim_retrieve_poses_gt(airsim_client, pose_client)
        i = 0
        if airsim_client.is_using_airsim:
            while (np.allclose(prev_gt_pose, new_gt_pose) and i < 10):
                time.sleep(0.05)
                new_gt_pose = airsim_retrieve_poses_gt(airsim_client, pose_client)
                i += 1
                if i==10:
                    print("waited until i==10")

        anim_time = airsim_client.getAnimationTime()
        assert anim_time != 0
        current_state.update_anim_time(anim_time)

def set_position(goal_state, airsim_client, current_state, pose_client, loop_mode):
    airsim_client.simPause(False)
    update_animation(airsim_client, pose_client, current_state)

    if loop_mode == "teleport_simulation" or loop_mode == "toy_example":
        airsim_client.simSetVehiclePose(goal_state)
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(goal_state.pitch, 0, 0))
        current_state.cam_pitch = goal_state.pitch
        airsim_client.simPause(True)

    elif loop_mode == "normal_simulation" or loop_mode == "calibration_with_momentum":
        #if (airsim_client.linecount < 5):
        #drone_speed = TOP_SPEED * airsim_client.linecount/5

        desired_pos, desired_yaw_deg, _ = goal_state.get_goal_pos_yaw_pitch(current_state.drone_orientation_gt)

        go_dist = np.linalg.norm(desired_pos[:, np.newaxis]-current_state.C_drone_gt.numpy()) 
        print("go_dist is", go_dist)
        if airsim_client.linecount < pose_client.CALIBRATION_LENGTH:
            drone_speed = go_dist
            if drone_speed > 2:
                drone_speed = 2
        else:
            if go_dist < 1:
                drone_speed = go_dist
            drone_speed = current_state.TOP_SPEED

        start_move = time.time()
        airsim_client.moveToPositionAsync(desired_pos[0], desired_pos[1], desired_pos[2], 
                                          drone_speed, current_state.DELTA_T, airsim.DrivetrainType.MaxDegreeOfFreedom, 
                                          airsim.YawMode(is_rate=False, yaw_or_rate=desired_yaw_deg), lookahead=-1, adaptive_lookahead=0).join()

        end_move = time.time()
        time_passed = end_move - start_move
        if (current_state.DELTA_T > time_passed):
            time.sleep(current_state.DELTA_T-time_passed)
            airsim_client.rotateToYawAsync(desired_yaw_deg, current_state.DELTA_T , margin = 5).join()

        cam_pitch = current_state.get_required_pitch()
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(cam_pitch, 0, 0))
        current_state.cam_pitch = cam_pitch
        airsim_client.simPause(True)