from helpers import * 
from PoseEstimationClient import PoseEstimationClient
from Potential_Error_Finder import Potential_Error_Finder
from pose3d_optimizer import *
from project_bones import *
from determine_positions import *
from PotentialStatesFetcher import PotentialStatesFetcher, PotentialState
from State import State
from file_manager import FileManager, get_bone_len_file_name
from drone_flight_client import DroneFlightClient
from crop import Crop
import copy
from PIL import Image

import pprint
import os
import cv2 as cv

gt_hv = []
est_hv = []
photo_time = 0

def get_client_gt_values(airsim_client, pose_client, simulated_value_dict, file_manager):
    """
    Description: 
        Sort out values read from simulator and return them

    Inputs: 
        airsim_client: an object of type VehicleClient or DroneFlightClient
        pose_client: an object of type PoseEstimationClient
        simulated_value_dict: dict type input of values read from AirSim simulator (or previously saved values)
        file_manager: object of type FileManager
    Returns:
        bone_pos_gt: numpy array of shape (3, num_of_joints)
        drone_orientation_gt: numpy array of shape (3,)
        drone_pos_gt: numpy array of shape (3,1)
    """

    drone_orientation_gt = np.array([simulated_value_dict['droneOrient'].x_val, simulated_value_dict['droneOrient'].y_val, 
                            simulated_value_dict['droneOrient'].z_val])

    #convert Unreal engine coordinates to AirSim coordinates.
    bone_pos_gt = np.zeros([3, 21])
    bone_ind = 0
    for ind, attribute in enumerate(airsim.attributes):
        if (attribute != 'dronePos' and attribute != 'droneOrient' and attribute != 'humanPos'):
            bone_pos_gt[:, bone_ind] = np.array([simulated_value_dict[attribute].x_val, simulated_value_dict[attribute].y_val, 
                                        -simulated_value_dict[attribute].z_val]) - airsim_client.DRONE_INITIAL_POS
            bone_pos_gt[:, bone_ind] = bone_pos_gt[:, bone_ind]/100
            bone_ind += 1

    if pose_client.USE_SINGLE_JOINT:
        temp = bone_pos_gt[:, 0]
        bone_pos_gt = temp[:, np.newaxis]
    if pose_client.model == "mpi":
        bone_pos_gt = rearrange_bones_to_mpi(bone_pos_gt)

    drone_pos_gt = np.array([simulated_value_dict['dronePos'].x_val, simulated_value_dict['dronePos'].y_val, 
                            -simulated_value_dict['dronePos'].z_val])
    drone_pos_gt = (drone_pos_gt - airsim_client.DRONE_INITIAL_POS)/100
    drone_pos_gt = drone_pos_gt[:, np.newaxis] 

    file_manager.record_gt_pose(bone_pos_gt, airsim_client.linecount)
    file_manager.record_drone_info(drone_pos_gt, drone_orientation_gt, airsim_client.linecount)

    assert bone_pos_gt.shape == (3,pose_client.num_of_joints)
    assert drone_orientation_gt.shape == (3,)
    assert drone_pos_gt.shape == (3,1)

    return bone_pos_gt, drone_orientation_gt, drone_pos_gt

def airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager):
    """
    Description: 
        Calls simulator to take picture and return GT values simulatenously. 

    Inputs: 
        airsim_client: an object of type VehicleClient or DroneFlightClient
        pose_client: an object of type PoseEstimationClient
        current_state: an object of type State
        file_manager: object of type FileManager
    Returns:
        image: photo taken at simulation step
    """
    airsim_client.simPause(False) #unpause drone to take picture
    if  airsim_client.is_using_airsim:
        time.sleep(0.1)

    response = airsim_client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)])

    airsim_client.simPause(True) #pause everything to start processing
    response = response[0]
    if airsim_client.is_using_airsim:
        gt_values = vector3r_arr_to_dict(response.bones)
        bone_pos_gt, drone_orientation_gt, drone_pos_gt = get_client_gt_values(airsim_client, pose_client, gt_values, file_manager)
        #multirotor_state = airsim_client.getMultirotorState()
        #estimated_state =  multirotor_state.kinematics_estimated
        #drone_pos_est = estimated_state.position

        current_state.store_frame_parameters(bone_pos_gt, drone_orientation_gt, drone_pos_gt)
    else:
        bone_pos_gt, drone_transformation_matrix = airsim_client.read_frame_gt_values()
        current_state.store_frame_transformation_matrix_joint_gt(bone_pos_gt, drone_transformation_matrix)

    #figure out a way to convert png bytes to float array
    image = response.image_data_uint8
    current_state.update_anim_time(airsim_client.getAnimationTime())
    #image_buffer =  Image.frombytes(mode="I", size=(airsim_client.SIZE_X, airsim_client.SIZE_Y), data=image)
   # print(image_buffer.shape)
    #image_buffer.show()

    return image

def take_photo(airsim_client, pose_client, current_state, file_manager, viewpoint = ""):
    """
    Description: 
        Calls simulator to take picture and return GT values simultaneously.
        Writes picture file.

    Inputs: 
        airsim_client: an object of type VehicleClient or DroneFlightClient
        pose_client: an object of type PoseEstimationClient
        current_state: an object of type State
        file_manager: object of type FileManager
        viewpoint: int. (default: ""), the viewpoint from which the drone is looking at the person
    Returns:
        photo: photo taken at simulation step
    """
    if airsim_client.is_using_airsim:
        photo = airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
        loc = file_manager.update_photo_loc(linecount=airsim_client.linecount, viewpoint=viewpoint)
        airsim.write_file(os.path.normpath(loc), photo)
        if airsim_client.linecount > 3 and pose_client.quiet:
            loc_rem = file_manager.take_photo_loc + '/img_' + str(airsim_client.linecount-2) + '.png'
            if os.path.isfile(loc_rem):
                os.remove(loc_rem)
    else:
        photo = airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
        file_manager.update_photo_loc(linecount=airsim_client.linecount, viewpoint=viewpoint)
    return photo

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
    port =  parameters["PORT"]

    #set random seeds once and for all
    np.random.seed(parameters["SEED"])
    torch.manual_seed(parameters["SEED"])

    #connect to the AirSim simulator
    if simulation_mode == "use_airsim":
        airsim_client = airsim.MultirotorClient(length_of_simulation, port=port)
        #airsim_client = airsim.MultirotorClient(length_of_simulation)
        airsim_client.confirmConnection()
        airsim_client.reset()
        if loop_mode == "normal_simulation":
            airsim_client.enableApiControl(True)
            airsim_client.armDisarm(True)
        airsim_client.initInitialDronePos()
        airsim_client.changeAnimation(ANIM_TO_UNREAL[file_manager.anim_num])
        if loop_mode == "normal_simulation":
            airsim_client.takeoffAsync(timeout_sec = 15)#.join()
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(CAMERA_PITCH_OFFSET, 0, 0))
    elif simulation_mode == "saved_simulation":
        airsim_client = DroneFlightClient(length_of_simulation, file_manager.anim_num, file_manager.non_simulation_files)
        #file_manager.label_list = airsim_client.label_list
    #pause airsim until we set stuff up 
    airsim_client.simPause(True)

    pose_client = PoseEstimationClient(param=energy_parameters, general_param=parameters, 
                    intrinsics_focal=airsim_client.focal_length, intrinsics_px=airsim_client.px, 
                    intrinsics_py=airsim_client.py, image_size=(airsim_client.SIZE_X, airsim_client.SIZE_Y))
    current_state = State(use_single_joint=pose_client.USE_SINGLE_JOINT, active_parameters=active_parameters,
                         model_settings=pose_client.model_settings(), anim_gt_array=file_manager.f_anim_gt_array, 
                         future_window_size=pose_client.FUTURE_WINDOW_SIZE)
    potential_states_fetcher = PotentialStatesFetcher(airsim_client=airsim_client, pose_client=pose_client, 
                                active_parameters=active_parameters, loop_mode=loop_mode)
    if loop_mode != "calibration":
        pose_client.read_bone_lengths_from_file(file_manager)
    file_manager.save_initial_drone_pos(airsim_client)

    #if pose_client.modes["mode_2d"] == "openpose":
    #    import openpose as openpose_module
    #if pose_client.modes["mode_lift"] == "lift":
    #    import liftnet as liftnet_module

    #shoulder_vector = initial_positions[R_SHOULDER_IND, :] - initial_positions[L_SHOULDER_IND, :] #find initial human orientation!
    #INITIAL_HUMAN_ORIENTATION = np.arctan2(-shoulder_vector[0], shoulder_vector[1]) #in unreal coordinates

    ################
    if loop_mode == "normal_simulation" or loop_mode == "teleport_simulation" or loop_mode == "toy_example" or loop_mode == "calibration":
        general_simulation_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, parameters)
    elif loop_mode == "openpose":
        openpose_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager)
    #elif loop_mode == "teleport":
     #   teleport_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, loop_mode, parameters)
    elif loop_mode == "save_gt_poses":
        save_gt_poses_loop(current_state, pose_client, airsim_client, file_manager)
    elif loop_mode == "create_dataset":
        create_test_set(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager)
    ################

    #calculate errors
    airsim_client.simPause(True)
    average_errors = pose_client.average_errors
    ave_current_error, ave_middle_error, ave_pastmost_error, ave_overall_error = pose_client.average_errors[pose_client.CURRENT_POSE_INDEX],  pose_client.average_errors[pose_client.MIDDLE_POSE_INDEX], pose_client.average_errors[pose_client.PASTMOST_POSE_INDEX], pose_client.ave_overall_error

    if loop_mode == "calibration":
        file_manager.save_bone_lengths(pose_client.boneLengths)

    print('End it!')
    pose_client.reset(file_manager.plot_loc)
    file_manager.close_files()

    return ave_current_error, ave_middle_error, ave_pastmost_error, ave_overall_error 

def general_simulation_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, parameters):
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
    potential_error_finder = Potential_Error_Finder(parameters)


    airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
    time.sleep(0.5)

    take_photo(airsim_client, pose_client, current_state, file_manager)
    
    initialize_empty_frames(airsim_client.linecount, pose_client, current_state, file_manager)

    airsim_client.simPause(True)
    potential_states_fetcher.reset(pose_client, airsim_client, current_state)
    potential_states_fetcher.get_potential_positions(pose_client.is_calibrating_energy)
    while airsim_client.linecount < airsim_client.length_of_simulation:   
        #### if we have the best traj finder 
        if potential_error_finder.find_best_traj and not pose_client.is_calibrating_energy:
            start_best_sim = time.time()   
            torch_seed_state = torch.get_rng_state()
            np_seed_state = np.random.get_state()
            current_anim_time = airsim_client.getAnimationTime()

            #find goal location
            for trajectory_ind in range(0, len(potential_states_fetcher.potential_trajectory_list)):
                print("* trajectory_ind", trajectory_ind)
                torch.set_rng_state(torch_seed_state)
                np.random.set_state(np_seed_state)                    
                goal_traj = potential_states_fetcher.choose_trajectory_using_trajind(trajectory_ind)
                for trial_ind in range(potential_error_finder.num_of_noise_trials):
                    #print("** trial ind", trial_ind)
                    potential_states_fetcher.restart_trajectory()
                    pose_client_copy = pose_client.deepcopy_PEC(trial_ind)
                    state_copy = current_state.deepcopy_state()
                    airsim_client.setAnimationTime(current_anim_time)
                    for future_ind in range(pose_client_copy.FUTURE_WINDOW_SIZE):
                        #print("*** future_ind", future_ind)
                        goal_state = potential_states_fetcher.move_along_trajectory()
                        #set position also updates animation
                        set_position(goal_state, airsim_client, state_copy, pose_client_copy, loop_mode=potential_states_fetcher.loop_mode)
                    
                        take_photo(airsim_client, pose_client_copy, state_copy, file_manager, trajectory_ind)           
                        determine_positions(airsim_client.linecount, pose_client_copy, state_copy, file_manager) 
                        goal_traj.record_error_for_trial(future_ind, pose_client_copy.middle_error, pose_client_copy.overall_error)
                goal_traj.find_overall_error()
            #file_manager.record_toy_example_results_error(linecount, self.potential_trajectory_list, self.goal_trajectory)
            potential_states_fetcher.restart_trajectory()
            torch.set_rng_state(torch_seed_state)
            np.random.set_state(np_seed_state)
            airsim_client.setAnimationTime(current_anim_time)
            end_best_sim = time.time()
            print("Simulating errors for all locations took", end_best_sim-start_best_sim, "seconds")

        #find goal location
        if airsim_client.linecount != 0:
            start2=time.time()
            potential_states_fetcher.choose_trajectory(pose_client, airsim_client.linecount, airsim_client.online_linecount, file_manager)
            goal_state = potential_states_fetcher.move_along_trajectory()
            end2= time.time()
            print("Choosing a trajectory took", end2-start2, "seconds")

            #move there
            set_position(goal_state, airsim_client, current_state, pose_client, loop_mode=potential_states_fetcher.loop_mode)

        #update state values read from AirSim and take picture
        take_photo(airsim_client, pose_client, current_state, file_manager)        

        #find human pose 
        start3=time.time() 
        determine_positions(airsim_client.linecount, pose_client, current_state, file_manager)
        end3=time.time()
        print("finding human pose took", end3-start3, "seconds")

        #plotting
        if not pose_client.quiet and airsim_client.linecount > 0:
            plot_drone_traj(pose_client, file_manager.plot_loc, airsim_client.linecount,  pose_client.animation)

        file_manager.write_error_values(pose_client.average_errors, airsim_client.linecount)
        
        if not pose_client.is_calibrating_energy and not pose_client.quiet and file_manager.loop_mode == "toy_example":
            plot_potential_errors_and_uncertainties_matrix(airsim_client.linecount, potential_states_fetcher.potential_trajectory_list,
                                                            potential_states_fetcher.goal_trajectory, potential_error_finder.find_best_traj, file_manager.plot_loc)
        airsim_client.increment_linecount(pose_client.is_calibrating_energy)

        potential_states_fetcher.reset(pose_client, airsim_client, current_state)
        potential_states_fetcher.get_potential_positions(pose_client.is_calibrating_energy)

        
def normal_simulation_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, loop_mode):
    """
    Description: 
        Simulation loop that is called when loop_mode is either "normal_simulation" or "teleport_simulation"
        Runs with AirSim.

    Inputs: 
        current_state: an object of type State
        pose_client: an object of type PoseEstimationClient
        airsim_client: an object of type VehicleClient or DroneFlightClient
        potential_states_fetcher: an object of type PotentialStatesFetcher
        file_manager: object of type FileManager
        loop_mode: string, either "normal_simulation" or "teleport_simulation"
    """

    #don't take a photo but retrieve initial gt values 
    airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
    #time.sleep(0.5)

    airsim_client.simPause(True)
    while airsim_client.linecount < airsim_client.length_of_simulation:    

        #update state values read from AirSim and take picture
        take_photo(airsim_client, pose_client, current_state, file_manager)

        #find human pose 
        determine_positions(airsim_client.linecount, pose_client, current_state, file_manager)

        #find goal location
        potential_states_fetcher.reset(pose_client, airsim_client, current_state)
        potential_states_fetcher.get_potential_positions(pose_client.is_calibrating_energy)
        potential_states_fetcher.choose_trajectory(pose_client, airsim_client.linecount, airsim_client.online_linecount, file_manager)
        goal_state = potential_states_fetcher.move_along_trajectory()

        #move there
        set_position(goal_state, airsim_client, current_state, pose_client, loop_mode=loop_mode)

        #plotting
        if not pose_client.quiet and airsim_client.linecount > 0:
            plot_drone_traj(pose_client, file_manager.plot_loc, airsim_client.linecount,  pose_client.animation)

        file_manager.write_error_values(pose_client.errors, airsim_client.linecount)
        airsim_client.increment_linecount(pose_client.is_calibrating_energy)
       
def openpose_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager):
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
                
                determine_openpose_error(airsim_client.linecount, pose_client, current_state, file_manager)

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
            airsim_client.updateAnimation(0.3)
    date_time_name = time.strftime("%Y-%m-%d-%H-%M")
    print("experiment ended at:", date_time_name)

def save_gt_poses_loop(current_state, pose_client, airsim_client, file_manager):
    while airsim_client.linecount < 1000:    
        #update state values read from AirSim and take picture
        airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
        print("currentpose",current_state.bone_pos_gt[:,0])
        anim_time = airsim_client.getAnimationTime()
        file_manager.write_gt_pose_values(anim_time, current_state.bone_pos_gt)

        #move there
        airsim_client.updateAnimation(0.05)
        airsim_client.increment_linecount(pose_client.is_calibrating_energy)


def create_test_set(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager):
    date_time_name = time.strftime("%Y-%m-%d-%H-%M")
    print("experiment began at:", date_time_name)
    airsim_client.simPause(False)

    for _ in range(62):        
        airsim_client.updateAnimation(current_state.DELTA_T)

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

def set_position(goal_state, airsim_client, current_state, pose_client, loop_mode):
    airsim_client.simPause(False)
    if not pose_client.is_calibrating_energy:
        airsim_client.updateAnimation(current_state.DELTA_T)
    anim_time = airsim_client.getAnimationTime()
    print("anim time is", anim_time)
    if anim_time != 0:
        current_state.update_anim_time(anim_time)
    else:
        airsim_client.setAnimationTime(current_state.anim_time + current_state.DELTA_T)
        current_state.update_anim_time(current_state.anim_time + current_state.DELTA_T)
        anim_time = airsim_client.getAnimationTime()
        assert anim_time != 0

    if loop_mode == "teleport_simulation" or loop_mode == "toy_example" or loop_mode == "calibration":
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