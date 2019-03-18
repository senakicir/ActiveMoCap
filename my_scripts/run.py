from helpers import * 
from NonAirSimClient import *
from PoseEstimationClient import PoseEstimationClient
from PoseEstimationClient_Simulation import PoseEstimationClient_Simulation
from pose3d_optimizer import *
from project_bones import *
from determine_positions import *
from PotentialStatesFetcher import PotentialStatesFetcher
from State import State, TOP_SPEED, TIME_HORIZON, DELTA_T
from file_manager import FileManager
from Dome_Experiment_Client import Dome_Experiment_Client
import copy

import pprint
import os
import cv2 as cv

gt_hv = []
est_hv = []
USE_AIRSIM = True
LENGTH_OF_SIMULATION = 200
photo_time = 0

def get_client_gt_values(airsim_client, pose_client, X):
    drone_orientation_gt = np.array([X['droneOrient'].x_val, X['droneOrient'].y_val, X['droneOrient'].z_val])

    gt_str = "" 
    DRONE_INITIAL_POS = airsim_client.DRONE_INITIAL_POS
    bone_ind = 0

    bone_pos_gt = np.zeros([3, 21])
    for ind, bone_i in enumerate(attributes):
        if (bone_i != 'dronePos' and bone_i != 'droneOrient' and bone_i != 'humanPos'):
            bone_pos_gt[:, bone_ind] = np.array([X[bone_i].x_val, X[bone_i].y_val, -X[bone_i].z_val]) - DRONE_INITIAL_POS
            bone_pos_gt[:, bone_ind] = bone_pos_gt[:, bone_ind]/100
            gt_str = gt_str + str(bone_pos_gt[0, bone_ind]) + '\t' + str(bone_pos_gt[1, bone_ind]) + '\t' +  str(bone_pos_gt[2, bone_ind]) + '\t'
            bone_ind += 1

    if pose_client.USE_SINGLE_JOINT:
        temp = bone_pos_gt[:, 0]
        bone_pos_gt = temp[:, np.newaxis]
    elif pose_client.model == "mpi":
        bone_pos_gt = rearrange_bones_to_mpi(bone_pos_gt)

    drone_pos_gt = np.array([X['dronePos'].x_val, X['dronePos'].y_val, -X['dronePos'].z_val])
    drone_pos_gt  = (drone_pos_gt - airsim_client.DRONE_INITIAL_POS)/100
    drone_pos_gt = drone_pos_gt[:, np.newaxis]        

    return bone_pos_gt, drone_orientation_gt, drone_pos_gt, gt_str

def airsim_retrieve_gt(airsim_client, pose_client, current_state):
    airsim_client.simPauseDrone(False)
    time.sleep(0.1)
    response = airsim_client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)])
    airsim_client.simPauseDrone(True)
    airsim_client.simPauseHuman(True)
    response = response[0]
    gt_numbers = vector3r_arr_to_dict(response.bones)
    bone_pos_gt, drone_orientation_gt, drone_pos_gt, gt_str = get_client_gt_values(airsim_client, pose_client, gt_numbers)

    #multirotor_state = airsim_client.getMultirotorState()
    #estimated_state =  multirotor_state.kinematics_estimated
    #drone_pos_est = estimated_state.position
    drone_pos_est = 0

    pose_client.f_groundtruth_str = gt_str
    current_state.store_frame_parameters(bone_pos_gt, drone_orientation_gt, drone_pos_gt, drone_pos_est)

    return response.image_data_uint8

def take_photo(airsim_client, pose_client, current_state, image_folder_loc):
    if USE_AIRSIM:
        photo = airsim_retrieve_gt(airsim_client, pose_client, current_state)
        loc = image_folder_loc + '/img_' + str(airsim_client.linecount) + '.png'
        airsim.write_file(os.path.normpath(loc), photo)
        loc_rem = image_folder_loc + '/img_' + str(airsim_client.linecount-2) + '.png'
        if os.path.isfile(loc_rem):
            os.remove(loc_rem)
    else:
        print("something is wrong!")
        response = airsim_client.simGetImages()
        bone_pos_gt = response.bone_pos
        drone_pos_est = 0
        photo = 0 
        drone_orientation_gt, drone_pos_gt = response.unreal_positions #fix this!
        pose_client.f_groundtruth_str = ""
        current_state.store_frame_parameters(bone_pos_gt, drone_orientation_gt, drone_pos_gt, drone_pos_est)

    return photo

def determine_calibration_mode(airsim_client, pose_client):
    if (airsim_client.linecount == pose_client.CALIBRATION_LENGTH):
        #client.switch_energy(energy_mode[cv2.getTrackbarPos('Calibration mode', 'Calibration for 3d pose')])
        pose_client.changeCalibrationMode(False)

def run_simulation(kalman_arguments, parameters, energy_parameters, active_parameters):
    errors = {}

    file_manager = FileManager(parameters)   

    USE_TRACKBAR = parameters["USE_TRACKBAR"]
    global USE_AIRSIM
    USE_AIRSIM = parameters["USE_AIRSIM"]
    loop_mode = parameters["LOOP_MODE"]
    #connect to the AirSim simulator
    if USE_AIRSIM:
        airsim_client = airsim.MultirotorClient()
        airsim_client.confirmConnection()
        if loop_mode == 0:
            airsim_client.enableApiControl(True)
            airsim_client.armDisarm(True)
        print('Taking off')
        airsim_client.initInitialDronePos()
        airsim_client.changeAnimation(ANIM_TO_UNREAL[file_manager.anim_num])
        airsim_client.changeCalibrationMode(True)
        if loop_mode == 0:
            airsim_client.takeoffAsync(timeout_sec = 20).join()
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(CAMERA_PITCH_OFFSET, 0, 0))
        time.sleep(2)
    else:
        f1, f2 = file_manager.get_nonairsim_client_names()
        airsim_client = NonAirSimClient(f1, f2)
    #pause airsim until we set stuff up 
    airsim_client.simPause(True)

    pose_client = PoseEstimationClient(energy_parameters,  Crop(loop_mode = loop_mode))
    current_state = State(use_single_joint=pose_client.USE_SINGLE_JOINT, model_settings=pose_client.model_settings())
    potential_states_fetcher = PotentialStatesFetcher(pose_client, active_parameters)
    
    file_manager.save_initial_drone_pos(airsim_client)

    #shoulder_vector = initial_positions[R_SHOULDER_IND, :] - initial_positions[L_SHOULDER_IND, :] #find initial human orientation!
    #INITIAL_HUMAN_ORIENTATION = np.arctan2(-shoulder_vector[0], shoulder_vector[1]) #in unreal coordinates

    if USE_TRACKBAR:
        create_trackbars(current_state.radius, active_parameters["Z_POS"])

    determine_calibration_mode(airsim_client, pose_client)

################
    if loop_mode == 0:
        normal_simulation_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager)
    elif loop_mode == 1:
        openpose_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager)
    elif loop_mode == 2:
        pose_client_sim = PoseEstimationClient_Simulation(energy_parameters,  Crop(loop_mode = loop_mode), pose_client)
        dome_experiment_client = Dome_Experiment_Client(parameters)
        dome_loop(current_state, pose_client, pose_client_sim, airsim_client, dome_experiment_client, potential_states_fetcher, file_manager)
################

    #calculate errors
    airsim_client.simPause(True)
    errors["ave_3d_err"] = sum(pose_client.error_3d)/len(pose_client.error_3d)
    errors["middle_3d_err"] = sum(pose_client.middle_pose_error)/len(pose_client.middle_pose_error)

    simple_plot(pose_client.processing_time, file_manager.estimate_folder_name, "processing_time", plot_title="Processing Time", x_label="Frames", y_label="Time")
    if (pose_client.modes["mode_3d"] == 3):
        simple_plot(pose_client.error_2d, file_manager.estimate_folder_name, "2D error", plot_title="error_2d", x_label="Frames", y_label="Error")
    simple_plot(pose_client.error_3d[:pose_client.CALIBRATION_LENGTH], file_manager.estimate_folder_name, "3D error", plot_title="calib_error_3d", x_label="Frames", y_label="Error")    
    #simple_plot(pose_client.error_3d[pose_client.CALIBRATION_LENGTH:], estimate_folder_name, "3D error", plot_title="online_error_3d", x_label="Frames", y_label="Error")
    
    print('End it!')
    airsim_client.reset()
    pose_client.reset(file_manager.plot_loc)
    file_manager.close_files()

    return errors

def normal_simulation_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager):
    end_test = False
    airsim_client.simPauseHuman(True)
    while (not end_test):
        #if USE_AIRSIM:
            #k = cv2.waitKey(1) & 0xFF
            #if k == 27:
            #    break        

        photo_loc = file_manager.get_photo_loc(airsim_client.linecount, USE_AIRSIM)

        take_photo(airsim_client, pose_client, current_state, file_manager.take_photo_loc)

        determine_calibration_mode(airsim_client, pose_client)

        determine_all_positions(airsim_client, pose_client, current_state, plot_loc=file_manager.plot_loc, photo_loc=photo_loc)

        cam_pitch = current_state.get_required_pitch()
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(cam_pitch, 0, 0))

        potential_states_fetcher.reset(pose_client, current_state)

        trajectory = potential_states_fetcher.trajectory 
        if airsim_client.linecount < pose_client.PRECALIBRATION_LENGTH:
            goal_state = potential_states_fetcher.precalibration()
            drone_speed = TOP_SPEED
        else:
            if (trajectory == 0):
                potential_states_fetcher.get_potential_positions_really_spherical_future()
                potential_states_fetcher.find_hessians_for_potential_states(pose_client)
                goal_state, _ = potential_states_fetcher.find_best_potential_state()
                potential_states_fetcher.plot_everything(airsim_client.linecount, file_manager.plot_loc, photo_loc)
            if (trajectory == 1):
                goal_state = potential_states_fetcher.constant_rotation_baseline_future()
            if (trajectory == 2): #RANDOM
                potential_states_fetcher.get_potential_positions_really_spherical_future()
                goal_state = potential_states_fetcher.find_random_next_state()
            if (trajectory == 4):
                goal_state = potential_states_fetcher.wobbly_baseline()
            if (trajectory == 5):
                goal_state = potential_states_fetcher.up_down_baseline()
            if (trajectory == 6):
                goal_state = potential_states_fetcher.left_right_baseline()
            drone_speed = TOP_SPEED

        desired_pos, desired_yaw_deg, _ = current_state.get_goal_pos_yaw_pitch(goal_state)

        #find desired drone speed
        #desired_vel = 5#(desired_pos - current_state.drone_pos)/TIME_HORIZON #how much the drone will have to move for this iteration
        #drone_speed = np.linalg.norm(desired_vel)    
        if (airsim_client.linecount < 5):
            drone_speed = drone_speed * airsim_client.linecount/5

        airsim_client.simPauseDrone(False)
        #add if here for calib
        if USE_AIRSIM:
            start_move = time.time()
            airsim_client.moveToPositionAsync(desired_pos[0], desired_pos[1], desired_pos[2], drone_speed, DELTA_T, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(is_rate=False, yaw_or_rate=desired_yaw_deg), lookahead=-1, adaptive_lookahead=0).join()
            end_move = time.time()
            time_passed = end_move - start_move
            print("time passed", time_passed)
            if (DELTA_T > time_passed):
                time.sleep(DELTA_T-time_passed)
                print("I was sleeping :(")
        else:
            airsim_client.moveToPositionAsync() #placeholderfunc
        airsim_client.simPauseDrone(True)

        current_state.cam_pitch = cam_pitch
        plot_drone_traj(pose_client, file_manager.plot_loc, airsim_client.linecount)
    
        file_manager.save_simulation_values(airsim_client, pose_client)

        airsim_client.linecount += 1
        print('linecount', airsim_client.linecount)

        if (not USE_AIRSIM):
            end_test = airsim_client.end
        else:
            if (airsim_client.linecount == LENGTH_OF_SIMULATION):
                end_test = True

def precalibration(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager):
    airsim_client.simPauseHuman(True)
    airsim_retrieve_gt(airsim_client, pose_client, current_state)

    for _ in range(pose_client.PRECALIBRATION_LENGTH):
        photo_loc = file_manager.get_photo_loc(airsim_client.linecount, USE_AIRSIM)
        take_photo(airsim_client, pose_client, current_state, file_manager.take_photo_loc)

        determine_all_positions(airsim_client, pose_client, current_state, plot_loc=file_manager.plot_loc, photo_loc=photo_loc)

        cam_pitch = current_state.get_required_pitch()
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(cam_pitch, 0, 0))
        current_state.cam_pitch = cam_pitch

        potential_states_fetcher.reset(pose_client, current_state)
        goal_state = potential_states_fetcher.precalibration()

        desired_pos, desired_yaw_deg, _ = current_state.get_goal_pos_yaw_pitch(goal_state)
        
        drone_speed = TOP_SPEED
        if (airsim_client.linecount < 5):
            drone_speed = drone_speed * airsim_client.linecount/5

        airsim_client.simPauseDrone(False)
        start_move = time.time()
        airsim_client.moveToPositionAsync(desired_pos[0], desired_pos[1], desired_pos[2], drone_speed, DELTA_T, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(is_rate=False, yaw_or_rate=desired_yaw_deg), lookahead=-1, adaptive_lookahead=0).join()
        end_move = time.time()
        time_passed = end_move - start_move
        print("time passed", time_passed)
        if (DELTA_T > time_passed):
            time.sleep(DELTA_T-time_passed)
            print("I was sleeping :(")
        airsim_client.simPauseDrone(True)

        plot_drone_traj(pose_client, file_manager.plot_loc, airsim_client.linecount)
    
        file_manager.save_simulation_values(airsim_client, pose_client)

        airsim_client.linecount += 1
        print('linecount', airsim_client.linecount)


       
def openpose_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager):
    #animations_to_test = ["64_06", "02_01", "05_08", "38_03"]
    date_time_name = time.strftime("%Y-%m-%d-%H-%M")
    print("experiment began at:", date_time_name)
    file_manager.write_openpose_prefix(potential_states_fetcher.THETA_LIST, potential_states_fetcher.PHI_LIST, pose_client.num_of_joints)

    for animation in range(1,19):
        airsim_client.simPauseDrone(True)
        #airsim_client.changeAnimation(ANIM_TO_UNREAL[animation])
        airsim_client.changeAnimation(animation)
        print("Animation:", animation)
        time.sleep(1)
        for _ in range(150): 
            airsim_retrieve_gt(airsim_client, pose_client, current_state)
            potential_states_fetcher.reset(pose_client, current_state)
            potential_states_fetcher.dome_experiment()

            for sample_ind in range(potential_states_fetcher.number_of_samples):
                photo_loc = file_manager.get_photo_loc(airsim_client.linecount, USE_AIRSIM)
                goal_state = potential_states_fetcher.potential_states_try[sample_ind]

                sim_pos = goal_state['position']

                airsim_client.simPauseDrone(False)
                airsim_client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(sim_pos[0],sim_pos[1],sim_pos[2]), airsim.to_quaternion(0, 0, goal_state["orientation"])), True)
                airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(goal_state['pitch'], 0, 0))
                current_state.cam_pitch = goal_state['pitch']
                
                take_photo(airsim_client, pose_client, current_state,  file_manager.take_photo_loc)
                airsim_client.simPauseDrone(True)
                
                determine_openpose_error(airsim_client, pose_client, current_state, plot_loc = file_manager.plot_loc, photo_loc = photo_loc)

                #SAVE ALL VALUES OF THIS SIMULATION
                file_manager.append_openpose_error(pose_client.openpose_error, pose_client.openpose_arm_error,  pose_client.openpose_leg_error)
                #file_manager.save_simulation_values(airsim_client, pose_client)

                airsim_client.linecount += 1
                #print('linecount', airsim_client.linecount)

            #print("WRITING ERROR NOW!")
            file_manager.write_openpose_error(current_state.bone_pos_gt)
            plot_drone_traj(pose_client, file_manager.plot_loc, airsim_client.linecount)
            pose_client.calib_res_list.clear()

            #implement a human pause function in airsim
            airsim_client.simPauseHuman(False)
            time.sleep(0.3)
            airsim_client.simPauseHuman(True)
    date_time_name = time.strftime("%Y-%m-%d-%H-%M")
    print("experiment ended at:", date_time_name)

def dome_loop(current_state, pose_client, pose_client_sim, airsim_client, dome_experiment_client, potential_states_fetcher, file_manager):
    date_time_name = time.strftime("%Y-%m-%d-%H-%M")
    print("experiment began at:", date_time_name)
    airsim_client.simPauseDrone(False)

    airsim_retrieve_gt(airsim_client, pose_client, current_state)
    dome_experiment_client.init_3d_pose(current_state.bone_pos_gt)
    pose_client.future_pose = current_state.bone_pos_gt
    pose_client.current_pose = current_state.bone_pos_gt

    potential_states_fetcher.reset(pose_client, current_state)
    potential_states_try = potential_states_fetcher.dome_experiment()

    #START AT POSITION 0
    goal_state = potential_states_fetcher.potential_states_try[0]
    sim_pos = goal_state['position']
    airsim_client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(sim_pos[0],sim_pos[1],sim_pos[2]), airsim.to_quaternion(0, 0, goal_state["orientation"])), True)
    airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(goal_state['pitch'], 0, 0))
    current_state.cam_pitch = goal_state['pitch']
    take_photo(airsim_client, pose_client, current_state,  file_manager.take_photo_loc)
    photo_loc = file_manager.get_photo_loc(airsim_client.linecount, USE_AIRSIM)

    initialize_with_gt(airsim_client, pose_client, current_state, plot_loc=file_manager.plot_loc, photo_loc=photo_loc)
    airsim_client.linecount += 1

    for exp_ind in range(1, 100):        
        potential_states_fetcher.reset(pose_client, current_state)
        potential_states_fetcher.potential_states_try = potential_states_try
        potential_states_fetcher.potential_states_go = potential_states_try

        if not pose_client.isCalibratingEnergy:
            airsim_client.simPauseHuman(False)
            time.sleep(DELTA_T)
            airsim_client.simPauseHuman(True)
            dome_experiment_client.adjust_3d_pose(current_state, pose_client)

            potential_states_fetcher.reset(pose_client, current_state)
            potential_states_try = potential_states_fetcher.dome_experiment()

        if dome_experiment_client.find_best_traj: #/and exp_ind >= predefined_traj_len:
            pose_client_sim.update_initial_param(pose_client)
            num_of_noise_trials = dome_experiment_client.num_of_noise_trials
            for state_ind in range(len(potential_states_try)):
                goal_state = potential_states_try[state_ind]
                sim_pos = goal_state['position']
                airsim_client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(sim_pos[0],sim_pos[1],sim_pos[2]), airsim.to_quaternion(0, 0, goal_state["orientation"])), True)
                airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(goal_state['pitch'], 0, 0))
                current_state.cam_pitch = goal_state['pitch']
                for trial_ind in range(num_of_noise_trials):
                    take_photo(airsim_client, pose_client_sim, current_state, file_manager.take_photo_loc)
                    photo_loc = file_manager.get_photo_loc(airsim_client.linecount, USE_AIRSIM)
                    determine_all_positions(airsim_client, pose_client_sim, current_state, plot_loc=file_manager.plot_loc, photo_loc=photo_loc)
                    dome_experiment_client.frame_overall_error_list[trial_ind], dome_experiment_client.frame_future_error_list[trial_ind] = pose_client_sim.get_error()
                    pose_client_sim.rewind_step()
                dome_experiment_client.record_noise_experiment_statistics(potential_states_fetcher, state_ind)

            best_index = np.argmin(potential_states_fetcher.overall_error_list)
            print("best index was", best_index, "with error", potential_states_fetcher.overall_error_list[state_ind])

        potential_states_fetcher.find_hessians_for_potential_states(pose_client)
        if exp_ind < dome_experiment_client.predefined_traj_len:
            goal_state = potential_states_fetcher.potential_states_try[exp_ind]
            potential_states_fetcher.goal_state_ind = exp_ind
        else:
            goal_state, _ = potential_states_fetcher.find_best_potential_state()    
        potential_states_fetcher.plot_everything(airsim_client.linecount, file_manager.plot_loc, "")

        sim_pos = goal_state['position']
        airsim_client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(sim_pos[0],sim_pos[1],sim_pos[2]), airsim.to_quaternion(0, 0, goal_state["orientation"])), True)
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(goal_state['pitch'], 0, 0))
        current_state.cam_pitch = goal_state['pitch']

        take_photo(airsim_client, pose_client, current_state,  file_manager.take_photo_loc)
        photo_loc = file_manager.get_photo_loc(airsim_client.linecount, USE_AIRSIM)

        determine_all_positions(airsim_client, pose_client, current_state, plot_loc=file_manager.plot_loc, photo_loc=photo_loc)

        airsim_client.linecount += 1
        plot_drone_traj(pose_client, file_manager.plot_loc, airsim_client.linecount)
        print('linecount', airsim_client.linecount)

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

def pause_function(airsim_client, pose_client):
    airsim_client.simPauseDrone(True)
    airsim_client.simPauseHuman(True)

def unpause_function(airsim_client, pose_client):
    airsim_client.simPauseDrone(False)
    if not pose_client.isCalibratingEnergy:
        airsim_client.simPauseHuman(False)