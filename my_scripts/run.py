from helpers import * 
from PoseEstimationClient import PoseEstimationClient
from Potential_Error_Finder import Potential_Error_Finder
from pose3d_optimizer import *
from project_bones import *
from determine_positions import *
from PotentialStatesFetcher import PotentialStatesFetcher, PotentialState
from State import State, TOP_SPEED, TIME_HORIZON, DELTA_T
from file_manager import FileManager
from drone_flight_client import DroneFlightClient
from crop import Crop
import copy

import pprint
import os
import cv2 as cv

gt_hv = []
est_hv = []
photo_time = 0

def get_client_gt_values(airsim_client, pose_client, X, file_manager):
    drone_orientation_gt = np.array([X['droneOrient'].x_val, X['droneOrient'].y_val, X['droneOrient'].z_val])

    gt_str = "" 
    DRONE_INITIAL_POS = airsim_client.DRONE_INITIAL_POS
    bone_ind = 0

    bone_pos_gt = np.zeros([3, 21])
    for ind, bone_i in enumerate(airsim.attributes):
        if (bone_i != 'dronePos' and bone_i != 'droneOrient' and bone_i != 'humanPos'):
            bone_pos_gt[:, bone_ind] = np.array([X[bone_i].x_val, X[bone_i].y_val, -X[bone_i].z_val]) - DRONE_INITIAL_POS
            bone_pos_gt[:, bone_ind] = bone_pos_gt[:, bone_ind]/100
            bone_ind += 1

    if pose_client.USE_SINGLE_JOINT:
        temp = bone_pos_gt[:, 0]
        bone_pos_gt = temp[:, np.newaxis]
    elif pose_client.model == "mpi":
        bone_pos_gt = rearrange_bones_to_mpi(bone_pos_gt)

    drone_pos_gt = np.array([X['dronePos'].x_val, X['dronePos'].y_val, -X['dronePos'].z_val])
    drone_pos_gt = (drone_pos_gt - airsim_client.DRONE_INITIAL_POS)/100
    drone_pos_gt = drone_pos_gt[:, np.newaxis] 

    return bone_pos_gt, drone_orientation_gt, drone_pos_gt

def airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager):
    airsim_client.simPauseDrone(False)
    if  airsim_client.is_using_airsim:
        time.sleep(0.1)
    response = airsim_client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)])
    airsim_client.simPauseDrone(True)
    airsim_client.simPauseHuman(True)
    response = response[0]

    if airsim_client.is_using_airsim:
        gt_joints = vector3r_arr_to_dict(response.bones)
        bone_pos_gt, drone_orientation_gt, drone_pos_gt = get_client_gt_values(airsim_client, pose_client, gt_joints, file_manager)
        #multirotor_state = airsim_client.getMultirotorState()
        #estimated_state =  multirotor_state.kinematics_estimated
        #drone_pos_est = estimated_state.position
        drone_pos_est = 0

        current_state.store_frame_parameters(bone_pos_gt, drone_orientation_gt, drone_pos_gt, drone_pos_est)
    else:
        bone_pos_gt, drone_transformation_matrix = airsim_client.read_frame_gt_values()
        current_state.store_frame_transformation_matrix_joint_gt(bone_pos_gt, drone_transformation_matrix)

    return response.image_data_uint8

def take_photo(airsim_client, pose_client, current_state, file_manager, viewpoint = ""):
    if airsim_client.is_using_airsim:
        photo = airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
        if viewpoint == "":
            loc = file_manager.take_photo_loc + '/img_' + str(airsim_client.linecount) + '.png'
        else:
            loc = file_manager.take_photo_loc + '/img_' + str(airsim_client.linecount) + "_viewpoint_" + str(viewpoint)  + '.png'
        file_manager.update_photo_loc(airsim_client.linecount, viewpoint)
        airsim.write_file(os.path.normpath(loc), photo)
        #loc_rem = file_manager.take_photo_loc + '/img_' + str(airsim_client.linecount-2) + '.png'
        #if os.path.isfile(loc_rem):
        #    os.remove(loc_rem)
    else:
        photo = airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
        file_manager.update_photo_loc(airsim_client.linecount, viewpoint)
    return photo

def determine_calibration_mode(linecount, pose_client):
    if (linecount == pose_client.CALIBRATION_LENGTH):
        #client.switch_energy(energy_mode[cv2.getTrackbarPos('Calibration mode', 'Calibration for 3d pose')])
        pose_client.changeCalibrationMode(False)

def run_simulation(kalman_arguments, parameters, energy_parameters, active_parameters):
    errors = {}

    file_manager = FileManager(parameters)   

    USE_TRACKBAR = parameters["USE_TRACKBAR"]
    simulation_mode = parameters["SIMULATION_MODE"]
    length_of_simulation = parameters["LENGTH_OF_SIMULATION"]
    loop_mode = parameters["LOOP_MODE"]

    #connect to the AirSim simulator
    if simulation_mode == "use_airsim":
        airsim_client = airsim.MultirotorClient(length_of_simulation)
        airsim_client.confirmConnection()
        #if loop_mode == "normal":
        #    airsim_client.enableApiControl(True)
         #   airsim_client.armDisarm(True)
        print('Taking off')
        airsim_client.initInitialDronePos()
        airsim_client.changeAnimation(ANIM_TO_UNREAL[file_manager.anim_num])
        airsim_client.changeCalibrationMode(True)
        #if loop_mode == "normal":
        #    airsim_client.takeoffAsync(timeout_sec = 20).join()
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(CAMERA_PITCH_OFFSET, 0, 0))
        time.sleep(2)
    elif simulation_mode == "saved_simulation":
        airsim_client = DroneFlightClient(length_of_simulation, file_manager.anim_num, file_manager.non_simulation_files)
        #file_manager.label_list = airsim_client.label_list
    #pause airsim until we set stuff up 
    airsim_client.simPause(True)

    pose_client = PoseEstimationClient(energy_parameters, loop_mode, animation=file_manager.anim_num, intrinsics_focal=airsim_client.focal_length, intrinsics_px=airsim_client.px, intrinsics_py=airsim_client.py, image_size=(airsim_client.SIZE_X, airsim_client.SIZE_Y))
    current_state = State(use_single_joint=pose_client.USE_SINGLE_JOINT, model_settings=pose_client.model_settings())
    potential_states_fetcher = PotentialStatesFetcher(airsim_client, pose_client, active_parameters)
    
    file_manager.save_initial_drone_pos(airsim_client)

    #shoulder_vector = initial_positions[R_SHOULDER_IND, :] - initial_positions[L_SHOULDER_IND, :] #find initial human orientation!
    #INITIAL_HUMAN_ORIENTATION = np.arctan2(-shoulder_vector[0], shoulder_vector[1]) #in unreal coordinates

    if USE_TRACKBAR:
        create_trackbars(current_state.radius, active_parameters["Z_POS"])

    determine_calibration_mode(airsim_client.linecount, pose_client)

################
    if loop_mode == "normal":
        #normal_simulation_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager)
        teleport_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, loop_mode, parameters)
    elif loop_mode == "openpose":
        openpose_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager)
    elif loop_mode == "teleport":
        teleport_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, loop_mode, parameters)
    elif loop_mode == "create_dataset":
        create_test_set(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager)
    
################

    #calculate errors
    airsim_client.simPause(True)
    if len(pose_client.error_3d) != 0:
        errors["ave_3d_err"] = sum(pose_client.error_3d)/len(pose_client.error_3d)
        errors["middle_3d_err"] = sum(pose_client.middle_pose_error)/len(pose_client.middle_pose_error)
    else:
        errors["ave_3d_err"] = None
        errors["middle_3d_err"] = None

    simple_plot(pose_client.processing_time, file_manager.estimate_folder_name, "processing_time", plot_title="Processing Time", x_label="Frames", y_label="Time")
    if (pose_client.modes["mode_3d"] == 3):
        simple_plot(pose_client.error_2d, file_manager.estimate_folder_name, "2D error", plot_title="error_2d", x_label="Frames", y_label="Error")
    #simple_plot(pose_client.error_3d[:pose_client.CALIBRATION_LENGTH], file_manager.estimate_folder_name, "3D error", plot_title="calib_error_3d", x_label="Frames", y_label="Error")    
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
        #if airsim_client.is_using_airsim:
            #k = cv2.waitKey(1) & 0xFF
            #if k == 27:
            #    break        

        photo_loc = file_manager.get_photo_loc(airsim_client.linecount)
        take_photo(airsim_client, pose_client, current_state, file_manager)

        determine_calibration_mode(airsim_client.linecount, pose_client)

        determine_positions(airsim_client.linecount, pose_client, current_state, file_manager.plot_loc, file_manager.get_photo_loc())

        cam_pitch = current_state.get_required_pitch()
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(cam_pitch, 0, 0))

        potential_states_fetcher.reset(pose_client, airsim_client, current_state)

        trajectory = potential_states_fetcher.trajectory 
        if airsim_client.linecount < pose_client.PRECALIBRATION_LENGTH:
            goal_state = potential_states_fetcher.precalibration()
            drone_speed = TOP_SPEED
        else:
            if (trajectory == "active"):
                potential_states_fetcher.get_potential_positions_really_spherical_future()
                potential_states_fetcher.find_hessians_for_potential_states(pose_client)
                goal_state, _ = potential_states_fetcher.find_best_potential_state()
                potential_states_fetcher.plot_everything(airsim_client.linecount, file_manager.plot_loc, photo_loc)
            if (trajectory == "constant_rotation"):
                goal_state = potential_states_fetcher.constant_rotation_baseline_future()
            if (trajectory == "random"): #RANDOM
                potential_states_fetcher.get_potential_positions_really_spherical_future()
                goal_state = potential_states_fetcher.find_random_next_state()
            if (trajectory == "wobbly_rotation"):
                goal_state = potential_states_fetcher.wobbly_baseline()
            if (trajectory == "updown"):
                goal_state = potential_states_fetcher.up_down_baseline()
            if (trajectory == "leftright"):
                goal_state = potential_states_fetcher.left_right_baseline()
            drone_speed = TOP_SPEED

        desired_pos, desired_yaw_deg, _ = goal_state.get_goal_pos_yaw_pitch(current_state.drone_orientation_gt)

        #find desired drone speed
        #desired_vel = 5#(desired_pos - current_state.drone_pos)/TIME_HORIZON #how much the drone will have to move for this iteration
        #drone_speed = np.linalg.norm(desired_vel)    
        if (airsim_client.linecount < 5):
            drone_speed = drone_speed * airsim_client.linecount/5

        airsim_client.simPauseDrone(False)
        #add if here for calib
        if airsim_client.is_using_airsim:
            start_move = time.time()
            airsim_client.moveToPositionAsync(desired_pos[0], desired_pos[1], desired_pos[2], drone_speed, 
                                              DELTA_T, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(is_rate=False, 
                                              yaw_or_rate=desired_yaw_deg), lookahead=-1, adaptive_lookahead=0).join()
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
        plot_drone_traj(pose_client, file_manager.plot_loc, airsim_client.linecount,  pose_client.animation)
    
        file_manager.save_simulation_values(airsim_client.linecount)

        airsim_client.linecount += 1
        print('linecount', airsim_client.linecount)

        if airsim_client.is_using_airsim:
            if (airsim_client.linecount == airsim_client.length_of_simulation):
                end_test = True
        else:
            end_test = airsim_client.end
            

def precalibration(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager):
    airsim_client.simPauseHuman(True)
    airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)

    for _ in range(pose_client.PRECALIBRATION_LENGTH):
        photo_loc = file_manager.get_photo_loc(airsim_client.linecount)
        take_photo(airsim_client, pose_client, current_state, file_manager)

        determine_positions(airsim_client.linecount, pose_client, current_state, file_manager)

        cam_pitch = current_state.get_required_pitch()
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(cam_pitch, 0, 0))
        current_state.cam_pitch = cam_pitch

        potential_states_fetcher.reset(pose_client, airsim_client, current_state)
        goal_state = potential_states_fetcher.precalibration()

        desired_pos, desired_yaw_deg, _ = goal_state.get_goal_pos_yaw_pitch(current_state.drone_orientation_gt)
        
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


        plot_drone_traj(pose_client, file_manager.plot_loc, airsim_client.linecount,  pose_client.animation)
    
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
            airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
            potential_states_fetcher.reset(pose_client, airsim_client, current_state)
            potential_states_fetcher.dome_experiment()

            for sample_ind in range(potential_states_fetcher.number_of_samples):
                photo_loc = file_manager.get_photo_loc(airsim_client.linecount)
                goal_state = potential_states_fetcher.potential_states_try[sample_ind]

                sim_pos = goal_state.position

                airsim_client.simPauseDrone(False)
                airsim_client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(sim_pos[0],sim_pos[1],sim_pos[2]), airsim.to_quaternion(0, 0, goal_state.orientation)), True)
                airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(goal_state['pitch'], 0, 0))
                current_state.cam_pitch = goal_state.pitch
                
                take_photo(airsim_client, pose_client, current_state, file_manager)
                airsim_client.simPauseDrone(True)
                
                determine_openpose_error(airsim_client.linecount, pose_client, current_state, file_manager)

                #SAVE ALL VALUES OF THIS SIMULATION
                file_manager.append_openpose_error(pose_client.openpose_error, pose_client.openpose_arm_error,  pose_client.openpose_leg_error)
                #file_manager.save_simulation_values(airsim_client, pose_client)

                airsim_client.linecount += 1
                #print('linecount', airsim_client.linecount)

            #print("WRITING ERROR NOW!")
            file_manager.write_openpose_error(current_state.bone_pos_gt)
            plot_drone_traj(pose_client, file_manager.plot_loc, airsim_client.linecount,  pose_client.animation)
            pose_client.calib_res_list.clear()

            #implement a human pause function in airsim
            airsim_client.simPauseHuman(False)
            time.sleep(0.3)
            airsim_client.simPauseHuman(True)
    date_time_name = time.strftime("%Y-%m-%d-%H-%M")
    print("experiment ended at:", date_time_name)

def teleport_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager, loop_mode, parameters):
    date_time_name = time.strftime("%Y-%m-%d-%H-%M")
    print("experiment began at:", date_time_name)
    airsim_client.simPauseDrone(False)
    potential_error_finder = Potential_Error_Finder(parameters)

    airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
    #potential_error_finder.init_3d_pose(current_state.bone_pos_gt)

    potential_states_fetcher.reset(pose_client, airsim_client, current_state)
    potential_states_fetcher.dome_experiment()
    # potential_states_fetcher.get_potential_positions_really_spherical_future()

    #START AT POSITION 0
    goal_state = potential_states_fetcher.potential_states_try[0]
    set_position(goal_state, airsim_client, current_state, loop_mode="teleport")

    take_photo(airsim_client, pose_client, current_state, file_manager, 0)
    initialize_empty_frames(airsim_client.linecount, pose_client, current_state, file_manager.plot_loc, file_manager.get_photo_loc())
    airsim_client.linecount += 1

    while airsim_client.linecount < airsim_client.length_of_simulation:    
        start1 = time.time()   
        if potential_error_finder.find_best_traj: #/and exp_ind >= predefined_traj_len:
            for state_ind in range(0, len(potential_states_fetcher.potential_states_try)):
                goal_state = potential_states_fetcher.potential_states_try[state_ind]
                set_position(goal_state, airsim_client, current_state, loop_mode="teleport")
                take_photo(airsim_client, pose_client, current_state, file_manager, state_ind)
                for trial_ind in range(potential_error_finder.num_of_noise_trials):
                    pose_client_copy = pose_client.deepcopy_PEC(trial_ind)
                    state_copy = current_state.deepcopy_state()
                    determine_positions(airsim_client.linecount, pose_client_copy, state_copy, file_manager.plot_loc, file_manager.get_photo_loc())
                    potential_error_finder.append_error(trial_ind, pose_client_copy.adjusted_optimized_poses, pose_client_copy.poses_3d_gt)
                file_manager.write_error_values(potential_error_finder.frame_overall_error_list, airsim_client.linecount)
                potential_error_finder.record_noise_experiment_statistics(potential_states_fetcher, state_ind)
        end1 = time.time()
        print("Simulating errors for all locations took", end1-start1, "seconds")

        if airsim_client.linecount < potential_error_finder.predefined_traj_len:
            goal_state = potential_states_fetcher.find_next_state_constant_rotation(airsim_client.linecount)    
        else:
            if potential_states_fetcher.trajectory == "active":
                start2 = time.time()
                potential_states_fetcher.find_hessians_for_potential_states(pose_client)
                goal_state, _ = potential_states_fetcher.find_best_potential_state()    
                end2 = time.time()
                print("Finding hessians and best potential state took", end2- start2)
                if potential_error_finder.find_best_traj:
                    potential_error_finder.find_correlations(potential_states_fetcher)
                    file_manager.write_correlation_values(airsim_client.linecount, potential_error_finder.correlation_current, potential_error_finder.cosine_current)
                    #plot_correlations(potential_error_finder, airsim_client.linecount, file_manager.plot_loc)

                potential_states_fetcher.plot_everything(airsim_client.linecount, file_manager, potential_error_finder.find_best_traj)
            elif potential_states_fetcher.trajectory == "constant_rotation":
                goal_state = potential_states_fetcher.find_next_state_constant_rotation(airsim_client.linecount)    
            elif potential_states_fetcher.trajectory == "constant_angle":
                goal_state = potential_states_fetcher.choose_state(0)   
            elif potential_states_fetcher.trajectory == "random":
                goal_state = potential_states_fetcher.find_random_next_state()    
            elif potential_states_fetcher.trajectory == "go_to_best":
                best_index = np.argmin(potential_states_fetcher.overall_error_mean_list)
                goal_state = potential_states_fetcher.choose_state(best_index)   
            elif potential_states_fetcher.trajectory == "go_to_worst":
                worst_index = np.argmax(potential_states_fetcher.overall_error_mean_list)
                goal_state = potential_states_fetcher.choose_state(worst_index)

            file_manager.write_uncertainty_values(potential_states_fetcher.uncertainty_list_whole, airsim_client.linecount)
        
        if airsim_client.linecount > pose_client.ONLINE_WINDOW_SIZE and potential_error_finder.find_best_traj:
            potential_error_finder.find_average_error_over_trials(goal_state.index)
            file_manager.write_average_error_over_trials(airsim_client.linecount, potential_error_finder)

        start3 = time.time()
        #set_position(goal_state, airsim_client, current_state, loop_mode=loop_mode)
        set_position(goal_state, airsim_client, current_state, loop_mode="teleport")

        take_photo(airsim_client, pose_client, current_state, file_manager, goal_state.index)
        #pose_client_sim.adjust_3d_pose(current_state, pose_client)

        determine_positions(airsim_client.linecount, pose_client, current_state, file_manager.plot_loc, file_manager.get_photo_loc())
        end3 = time.time()
        print("Finding pose at chosen loc took", end3-start3)

        airsim_client.linecount += 1
        print('linecount', airsim_client.linecount)
        if not pose_client.quiet:
            plot_drone_traj(pose_client, file_manager.plot_loc, airsim_client.linecount, pose_client.animation)

        if not pose_client.isCalibratingEnergy:
            airsim_client.simPauseHuman(False)
            if airsim_client.is_using_airsim:
                time.sleep(DELTA_T)
            airsim_client.simPauseHuman(True)
            #pose_client_sim.update_internal_3d_pose()

        potential_states_fetcher.reset(pose_client, airsim_client, current_state)
        if loop_mode == "teleport":
            potential_states_fetcher.dome_experiment()
        elif loop_mode == "normal":
            potential_states_fetcher.get_potential_positions_really_spherical_future()
        end4 = time.time()
        print("One iter took", end4-start1)


def create_test_set(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager):
    date_time_name = time.strftime("%Y-%m-%d-%H-%M")
    print("experiment began at:", date_time_name)
    airsim_client.simPauseDrone(False)

    for _ in range(200):        
        airsim_client.simPauseHuman(False)
        time.sleep(DELTA_T)
        airsim_client.simPauseHuman(True)

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
            openpose_res, liftnet_res = determine_openpose_error(airsim_client.linecount, pose_client, current_state, file_manager.plot_loc, file_manager.get_photo_loc())
            file_manager.prepare_test_set(current_state, openpose_res, liftnet_res, airsim_client.linecount, state_ind)

        airsim_client.linecount += 1
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

def set_position(goal_state, airsim_client, current_state, loop_mode):
    if loop_mode == "teleport":
        airsim_client.simSetVehiclePose(goal_state)
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(goal_state.pitch, 0, 0))
        current_state.cam_pitch = goal_state.pitch

    elif loop_mode == "normal":
        if (airsim_client.linecount < 5):
            drone_speed = TOP_SPEED * airsim_client.linecount/5
        desired_pos, desired_yaw_deg, _ = goal_state.get_goal_pos_yaw_pitch(current_state.drone_orientation_gt)
        start_move = time.time()
        airsim_client.moveToPositionAsync(desired_pos[0], desired_pos[1], desired_pos[2], 
                                          drone_speed, DELTA_T, airsim.DrivetrainType.MaxDegreeOfFreedom, 
                                          airsim.YawMode(is_rate=False, yaw_or_rate=desired_yaw_deg), lookahead=-1, adaptive_lookahead=0).join()
        end_move = time.time()
        time_passed = end_move - start_move
        if (DELTA_T > time_passed):
            time.sleep(DELTA_T-time_passed)

        cam_pitch = current_state.get_required_pitch()
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(cam_pitch, 0, 0))
        current_state.cam_pitch = cam_pitch