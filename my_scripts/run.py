from helpers import * 
from NonAirSimClient import *
from PoseEstimationClient import *
from pose3d_optimizer import *
from project_bones import *
from determine_positions import *
from PotentialStatesFetcher import PotentialStatesFetcher, THETA_LIST, PHI_LIST
from State import State, TOP_SPEED, TIME_HORIZON, DELTA_T
from file_manager import FileManager

import pprint
import os

gt_hv = []
est_hv = []
USE_AIRSIM = True
LENGTH_OF_SIMULATION = 200
photo_time = 0

def get_client_unreal_values(client, X):
    unreal_positions = np.zeros([5,3])
    if (USE_AIRSIM):
        keys = {'humanPos': HUMAN_POS_IND, 'dronePos' : DRONE_POS_IND, 'droneOrient': DRONE_ORIENTATION_IND, 'left_arm': L_SHOULDER_IND, 'right_arm': R_SHOULDER_IND}
        for key, value in keys.items():
            element = X[key]
            if (key != 'droneOrient'):
                unreal_positions[value, :] = np.array([element.x_val, element.y_val, -element.z_val])
                unreal_positions[value, :]  = (unreal_positions[value, :] - client.DRONE_INITIAL_POS)/100
            else:
                unreal_positions[value, :] = np.array([element.x_val, element.y_val, element.z_val])
    else:
        do_nothing()
    return unreal_positions

def take_photo(airsim_client, pose_client, image_folder_loc):
    if USE_AIRSIM:
        response = airsim_client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)])

        airsim_client.simPause(True)
        response = response[0]
        X = response.bones  
        gt_numbers = vector3r_arr_to_dict(X)
        unreal_positions = get_client_unreal_values(airsim_client, gt_numbers)
        gt_str = ""
        bone_pos = np.zeros([3, len(gt_numbers)-3])
        DRONE_INITIAL_POS = airsim_client.DRONE_INITIAL_POS
        for bone_ind, bone_i in enumerate(attributes):
            gt_str = gt_str + str(gt_numbers[bone_i].x_val) + '\t' + str(gt_numbers[bone_i].y_val) + '\t' +  str(gt_numbers[bone_i].z_val) + '\t'
            if (bone_ind >= 3):
                bone_pos[:, bone_ind-3] = np.array([gt_numbers[bone_i].x_val, gt_numbers[bone_i].y_val, -gt_numbers[bone_i].z_val]) - DRONE_INITIAL_POS
        bone_pos = bone_pos / 100

        multirotor_state = airsim_client.getMultirotorState()
        estimated_state =  multirotor_state.kinematics_estimated
        drone_pos = estimated_state.position
        #drone_orient = airsim.to_eularian_angles(estimated_state.orientation)
        #CHANGE THIS
        drone_orient = unreal_positions[DRONE_ORIENTATION_IND]

        airsim_client.updateSynchronizedData(unreal_positions, bone_pos, drone_pos, drone_orient)

        #img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        #img_rgba = img1d.reshape(response.height, response.width, 4)  
        #airsim_client.photo = img_rgba[:,:,:,0:3]
        
        loc = image_folder_loc + '/img_' + str(airsim_client.linecount) + '.png'
        airsim.write_file(os.path.normpath(loc), response.image_data_uint8)

    else:
        response = airsim_client.simGetImages()
        bone_pos = response.bone_pos
        unreal_positions = response.unreal_positions
        drone_orient = airsim_client.getPitchRollYaw()
        drone_pos = airsim_client.getPosition()
        airsim_client.updateSynchronizedData(unreal_positions, bone_pos, drone_pos, drone_orient)
        gt_str = ""
    pose_client.f_groundtruth_str = gt_str

    return response.image_data_uint8

def determine_calibration_mode(airsim_client, pose_client, COMPUTER_VISION_MODE):
    if not COMPUTER_VISION_MODE:
        if (airsim_client.linecount == pose_client.CALIBRATION_LENGTH):
            #client.switch_energy(energy_mode[cv2.getTrackbarPos('Calibration mode', 'Calibration for 3d pose')])
            airsim_client.changeCalibrationMode(False)
            pose_client.changeCalibrationMode(False)

def run_simulation_trial(kalman_arguments, parameters, energy_parameters, active_parameters):

    errors_pos = []
    errors_vel = []
    errors = {}
    end_test = False

    file_manager = FileManager(parameters)   

    USE_TRACKBAR = parameters["USE_TRACKBAR"]
    global USE_AIRSIM
    USE_AIRSIM = parameters["USE_AIRSIM"]
    trajectory = active_parameters["TRAJECTORY"]
    z_pos = active_parameters["Z_POS"]
    COMPUTER_VISION_MODE = active_parameters["COMPUTER_VISION_MODE"]
    #connect to the AirSim simulator
    if USE_AIRSIM:
        airsim_client = airsim.MultirotorClient()
        airsim_client.confirmConnection()
        airsim_client.enableApiControl(True)
        airsim_client.armDisarm(True)
        print('Taking off')
        airsim_client.initInitialDronePos()
        airsim_client.changeAnimation(ANIM_TO_UNREAL[file_manager.anim_num])
        airsim_client.changeCalibrationMode(True)
        airsim_client.takeoffAsync(timeout_sec = 20).join()
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(CAMERA_PITCH_OFFSET, 0, 0))
        time.sleep(2)
        #airsim_client.moveToZAsync(z_pos, 10, timeout_sec = 20).join()
        #time.sleep(60)
    else:
        airsim_client = NonAirSimClient(file_manager.get_nonairsim_client_names())
    #pause airsim until we set stuff up 
    airsim_client.simPause(True)

    pose_client = PoseEstimationClient(energy_parameters,  Crop(openpose_test = COMPUTER_VISION_MODE))
    current_state = State()
    potential_states_fetcher = PotentialStatesFetcher(pose_client, active_parameters)
    
    file_manager.save_initial_drone_pos(airsim_client)

    #shoulder_vector = initial_positions[R_SHOULDER_IND, :] - initial_positions[L_SHOULDER_IND, :] #find initial human orientation!
    #INITIAL_HUMAN_ORIENTATION = np.arctan2(-shoulder_vector[0], shoulder_vector[1]) #in unreal coordinates

    if USE_TRACKBAR:
        # create trackbars for angle change
        cv2.namedWindow('Drone Control')
        cv2.createTrackbar('Angle','Drone Control', 0, 360, do_nothing)
        #cv2.setTrackbarPos('Angle', 'Angle Control', int(degrees(some_angle-INITIAL_HUMAN_ORIENTATION)))
        #cv2.setTrackbarPos('Angle', 'Drone Control', int(degrees(current_state.some_angle)))

        cv2.createTrackbar('Radius','Drone Control', 3, 10, do_nothing)
        cv2.setTrackbarPos('Radius', 'Drone Control', int(current_state.radius))

        cv2.createTrackbar('Z','Drone Control', 3, 20, do_nothing)
        cv2.setTrackbarPos('Z', 'Drone Control', z_pos)

    #if (pose_client.modes["mode_3d"] == 3 and USE_AIRSIM):
    #    cv2.namedWindow('Calibration for 3d pose')
    #    cv2.createTrackbar('Calibration mode','Calibration for 3d pose', 0, 1, do_nothing)
    #    cv2.setTrackbarPos('Calibration mode','Calibration for 3d pose', 1)

    airsim_client.simPause(False)

################
    if (not COMPUTER_VISION_MODE):
        normal_simulation_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager)
    else:
        openpose_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager)
################

    #calculate errors
    airsim_client.simPause(True)
    errors["ave_3d_err"] = sum(pose_client.error_3d)/len(pose_client.error_3d)
    errors["middle_3d_err"] = sum(pose_client.middle_pose_error)/len(pose_client.middle_pose_error)

    estimate_folder_name = foldernames_anim["estimates"]
    simple_plot(pose_client.processing_time, estimate_folder_name, "processing_time", plot_title="Processing Time", x_label="Frames", y_label="Time")
    if (pose_client.modes["mode_3d"] == 3):
        simple_plot(pose_client.error_2d, estimate_folder_name, "2D error", plot_title="error_2d", x_label="Frames", y_label="Error")
    simple_plot(pose_client.error_3d[:pose_client.CALIBRATION_LENGTH], estimate_folder_name, "3D error", plot_title="calib_error_3d", x_label="Frames", y_label="Error")    
    #simple_plot(pose_client.error_3d[pose_client.CALIBRATION_LENGTH:], estimate_folder_name, "3D error", plot_title="online_error_3d", x_label="Frames", y_label="Error")
    
    print('End it!')
    airsim_client.reset()
    pose_client.reset(file_manager.plot_loc)
    file_manager.close_files()

    return errors

def normal_simulation_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager):
    end_test = False
    while (not end_test):
        #if USE_AIRSIM:
            #k = cv2.waitKey(1) & 0xFF
            #if k == 27:
            #    break        

        photo_loc = file_manager.get_photo_loc(airsim_client.linecount, USE_AIRSIM)

        airsim_client.simPause(False)
        take_photo(airsim_client, pose_client, file_manager.take_photo_loc)
        airsim_client.simPause(True)

        determine_calibration_mode(airsim_client, pose_client, False)

        positions, unreal_positions = determine_all_positions(airsim_client, pose_client, plot_loc=file_manager.plot_loc, photo_loc=photo_loc)

        current_state.updateState(positions) #updates human pos, human orientation, human vel, drone pos
        cam_pitch = current_state.get_current_pitch()
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(cam_pitch, 0, 0))

        potential_states_fetcher.reset(pose_client, current_state.drone_pos)
        
        if airsim_client.linecount < 20:
            goal_state = potential_states_fetcher.precalibration()
            drone_speed = TOP_SPEED
        else:
            if (trajectory == 0):
                potential_states_fetcher.get_potential_positions_really_spherical_future()
                potential_states_fetcher.find_hessians_for_potential_states(pose_client, pose_client.P_world)
                goal_state = potential_states_fetcher.find_best_potential_state()
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

        airsim_client.simPause(False)
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

        airsim_client.simPause(True)

        pose_client.cam_pitch = cam_pitch
        plot_drone_traj(pose_client, plot_loc_, airsim_client.linecount)
    
        file_manager.save_simulation_values(airsim_client, pose_client)

        airsim_client.linecount += 1
        print('linecount', airsim_client.linecount)

        if (not USE_AIRSIM):
            end_test = airsim_client.end
        else:
            if (airsim_client.linecount == LENGTH_OF_SIMULATION):
                end_test = True
       
        airsim_client.simPause(False)

def openpose_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager):
    for different_pose_ind in range(1): 
        end_test = False
        while (not end_test):
            photo_loc = file_manager.get_photo_loc(airsim_client.linecount, USE_AIRSIM)

            potential_states_fetcher.reset(pose_client, current_state.drone_pos)
            goal_state, openpose_str = potential_states_fetcher.test_openpose_mode()

            pose_client.f_openpose_str = openpose_str
            sim_pos = goal_state['position']
            airsim_client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(sim_pos[0],sim_pos[1],sim_pos[2]), airsim.to_quaternion(0, 0, goal_state["orientation"])), False)
            take_photo(airsim_client, pose_client, file_manager.take_photo_loc)
            airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(goal_state['pitch'], 0, 0))
            pose_client.cam_pitch = goal_state['pitch']
            
            determine_openpose_error(airsim_client, pose_client, plot_loc = file_manager.plot_loc, photo_loc = photo_loc)
            file_manager.write_openpose_error(pose_client.f_openpose_str)

            airsim_client.simPause(True)
            plot_drone_traj(pose_client, file_manager.plot_loc, airsim_client.linecount)

            #SAVE ALL VALUES OF THIS SIMULATION
            file_manager.save_simulation_values(airsim_client, pose_client)

            airsim_client.linecount += 1
            print('linecount', airsim_client.linecount)

            if  (airsim_client.linecount == len(THETA_LIST)*len(PHI_LIST)):
                end_test = True
            airsim_client.simPause(False)

        #implement a human pause function in airsim
        airsim_client.simPauseHuman(False)
        time.sleep(1)
        airsim_client.simPauseHuman(True)
        #airsim_client.changeAnimation(ANIM_TO_UNREAL[ANIMATION_NUM])


###### REMOVE
def run_simulation(kalman_arguments, parameters, energy_parameters, active_parameters):

    errors_pos = []
    errors_vel = []
    errors = {}
    end_test = False
       
    USE_TRACKBAR = parameters["USE_TRACKBAR"]
    global USE_AIRSIM
    USE_AIRSIM = parameters["USE_AIRSIM"]
    ANIMATION_NUM = parameters["ANIMATION_NUM"]
    experiment_name = parameters["EXPERIMENT_NAME"]
    test_set_name = parameters["TEST_SET_NAME"]
    file_names = parameters["FILE_NAMES"]
    folder_names = parameters["FOLDER_NAMES"]
    trajectory = active_parameters["TRAJECTORY"]
    z_pos = active_parameters["Z_POS"]
    COMPUTER_VISION_MODE = active_parameters["COMPUTER_VISION_MODE"]
    #connect to the AirSim simulator
    if USE_AIRSIM:
        airsim_client = airsim.MultirotorClient()
        airsim_client.confirmConnection()
        airsim_client.enableApiControl(True)
        airsim_client.armDisarm(True)
        print('Taking off')
        airsim_client.initInitialDronePos()
        airsim_client.changeAnimation(ANIM_TO_UNREAL[ANIMATION_NUM])
        airsim_client.changeCalibrationMode(True)
        airsim_client.takeoffAsync(timeout_sec = 20).join()
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(CAMERA_PITCH_OFFSET, 0, 0))
        time.sleep(2)
        #airsim_client.moveToZAsync(z_pos, 10, timeout_sec = 20).join()
        #time.sleep(60)
    else:
        
        filename_bones = 'test_sets/'+test_set_name+'/groundtruth.txt'
        filename_output = 'test_sets/'+test_set_name+'/a_flight.txt'
        airsim_client = NonAirSimClient(filename_bones, filename_output)

    pose_client = PoseEstimationClient(energy_parameters,  Crop(openpose_test = COMPUTER_VISION_MODE))

    #define some variables
    airsim_client.linecount = 0
    gt_hp = []
    est_hp = []
    f_output, f_groundtruth, f_reconstruction, f_openpose_error = open_files(file_names, experiment_name)
    filenames_anim = file_names[experiment_name]


    foldernames_anim = folder_names[experiment_name]
    plot_loc_ = foldernames_anim["superimposed_images"]

    #save drone initial position
    f_groundtruth_prefix = "-1\t" + str(airsim_client.DRONE_INITIAL_POS[0,]) + "\t" + str(airsim_client.DRONE_INITIAL_POS[1,]) + "\t" + str(airsim_client.DRONE_INITIAL_POS[2,])
    for i in range(0,70):
        f_groundtruth_prefix = f_groundtruth_prefix + "\t"
    f_groundtruth.write(f_groundtruth_prefix + "\n")
    airsim_client.simPause(True)
    current_state = State()
    potential_states_fetcher = PotentialStatesFetcher(pose_client, active_parameters)

    #shoulder_vector = initial_positions[R_SHOULDER_IND, :] - initial_positions[L_SHOULDER_IND, :] #find initial human orientation!
    #INITIAL_HUMAN_ORIENTATION = np.arctan2(-shoulder_vector[0], shoulder_vector[1]) #in unreal coordinates

    if USE_TRACKBAR:
        # create trackbars for angle change
        cv2.namedWindow('Drone Control')
        cv2.createTrackbar('Angle','Drone Control', 0, 360, do_nothing)
        #cv2.setTrackbarPos('Angle', 'Angle Control', int(degrees(some_angle-INITIAL_HUMAN_ORIENTATION)))
        #cv2.setTrackbarPos('Angle', 'Drone Control', int(degrees(current_state.some_angle)))

        cv2.createTrackbar('Radius','Drone Control', 3, 10, do_nothing)
        cv2.setTrackbarPos('Radius', 'Drone Control', int(current_state.radius))

        cv2.createTrackbar('Z','Drone Control', 3, 20, do_nothing)
        cv2.setTrackbarPos('Z', 'Drone Control', z_pos)

    #if (pose_client.modes["mode_3d"] == 3 and USE_AIRSIM):
    #    cv2.namedWindow('Calibration for 3d pose')
    #    cv2.createTrackbar('Calibration mode','Calibration for 3d pose', 0, 1, do_nothing)
    #    cv2.setTrackbarPos('Calibration mode','Calibration for 3d pose', 1)
    airsim_client.simPause(False)

    while (not end_test):
        #if USE_AIRSIM:
            #k = cv2.waitKey(1) & 0xFF
            #if k == 27:
            #    break        

        if (USE_AIRSIM):
            photo_loc_ = foldernames_anim["images"] + '/img_' + str(airsim_client.linecount) + '.png'
        else:
            photo_loc_ = 'test_sets/'+test_set_name+'/images/img_' + str(airsim_client.linecount) + '.png'

        take_photo(airsim_client, pose_client, foldernames_anim["images"])
        airsim_client.simPause(True)
    
        if (airsim_client.linecount == pose_client.CALIBRATION_LENGTH):
            #client.switch_energy(energy_mode[cv2.getTrackbarPos('Calibration mode', 'Calibration for 3d pose')])
            airsim_client.changeCalibrationMode(False)
            pose_client.changeCalibrationMode(False)
            #global_plot_ind =0

        positions, unreal_positions = determine_all_positions(airsim_client, pose_client, plot_loc = plot_loc_, photo_loc = photo_loc_)

        if COMPUTER_VISION_MODE: 
            f_openpose_error.write(pose_client.f_openpose_str)

        current_state.updateState(positions) #updates human pos, human orientation, human vel, drone pos
        cam_pitch = current_state.get_current_pitch()
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(cam_pitch, 0, 0))

        #finds desired position and yaw angle
        #if (USE_TRACKBAR):
          #  [desired_pos, desired_yaw_deg] = current_state.get_desired_pos_and_yaw_trackbar()

        potential_states_fetcher.reset(pose_client, current_state.drone_pos)
        
        if airsim_client.linecount < 20:
            goal_state = potential_states_fetcher.precalibration()
            drone_speed = TOP_SPEED
        else:
            if (trajectory == 0):
                potential_states_fetcher.get_potential_positions_really_spherical_future()
                potential_states_fetcher.find_hessians_for_potential_states(pose_client, pose_client.P_world)
                goal_state = potential_states_fetcher.find_best_potential_state()
                potential_states_fetcher.plot_everything(airsim_client.linecount, plot_loc_, photo_loc_)
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

        airsim_client.simPause(False)
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

        airsim_client.simPause(True)

        pose_client.cam_pitch = cam_pitch
        plot_drone_traj(pose_client, plot_loc_, airsim_client.linecount)
    
        #SAVE ALL VALUES OF THIS SIMULATION
        gt_hp.append(unreal_positions[HUMAN_POS_IND, :])
        est_hp.append(current_state.human_pos)
        errors_pos.append(np.linalg.norm(unreal_positions[HUMAN_POS_IND, :]-current_state.human_pos))
        if (airsim_client.linecount > 0):
            gt_hv.append((gt_hp[-1]-gt_hp[-2])/DELTA_T)
            est_hv.append(current_state.human_vel)
            errors_vel.append(np.linalg.norm( (gt_hp[-1]-gt_hp[-2])/DELTA_T - current_state.human_vel))

        save_simulation_values(f_output, f_reconstruction, f_groundtruth, f_groundtruth_str, airsim_client, pose_client)

        airsim_client.linecount += 1
        print('linecount', airsim_client.linecount)

        if (not USE_AIRSIM):
            end_test = airsim_client.end
        else:
            if (airsim_client.linecount == LENGTH_OF_SIMULATION):
                end_test = True
       
        airsim_client.simPause(False)

    #calculate errors
    airsim_client.simPause(True)
    errors["ave_3d_err"] = sum(pose_client.error_3d)/len(pose_client.error_3d)
    errors["middle_3d_err"] = sum(pose_client.middle_pose_error)/len(pose_client.middle_pose_error)

    estimate_folder_name = foldernames_anim["estimates"]
    #plot_error(np.array(gt_hp), np.array(est_hp), np.array(gt_hv), np.array(est_hv), errors, estimate_folder_name)
    simple_plot(pose_client.processing_time, estimate_folder_name, "processing_time", plot_title="Processing Time", x_label="Frames", y_label="Time")
    if (pose_client.modes["mode_3d"] == 3):
        simple_plot(pose_client.error_2d, estimate_folder_name, "2D error", plot_title="error_2d", x_label="Frames", y_label="Error")
    simple_plot(pose_client.error_3d[:pose_client.CALIBRATION_LENGTH], estimate_folder_name, "3D error", plot_title="calib_error_3d", x_label="Frames", y_label="Error")    
    #simple_plot(pose_client.error_3d[pose_client.CALIBRATION_LENGTH:], estimate_folder_name, "3D error", plot_title="online_error_3d", x_label="Frames", y_label="Error")
    
    print('End it!')
    f_groundtruth.close()
    f_output.close()
    f_reconstruction.close()
    f_openpose_error.close()
    airsim_client.simPause(False)

    airsim_client.reset()
    pose_client.reset(plot_loc_)

    return errors


def run_openpose_test(parameters, energy_parameters, active_parameters):
    end_test = False
       
    ANIMATION_NUM = parameters["ANIMATION_NUM"]
    experiment_name = parameters["EXPERIMENT_NAME"]
    file_names = parameters["FILE_NAMES"]
    folder_names = parameters["FOLDER_NAMES"]
    #connect to the AirSim simulator

    airsim_client = airsim.MultirotorClient()
    airsim_client.confirmConnection()
    airsim_client.enableApiControl(True)
    airsim_client.armDisarm(True)
    print('Taking off')
    airsim_client.initInitialDronePos()
    airsim_client.changeAnimation(ANIM_TO_UNREAL[ANIMATION_NUM])
    airsim_client.simPauseHuman(True)
    airsim_client.takeoffAsync(timeout_sec = 20).join()
    time.sleep(2)

    airsim_client.simPause(True)
    pose_client = PoseEstimationClient(energy_parameters,  Crop(openpose_test = True))

    #define some variables
    airsim_client.linecount = 0
    f_output, f_groundtruth, f_reconstruction, f_openpose_error = open_files(file_names, experiment_name)
    
    foldernames_anim = folder_names[experiment_name]    
    plot_loc_ = foldernames_anim["superimposed_images"]

    #save drone initial position
    f_groundtruth_prefix = "-1\t" + str(airsim_client.DRONE_INITIAL_POS[0,]) + "\t" + str(airsim_client.DRONE_INITIAL_POS[1,]) + "\t" + str(airsim_client.DRONE_INITIAL_POS[2,])
    for i in range(0,70):
        f_groundtruth_prefix = f_groundtruth_prefix + "\t"
    f_groundtruth.write(f_groundtruth_prefix + "\n")

    current_state = State()
    potential_states_fetcher = PotentialStatesFetcher(pose_client, active_parameters)
    airsim_client.simPause(False)

    for different_pose_ind in range(1): 
        while (not end_test):
            photo_loc_ = foldernames_anim["images"] + '/img_' + str(airsim_client.linecount) + '.png'

            potential_states_fetcher.reset(pose_client, current_state.drone_pos)
            goal_state, openpose_str = potential_states_fetcher.test_openpose_mode()

            pose_client.f_openpose_str = openpose_str
            sim_pos = goal_state['position']
            airsim_client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(sim_pos[0],sim_pos[1],sim_pos[2]), airsim.to_quaternion(0, 0, goal_state["orientation"])), False)
            take_photo(airsim_client, pose_client, foldernames_anim["images"])
            airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(goal_state['pitch'], 0, 0))
            pose_client.cam_pitch = goal_state['pitch']
            
            determine_openpose_error(airsim_client, pose_client, plot_loc = plot_loc_, photo_loc = photo_loc_)

            f_openpose_error.write(pose_client.f_openpose_str)

            airsim_client.simPause(True)
            plot_drone_traj(pose_client, plot_loc_, airsim_client.linecount)

            #SAVE ALL VALUES OF THIS SIMULATION
            save_simulation_values(f_output, f_reconstruction, f_groundtruth, airsim_client, pose_client)

            airsim_client.linecount += 1
            print('linecount', airsim_client.linecount)

            if  (airsim_client.linecount == len(THETA_LIST)*len(PHI_LIST)):
                end_test = True
            airsim_client.simPause(False)

        #implement a human pause function in airsim
        airsim_client.simPauseHuman(False)
        time.sleep(1)
        airsim_client.simPauseHuman(True)
        #airsim_client.changeAnimation(ANIM_TO_UNREAL[ANIMATION_NUM])
        

    #calculate errors
    airsim_client.simPause(True)

    print('End it!')
    f_groundtruth.close()
    f_output.close()
    f_reconstruction.close()
    f_openpose_error.close()
    airsim_client.simPause(False)

    airsim_client.reset()
    pose_client.reset(plot_loc_)
