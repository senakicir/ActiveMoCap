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

def get_client_gt_values(airsim_client, pose_client, X):
    drone_oriention_gt = np.array([X['droneOrient'].x_val, X['droneOrient'].y_val, X['droneOrient'].z_val])

    gt_str = "" 
    bone_pos_gt = np.zeros([3, 21])
    DRONE_INITIAL_POS = airsim_client.DRONE_INITIAL_POS
    bone_ind = 0
    for ind, bone_i in enumerate(attributes):
        gt_str = gt_str + str(bone_pos_gt[0, ind]) + '\t' + str(bone_pos_gt[1, ind]) + '\t' +  str(bone_pos_gt[2, ind]) + '\t'
        if (bone_i != 'dronePos' and bone_i != 'droneOrient' and bone_i != 'humanPos'):
            bone_pos_gt[:, bone_ind] = np.array([X[bone_i].x_val, X[bone_i].y_val, -X[bone_i].z_val]) - DRONE_INITIAL_POS
            bone_pos_gt[:, bone_ind] = bone_pos_gt[:, bone_ind]/100
            bone_ind += 1

    if pose_client.model == "mpi":
        bone_pos_gt = transform_gt_bones(bone_pos_gt)

    drone_pos_gt = np.array([X['dronePos'].x_val, X['dronePos'].y_val, -X['dronePos'].z_val])
    drone_pos_gt  = (drone_pos_gt - airsim_client.DRONE_INITIAL_POS)/100
    drone_pos_gt = drone_pos_gt[:, np.newaxis]

    pose_client.store_frame_parameters(drone_oriention_gt, bone_pos_gt, drone_pos_gt, gt_str)

def take_photo(airsim_client, pose_client, image_folder_loc):
    if USE_AIRSIM:
        response = airsim_client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)])

        airsim_client.simPauseDrone(True)
        airsim_client.simPauseHuman(True)
        response = response[0]
        gt_numbers = vector3r_arr_to_dict(response.bones)
        get_client_gt_values(airsim_client, pose_client, gt_numbers)

        multirotor_state = airsim_client.getMultirotorState()
        estimated_state =  multirotor_state.kinematics_estimated
        pose_client.drone_pos_est = estimated_state.position

        #img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        #img_rgba = img1d.reshape(response.height, response.width, 4)  
        #airsim_client.photo = img_rgba[:,:,:,0:3]
        
        loc = image_folder_loc + '/img_' + str(airsim_client.linecount) + '.png'
        airsim.write_file(os.path.normpath(loc), response.image_data_uint8)

    else:
        response = airsim_client.simGetImages()
        bone_pos_gt = response.bone_pos
        drone_oriention_gt, human_orientation_gt, drone_pos_gt = response.unreal_positions #fix this!
        pose_client.drone_pos_est = airsim_client.getPosition()
        pose_client.store_frame_parameters(drone_oriention_gt, human_orientation_gt, bone_pos_gt, drone_pos_gt, "")

    return response.image_data_uint8

def determine_calibration_mode(airsim_client, pose_client):
    if (airsim_client.linecount == pose_client.CALIBRATION_LENGTH):
        #client.switch_energy(energy_mode[cv2.getTrackbarPos('Calibration mode', 'Calibration for 3d pose')])
        pose_client.changeCalibrationMode(False)

def run_simulation_trial(kalman_arguments, parameters, energy_parameters, active_parameters):
    errors = {}

    file_manager = FileManager(parameters)   

    USE_TRACKBAR = parameters["USE_TRACKBAR"]
    global USE_AIRSIM
    USE_AIRSIM = parameters["USE_AIRSIM"]
    z_pos = active_parameters["Z_POS"]
    loop_mode = active_parameters["LOOP_MODE"]
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
        f1, f2 = file_manager.get_nonairsim_client_names()
        airsim_client = NonAirSimClient(f1, f2)
    #pause airsim until we set stuff up 
    airsim_client.simPause(True)

    pose_client = PoseEstimationClient(energy_parameters,  Crop(loop_mode = loop_mode))
    current_state = State()
    potential_states_fetcher = PotentialStatesFetcher(pose_client, active_parameters)
    
    file_manager.save_initial_drone_pos(airsim_client)

    #shoulder_vector = initial_positions[R_SHOULDER_IND, :] - initial_positions[L_SHOULDER_IND, :] #find initial human orientation!
    #INITIAL_HUMAN_ORIENTATION = np.arctan2(-shoulder_vector[0], shoulder_vector[1]) #in unreal coordinates

    if USE_TRACKBAR:
        create_trackbars(current_state.radius, z_pos)

################
    if loop_mode == 0:
        normal_simulation_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager)
    elif loop_mode == 1:
        openpose_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager)
    elif loop_mode == 2:
        dome_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager)

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

        take_photo(airsim_client, pose_client, file_manager.take_photo_loc)

        determine_calibration_mode(airsim_client, pose_client)

        determine_all_positions(airsim_client, pose_client, plot_loc=file_manager.plot_loc, photo_loc=photo_loc)

        current_state.updateState(pose_client) #updates human pos, human orientation, human vel, drone pos
        cam_pitch = current_state.get_current_pitch()
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(cam_pitch, 0, 0))

        potential_states_fetcher.reset(pose_client, current_state.drone_pos)

        trajectory = potential_states_fetcher.trajectory 
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
                print("i am here")
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

        print("desired_pos", desired_pos)
        print("current pos", current_state.drone_pos)
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

        pose_client.cam_pitch = cam_pitch
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
    for _ in range(20):
        photo_loc = file_manager.get_photo_loc(airsim_client.linecount, USE_AIRSIM)
        take_photo(airsim_client, pose_client, file_manager.take_photo_loc)

        determine_all_positions(airsim_client, pose_client, plot_loc=file_manager.plot_loc, photo_loc=photo_loc)

        current_state.updateState(pose_client) #updates human pos, human orientation, human vel, drone pos
        cam_pitch = current_state.get_current_pitch()
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(cam_pitch, 0, 0))

        potential_states_fetcher.reset(pose_client, current_state.drone_pos)

        goal_state = potential_states_fetcher.precalibration()
        drone_speed = TOP_SPEED

        desired_pos, desired_yaw_deg, _ = current_state.get_goal_pos_yaw_pitch(goal_state)

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

        pose_client.cam_pitch = cam_pitch
        plot_drone_traj(pose_client, file_manager.plot_loc, airsim_client.linecount)
    
        file_manager.save_simulation_values(airsim_client, pose_client)

        airsim_client.linecount += 1
        print('linecount', airsim_client.linecount)


       
def openpose_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager):
    animations_to_test = ["05_08", "38_03", "64_06", "02_01"]
    file_manager.write_openpose_prefix(THETA_LIST, PHI_LIST, pose_client.num_of_joints)

    for animation in animations_to_test:
        airsim_client.simPauseDrone(True)
        airsim_client.changeAnimation(ANIM_TO_UNREAL[animation])
        #implement a human pause function in airsim

        for _ in range(2): 
            photo_loc = file_manager.get_photo_loc(airsim_client.linecount, USE_AIRSIM)

            #????????????
            take_photo(airsim_client, pose_client, file_manager.take_photo_loc)
            determine_openpose_error(airsim_client, pose_client, plot_loc = file_manager.plot_loc, photo_loc = photo_loc)
            
            potential_states_fetcher.reset(pose_client, current_state.drone_pos)
            potential_states_fetcher.dome_experiment()

            num_of_samples = len(THETA_LIST)*len(PHI_LIST)
            for sample_ind in range(num_of_samples):
                photo_loc = file_manager.get_photo_loc(airsim_client.linecount, USE_AIRSIM)

                goal_state = potential_states_fetcher.potential_states_try[sample_ind]

                sim_pos = goal_state['position']

                airsim_client.simPauseDrone(False)
                airsim_client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(sim_pos[0],sim_pos[1],sim_pos[2]), airsim.to_quaternion(0, 0, goal_state["orientation"])), False)
                airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(goal_state['pitch'], 0, 0))
                take_photo(airsim_client, pose_client, file_manager.take_photo_loc)
                airsim_client.simPauseDrone(True)

                pose_client.cam_pitch = goal_state['pitch']
                
                determine_openpose_error(airsim_client, pose_client, plot_loc = file_manager.plot_loc, photo_loc = photo_loc)

                plot_drone_traj(pose_client, file_manager.plot_loc, airsim_client.linecount)

                #SAVE ALL VALUES OF THIS SIMULATION
                file_manager.append_openpose_error(pose_client.openpose_error)
                #file_manager.save_simulation_values(airsim_client, pose_client)

                airsim_client.linecount += 1 #THIS IS CONFUSING
                print('linecount', airsim_client.linecount)

            print("WRITING ERROR NOW!")
            file_manager.write_openpose_error(pose_client.current_pose_GT)

            #implement a human pause function in airsim
            airsim_client.simPauseHuman(False)
            time.sleep(1.5)
            airsim_client.simPauseHuman(True)

def dome_loop(current_state, pose_client, airsim_client, potential_states_fetcher, file_manager):
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

        airsim_client.simPauseDrone(False)
        airsim_client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(sim_pos[0],sim_pos[1],sim_pos[2]), airsim.to_quaternion(0, 0, goal_state["orientation"])), False)
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(goal_state['pitch'], 0, 0))
        take_photo(airsim_client, pose_client, file_manager.take_photo_loc)
        airsim_client.simPauseDrone(True)

        #SAVE ALL VALUES OF THIS SIMULATION
        file_manager.save_simulation_values(airsim_client, pose_client)

        airsim_client.linecount += 1
        print('linecount', airsim_client.linecount)

        if  (airsim_client.linecount == len(THETA_LIST)*len(PHI_LIST)):
            end_test = True
        airsim_client.simPause(False)

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
