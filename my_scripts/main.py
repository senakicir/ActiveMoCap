from helpers import * 
from State import *
from NonAirSimClient import *
from PoseEstimationClient import *
from pose3d_optimizer import *
from project_bones import *
from determine_positions import *
import pprint

gt_hv = []
est_hv = []
USE_AIRSIM = False
LENGTH_OF_SIMULATION = 100
photo_time = 0

def get_client_unreal_values(client, X):
    unreal_positions = np.zeros([5,3])
    if (USE_AIRSIM):
        DRONE_INITIAL_POS = client.DRONE_INITIAL_POS
        keys = {'humanPos': HUMAN_POS_IND, 'dronePos' : DRONE_POS_IND, 'droneOrient': DRONE_ORIENTATION_IND, 'left_arm': L_SHOULDER_IND, 'right_arm': R_SHOULDER_IND}
        for key, value in keys.items():
            element = X[key]
            if (key != 'droneOrient'):
                unreal_positions[value, :] = np.array([element.x_val, element.y_val, -element.z_val])
                unreal_positions[value, :]  = (unreal_positions[value, :] - DRONE_INITIAL_POS)/100
            else:
                unreal_positions[value, :] = np.array([element.x_val, element.y_val, element.z_val])
    else:
        do_nothing()
    return unreal_positions

def take_photo(airsim_client, image_folder_loc):
    if USE_AIRSIM:
        ##timedebug
        s1 = time.time()
        response = airsim_client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)])
        response = response[0]
        X = response.bones  
        global photo_time
        photo_time = time.time() - s1
        print("Get image from airsim takes" , photo_time)

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

    return response.image_data_uint8, gt_str

def main(kalman_arguments, parameters, energy_parameters):

    errors_pos = []
    errors_vel = []
    errors = {}
    end_test = False
       
    USE_TRACKBAR = parameters["USE_TRACKBAR"]
    global USE_AIRSIM
    USE_AIRSIM = parameters["USE_AIRSIM"]
    ANIMATION_NUM = parameters["ANIMATION_NUM"]
    test_set_name = parameters["TEST_SET_NAME"]
    file_names = parameters["FILE_NAMES"]
    folder_names = parameters["FOLDER_NAMES"]
    IS_ACTIVE = parameters["ACTIVE"]

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
        airsim_client.takeoffAsync(timeout_sec = 20)
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(CAMERA_PITCH_OFFSET, 0, 0))
        #airsim_client.moveToZAsync(-z_pos, 2, timeout_sec = 5, yaw_mode = airsim.YawMode(), lookahead = -1, adaptive_lookahead = 1)
        time.sleep(20)
    else:
        filename_bones = 'test_sets/'+test_set_name+'/groundtruth.txt'
        filename_output = 'test_sets/'+test_set_name+'/a_flight.txt'
        airsim_client = NonAirSimClient(filename_bones, filename_output)

    pose_client = PoseEstimationClient(energy_parameters)

    #define some variables
    airsim_client.linecount = 0

    gt_hp = []
    est_hp = []
   # global_plot_ind = 0

    filenames_anim = file_names[ANIMATION_NUM]
    foldernames_anim = folder_names[ANIMATION_NUM]
    f_output = open(filenames_anim["f_output"], 'w')
    f_groundtruth = open(filenames_anim["f_groundtruth"], 'w')

    #save drone initial position
    f_groundtruth_prefix = "-1\t" + str(airsim_client.DRONE_INITIAL_POS[0,]) + "\t" + str(airsim_client.DRONE_INITIAL_POS[1,]) + "\t" + str(airsim_client.DRONE_INITIAL_POS[2,])
    for i in range(0,70):
        f_groundtruth_prefix = f_groundtruth_prefix + "\t"
    f_groundtruth.write(f_groundtruth_prefix + "\n")
    take_photo(airsim_client, foldernames_anim["images"])

    plot_loc_ = foldernames_anim["superimposed_images"]
    if USE_AIRSIM:
        photo_loc_ = foldernames_anim["images"] + '/img_' + str(airsim_client.linecount) + '.png'
    else:
        photo_loc_ = 'test_sets/'+test_set_name+'/images/img_' + str(airsim_client.linecount) + '.png'

    airsim_client.simPause(True)
    initial_positions, _  = determine_all_positions(airsim_client, pose_client, plot_loc=plot_loc_, photo_loc=photo_loc_)
    airsim_client.simPause(False)

    current_state = State(initial_positions)
    current_state.radius = 14
    #shoulder_vector = initial_positions[R_SHOULDER_IND, :] - initial_positions[L_SHOULDER_IND, :] #find initial human orientation!
    #INITIAL_HUMAN_ORIENTATION = np.arctan2(-shoulder_vector[0], shoulder_vector[1]) #in unreal coordinates

    print ('Drone started %.2f m. from the hiker.\n' % current_state.radius)

    if USE_TRACKBAR:
        # create trackbars for angle change
        cv2.namedWindow('Drone Control')
        cv2.createTrackbar('Angle','Drone Control', 0, 360, do_nothing)
        #cv2.setTrackbarPos('Angle', 'Angle Control', int(degrees(some_angle-INITIAL_HUMAN_ORIENTATION)))
        cv2.setTrackbarPos('Angle', 'Drone Control', int(degrees(current_state.some_angle)))

        cv2.createTrackbar('Radius','Drone Control', 3, 10, do_nothing)
        cv2.setTrackbarPos('Radius', 'Drone Control', int(current_state.radius))

        cv2.createTrackbar('Z','Drone Control', 3, 20, do_nothing)
        cv2.setTrackbarPos('Z', 'Drone Control', z_pos)

    #if (pose_client.modes["mode_3d"] == 3 and USE_AIRSIM):
    #    cv2.namedWindow('Calibration for 3d pose')
    #    cv2.createTrackbar('Calibration mode','Calibration for 3d pose', 0, 1, do_nothing)
    #    cv2.setTrackbarPos('Calibration mode','Calibration for 3d pose', 1)

    while (not end_test):

        if USE_AIRSIM:
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        _, f_groundtruth_str = take_photo(airsim_client, foldernames_anim["images"])

        if (airsim_client.linecount == pose_client.CALIBRATION_LENGTH):
            #client.switch_energy(energy_mode[cv2.getTrackbarPos('Calibration mode', 'Calibration for 3d pose')])
            airsim_client.changeCalibrationMode(False)
            pose_client.changeCalibrationMode(False)
            global_plot_ind =0

        if (USE_AIRSIM):
            photo_loc_ = foldernames_anim["images"] + '/img_' + str(airsim_client.linecount) + '.png'
        else:
            photo_loc_ = 'test_sets/'+test_set_name+'/images/img_' + str(airsim_client.linecount) + '.png'

        airsim_client.simPause(True)
        positions, unreal_positions = determine_all_positions(airsim_client, pose_client, plot_loc = plot_loc_, photo_loc = photo_loc_)
        airsim_client.simPause(False)


        current_state.updateState(positions) #updates human pos, human orientation, human vel, drone pos

        #finds desired position and yaw angle
        if (USE_TRACKBAR):
            [desired_pos, desired_yaw_deg] = current_state.get_desired_pos_and_yaw_trackbar()
        elif (IS_ACTIVE):
            if(pose_client.isCalibratingEnergy):
                [desired_pos, desired_yaw_deg] = current_state.get_desired_pos_and_angle_fixed_rotation()
            else:
                [desired_pos, desired_yaw_deg] = current_state.get_goal_pos_and_yaw_active(pose_client)
        else:
            [desired_pos, desired_yaw_deg] = current_state.get_desired_pos_and_angle_fixed_rotation()
        
        #find desired drone speed
        desired_vel = (desired_pos - current_state.drone_pos)/TIME_HORIZON #how much the drone will have to move for this iteration
        drone_speed = np.linalg.norm(desired_vel)        

        #move drone!
        if (airsim_client.linecount <5):
            damping_speed = 0.025*airsim_client.linecount
        else:
            damping_speed = 0.5
        airsim_client.moveToPositionAsync(desired_pos[0], desired_pos[1], desired_pos[2], drone_speed*damping_speed, DELTA_T, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(is_rate=False, yaw_or_rate=desired_yaw_deg), lookahead=-1, adaptive_lookahead=0)
        time.sleep(DELTA_T) 

        airsim_client.simPause(True)
        plot_drone_traj(pose_client, plot_loc_, airsim_client.linecount)
        #if (airsim_client.linecount % 1 == 0 and not pose_client.quiet):
            #plot_global_motion(pose_client, plot_loc_, global_plot_ind)
            #plot_covariance_as_ellipse(pose_client, plot_loc_, global_plot_ind)

        #SAVE ALL VALUES OF THIS SIMULATION
        gt_hp.append(unreal_positions[HUMAN_POS_IND, :])
        est_hp.append(current_state.human_pos)
        errors_pos.append(np.linalg.norm(unreal_positions[HUMAN_POS_IND, :]-current_state.human_pos))
        if (airsim_client.linecount > 0):
            gt_hv.append((gt_hp[-1]-gt_hp[-2])/DELTA_T)
            est_hv.append(current_state.human_vel)
            errors_vel.append(np.linalg.norm( (gt_hp[-1]-gt_hp[-2])/DELTA_T - current_state.human_vel))

        f_output_str = str(airsim_client.linecount)+pose_client.f_string + '\n'
        f_output.write(f_output_str)
        f_groundtruth_str =  str(airsim_client.linecount) + '\t' +f_groundtruth_str + '\n'
        f_groundtruth.write(f_groundtruth_str)
        airsim_client.simPause(False)

        airsim_client.linecount += 1
        print('linecount', airsim_client.linecount)

        if (not USE_AIRSIM):
            end_test = airsim_client.end
        else:
            if (airsim_client.linecount == LENGTH_OF_SIMULATION):
                end_test = True

    #calculate errors
    error_arr_pos = np.asarray(errors_pos)
    errors["error_ave_pos"] = np.mean(error_arr_pos)
    errors["error_std_pos"] = np.std(error_arr_pos)

    error_arr_vel = np.asarray(errors_vel)
    errors["error_ave_vel"] = np.mean(error_arr_vel)
    errors["error_std_vel"] = np.std(error_arr_vel)
    errors["ave_3d_err"] = sum(pose_client.error_3d)/len(pose_client.error_3d)

    gt_hp_arr = np.asarray(gt_hp)
    est_hp_arr = np.asarray(est_hp)
    gt_hv_arr = np.asarray(gt_hv)
    est_hv_arr = np.asarray(est_hv)
    estimate_folder_name = foldernames_anim["estimates"]
    plot_error(gt_hp_arr, est_hp_arr, gt_hv_arr, est_hv_arr, errors, estimate_folder_name)
    simple_plot(pose_client.processing_time, estimate_folder_name, "processing_time", plot_title="Processing Time", x_label="Frames", y_label="Time")
    if (pose_client.modes["mode_3d"] == 3):
        simple_plot(pose_client.error_2d, estimate_folder_name, "2D error", plot_title="error_2d", x_label="Frames", y_label="Error")
    simple_plot(pose_client.error_3d[:pose_client.CALIBRATION_LENGTH], estimate_folder_name, "3D error", plot_title="calib_error_3d", x_label="Frames", y_label="Error")    
    simple_plot(pose_client.error_3d[pose_client.CALIBRATION_LENGTH:], estimate_folder_name, "3D error", plot_title="flight_error_3d", x_label="Frames", y_label="Error")
    
    if (pose_client.calc_hess and not pose_client.quiet):
        plot_covariances(pose_client, plot_loc_, "future_current_cov_")

    print('End it!')
    f_groundtruth.close()
    f_output.close()

    airsim_client.reset()
    pose_client.reset(plot_loc_) #DO NOT FORGET

    airsim_client.changeAnimation(0) #reset animation

    return errors

if __name__ == "__main__":
    kalman_arguments = {"KALMAN_PROCESS_NOISE_AMOUNT" :1, "KALMAN_MEASUREMENT_NOISE_AMOUNT_XY" : 1e-3}
    kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_Z"] = 1000 * kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_XY"]
    use_trackbar = False
    use_airsim = True
    param_read_M = False
    param_find_M = False
    is_quiet = False
    calculate_hess = True
    active = True
    flight_window_size = 6
    calibration_length = 12

    #mode_3d: 0- gt, 1- naiveback, 2- energy pytorch, 3-energy scipy
    #mode_2d: 0- gt, 1- openpose
    #mode_lift: 0- gt, 1- lift
    modes = {"mode_3d":3, "mode_2d":0, "mode_lift":0} 
   
    animations = ["02_01"]#["64_06"]
    test_set = {}
    for animation_num in animations:
        test_set[animation_num] = TEST_SETS[animation_num]

    file_names, folder_names, f_notes_name, _ = reset_all_folders(animations)

    parameters = {"USE_TRACKBAR": use_trackbar, "USE_AIRSIM": use_airsim, "FILE_NAMES": file_names, "FOLDER_NAMES": folder_names, "ACTIVE": active}
    
    weights_ =  {'proj': 0.0003332222592469177, 'smooth': 0.3332222592469177, 'bone': 0.3332222592469177, 'lift': 0.3332222592469177}
    weights = normalize_weights(weights_)

    energy_parameters = {"CALCULATE_HESSIAN":calculate_hess,"FLIGHT_WINDOW_SIZE": flight_window_size, "CALIBRATION_LENGTH": calibration_length, "PARAM_FIND_M": param_find_M, "PARAM_READ_M": param_read_M, "QUIET": is_quiet, "MODES": modes, "MODEL": "mpi", "METHOD": "trf", "FTOL": 1e-3, "WEIGHTS": weights}
    fill_notes(f_notes_name, parameters, energy_parameters)   

    if (use_airsim):
        for animation_num in animations:
            parameters["ANIMATION_NUM"]=  animation_num
            parameters["TEST_SET_NAME"]= ""
            errors = main(kalman_arguments, parameters, energy_parameters)
            print(errors)
    else:
        for animation_num, test_set in test_set.items():
            parameters["ANIMATION_NUM"]=  animation_num
            parameters["TEST_SET_NAME"]= test_set
            errors = main(kalman_arguments, parameters, energy_parameters)
            print(errors)
