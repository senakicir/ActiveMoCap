from helpers import * 
from State import *
from NonAirSimClient import *
from pose_est_client import *
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
        airsim_client.simPause(True)
        response = airsim_client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)])
        airsim_client.simPause(False)
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

def main(kalman_arguments = None, parameters = None, energy_parameters = None):

    errors_pos = []
    errors_vel = []
    errors = {}
    end_test = False

    if (kalman_arguments == None):
        kalman_arguments = {"KALMAN_PROCESS_NOISE_AMOUNT" : 3.72759372031e-11, "KALMAN_MEASUREMENT_NOISE_AMOUNT_XY" : 7.19685673001e-08}
        kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_Z"] = 77.4263682681 * kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_XY"]
    MEASUREMENT_NOISE_COV = np.array([[kalman_arguments["KALMAN_PROCESS_NOISE_AMOUNT"], 0, 0], [0, kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_XY"], 0], [0, 0, kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_Z"]]])

    if (parameters == None):
        parameters = {"QUIET": False, "USE_TRACKBAR": False, "USE_AIRSIM": True, "ANIMATION_NUM": 1, "TEST_SET_NAME": "test_set_1", "FILE_NAMES": "", "FOLDER_NAMES": "", "MODEL": "mpi"}
    if (energy_parameters == None):
        energy_parameters = {"FTOL": 1e-2, "METHOD": "trf", "WEIGHTS": {"proj":0.25,"smooth":0.25, "bone":0.25, "lift":0.25}}
    
    USE_TRACKBAR = parameters["USE_TRACKBAR"]
    global USE_AIRSIM
    USE_AIRSIM = parameters["USE_AIRSIM"]
    ANIMATION_NUM = parameters["ANIMATION_NUM"]
    test_set_name = parameters["TEST_SET_NAME"]
    file_names = parameters["FILE_NAMES"]
    folder_names = parameters["FOLDER_NAMES"]
    quiet = parameters["QUIET"]

    #connect to the AirSim simulator
    if USE_AIRSIM:
        airsim_client = airsim.MultirotorClient()
        airsim_client.confirmConnection()
        airsim_client.enableApiControl(True)
        airsim_client.armDisarm(True)
        print('Taking off')
        airsim_client.initInitialDronePos()
        pose_client = PoseEstimationClient(parameters["MODEL"])
        airsim_client.changeAnimation(ANIMATION_NUM)
        airsim_client.changeCalibrationMode(True)
        pose_client.changeCalibrationMode(True)
        airsim_client.takeoffAsync(timeout_sec = 20)
        airsim_client.moveToZAsync(-z_pos, 2, timeout_sec = 5, yaw_mode = airsim.YawMode(), lookahead = -1, adaptive_lookahead = 1)
        time.sleep(5)
        airsim_client.simSetCameraOrientation(str(0), airsim.to_quaternion(CAMERA_PITCH_OFFSET, 0, 0))
    else:
        filename_bones = 'test_sets/'+test_set_name+'/groundtruth.txt'
        filename_output = 'test_sets/'+test_set_name+'/a_flight.txt'
        airsim_client = NonAirSimClient(filename_bones, filename_output)

    #define some variables
    airsim_client.linecount = 0
    pose_client.modes = parameters["MODES"]
    pose_client.method = energy_parameters["METHOD"]
    pose_client.ftol = energy_parameters["FTOL"]
    pose_client.weights = energy_parameters["WEIGHTS"]

    if pose_client.model =="mpi":
        pose_client.boneLengths = torch.zeros([14,1])
    else:
        pose_client.boneLengths = torch.zeros([20,1])

    gt_hp = []
    est_hp = []
    processing_time = []
    plot_info = []
    global_plot_ind = 0

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

    initial_positions, _, _, _  = determine_all_positions(airsim_client, pose_client, MEASUREMENT_NOISE_COV, plot_loc=plot_loc_, photo_loc=photo_loc_, quiet=quiet)

    current_state = State(initial_positions)#, kalman_arguments['KALMAN_PROCESS_NOISE_AMOUNT'])
    
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

    if (pose_client.modes["mode_3d"] == 3 and USE_AIRSIM):
        cv2.namedWindow('Calibration for 3d pose')
        cv2.createTrackbar('Calibration mode','Calibration for 3d pose', 0, 1, do_nothing)
        cv2.setTrackbarPos('Calibration mode','Calibration for 3d pose', 1)

    while (not end_test):

        start = time.time()
        if USE_AIRSIM:
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        ##timedebug
        photo, f_groundtruth_str = take_photo(airsim_client, foldernames_anim["images"])

        #set the mode for energy, calibration mode or no?
        #if (USE_AIRSIM):
        if (airsim_client.linecount == CALIBRATION_LENGTH):
            #client.switch_energy(energy_mode[cv2.getTrackbarPos('Calibration mode', 'Calibration for 3d pose')])
            airsim_client.changeCalibrationMode(False)
            pose_client.changeCalibrationMode(False)
            global_plot_ind =0
            plot_info = []

        if (USE_AIRSIM):
            photo_loc_ = foldernames_anim["images"] + '/img_' + str(airsim_client.linecount) + '.png'
        else:
            photo_loc_ = 'test_sets/'+test_set_name+'/images/img_' + str(airsim_client.linecount) + '.png'

        positions, unreal_positions, cov, plot_end = determine_all_positions(airsim_client, pose_client, MEASUREMENT_NOISE_COV, plot_loc = plot_loc_, photo_loc = photo_loc_, quiet=quiet)
        inFrame = True #TO DO
        
        current_state.updateState(positions, inFrame, cov) #updates human pos, human orientation, human vel, drone pos

        gt_hp.append(unreal_positions[HUMAN_POS_IND, :])
        est_hp.append(current_state.human_pos)
        errors_pos.append(np.linalg.norm(unreal_positions[HUMAN_POS_IND, :]-current_state.human_pos))
        if (airsim_client.linecount > 0):
            gt_hv.append((gt_hp[-1]-gt_hp[-2])/DELTA_T)
            est_hv.append(current_state.human_vel)
            errors_vel.append(np.linalg.norm( (gt_hp[-1]-gt_hp[-2])/DELTA_T - current_state.human_vel))

        #finds desired position and angle
        if (USE_TRACKBAR):
            [desired_pos, desired_yaw] = current_state.getDesiredPosAndAngleTrackbar()
        else:
            [desired_pos, desired_yaw] = current_state.getDesiredPosAndAngle()
        
        #find desired drone speed
        delta_pos = desired_pos - current_state.drone_pos #how much the drone will have to move for this iteration
        desired_vel = delta_pos/TIME_HORIZON
        drone_speed = np.linalg.norm(desired_vel)

        #update drone position
        curr_pos = current_state.drone_pos
        new_pos = desired_pos

        #angle required to face the hiker
        angle = current_state.drone_orientation
        current_yaw_deg = degrees(angle[2])
        yaw_candidates = np.array([degrees(desired_yaw), degrees(desired_yaw) - 360, degrees(desired_yaw) +360])
        min_diff = np.array([abs(current_yaw_deg -  yaw_candidates[0]), abs(current_yaw_deg -  yaw_candidates[1]), abs(current_yaw_deg -  yaw_candidates[2])])
        desired_yaw_deg = yaw_candidates[np.argmin(min_diff)]

        #move drone!
        if (airsim_client.linecount <5):
            damping_speed = 0.025*airsim_client.linecount
        else:
            damping_speed = 0.5
        airsim_client.moveToPositionAsync(new_pos[0], new_pos[1], new_pos[2], drone_speed*damping_speed, DELTA_T, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(is_rate=False, yaw_or_rate=desired_yaw_deg), lookahead=-1, adaptive_lookahead=0)
        end = time.time()
        #elapsed_time = end - start
        #print("elapsed time: ", elapsed_time)
        processing_time.append(plot_end["eval_time"])
        time.sleep(DELTA_T) 
        if (airsim_client.linecount % 3 == 0 and not quiet):
            plot_info.append(plot_end)
            plot_global_motion(plot_info, plot_loc_, global_plot_ind, pose_client.model, pose_client.isCalibratingEnergy)
            global_plot_ind +=1

        #SAVE ALL VALUES OF THIS SIMULATION       
        f_output_str = str(airsim_client.linecount)+plot_end["f_string"] + '\n'
        f_output.write(f_output_str)
        f_groundtruth_str =  str(airsim_client.linecount) + '\t' +f_groundtruth_str + '\n'
        f_groundtruth.write(f_groundtruth_str)

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
    errors["ave_3d_err"] = sum(client.error_3d)/len(client.error_3d)

    gt_hp_arr = np.asarray(gt_hp)
    est_hp_arr = np.asarray(est_hp)
    gt_hv_arr = np.asarray(gt_hv)
    est_hv_arr = np.asarray(est_hv)
    estimate_folder_name = foldernames_anim["estimates"]
    plot_error(gt_hp_arr, est_hp_arr, gt_hv_arr, est_hv_arr, errors, estimate_folder_name)
    simple_plot(processing_time, estimate_folder_name, "processing_time", plot_title="Processing Time", x_label="Frames", y_label="Time")
    if (pose_client.modes["mode_3d"] == 3):
        simple_plot(client.error_2d, estimate_folder_name, "2D error", plot_title="error_2d", x_label="Frames", y_label="Error")
    simple_plot(client.error_3d[:CALIBRATION_LENGTH], estimate_folder_name, "3D error", plot_title="calib_error_3d", x_label="Frames", y_label="Error")    
    simple_plot(client.error_3d[CALIBRATION_LENGTH:], estimate_folder_name, "3D error", plot_title="flight_error_3d", x_label="Frames", y_label="Error")

    print('End it!')
    f_groundtruth.close()
    f_output.close()

    airsim_client.reset()
    pose_client.reset()
    airsim_client.changeAnimation(0) #reset animation

    return errors

if __name__ == "__main__":
    kalman_arguments = {"KALMAN_PROCESS_NOISE_AMOUNT" : 5.17947467923e-10, "KALMAN_MEASUREMENT_NOISE_AMOUNT_XY" : 1.38949549437e-08}
    kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_Z"] = 517.947467923 * kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_XY"]
    use_airsim = False
    #mode_3d: 0- gt, 1- naiveback, 2- energy pytorch, 3-energy scipy
    #mode_2d: 0- gt, 1- openpose
    #mode_lift: 0- gt, 1- lift
    modes = {"mode_3d":3, "mode_2d":1, "mode_lift":1} 
   
    use_trackbar = False

    #animations = [0,1,2,3]
    animations = ["02_01"]
    test_set = {}
    for animation_num in animations:
        test_set[animation_num] = TEST_SETS[animation_num]

    file_names, folder_names, f_notes_name = reset_all_folders(animations)

    parameters = {"QUIET": False, "USE_TRACKBAR": use_trackbar, "MODES": modes, "USE_AIRSIM": use_airsim, "FILE_NAMES": file_names, "FOLDER_NAMES": folder_names, "MODEL": "mpi"}
    
    weights_ = {'proj': 0.010646024544911734, 'smooth': 0.4941446865002986, 'bone': 0.0010646024544911734, 'lift': 0.4941446865002986}#'smoothpose': 0.01,}
    weights = {}
    weights_sum = sum(weights_.values())
    for loss_key in LOSSES:
        weights[loss_key] = weights_[loss_key]/weights_sum

    energy_parameters = {"METHOD": "trf", "FTOL": 1e-3, "WEIGHTS": weights}
    fill_notes(f_notes_name, parameters, energy_parameters)   

    if (use_airsim):
        for animation_num in animations:

            parameters["ANIMATION_NUM"]= animation_num
            parameters["TEST_SET_NAME"]= ""
            errors = main(kalman_arguments, parameters, energy_parameters)
            print(errors)
    else:
        for animation_num, test_set in test_set.items():
            parameters["ANIMATION_NUM"]= animation_num
            parameters["TEST_SET_NAME"]= test_set
            errors = main(kalman_arguments, parameters, energy_parameters)
            print(errors)
