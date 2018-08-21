from helpers import * 
from State import *
from NonAirSimClient import *
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
    if (USE_AIRSIM == True):
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

def take_photo(client, image_folder_loc):
    if (USE_AIRSIM == True):
        ##timedebug
        s1 = time.time()
        client.simPause(True)
        response = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)])
        client.simPause(False)
        response = response[0]
        X = response.bones  
        global photo_time
        photo_time = time.time() - s1
        print("Get image from airsim takes" , photo_time)

        gt_numbers = vector3r_arr_to_dict(X)
        unreal_positions = get_client_unreal_values(client, gt_numbers)
        gt_str = ""
        bone_pos = np.zeros([3, len(gt_numbers)-3])
        DRONE_INITIAL_POS = client.DRONE_INITIAL_POS
        for bone_ind, bone_i in enumerate(attributes):
            gt_str = gt_str + str(gt_numbers[bone_i].x_val) + '\t' + str(gt_numbers[bone_i].y_val) + '\t' +  str(gt_numbers[bone_i].z_val) + '\t'
            if (bone_ind >= 3):
                bone_pos[:, bone_ind-3] = np.array([gt_numbers[bone_i].x_val, gt_numbers[bone_i].y_val, -gt_numbers[bone_i].z_val]) - DRONE_INITIAL_POS
        bone_pos = bone_pos / 100

        multirotor_state = client.getMultirotorState()
        estimated_state =  multirotor_state.kinematics_estimated
        drone_pos = estimated_state.position
        #drone_orient = airsim.to_eularian_angles(estimated_state.orientation)
        #CHANGE THIS
        drone_orient = unreal_positions[DRONE_ORIENTATION_IND]

        client.updateSynchronizedData(unreal_positions, bone_pos, drone_pos, drone_orient)
        
        loc = image_folder_loc + '/img_' + str(client.linecount) + '.png'
        airsim.write_file(os.path.normpath(loc), response.image_data_uint8)

    else:
        response = client.simGetImages()
        bone_pos = response.bone_pos
        unreal_positions = response.unreal_positions
        drone_orient = client.getPitchRollYaw()
        drone_pos = client.getPosition()
        client.updateSynchronizedData(unreal_positions, bone_pos, drone_pos, drone_orient)
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
        parameters = {"USE_TRACKBAR": False, "MODE_3D": 0, "MODE_2D":0, "USE_AIRSIM": True, "ANIMATION_NUM": 1, "TEST_SET_NAME": "test_set_1", "FILE_NAMES": "", "FOLDER_NAMES": "", "MODEL": "mpi"}
    if (energy_parameters == None):
        energy_parameters = {"LR_MU": [0.2, 0.8], "ITER": 3000, "WEIGHTS": {"proj":1,"smooth":0.5, "bone":10}}
    
    USE_TRACKBAR = parameters["USE_TRACKBAR"]
    mode_3d = parameters["MODE_3D"]
    mode_2d = parameters["MODE_2D"]
    global USE_AIRSIM
    USE_AIRSIM = parameters["USE_AIRSIM"]
    ANIMATION_NUM = parameters["ANIMATION_NUM"]
    test_set_name = parameters["TEST_SET_NAME"]
    file_names = parameters["FILE_NAMES"]
    folder_names = parameters["FOLDER_NAMES"]

    #connect to the AirSim simulator
    if (USE_AIRSIM == True):
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        print('Taking off')
        client.initInitialDronePos()
        client.changeAnimation(ANIMATION_NUM)
        client.changeCalibrationMode(True)
        client.takeoffAsync(timeout_sec = 20)
        client.moveToZAsync(-z_pos, 2, timeout_sec = 5, yaw_mode = airsim.YawMode(), lookahead = -1, adaptive_lookahead = 1)
        time.sleep(5)
        client.simSetCameraOrientation(str(0), airsim.to_quaternion(CAMERA_PITCH_OFFSET, 0, 0))
    else:
        filename_bones = 'test_sets/'+test_set_name+'/groundtruth.txt'
        filename_output = 'test_sets/'+test_set_name+'/a_flight.txt'
        client = NonAirSimClient(filename_bones, filename_output)

    #define some variables
    client.linecount = 0
    client.lr = energy_parameters["LR_MU"][0]
    client.mu = energy_parameters["LR_MU"][1]
    client.iter = energy_parameters["ITER"]
    client.weights = energy_parameters["WEIGHTS"]
    client.model = parameters["MODEL"]

    if client.model =="mpi":
        client.boneLengths = torch.zeros([14,1])
    else:
        client.boneLengths = torch.zeros([20,1])

    gt_hp = []
    est_hp = []

    filenames_anim = file_names[ANIMATION_NUM]
    foldernames_anim = folder_names[ANIMATION_NUM]
    f_output = open(filenames_anim["f_output"], 'w')
    f_groundtruth = open(filenames_anim["f_groundtruth"], 'w')

    #save drone initial position
    f_groundtruth_prefix = "-1\t" + str(client.DRONE_INITIAL_POS[0,]) + "\t" + str(client.DRONE_INITIAL_POS[1,]) + "\t" + str(client.DRONE_INITIAL_POS[2,])
    for i in range(0,70):
        f_groundtruth_prefix = f_groundtruth_prefix + "\t"
    f_groundtruth.write(f_groundtruth_prefix + "\n")
    photo, _ = take_photo(client, foldernames_anim["images"])

    plot_loc_ = foldernames_anim["superimposed_images"]
    if (USE_AIRSIM==True):
        photo_loc_ = foldernames_anim["images"] + '/img_' + str(client.linecount) + '.png'
    else:
        photo_loc_ = 'test_sets/'+test_set_name+'/images/img_' + str(client.linecount) + '.png'

    initial_positions, _, _, _, _ = determine_all_positions(mode_3d, mode_2d, client, MEASUREMENT_NOISE_COV, plot_loc=plot_loc_, photo_loc=photo_loc_)

    current_state = State(initial_positions, kalman_arguments['KALMAN_PROCESS_NOISE_AMOUNT'])
    
    #shoulder_vector = initial_positions[R_SHOULDER_IND, :] - initial_positions[L_SHOULDER_IND, :] #find initial human orientation!
    #INITIAL_HUMAN_ORIENTATION = np.arctan2(-shoulder_vector[0], shoulder_vector[1]) #in unreal coordinates

    print ('Drone started %.2f m. from the hiker.\n' % current_state.radius)

    if (USE_TRACKBAR == True):
        # create trackbars for angle change
        cv2.namedWindow('Drone Control')
        cv2.createTrackbar('Angle','Drone Control', 0, 360, do_nothing)
        #cv2.setTrackbarPos('Angle', 'Angle Control', int(degrees(some_angle-INITIAL_HUMAN_ORIENTATION)))
        cv2.setTrackbarPos('Angle', 'Drone Control', int(degrees(current_state.some_angle)))

        cv2.createTrackbar('Radius','Drone Control', 3, 10, do_nothing)
        cv2.setTrackbarPos('Radius', 'Drone Control', int(current_state.radius))

        cv2.createTrackbar('Z','Drone Control', 3, 20, do_nothing)
        cv2.setTrackbarPos('Z', 'Drone Control', z_pos)

    if (mode_3d == 3 and USE_AIRSIM == True):
        cv2.namedWindow('Calibration for 3d pose')
        cv2.createTrackbar('Calibration mode','Calibration for 3d pose', 0, 1, do_nothing)
        cv2.setTrackbarPos('Calibration mode','Calibration for 3d pose', 1)

    while (end_test == False):

        start = time.time()
        if USE_AIRSIM == True:
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        ##timedebug
        photo, f_groundtruth_str = take_photo(client, foldernames_anim["images"])

        #set the mode for energy, calibration mode or no?
        #if (USE_AIRSIM == True):
        if (client.linecount == CALIBRATION_LENGTH):
            #client.switch_energy(energy_mode[cv2.getTrackbarPos('Calibration mode', 'Calibration for 3d pose')])
            client.changeCalibrationMode(False)
        

        if (USE_AIRSIM==True):
            photo_loc_ = foldernames_anim["images"] + '/img_' + str(client.linecount) + '.png'
        else:
            photo_loc_ = 'test_sets/'+test_set_name+'/images/img_' + str(client.linecount) + '.png'

        positions, unreal_positions, cov, inFrame, f_output_str = determine_all_positions(mode_3d, mode_2d, client, MEASUREMENT_NOISE_COV, plot_loc = plot_loc_, photo_loc = photo_loc_)
        inFrame = True #TO DO
        
        current_state.updateState(positions, inFrame, cov) #updates human pos, human orientation, human vel, drone pos

        gt_hp.append(unreal_positions[HUMAN_POS_IND, :])
        est_hp.append(current_state.human_pos)
        errors_pos.append(np.linalg.norm(unreal_positions[HUMAN_POS_IND, :]-current_state.human_pos))
        if (client.linecount > 0):
            gt_hv.append((gt_hp[-1]-gt_hp[-2])/DELTA_T)
            est_hv.append(current_state.human_vel)
            errors_vel.append(np.linalg.norm( (gt_hp[-1]-gt_hp[-2])/DELTA_T - current_state.human_vel))

        #finds desired position and angle
        if (USE_TRACKBAR == True):
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
        damping_speed = 1
        client.moveToPositionAsync(new_pos[0], new_pos[1], new_pos[2], drone_speed*damping_speed, DELTA_T, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(is_rate=False, yaw_or_rate=desired_yaw_deg), lookahead=-1, adaptive_lookahead=0)
        end = time.time()
        elapsed_time = end - start
        print("elapsed time: ", elapsed_time)
        time.sleep(DELTA_T)

        #SAVE ALL VALUES OF THIS SIMULATION       
        f_output_str = str(client.linecount)+f_output_str + '\n'
        f_output.write(f_output_str)
        f_groundtruth_str =  str(client.linecount) + '\t' +f_groundtruth_str + '\n'
        f_groundtruth.write(f_groundtruth_str)

        client.linecount += 1
        print('linecount', client.linecount)

        if (USE_AIRSIM == False):
            end_test = client.end
        else:
            if (client.linecount == LENGTH_OF_SIMULATION):
                end_test = True

    #calculate errors
    error_arr_pos = np.asarray(errors_pos)
    errors["error_ave_pos"] = np.mean(error_arr_pos)
    errors["error_std_pos"] = np.std(error_arr_pos)

    error_arr_vel = np.asarray(errors_vel)
    errors["error_ave_vel"] = np.mean(error_arr_vel)
    errors["error_std_vel"] = np.std(error_arr_vel)

    gt_hp_arr = np.asarray(gt_hp)
    est_hp_arr = np.asarray(est_hp)
    gt_hv_arr = np.asarray(gt_hv)
    est_hv_arr = np.asarray(est_hv)
    estimate_folder_name = foldernames_anim["estimates"]
    plot_error(gt_hp_arr, est_hp_arr, gt_hv_arr, est_hv_arr, errors, estimate_folder_name)
    if (mode_3d == 3):
        plot_loss_2d(client, estimate_folder_name)
    plot_loss_3d(client, estimate_folder_name)

    print('End it!')
    f_groundtruth.close()
    f_output.close()

    client.reset()
    client.changeAnimation(0) #reset animation

    return errors

if __name__ == "__main__":
    kalman_arguments = {"KALMAN_PROCESS_NOISE_AMOUNT" : 5.17947467923e-10, "KALMAN_MEASUREMENT_NOISE_AMOUNT_XY" : 1.38949549437e-08}
    kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_Z"] = 517.947467923 * kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_XY"]
    use_airsim = True
    mode_3d = 0 #0 - gt, 1- naiveback, 2- energy, 3-energy scipy
    mode_2d = 0 # 0- gt, 1- openpose
    use_trackbar = False

    #animations = [0,1,2,3]
    animations = [4]
    test_set = {}
    for animation_num in animations:
        test_set[animation_num] = TEST_SETS[animation_num]

    file_names, folder_names, f_notes_name = reset_all_folders(animations)

    parameters = {"USE_TRACKBAR": use_trackbar, "MODE_3D": mode_3d, "MODE_2D": mode_2d, "USE_AIRSIM": use_airsim, "FILE_NAMES": file_names, "FOLDER_NAMES": folder_names, "MODEL": "mpi"}
    
    weights_ = {'proj': 0.01, 'smooth': 0.8, 'bone': 0.3, 'lift': 0.4}#'smoothpose': 0.01,}
    weights = {}
    weights_sum = sum(weights_.values())
    for loss_key in LOSSES:
        weights[loss_key] = weights_[loss_key]/weights_sum

    energy_parameters = {"LR_MU": [1, 0.8], "ITER": 4000, "WEIGHTS": weights}

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
