import setup_path 
import airsim

import shutil
import skimage.io
import numpy as np
import torch as torch
from pandas import read_csv

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import time, os
import cv2
from math import degrees, radians, pi, ceil, exp, atan2, sqrt, cos, sin, acos

TEST_SETS = {"t": "test_set_t", "05_08": "test_set_05_08", "38_03": "test_set_38_03", "64_06": "test_set_64_06", "02_01": "test_set_02_01"}
ANIM_TO_UNREAL = {"t": 0, "05_08": 1, "38_03": 2, "64_06": 3, "02_01": 4, "06_03":5, "05_11":6, "05_15":7, "06_09":8,"07_10": 9, 
                 "07_05": 10, "64_11": 11, "64_22":12, "64_26":13, "13_06":14, "14_32":15,"06_13":16,"14_01":17, "28_19":18, 
                 "noise":-1}

bones_h36m = [[0, 1], [1, 2], [2, 3], [3, 19], #right leg
              [0, 4], [4, 5], [5, 6], [6, 20], #left leg
              [0, 7], [7, 8], [8, 9], [9, 10], #middle
              [8, 14], [14, 15], [15, 16], [16, 17], #left arm
              [8, 11], [11, 12], [12, 13], [13, 18]] #right arm

joint_indices_h36m=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
joint_names_h36m = ['hip','right_up_leg','right_leg','right_foot','left_up_leg','left_leg', 'left_foot','spine1','neck', 'head', 'head_top', 'left_arm','left_forearm','left_hand','right_arm','right_forearm','right_hand', 'right_hand_tip', 'left_hand_tip', 'right_foot_tip', 'left_foot_tip']

bones_mpi = [[0, 1], [14, 1], #middle
            [1, 2], [2, 3], [3, 4], #right arm
            [1, 5], [5, 6], [6, 7],  #left arm
            [14, 8], [8, 9], [9, 10], #right leg
            [14, 11], [11, 12], [12, 13]] #left leg
joint_names_mpi = ['head','neck','right_arm','right_forearm','right_hand','left_arm', 'left_forearm','left_hand','right_up_leg','right_leg', 'right_foot', 'left_up_leg', 'left_leg', 'left_foot', 'spine1']

EPSILON = 0.00000001

SIZE_X = 1024
SIZE_Y = 576

CAMERA_OFFSET_X = 45/100
CAMERA_OFFSET_Y = 0
CAMERA_OFFSET_Z = 0#-4.92
CAMERA_ROLL_OFFSET = 0
CAMERA_PITCH_OFFSET = 0
CAMERA_YAW_OFFSET = 0

CURRENT_POSE_INDEX = 1
FUTURE_POSE_INDEX = 0
MIDDLE_POSE_INDEX = 3

plt.figure()
plt.close()

max_radii = 1

def find_bone_map():
    bones_map_to_mpi = []
    for ind, value in enumerate(joint_names_mpi):
        bones_map_to_mpi.append(joint_names_h36m.index(value))
    return bones_map_to_mpi

bones_map_to_mpi = find_bone_map()

def rearrange_bones_to_mpi(bones_unarranged, is_torch = True):
    if (is_torch):
        bones_rearranged = torch.zeros(3, 15)
        bones_rearranged = bones_unarranged[:, bones_map_to_mpi]
    else:
        bones_rearranged = np.zeros([3,15])
        bones_rearranged = bones_unarranged[:, bones_map_to_mpi]
    return bones_rearranged

def split_bone_connections(bone_connections):
    if (bone_connections == bones_h36m):
        left_bone_connections = [[8, 14], [14, 15], [15, 16], [16, 17], [0, 4], [4, 5], [5, 6], [6, 20]]
        right_bone_connections = [[8, 11], [11, 12], [12, 13], [13, 18], [0, 1], [1, 2], [2, 3], [3, 19]]
        middle_bone_connections = [[0, 7], [7, 8], [8, 9], [9, 10]]
    elif (bone_connections == bones_mpi):
        left_bone_connections = [[1, 5], [5, 6], [6, 7],[14, 11], [11, 12], [12, 13]]
        right_bone_connections = [[1, 2], [2, 3], [3, 4], [14, 8], [8, 9], [9, 10]]
        middle_bone_connections = [[0, 1], [14, 1]]
    return left_bone_connections, right_bone_connections, middle_bone_connections

additional_directions = [[4, 10], [7,13], [3,9], [6, 12], [14,3], [14, 6]]
lift_bone_directions = bones_mpi + additional_directions

def return_lift_bone_connections(bone_connections):
    if (bone_connections == bones_mpi):
        return lift_bone_directions
    elif (bone_connections == bones_h36m):
        #todo
        return lift_bone_directions

def return_arm_connection(bone_connections):
    if (bone_connections == bones_h36m):
        left_arm_connections = [[8, 14], [14, 15], [15, 16], [16, 17]]
        right_arm_connections = [[8, 11], [11, 12], [12, 13], [13, 18]]
    elif (bone_connections == bones_mpi):
        left_arm_connections = [[1, 5], [5, 6], [6, 7]]
        right_arm_connections = [[1, 2], [2, 3], [3, 4]]
    return right_arm_connections, left_arm_connections

def return_arm_joints(model="mpi"):
    if (model == "mpi"):
        arm_joints = [5,6,7,2,3,4]
        left_arm_joints = [5, 6, 7]
        right_arm_joints = [2, 3, 4]
    return arm_joints, right_arm_joints, left_arm_joints

def return_leg_joints(model="mpi"):
    if (model == "mpi"):
        leg_joints = [11,12,13,8,9,10]
        left_leg_joints = [11, 12, 13]
        right_leg_joints = [8, 9, 10]
    return leg_joints, right_leg_joints, left_leg_joints

def model_settings(model):
    if (model == "mpi"):
        bone_connections = bones_mpi
        joint_names = joint_names_mpi
        num_of_joints = 15
    else:
        bone_connections = bones_h36m
        joint_names = joint_names_h36m
        num_of_joints = 21
    return bone_connections, joint_names, num_of_joints


def normalize_weights(weights_):    
    weights = {}
    weights_sum = sum(weights_.values())
    for loss_key in weights_:
        weights[loss_key] = weights_[loss_key]/weights_sum
    return weights

def add_noise_to_pose(pose, noise_std):
    noise = torch.normal(torch.zeros(pose.shape), torch.ones(pose.shape)*noise_std).float()
    pose = pose.float().clone() + noise
    return pose 

def range_angle(angle, limit=360, is_radians = True):
    if is_radians:
        angle = degrees(angle)
    if angle > limit:
        angle = angle - 360
    elif angle < limit-360:
        angle = angle + 360
    if is_radians:
        angle = radians(angle)
    return angle

def save_bone_positions_2(index, bones, f_output):
    bones = [ v for v in bones.values() ]
    line = str(index)
    for i in range(0, len(bones)):
        line = line+'\t'+str(bones[i][b'x_val'])+'\t'+str(bones[i][b'y_val'])+'\t'+str(bones[i][b'z_val'])
    line = line+'\n'
    f_output.write(line)

def do_nothing(x):
    pass

def find_M(plot_info, hip_index, num_of_joints):
    p_GT = np.zeros([3*len(plot_info),num_of_joints])
    p_est = np.zeros([3*len(plot_info),num_of_joints])
    for frame_ind, frame_plot_info in enumerate(plot_info):
        predicted_bones = frame_plot_info["est"]
        bones_GT = frame_plot_info["GT"]
        root_GT = bones_GT[:,hip_index]
        root_est = predicted_bones[:,hip_index]
        p_GT[3*frame_ind:3*(frame_ind+1),:]= bones_GT-root_GT[:, np.newaxis]
        p_est[3*frame_ind:3*(frame_ind+1),:]= predicted_bones-root_est[:, np.newaxis]


    #remove spine row from both arrays
    p_est = np.delete(p_est, hip_index, 1)
    p_GT = np.delete(p_GT, hip_index, 1)
    filename = "M_rel.txt"
    
    X = np.linalg.inv(np.dot(p_est.T, p_est))
    M = np.dot(np.dot(X, p_est.T), p_GT)

    M = np.insert(M, hip_index, 0, axis=1)
    M = np.insert(M, hip_index, 0, axis=0)
    M[hip_index, hip_index] = 1

    M_file = open(filename, 'w')
    M_str = ""
    for i in range(0, num_of_joints):
        for j in range(0, num_of_joints):
            M_str += str(M[i,j]) + "\t"
        if (i != num_of_joints-1):
            M_str += "\n"
    M_file.write(M_str)

    return M

def read_M(num_of_joints, name = "M_rel"):
    filename = name+".txt"
    if os.path.exists(filename):
        X = read_csv(filename, sep='\t', header=None).ix[:,:].values.astype('float')     
        return X[:,0:num_of_joints]
    else:
        return np.eye(num_of_joints)

def move_M(destination_folder):
    os.rename("M_rel.txt", destination_folder+"/M_rel.txt")

def drone_flight_filenames():
    date_time_name = '2019-05-15-18-13'
    mode = "ransac"

    main_dir = "/cvlabdata2/home/kicirogl/ActiveDrone/drone_flight/2019_02_isinsu/video_1_full_framerate_2_trial_2"
    input_image_dir = main_dir
    general_output_folder = main_dir + "/drone_flight_dataset/" 
    gt_folder_dir = general_output_folder + date_time_name + "_" + mode + '/'
    openpose_liftnet_image_dir = general_output_folder + "openpose_liftnet_images"

    drone_flight_filenames = {"input_image_dir": input_image_dir, 
            "openpose_liftnet_image_dir": openpose_liftnet_image_dir, 
            "gt_folder_dir": gt_folder_dir,
            "f_drone_pos_reoriented": gt_folder_dir + "drone_pos_reoriented.txt", 
            "f_groundtruth_reoriented": gt_folder_dir + "groundtruth_reoriented.txt", 
            "f_pose_2d": general_output_folder + "pose_2d.txt", 
            "f_pose_lift": general_output_folder + "pose_lift.txt",
            "f_intrinsics": general_output_folder + "intrinsics.txt"}

    return drone_flight_filenames

def reset_all_folders(animation_list, base = ""):
    if (base == ""):
        base = "temp_main"
    if (base == "grid_search"):
        base = "grid_search"

    date_time_name = time.strftime("%Y-%m-%d-%H-%M")
    folder_names = [base, base + '/' + date_time_name]
    main_folder_name = base + '/' + date_time_name

    for a_folder_name in folder_names:
        if not os.path.exists(a_folder_name):
            os.makedirs(a_folder_name)
    
    file_names = {}
    folder_names = {}
    for animation in animation_list:
        sub_folder_name = main_folder_name + "/" + str(animation)
        for ind in range(animation_list[animation]):
            experiment_folder_name = sub_folder_name + "/" + str(ind)
            if not os.path.exists(experiment_folder_name):
                os.makedirs(experiment_folder_name)
            key = str(animation) + "_" + str(ind)
            folder_names[key] = {"images": experiment_folder_name + '/images', "estimates": experiment_folder_name + '/estimates', "superimposed_images":  experiment_folder_name + '/superimposed_images'}
            for a_folder_name in folder_names[key].values():
                if not os.path.exists(a_folder_name):
                    os.makedirs(a_folder_name)
            file_names[key] = {"f_error": experiment_folder_name +  '/error.txt', 
                "f_groundtruth": experiment_folder_name +  '/groundtruth.txt', 
                "f_reconstruction": experiment_folder_name +  '/reconstruction.txt', 
                "f_uncertainty": experiment_folder_name +  '/uncertainty.txt',
                "f_average_error" : experiment_folder_name + '/average_error.txt',
                "f_drone_pos": experiment_folder_name +  '/drone_pos.txt', 
                "f_initial_drone_pos": experiment_folder_name +  '/initial_drone_pos.txt', 
                "f_openpose_error": experiment_folder_name +  '/openpose_error.txt', 
                "f_openpose_arm_error": experiment_folder_name +  '/openpose_arm_error.txt',  
                "f_openpose_leg_error": experiment_folder_name +  '/openpose_leg_error.txt',
                "f_openpose_results": experiment_folder_name +  '/openpose_results.txt'}

    f_notes_name = main_folder_name + "/notes.txt"
    return file_names, folder_names, f_notes_name, date_time_name

def fill_notes(f_notes_name, parameters, energy_parameters, active_parameters):
    f_notes = open(f_notes_name, 'w')
    notes_str = "General Parameters:\n"
    for key, value in parameters.items():
        if (key !=  "FILE_NAMES" and key != "FOLDER_NAMES"):
            notes_str += str(key) + " : " + str(value)
            notes_str += '\n'

    notes_str += '\nEnergy Parameters:\n'
    for key, value in energy_parameters.items():
        notes_str += str(key) + " : " + str(value)
        notes_str += '\n'

    notes_str += '\nActive motion Parameters:\n'
    for key, value in active_parameters.items():
        notes_str += str(key) + " : " + str(value)
        notes_str += '\n'

    f_notes.write(notes_str)
    f_notes.close()

def append_error_notes(f_notes_name, err_1, err_2):
    f_notes = open(f_notes_name, 'a')
    notes_str = "\n---\nResults:\n"
    notes_str += "last frame error: ave:" + str(np.mean(np.array(err_1), axis=0)) + '\tstd:' + str(np.std(np.array(err_1), axis=0)) +"\n"
    notes_str += "mid frame error: ave:" + str(np.mean(np.array(err_2), axis=0)) + '\tstd:' + str(np.std(np.array(err_2), axis=0)) +"\n"
    f_notes.write(notes_str)
    f_notes.close()

def plot_error(gt_hp_arr, est_hp_arr, gt_hv_arr, est_hv_arr, errors, folder_name):
    #PLOT STUFF HERE AT THE END OF SIMULATION
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    p1, =ax.plot(est_hp_arr[:, 0], est_hp_arr[:, 1], est_hp_arr[:, 2], c='r', marker='^', label="Estimated")
    p2, =ax.plot(gt_hp_arr[:, 0], gt_hp_arr[:, 1], gt_hp_arr[:, 2], c='b', marker='^', label="GT")
    plt.legend(handles=[p1, p2])

    plt.title(str(errors["error_ave_pos"]))
    plt.savefig(folder_name + '/est_pos_final.png', bbox_inches='tight', pad_inches=0)
    #plt.close()

    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    p1, = ax.plot(est_hv_arr[:, 0], est_hv_arr[:, 1], est_hv_arr[:, 2], c='r', marker='^', label="Estimated")
    p2, = ax.plot(gt_hv_arr[:, 0], gt_hv_arr[:, 1], gt_hv_arr[:, 2], c='b', marker='^', label="GT")
    plt.legend(handles=[p1, p2])
    plt.title(str(errors["error_ave_vel"]))
    plt.savefig(folder_name + '/est_vel_final.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    #################

def simple_plot(data, folder_name, plot_name, plot_title="", x_label="", y_label=""):
    _ = plt.figure()
    average = sum(data)/len(data)
    p1, = plt.plot(data, label="Average: "+str(average))
    if (plot_title != ""): 
        plt.title(plot_title)
    if (x_label != ""): 
       plt.xlabel(x_label)
    if (y_label != ""): 
        plt.ylabel(y_label)
    plt.legend(handles=[p1])
    plt.savefig(folder_name + '/' + plot_title + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def simple_plot2(xdata, ydata, folder_name, plot_name, plot_title="", x_label="", y_label=""):
    _ = plt.figure()
    _, = plt.semilogx(xdata, ydata)
    if (plot_title != ""): 
        plt.title(plot_title)
    if (x_label != ""): 
       plt.xlabel(x_label)
    if (y_label != ""): 
        plt.ylabel(y_label)
    plt.savefig(folder_name + '/' + plot_title + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_matrix(matrix, plot_loc, ind, plot_title, custom_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.title(plot_title)    
    matrix_plot_loc = plot_loc +'/'+ custom_name + str(ind) + '.png'
    plt.savefig(matrix_plot_loc, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_matrices(matrix1, matrix2, plot_loc, ind, custom_name):
    fig = plt.figure()
    ax1 = plt.subplot(1,2,1)
    im = ax1.imshow(matrix1)
    plt.title("current frame")    

    ax2 = plt.subplot(1,2,2)
    im = ax2.imshow(matrix2)
    plt.title("future frame")    

    divider = make_axes_locatable(ax2)

    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    matrix_plot_loc = plot_loc +'/'+ custom_name + str(ind) + '.png'
    plt.savefig(matrix_plot_loc, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_covariances(pose_client, plot_loc, custom_name):    
    calib_list_min = []
    calib_list_max = []
    for img in pose_client.calib_cov_list:
        calib_list_min.append(img.min())
        calib_list_max.append(img.max())
    vmin_calib = min(calib_list_min)
    vmax_calib = max(calib_list_max)

    online_list_min = []
    online_list_max = []
    for ele in pose_client.online_cov_list:
        curr = ele["curr"]
        future = ele["future"]
        online_list_min.append(curr.min())
        online_list_max.append(curr.max())
        online_list_min.append(future.min())
        online_list_max.append(future.max())
    vmin_online = min(online_list_min)
    vmax_online = max(online_list_max)

    cmap = cm.viridis
    norm_calib = colors.Normalize(vmin=vmin_calib, vmax=vmax_calib)
    norm_online = colors.Normalize(vmin=vmin_online, vmax=vmax_online)

    for ind, matrix1 in enumerate(pose_client.calib_cov_list):
        fig = plt.figure()
        ax1 = plt.subplot(1,1,1)

        im = ax1.imshow(matrix1, cmap=cmap, norm=norm_calib)
        fig.subplots_adjust(right=0.8, hspace = 0.5)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        plt.suptitle("Current Frame Covariance Matrix")  
        matrix_plot_loc = plot_loc +'/'+ custom_name + str(ind+1) + '.png'
        plt.savefig(matrix_plot_loc, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    for ind, ele in enumerate(pose_client.online_cov_list):
        curr = ele["curr"]
        future = ele["future"]
        fig, _ = plt.subplots(1,2)

        ax1 = plt.subplot(1,2,1)
        _ = ax1.imshow(curr, cmap=cmap, norm=norm_online)
        plt.title("Current frame")    

        ax2 = plt.subplot(1,2,2)
        im2 = ax2.imshow(future, cmap=cmap, norm=norm_online)
        plt.title("Future frame")  

        fig.suptitle('Covariance matrix')
        fig.subplots_adjust(right=0.8, hspace = 0.5)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im2, cax=cbar_ax)
        matrix_plot_loc = plot_loc +'/'+ custom_name + str(pose_client.CALIBRATION_LENGTH+ind+1) + '.png'
        plt.savefig(matrix_plot_loc, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

def superimpose_on_image(openpose, plot_loc, ind, bone_connections, photo_location, custom_name=None, scale=-1, projection = np.zeros([1,1])):
    if custom_name == None:
        name = '/superimposed_'
    else: 
        name = '/'+custom_name

    superimposed_plot_loc = plot_loc + name + str(ind) + '.png'

    im = plt.imread(photo_location)
    im = np.array(im[:,:,0:3])

    if (scale != -1):
        scale_ = scale / im.shape[0]
        im = cv2.resize(im, (0, 0), fx=scale_, fy=scale_, interpolation=cv2.INTER_CUBIC)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im)

    if np.count_nonzero(projection) != 0:
        for i, bone in enumerate(bone_connections):
            p0, = ax.plot( projection[0, bone], projection[1,bone], color = "w", linewidth=3, label="Reprojection")

    left_bone_connections, right_bone_connections, middle_bone_connections = split_bone_connections(bone_connections)
    for i, bone in enumerate(left_bone_connections):    
        p1, = ax.plot( openpose[0, bone], openpose[1,bone], color = "r", linewidth=1, label="OpenPose Left")   
    for i, bone in enumerate(right_bone_connections):    
        p2, = ax.plot( openpose[0, bone], openpose[1,bone], color = "b", linewidth=1, label="OpenPose Right")   
    for i, bone in enumerate(middle_bone_connections):    
        ax.plot( openpose[0, bone], openpose[1,bone], color = "b", linewidth=1) 

    if np.count_nonzero(projection) != 0:
        plot_handles = [p0,p1,p2]
    else:
        plot_handles = [p1,p2]

    plt.legend(handles=plot_handles)
    plt.savefig(superimposed_plot_loc, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_heatmaps(heatmaps, ind, plot_loc, custom_name=None, scales=None, poses=None, bone_connections=None):
    if custom_name == None:
        name = '/heatmaps'
    else: 
        name = '/'+custom_name
    
    fig = plt.figure()
    if (heatmaps.ndim == 3):
        ave_heatmaps = np.mean(heatmaps, axis=0)
        plt.imshow(ave_heatmaps)
    elif (heatmaps.ndim == 4):
        left_bone_connections, right_bone_connections, middle_bone_connections = split_bone_connections(bone_connections)
        ave_heatmaps_scales = np.mean(heatmaps, axis=1)
        for scale_ind in range(0, ave_heatmaps_scales.shape[0]):
            plt.subplot(2, int(ceil(ave_heatmaps_scales.shape[0]/2)),scale_ind+1)
            plt.imshow(ave_heatmaps_scales[scale_ind, :, :])
            for i, bone in enumerate(left_bone_connections):    
                plt.plot( poses[scale_ind, 0, bone], poses[scale_ind, 1,bone], color = "r", linewidth=2)   
            for i, bone in enumerate(right_bone_connections):    
                plt.plot( poses[scale_ind, 0, bone], poses[scale_ind, 1,bone], color = "b", linewidth=2)   
            for i, bone in enumerate(middle_bone_connections):    
                plt.plot( poses[scale_ind, 0, bone], poses[scale_ind, 1,bone], color = "b", linewidth=2)   
            plt.title(str(scales[scale_ind]))

    heatmap_loc = plot_loc + name + str(ind) + '.png'

    plt.savefig(heatmap_loc, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_image(img, ind, plot_loc, custom_name=None):
    if custom_name == None:
        name = '/cropped_image'
    else: 
        name = '/'+custom_name

    fig = plt.figure()
    plt.imshow(img)
    img_loc = plot_loc + name + str(ind) + '.png'
    plt.savefig(img_loc, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_human(bones_GT, predicted_bones, location, ind,  bone_connections, use_single_joint, error = -5, custom_name = None, orientation = "z_up", label_names =None, additional_text =None):   
    if custom_name == None:
        name = '/plot3d_'
    else: 
        name = '/'+custom_name

    if label_names == None:
        blue_label = "GT"
        red_label = "Estimate"
    else:
        blue_label = label_names[0]
        red_label = label_names[1]

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')

    X = bones_GT[0,:]
    if orientation == "z_up":
        # maintain aspect ratio
        Y = bones_GT[1,:]
        Z = -bones_GT[2,:]
        
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.8
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.view_init(elev=30., azim=135)

    if not use_single_joint:
        left_bone_connections, right_bone_connections, middle_bone_connections = split_bone_connections(bone_connections)

        #plot joints
        for i, bone in enumerate(left_bone_connections):
            plot1, = ax.plot(bones_GT[0,bone], bones_GT[1,bone], -bones_GT[2,bone], c='xkcd:light blue', marker='^', label=blue_label + " left")
        for i, bone in enumerate(right_bone_connections):
            plot1_r, = ax.plot(bones_GT[0,bone], bones_GT[1,bone], -bones_GT[2,bone], c='xkcd:royal blue', marker='^', label=blue_label + " right")
        for i, bone in enumerate(middle_bone_connections):
            ax.plot(bones_GT[0,bone], bones_GT[1,bone], -bones_GT[2,bone], c='xkcd:royal blue', marker='^')

        for i, bone in enumerate(left_bone_connections):
            plot2, = ax.plot(predicted_bones[0,bone], predicted_bones[1,bone], -predicted_bones[2,bone], c='xkcd:light red', marker='^', label=red_label + " left")
        for i, bone in enumerate(right_bone_connections):
            plot2_r, = ax.plot(predicted_bones[0,bone], predicted_bones[1,bone], -predicted_bones[2,bone], c='xkcd:blood red', marker='^', label=red_label + " right")
        for i, bone in enumerate(middle_bone_connections):
            ax.plot(predicted_bones[0,bone], predicted_bones[1,bone], -predicted_bones[2,bone], c='xkcd:blood red', marker='^')
        ax.legend(handles=[plot1, plot1_r, plot2, plot2_r])
    else:
        plot1, = ax.plot(bones_GT[0,:], bones_GT[1,:], -bones_GT[2,:], c='xkcd:royal blue', marker='^')
        plot2, = ax.plot(predicted_bones[0,:], predicted_bones[1,:], -predicted_bones[2,:], c='xkcd:blood red', marker='^')
        #ax.legend(handles=[plot1, plot2])

    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


    if (error != -5):
        ax.text2D(0, 0.3, "error: %.4f" %error, transform=ax.transAxes)
        if (additional_text != None):
            ax.text2D(0, 0.35, "running ave error: %.4f" %additional_text, transform=ax.transAxes)

    plt.title("3D Human Pose")
    plot_3d_pos_loc = location + name + str(ind) + '.png'
    plt.savefig(plot_3d_pos_loc)
    plt.close()


def plot_drone_traj(pose_client, plot_loc, ind):
    if (pose_client.isCalibratingEnergy):
        plot_info = pose_client.calib_res_list
        file_name = plot_loc + '/drone_traj_'+ str(ind) + '.png'
    else:
        plot_info = pose_client.online_res_list
        file_name = plot_loc + '/drone_traj_'+ str(ind) + '.png'
    file_name_2 = plot_loc + '/drone_traj_2_'+ str(ind) + '.png'
    
    bone_connections, _, _, _ = pose_client.model_settings()
    left_bone_connections, right_bone_connections, middle_bone_connections = split_bone_connections(bone_connections)

    # fig = plt.figure( figsize=(12, 4))
    # ax = fig.add_subplot(151, projection='3d')

    # last_frame_plot_info = plot_info[-1]
    # predicted_bones = last_frame_plot_info["est"]
    # bones_GT = last_frame_plot_info["GT"]

    # X = np.concatenate([bones_GT[0,:], predicted_bones[0,:]])
    # Y = np.concatenate([bones_GT[1,:], predicted_bones[1,:]])
    # Z = np.concatenate([-bones_GT[2,:], -predicted_bones[2,:]])

    # #plot drone
    # drone_x, drone_y, drone_z = [],[],[]
    # for frame_ind in range (0, len(plot_info)):
    #     frame_plot_info = plot_info[frame_ind]
    #     drone = frame_plot_info["drone"].squeeze()
    #     drone_x.append(drone[0])
    #     drone_y.append(drone[1])
    #     drone_z.append(-drone[2])

    #     X = np.concatenate([X, [drone[0]]])
    #     Y = np.concatenate([Y, [drone[1]]])
    #     Z = np.concatenate([Z, [-drone[2]]])

    # plotd, = ax.plot(drone_x, drone_y, drone_z, c='xkcd:black', marker='^', label="drone")

    # #plot final frame human
    # for i, bone in enumerate(left_bone_connections):
    #     plot1, = ax.plot(bones_GT[0,bone], bones_GT[1,bone], -bones_GT[2,bone], c='xkcd:light blue', label="GT left")
    # for i, bone in enumerate(right_bone_connections):
    #     plot1_r, = ax.plot(bones_GT[0,bone], bones_GT[1,bone], -bones_GT[2,bone], c='xkcd:royal blue', label="GT right")
    # for i, bone in enumerate(middle_bone_connections):
    #     ax.plot(bones_GT[0,bone], bones_GT[1,bone], -bones_GT[2,bone], c='xkcd:royal blue')

    # for i, bone in enumerate(left_bone_connections):
    #     plot2, = ax.plot(predicted_bones[0,bone], predicted_bones[1,bone], -predicted_bones[2,bone], c='xkcd:light red', label="estimate left")
    # for i, bone in enumerate(right_bone_connections):
    #     plot2_r, = ax.plot(predicted_bones[0,bone], predicted_bones[1,bone], -predicted_bones[2,bone], c='xkcd:blood red', label="right left")
    # for i, bone in enumerate(middle_bone_connections):
    #     ax.plot(predicted_bones[0,bone], predicted_bones[1,bone], -predicted_bones[2,bone], c='xkcd:blood red')

    # ax.legend(handles=[plot1, plot1_r, plot2, plot2_r, plotd])

    # max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.4
    # mid_x = (X.max()+X.min()) * 0.5
    # mid_y = (Y.max()+Y.min()) * 0.5
    # mid_z = (Z.max()+Z.min()) * 0.5
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.title("Drone Trajectory")

    # ax = fig.add_subplot(152)
    # ax.plot(drone_x, drone_y, c='xkcd:black', marker='^')
    # plt.title("top down")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")

    # ax = fig.add_subplot(153)
    # ax.plot(drone_z, c='xkcd:black', marker='^')
    # plt.title("z")
    # ax.set_xlabel("frame")
    # ax.set_ylabel("z")

    # ax = fig.add_subplot(154)
    # ax.plot(drone_x, c='xkcd:black', marker='^')
    # plt.title("x")
    # ax.set_xlabel("frame")
    # ax.set_ylabel("x")

    # ax = fig.add_subplot(155)
    # ax.plot(drone_y, c='xkcd:black', marker='^')
    # plt.title("y")
    # ax.set_xlabel("frame")
    # ax.set_ylabel("y")

    # plt.savefig(file_name)
    # plt.close()

#####################
    fig = plt.figure( figsize=(4, 4))
    left_bone_connections, right_bone_connections, middle_bone_connections = split_bone_connections(bone_connections)
    ax = fig.add_subplot(111, projection='3d')

    last_frame_plot_info = plot_info[-1]
    predicted_bones = last_frame_plot_info["est"]
    bones_GT = last_frame_plot_info["GT"]

    X = np.concatenate([bones_GT[0,:], predicted_bones[0,:]])
    Y = np.concatenate([bones_GT[1,:], predicted_bones[1,:]])
    Z = np.concatenate([-bones_GT[2,:], -predicted_bones[2,:]])

    #plot drone
    drone_x, drone_y, drone_z = [],[],[]
    for frame_ind in range (0, len(plot_info)):
        frame_plot_info = plot_info[frame_ind]
        drone = frame_plot_info["drone"].squeeze()
        drone_x.append(drone[0])
        drone_y.append(drone[1])
        drone_z.append(-drone[2])

        X = np.concatenate([X, [drone[0]]])
        Y = np.concatenate([Y, [drone[1]]])
        Z = np.concatenate([Z, [-drone[2]]])

    plotd, = ax.plot(drone_x, drone_y, drone_z, c='xkcd:black', marker='^', label="drone")

    #plot final frame human
    if not pose_client.USE_SINGLE_JOINT:
        for i, bone in enumerate(left_bone_connections):
            plot1, = ax.plot(bones_GT[0,bone], bones_GT[1,bone], -bones_GT[2,bone], c='xkcd:light blue', label="GT left")
        for i, bone in enumerate(right_bone_connections):
            plot1_r, = ax.plot(bones_GT[0,bone], bones_GT[1,bone], -bones_GT[2,bone], c='xkcd:royal blue', label="GT right")
        for i, bone in enumerate(middle_bone_connections):
            ax.plot(bones_GT[0,bone], bones_GT[1,bone], -bones_GT[2,bone], c='xkcd:royal blue')

        for i, bone in enumerate(left_bone_connections):
            plot2, = ax.plot(predicted_bones[0,bone], predicted_bones[1,bone], -predicted_bones[2,bone], c='xkcd:light red', label="estimate left")
        for i, bone in enumerate(right_bone_connections):
            plot2_r, = ax.plot(predicted_bones[0,bone], predicted_bones[1,bone], -predicted_bones[2,bone], c='xkcd:blood red', label="right left")
        for i, bone in enumerate(middle_bone_connections):
            ax.plot(predicted_bones[0,bone], predicted_bones[1,bone], -predicted_bones[2,bone], c='xkcd:blood red')
        ax.legend(handles=[plot1, plot1_r, plot2, plot2_r, plotd])
    else:
        plot1, = ax.plot(predicted_bones[0,:], predicted_bones[1,:], -predicted_bones[2,:], c='xkcd:blood red')
        plot2, = ax.plot(bones_GT[0,:], bones_GT[1,:], -bones_GT[2,:], c='xkcd:royal blue')
        #ax.legend(handles=[plot1,plot2]

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.4
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    #ax.set_xlim(predicted_bones[0,0]-7, predicted_bones[0,0] +7)
    #ax.set_ylim(predicted_bones[1,0]-7, predicted_bones[1,0] +7)
    #ax.set_zlim(-predicted_bones[2,0]-2, -predicted_bones[2,0] +9)

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Drone Trajectory")
    plt.savefig(file_name_2)
    plt.close()

def plot_optimization_losses(pltpts, location, ind, loss_dict):
    plt.figure()
    for loss_ind, loss_key in enumerate(loss_dict):
        x_axis = np.linspace(1,len(pltpts[loss_key]),len(pltpts[loss_key]))
        plt.subplot(1,len(loss_dict),loss_ind+1)
        plt.semilogy(x_axis, pltpts[loss_key])
        plt.xlabel("iter")
        plt.title(loss_key)
    plot_3d_pos_loc = location + '/loss_' + str(ind) + '.png'
    plt.savefig(plot_3d_pos_loc, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_2d_projection(pose, plot_loc, ind, bone_connections, custom_name="proj_2d"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.ones([576,1024]))

    left_bone_connections, right_bone_connections, middle_bone_connections = split_bone_connections(bone_connections)
    for _, bone in enumerate(left_bone_connections):    
        p1, = ax.plot( pose[0, bone], pose[1,bone], color = "r", linewidth=1, label="Left")   
    for i, bone in enumerate(right_bone_connections):    
        p2, = ax.plot( pose[0, bone], pose[1,bone], color = "b", linewidth=1, label="Right")   
    for i, bone in enumerate(middle_bone_connections):    
        ax.plot(pose[0, bone], pose[1,bone], color = "b", linewidth=1)   
    ax.set_title(str(ind))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False) 

    plot_3d_pos_loc = plot_loc + '/' +custom_name+ "_" + str(ind) + '.png'
    plt.savefig(plot_3d_pos_loc, bbox_inches='tight', pad_inches=0)
    plt.close()

def vector3r_arr_to_dict(input):
    output = dict()
    for attribute in airsim.attributes:
        output[attribute] = getattr(input, attribute)
    return output

def normalize_pose(pose_3d_input, hip_index, is_torch = True):
    if (is_torch):
        hip_pos = pose_3d_input[:, hip_index].unsqueeze(1)
        relative_pos = torch.sub(pose_3d_input, hip_pos)
        max_z = torch.max(relative_pos[2,:])
        min_z = torch.min(relative_pos[2,:])
        result = (relative_pos)/(max_z - min_z)
    else:
        hip_pos = pose_3d_input[:, hip_index]
        relative_pos = pose_3d_input - hip_pos[:, np.newaxis]
        max_z = np.max(relative_pos[2,:])
        min_z = np.min(relative_pos[2,:])
        result = (relative_pos)/(max_z - min_z)
    return result, relative_pos

def numpy_to_tuples(pose_2d):
    tuple_list = []
    for i in range(0, pose_2d.shape[1]):
        tuple_list.append(pose_2d[:, i].tolist())
    return tuple_list

def create_heatmap(kpt, grid_x, grid_y, stride=1, sigma=15):
    """
    Creates the heatmap of the given size with the given joints.
    """
    heatmap = np.zeros((kpt.shape[1]+1, grid_y, grid_x), dtype='float32')
    num_point, height, width = heatmap.shape

    length = kpt.shape[1]

    x = np.arange(0, grid_x, 1)
    y = np.arange(0, grid_y, 1)
    xx, yy = np.meshgrid(x, y)

    for j in range(length):
        x = kpt[0,j]
        y = kpt[1,j]
        
        dis = ((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma / sigma
        heatmap[j,:,:] = np.exp(-dis)
        heatmap[j, dis > 4.6052] = 0

    # Add the background channel
    heatmap[-1, :, :] = 1.0 - np.max(heatmap[:-1, :, :], axis=0)

    return heatmap

def matrix_to_ellipse(matrix, center):
    _, s, rotation = np.linalg.svd(matrix)
    radii = 3*np.sqrt(s)/max_radii

    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

    return x,y,z


def shape_cov(cov, hip_index, num_of_joints, frame_index):
    H = np.zeros([3,3])
    offset = hip_index + frame_index*3*num_of_joints
    H[:,0] = np.array([cov[offset, offset], cov[num_of_joints+offset, offset], cov[num_of_joints*2+offset, offset],])
    H[:,1] = np.array([cov[offset, num_of_joints+offset], cov[num_of_joints+offset, num_of_joints+offset], cov[num_of_joints*2+offset, num_of_joints+offset],])
    H[:,2] = np.array([cov[offset, num_of_joints*2+offset], cov[num_of_joints+offset, num_of_joints*2+offset], cov[num_of_joints*2+offset, num_of_joints*2+offset],])
    return H

def shape_cov_ave_joints(cov, num_of_joints):
    new_cov = np.cov([3,3])
    for i in range(3):
        for j in range(3):
                small_cov = cov[i*num_of_joints:(i+1)*num_of_joints, j*num_of_joints:(j+1)*num_of_joints]
                new_cov[i, j] = np.mean(small_cov.flatten())
    return new_cov

def choose_frame_from_cov(cov, frame_index, num_of_joints):
    return cov[frame_index*(3*num_of_joints):(frame_index+1)*(3*num_of_joints), frame_index*(3*num_of_joints):(frame_index+1)*(3*num_of_joints)]

def shape_cov_hip(cov, frame_index):
    H = np.zeros([3,3])
    offset = frame_index*3
    H[:,0] = np.array([cov[offset, offset], cov[1+offset, offset], cov[2+offset, offset],])
    H[:,1] = np.array([cov[offset, 1+offset], cov[1+offset, 1+offset], cov[2+offset, 1+offset],])
    H[:,2] = np.array([cov[offset, 2+offset], cov[1+offset, 2+offset], cov[2+offset, 2+offset],])
    return H

def shape_cov_mini(cov, hip_index, frame_index):
    H = np.zeros([3,3])
    H[:,0] = np.array([cov[hip_index, hip_index], cov[1+hip_index, hip_index], cov[2+hip_index, hip_index],])
    H[:,1] = np.array([cov[hip_index, 1+hip_index], cov[1+hip_index, 1+hip_index], cov[2+hip_index, 1+hip_index],])
    H[:,2] = np.array([cov[hip_index, 2+hip_index], cov[1+hip_index, 2+hip_index], cov[2+hip_index, 2+hip_index],])
    return H

def shape_cov_general(cov, num_of_joints, frame_index = 0):    
    H = np.zeros([num_of_joints,3,3])
    for joint_ind in range(num_of_joints):
        x = frame_index*(3*num_of_joints)+joint_ind+(0*num_of_joints)
        y = frame_index*(3*num_of_joints)+joint_ind+(1*num_of_joints)
        z = frame_index*(3*num_of_joints)+joint_ind+(2*num_of_joints)
        H[joint_ind, :, 0] = np.array([cov[x,x], cov[x,y], cov[x,z],])
        H[joint_ind, :, 1] = np.array([cov[y,x], cov[y,y], cov[y,z],])
        H[joint_ind, :, 2] = np.array([cov[z,x], cov[z,y], cov[z,z],])
    return H

def plot_covariance_as_ellipse(pose_client, plot_loc, ind):
    if (pose_client.isCalibratingEnergy):
        plot_info = pose_client.calib_res_list
        file_name = plot_loc + '/ellipse_calib_'+ str(ind) + '.png'
    else:
        plot_info = pose_client.online_res_list
        file_name = plot_loc + '/ellipse_online_'+ str(ind) + '.png'

    bone_connections, _, _, hip_index = pose_client.model_settings()
    left_bone_connections, right_bone_connections, middle_bone_connections = split_bone_connections(bone_connections)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #plot final frame human
    last_frame_plot_info = plot_info[-1]
    predicted_bones = last_frame_plot_info["est"]
    bones_GT = last_frame_plot_info["GT"]
    for i, bone in enumerate(left_bone_connections):
        plot1, = ax.plot(bones_GT[0,bone], bones_GT[1,bone], -bones_GT[2,bone], c='xkcd:light blue', label="GT left")
    for i, bone in enumerate(right_bone_connections):
        plot1_r, = ax.plot(bones_GT[0,bone], bones_GT[1,bone], -bones_GT[2,bone], c='xkcd:royal blue', label="GT right")
    for i, bone in enumerate(middle_bone_connections):
        ax.plot(bones_GT[0,bone], bones_GT[1,bone], -bones_GT[2,bone], c='xkcd:royal blue')

    for i, bone in enumerate(left_bone_connections):
        plot2, = ax.plot(predicted_bones[0,bone], predicted_bones[1,bone], -predicted_bones[2,bone], c='xkcd:light red', label="estimate left")
    for i, bone in enumerate(right_bone_connections):
        plot2_r, = ax.plot(predicted_bones[0,bone], predicted_bones[1,bone], -predicted_bones[2,bone], c='xkcd:blood red', label="right left")
    for i, bone in enumerate(middle_bone_connections):
        ax.plot(predicted_bones[0,bone], predicted_bones[1,bone], -predicted_bones[2,bone], c='xkcd:blood red')

    X = np.concatenate([bones_GT[0,:], predicted_bones[0,:]])
    Y = np.concatenate([bones_GT[1,:], predicted_bones[1,:]])
    Z = np.concatenate([-bones_GT[2,:], -predicted_bones[2,:]])

    #plot drone
    for frame_ind in range (0, len(plot_info), 3):
        frame_plot_info = plot_info[frame_ind]
        drone = frame_plot_info["drone"]
        plotd, = ax.plot(drone[0], drone[1], -drone[2], c='xkcd:lime', marker='^', label="drone")

        X = np.concatenate([X, drone[0]])
        Y = np.concatenate([Y, drone[1]])
        Z = np.concatenate([Z, -drone[2]])

    #plot ellipse
    center = np.copy(predicted_bones[:, hip_index])
    center[2]= -center[2]
    x,y,z = matrix_to_ellipse(pose_client.measurement_cov, center)
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
    ax.legend(handles=[plot1, plot1_r, plot2, plot2_r, plotd])

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.45
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def rotation_matrix_to_euler(R) :
    sy = sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = atan2(R[2,1] , R[2,2])
        y = atan2(-R[2,0], sy)
        z = atan2(R[1,0], R[0,0])
    else :
        x = atan2(-R[1,2], R[1,1])
        y = atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def euler_to_rotation_matrix(roll, pitch, yaw, returnTensor=True):
    if (returnTensor == True):
        return torch.FloatTensor([[cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll), cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll)],
                    [sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll), sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll)],
                    [-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)]])
    return np.array([[cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll), cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll)],
                    [sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll), sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll)],
                    [-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)]])


def plot_potential_states(current_human_pose, future_human_pose, gt_human_pose, potential_states, C_drone, R_drone, hip_index, plot_loc, ind):
    current_human_pos = current_human_pose[0:2, hip_index]
    future_human_pos =  future_human_pose[0:2, hip_index]
    gt_human_pos = gt_human_pose[0:2, hip_index]
    
    fig, ax = plt.subplots()
    plt.axis(v=['scaled'])

    #plot the people
    plot1, = ax.plot(float(current_human_pos[0]), float(current_human_pos[1]), c='xkcd:light red', marker='^', label="current human pos")
    plot2, = ax.plot(float(future_human_pos[0]), float(future_human_pos[1]), c='xkcd:royal blue', marker='^', label="future human pos")
    plot5, = ax.plot(float(gt_human_pos[0]), float(gt_human_pos[1]), c='xkcd:orchid', marker='^', label="GT current human pos")

    #plot potential states
    for state_ind, potential_state in enumerate(potential_states):
        yaw = potential_state.orientation
        state_pos =  potential_state.position
        plot4, = plt.plot([float(state_pos[0] - (cos(yaw)*0.5)/2), float(state_pos[0] + (cos(yaw)*0.5)/2)], [float(state_pos[1] - (sin(yaw)*0.5)/2), float(state_pos[1] + (sin(yaw)*0.5)/2)], c='xkcd:hot pink', label="potential state")
        plt.text(state_pos[0], state_pos[1], str(state_ind))
    
    #plot current drone state
    current_drone_pos = C_drone[0:2]
    _,_,yaw = rotation_matrix_to_euler(R_drone)
    plot3, = plt.plot([float(current_drone_pos[0] - (cos(yaw)*0.5)/2), float(current_drone_pos[0] + (cos(yaw)*0.5)/2)], [float(current_drone_pos[1] - (sin(yaw)*0.5)/2), float(current_drone_pos[1] + (sin(yaw)*0.5)/2)], c='xkcd:black', label="current drone pos")

    plt.legend(handles=[plot1, plot2, plot3, plot4, plot5])
    #plt.show()
    file_name = plot_loc + "/potential_states_" + str(ind) + ".png"
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_potential_hessians(hessians, linecount, plot_loc, custom_name = None):
    if custom_name == None:
        name = '/potential_covs_'
    else: 
        name = '/'+custom_name

    if (len(hessians) > 6):
        nrows, ncols = 3, 3
    elif (len(hessians) > 3):
        nrows, ncols = 3, 2
    else:
        nrows, ncols = 3, 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8,8))

    list_min = []
    list_max = []
    for img in hessians:
        list_min.append(img.min())
        list_max.append(img.max())
    vmin = min(list_min)
    vmax = max(list_max)

    cmap = cm.viridis
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    plt.suptitle("Covariance matrices for potential states")
    for ind in range(min(nrows*ncols, len(hessians))):
        hess = hessians[ind]
        #shaped_hess = shape_cov_ave_joints(hess, num_of_joints)
        shaped_hess = hess
        ax = axes.flat[ind]
        im = ax.imshow(shaped_hess, cmap=cmap, norm=norm)
        ax.set_title(str(ind))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #eigenvals, _ = np.linalg.eig(hess)
        #uncertainty1 = float("{0:.4f}".format(np.linalg.det(hess)))
        #uncertainty2 = float("{0:.4f}".format(np.max(eigenvals)))
        #uncertainty3 = float("{0:.4f}".format(np.sum(eigenvals)))
        #ax.text(0.05,5,str(uncertainty1), color="white")
        #ax.text(0.05,10,str(uncertainty2), color="white")   
        #ax.text(0.05,15,str(uncertainty3), color="white")   

    fig.subplots_adjust(right=0.8, hspace = 0.5)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    file_name = plot_loc + name + str(linecount) + ".png"
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0, dpi=1000)
    plt.close(fig)
    
def plot_potential_projections(pose2d_list, linecount, plot_loc, photo_loc, bone_connections):
    left_bone_connections, right_bone_connections, middle_bone_connections = split_bone_connections(bone_connections)

    superimposed_plot_loc = plot_loc + "/potential_projections_" + str(linecount) + '.png'

    if (len(pose2d_list) > 6):
        nrows, ncols = 3, 3
    elif (len(pose2d_list) > 3):
        nrows, ncols = 3, 2
    else:
        nrows, ncols = 3, 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    im = plt.imread(photo_loc)
    im = np.array(im[:,:,0:3])
    for ind, pose in enumerate(pose2d_list):
        ax = axes.flat[ind]
        ax.imshow(im)
    
        #plot part
        for _, bone in enumerate(left_bone_connections):    
            p1, = ax.plot( pose[0, bone], pose[1,bone], color = "r", linewidth=1, label="Left")   
        for i, bone in enumerate(right_bone_connections):    
            p2, = ax.plot( pose[0, bone], pose[1,bone], color = "b", linewidth=1, label="Right")   
        for i, bone in enumerate(middle_bone_connections):    
            ax.plot( pose[0, bone], pose[1,bone], color = "b", linewidth=1)   
        ax.set_title(str(ind))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False) 

    plt.savefig(superimposed_plot_loc, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_potential_projections_noimage(pose2d_list, linecount, plot_loc, bone_connections):
    left_bone_connections, right_bone_connections, middle_bone_connections = split_bone_connections(bone_connections)

    superimposed_plot_loc = plot_loc + "/potential_projections_noimage_" + str(linecount) + '.png'

    if (len(pose2d_list) > 6):
        nrows, ncols = 3, 3
    elif (len(pose2d_list) > 3):
        nrows, ncols = 3, 2
    else:
        nrows, ncols = 3, 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    im = np.zeros([SIZE_Y, SIZE_X])
    for ind, pose in enumerate(pose2d_list):
        ax = axes.flat[ind]
        ax.imshow(im)
    
        #plot part
        for _, bone in enumerate(left_bone_connections):    
            p1, = ax.plot( pose[0, bone], pose[1,bone], color = "r", linewidth=1, label="Left")   
        for i, bone in enumerate(right_bone_connections):    
            p2, = ax.plot( pose[0, bone], pose[1,bone], color = "b", linewidth=1, label="Right")   
        for i, bone in enumerate(middle_bone_connections):    
            ax.plot( pose[0, bone], pose[1,bone], color = "b", linewidth=1)   
        ax.set_title(str(ind))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False) 


    plt.savefig(superimposed_plot_loc, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_potential_errors(potential_states_fetcher, plot_loc, linecount):
    hip_index, num_of_joints = potential_states_fetcher.hip_index, potential_states_fetcher.number_of_joints
    current_human_pos = potential_states_fetcher.current_human_pos[:, hip_index]
    future_human_pos =  potential_states_fetcher.future_human_pos[:, hip_index]
    gt_human_pos = potential_states_fetcher.human_GT[:, hip_index]
    
    fig = plt.figure(figsize=(8,12))

    potential_states = potential_states_fetcher.potential_states_go
    uncertainty_list_whole = potential_states_fetcher.uncertainty_list_whole
    uncertainty_list_future = potential_states_fetcher.uncertainty_list_future
    overall_error_list = potential_states_fetcher.overall_error_list
    overall_std_list = potential_states_fetcher.overall_error_std_list 
    future_error_list = potential_states_fetcher.future_error_list
    future_std_list = potential_states_fetcher.future_error_std_list
    
    cmap = cm.cool
    norms = []
    axes = []
    titles = ["Overall Uncertainty", "Future Uncertainty", "Overall Error Mean", "Future Error Mean", "Overall Error Std", "Future Error Std"]
    lists = [uncertainty_list_whole, uncertainty_list_future, overall_error_list, future_error_list, overall_std_list, future_std_list]
    for ind, a_list in enumerate(lists):
        norms.append(colors.Normalize(vmin=(np.min(a_list)), vmax=(np.max(a_list))))
        axes.append(fig.add_subplot(3,2,ind+1, projection='3d'))

    #for ax limits
    X = np.array([current_human_pos[0], future_human_pos[0], gt_human_pos[0]])
    Y = np.array([current_human_pos[1], future_human_pos[1], gt_human_pos[1]])
    Z = np.array([-current_human_pos[2], -future_human_pos[2], -gt_human_pos[2]])

    #plot ellipses
    for state_ind, potential_state in enumerate(potential_states):
        state_pos =  potential_state.position
        center = np.copy(state_pos)
        center[2] = -center[2]

        X = np.concatenate([X, np.array([center[0]])])
        Y = np.concatenate([Y, np.array([center[1]])])
        Z = np.concatenate([Z, np.array([center[2]])])

        markersize=30
        text_color="b"
        if (state_ind == potential_states_fetcher.goal_state_ind):
            markersize=100
            text_color="r"
        for list_ind, a_list in enumerate(lists):
            plot5=axes[list_ind].scatter([center[0]], [center[1]], [center[2]], marker='^', c=[a_list[state_ind]], cmap=cmap, norm=norms[list_ind], s=markersize, alpha=1)
            if state_ind == 0:
                plt.colorbar(plot5, ax=axes[list_ind])#, shrink = 0.8)    

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.4
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5

    for ind, ax in enumerate(axes):
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(titles[ind])
        plot1, = ax.plot([current_human_pos[0]], [current_human_pos[1]], [-current_human_pos[2]], c='xkcd:light red', marker='*', label="current human pos")
        plot2, = ax.plot([gt_human_pos[0]], [gt_human_pos[1]], [-gt_human_pos[2]], c='xkcd:orchid', marker='*', label="GT current human pos")
        #ax.legend(handles=[plot1, plot2])

    file_name = plot_loc + "/potential_errors_" + str(linecount) + ".png"
    plt.savefig(file_name)
    plt.close(fig)


def plot_potential_ellipses(potential_states_fetcher, plot_loc, ind, ellipses=True, top_down=True, plot_errors=False):
    hip_index, num_of_joints = potential_states_fetcher.hip_index, potential_states_fetcher.number_of_joints
    current_human_pos = potential_states_fetcher.current_human_pos[:, hip_index]
    future_human_pos =  potential_states_fetcher.future_human_pos[:, hip_index]
    gt_human_pos = potential_states_fetcher.human_GT[:, hip_index]
    

    if top_down:
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(121, projection='3d')
        ax_top_down = fig.add_subplot(122) 
    else:
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')

    potential_states = potential_states_fetcher.potential_states_go
    covs = potential_states_fetcher.potential_covs_whole
    if plot_errors:
        error_list = potential_states_fetcher.error_list
        cmap = cm.cool
        norm = colors.Normalize(vmin=(np.min(error_list)), vmax=(np.max(error_list)))

    #plot the people
    plot1, = ax.plot([current_human_pos[0]], [current_human_pos[1]], [-current_human_pos[2]], c='xkcd:light red', marker='^', label="current human pos")
    plot2, = ax.plot([future_human_pos[0]], [future_human_pos[1]], [-future_human_pos[2]], c='xkcd:royal blue', marker='^', label="future human pos")
    plot3, = ax.plot([gt_human_pos[0]], [gt_human_pos[1]], [-gt_human_pos[2]], c='xkcd:orchid', marker='^', label="GT current human pos")

    #for ax limits
    X = np.array([current_human_pos[0], future_human_pos[0], gt_human_pos[0]])
    Y = np.array([current_human_pos[1], future_human_pos[1], gt_human_pos[1]])
    Z = np.array([-current_human_pos[2], -future_human_pos[2], -gt_human_pos[2]])

    #plot ellipses
    centers = []
    for state_ind, potential_state in enumerate(potential_states):
        state_pos =  potential_state.position
        center = np.copy(state_pos)
        center[2] = -center[2]
        centers.append(center)
    
    if ind < 3:
        radii_list = np.zeros([len(covs), 3])
        for state_ind, cov in enumerate(covs):
            shaped_cov = shape_cov(cov, hip_index, num_of_joints, FUTURE_POSE_INDEX)
            _, s, _ = np.linalg.svd(shaped_cov)
            radii = np.sqrt(s)
            radii_list[state_ind, :] = radii[0:3]
        global max_radii
        max_radii = np.max(radii_list)

    for state_ind, center in enumerate(centers):
        markersize=30
        text_color="b"
        if (state_ind == potential_states_fetcher.goal_state_ind):
            markersize=100
            text_color="r"
        if ellipses:
            x, y, z = matrix_to_ellipse(matrix=shape_cov(covs[state_ind], hip_index, num_of_joints, FUTURE_POSE_INDEX), center=center)
            ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
            ax.text(center[0], center[1], center[2], str(state_ind), color=text_color)
            if top_down:
                ax_top_down.plot(x,y)
                ax_top_down.text(center[0], center[1], str(state_ind), color=text_color)
        else:
            if plot_errors:
                ax.scatter([center[0]], [center[1]], [center[2]], marker='^', c=[error_list[state_ind]], cmap=cmap, norm=norm, s=markersize, alpha=1)
                point_text = '{0:d}:{1:.4f}'.format(state_ind, error_list[state_ind])
                ax.text(center[0], center[1], center[2], point_text, color="b")
            else:
                ax.plot([center[0]], [center[1]], [center[2]], marker='^', c=text_color, markersize=markersize)
                ax.text(center[0], center[1], center[2], str(state_ind))
                if top_down:
                    ax_top_down.text(center[0], center[1], str(state_ind))


        X = np.concatenate([X, np.array([center[0]])])
        Y = np.concatenate([Y, np.array([center[1]])])
        Z = np.concatenate([Z, np.array([center[2]])])
    #curr_center = centers[potential_states_fetcher.current_state_ind]
    #goal_center = centers[potential_states_fetcher.goal_state_ind]
    #ax.plot([goal_center[0]], [goal_center[1]], [goal_center[2]], marker='^', color='xkcd:dark orange')

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.4
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if top_down:
        ax_top_down.set_xlim(mid_x - max_range*2, mid_x + max_range*2)
        ax_top_down.set_ylim(mid_y - max_range*2, mid_y + max_range*2)
        ax_top_down.set_xlabel('X')
        ax_top_down.set_ylabel('Y')

    ax.legend(handles=[plot1, plot2, plot3])

    file_name = plot_loc + "/potential_ellipses_" + str(ellipses)+ "_" + str(ind) + ".png"
    plt.savefig(file_name)
    plt.close(fig)

def plot_correlations(pose_client, linecount, plot_loc):
    fig = plt.figure()
    ax_current = fig.add_subplot(121)
    ax_future = fig.add_subplot(122)

    corr = [pose_client.correlation_current, pose_client.correlation_future]
    for ind, ax in enumerate([ax_current, ax_future]):
        ax.plot(corr[ind], marker="^")
        point_text = 'Mean: {0:.4f}, Std:{1:.4f}'.format(np.mean(corr[ind]), np.std(corr[ind]))
        ax.text(0.02, 0.9, point_text, fontsize=14, transform=ax.transAxes)

    corr_plot_loc = plot_loc +'/correlations_' + str(linecount) + '.png'
    plt.savefig(corr_plot_loc, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    fig = plt.figure()
    ax_current = fig.add_subplot(121)
    ax_future = fig.add_subplot(122)

    corr = [pose_client.cosine_current, pose_client.cosine_future]
    for ind, ax in enumerate([ax_current, ax_future]):
        ax.plot(corr[ind], marker="^")
        point_text = 'Mean: {0:.4f}, Std:{1:.4f}'.format(np.mean(corr[ind]), np.std(corr[ind]))
        ax.text(0.02, 0.9, point_text, fontsize=14, transform=ax.transAxes)

    corr_plot_loc = plot_loc +'/cosine_metric_' + str(linecount) + '.png'
    plt.savefig(corr_plot_loc, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
