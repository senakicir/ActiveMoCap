import setup_path 
import airsim

import shutil
import skimage.io
import numpy as np
import torch as torch
from pandas import read_csv

from torch.autograd import Variable
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import time, os
import cv2
from math import degrees, radians, pi, ceil, exp, atan2, sqrt, cos, sin, acos


energy_mode = {1:True, 0:False}
LOSSES = ["proj", "smooth", "bone", "lift"]#, "smoothpose"]
CALIBRATION_LOSSES = ["proj", "sym"]
FUTURE_LOSSES = ["proj", "smooth", "bone", "lift"]#, "smoothpose"]
TOP_SPEED = 3

attributes = ['dronePos', 'droneOrient', 'humanPos', 'hip', 'right_up_leg', 'right_leg', 'right_foot', 'left_up_leg', 'left_leg', 'left_foot', 'spine1', 'neck', 'head', 'head_top','left_arm', 'left_forearm', 'left_hand','right_arm','right_forearm','right_hand', 'right_hand_tip', 'left_hand_tip' ,'right_foot_tip' ,'left_foot_tip']
TEST_SETS = {"t": "test_set_t", "05_08": "test_set_05_08", "38_03": "test_set_38_03", "64_06": "test_set_64_06", "02_01": "test_set_02_01"}
ANIM_TO_UNREAL = {"t": 0, "05_08": 1, "38_03": 2, "64_06": 3, "02_01": 4}

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
FOCAL_LENGTH = SIZE_X/2
px = SIZE_X/2
py = SIZE_Y/2
CAMERA_OFFSET_X = 45/100
CAMERA_OFFSET_Y = 0
CAMERA_OFFSET_Z = 0#-4.92
CAMERA_ROLL_OFFSET = 0
CAMERA_PITCH_OFFSET = 0
CAMERA_YAW_OFFSET = 0
FRAME_START_OPTIMIZING = 5

CURRENT_POSE_INDEX = 1
FUTURE_POSE_INDEX = 0
MIDDLE_POSE_INDEX = 3

def find_bone_map():
    bones_map_to_mpi = []
    for ind, value in enumerate(joint_names_mpi):
        bones_map_to_mpi.append(joint_names_h36m.index(value))
    return bones_map_to_mpi

bones_map_to_mpi = find_bone_map()

def rearrange_bones_to_mpi(bones_unarranged, is_torch = True):
    if (is_torch):
        bones_rearranged = Variable(torch.zeros(3, 15))
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

def model_settings(model, bone_pos_3d_GT = Variable(torch.zeros(3,21))):
    if (model == "mpi"):
        bone_pos_3d_GT = rearrange_bones_to_mpi(bone_pos_3d_GT)
        bone_connections = bones_mpi
        joint_names = joint_names_mpi
        num_of_joints = 15
    else:
        bone_connections = bones_h36m
        joint_names = joint_names_h36m
        num_of_joints = 21
    return bone_connections, joint_names, num_of_joints, bone_pos_3d_GT

def normalize_weights(weights_):    
    weights = {}
    weights_sum = sum(weights_.values())
    for loss_key in LOSSES:
        weights[loss_key] = weights_[loss_key]/weights_sum
    return weights

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

def find_M(plot_info, model):
    _,joint_names,num_of_joints,_= model_settings(model) 
    spine_index = joint_names.index('spine1')
    p_GT = np.zeros([3*len(plot_info),num_of_joints])
    p_est = np.zeros([3*len(plot_info),num_of_joints])
    for frame_ind, frame_plot_info in enumerate(plot_info):
        predicted_bones = frame_plot_info["est"]
        bones_GT = frame_plot_info["GT"]
        root_GT = bones_GT[:,spine_index]
        root_est = predicted_bones[:,spine_index]
        p_GT[3*frame_ind:3*(frame_ind+1),:]= bones_GT-root_GT[:, np.newaxis]
        p_est[3*frame_ind:3*(frame_ind+1),:]= predicted_bones-root_est[:, np.newaxis]


    #remove spine row from both arrays
    p_est = np.delete(p_est, spine_index, 1)
    p_GT = np.delete(p_GT, spine_index, 1)
    filename = "M_rel.txt"
    
    X = np.linalg.inv(np.dot(p_est.T, p_est))
    M = np.dot(np.dot(X, p_est.T), p_GT)

    M = np.insert(M, spine_index, 0, axis=1)
    M = np.insert(M, spine_index, 0, axis=0)
    M[spine_index, spine_index] = 1

    M_file = open(filename, 'w')
    M_str = ""
    for i in range(0, num_of_joints):
        for j in range(0, num_of_joints):
            M_str += str(M[i,j]) + "\t"
        if (i != num_of_joints-1):
            M_str += "\n"
    M_file.write(M_str)

    return M

def read_M(model, name = "M_rel"):
    filename = name+".txt"
    _,_,num_of_joints,_= model_settings(model)
    if os.path.exists(filename):
        X = read_csv(filename, sep='\t', header=None).ix[:,:].as_matrix().astype('float')     
        return X[:,0:num_of_joints]  
    else:
        return np.eye(num_of_joints)

def move_M(destination_folder):
    os.rename("M_rel.txt", destination_folder+"/M_rel.txt")

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
            file_names[key] = {"f_output": experiment_folder_name +  '/a_flight.txt', "f_groundtruth": experiment_folder_name +  '/groundtruth.txt'}

    f_notes_name = main_folder_name + "/notes.txt"
    return file_names, folder_names, f_notes_name, date_time_name


def fill_notes(f_notes_name, parameters, energy_parameters):
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
    fig1 = plt.figure()
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
    #plot part
    colors = ["y", "r"]

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


def plot_human(bones_GT, predicted_bones, location, ind,  bone_connections, error = -5, custom_name = None, orientation = "z_up", label_names =None, additional_text =None):   
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

    fig = plt.figure()
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
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


    if (error != -5):
        ax.text2D(0, 0.3, "error: %.4f" %error, transform=ax.transAxes)
        if (additional_text != None):
            ax.text2D(0, 0.35, "running ave error: %.4f" %additional_text, transform=ax.transAxes)

    plt.title("3D Human Pose")
    plot_3d_pos_loc = location + name + str(ind) + '.png'
    plt.savefig(plot_3d_pos_loc, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_global_motion(pose_client, plot_loc, ind):
    if (pose_client.isCalibratingEnergy):
        plot_info = pose_client.calib_res_list
        file_name = plot_loc + '/global_plot_calib_'+ str(ind) + '.png'
    else:
        plot_info = pose_client.online_res_list
        file_name = plot_loc + '/global_plot_online_'+ str(ind) + '.png'

    fig = plt.figure()
    bone_connections, _, _, _ = model_settings(pose_client.model)
    left_bone_connections, right_bone_connections, middle_bone_connections = split_bone_connections(bone_connections)
    ax = fig.add_subplot(111, projection='3d')
    for frame_ind in range (0, len(plot_info), 3):
        frame_plot_info = plot_info[frame_ind]
        predicted_bones = frame_plot_info["est"]
        bones_GT = frame_plot_info["GT"]
        drone = frame_plot_info["drone"]

        #plot joints
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

        plotd, = ax.plot(drone[0], drone[1], -drone[2], c='xkcd:lime', marker='^', label="drone")

        if frame_ind == 0:
            X = np.concatenate([np.concatenate([bones_GT[0,:], predicted_bones[0,:]]), drone[0]])
            Y = np.concatenate([np.concatenate([bones_GT[1,:], predicted_bones[1,:]]), drone[1]])
            Z = np.concatenate([np.concatenate([-bones_GT[2,:], -predicted_bones[2,:]]), -drone[2]])
        else:
            X = np.concatenate([X, np.concatenate([np.concatenate([bones_GT[0,:], predicted_bones[0,:]]), drone[0]])])
            Y = np.concatenate([Y, np.concatenate([np.concatenate([bones_GT[1,:], predicted_bones[1,:]]), drone[1]])])
            Z = np.concatenate([Z, np.concatenate([np.concatenate([-bones_GT[2,:], -predicted_bones[2,:]]), -drone[2]])])

    ax.legend(handles=[plot1, plot1_r, plot2, plot2_r, plotd])

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

    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_drone_traj(pose_client, plot_loc, ind):
    if (pose_client.isCalibratingEnergy):
        plot_info = pose_client.calib_res_list
        file_name = plot_loc + '/drone_traj_'+ str(ind) + '.png'
    else:
        plot_info = pose_client.online_res_list
        file_name = plot_loc + '/drone_traj_'+ str(ind) + '.png'

    fig = plt.figure()
    bone_connections, _, _, _ = model_settings(pose_client.model)
    left_bone_connections, right_bone_connections, middle_bone_connections = split_bone_connections(bone_connections)
    ax = fig.add_subplot(141, projection='3d')

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

    ax.legend(handles=[plot1, plot1_r, plot2, plot2_r, plotd])

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
    plt.title("Drone Trajectory")

    ax = fig.add_subplot(142)
    ax.plot(drone_z, c='xkcd:black', marker='^')
    plt.title("Drone Trajectory, z coordinate")
    ax.set_xlabel("frame")
    ax.set_ylabel("z")

    ax = fig.add_subplot(143)
    ax.plot(drone_x, c='xkcd:black', marker='^')
    plt.title("Drone Trajectory, x coordinate")
    ax.set_xlabel("frame")
    ax.set_ylabel("x")

    ax = fig.add_subplot(144)
    ax.plot(drone_y, c='xkcd:black', marker='^')
    plt.title("Drone Trajectory, y coordinate")
    ax.set_xlabel("frame")
    ax.set_ylabel("y")

    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
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

def vector3r_arr_to_dict(input):
    output = dict()
    for attribute in attributes:
        output[attribute] = getattr(input, attribute)
    return output

def normalize_pose(pose_3d_input, joint_names, is_torch = True):
    if (is_torch):
        hip_pos = pose_3d_input[:, joint_names.index('spine1')].unsqueeze(1)
        relative_pos = torch.sub(pose_3d_input, hip_pos)
        max_z = torch.max(relative_pos[2,:])
        min_z = torch.min(relative_pos[2,:])
        result = (relative_pos)/(max_z - min_z)
    else:
        hip_pos = pose_3d_input[:, joint_names.index('spine1')]
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

def matrix_to_ellipse(matrix, center, plot_scale = 1):
    _, s, rotation = np.linalg.svd(matrix)
    radii = np.sqrt(s)/plot_scale

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

def shape_cov(cov, model, frame_index):
    _, joint_names, num_of_joints, _ = model_settings(model)
    hip_index = joint_names.index('spine1')
    H = np.zeros([3,3])
    offset = hip_index + frame_index*3*num_of_joints
    H[:,0] = np.array([cov[offset, offset], cov[num_of_joints+offset, offset], cov[num_of_joints*2+offset, offset],])
    H[:,1] = np.array([cov[offset, num_of_joints+offset], cov[num_of_joints+offset, num_of_joints+offset], cov[num_of_joints*2+offset, num_of_joints+offset],])
    H[:,2] = np.array([cov[offset, num_of_joints*2+offset], cov[num_of_joints+offset, num_of_joints*2+offset], cov[num_of_joints*2+offset, num_of_joints*2+offset],])
    return H

def shape_cov_hip(cov, model, frame_index):
    H = np.zeros([3,3])
    offset = frame_index*3
    H[:,0] = np.array([cov[offset, offset], cov[1+offset, offset], cov[2+offset, offset],])
    H[:,1] = np.array([cov[offset, 1+offset], cov[1+offset, 1+offset], cov[2+offset, 1+offset],])
    H[:,2] = np.array([cov[offset, 2+offset], cov[1+offset, 2+offset], cov[2+offset, 2+offset],])
    return H

def shape_cov_mini(cov, model, frame_index):
    _, joint_names, _, _ = model_settings(model)
    hip_index = joint_names.index('spine1')
    H = np.zeros([3,3])
    H[:,0] = np.array([cov[hip_index, hip_index], cov[1+hip_index, hip_index], cov[2+hip_index, hip_index],])
    H[:,1] = np.array([cov[hip_index, 1+hip_index], cov[1+hip_index, 1+hip_index], cov[2+hip_index, 1+hip_index],])
    H[:,2] = np.array([cov[hip_index, 2+hip_index], cov[1+hip_index, 2+hip_index], cov[2+hip_index, 2+hip_index],])
    return H

def shape_cov_test(cov, model):
    _, _, num_of_joints, _ = model_settings(model)
    return cov[0:num_of_joints*3, 0:num_of_joints*3]

def shape_cov_test2(cov, model):
    _, joint_names, num_of_joints, _ = model_settings(model)
    hip_index = joint_names.index('spine1')

    ind = []
    for frame in range(0,7):
        for i in range(0,3):
            ind.append(hip_index+frame*3*num_of_joints+i)
    x,y = np.meshgrid(ind, ind)
    new_cov = cov[x, y]
    
    print(new_cov.shape)
    return new_cov

def plot_covariance_as_ellipse(pose_client, plot_loc, ind):
    if (pose_client.isCalibratingEnergy):
        plot_info = pose_client.calib_res_list
        file_name = plot_loc + '/ellipse_calib_'+ str(ind) + '.png'
    else:
        plot_info = pose_client.online_res_list
        file_name = plot_loc + '/ellipse_online_'+ str(ind) + '.png'

    bone_connections, joint_names, _, _ = model_settings(pose_client.model)
    hip_index = joint_names.index('spine1')
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

def euler_to_rotation_matrix(roll, pitch, yaw, returnTensor=False):
    if (returnTensor == True):
        return torch.FloatTensor([[cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll), cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll)],
                    [sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll), sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll)],
                    [-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)]])
    return np.array([[cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll), cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll)],
                    [sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll), sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll)],
                    [-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)]])


def plot_potential_states(current_human_pose, future_human_pose, gt_human_pose, potential_states, C_drone, R_drone, model, plot_loc, ind):
    _, joint_names, _, _ = model_settings(model)
    hip_index = joint_names.index('spine1')

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
        yaw = potential_state["orientation"]
        state_pos =  potential_state["position"]
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
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

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
    for ind, hess in enumerate(hessians):
        ax = axes.flat[ind]
        im = ax.imshow(hess, cmap=cmap, norm=norm)
        ax.set_title(str(ind))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        eigenvals, _ = np.linalg.eig(hess)
        uncertainty1 = float("{0:.4f}".format(np.linalg.det(hess)))
        uncertainty2 = float("{0:.4f}".format(np.max(eigenvals)))
        uncertainty3 = float("{0:.4f}".format(np.sum(eigenvals)))
        ax.text(0.05,0.05,str(uncertainty1))
        ax.text(0.05,0.5,str(uncertainty2))   
        ax.text(0.05,1.05,str(uncertainty3))   

    fig.subplots_adjust(right=0.8, hspace = 0.5)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    file_name = plot_loc + name + str(linecount) + ".png"
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
def plot_potential_projections(pose2d_list, linecount, plot_loc, photo_loc, model):
    bone_connections, joint_names, _, _ = model_settings(model)
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


def plot_potential_ellipses(current_human_pose, future_human_pose, gt_human_pose, potential_states_fetcher, model, plot_loc, ind, ellipses = True):
    _, joint_names, _, _ = model_settings(model)
    hip_index = joint_names.index('spine1')

    current_human_pos = current_human_pose[:, hip_index]
    future_human_pos =  future_human_pose[:, hip_index]
    gt_human_pos = gt_human_pose[:, hip_index]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    potential_states = potential_states_fetcher.potential_states
    
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
        state_pos =  potential_state["position"]
        center = np.copy(state_pos)
        center[2] = -center[2]
        centers.append(center)
        if ellipses:
            covs = potential_states_fetcher.potential_covs_normal
            x,y,z = matrix_to_ellipse(covs[state_ind], center, 6)
            ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
        else:
            ax.plot([center[0]], [center[1]], [center[2]], marker='^', color='b')
        ax.text(center[0], center[1], center[2], str(state_ind))

        X = np.concatenate([X, np.array([center[0]])])
        Y = np.concatenate([Y, np.array([center[1]])])
        Z = np.concatenate([Z, np.array([center[2]])])
    #curr_center = centers[potential_states_fetcher.current_state_ind]
    goal_center = centers[potential_states_fetcher.goal_state_ind]
    ax.plot([goal_center[0]], [goal_center[1]], [goal_center[2]], marker='^', color='xkcd:dark orange')

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

    ax.legend(handles=[plot1, plot2, plot3])
    #plt.show()
    file_name = plot_loc + "/potential_ellipses_" + str(ind) + ".png"
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)