import setup_path 
import airsim

import shutil, skimage.io
import numpy as np
import torch as torch
from torch.autograd import Variable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time, os
import cv2
from math import degrees, radians, pi, ceil, exp




energy_mode = {1:True, 0:False}
LOSSES = ["proj", "smooth", "bone", "lift"]#, "smoothpose"]
CALIBRATION_LOSSES = ["proj", "sym"]
attributes = ['dronePos', 'droneOrient', 'humanPos', 'hip', 'right_up_leg', 'right_leg', 'right_foot', 'left_up_leg', 'left_leg', 'left_foot', 'spine1', 'neck', 'head', 'head_top','left_arm', 'left_forearm', 'left_hand','right_arm','right_forearm','right_hand', 'right_hand_tip', 'left_hand_tip' ,'right_foot_tip' ,'left_foot_tip']
TEST_SETS = {"t": "test_set_t", "05_08": "test_set_05_08", "38_03": "test_set_38_03", "64_06": "test_set_64_06", "02_01": "test_set_02_01"}

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
        rigth_arm_connections = [[8, 11], [11, 12], [12, 13], [13, 18]]
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
        sub_folder_name = main_folder_name + "/" + str(TEST_SETS[animation])
        folder_names[animation] = {"images": sub_folder_name + '/images', "estimates": sub_folder_name + '/estimates', "superimposed_images":  sub_folder_name + '/superimposed_images'}
        for a_folder_name in folder_names[animation].values():
            if not os.path.exists(a_folder_name):
                os.makedirs(a_folder_name)
        file_names[animation] = {"f_output": sub_folder_name +  '/a_flight.txt', "f_groundtruth": sub_folder_name +  '/groundtruth.txt'}

    f_notes_name = main_folder_name + "/notes.txt"
    return file_names, folder_names, f_notes_name


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
    fig1 = plt.figure()
    p1, = plt.semilogx(xdata, ydata)
    if (plot_title != ""): 
        plt.title(plot_title)
    if (x_label != ""): 
       plt.xlabel(x_label)
    if (y_label != ""): 
        plt.ylabel(y_label)
    plt.savefig(folder_name + '/' + plot_title + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()

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
    plt.legend(handles=[p0,p1,p2])
    plt.savefig(superimposed_plot_loc, bbox_inches='tight', pad_inches=0)
    plt.close()

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
    plt.close()


def plot_drone_and_human(bones_GT, predicted_bones, location, ind,  bone_connections, error = -5, custom_name = None, orientation = "z_up", label_names =None):   
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
    gs1 = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs1[0], projection='3d')

    X = bones_GT[0,:]
    if orientation == "z_up":
        # maintain aspect ratio
        Y = bones_GT[1,:]
        Z = -bones_GT[2,:]
    elif (orientation == "y_up"):
        # maintain aspect ratio
        Y = bones_GT[2,:]
        Z = -bones_GT[1,:]
        
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.8
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    #plot drone
    #plot1 = ax.scatter(drone[0], drone[1], drone[2], c='r', marker='o')

    left_bone_connections, right_bone_connections, middle_bone_connections = split_bone_connections(bone_connections)

    if (orientation == "z_up"):
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

    elif (orientation == "y_up"):
        #plot joints
        for i, bone in enumerate(bone_connections):
            plot1, = ax.plot(bones_GT[0,bone], bones_GT[2,bone], -bones_GT[1,bone], c='b', marker='^', label=blue_label)
        for i, bone in enumerate(bone_connections):
            plot2, = ax.plot(predicted_bones[0,bone], predicted_bones[2,bone], -predicted_bones[1,bone], c='r', marker='^', label=red_label)
        ax.legend(handles=[plot1, plot2])

        ax.set_xlabel('X')
        ax.set_zlabel('Y')
        ax.set_ylabel('Z')

    if (error != -5):
        plt.title("error: %.4f" %error)

    plot_3d_pos_loc = location + name + str(ind) + '.png'
    plt.savefig(plot_3d_pos_loc, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_global_motion(plot_info, plot_loc, ind, model, isCalib):
    fig = plt.figure()
    bone_connections, _, num_of_joints, _ = model_settings(model)
    left_bone_connections, right_bone_connections, middle_bone_connections = split_bone_connections(bone_connections)
    gs1 = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs1[0], projection='3d')
    for frame_ind, frame_plot_info in enumerate(plot_info):
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
    if isCalib:
        file_name = plot_loc + '/global_plot_calib_'+ str(ind) + '.png'
    else:
        file_name = plot_loc + '/global_plot_flight_'+ str(ind) + '.png'
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