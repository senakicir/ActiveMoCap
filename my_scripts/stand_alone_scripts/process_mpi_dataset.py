import torch as torch
import numpy as np
from scipy.io import loadmat
import pdb
import os
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors

#original_joint_names_dataset = ['spine3', 'spine4', 'spine2', 'spine', 'pelvis', 'neck', 'head', 'head_top',
          #           'left_clavicle', 'left_shoulder', 'left_elbow','left_wrist', 'left_hand',  
           #          'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist', 'right_hand', 
            #         'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe', 'right_hip' , 
             #        'right_knee', 'right_ankle', 'right_foot', 'right_toe'] 
original_joint_names_dataset = ['spine3', 'spine4', 'spine2', 'spine', 'pelvis', 'neck', 'head', 'head_top',
                     'left_clavicle', 'left_shoulder', 'left_elbow','left_wrist', 'left_hand',  
                    'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist', 'right_hand', 
                    'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe', 'right_hip' , 
                    'right_knee', 'right_ankle', 'right_foot', 'right_toe'] 
joint_indices_dataset = [3, 1, 4, 5, 5, 2, 6, 7, 6, 9, 10, 11, 12, 6, 14, 15, 16, 17, 5, 19, 20, 21, 22, 5, 24, 25, 26, 27 ]

joint_names_mpi = ['head','neck','right_shoulder','right_elbow','right_wrist',
                'left_shoulder', 'left_elbow','left_wrist','right_hip','right_knee', 
                'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'spine']

bones_mpi = [[0, 1], [14, 1], #middle
            [1, 2], [2, 3], [3, 4], #right arm
            [1, 5], [5, 6], [6, 7],  #left arm
            [14, 8], [8, 9], [9, 10], #right leg
            [14, 11], [11, 12], [12, 13]] #left leg

def find_bone_map():
    joint_map = []
    for ind, value in enumerate(joint_names_mpi):
        joint_map.append(original_joint_names_dataset.index(value))
    return joint_map

def rearrange_bones_to_mpi(joints_unarranged, joint_map):
    joints_rearranged = np.zeros([3,15])
    joints_rearranged = joints_unarranged[:, joint_map]
    return joints_rearranged


def put_poses_in_matrix():
    #pdb.set_trace()
    camera_list_poses = internal_annot_file["univ_annot3"]
    pose_matrix = np.zeros([14, 6031, 3, 15])
    for camera_ind in range(14):
        camera_dir = new_dataset_loc+"/camera_"+str(camera_ind)
        if not os.path.exists(camera_dir):
            os.makedirs(camera_dir)  
        poses_file = open(camera_dir+"/rel_3d_poses.txt", "w")
        camera_i = camera_list_poses[camera_ind][0] 
        anim_time = 0
        for linecount in range(camera_i.shape[0]):
            anim_time += 0.02
            flattened_pose = camera_i[linecount]
            pose = flattened_pose.reshape((3,-1), order="F")
            rearranged_pose = rearrange_bones_to_mpi(pose, joint_map)
            pose_matrix[camera_ind, linecount, :, :] = rearranged_pose.copy()
            pose_str = ""
            for i in range(rearranged_pose.shape[1]):
                pose_str += str(rearranged_pose[0, i]) + '\t' + str(rearranged_pose[1, i]) + '\t' + str(rearranged_pose[2, i])
            poses_file.write(str(anim_time)+ '\t'+ pose_str + '\n')
    return pose_matrix

def get_stripped_word_list(file_name):
    line = file_name.readline()
    word_list = line.split()
    stripped_word_list = [word.strip() for word in word_list]
    first_word = stripped_word_list[0]
    return first_word, stripped_word_list

def get_camera_transformation_matrix(cam_c, cam_up, cam_right):
    camera_transformation_matrix = np.zeros([14,4,4])
    for cam_index in range(14):
        camera_transformation_matrix[cam_index, 3, :] = np.array([0,0,0,1]) 
        camera_transformation_matrix[cam_index, 0:3, 3] = cam_c[cam_index, :].copy()
        camera_transformation_matrix[cam_index, 0:3, 0] = cam_right[cam_index, :].copy()
        camera_transformation_matrix[cam_index, 0:3, 2] = cam_up[cam_index, :].copy()
        vec_forward = np.cross(cam_right[cam_index, :], cam_up[cam_index, :])
        camera_transformation_matrix[cam_index, 0:3, 1] = vec_forward.copy()
    return camera_transformation_matrix


def record_camera_poses(camera_calib_file, new_dataset_loc):
    #camera_calib_file[]
    camera_poses_file = open(new_dataset_loc+"/camera_poses.txt", "w")
    line = camera_calib_file.readline()
    first_word = ""

    cam_c = np.zeros((14, 3))
    cam_up = np.zeros((14, 3))
    cam_right = np.zeros((14, 3))

    cam_intrinsics = np.zeros((14,5))
    for cam_index in range(14):
        camera_dir = new_dataset_loc+"/camera_"+str(cam_index)
        if not os.path.exists(camera_dir):
            os.makedirs(camera_dir)  

        intrinsics_file = open(camera_dir+"/intrinsics.txt", "w")
        intrinsics_file.write("focal_length\tpx\tpy\tsize_x\tsize_y\n")
        first_word, stripped_word_list = get_stripped_word_list(camera_calib_file)
        assert first_word == "camera"

        while (first_word != "focalLength"):
            first_word, stripped_word_list = get_stripped_word_list(camera_calib_file)
        focal_length = float(stripped_word_list[1])
        while (first_word != "centerOffset"):
            first_word, stripped_word_list = get_stripped_word_list(camera_calib_file)
        px = float(stripped_word_list[1])
        py = float(stripped_word_list[2])
        size_x = -1
        size_y = -1
        cam_intrinsics[cam_index, :]= np.array([focal_length, px, py, size_x, size_y])
        intrinsics_file.write(str(focal_length)+"\t"+str(px)+"\t"+str(py)+"\t"+str(size_x)+"\t"+str(size_y))

        while (first_word != "origin"):
            first_word, stripped_word_list = get_stripped_word_list(camera_calib_file)
        cam_c[cam_index, :] = np.array([float(stripped_word_list[1]), float(stripped_word_list[2]), float(stripped_word_list[3])])
        while (first_word != "up"):
            first_word, stripped_word_list = get_stripped_word_list(camera_calib_file)
        cam_up[cam_index, :] = np.array([float(stripped_word_list[1]), float(stripped_word_list[2]), float(stripped_word_list[3])])
        while (first_word != "right"):
            first_word, stripped_word_list = get_stripped_word_list(camera_calib_file)
        cam_right[cam_index, :] = np.array([float(stripped_word_list[1]), float(stripped_word_list[2]), float(stripped_word_list[3])])

    camera_transformation_matrix = get_camera_transformation_matrix(cam_c, cam_up, cam_right)
    for cam_index in range(14):
        flattened_cam_matrix = np.reshape(camera_transformation_matrix[cam_index, :], (16,-1))
        cam_str = str(cam_index)+'\t'
        for i in range(16):
            cam_str += str(flattened_cam_matrix[i]) +'\t'
        camera_poses_file.write(cam_str+'\n')
    return camera_transformation_matrix
    
   
def display_pose(pose):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111,  projection='3d')
    X = pose[0,:]
    Y = pose[2,:]
    Z = -pose[1,:]
    for _, bone in enumerate(bones_mpi):
        ax.plot(X[bone], Y[bone], Z[bone], c='xkcd:black')

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.4
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()
    plt.close(fig)

def save_poses(pose_matrix, new_dataset_loc):
    poses_file = open(new_dataset_loc+"/gt_3d_poses.txt", "w")
    anim_time = 0
    for i in pose_matrix.shape[0]:
        anim_time += 0.02
        pose_str = ""
        for j in range(pose_matrix.shape[1]):
            pose_str += str(pose_matrix[i, 0, j]) + '\t' + str(pose_matrix[i, 1, j]) + '\t' + str(pose_matrix[i, 2, j])
        poses_file.write(str(anim_time)+ '\t'+ pose_str + '\n')


dataset_loc = "/Users/kicirogl/workspace/cvlabdata1/home/rhodin/datasets/MPI-3D-HP-allcam/sotnychenko_seq_1"
internal_annot_file = loadmat(dataset_loc+"/internal_annot.mat")
camera_calib_file = open(dataset_loc+"/cameras.calib", "r")


new_dataset_loc = "/Users/kicirogl/workspace/cvlabdata2/home/kicirogl/ActiveDrone/test_sets/mpi_inf_3dhp"
joint_map = find_bone_map()
camera_pose_matrix = put_poses_in_matrix()
camera_matrix = record_camera_poses(camera_calib_file, new_dataset_loc)
