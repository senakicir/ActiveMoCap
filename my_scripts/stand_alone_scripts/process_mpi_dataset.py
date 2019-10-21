import torch as torch
import numpy as np
from scipy.io import loadmat
import pdb
from configobj import ConfigObj

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

def find_bone_map():
    joint_map = []
    for ind, value in enumerate(joint_names_mpi):
        joint_map.append(original_joint_names_dataset.index(value))
    return joint_map

def rearrange_bones_to_mpi(joints_unarranged, joint_map):
    joints_rearranged = np.zeros([3,15])
    joints_rearranged = joints_unarranged[:, joint_map]
    return joints_rearranged


dataset_loc = "/Users/kicirogl/workspace/cvlabdata1/home/rhodin/datasets/MPI-3D-HP-allcam/sotnychenko_seq_1"
internal_annot_file = loadmat(dataset_loc+"/internal_annot.mat")
camera_calib_file = ConfigObj(dataset_loc+"/cameras.calib")
pdb.set_trace()

new_dataset_loc = "/Users/kicirogl/workspace/cvlabdata2/home/kicirogl/ActiveDrone/test_sets/mpi_inf_3dhp"
cameras_file = open(new_dataset_loc+"/cameras.txt", "w")
joint_map = find_bone_map()
record_gt_poses()
record_camera_poses()

def record_gt_poses():
    camera_list_poses = internal_annot_file["annot3"]
    for i in range(14):
        anim_time = 0
        poses_file = open(new_dataset_loc+"/camera_"+str(i)+"/gt_3d_poses.txt", "w")
        camera_i = camera_list_poses[i][0] 
        for linecount in range(camera_i.shape[0]):
            flattened_pose = camera_i[linecount]
            anim_time += 1/50
            pose = flattened_pose.reshape([3,:])
            rearranged_pose = rearrange_bones_to_mpi(pose, joint_map)
            pose_str = ""
            for i in range(rearranged_pose.shape[1]):
                pose_str += str(rearranged_pose[0, i]) + '\t' + str(rearranged_pose[1, i]) + '\t' + str(rearranged_pose[2, i])
            poses_file.write(str(anim_time)+ '\t'+ pose_str + '\n')


def record_camera_poses():
    camera_calib_file[]
    poses_file = open(new_dataset_loc+"/camera_calib.txt", "w")
    
