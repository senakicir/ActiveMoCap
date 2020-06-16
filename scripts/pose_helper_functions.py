import torch
import numpy as np
from math import degrees, radians, pi, ceil, exp, atan2, sqrt, cos, sin, acos, ceil

bones_h36m = [[0, 1], [1, 2], [2, 3], [3, 19], #right leg
              [0, 4], [4, 5], [5, 6], [6, 20], #left leg
              [0, 7], [7, 8], [8, 9], [9, 10], #middle
              [8, 14], [14, 15], [15, 16], [16, 17], #left arm
              [8, 11], [11, 12], [12, 13], [13, 18]] #right arm

joint_indices_h36m=list(range(20))
joint_names_h36m = ['hip','right_up_leg','right_leg','right_foot','left_up_leg','left_leg', 'left_foot','spine1','neck', 'head', 'head_top', 'left_arm','left_forearm','left_hand','right_arm','right_forearm','right_hand', 'right_hand_tip', 'left_hand_tip', 'right_foot_tip', 'left_foot_tip']

bones_mpi = [[0, 1], [14, 1], #middle
            [1, 2], [2, 3], [3, 4], #right arm
            [1, 5], [5, 6], [6, 7],  #left arm
            [14, 8], [8, 9], [9, 10], #right leg
            [14, 11], [11, 12], [12, 13]] #left leg
joint_names_mpi = ['head','neck','right_arm','right_forearm','right_hand','left_arm', 'left_forearm','left_hand','right_up_leg','right_leg', 'right_foot', 'left_up_leg', 'left_leg', 'left_foot', 'spine1']


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


def add_noise_to_pose(pose, my_rng, noise_type, test_set_name=None):
    noise = my_rng.get_pose_noise(pose.shape, noise_type)
    if (pose.is_cuda):
        my_device = torch.device("cuda")
        noise = noise.to(my_device)

    if test_set_name=="mpi_inf_3dhp" and noise_type =="proj":
        ## because the size of the person changes dramatically from image to image 
        ## in the mpi dataset and the openpose noise was calculated
        ## according to the simulor data. therefore, we need to normalize the body size here.
        body_size = torch.max(pose[1,:])-torch.min(pose[1,:])
        noise = body_size*(noise/150) 

    pose = pose.float().clone() + noise
    return pose 

def calculate_bone_lengths(bones, bone_connections, batch):
    if batch:
        return (torch.sum(torch.pow(bones[:, :, bone_connections[:,0]] - bones[:, :, bone_connections[:,1]], 2), dim=1)).float()
    else:  
        return (torch.sum(torch.pow(bones[:, bone_connections[:,0]] - bones[:, bone_connections[:,1]], 2), dim=0)).float()

def calculate_bone_lengths_sqrt(bones, bone_connections, batch):
    if batch:
        return torch.sqrt(torch.sum(torch.pow(bones[:, :, bone_connections[:,0]] - bones[:, :, bone_connections[:,1]], 2), dim=1)).float()
    else:  
        return torch.sqrt(torch.sum(torch.pow(bones[:, bone_connections[:,0]] - bones[:, bone_connections[:,1]], 2), dim=0)).float()    

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
