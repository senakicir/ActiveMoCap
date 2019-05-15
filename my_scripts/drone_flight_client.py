from helpers import *
import pandas as pd
import torch
import numpy as np
import pdb
from PotentialStatesFetcher import PotentialState_Drone_Flight

def read_transformation_matrix(f_drone_pos):
    whole_file = pd.read_csv(f_drone_pos, sep='\t').as_matrix()
    label_list = whole_file[:,0].tolist()
    transformation_matrix = whole_file[:,1:-1].astype('float')
    num_of_files = transformation_matrix.shape[0]
    transformation_matrix_tensor = torch.zeros([num_of_files, 4, 4])
    inv_transformation_matrix_tensor = torch.zeros([num_of_files, 4, 4])
    for ind in range(num_of_files):
        transformation_matrix_tensor[ind, :,:] = torch.from_numpy(np.reshape(transformation_matrix[ind, :], (4,4))).float()
        inv_transformation_matrix_tensor[ind, :,:] = torch.inverse(transformation_matrix_tensor[ind, :,:] )
    return label_list, transformation_matrix_tensor, inv_transformation_matrix_tensor

def read_pose_from_file(input_file, dim, num_of_joints):
    whole_file = pd.read_csv(input_file, sep='\t').as_matrix()[:,:-1].astype('float')
    pose = torch.zeros(whole_file.shape[0], dim, num_of_joints)
    for i in range(whole_file.shape[0]):
        pose[i, :, : ] = torch.from_numpy(np.reshape(whole_file[i, :], newshape=(dim, -1), order="F")).float()
    return pose

def read_gt_pose_single(f_groundtruth):
    gt_str = f_groundtruth.read().split('\t')
    gt_str.pop()
    pose3d_list = [float(joint) for joint in gt_str]
    pose_3d_gt = np.reshape(np.array(pose3d_list, dtype=float), newshape=(3,-1), order="F")
    return pose_3d_gt

def read_intrinsics(f_intrinsics):
    intrinsics_str = f_intrinsics.read().split('\t')
    intrinsics_str.pop()
    intrinsics = [float(value) for value in intrinsics_str]
    return intrinsics[0], intrinsics[1], intrinsics[2]


class DroneFlightClient(object):
    def __init__(self, drone_flight_files):
        #take filenames and save them
        self.linecount = 0
        self.actual_position = 0

        self.is_using_airsim = False
        self.end = False
        self.files = drone_flight_files

        self.label_list, self.transformation_matrix_tensor, self.inv_transformation_matrix_tensor = read_transformation_matrix(self.files["f_drone_pos_reoriented"])
        self.focal_length, self.px, self.py = read_intrinsics(self.files["f_intrinsics"])
        self.groundtruth_matrix =  read_gt_pose_single(self.files["f_groundtruth_reoriented"])
        self.openpose_res_2d = read_pose_from_file(self.files["f_pose_2d"], 2, 15)

        self.DRONE_INITIAL_POS = np.zeros([3,1])     
        self.num_of_data = len(self.label_list)

    def simPauseHuman(self, arg1):
        pass

    def simPauseDrone(self, arg1):
        pass

    def simPause(self, arg1):
        pass

    def simSetCameraOrientation(self, arg1, arg2):
        pass

    def moveToPositionAsync(self, sth):
        self.actual_position = 0

    def read_frame_gt_values(self):
        drone_transformation_matrix =  self.transformation_matrix_tensor[self.actual_position, :, :]
        return self.groundtruth_matrix, drone_transformation_matrix

    def simGetImages(self, arg):
        return [DummyPhotoResponse()]

    def reset(self):
        self.linecount = 0
        self.actual_position = 0
        self.end = False

    def changeAnimation(self, arg):
        pass

    def simSetVehiclePose(self, goal_state):
        self.actual_position = goal_state.index

    def get_linecount(self):
        return self.actual_position

    def get_drone_flight_states(self):
        potential_states = []
        for ind in range(self.num_of_data):
            self.transformation_matrix_tensor[ind, :, :]
            potential_states.append(PotentialState_Drone_Flight(self.transformation_matrix_tensor[ind, :, :], ind))
        return potential_states

class DummyPhotoResponse(object):
    def __init__(self):
        self.bone_pos = np.array([])
        self.image_data_uint8 = np.uint8(0)

