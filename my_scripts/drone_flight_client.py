from helpers import *
import pandas as pd
import torch
import numpy as np
import pdb
from PotentialStatesFetcher import PotentialState_Drone_Flight
from Lift_Client import calculate_bone_directions

def read_transformation_matrix(f_drone_pos, test_set):
    if test_set != "drone_flight":
        whole_file = pd.read_csv(f_drone_pos, sep='\t').as_matrix()
        #label_list = whole_file[:,0]
        num_of_files = whole_file[-1,0] + 1
        num_of_samples = np.max(whole_file[:,1]) + 1
        
        transformation_matrix = whole_file[:,1:-1].astype('float')

        transformation_matrix_tensor = torch.zeros([num_of_files, num_of_samples, 4, 4])
        inv_transformation_matrix_tensor = torch.zeros([num_of_files, num_of_samples, 4, 4])
        ind = 0 
        for linecount in range(num_of_files):
            for samplecount in range(num_of_samples):
                transformation_matrix_tensor[linecount, samplecount, :,:] = torch.from_numpy(np.reshape(transformation_matrix[ind, :], (4,4))).float()
                inv_transformation_matrix_tensor[linecount, samplecount, :,:] = torch.inverse(transformation_matrix_tensor[linecount, samplecount, :,:] )
                ind += 1

    return num_of_files, num_of_samples, transformation_matrix_tensor, inv_transformation_matrix_tensor

def read_pose_from_file(input_file, dim, num_of_joints):
    whole_file = pd.read_csv(input_file, sep='\t').as_matrix()[:,:-1].astype('float')
    pose = torch.zeros(whole_file.shape[0], dim, num_of_joints)
    for i in range(whole_file.shape[0]):
        pose[i, :, : ] = torch.from_numpy(np.reshape(whole_file[i, :], newshape=(dim, -1), order="F")).float()
    return pose

def read_gt_pose(input_file, test_set_name):
    if test_set_name == "drone_flight":
        gt_3d_pose = torch.zeros([1, 3, 15])
        gt_3d_pose[0, :,:] = torch.from_numpy(read_gt_pose_single(input_file)).float()
        return gt_3d_pose
    else:
        return read_pose_from_file(input_file, dim=3, num_of_joints=15)


def read_gt_pose_single(f_groundtruth):
    gt_str = f_groundtruth.read().split('\t')
    gt_str.pop()
    pose3d_list = [float(joint) for joint in gt_str]
    pose_3d_gt = np.reshape(np.array(pose3d_list, dtype=float), newshape=(3,-1), order="F")
    return pose_3d_gt

def read_intrinsics(f_intrinsics):
    if f_intrinsics != None:
        intrinsics_str = f_intrinsics.read().split('\t')
        intrinsics_str.pop()
        intrinsics = [float(value) for value in intrinsics_str]
        return intrinsics[0], intrinsics[1], intrinsics[2]
    else:
        SIZE_X = 1024
        SIZE_Y = 576
        return SIZE_X/2, SIZE_X/2, SIZE_Y/2

def get_pose_matrix(input_file, num_of_data, num_of_samples, dim):
    overall_matrix = read_pose_from_file(input_file, dim=dim, num_of_joints=15)
    openpose_matrix = torch.zeros(num_of_data, num_of_samples, dim=dim, 15)
    ind = 0
    for linecount in range(num_of_files):
        for samplecount in range(num_of_samples):
            openpose_matrix[linecount, samplecount, :, :] = overall_matrix[ind, :, :]
            ind += 1
    return openpose_matrix

class DroneFlightClient(object):
    def __init__(self, test_set_name, non_simulation_files):
        #take filenames and save them
        self.linecount = 0
        self.chosen_sample = 0

        self.is_using_airsim = False
        self.end = False
        self.files = non_simulation_files
        self.test_set_name = test_set_name

        self.num_of_data, self.num_of_samples, self.transformation_matrix_tensor, self.inv_transformation_matrix_tensor = read_transformation_matrix(self.files["f_drone_pos"], test_set)

        self.focal_length, self.px, self.py = read_intrinsics(self.files["f_intrinsics"])
        self.groundtruth_matrix =  read_gt_pose(self.files["f_groundtruth"], test_set_name)
        self.openpose_res_matrix = get_pose_matrix(self.files["f_pose_2d"], self.num_of_data, self.num_of_samples, dim=2)
        self.liftnet_res = read_pose_from_file(self.files["f_pose_lift"], dim=3, num_of_joints=15)

        self.DRONE_INITIAL_POS = np.zeros([3,1])     

    def simPauseHuman(self, arg1):
        pass

    def simPauseDrone(self, arg1):
        pass

    def simPause(self, arg1):
        pass

    def simSetCameraOrientation(self, arg1, arg2):
        pass

    def moveToPositionAsync(self, sth):
        self.chosen_sample = 0

    def read_frame_gt_values(self):
        drone_transformation_matrix =  self.transformation_matrix_tensor[self.linecount, self.chosen_sample, :, :]
        gt_3d_pose = self.groundtruth_matrix[self.linecount, :, :]
        self.current_openpose_res = self.openpose_res_matrix[self.linecount, self.chosen_sample, :, :]
        self.current_liftnet_res = self.liftnet_res[self.linecount, :, :]
        return gt_3d_pose, drone_transformation_matrix

    def simGetImages(self, arg):
        return [DummyPhotoResponse()]

    def reset(self):
        self.linecount = 0
        self.chosen_sample = 0
        self.current_openpose_res = 0
        self.end = False

    def changeAnimation(self, arg):
        pass

    def simSetVehiclePose(self, goal_state):
        self.chosen_sample = goal_state.index

    def get_linecount(self):
        return self.linecount

    def get_drone_flight_states(self):
        potential_states = []
        for ind in range(self.num_of_data):
            potential_states.append(PotentialState_Drone_Flight(self.transformation_matrix_tensor[self.linecount, ind, :, :], ind))
        return potential_states

class DummyPhotoResponse(object):
    def __init__(self):
        self.bone_pos = np.array([])
        self.image_data_uint8 = np.uint8(0)

