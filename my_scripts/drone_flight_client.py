from helpers import *
import pandas as pd
import torch
import numpy as np
import pdb
from PotentialStatesFetcher import PotentialState_Drone_Flight
from Lift_Client import calculate_bone_directions

def read_transformation_matrix(f_drone_pos, test_set):
    whole_file = pd.read_csv(f_drone_pos, sep='\t', header=None).values
    num_of_files = int(whole_file[-1,0])+1
    num_of_samples = int(np.max(whole_file[:,1]))+1
    print("num of files", num_of_files, ". num of samples", num_of_samples)
    
    transformation_matrix = whole_file[:,2:-1].astype('float')

    transformation_matrix_tensor = torch.zeros([num_of_files, num_of_samples, 4, 4])
    inv_transformation_matrix_tensor = torch.zeros([num_of_files, num_of_samples, 4, 4])
    ind = 0 
    for linecount in range(num_of_files):
        for samplecount in range(num_of_samples):
            assert linecount == whole_file[ind, 0]
            assert samplecount == whole_file[ind, 1]
            transformation_matrix_tensor[linecount, samplecount, :, :] = torch.from_numpy(np.reshape(transformation_matrix[ind, :], (4,4))).float()
            inv_transformation_matrix_tensor[linecount, samplecount, :, :] = torch.inverse(transformation_matrix_tensor[linecount, samplecount, :,:] )
            ind += 1

    return num_of_files, num_of_samples//2, transformation_matrix_tensor[:, 0:num_of_samples//2, :, :], inv_transformation_matrix_tensor[:, 0:num_of_samples//2, :, :]

def read_pose_from_file(input_file, dim, num_of_joints):
    whole_file = pd.read_csv(input_file, sep='\t', header=None).values[:,1:-1].astype('float')
    pose = torch.zeros(whole_file.shape[0], dim, num_of_joints)
    for i in range(whole_file.shape[0]):
        pose[i, :, : ] = torch.from_numpy(np.reshape(whole_file[i, :], newshape=(dim, -1), order="F")).float()
    return pose

def read_gt_pose(input_file, test_set_name):
    gt_pose = read_pose_from_file(input_file, dim=3, num_of_joints=15)
    return gt_pose

def read_intrinsics(f_intrinsics):
    if f_intrinsics != None:
        intrinsics = pd.read_csv(f_intrinsics, sep='\t', header=None).values[0,:].astype('float')
        return intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], intrinsics[4]
    else:
        SIZE_X = 1024
        SIZE_Y = 576
        return SIZE_X/2, SIZE_X/2, SIZE_Y/2, int(SIZE_X), int(SIZE_Y)

def get_pose_matrix(input_file, num_of_data, num_of_samples, dim):
    overall_matrix = read_pose_from_file(input_file, dim=dim, num_of_joints=15)
    openpose_matrix = torch.zeros((num_of_data, num_of_samples, dim, 15))
    ind = 0
    for linecount in range(num_of_files):
        for samplecount in range(num_of_samples):
            openpose_matrix[linecount, samplecount, :, :] = overall_matrix[ind, :, :]
            ind += 1
    return openpose_matrix

class DroneFlightClient(object):
    def __init__(self, length_of_simulation, test_set_name, non_simulation_files):
        #take filenames and save them
        self.linecount = 0
        self.online_linecount = 0
        self.chosen_sample = 0
        self.length_of_simulation = length_of_simulation

        self.is_using_airsim = False
        self.end = False
        self.files = non_simulation_files
        self.test_set_name = test_set_name

        self.num_of_data, self.num_of_samples, self.transformation_matrix_tensor, self.inv_transformation_matrix_tensor = read_transformation_matrix(self.files["f_drone_pos"], self.test_set_name)

        self.focal_length, self.px, self.py, self.SIZE_X, self.SIZE_Y = read_intrinsics(self.files["f_intrinsics"])
        self.groundtruth_matrix =  read_gt_pose(self.files["f_groundtruth"], test_set_name)
        self.openpose_res_matrix = 0#get_pose_matrix(self.files["f_pose_2d"], self.num_of_data, self.num_of_samples, dim=2)
        self.liftnet_res = 0#read_pose_from_file(self.files["f_pose_lift"], dim=3, num_of_joints=15)

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
        pass

    def read_frame_gt_values(self):
        linecount_ind = self.linecount 
        if self.test_set_name == "drone_flight":
            linecount_ind = 0

        drone_transformation_matrix =  self.transformation_matrix_tensor[linecount_ind, self.chosen_sample, :, :].clone()
        gt_3d_pose = self.groundtruth_matrix[linecount_ind, :, :].clone()
        #self.current_openpose_res = self.openpose_res_matrix[self.linecount, self.chosen_sample, :, :]
        #self.current_liftnet_res = self.liftnet_res[self.linecount, :, :]
        return gt_3d_pose.numpy(), drone_transformation_matrix

    def simGetImages(self, arg):
        return [DummyPhotoResponse()]

    def reset(self):
        self.linecount = 0
        self.online_linecount = 0
        self.chosen_sample = 0
        self.current_openpose_res = 0
        self.end = False

    def changeAnimation(self, arg):
        pass

    def simSetVehiclePose(self, goal_state):
        self.chosen_sample = goal_state.index

    def get_linecount(self):
        return self.linecount

    def increment_linecount(self, isCalibratingEnergy):
        self.linecount += 1
        if not isCalibratingEnergy:
            self.online_linecount += 1

    def get_drone_flight_states(self):
        linecount_ind = self.linecount 
        if self.test_set_name == "drone_flight":
            linecount_ind = 0
       
        potential_states = []
        for sample_ind in range(self.num_of_samples):
            potential_states.append(PotentialState_Drone_Flight(self.transformation_matrix_tensor[linecount_ind, sample_ind, :, :], sample_ind))
        return potential_states

class DummyPhotoResponse(object):
    def __init__(self):
        self.bone_pos = np.array([])
        self.image_data_uint8 = np.uint8(0)

