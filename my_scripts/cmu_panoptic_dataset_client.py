import pandas as pd
import torch
import numpy as np
from PotentialStatesFetcher import PotentialState_External_Dataset
from Lift_Client import calculate_bone_directions
from external_dataset_client import External_Dataset_Client
from State import find_pose_and_frame_at_time
import json


class CMU_Panoptic_Dataset_Client(External_Dataset_Client):
    def __init__(self, seq, length_of_simulation, test_sets_loc, experiment_ind):
        if seq == "cmu_panoptic_pose_1":
            test_set_name = "cmu_panoptic_pose_1"
            folder_name = "171204_pose1"
        elif seq == "cmu_panoptic_dance_3":
            test_set_name = "cmu_panoptic_dance_3"
            folder_name = "150821_dance3" 
            
        super(CMU_Panoptic_Dataset_Client, self).__init__(length_of_simulation, test_set_name)
        self.num_of_camera_views = 31
        
        #TODO
        self.constant_rotation_camera_sequence = list(range(self.num_of_camera_views))
        initial_cam_view_list = [1,4,5,7,8]

        self.chosen_cam_view = initial_cam_view_list[experiment_ind]
        self.initial_cam_view =  self.chosen_cam_view 
        
        self.default_initial_anim_time = 34
        self.internal_anim_time = self.default_initial_anim_time

        self.framecount = 1
        self.num_of_joints = 15
        self.external_dataset_states = []

        self.files = self.get_cmu_dataset_files(test_sets_loc, folder_name)


        self.image_main_dir = self.files["image_main_dir"]
        self.intrinsics = self.read_intrinsics(self.files["f_intrinsics_dict"])
        self.transformation_matrix_tensor, self.inv_transformation_matrix_tensor = self.read_transformation_matrix(self.files["f_extrinsics_dict"])
        self.groundtruth_matrix =  self.read_gt_pose_from_file(self.files["f_groundtruth_poses"])

    def get_cmu_dataset_files(self, test_sets_loc, folder_name):
        files = {}

        my_test_set_loc = test_sets_loc + "/" + self.test_set_name

        intrinsics_dict = {}
        for cam_index in range(self.num_of_camera_views):
            intrinsics_dict[cam_index] = my_test_set_loc + "/camera_" + str(cam_index) + "/intrinsics.json"
        files["f_intrinsics_dict"] = intrinsics_dict

        extrinsics_dict = {}
        for cam_index in range(self.num_of_camera_views):
            extrinsics_dict[cam_index] = my_test_set_loc + "/camera_" + str(cam_index) + "/extrinsics.json"
        files["f_extrinsics_dict"] = extrinsics_dict
        
        files["image_main_dir"] =  my_test_set_loc
        files["f_groundtruth_poses"] = my_test_set_loc + "/gt_3d_poses.txt"
        return files

    def read_intrinsics(self, f_intrinsics_dict):
        assert f_intrinsics_dict is not None

        intrinsics = {}
        K_torch = torch.zeros([self.num_of_camera_views, 3, 3])

        for cam_index in range(self.num_of_camera_views):
            f_intrinsics = f_intrinsics_dict[cam_index]
            with open(f_intrinsics) as json_file:
                intrinsics_dict = json.load(json_file)
                K_torch[cam_index, :, :] = torch.FloatTensor(intrinsics_dict["K"])
                #all image sizes are uniform in this dataset
                size_x = intrinsics_dict["size_x"] 
                size_y = intrinsics_dict["size_y"]
                #since this dataset does not provide focal length
                #and focal length is found at [0,0] and [1,1] we take their average.
                #projection does not use this value (it uses K directly)
                #but back-projection does!
                intrinsics[cam_index] = {}
                intrinsics[cam_index]["f"] = (K_torch[cam_index, 0,0] + K_torch[cam_index, 1,1])/2
        intrinsics["size_x"] = size_x
        intrinsics["size_y"] = size_y
        intrinsics["K_torch"] = K_torch
        intrinsics["flip_x_y"] = torch.eye(3)
        return intrinsics

    def read_transformation_matrix(self, f_extrinsics_dict):
        assert f_extrinsics_dict is not None

        transformation_matrix_tensor = torch.zeros([self.num_of_camera_views, 4, 4])
        inv_transformation_matrix_tensor = torch.zeros([self.num_of_camera_views, 4, 4])

        for cam_index in range(self.num_of_camera_views):
            f_extrinsics = f_extrinsics_dict[cam_index]
            with open(f_extrinsics) as json_file:
                extrinsics_dict = json.load(json_file)
                transformation_matrix_tensor[cam_index, :, :] = torch.FloatTensor(extrinsics_dict["extrinsics_matrix"])
                inv_transformation_matrix_tensor[cam_index, :, :] = torch.inverse(transformation_matrix_tensor[cam_index, :,:] )

        #prepare potential trajectories also since they will never change!
        self.external_dataset_states = []
        for camera_ind in range(self.num_of_camera_views):
            self.external_dataset_states.append(PotentialState_External_Dataset(transformation_matrix_tensor[camera_ind, :, :], camera_ind, camera_ind))
        return transformation_matrix_tensor, inv_transformation_matrix_tensor


    def read_gt_pose_from_file(self, input_file):
        whole_file = pd.read_csv(input_file, sep='\t', header=None, skiprows=[0]).to_numpy()[:,:-1].astype('float')
        return whole_file

    def read_gt_pose_at_animtime(self, anim_time):
        pose, framecount = find_pose_and_frame_at_time(anim_time, self.groundtruth_matrix, num_of_joints=15)
        self.framecount = framecount
        return pose

    def get_photo_index(self):
        return self.framecount


    def read_frame_gt_values(self, anim_time):
        assert self.internal_anim_time == anim_time
        drone_transformation_matrix =  self.transformation_matrix_tensor[self.chosen_cam_view, :, :].clone()
        gt_3d_pose = self.read_gt_pose_at_animtime(anim_time).copy()
        return gt_3d_pose, drone_transformation_matrix, self.chosen_cam_view 

    def get_external_dataset_states(self, pose_2d_mode):
        if pose_2d_mode != "openpose":
            states = self.external_dataset_states
        else:
            states = self.external_dataset_states
        return states
