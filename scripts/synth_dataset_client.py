import pandas as pd
import torch
import numpy as np

from PotentialStatesFetcher import PotentialState_External_Dataset
from Lift_Client import calculate_bone_directions
from external_dataset_client import External_Dataset_Client
from State import find_pose_and_frame_at_time

def get_synth_dataset_filenames(test_sets_loc, test_set_name):
    files = {}
    my_test_set_loc = test_sets_loc + "/" + test_set_name
    files["f_camera_pos"] = my_test_set_loc + "/camera_poses.txt"
    files["f_groundtruth_poses"] = my_test_set_loc + "/gt_3d_poses.txt"
    files["f_intrinsics"] = my_test_set_loc + "/intrinsics.txt"
    files["image_main_dir"] =  my_test_set_loc
    return files

class Synth_Dataset_Client(External_Dataset_Client):
    def __init__(self, length_of_simulation, test_sets_loc, test_set_name, experiment_ind):
        super(Synth_Dataset_Client, self).__init__(length_of_simulation, test_set_name)
        if test_set_name == "06_13" or test_set_name == "13_06":
            self.num_of_camera_views = 8
        else:
            self.num_of_camera_views = 18
        self.default_initial_anim_time = 1.0
        self.internal_anim_time = self.default_initial_anim_time

        self.constant_rotation_camera_sequence = list(range(self.num_of_camera_views))
        initial_cam_view_list = list(range(5))#[0]*5
        self.chosen_cam_view = initial_cam_view_list[experiment_ind]
        self.initial_cam_view =  self.chosen_cam_view 

        self.num_of_joints = 15
        self.synth_dataset_states = []
        self.files = get_synth_dataset_filenames(test_sets_loc, self.test_set_name)
        self.transformation_matrix_tensor, self.inv_transformation_matrix_tensor = self.read_transformation_matrix(self.files["f_camera_pos"])
        self.image_main_dir = self.files["image_main_dir"]
        self.intrinsics = self.read_intrinsics(self.files["f_intrinsics"])
        self.groundtruth_matrix =  self.read_gt_pose_from_file(self.files["f_groundtruth_poses"])
        
    def read_intrinsics(self, f_intrinsics):
        file_contents = pd.read_csv(f_intrinsics, sep='\t', skiprows=[0], header=None).to_numpy()[0,:].astype('float')
        focal_length = file_contents[0]
        px = file_contents[1]
        py = file_contents[2]
        size_x = file_contents[3]
        size_y = file_contents[4]
        flip_x_y = torch.FloatTensor([[0,1,0],[-1,0,0],[0,0,1]])
        K_torch =  torch.FloatTensor([[focal_length,0,px],[0,focal_length,py],[0,0,1]])
        intrinsics = {"f":focal_length,"size_x":size_x, "size_y":size_y, "K_torch": K_torch, "flip_x_y": flip_x_y}
       
        return intrinsics

    def read_transformation_matrix(self, f_camera_pos):
        whole_file =  pd.read_csv(f_camera_pos, sep='\t', header=None).to_numpy()[:,:-1].astype("float")
        transformation_matrix = whole_file[:,2:]

        transformation_matrix_tensor = torch.zeros([self.length_of_simulation, self.num_of_camera_views, 4, 4])
        inv_transformation_matrix_tensor = torch.zeros([self.length_of_simulation, self.num_of_camera_views, 4, 4])
        for linecount in range(self.length_of_simulation):
            for cam_index in range(self.num_of_camera_views):
                assert linecount == whole_file[linecount*self.num_of_camera_views+cam_index, 0]
                assert cam_index == whole_file[linecount*self.num_of_camera_views+cam_index, 1]
                transformation_matrix_tensor[linecount,cam_index, :, :] = torch.from_numpy(np.reshape(transformation_matrix[linecount*self.num_of_camera_views+cam_index, :], (4,4))).float()
                inv_transformation_matrix_tensor[linecount,cam_index, :, :] = torch.inverse(transformation_matrix_tensor[linecount, cam_index, :,:] )
        return transformation_matrix_tensor, inv_transformation_matrix_tensor

    def read_gt_pose_from_file(self, input_file):
        whole_file = pd.read_csv(input_file, sep='\t', header=None, skiprows=[0]).to_numpy()[:,:-1].astype('float')
        return whole_file

    def read_gt_pose_at_animtime(self, anim_time):
        pose, _ = find_pose_and_frame_at_time(anim_time, self.groundtruth_matrix, num_of_joints=15)
        return pose

    def get_photo_index(self):
        return self.linecount

    def read_frame_gt_values(self, ignored_arg):

        drone_transformation_matrix =  self.transformation_matrix_tensor[self.linecount, self.chosen_cam_view, :, :].clone()
        gt_3d_pose = self.read_gt_pose_at_animtime(self.internal_anim_time).copy()
        return gt_3d_pose, drone_transformation_matrix, self.chosen_cam_view 

    def get_external_dataset_states(self, ignored_arg):
        self.synth_dataset_states = []
        for camera_ind in range(self.num_of_camera_views):
            self.synth_dataset_states.append(PotentialState_External_Dataset(self.transformation_matrix_tensor[self.linecount, camera_ind, :, :].clone(), camera_ind, camera_ind))
        return self.synth_dataset_states
