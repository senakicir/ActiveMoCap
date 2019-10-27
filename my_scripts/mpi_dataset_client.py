import pandas as pd
import torch
import numpy as np
from PotentialStatesFetcher import PotentialState_External_Dataset
from Lift_Client import calculate_bone_directions
from external_dataset_client import External_Dataset_Client
from State import find_pose_at_time

class MPI_Dataset_Client(External_Dataset_Client):
    def __init__(self, length_of_simulation, test_sets_loc):
        test_set_name = "mpi_inf_3dhp"
        External_Dataset_Client.__init__(length_of_simulation, test_set_name, non_simulation_files)
        self.num_of_camera_views = 14
        self.num_of_joints = 15
        self.mpi_dataset_states = []
        self.files = self.get_mpi_dataset_filenames(test_sets_loc)
        self.transformation_matrix_tensor, self.inv_transformation_matrix_tensor = self.read_transformation_matrix(self.files["f_camera_pos"])
        self.focal_length, self.px, self.py, self.SIZE_X, self.SIZE_Y = self.read_intrinsics(self.files["f_intrinsics_dict"])
        self.groundtruth_matrix =  self.read_gt_pose_from_file(self.files["f_groundtruth_poses"], test_set_name)

    def get_mpi_dataset_filenames(self, test_sets_loc):
        files = {}
        my_test_set_loc = test_sets_loc + "/" + self.test_set_name
        files["f_camera_pos"] = my_test_set_loc + "/camera_poses.txt"
        files["f_groundtruth_poses"] = my_test_set_loc + "/gt_3d_poses.txt"
        intrinsics_dict = {}
        for cam_index in range(14):
            intrinsics_dict[cam_index] = my_test_set_loc + "/camera" + str(cam_index) + "/intrinsics.txt"
        files["f_intrinsics_dict"] = intrinsics_dict
        return files
        
    def read_transformation_matrix(self, f_camera_pos):
        whole_file =  pd.read_csv(f_camera_pos, sep='\t', header=None)[:,:-1].to_numpy()
        assert whole_file[0,0] == 0 and whole_file[-1,0] == self.num_of_camera_views-1 
        transformation_matrix = whole_file[:,1:]

        transformation_matrix_tensor = torch.zeros([self.num_of_camera_views, 4, 4])
        inv_transformation_matrix_tensor = torch.zeros([self.num_of_camera_views, 4, 4])
        ind = 0 
        for cam_index in range(self.num_of_camera_views):
            assert cam_index == whole_file[cam_index, 0]
            transformation_matrix_tensor[cam_index, :, :] = torch.from_numpy(np.reshape(transformation_matrix[cam_index, :], (4,4))).float()
            inv_transformation_matrix_tensor[cam_index, :, :] = torch.inverse(transformation_matrix_tensor[cam_index, :,:] )

        #prepare potential trajectories also since they will never change!
        self.mpi_dataset_states = []
        for camera_ind in range(self.num_of_camera_views):
            mpi_dataset_states.append(PotentialState_External_Dataset(self.transformation_matrix_tensor[camera_ind, :, :], camera_ind))
        return transformation_matrix_tensor, inv_transformation_matrix_tensor

    def read_gt_pose_from_file(self, input_file):
        whole_file = pd.read_csv(input_file, sep='\t', header=None)[:,:-1].to_numpy().astype('float')
        return whole_file

    def read_gt_pose_at_animtime(self, anim_time):
        return find_pose_at_time(anim_time, self.groundtruth_matrix, num_of_joints=15)

    def read_intrinsics(self, f_intrinsics_dict):
        assert f_intrinsics is not None
        intrinsics = {}
        for cam_index in range(self.num_of_camera_views):
            temp_dict = {}
            f_intrinsics = f_intrinsics_dict[cam_index]
            file_contents = pd.read_csv(f_intrinsics, sep='\t', skip_rows=[0], header=None).to_numpy().astype('np.float')
            temp_dict["f"] = file_contents[0]
            temp_dict["px"] = file_contents[1]
            temp_dict["py"] = file_contents[2]
            temp_dict["size_x"] = file_contents[3]
            temp_dict["size_y"] = file_contents[4]
            intrinsics[cam_index] = temp_dict
        return intrinsics

    def read_frame_gt_values(self, anim_time):
        linecount_ind = self.linecount 
        drone_transformation_matrix =  self.transformation_matrix_tensor[self.chosen_cam_view, :, :].clone()
        gt_3d_pose = read_gt_pose_at_animtime(anim_time).copy()
        return gt_3d_pose, drone_transformation_matrix, self.chosen_cam_view 

    def get_external_dataset_states(self):
        return self.mpi_dataset_states
