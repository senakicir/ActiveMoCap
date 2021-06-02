import pandas as pd
import torch
import numpy as np
from PotentialStatesFetcher import PotentialState_External_Dataset
from Lift_Client import calculate_bone_directions
from external_dataset_client import External_Dataset_Client
from State import find_pose_and_frame_at_time

class MPI_Dataset_Client(External_Dataset_Client):
    def __init__(self, length_of_simulation, test_sets_loc, experiment_ind):
        test_set_name = "mpi_inf_3dhp"
        super(MPI_Dataset_Client, self).__init__(length_of_simulation, test_set_name)
        self.num_of_camera_views = 14
        self.default_initial_anim_time = 16.02
        self.internal_anim_time = self.default_initial_anim_time

        self.constant_rotation_camera_sequence = [4,3,2,6,7,10,5,8,9,0,1]
        initial_cam_view_list = [1,4,5,7,8]
        self.chosen_cam_view = initial_cam_view_list[experiment_ind]
        self.initial_cam_view =  self.chosen_cam_view 
        self.framecount = 800
        self.num_of_joints = 15
        self.files = self.get_mpi_dataset_filenames(test_sets_loc)
        self.transformation_matrix_tensor, self.inv_transformation_matrix_tensor = self.read_transformation_matrix(self.files["f_camera_pos"])
        self.image_main_dir = self.files["image_main_dir"]
        self.intrinsics = self.read_intrinsics(self.files["f_intrinsics_dict"])
        self.groundtruth_matrix =  self.read_gt_pose_from_file(self.files["f_groundtruth_poses"])

    def get_mpi_dataset_filenames(self, test_sets_loc):
        files = {}
        my_test_set_loc = test_sets_loc + "/" + self.test_set_name
        files["f_camera_pos"] = my_test_set_loc + "/camera_poses.txt"
        files["f_groundtruth_poses"] = my_test_set_loc + "/gt_3d_poses.txt"
        intrinsics_dict = {}
        for cam_index in range(self.num_of_camera_views):
            intrinsics_dict[cam_index] = my_test_set_loc + "/camera_" + str(cam_index) + "/intrinsics.txt"

        files["f_intrinsics_dict"] = intrinsics_dict
        files["image_main_dir"] =  my_test_set_loc
        return files
        
    def read_intrinsics(self, f_intrinsics_dict):
        assert f_intrinsics_dict is not None
        intrinsics = {}

        K_torch = torch.zeros(self.num_of_camera_views, 3, 3)

        for cam_index in range(self.num_of_camera_views):
            f_intrinsics = f_intrinsics_dict[cam_index]
            file_contents = pd.read_csv(f_intrinsics, sep='\t', skiprows=[0], header=None).to_numpy()[0,:].astype('float')
            f = file_contents[0]
            px = file_contents[1]
            py = file_contents[2]
            size_x = file_contents[3]
            size_y = file_contents[4]

            K_torch[cam_index, :, :] = (torch.FloatTensor([[f,0,px],[0,f,py],[0,0,1]]))
            
            intrinsics[cam_index]["f"] = f
        intrinsics["size_x"] = size_x
        intrinsics["size_y"] = size_y
        intrinsics["K_torch"] = K_torch
        intrinsics["flip_x_y"] = torch.eye(3).to(self.device)
        return intrinsics

    def read_transformation_matrix(self, f_camera_pos):
        whole_file =  pd.read_csv(f_camera_pos, sep='\t', header=None).to_numpy()[:,:-1].astype("float")
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
            states = self.external_dataset_states[0:11]
        return states
