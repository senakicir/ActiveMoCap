import torch as torch
from math import pi, cos, sin, degrees
import numpy as np
from helpers import euler_to_rotation_matrix, do_nothing, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z, CAMERA_ROLL_OFFSET, CAMERA_PITCH_OFFSET, CAMERA_YAW_OFFSET, EPSILON
from pose_helper_functions import add_noise_to_pose


neat_tensor = torch.FloatTensor([[0, 0, 0, 1]]) #this tensor is neat!
DEFAULT_TORSO_SIZE = 0.3

class Projection_Client(object):
    def __init__(self, test_set, future_window_size, estimation_window_size, num_of_joints, intrinsics, noise_2d_std, device):
        self.device = device
        self.test_set = test_set
        self.FUTURE_WINDOW_SIZE = future_window_size
        self.ESTIMATION_WINDOW_SIZE = estimation_window_size
        self.intrinsics = intrinsics
        self.num_of_joints = num_of_joints
        self.noise_2d_std = noise_2d_std

        if self.test_set != "mpi_inf_3dhp":
            focal_length = intrinsics["f"]
            px = intrinsics["px"]
            py = intrinsics["py"]
            self.size_x, self.size_y = intrinsics["size_x"], intrinsics["size_y"]

            self.K_torch = (torch.FloatTensor([[focal_length,0,px],[0,focal_length,py],[0,0,1]])).to(self.device)
            self.K_inv_torch = torch.inverse(self.K_torch)
        else:
            self.K_torch = torch.zeros([len(intrinsics), 3, 3]).to(self.device)
            self.K_inv_torch = torch.zeros([len(intrinsics), 3, 3]).to(self.device)

            for cam_index in range(len(intrinsics)):
                cam_intrinsics = intrinsics[cam_index]
                focal_length = cam_intrinsics["f"]
                px = cam_intrinsics["px"]
                py = cam_intrinsics["py"]
                self.size_x, self.size_y = cam_intrinsics["size_x"], cam_intrinsics["size_y"]
                self.K_torch[cam_index, :, :] = torch.FloatTensor([[focal_length,0,px],[0,focal_length,py],[0,0,1]])
                self.K_inv_torch[cam_index, :, :] = torch.inverse(self.K_torch[cam_index, :, :])
        

        if self.test_set == "drone_flight" :
            self.flip_x_y_single = torch.eye(3).to(self.device)
        elif self.test_set == "mpi_inf_3dhp":
            self.flip_x_y_single = torch.eye(3).to(self.device)
        else:
            self.flip_x_y_single = torch.FloatTensor([[0,1,0],[-1,0,0],[0,0,1]]).to(self.device)
        self.flip_x_y_single_inv = torch.inverse(self.flip_x_y_single)
        self.flip_x_y_pre = torch.cat((self.flip_x_y_single, torch.zeros(3,1).to(self.device)), dim=1)
        self.flip_x_y_pre_inv = torch.cat((self.flip_x_y_single_inv, torch.zeros(3,1).to(self.device)), dim=1)

        self.ones_tensor_single = torch.ones([1, self.num_of_joints]).to(self.device)*1.0
        self.ones_tensor_future = torch.ones(self.FUTURE_WINDOW_SIZE, 1, self.num_of_joints).to(self.device)*1.0

        self.single_backproj_res = torch.zeros([3, self.num_of_joints]).to(self.device)

    def append_data():
        raise NotImplementedError

    def reset(self, data_list):
        self.window_size = len(data_list)
        self.pose_2d_tensor = torch.zeros(self.window_size , 2, self.num_of_joints).to(self.device)
        self.inverse_transformation_matrix = torch.zeros(self.window_size , 4, 4).to(self.device)
        queue_index = 0
        cam_list = []
        for (cam_index, bone_2d, _, inverse_transformation_matrix) in data_list:
            cam_list.append(cam_index)
            self.pose_2d_tensor[queue_index, :, :] = bone_2d.clone()
            self.inverse_transformation_matrix[queue_index, :, :]= inverse_transformation_matrix.clone()
            queue_index += 1

        self.ones_tensor = torch.ones(self.window_size, 1, self.num_of_joints).to(self.device)
        self.flip_x_y_batch = self.flip_x_y_pre.repeat(self.window_size , 1, 1)
        
        if self.test_set != "mpi_inf_3dhp":
            self.camera_intrinsics = self.K_torch.repeat(self.window_size , 1,1)
        else:
            self.camera_intrinsics =  self.K_torch[cam_list, :, :]

    def reset_future(self, data_list, potential_trajectory, use_hessian_mode):
        if use_hessian_mode == "whole":
            self.hessian_size = self.ESTIMATION_WINDOW_SIZE + self.FUTURE_WINDOW_SIZE
        elif use_hessian_mode == "partial":
            self.hessian_size = self.ESTIMATION_WINDOW_SIZE 

        self.pose_2d_tensor = torch.zeros(self.hessian_size, 2, self.num_of_joints).to(self.device)
        self.inverse_transformation_matrix = torch.zeros(self.hessian_size , 4, 4).to(self.device)

        ##find future projections
        self.inverse_transformation_matrix[:self.FUTURE_WINDOW_SIZE, :, :] = potential_trajectory.inv_transformation_matrix.clone()
        cam_list = potential_trajectory.cam_list.copy()

        self.pose_2d_tensor[:self.FUTURE_WINDOW_SIZE, :, :] = potential_trajectory.potential_2d_poses.clone()

        queue_index = self.FUTURE_WINDOW_SIZE
        for cam_index, bone_2d, _, inverse_transformation_matrix in data_list:
            cam_list.append(cam_index)
            self.pose_2d_tensor[queue_index, :, :] = bone_2d.clone()
            self.inverse_transformation_matrix[queue_index, :, :]= inverse_transformation_matrix.clone()
            queue_index += 1
            if queue_index == self.hessian_size:
                break
        self.ones_tensor = torch.ones(self.hessian_size, 1, self.num_of_joints).to(self.device)*1.0
        self.flip_x_y_batch = self.flip_x_y_pre.repeat(self.hessian_size , 1, 1)

        if self.test_set != "mpi_inf_3dhp":
            self.camera_intrinsics = self.K_torch.repeat(self.hessian_size , 1,1)
        else:
            self.camera_intrinsics = self.K_torch[cam_list, :, :]


    def deepcopy_projection_client(self):
        return Projection_Client(self.test_set, self.FUTURE_WINDOW_SIZE, self.ESTIMATION_WINDOW_SIZE, self.num_of_joints, self.intrinsics, self.noise_2d_std, self.device) 

    def take_projection(self, pose_3d):
        return self.take_batch_projection(pose_3d, self.inverse_transformation_matrix, self.ones_tensor, self.camera_intrinsics, self.flip_x_y_batch)

    def take_single_projection(self, P_world, inv_transformation_matrix, cam_index):
        if self.test_set != "mpi_inf_3dhp":
            camera_intrinsics = self.K_torch
        else:
            camera_intrinsics = self.K_torch[cam_index, :, :]

        P_world_device = P_world.to(self.device)
        inv_transformation_matrix_device = inv_transformation_matrix.to(self.device)
        P_world = torch.cat((P_world_device, self.ones_tensor_single), 0)
        P_camera = self.world_to_camera(P_world, inv_transformation_matrix_device, self.flip_x_y_single)
        proj_homog = torch.mm(camera_intrinsics, P_camera)
        
        z = proj_homog[2,:]
        result = torch.zeros([2, self.num_of_joints]).to(self.device)
        result[0,:] = proj_homog[0,:]/z
        result[1,:] = proj_homog[1,:]/z
        return result

    def take_batch_projection(self, P_world, inv_transformation_matrix, ones_tensor, camera_intrinsics, flip_x_y):
        P_world = torch.cat((P_world, ones_tensor), dim=1)
        P_camera = self.world_to_camera(P_world, inv_transformation_matrix, flip_x_y)
        proj_homog = torch.bmm(camera_intrinsics, P_camera)
        
        z = proj_homog[:,2,:]
        result = torch.zeros([P_world.shape[0], 2, self.num_of_joints]).to(self.device)
        result[:,0,:] = proj_homog[:,0,:]/z
        result[:,1,:] = proj_homog[:,1,:]/z
        return result

    def take_single_backprojection(self, pose_2d, transformation_matrix, joint_names, cam_index=0):
        if self.test_set != "mpi_inf_3dhp":
            camera_intrinsics_inv = self.K_inv_torch
            focal_length = self.intrinsics["f"]
        else:
            camera_intrinsics_inv = self.K_inv_torch[cam_index, :, :]
            focal_length = self.intrinsics[cam_index]["f"]


        transformation_matrix_device = transformation_matrix.to(self.device)
        img_torso_size = torch.norm(pose_2d[:, joint_names.index('neck')] - pose_2d[:, joint_names.index('spine1')]).float()
        if img_torso_size == 0:
            return torch.normal(torch.zeros(3, self.num_of_joints), torch.ones(3, self.num_of_joints)*10).float()

        z_val = ((focal_length * DEFAULT_TORSO_SIZE) / (img_torso_size)).float()

        self.single_backproj_res[0,:] = pose_2d[0,:]*z_val
        self.single_backproj_res[1,:] = pose_2d[1,:]*z_val
        self.single_backproj_res[2,:] = self.ones_tensor_single*z_val
        
        self.single_backproj_res = torch.mm(camera_intrinsics_inv, self.single_backproj_res)
        P_world = self.camera_to_world(self.single_backproj_res, transformation_matrix_device)

        return P_world.detach().cpu()

    def world_to_camera(self, P_world, inv_transformation_matrix, flip_x_y):
        if inv_transformation_matrix.dim() == 3:
            P_camera = torch.bmm(inv_transformation_matrix, P_world)
            P_camera = torch.bmm(flip_x_y, P_camera)
        else:
            P_camera = torch.mm(inv_transformation_matrix, P_world)
            P_camera = torch.mm(flip_x_y, P_camera[0:3,:])
        return P_camera  

    def camera_to_world(self, P_camera, transformation_matrix):
        if transformation_matrix.dim() == 3:
            ones_tensor = torch.ones(transformation_matrix.shape[0], 1, self.num_of_joints).to(self.device)*1.0
            flip_x_y_inv = self.flip_x_y_single_inv.repeat(transformation_matrix.shape[0], 1, 1)

            P_camera_flipped = torch.bmm(flip_x_y_inv, P_camera)
            P_camera_concat = torch.cat((P_camera_flipped, ones_tensor),dim=1)
            P_world_ = torch.bmm(transformation_matrix, P_camera_concat)
            P_world = P_world_[:,0:3,:].clone()
        else:
            P_camera_flipped = torch.mm(self.flip_x_y_single_inv, P_camera)
            P_camera_concat = torch.cat((P_camera_flipped, self.ones_tensor_single),0)
            P_world_ = torch.mm(transformation_matrix, P_camera_concat)
            P_world = P_world_[0:3,:].clone()
        return P_world
