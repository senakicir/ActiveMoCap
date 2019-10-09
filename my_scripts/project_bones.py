import torch as torch
from math import pi, cos, sin, degrees
import numpy as np
from helpers import euler_to_rotation_matrix, do_nothing, CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z, CAMERA_ROLL_OFFSET, CAMERA_PITCH_OFFSET, CAMERA_YAW_OFFSET, EPSILON
from pose_helper_functions import add_noise_to_pose
import pdb

neat_tensor = torch.FloatTensor([[0, 0, 0, 1]]) #this tensor is neat!
DEFAULT_TORSO_SIZE = 0.3

C_cam_torch = torch.FloatTensor([[CAMERA_OFFSET_X], [CAMERA_OFFSET_Y], [CAMERA_OFFSET_Z]])

class Projection_Client(object):
    def __init__(self, test_set, future_window_size, num_of_joints, focal_length, px, py, noise_2d_std, device):
        self.device = device
        self.test_set = test_set
        self.FUTURE_WINDOW_SIZE = future_window_size
        self.focal_length = focal_length
        self.px = px
        self.py = py

        self.K_torch = (torch.FloatTensor([[self.focal_length,0,self.px],[0,self.focal_length,self.py],[0,0,1]])).to(self.device)
        self.K_inv_torch = torch.inverse(self.K_torch)
        self.num_of_joints = num_of_joints

        if self.test_set == "drone_flight":
            self.flip_x_y_single = torch.eye(3).to(self.device)
            self.flip_x_y_single_inv = torch.eye(3)
        else:
            self.flip_x_y_single = torch.FloatTensor([[0,1,0],[-1,0,0],[0,0,1]]).to(self.device)
            self.flip_x_y_single_inv = torch.inverse(self.flip_x_y_single)
        self.flip_x_y_pre = torch.cat((self.flip_x_y_single, torch.zeros(3,1).to(self.device)), dim=1)

        self.noise_2d_std = noise_2d_std
        self.ones_tensor_single = torch.ones([1, self.num_of_joints]).to(self.device)*1.0
        self.ones_tensor_future = torch.ones(self.FUTURE_WINDOW_SIZE, 1, self.num_of_joints).to(self.device)

        self.single_backproj_res = torch.zeros([3, self.num_of_joints]).to(self.device)

    def reset(self, data_list):
        self.window_size = len(data_list)
        self.pose_2d_tensor = torch.zeros(self.window_size , 2, self.num_of_joints).to(self.device)
        self.inverse_transformation_matrix = torch.zeros(self.window_size , 4, 4).to(self.device)
        queue_index = 0
        for bone_2d, _, inverse_transformation_matrix in data_list:
            self.pose_2d_tensor[queue_index, :, :] = bone_2d.clone()
            self.inverse_transformation_matrix[queue_index, :, :]= inverse_transformation_matrix.clone()
            queue_index += 1

        self.ones_tensor = torch.ones(self.window_size, 1, self.num_of_joints).to(self.device)
        self.flip_x_y_batch = self.flip_x_y_pre.repeat(self.window_size , 1, 1)
        self.camera_intrinsics = self.K_torch.repeat(self.window_size , 1,1)

    def reset_future(self, data_list, future_poses, potential_trajectory):
        self.online_window_size = len(data_list)+self.FUTURE_WINDOW_SIZE
        self.pose_2d_tensor = torch.zeros(self.online_window_size, 2, self.num_of_joints).to(self.device)
        self.inverse_transformation_matrix = torch.zeros(self.online_window_size , 4, 4).to(self.device)

        ##find future projections
        self.inverse_transformation_matrix[:self.FUTURE_WINDOW_SIZE, :, :] = potential_trajectory.inv_transformation_matrix.clone()
        camera_intrinsics = self.K_torch.repeat(self.FUTURE_WINDOW_SIZE , 1,1)
        flip_x_y = self.flip_x_y_pre.repeat(self.FUTURE_WINDOW_SIZE , 1, 1)
        future_projection = self.take_batch_projection(future_poses, self.inverse_transformation_matrix[:self.FUTURE_WINDOW_SIZE, :, :], self.ones_tensor_future, camera_intrinsics, flip_x_y) 
        #add some noise to future projection so that the error is not zero
        self.pose_2d_tensor[:self.FUTURE_WINDOW_SIZE, :, :] = add_noise_to_pose(future_projection, self.noise_2d_std)
       
        queue_index = self.FUTURE_WINDOW_SIZE
        for bone_2d, _, inverse_transformation_matrix in data_list:
            self.pose_2d_tensor[queue_index, :, :] = bone_2d.clone()
            self.inverse_transformation_matrix[queue_index, :, :]= inverse_transformation_matrix.clone()
            queue_index += 1

        self.ones_tensor = torch.ones(self.online_window_size, 1, self.num_of_joints).to(self.device)*1.0
        self.flip_x_y_batch = self.flip_x_y_pre.repeat(self.online_window_size , 1, 1)
        self.camera_intrinsics = self.K_torch.repeat(self.online_window_size , 1,1)

    def deepcopy_projection_client(self):
        return Projection_Client(self.test_set, self.FUTURE_WINDOW_SIZE, self.num_of_joints, self.focal_length, self.px, self.py, self.noise_2d_std, self.device) 

    def take_projection(self, pose_3d):
        return self.take_batch_projection(pose_3d, self.inverse_transformation_matrix, self.ones_tensor, self.camera_intrinsics, self.flip_x_y_batch)

    def take_single_projection(self, P_world, inv_transformation_matrix):
        P_world_device = P_world.to(self.device)
        inv_transformation_matrix_device = inv_transformation_matrix.to(self.device)
        P_world = torch.cat((P_world_device, self.ones_tensor_single), 0)
        P_camera = self.world_to_camera(P_world, inv_transformation_matrix_device, self.flip_x_y_single)
        proj_homog = torch.mm(self.K_torch, P_camera)
        
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

    def take_single_backprojection(self, pose_2d, transformation_matrix, joint_names):
        transformation_matrix_device = transformation_matrix.to(self.device)
        img_torso_size = torch.norm(pose_2d[:, joint_names.index('neck')] - pose_2d[:, joint_names.index('spine1')]).float()
        if img_torso_size == 0:
            return torch.normal(torch.zeros(3, self.num_of_joints), torch.ones(3, self.num_of_joints)*10).float()

        z_val = ((self.focal_length * DEFAULT_TORSO_SIZE) / (img_torso_size)).float()

        self.single_backproj_res[0,:] = pose_2d[0,:]*z_val
        self.single_backproj_res[1,:] = pose_2d[1,:]*z_val
        self.single_backproj_res[2,:] = self.ones_tensor_single*z_val
        
        self.single_backproj_res = torch.mm(self.K_inv_torch, self.single_backproj_res)
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
        P_camera = torch.mm(self.flip_x_y_single_inv, P_camera)
        P_camera = torch.cat((P_camera, self.ones_tensor_single),0)
        P_world_ = torch.mm(transformation_matrix, P_camera)
        P_world = P_world_.clone()
        return P_world[0:3,:]